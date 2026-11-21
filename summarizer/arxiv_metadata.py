"""arXiv metadata harvested from OAI-PMH, for the statistics guides (authors per paper, journal versions).

`harvest` walks ListRecords (metadataPrefix=arXiv) from a date and appends one compact JSON line per record
to a gzip file; it saves the resumption token after each page, so an interrupted harvest starts again where
it stopped. `load` reads the lines back. Nothing here needs the database: the harvest runs on a laptop and the
guides it feeds are committed as Markdown (see management/commands/data_guides.py).
"""
import gzip
import json
import os
import re
import time
import xml.etree.ElementTree as ET

import requests

OAI = 'https://oaipmh.arxiv.org/oai'
NS = {'o': 'http://www.openarchives.org/OAI/2.0/', 'a': 'http://arxiv.org/OAI/arXiv/'}
USER_AGENT = 'SummarizePaper-statistics/1.0 (https://www.summarizepaper.com/en/contact/)'
SPACING = 3  # seconds between requests, arXiv's rule for automated access
# an author entry that names a group, not a person ('ATLAS Collaboration', 'the Fermi-LAT team')
COLLABORATION = re.compile(r'\b(collaborations?|consortium|team|group|survey|observatory|experiment)\b', re.I)
COMMENTS = 300  # characters of the comments kept (where 'Accepted at NeurIPS 2022' is written)


class OAIError(Exception):
    def __init__(self, code, text):
        super().__init__('%s: %s' % (code, text))
        self.code = code


def parse(xml_bytes):
    """(records, token, latest datestamp): records as dicts, token None on the last page."""
    root = ET.fromstring(xml_bytes)
    error = root.find('o:error', NS)
    if error is not None:
        raise OAIError(error.get('code'), (error.text or '').strip())
    records, latest = [], ''
    for rec in root.iterfind('.//o:record', NS):
        header = rec.find('o:header', NS)
        stamp = header.findtext('o:datestamp', '', NS) if header is not None else ''
        latest = max(latest, stamp)
        if header is not None and header.get('status') == 'deleted':
            continue
        meta = rec.find('.//a:arXiv', NS)
        if meta is None:
            continue

        def text(tag):
            node = meta.find('a:' + tag, NS)
            return ' '.join(node.text.split()) if node is not None and node.text else ''
        authors = meta.findall('a:authors/a:author', NS)
        groups = sum(1 for a in authors if COLLABORATION.search(
            ' '.join(filter(None, (a.findtext('a:keyname', '', NS), a.findtext('a:forenames', '', NS))))))
        records.append({
            'id': text('id'),
            'cats': text('categories').split(),
            'authors': len(authors),
            'groups': groups,
            'jref': text('journal-ref'),
            'doi': text('doi').split()[0] if text('doi') else '',
            'comments': text('comments')[:COMMENTS],
        })
    token = root.find('.//o:resumptionToken', NS)
    return records, (token.text.strip() if token is not None and token.text and token.text.strip() else None), latest


def harvest(path, since, log=print, update=False):
    """Append every record changed since `since` (YYYY-MM-DD) to `path` (.jsonl.gz); resumable.

    The state file keeps the resumption token and the latest datestamp seen: arXiv's tokens expire (at midnight
    UTC), and an expired one restarts the list from that date (records seen twice are merged by `load`)."""
    state_path = path + '.state'
    state = json.load(open(state_path)) if os.path.exists(state_path) else {'token': '', 'from': since, 'done': False}
    if state['done'] and update:  # a later run adds what changed since the last datestamp seen (yearly refresh)
        state = {'token': '', 'from': state['from'], 'done': False}
    if state['done']:
        log('harvest already complete')
        return
    token = state['token']
    session = requests.Session()
    session.headers['User-Agent'] = USER_AGENT
    pages = 0
    while True:
        params = {'verb': 'ListRecords', 'resumptionToken': token} if token else \
            {'verb': 'ListRecords', 'metadataPrefix': 'arXiv', 'from': state['from']}
        for attempt in range(8):
            try:
                response = session.get(OAI, params=params, timeout=180)
            except requests.RequestException as exc:
                log('network error %s, retry' % exc)
                time.sleep(30 * (attempt + 1))
                continue
            if response.status_code == 503:
                wait = int(response.headers.get('Retry-After', '30') or 30)
                log('503, waiting %d s' % wait)
                time.sleep(max(wait, 10))
                continue
            if response.status_code >= 500:
                time.sleep(30 * (attempt + 1))
                continue
            response.raise_for_status()
            break
        else:
            raise RuntimeError('OAI-PMH kept failing; run again to resume')
        try:
            records, token, latest = parse(response.content)
        except OAIError as exc:
            if exc.code != 'badResumptionToken' or not token:
                raise
            log('token expired, restarting from %s' % state['from'])
            token = state['token'] = ''
            continue
        with gzip.open(path, 'at', encoding='utf-8') as out:
            for record in records:
                out.write(json.dumps(record, separators=(',', ':')) + '\n')
        state = {'token': token or '', 'from': max(state['from'], latest[:10]) if latest else state['from'], 'done': not token}
        with open(state_path, 'w') as f:
            json.dump(state, f)
        pages += 1
        if pages % 20 == 0:
            log('%d pages, last id %s' % (pages, records[-1]['id'] if records else '-'))
        if not token:
            log('harvest complete after %d pages' % pages)
            return
        time.sleep(SPACING)


def titles(ids):
    """{arXiv id: title} from the arXiv API, {} when it cannot be reached (the page then shows the ids)."""
    if not ids:
        return {}
    try:
        response = requests.get('https://export.arxiv.org/api/query', params={'id_list': ','.join(ids), 'max_results': len(ids)},
                                headers={'User-Agent': USER_AGENT}, timeout=60)
        response.raise_for_status()
        root = ET.fromstring(response.content)
    except (requests.RequestException, ET.ParseError):
        return {}
    atom = '{http://www.w3.org/2005/Atom}'
    found = {}
    for entry in root.iter(atom + 'entry'):
        arxiv_id = re.sub(r'v\d+$', '', (entry.findtext(atom + 'id') or '').rsplit('/abs/', 1)[-1])
        title = ' '.join((entry.findtext(atom + 'title') or '').split())
        if arxiv_id and title:
            found[arxiv_id] = title
    return found


def load(path):
    """Records by arXiv id (a record harvested twice after a resume keeps its last line)."""
    records = {}
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            record = json.loads(line)
            records[record['id']] = record
    return records
