"""The papers the rising topics count, each dated by the submission of its first version.

arXiv's OAI-PMH interface (arXivRaw) gives every record changed since a day with the date of its
version 1, so a week of submissions is complete a few days after it ends, however often its papers
are revised later, and years of history can be read again the same way. bioRxiv and medRxiv come
from their API, version 1 only, dated by their posting day.
"""
import datetime
import email.utils
import logging
import time
import xml.etree.ElementTree as ET

import requests

from . import sources
from .digest import BIO_ARCHIVES, bio_page

logger = logging.getLogger(__name__)

OAI = 'https://oaipmh.arxiv.org/oai'
NS = {'oai': 'http://www.openarchives.org/OAI/2.0/', 'raw': 'http://arxiv.org/OAI/arXivRaw/'}
PAUSE = 2  # seconds between OAI pages; arXiv answers 503 with Retry-After when it wants more
MAX_TRIES = 6


def _text(node, path):
    found = node.find(path, NS)
    return ' '.join((found.text or '').split()) if found is not None else ''


def _first_version_day(meta):
    for version in meta.findall('raw:version', NS):
        if version.get('version') == 'v1':
            try:
                return email.utils.parsedate_to_datetime(_text(version, 'raw:date')).date()
            except (TypeError, ValueError):
                return None
    return None


def _authors(text):
    """'A. Lee, B. Ray and C. Day' -> ['A. Lee', 'B. Ray', 'C. Day'] (arXivRaw gives one string)."""
    text = text.replace(' and ', ', ')
    return [a.strip() for a in text.split(',') if a.strip()][:200]


def _oai_page(params):
    for attempt in range(MAX_TRIES):
        try:
            response = requests.get(OAI, params=params, timeout=120, headers={'User-Agent': sources.USER_AGENT})
        except requests.RequestException as e:
            logger.info('OAI request failed (%r), retrying', e)
            time.sleep(30 * (attempt + 1))
            continue
        if response.status_code == 200:
            return ET.fromstring(response.content)
        wait = int(response.headers.get('Retry-After', '0') or 0) or 30 * (attempt + 1)
        logger.info('OAI answered %s, retrying in %ds', response.status_code, wait)
        time.sleep(min(wait, 600))
    raise IOError('arXiv OAI-PMH unavailable')


def arxiv_papers(since, until=None, first=None):
    """Every arXiv paper changed between `since` and `until` whose version 1 is from `first` (default `since`) on,
    as {'key', 'day', 'title', 'abstract', 'authors', 'categories'}."""
    first = first or since
    params = {'verb': 'ListRecords', 'metadataPrefix': 'arXivRaw', 'from': since.isoformat()}
    if until:
        params['until'] = until.isoformat()
    while True:
        root = _oai_page(params)
        error = root.find('oai:error', NS)
        if error is not None:
            if error.get('code') == 'noRecordsMatch':
                return
            raise IOError('OAI error %s: %s' % (error.get('code'), error.text))
        listing = root.find('oai:ListRecords', NS)
        for record in listing.findall('oai:record', NS) if listing is not None else []:
            meta = record.find('oai:metadata/raw:arXivRaw', NS)
            if meta is None:
                continue  # a deleted record
            day = _first_version_day(meta)
            if day is None or day < first:
                continue
            yield {'key': _text(meta, 'raw:id'), 'day': day, 'title': _text(meta, 'raw:title'),
                   'abstract': _text(meta, 'raw:abstract'), 'authors': _authors(_text(meta, 'raw:authors')),
                   'categories': _text(meta, 'raw:categories').split()}
        token = listing.find('oai:resumptionToken', NS) if listing is not None else None
        if token is None or not (token.text or '').strip():
            return
        params = {'verb': 'ListRecords', 'resumptionToken': token.text.strip()}
        time.sleep(PAUSE)


def bio_papers(first, last):
    """The bioRxiv and medRxiv preprints first posted from `first` to `last`."""
    for server in BIO_ARCHIVES:
        cursor = 0
        while True:
            rows, total = bio_page(server, first.isoformat(), last.isoformat(), cursor)
            for row in rows:
                if str(row.get('version')) != '1' or not row.get('doi'):
                    continue
                paper = sources.biorxiv_paper(row, server)
                try:
                    day = datetime.date.fromisoformat(row.get('date', ''))
                except ValueError:
                    continue
                yield {'key': paper['key'], 'day': day, 'title': paper['title'], 'abstract': paper['abstract'],
                       'authors': paper['authors'], 'categories': [paper['category']]}
            cursor += len(rows)
            if not rows or cursor >= total:
                break
