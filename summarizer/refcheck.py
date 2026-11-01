"""Bibliography check (Premium): what is wrong with the references of a paper before it is submitted.

From a .bib (pasted, uploaded or exported from a Zotero collection), and optionally the manuscript: its LaTeX
(one file, or an Overleaf archive with its .bib), the source of an arXiv paper, or a PDF, whose printed reference
list is read when there is no .bib (see refcheck_sources.py).

1. Each entry is identified: its DOI, its arXiv id, or, for the entries with neither, a conservative
   Crossref/ADS/OpenAlex cascade using titles, authors and locators (bibliography.py), several at once.
   Matched metadata is compared field by field; an unresolved entry is never called incorrect.
2. Retracted papers come from OpenAlex (is_retracted, fed by Retraction Watch) and from the notices Crossref
   links to the article; expressions of concern and corrections too; papers withdrawn from arXiv from their
   arXiv comment.
3. A preprint cited as such may have been published since: arXiv's own DOI and journal reference, Semantic
   Scholar's, and for a preprint server's DOI (bioRxiv, medRxiv, ChemRxiv, SSRN...) Crossref's explicit
   link to its published version. A missing DOI, volume or pages is filled from the record that identifies the
   entry. All the corrections of an entry make one corrected entry; the researcher chooses which to download.
4. An arXiv version older than the latest one, and the same paper cited under two keys (its preprint and its
   published version included) are reported too; with the LaTeX, the keys cited but missing from the .bib and
   the entries never cited.
5. With the LaTeX, each sentence with citations is read with the abstracts of all the papers it cites, and the
   sentences they do not support are read again against the full text of the papers when it may be read, with
   the passage that supports or contradicts them (refcheck_claims.py). It is a warning, not a verdict.

The researcher marks what is intended or fixed, checks again after correcting (what was solved and what is new
is shown, the marks are kept), shares the report, keeps the references in a collection, and can watch them
until the paper is published (refcheck_watch.py).
"""
import contextvars
import html
import json
import logging
import re
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, bibliography, citations, digest, generated, refcheck_claims, refcheck_sources, result_share, sources
from .models import Collection, Generated, ReferenceWatch, ZoteroAccount

logger = logging.getLogger(__name__)

MAX_UPLOAD = 5 * 1024 * 1024
MAX_ARCHIVE = 40 * 1024 * 1024
MAX_ENTRIES = 400
TITLE_SEARCHES = 80  # bounded cascades for entries without identifiers; remaining entries are disclosed
MAX_LOOKUPS = 900
RESOLVE_THREADS = 6
OPENALEX_BATCH = 50
ARXIV_BATCH = 100
S2_BATCH = 'https://api.semanticscholar.org/graph/v1/paper/batch'
PER_DAY = 10
QUOTA_KEY = 'refcheck-quota:%s:%s'
RECHECK_KEY = 'refcheck-recheck:%s'
VERSION = 3
LEVELS = ('error', 'warning', 'info')
MARKS = ('intended', 'fixed', 'ignored')
RETRACTIONS = ('retraction', 'withdrawal', 'removal')
CORRECTIONS = ('correction', 'erratum', 'corrigendum')
# the fields a reference cannot do without, by entry type (an editor stands for the author of a book)
REQUIRED = {'article': ('author', 'title', 'journal', 'year'), 'book': ('author', 'title', 'publisher', 'year'),
            'inproceedings': ('author', 'title', 'booktitle', 'year'), 'incollection': ('author', 'title', 'booktitle', 'year'),
            'phdthesis': ('author', 'title', 'school', 'year'), 'mastersthesis': ('author', 'title', 'school', 'year'),
            'techreport': ('author', 'title', 'institution', 'year')}
LIMITED = contextvars.ContextVar('openalex_limited', default=False)  # set when OpenAlex calls were skipped
STEPS = [
    ('read', _('Reading your references')),
    ('identify', _('Matching references with Crossref, NASA ADS and OpenAlex')),
    ('published', _('Looking for retractions, corrections and preprints published since')),
    ('claims', _('An AI reads each sentence with a citation next to the abstracts of the papers it cites')),
    ('deep', _('Reading the full text of the cited papers for the sentences flagged')),
]
KIND_LABELS = {
    'retracted': _('Retracted'), 'withdrawn': _('Withdrawn'), 'concern': _('Expression of concern'),
    'corrected': _('Correction published'), 'published': _('Final publication available'), 'duplicate': _('Duplicate'),
    'arxiv_missing': _('Not found'), 'old_version': _('Older version'), 'metadata': _('Reference to correct'),
    'incomplete': _('Incomplete entry'), 'unknown': _('Reference unresolved'),
}
FAILURES = {
    'no_references': _('We found no reference: send a .bib file, an archive with your .bib, or a PDF with its reference list.'),
    'archive': _('We could not open this archive. Send a .zip or .tar.gz of your LaTeX project.'),
    'too_large': _('This archive is too large. Send only your .tex and .bib files.'),
    'pdf': _('We could not read the text of this PDF.'),
    'arxiv': _('arXiv did not answer. Try again in a few minutes.'),
    'arxiv_missing': _('arXiv has no paper with this identifier.'),
    'arxiv_pdf_only': _('arXiv has no LaTeX source for this paper, only its PDF. Send the PDF instead.'),
    'zotero': _('Zotero did not answer. Check your Zotero key on the integrations page, or try again later.'),
}


# --- BibTeX ------------------------------------------------------------------

def _value(text, i):
    """The value starting at text[i] ({...}, "..." or a bare word) and the index after it."""
    if text[i] == '{':
        depth, j = 0, i
        while j < len(text):
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            if depth == 0:
                return text[i + 1:j], j + 1
            j += 1
        return text[i + 1:], len(text)
    if text[i] == '"':
        j = i + 1
        while j < len(text) and (text[j] != '"' or text[j - 1] == '\\'):
            j += 1
        return text[i + 1:j], j + 1
    match = re.match(r'[^,}\s]*', text[i:])
    return match.group(0), i + len(match.group(0))


def parse_bibtex(text):
    """[{'key', 'type', 'fields': {name: value}, 'raw'}] of the entries (not @string, @comment, @preamble)."""
    entries = []
    for match in re.finditer(r'@(\w+)\s*([{(])', text):
        kind = match.group(1).lower()
        if kind in ('string', 'comment', 'preamble'):
            continue
        start, i = match.start(), match.end()
        key_match = re.match(r'\s*([^,\s]+)\s*,', text[i:])
        if not key_match:
            continue
        i += key_match.end()
        fields = {}
        while i < len(text):
            field = re.match(r'\s*([\w\-:.]+)\s*=\s*', text[i:])
            if not field:
                break
            i += field.end()
            if i >= len(text):
                break
            value, i = _value(text, i)
            fields[field.group(1).lower()] = ' '.join(value.split())
            rest = re.match(r'\s*(#\s*[^,}]+)?\s*,?', text[i:])  # string concatenations are dropped
            i += rest.end()
            if i < len(text) and text[i] in '})':
                break
        end = text.find('\n@', i)
        raw = text[start:i + 1] if i < len(text) and text[i] in '})' else text[start:end if end > 0 else len(text)]
        entries.append({'key': key_match.group(1), 'type': kind, 'fields': fields, 'raw': raw.strip()})
        if len(entries) >= MAX_ENTRIES:
            break
    return entries


def plain(text):
    """BibTeX braces and accents commands out, for showing and comparing titles."""
    text = re.sub(r'\\[a-zA-Z]+\s*(?=\{)|[{}]', '', text or '')
    return ' '.join(re.sub(r'\\[\'"`^~=.]', '', text).split())


def identify(entry):
    """(arxiv id without version or '', the version cited or '', DOI or '') from the entry's fields."""
    f = entry['fields']
    doi = sources.clean_doi(f.get('doi', '')) or ''
    arxiv, version = '', ''
    match = sources.ARXIV_DOI_RE.match(doi)
    if match:
        arxiv, doi = match.group(1), ''
    # the eprint field is an arXiv id; elsewhere, an id only counts next to the word arXiv
    candidates = [(f.get('eprint', ''), True)] + [(f.get(k, ''), False) for k in ('url', 'journal', 'note', 'howpublished')]
    if f.get('journal', '').strip('{} ').lower() == 'corr':  # DBLP: journal = {CoRR}, volume = {abs/1412.6980}
        candidates.append((f.get('volume', ''), True))
    for value, trusted in candidates:
        found = re.search(r'(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?', value)
        if found and (trusted or 'arxiv' in value.lower()):
            arxiv, version = arxiv or found.group(1), found.group(2) or ''
            break
    if not doi:
        found = sources.DOI_RE.search(f.get('url', ''))
        if found and not sources.ARXIV_DOI_RE.match(found.group(1)):
            doi = found.group(1).rstrip('.,;)').lower()
    return arxiv, version, doi.lower()


def is_preprint_entry(entry):
    """Whether the entry cites a preprint: an arXiv DOI or journal, or no journal, book or proceedings at all."""
    f = entry['fields']
    journal = f.get('journal', '').lower()
    if sources.ARXIV_DOI_RE.match(f.get('doi', '')) or 'arxiv' in journal:
        return True
    if journal or f.get('booktitle') or entry['type'] in ('book', 'inbook', 'incollection', 'inproceedings', 'phdthesis'):
        return False
    return not f.get('doi')


# --- lookups -----------------------------------------------------------------

def arxiv_records(ids):
    """{arxiv id: arXiv API record} (latest version), by batches of ARXIV_BATCH."""
    found = {}
    ids = list(dict.fromkeys(ids))
    for start in range(0, len(ids), ARXIV_BATCH):
        try:
            papers = digest.fetch_ids(ids[start:start + ARXIV_BATCH])
        except digest.ArxivUnavailable:
            continue
        for paper in papers:
            versioned = paper['arxiv_id'].replace('--', '/')
            found[re.sub(r'v\d+$', '', versioned)] = dict(paper, version=re.search(r'v(\d+)$', versioned).group(1)
                                                          if re.search(r'v\d+$', versioned) else '')
    return found


def openalex_works(dois):
    """{doi: {'title', 'retracted', 'abstract'}} from OpenAlex, by batches of OPENALEX_BATCH."""
    found = {}
    dois = list(dict.fromkeys(d for d in dois if d))
    for start in range(0, len(dois), OPENALEX_BATCH):
        batch = dois[start:start + OPENALEX_BATCH]
        if not sources.tool_search_allowed():
            LIMITED.set(True)
            break
        try:
            body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
                'filter': 'doi:%s' % '|'.join(batch), 'per_page': len(batch),
                'select': 'doi,display_name,is_retracted,abstract_inverted_index'}))
        except sources.SourceUnavailable:
            continue
        for work in (body or {}).get('results', []):
            doi = (work.get('doi') or '').split('doi.org/')[-1].lower()
            found[doi] = {'title': work.get('display_name') or '', 'retracted': bool(work.get('is_retracted')),
                          'abstract': sources.inverted_abstract(work.get('abstract_inverted_index'))}
    return found


def by_title(title):
    """(doi, arxiv id) of the OpenAlex work with this title, ('', '') when no title matches closely."""
    if not sources.tool_search_allowed():
        LIMITED.set(True)
        return '', ''
    try:
        body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
            'search': title[:300], 'per_page': 3, 'select': 'doi,display_name,locations'}))
    except sources.SourceUnavailable:
        return '', ''
    wanted = normalized(title)
    for work in (body or {}).get('results', []):
        if normalized(work.get('display_name') or '') == wanted:
            doi = (work.get('doi') or '').split('doi.org/')[-1].lower()
            match = sources.ARXIV_DOI_RE.match(doi)
            return ('', match.group(1)) if match else (doi, sources.arxiv_of(work))
    return '', ''


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', plain(text).lower()).strip()


def s2_published(arxiv_ids):
    """{arxiv id: journal DOI} from Semantic Scholar, for preprints arXiv does not link to a journal."""
    if not arxiv_ids:
        return {}
    try:
        response = citations.s2_call(lambda headers: requests.post(
            S2_BATCH, params={'fields': 'externalIds,journal,publicationTypes'}, headers=headers,
            json={'ids': ['ARXIV:%s' % a for a in arxiv_ids[:500]]}, timeout=60))
        response.raise_for_status()
        rows = response.json()
    except (requests.RequestException, ValueError):
        return {}
    found = {}
    for arxiv_id, row in zip(arxiv_ids, rows):
        doi = ((row or {}).get('externalIds') or {}).get('DOI') or ''
        if doi and not sources.ARXIV_DOI_RE.match(doi) and ((row.get('journal') or {}).get('name')):
            found[arxiv_id] = doi.lower()
    return found


def published_fields(work, doi):
    """BibTeX fields of the published version from its Crossref record: title, journal, year, volume, pages, doi."""
    year = ((work.get('published') or work.get('issued') or {}).get('date-parts') or [[None]])[0][0]
    fields = {'title': ' '.join(html.unescape(sources._strip_tags((work.get('title') or [''])[0])).split()),
              'journal': html.unescape((work.get('container-title') or [''])[0]), 'year': str(year or ''),
              'volume': work.get('volume') or '', 'pages': work.get('page') or work.get('article-number') or '',
              'doi': doi}
    return {k: v for k, v in fields.items() if v}


def with_fields(entry, fields, kind=None, published=False, drop=()):
    """Rewrite only reviewed fields, preserving the key and the remaining source fields."""
    drop = set(drop) | ({'archiveprefix', 'primaryclass', 'howpublished'} if published else set())
    merged = {k: v for k, v in entry['fields'].items() if k not in drop}
    merged.update(fields)
    lines = ['@%s{%s,' % (kind or entry['type'], entry['key'])]
    lines += ['  %s = {%s},' % (name, re.sub(r'(?<!\\)&', r'\\&', value) if name == 'journal' else value)
              for name, value in merged.items()]
    return '\n'.join(lines) + '\n}'


# --- the check ---------------------------------------------------------------

def first_author(entry):
    names = bibliography.families(entry['fields'].get('author', '') or entry['fields'].get('editor', ''))
    return names[0] if names else ''


def notices_of(works, *dois):
    """[(type, doi)] of the notices Crossref links to these DOIs."""
    from .publication_status import notices
    found = []
    for doi in dois:
        if doi:
            found += [(n['type'], n['doi']) for n in notices(works.get(doi) or {})]
    return list(dict.fromkeys(found))


def preprint_link(work):
    """The DOI of the published version Crossref links a preprint server's record to, else ''."""
    if work.get('type') != 'posted-content':
        return ''
    related = (work.get('relation') or {}).get('is-preprint-of') or []
    return next((sources.clean_doi(r.get('id', '')).lower() for r in related
                 if r.get('id-type') == 'doi' and sources.clean_doi(r.get('id', ''))), '')


def biorxiv_published(doi):
    """The journal DOI bioRxiv or medRxiv gives for one of its preprints (Crossref often has no link), else ''."""
    if not doi.startswith('10.1101/'):
        return ''
    key = 'biorxiv-published:%s' % doi
    found = cache.get(key)
    if found is not None:
        return found
    found = ''
    for server in ('biorxiv', 'medrxiv'):
        try:
            response = requests.get('https://api.biorxiv.org/details/%s/%s' % (server, doi), timeout=20)
            rows = response.json().get('collection') or [] if response.status_code == 200 else []
        except (requests.RequestException, ValueError):
            return ''  # not cached: asked again next time
        published = next((sources.clean_doi(r.get('published') or '') for r in reversed(rows)
                          if sources.clean_doi(r.get('published') or '')), '')
        if rows:
            found = (published or '').lower()
            break
    cache.set(key, found, 86400)
    return found


def completion(entry, matched):
    """(fields the matched record fills, required fields still missing) of an entry."""
    f, filled = entry['fields'], {}
    if matched:
        found = matched['fields']
        if matched.get('doi') and not f.get('doi') and not sources.ARXIV_DOI_RE.match(matched['doi']):
            filled['doi'] = matched['doi']
        for name in ('volume', 'pages', 'year'):
            if found.get(name) and not f.get(name):
                filled[name] = found[name]
        if entry['type'] == 'article' and found.get('journal') and not f.get('journal'):
            filled['journal'] = found['journal']
    missing = [name for name in REQUIRED.get(entry['type'], ())
               if not f.get(name) and name not in filled and not (name == 'author' and f.get('editor'))]
    return filled, missing


def check(bib, tex, lang, step=lambda name: None, origin='bib'):
    """The report on a bibliography. tex: the LaTeX as a string, or [(file name, LaTeX)] in reading order."""
    LIMITED.set(False)
    texts = [('main.tex', tex)] if isinstance(tex, str) else list(tex or [])
    texts = [(name, text) for name, text in texts if text and text.strip()]
    entries = parse_bibtex(bib)
    if not entries:
        raise generated.SourceUnavailable('no_references')
    by_key = {}
    for entry in entries:
        by_key.setdefault(entry['key'], entry)
    step('identify')
    budget = {'limited': False}

    def openalex_allowed():
        allowed = sources.tool_search_allowed()
        budget['limited'] |= not allowed
        return allowed
    resolver = bibliography.Resolver(openalex_allowed, max_calls=MAX_LOOKUPS, seconds=240)
    ids = {e['key']: identify(e) for e in entries}
    jobs, skipped, searched = [], set(), 0
    for entry in entries:
        arxiv, _version, doi = ids[entry['key']]
        if doi:
            jobs.append((entry, doi))
        elif not arxiv:
            if searched < TITLE_SEARCHES:
                searched += 1
                jobs.append((entry, ''))
            else:
                skipped.add(entry['key'])

    def resolve(job):
        entry, doi = job
        try:
            return entry['key'], resolver.resolve(entry, doi) if doi else resolver.resolve(entry)
        except Exception:
            logger.exception('Reference %s could not be resolved', entry['key'])
            return entry['key'], None
    metadata = {}
    with ThreadPoolExecutor(RESOLVE_THREADS) as pool:
        for n, (key, matched) in enumerate(pool.map(resolve, jobs)):
            if n % 10 == 9:
                step('identify')  # heartbeat while a long bibliography is resolved
            if matched:
                metadata[key] = matched
                arxiv, version, doi = ids[key]
                ids[key] = (arxiv or matched.get('arxiv', ''), version, doi or matched['doi'])

    step('published')
    records = arxiv_records([a for a, _, _ in ids.values() if a])
    unlinked = [a for a, r in records.items() if not r.get('doi')]
    s2 = s2_published([a for a in unlinked if any(ids[e['key']][0] == a and is_preprint_entry(e) for e in entries)])
    later, as_preprint = {}, {}
    for entry in entries:
        arxiv, _version, doi = ids[entry['key']]
        record = records.get(arxiv) if arxiv else None
        published = ((record or {}).get('doi', '') or s2.get(arxiv, '')).lower()
        later[entry['key']] = published if published != doi else ''
        as_preprint[entry['key']] = bool(record) and is_preprint_entry(entry)

    def crossref(doi):
        try:
            return doi, resolver.work(doi)
        except Exception:
            logger.exception('Crossref record of %s could not be read', doi)
            return doi, {}
    wanted = list(dict.fromkeys(d for d in [i[2] for i in ids.values()] + list(later.values()) if d))
    with ThreadPoolExecutor(RESOLVE_THREADS) as pool:
        works = dict(pool.map(crossref, wanted))
    servers = [ids[e['key']][2] for e in entries if (works.get(ids[e['key']][2]) or {}).get('type') == 'posted-content'
               and not preprint_link(works[ids[e['key']][2]])]
    with ThreadPoolExecutor(RESOLVE_THREADS) as pool:
        announced = dict(zip(servers, pool.map(biorxiv_published, servers)))
    for entry in entries:  # a preprint server's DOI (bioRxiv, SSRN...) and its journal version
        doi = ids[entry['key']][2]
        linked = preprint_link(works.get(doi) or {}) or announced.get(doi, '')
        if linked and linked != doi:
            later[entry['key']], as_preprint[entry['key']] = linked, True
            if linked not in works:
                works[linked] = crossref(linked)[1]
    retracted = openalex_works([d for d in wanted] + [d for d in later.values() if d and d not in wanted])

    items, updates = [], {}

    def update(key, fields, kind=None, drop=()):
        found = updates.setdefault(key, {'fields': {}, 'kind': None, 'drop': set()})
        found['fields'].update(fields)
        found['kind'] = kind or found['kind']
        found['drop'] |= set(drop)

    for position, entry in enumerate(entries):
        key = entry['key']
        arxiv, version, doi = ids[key]
        record = records.get(arxiv) if arxiv else None
        published = later[key]
        title = plain(entry['fields'].get('title', '')) or (record or {}).get('title', '') or key
        matched = metadata.get(key)
        issues = []
        if not matched and not record and not (works.get(doi) or {}).get('title'):
            issues.append({'kind': 'unknown', 'level': 'info',
                           'reason': 'limit' if key in skipped or resolver.limited else
                                     'unavailable' if resolver.unavailable else 'no_match'})
        if arxiv and record is None:
            issues.append({'kind': 'arxiv_missing', 'level': 'warning', 'arxiv': arxiv})
        if record and re.search(r'\bwithdrawn\b', record.get('comment', ''), re.I):
            issues.append({'kind': 'withdrawn', 'level': 'error', 'detail': record['comment'][:300]})
        notices = notices_of(works, doi, published)
        retraction = next((n for n in notices if n[0] in RETRACTIONS), None)
        if retraction or any(retracted.get(d, {}).get('retracted') for d in (doi, published) if d):
            issues.append({'kind': 'retracted', 'level': 'error',
                           'doi': next(d for d in (doi, published) if d), 'notice': retraction[1] if retraction else ''})
        for kind, notice in notices:
            if kind == 'expression_of_concern':
                issues.append({'kind': 'concern', 'level': 'warning', 'notice': notice})
            elif kind in CORRECTIONS and not any(i['kind'] == 'corrected' for i in issues):
                issues.append({'kind': 'corrected', 'level': 'warning', 'notice': notice})
        if published and as_preprint[key]:
            fields = published_fields(works.get(published) or {}, published)
            server = (works.get(doi) or {}).get('type') == 'posted-content'
            update(key, fields, 'article', drop=('archiveprefix', 'primaryclass', 'howpublished') +
                   (('publisher', 'institution') if server else ()))
            issues.append({'kind': 'published', 'level': 'warning', 'doi': published, 'corrects': True,
                           'journal': fields.get('journal') or (record or {}).get('journal_ref', '')})
        # a preprint published since is corrected from its published version, not from the preprint's record
        changes = bibliography.differences(entry, matched) if matched and not (published and as_preprint[key]) else []
        if changes:
            fields = {c['field']: c['after'] for c in changes if c['field'] != 'type'}
            if matched['doi']:
                fields['doi'] = matched['doi']
            kind = matched['type'] if any(c['field'] == 'type' for c in changes) else None
            if kind == 'book' and matched['fields'].get('isbn') and not entry['fields'].get('isbn'):
                fields['isbn'] = matched['fields']['isbn']
            # the record of the preprint never overrides its published version
            update(key, {k: v for k, v in fields.items() if k not in updates.get(key, {}).get('fields', {})}, kind)
            issues.append({'kind': 'metadata', 'level': 'warning', 'changes': changes, 'corrects': True,
                           'source': matched['source'], 'url': matched['url']})
        filled, missing = completion(entry, matched) if not (published and as_preprint[key]) else ({}, [])
        filled = {k: v for k, v in filled.items() if k not in updates.get(key, {}).get('fields', {})}
        if filled or missing:
            if filled:
                update(key, filled)
            issues.append({'kind': 'incomplete', 'level': 'info', 'filled': filled, 'missing': missing,
                           'corrects': bool(filled)})
        if record and version and record.get('version') and int(version[1:]) < int(record['version']):
            issues.append({'kind': 'old_version', 'level': 'info', 'cited': version, 'latest': 'v%s' % record['version'],
                           'arxiv': arxiv})
        abstract = (record or {}).get('abstract') or retracted.get(doi, {}).get('abstract', '') or \
            retracted.get(published, {}).get('abstract', '')
        items.append({'key': key, 'title': title, 'arxiv': arxiv, 'doi': doi or published, 'issues': issues,
                      'position': position, 'abstract': abstract,
                      'source': matched['source'] if matched else 'arXiv' if record else 'Crossref' if works.get(doi) else '',
                      'source_url': matched['url'] if matched else 'https://arxiv.org/abs/' + arxiv if record else
                                    'https://doi.org/' + doi if works.get(doi) else '',
                      'verification': 'incorrect' if changes else 'published' if published and as_preprint[key] else
                                      'identified' if matched or record or works.get(doi) else 'unverified'})

    # the same work under two keys: same DOI or arXiv id, a preprint and its published version, or the same title
    owner = {}
    for item, entry in zip(items, entries):
        refs = [('doi', ids[item['key']][2]), ('arxiv', item['arxiv']), ('doi', later[item['key']])]
        title, author = normalized(entry['fields'].get('title', '')), first_author(entry)
        if len(title) >= 25 and author:
            refs.append(('title', title + '|' + author))
        for ref in refs:
            other = owner.get(ref) if ref[1] else None
            if other and other != item['key']:
                pair = ref[0] == 'doi' and (ref[1] == later[item['key']]) != (ref[1] == later[other])
                item['issues'].insert(0, {'kind': 'duplicate', 'level': 'warning', 'other': other, 'pair': pair})
                break
        for ref in refs:
            if ref[1]:
                owner.setdefault(ref, item['key'])

    corrected = {}
    if origin == 'bib':
        for key, found in updates.items():
            corrected[key] = with_fields(by_key[key], found['fields'], found['kind'], drop=found['drop'])
    for item in items:
        for issue in item['issues']:
            if issue.pop('corrects', False) and item['key'] in corrected:
                issue['bibtex'] = corrected[item['key']]

    claims = {'claims': [], 'confirmed': [], 'checked': 0, 'citing': 0, 'readable': 0, 'full_texts_tried': 0, 'full_texts_read': 0}
    undefined, uncited = [], []
    if texts:
        step('claims')
        papers = {i['key']: {'title': i['title'], 'abstract': i['abstract'], 'arxiv': i['arxiv'], 'doi': i['doi']}
                  for i in items}
        claims = refcheck_claims.run(texts, papers, lang, lambda: step('deep'))
        keys, everything = refcheck_claims.cited_keys(texts)
        undefined = [k for k in keys if k not in by_key]
        if not everything and origin == 'bib':
            uncited = [e['key'] for e in entries if e['key'] not in set(keys)]
        for claim in claims['claims'] + claims['confirmed']:
            claim['papers'] = [paper_card(papers[k], k) for k in claim['concerned'] if k in papers]
    for item in items:
        item.pop('abstract')
    rank = {level: i for i, level in enumerate(LEVELS)}
    problems = [i for i in items if i['issues']]
    problems.sort(key=lambda i: (min(rank[x['level']] for x in i['issues']), i['position']))
    fixed_bib = bib
    for entry in entries:
        if entry['key'] in corrected and entry['raw'] in fixed_bib:
            fixed_bib = fixed_bib.replace(entry['raw'], corrected[entry['key']], 1)
    return {
        'version': VERSION, 'origin': origin, 'entries': len(entries), 'problems': problems,
        'claims': [claim_shown(c) for c in claims['claims']], 'confirmed': [claim_shown(c) for c in claims['confirmed']],
        'claims_checked': claims['checked'], 'claims_citing': claims['citing'], 'claims_readable': claims['readable'],
        'with_tex': bool(texts), 'files': [name for name, _text in texts],
        'counts': {level: sum(any(x['level'] == level for x in i['issues']) for i in items) for level in LEVELS},
        'fixed_bib': fixed_bib if corrected else '', 'corrected': len(corrected),
        'corrections': {k: {'before': by_key[k]['raw'], 'after': v} for k, v in corrected.items()},
        'limited': LIMITED.get() or budget['limited'],
        'summary': {status: sum(i['verification'] == status for i in items)
                    for status in ('incorrect', 'published', 'identified', 'unverified')},
        'identified': [{k: i[k] for k in ('key', 'title', 'source', 'source_url')} for i in items
                       if i['verification'] == 'identified' and not i['issues']],
        'sources_unavailable': sorted(resolver.unavailable), 'lookup_limited': resolver.limited or bool(skipped),
        'metadata_checked': len(metadata), 'undefined': undefined, 'uncited': uncited,
        'refs': [{k: i[k] for k in ('key', 'title', 'arxiv', 'doi')} for i in items if i['arxiv'] or i['doi']],
        'full_texts_tried': claims['full_texts_tried'], 'full_texts_read': claims['full_texts_read'],
        'profile': profile(entries, items, as_preprint, texts),
    }


def manuscript_authors(texts):
    """The family names of the manuscript's authors, from its \\author commands."""
    names = []
    for _name, tex in texts[:1]:
        for found in re.findall(r'\\author\s*(?:\[[^\]]*\])?\s*\{((?:[^{}]|\{[^{}]*\})*)\}', refcheck_sources.strip_comments(tex)):
            found = re.sub(r'\\(?:thanks|footnote|inst|affil|email)\s*\{(?:[^{}]|\{[^{}]*\})*\}|\$[^$]*\$|\^\{?[^}\s]*\}?', ' ', found)
            found = re.sub(r'\\(?:and|AND)\b|\\\\|,|;', ' and ', found)
            names += bibliography.families(re.sub(r'\\[a-zA-Z]+|[{}~]', ' ', found))
    return [n for n in dict.fromkeys(names) if len(n) > 1]


def profile(entries, items, as_preprint, texts):
    """What a referee sees at a glance of a bibliography: its age, its preprints, its venues, its self-citations."""
    now = timezone.now().year
    years = sorted(int(e['fields']['year']) for e in entries if re.fullmatch(r'(1[5-9]|20)\d\d', e['fields'].get('year', '').strip()))
    venues = {}
    for entry in entries:
        journal = plain(entry['fields'].get('journal', '') or entry['fields'].get('booktitle', ''))
        if journal and 'arxiv' not in journal.lower():
            name = bibliography.journal_name(journal)
            venues[name] = venues.get(name, 0) + 1
    mine = set(manuscript_authors(texts))
    own = [e['key'] for e in entries if mine & set(bibliography.families(e['fields'].get('author', '')))] if mine else None
    return {
        'with_id': sum(1 for i in items if i['arxiv'] or i['doi']),
        'preprints': sum(1 for e in entries if as_preprint.get(e['key']) or is_preprint_entry(e) and ids_arxiv(e)),
        'median_age': now - years[len(years) // 2] if years else None, 'dated': len(years),
        'recent': sum(1 for y in years if y >= now - 5), 'oldest': years[0] if years else None,
        'newest': years[-1] if years else None,
        'venues': sorted(venues.items(), key=lambda v: (-v[1], v[0]))[:5],
        'self': len(own) if own is not None else None, 'self_keys': (own or [])[:30], 'authors': sorted(mine)[:20],
    }


def ids_arxiv(entry):
    return identify(entry)[0]


def paper_card(paper, key):
    """What the report shows of a cited paper next to a flagged sentence."""
    chat = paper['arxiv'] or (sources.doi_key(paper['doi']) if paper['doi'] else '')
    return {'key': key, 'title': paper['title'], 'abstract': paper['abstract'][:1500], 'chat': chat,
            'url': 'https://arxiv.org/abs/' + paper['arxiv'] if paper['arxiv'] else
                   'https://doi.org/' + paper['doi'] if paper['doi'] else ''}


def claim_shown(claim):
    shown = {k: claim.get(k) for k in ('keys', 'concerned', 'sentence', 'file', 'line', 'verdict', 'use', 'reason',
                                       'passage', 'numbers_absent', 'not_in_full_text', 'full_text', 'papers')
             if claim.get(k) not in (None, '', [])}
    shown['key'] = claim['concerned'][0] if claim.get('concerned') else claim['keys'][0]
    shown['title'] = next((p['title'] for p in claim.get('papers', []) if p['key'] == shown['key']), '')
    return shown


def carried(result, previous):
    """Marks and novelty against the previous check of the same manuscript; what it reported and is now solved."""
    old = {(p['key'], i['kind']): i.get('mark', '') for p in previous.get('problems', []) for i in p['issues']}
    now = set()
    for problem in result['problems']:
        for issue in problem['issues']:
            now.add((problem['key'], issue['kind']))
            if (problem['key'], issue['kind']) in old:
                if old[(problem['key'], issue['kind'])]:
                    issue['mark'] = old[(problem['key'], issue['kind'])]
            else:
                issue['new'] = True
    sentences = {' '.join(c['sentence'].split()): c.get('mark', '') for c in previous.get('claims', [])}
    for claim in result['claims']:
        sentence = ' '.join(claim['sentence'].split())
        if sentence in sentences:
            if sentences[sentence]:
                claim['mark'] = sentences[sentence]
        else:
            claim['new'] = True
    return [{'key': k, 'kind': kind} for (k, kind), _mark in old.items() if (k, kind) not in now and kind != 'unknown'][:200]


def write(load, lang, user=None, previous=None, with_missed=False):
    def work(pk):
        generated.progress(pk, 'read')
        try:
            bib, texts, origin, pdf = load()
        except refcheck_sources.Unreadable as e:
            raise generated.SourceUnavailable(str(e))
        result = check(bib, texts, lang, lambda name: generated.progress(pk, name), origin=origin)
        if previous is not None:
            result['previous'] = {'key': previous.key, 'date': previous.created.date().isoformat(),
                                  'solved': carried(result, previous.data())}
        if with_missed and user is not None:
            result['missed_key'] = start_missed(user, lang, bib, texts, pdf)
        Generated.objects.filter(pk=pk).update(status='done', content=json.dumps(result), updated=timezone.now())
        ready_email(pk, result)  # once the link works: a check that reads full texts takes minutes
        return result
    return work


def ready_email(pk, result):
    """Tell the researcher their check is ready, with what it found."""
    from django.conf import settings
    from django.core.mail import EmailMultiAlternatives
    from django.template.loader import render_to_string
    from django.utils import translation
    from . import mailcheck
    row = Generated.objects.select_related('user').filter(pk=pk).first()
    user = row.user if row is not None else None
    if user is None or not user.email or mailcheck.undeliverable_reason(user.email):
        return
    with translation.override(row.lang):
        context = {'user': user, 'result': result, 'claims': len(result.get('claims', [])),
                   'link': '%s/%s/refcheck/%s/' % (digest.BASE_URL, row.lang, row.key)}
        subject = render_to_string('summarizer/refcheck_email_subject.txt', context).strip()
        text = render_to_string('summarizer/refcheck_email.txt', context)
    try:
        EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email]).send()
    except Exception:
        logger.exception('Could not send the bibliography check %s to user %s', row.key, user.pk)


def start_missed(user, lang, bib, texts, pdf):
    """'What did I miss?' on the same manuscript and references; '' when there is no text to read."""
    from . import missed
    draft = '\n\n'.join(' '.join(refcheck_claims.readable(s) for _line, s in refcheck_claims.sentences(t)) for _n, t in texts)
    if pdf is None and len(draft.strip()) < 200:
        return ''
    row = generated.start('missed', uuid.uuid4().hex, lang, missed.write(pdf, draft, lang, bib), user=user)
    return row.key


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def free_recheck(previous):
    """The first check again of a report is not counted: it is the use the tool is for."""
    return previous is not None and cache.add(RECHECK_KEY % previous.key, True, 90 * 86400)


def borrowed(request, value):
    """(label, BibTeX) of the references of one of the user's collections or literature reviews, None otherwise."""
    kind, _sep, key = (value or '').partition(':')
    try:
        if kind == 'collection' and key.isdigit():
            from . import library
            collection = library.editable(request, int(key))
            return collection.name, library.collection_export(request, collection.pk, 'bib').content.decode('utf-8')
        if kind == 'literature' and key:
            from . import literature
            row = Generated.objects.filter(kind='literature', key=key, user=request.user, status='done').first()
            if row is not None:
                result = row.data()
                return result.get('title') or _('Literature review'), literature.export_result(row, 'bib').content.decode('utf-8')
    except Http404:
        return None
    return None


def _text(upload, pasted):
    if upload is None:
        return pasted
    if upload.size > MAX_UPLOAD:
        return None
    return refcheck_sources.decode(upload.read())


def loader(bib, tex, tex_name, archive, pdf, arxiv, zotero):
    """What the check reads, gathered in the background: (bib, texts, origin, PDF bytes or None)."""
    def load():
        texts, origin, references = [], 'bib', bib
        if zotero is not None:
            references = refcheck_sources.zotero_bibtex(*zotero) + '\n\n' + references
        files = refcheck_sources.archive_files(archive) if archive is not None else \
            refcheck_sources.arxiv_source(arxiv) if arxiv else None
        if files is not None:
            texts = refcheck_sources.manuscript(files)
            if not parse_bibtex(references):
                references, origin = refcheck_sources.references_of(files, texts)
        if tex:
            texts = [(tex_name, tex)]
        if pdf is not None and not parse_bibtex(references):
            references, origin = refcheck_sources.pdf_bibtex(pdf), 'pdf'
        if not parse_bibtex(references):
            raise refcheck_sources.Unreadable('no_references')
        return references, texts, origin, pdf
    return load


def refcheck_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/refcheck.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'refcheck')
    error = None
    previous = Generated.objects.filter(kind='refcheck', user=request.user, status='done',
                                        key=request.POST.get('previous') or request.GET.get('previous') or '-').first()
    zotero = ZoteroAccount.objects.filter(user=request.user).first() if access.has_tool(request.user, 'zotero') else None
    origin = request.POST.get('from') or request.GET.get('from') or ''
    lent = borrowed(request, origin) if origin else None
    if request.method == 'POST' and paid:
        bib = _text(request.FILES.get('bib'), request.POST.get('bibtext', ''))
        if lent is not None and bib is not None:
            bib = lent[1] + '\n\n' + bib
        source = request.FILES.get('source')
        tex, tex_name, archive, pdf = request.POST.get('textext', '').strip(), 'main.tex', None, None
        arxiv = refcheck_sources.arxiv_id(request.POST.get('arxiv', ''))
        collection = request.POST.get('zotero', '').strip()
        name = (source.name if source is not None else '').lower()
        if bib is None or (source is not None and source.size > (MAX_ARCHIVE if name.endswith(('.zip', '.gz', '.tgz', '.tar')) else MAX_UPLOAD)):
            error = 'file'
        elif request.POST.get('arxiv', '').strip() and not arxiv:
            error = 'arxiv'
        elif source is not None and not name.endswith(('.tex', '.txt', '.zip', '.gz', '.tgz', '.tar', '.pdf')):
            error = 'format'
        else:
            if source is not None:
                data = source.read()
                if name.endswith('.pdf'):
                    pdf = data
                elif name.endswith(('.tex', '.txt')):
                    tex, tex_name = refcheck_sources.decode(data), source.name
                else:
                    archive = data
            if bib.strip() and not parse_bibtex(bib):
                error = 'empty'
            elif not (bib.strip() or archive is not None or arxiv or pdf is not None or (collection and zotero)):
                error = 'empty'
            elif not (free_recheck(previous) or _spend(request.user)):
                error = 'quota'
            else:
                with_missed = bool(request.POST.get('missed')) and access.has_tool(request.user, 'missed') and \
                    missed_quota(request.user)
                load = loader(bib, tex, tex_name, archive, pdf, arxiv, (zotero, collection) if collection and zotero else None)
                row = generated.start('refcheck', uuid.uuid4().hex, get_language(),
                                      write(load, get_language(), request.user, previous, with_missed), user=request.user)
                return redirect('/%s/refcheck/%s/' % (get_language(), row.key))
    checks = Generated.objects.filter(kind='refcheck', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/refcheck.html', {
        'paid': paid, 'error': error, 'checks': checks, 'per_day': PER_DAY, 'previous': previous, 'zotero': zotero,
        'lent': lent, 'lent_from': origin if lent else '',
        'recheck_free': previous is not None and cache.get(RECHECK_KEY % previous.key) is None,
        'missed_allowed': access.has_tool(request.user, 'missed'),
        'upload_max_bytes': MAX_UPLOAD, 'upload_max_mb': MAX_UPLOAD // (1024 * 1024),
        'archive_max_bytes': MAX_ARCHIVE, 'archive_max_mb': MAX_ARCHIVE // (1024 * 1024)})


def missed_quota(user):
    from . import missed
    return missed._spend(user)


def _row(request, key):
    if not access.has_tool(request.user, 'refcheck'):
        raise Http404
    row = Generated.objects.filter(kind='refcheck', key=key, user=request.user).first()
    if row is None:
        raise Http404
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    return row


def shown(row):
    """The report as its page shows it: marked problems apart, counts of what is left to look at."""
    result = row.data() if row.status == 'done' else None
    if not result:
        return result
    rank = {level: i for i, level in enumerate(LEVELS)}
    for problem in result.get('problems', []):
        for issue in problem['issues']:
            issue['label'] = str(KIND_LABELS.get(issue['kind'], issue['kind']))
        problem['active'] = [i for i in problem['issues'] if not i.get('mark')]
        problem['marked'] = [i for i in problem['issues'] if i.get('mark')]
        problem['level'] = min((i['level'] for i in problem['active'] or problem['issues']), key=rank.get)
        problem['kinds'] = ' '.join(dict.fromkeys(i['kind'] for i in problem['issues']))
        problem['correction'] = (result.get('corrections') or {}).get(problem['key'])
    active = [p for p in result.get('problems', []) if p['active']]
    result['active'] = active
    result['done'] = [p for p in result.get('problems', []) if not p['active']]
    result['counts'] = {level: sum(any(i['level'] == level for i in p['active']) for p in active) for level in LEVELS}
    result['kinds'] = {}
    for problem in active:
        for issue in problem['active']:
            result['kinds'][issue['kind']] = result['kinds'].get(issue['kind'], 0) + 1
    for solved in (result.get('previous') or {}).get('solved', []):
        solved['label'] = str(KIND_LABELS.get(solved['kind'], solved['kind']))
    for index, claim in enumerate(result.get('claims', [])):
        claim['index'] = index
    result['open_claims'] = [c for c in result.get('claims', []) if not c.get('mark')]
    result['marked_claims'] = [c for c in result.get('claims', []) if c.get('mark')]
    result['new_count'] = sum(1 for p in active for i in p['active'] if i.get('new')) + \
        sum(1 for c in result['open_claims'] if c.get('new'))
    return result


def mark(row, target, value):
    """Mark an issue ('<entry key>|<kind>') or a flagged sentence ('claim|<index>'): intended, fixed, ignored or nothing."""
    data = row.data()
    value = value if value in MARKS else ''
    key, _sep, kind = target.rpartition('|')
    if key == 'claim':
        try:
            data['claims'][int(kind)]['mark'] = value
        except (KeyError, IndexError, ValueError):
            return False
    else:
        issues = [i for p in data.get('problems', []) if p['key'] == key for i in p['issues'] if i['kind'] == kind]
        if not issues:
            return False
        for issue in issues:
            issue['mark'] = value
    row.content = json.dumps(data)
    row.save(update_fields=['content', 'updated'])
    return True


def save_collection(request, row, result):
    from . import collection_tools, library
    paid = access.has_tool(request.user, 'collections')
    if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
        return redirect('/%s/refcheck/%s/?m=too_many' % (get_language(), row.key))
    ids = [i for i in dict.fromkeys(library.canonical_id(r['arxiv'] or r['doi']) for r in result.get('refs', [])) if i]
    if not ids:
        return redirect('/%s/refcheck/%s/?m=no_ids' % (get_language(), row.key))
    name = '%s · %s' % (_('References'), timezone.localtime(row.created).strftime('%Y-%m-%d'))
    collection = Collection.objects.create(user=request.user, name=name[:120])
    collection_tools.start_import(request, collection, ids, paid)
    return redirect('/%s/collections/%s/' % (get_language(), collection.pk))


@login_required
def refcheck_result(request, key):
    row = _row(request, key)
    if request.method == 'POST' and row.status == 'done':
        action = request.POST.get('action')
        if result_share.act(row, action):
            return redirect('/%s/refcheck/%s/' % (get_language(), row.key))
        if action == 'mark':
            done = mark(row, request.POST.get('target', ''), request.POST.get('mark', ''))
            if request.headers.get('X-Requested-With') == 'fetch':  # the page updates itself, without reloading
                return JsonResponse({'ok': done})
            return redirect('/%s/refcheck/%s/#%s' % (get_language(), row.key, request.POST.get('anchor', 'problems')))
        if action == 'watch':
            from . import refcheck_watch
            refcheck_watch.start(request.user, row, row.data())
            return redirect('/%s/refcheck/%s/?m=watching#next' % (get_language(), row.key))
        if action == 'unwatch':
            ReferenceWatch.objects.filter(user=request.user, check_key=row.key).delete()
            return redirect('/%s/refcheck/%s/#next' % (get_language(), row.key))
        if action == 'collection':
            return save_collection(request, row, row.data())
    result = shown(row)
    return render(request, 'summarizer/refcheck_result.html', {
        'row': row, 'result': result, 'steps': STEPS, 'message': request.GET.get('m'),
        'failure': FAILURES.get(row.error) if row.status == 'no_source' else None,
        'watch': ReferenceWatch.objects.filter(user=request.user, check_key=row.key).first(),
        'missed': Generated.objects.filter(kind='missed', key=result.get('missed_key'), user=request.user).first()
        if result and result.get('missed_key') else None,
        'share_url': result_share.shared_url(request, row, 'refcheck')})


def bib_file(data, selected=None, only=False):
    """The corrected .bib with the corrections chosen (all by default), or only the corrected entries."""
    corrections = data.get('corrections')
    if not corrections:
        return data.get('fixed_bib') or ''
    keys = [k for k in (selected if selected is not None else corrections) if k in corrections]
    if only:
        return '\n\n'.join(corrections[k]['after'] for k in keys) + '\n'
    text = data['fixed_bib']
    for key, correction in corrections.items():
        if key not in keys:
            text = text.replace(correction['after'], correction['before'], 1)
    return text


def _bib(request, row):
    data = row.data() if row.status == 'done' else {}
    if not data.get('fixed_bib'):
        raise Http404
    selected = request.GET.getlist('k') if request.GET.get('sel') else None
    only = bool(request.GET.get('only'))
    response = HttpResponse(bib_file(data, selected, only), content_type='application/x-bibtex; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="%s"' % ('corrected-entries.bib' if only else 'references-corrected.bib')
    return response


@login_required
def refcheck_bib(request, key):
    return _bib(request, _row(request, key))


def refcheck_shared(request, token):
    """The read-only page of a shared check, for co-authors: the report and its corrected .bib, never who ran it."""
    row = result_share.shared_row('refcheck', token)
    return render(request, 'summarizer/refcheck_result.html', {'row': row, 'result': shown(row), 'shared': True})


def refcheck_shared_bib(request, token):
    return _bib(request, result_share.shared_row('refcheck', token))
