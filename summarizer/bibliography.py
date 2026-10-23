"""Conservative reference resolution and field comparisons; no model-generated metadata.

Search results are candidates, not evidence of identity. A locator can identify a
paper despite a damaged title; similar titles cannot override conflicting locators.
"""
import hashlib
import html
import json
import re
import time
import unicodedata
from difflib import SequenceMatcher
from urllib.parse import quote

from django.conf import settings
from django.core.cache import cache

from pylatexenc.latex2text import LatexNodes2Text

from . import sources


def normalized(value):
    text = LatexNodes2Text().latex_to_text(html.unescape(sources._strip_tags(str(value or ""))))
    text = unicodedata.normalize("NFKD", text).casefold()
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"[^\w]+", " ", text).strip()

ADS = 'https://api.adsabs.harvard.edu/v1/search/query'
TTL = 86400
MAX_CALLS = 180
# Common astronomical BibTeX macros, abbreviations and full journal names.
JOURNALS = {
    'aap': ('A&A', 'Astronomy and Astrophysics', 'Astron. Astrophys.'),
    'aj': ('AJ', 'The Astronomical Journal', 'Astronomical Journal'),
    'apj': ('ApJ', 'The Astrophysical Journal', 'Astrophysical Journal'),
    'apjl': ('ApJL', 'The Astrophysical Journal Letters', 'Astrophysical Journal Letters'),
    'apjs': ('ApJS', 'The Astrophysical Journal Supplement Series'),
    'mnras': ('MNRAS', 'Monthly Notices of the Royal Astronomical Society'),
    'araa': ('ARA&A', 'Annual Review of Astronomy and Astrophysics'),
    'pasp': ('PASP', 'Publications of the Astronomical Society of the Pacific'),
}


def journal_key(value):
    raw = value.strip().strip('{}').lstrip('\\')
    key = normalized(raw.replace('&', ' and '))
    for macro, names in JOURNALS.items():
        if key in [macro] + [normalized(n.replace('&', ' and ')) for n in names]:
            return macro
    return re.sub(r'^the ', '', key)


def journal_name(value):
    aliases = JOURNALS.get(journal_key(value))
    return aliases[1] if aliases else value


def families(value):
    names = []
    for author in re.split(r'\s+and\s+', value, flags=re.I):
        name = normalized(author.split(',')[0] if ',' in author else author)
        if name and name not in ('others', 'et al'):
            names.append(name if ',' in author else name.split()[-1])
    return names


def pages(value):
    parts = re.split(r'\s*[-–—]+\s*', value.strip())
    parts = [normalized(p).replace(' ', '') for p in parts if p]
    if len(parts) == 2 and all(p.isdigit() for p in parts) and len(parts[1]) < len(parts[0]):
        parts[1] = parts[0][:-len(parts[1])] + parts[1]
    return parts


def first(value):
    return (value or [''])[0]


def crossref_record(work):
    doi = sources.clean_doi(work.get('DOI', '')) or ''
    kind = work.get('type', '')
    kind = 'book' if kind in ('book', 'monograph', 'edited-book', 'reference-book') else \
        'incollection' if kind == 'book-chapter' else 'article' if kind == 'journal-article' else ''
    dates = []
    for key in ('published-print', 'published', 'issued', 'published-online'):
        parts = (work.get(key) or {}).get('date-parts') or []
        if parts and parts[0] and parts[0][0]:
            dates.append(str(parts[0][0]))
    authors = [a.get('family', '') for a in work.get('author', []) if a.get('family')]
    fields = {'title': sources._strip_tags(first(work.get('title'))), 'journal': first(work.get('container-title')),
              'year': first(dates), 'volume': work.get('volume', ''),
              'pages': work.get('page') or work.get('article-number') or '', 'publisher': work.get('publisher', ''),
              'doi': doi, 'isbn': first(work.get('ISBN'))}
    return {'fields': {k: html.unescape(str(v)) for k, v in fields.items() if v}, 'type': kind, 'doi': doi,
            'authors': [normalized(a) for a in authors], 'years': dates, 'source': 'Crossref',
            'url': 'https://doi.org/' + quote(doi, safe='/') if doi else '', 'arxiv': ''}


def openalex_record(work):
    b = work.get('biblio') or {}
    journal = ((work.get('primary_location') or {}).get('source') or {}).get('display_name', '')
    fields = {'title': work.get('display_name', ''), 'journal': journal, 'year': str(work.get('publication_year') or ''),
              'volume': b.get('volume'), 'pages': '-'.join(str(v) for v in [b.get('first_page'), b.get('last_page')] if v)}
    doi = sources.clean_doi(work.get('doi', '')) or ''
    return {'fields': {k: html.unescape(str(v)) for k, v in fields.items() if v}, 'doi': doi,
            'type': {'article': 'article', 'book': 'book', 'book-chapter': 'incollection'}.get(work.get('type'), ''),
            'authors': [normalized((a.get('author') or {}).get('display_name', '')).split()[-1]
                        for a in work.get('authorships', []) if normalized((a.get('author') or {}).get('display_name', ''))],
            'years': [fields['year']], 'source': 'OpenAlex', 'url': work.get('id', ''), 'arxiv': sources.arxiv_of(work)}


def ads_record(work):
    authors = work.get('author') or []
    f = {'title': first(work.get('title')), 'journal': work.get('pub', ''), 'year': str(work.get('year') or ''),
         'volume': work.get('volume', ''), 'pages': '-'.join(work.get('page') or []),
         'doi': first(work.get('doi')), 'isbn': first(work.get('isbn'))}
    kind = {'article': 'article', 'book': 'book', 'inbook': 'incollection', 'inproceedings': 'inproceedings'}.get(work.get('doctype'), '')
    return {'fields': {k: str(v) for k, v in f.items() if v}, 'type': kind,
            'doi': sources.clean_doi(f['doi']) or '', 'authors': [normalized(a.split(',')[0]) for a in authors],
            'years': [f['year']], 'source': 'NASA ADS',
            'url': 'https://ui.adsabs.harvard.edu/abs/' + quote(work.get('bibcode', ''), safe='') + '/abstract', 'arxiv': ''}


def match_score(entry, candidate):
    f, c = entry['fields'], candidate['fields']
    own_authors = families(f.get('author', ''))
    authors = candidate['authors']
    if not own_authors or not authors or own_authors[0] not in authors:
        return 0
    # A shared surname alone is insufficient for a multi-author reference.
    if len(own_authors) > 1 and len(set(own_authors) & set(authors)) < 2:
        return 0
    title, other = normalized(f.get('title', '')), normalized(c.get('title', ''))
    similarity = SequenceMatcher(None, title, other).ratio() if title and other else 0
    checks = [journal_key(f['journal']) == journal_key(c['journal']) if f.get('journal') and c.get('journal') else None,
              normalized(f['volume']) == normalized(c['volume']) if f.get('volume') and c.get('volume') else None,
              pages(f['pages'])[:1] == pages(c['pages'])[:1] if f.get('pages') and c.get('pages') else None]
    year = f.get('year', '')
    same_year = year in candidate['years'] if year else False
    # Title-independent recovery requires ALL of author/year/journal/volume/first page.
    if same_year and all(v is True for v in checks):
        return 130 + similarity
    if f.get('publisher') and not f.get('journal') and candidate['type'] == 'article' and c.get('journal'):
        return 0  # a review of a book may repeat its title and name its author
    if similarity < .90 or len(title) < 20:
        return 0
    agree, conflict = checks.count(True), checks.count(False)
    if conflict >= 2 or (conflict and agree + int(same_year) < 2):
        return 0
    if candidate['type'] == 'book' and year and not same_year:
        # Edition dates differ between catalogues. Never replace a book by a later edition.
        if not any(y.isdigit() and year.isdigit() and abs(int(y) - int(year)) <= 1 for y in candidate['years']):
            return 0
    if not (same_year or agree or (candidate['type'] == 'book' and f.get('publisher') and
                                  normalized(f['publisher']) == normalized(c.get('publisher', '')))):
        return 0
    return 90 + agree * 5 + int(same_year) * 5 + similarity - (8 if candidate['type'] == 'book' and year and not same_year else 0)


def select(entry, candidates):
    scored = {}
    for c in candidates:
        score = match_score(entry, c) if c['url'] else 0
        if score:
            key = c['doi'] or c['url']
            if key not in scored or score > scored[key][0]:
                scored[key] = (score, c)
    rows = sorted(scored.values(), key=lambda row: row[0], reverse=True)
    if not rows:
        return None, False
    if len(rows) > 1 and rows[0][0] - rows[1][0] < 4:
        return None, True
    return rows[0][1], False


class Resolver:
    def __init__(self, openalex_allowed=lambda: True):
        self.remaining = MAX_CALLS
        self.deadline = time.monotonic() + 180
        self.unavailable = set()
        self.checked = set()
        self.limited = False
        self.ambiguous = False
        self.openalex_allowed = openalex_allowed

    def query(self, source, url, params=None):
        self.checked.add(source)
        key = 'bib-metadata:v1:' + hashlib.sha256(json.dumps([source, url, params], sort_keys=True).encode()).hexdigest()
        cached = cache.get(key)
        if cached is not None:
            return cached
        if not self.remaining or time.monotonic() > self.deadline:
            self.limited = True
            return {}
        if source == 'OpenAlex' and url.endswith('/works') and not self.openalex_allowed():
            self.limited = True
            return {}
        self.remaining -= 1
        try:
            if source == 'NASA ADS':
                response = sources._get(url, params, timeout=15, headers={'Authorization': 'Bearer ' + settings.ADS_TOKEN})
                if response.status_code != 200:
                    raise sources.SourceUnavailable('ADS unavailable')
                body = response.json()
            else:
                body = sources._json(url, sources._openalex_params(params) if source == 'OpenAlex' else params)
            body = body or {}
            if not isinstance(body, dict):
                raise sources.SourceUnavailable('Malformed bibliographic response')
        except (sources.SourceUnavailable, ValueError):
            self.unavailable.add(source)
            return {}
        cache.set(key, body, TTL)
        return body

    def doi(self, doi):
        body = self.query('Crossref', sources.CROSSREF % quote(doi, safe=''), {'mailto': 'contact@summarizepaper.com'})
        record = crossref_record(body.get('message') or {})
        return record if record['doi'] == doi and record['fields'].get('title') else None

    def crossref(self, params):
        body = self.query('Crossref', 'https://api.crossref.org/works', dict(params, rows=5, mailto='contact@summarizepaper.com'))
        return [crossref_record(w) for w in (body.get('message') or {}).get('items', [])]

    def ads(self, query):
        if not getattr(settings, 'ADS_TOKEN', ''):
            return []
        body = self.query('NASA ADS', ADS, {'q': query, 'rows': 5,
            'fl': 'bibcode,title,author,year,pub,volume,page,doi,isbn,doctype'})
        return [ads_record(w) for w in (body.get('response') or {}).get('docs', [])]

    def openalex(self, title):
        body = self.query('OpenAlex', sources.OPENALEX + '/works', {'search': title[:300], 'per_page': 5,
            'select': 'id,doi,display_name,type,publication_year,biblio,authorships,primary_location,locations'})
        return [openalex_record(w) for w in body.get('results', [])]

    def resolve(self, entry, doi=''):
        self.ambiguous = False
        if doi:
            record = self.doi(doi)
            if record:
                return record
            candidates = self.ads('doi:"%s"' % normalized_query(doi))
            return next((c for c in candidates if c['doi'] == doi), None)
        f = entry['fields']
        title, authors = f.get('title', ''), families(f.get('author', ''))
        if not authors:
            return None
        title = sources._strip_tags(title)
        author = authors[0]
        stages = []
        if title:
            title_candidates = self.crossref({'query.bibliographic': title[:300], 'query.author': author})
            title_candidates += self.ads('title:"%s" author:"%s"' % (normalized_query(title), normalized_query(author)))
            stages = [lambda: title_candidates, lambda: self.openalex(title),
                      lambda: self.crossref({'query.bibliographic': ' '.join([title[:220], f.get('year', ''), author]), 'query.author': author})]
        if all(f.get(k) for k in ('journal', 'volume', 'pages', 'year')) and re.fullmatch(r'\d{4}', f['year']):
            locator = ' '.join([journal_name(f['journal']), f['year'], f['volume'], first(pages(f['pages']))])
            stages += [lambda: self.ads('author:"%s" year:%s volume:"%s" page:"%s"' %
                (normalized_query(author), normalized_query(f['year']), normalized_query(f['volume']), normalized_query(first(pages(f['pages']))))),
                lambda: self.crossref({'query.bibliographic': locator, 'query.author': author,
                                       'query.container-title': journal_name(f['journal'])})]
        candidates = []
        for stage in stages:
            candidates.extend(stage())
            record, ambiguous = select(entry, candidates)
            self.ambiguous |= ambiguous
            if record:
                if record['doi'] and record['source'] != 'Crossref':
                    detail = self.doi(record['doi'])
                    if detail and match_score(entry, detail):
                        return detail
                return record
        return None


def normalized_query(value):
    """Keep user BibTeX out of ADS's query syntax (fixed field names/operators only)."""
    return re.sub(r'[^\w\s.,/&-]', ' ', value)[:300]


def differences(entry, record):
    f, c = entry['fields'], record['fields']
    found = []
    for field in ('title', 'year', 'journal', 'volume', 'pages'):
        before, after = f.get(field, ''), c.get(field, '')
        if not before or not after:
            continue  # missing source fields are not evidence of an error
        a, b = normalized(before), normalized(after)
        if field == 'year':
            if before in record['years']:
                continue  # online and print years may both be legitimate
            if record['type'] == 'book' and any(y.isdigit() and before.isdigit() and abs(int(y)-int(before)) <= 1 for y in record['years']):
                continue  # catalogue/edition ambiguity, not a proven wrong year
        if field == 'journal':
            a, b = journal_key(before), journal_key(after)
        if field == 'pages':
            a, b = pages(before), pages(after)
            if a[:1] == b[:1] and (len(a) == 1 or len(b) == 1):
                continue
        if a != b:
            found.append({'field': field, 'before': before, 'after': after})
    if record['type'] and entry['type'] != record['type'] and \
            (record['type'] == 'book' or (entry['type'] == 'book' and record['type'] == 'article')):
        found.append({'field': 'type', 'before': '@' + entry['type'], 'after': '@' + record['type']})
    return found
