"""Papers from other sources than arXiv: journals and preprint servers.

A paper is known by a key that takes the place of the arXiv id everywhere
(URLs, ArxivPaper.arxiv_id, PaperText, digests...):

    doi_<doi>       a DOI, lowercased, '/' written '--' as for old arXiv ids
                    ('10.1093/mnras/stad1234' -> 'doi_10.1093--mnras--stad1234')
    doi_h<hash>     a DOI with characters a URL or a channel group cannot hold
    hal_<id>        a HAL deposit without DOI ('hal_hal-01234567')
    pmc_<pmcid>     a Europe PMC article without DOI ('pmc_pmc1234567')

Anything else is an arXiv id. Where the metadata and the text come from:

    bioRxiv, medRxiv   api.biorxiv.org; text from the JATS XML, else the PDF
    OSF preprints      api.osf.io (PsyArXiv, SocArXiv...); text from the PDF
    HAL                api.archives-ouvertes.fr; text from the PDF
    Europe PMC         ebi.ac.uk/europepmc; text from the JATS XML
    any other DOI      OpenAlex (single records are free), Crossref if OpenAlex
                       fails; text from Europe PMC, an open PDF, or the copy
                       OpenAlex keeps (content API, needs OPENALEX_API_KEY)

As for arXiv, the full text is read only when the license allows building on
it (see is_open_license); otherwise the summary is written from the abstract.
Metadata is kept in ExternalPaper and refreshed at most once a day.
"""
import datetime
import functools
import hashlib
import json
import logging
import re
import urllib.parse
from xml.etree import ElementTree

import requests
from django.conf import settings
from django.utils import timezone

logger = logging.getLogger(__name__)

USER_AGENT = 'SummarizePaper/1.0 (+https://www.summarizepaper.com; mailto:contact@summarizepaper.com)'
TIMEOUT = 20
PDF_TIMEOUT = 120
MAX_PDF_BYTES = 60 * 1024 * 1024
MIN_TEXT = 2000  # shorter extractions are a cover page or an error page, not the paper
REFRESH = datetime.timedelta(days=1)
# The free OpenAlex key gives $1 a day, shared by searches ($0.001) and the full texts it keeps ($0.01 each):
# at most this many texts a day, so a busy day of summaries never leaves the search without budget.
OPENALEX_CONTENT_PER_DAY = 50
OPENALEX_CONTENT_KEY = 'openalex-content:%s'
MAX_KEY = 90  # channel group names ('ar_<key>_<lang>') must stay under 100 characters

OPENALEX = 'https://api.openalex.org'
OPENALEX_CONTENT = 'https://content.openalex.org/works/%s.%s'
CROSSREF = 'https://api.crossref.org/works/%s'
BIORXIV = 'https://api.biorxiv.org/details/%s/%s/na/json'
BIORXIV_PDF = 'https://www.%s.org/content/%sv%s.full.pdf'
OSF = 'https://api.osf.io/v2/preprints/%s/'
HAL = 'https://api.archives-ouvertes.fr/search/'
HAL_FIELDS = ('halId_s,title_s,abstract_s,authFullName_s,producedDate_s,uri_s,fileMain_s,licence_s,'
              'doiIdentifier_s,journalTitle_s,domain_s,docType_s')
EPMC = 'https://www.ebi.ac.uk/europepmc/webservices/rest/search'
EPMC_XML = 'https://www.ebi.ac.uk/europepmc/webservices/rest/%s/fullTextXML'

BIORXIV_PREFIXES = ('10.1101/', '10.64898/')  # openRxiv moved to 10.64898 in 2025
OSF_PROVIDERS = {
    'osf': 'OSF Preprints', 'psyarxiv': 'PsyArXiv', 'socarxiv': 'SocArXiv', 'edarxiv': 'EdArXiv',
    'engrxiv': 'engrXiv', 'metaarxiv': 'MetaArXiv', 'sportrxiv': 'SportRxiv', 'lawarxiv': 'LawArXiv',
    'mindrxiv': 'MindRxiv', 'africarxiv': 'AfricArXiv', 'eartharxiv': 'EarthArXiv', 'ecoevorxiv': 'EcoEvoRxiv',
}

DOI_RE = re.compile(r'\b(10\.\d{4,9}/[^\s"<>]+)', re.I)
SAFE_DOI = re.compile(r'^10\.\d{4,9}/[a-z0-9._\-/]+$')
KEY_RE = re.compile(r'^(doi_[a-z0-9._\-]+|hal_[a-z][a-z0-9]*(?:-[a-z0-9]+)*-\d{6,}(?:v\d+)?|pmc_pmc\d+)$')
HAL_ID_RE = re.compile(r'\b((?:hal|halshs|hal-pasteur|tel|inria|inserm|cea|ird|insu|in2p3|cel|medihal|sic|edutice|'
                       r'dumas|ineris|irsn|pastel|memsic|halsde|emse|ensl|mnhn|ijn|hceres|anses)-\d{8})(?:v\d+)?\b', re.I)
PMC_RE = re.compile(r'\b(PMC\d{4,9})\b', re.I)
PMID_RE = re.compile(r'(?:pmid[:\s]*|pubmed\.ncbi\.nlm\.nih\.gov/)(\d{5,9})\b', re.I)
OSF_URL_RE = re.compile(r'osf\.io/(?:preprints/[a-z]+/)?([a-z0-9]{5}(?:_v\d+)?)\b', re.I)
ARXIV_DOI_RE = re.compile(r'^10\.48550/arxiv\.(.+)$', re.I)
ARXIV_ABS_RE = re.compile(r'arxiv\.org/(?:abs|pdf)/([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})', re.I)

OPEN_LICENSES = {'by', 'by-sa', 'by-nc', 'by-nc-sa', 'zero', 'mark'}

_session = requests.Session()
_session.headers['User-Agent'] = USER_AGENT


class SourceUnavailable(Exception):
    """The source did not answer (or not in a way we understand); the page asks to retry later."""


def in_thread(function, *args):
    """Run a lookup in a worker thread (asyncio.to_thread): its slow downloads hold neither the event loop
    nor Channels' single thread for database calls, and its database connection is closed afterwards.

    Every function run in a thread we start ourselves (asyncio.to_thread, a ThreadPoolExecutor) must go through
    this when it may touch the database, even indirectly (a search remembers papers, an AI call counts its
    tokens): Django gives a connection back to the pool only when its thread closes it, and the pool holds 8.
    Worker threads that kept theirs blocked the whole site once a day."""
    from django.db import connections
    try:
        return function(*args)
    finally:
        connections.close_all()


def closing(function):
    """function, run through in_thread: for pool.map(closing(f), items) and pool.submit(closing(f), ...)."""
    return functools.partial(in_thread, function)


# --- keys --------------------------------------------------------------------

def is_external(key):
    return bool(key) and key.startswith(('doi_', 'hal_', 'pmc_'))


def valid_key(key):
    return len(key) <= MAX_KEY and bool(KEY_RE.match(key))


def clean_doi(text):
    """'https://doi.org/10.1101/2024.01.01.123456v2.' -> '10.1101/2024.01.01.123456' (None without a DOI)."""
    match = DOI_RE.search(urllib.parse.unquote(text or ''))
    if not match:
        return None
    doi = match.group(1).rstrip('.,;)]}').lower()
    for suffix in ('.full.pdf', '.full', '.abstract', '.pdf', '/full', '/abstract', '/pdf'):
        if doi.endswith(suffix):
            doi = doi[:-len(suffix)]
    if doi.startswith(BIORXIV_PREFIXES):
        doi = re.sub(r'v\d+$', '', doi)  # a bioRxiv link carries the version, the DOI does not
    return doi


def doi_key(doi):
    doi = doi.lower()
    encoded = doi.replace('/', '--')
    if SAFE_DOI.match(doi) and encoded.replace('--', '/') == doi and len(encoded) + 4 <= MAX_KEY:
        return 'doi_' + encoded
    return 'doi_h' + hashlib.sha1(doi.encode()).hexdigest()[:24]


def key_doi(key):
    """The DOI of a doi_ key ('' for other keys or an unknown hashed DOI)."""
    if not key.startswith('doi_'):
        return ''
    if key.startswith('doi_h'):
        from .models import ExternalPaper
        return ExternalPaper.objects.filter(key=key).values_list('doi', flat=True).first() or ''
    return key[4:].replace('--', '/')


def label(key):
    """How the identifier is shown: 'DOI: 10.1093/mnras/stad1234', 'HAL: hal-01234567'..."""
    if key.startswith('doi_'):
        return 'DOI: %s' % key_doi(key)
    if key.startswith('hal_'):
        return 'HAL: %s' % key[4:]
    if key.startswith('pmc_'):
        return 'Europe PMC: %s' % key[4:].upper()
    return 'arXiv: %s' % key.replace('--', '/')


def ref_url(key):
    """Where the paper lives outside SummarizePaper."""
    if key.startswith('doi_'):
        return 'https://doi.org/%s' % key_doi(key)
    if key.startswith('hal_'):
        return 'https://hal.science/%s' % key[4:]
    if key.startswith('pmc_'):
        return 'https://europepmc.org/article/PMC/%s' % key[4:].upper()
    return 'https://arxiv.org/abs/%s' % key.replace('--', '/')


# --- licenses ----------------------------------------------------------------

def normalize_license(value):
    """A license as each source writes it -> a creativecommons.org URL ('' when it is not a CC license)."""
    text = (value or '').strip().lower()
    if not text:
        return ''
    if 'arxiv.org/licenses' in text:
        return value.strip()
    match = re.search(r'creativecommons\.org/(licenses|publicdomain)/([a-z\-]+)(?:/(\d\.\d))?', text)
    if match:
        kind, code, version = match.groups()
        return 'https://creativecommons.org/%s/%s/%s' % (kind, code, version + '/' if version else '')
    if '/' in text:
        return ''  # a publisher's own terms (elsevier.com/tdm/userlicense...), whatever words its URL has
    if 'cc0' in text or 'zero' in text:
        return 'https://creativecommons.org/publicdomain/zero/1.0/'
    if text in ('public-domain', 'pd', 'public domain'):
        return 'https://creativecommons.org/publicdomain/mark/1.0/'
    words = re.sub(r'[^a-z0-9.]+', ' ', text.replace('attribution', 'by').replace('noncommercial', 'nc')
                   .replace('non-commercial', 'nc').replace('sharealike', 'sa').replace('share-alike', 'sa')
                   .replace('noderivatives', 'nd').replace('no-derivatives', 'nd').replace('noderivs', 'nd')).split()
    if 'by' not in words:
        return ''
    code = 'by' + ''.join('-' + part for part in ('nc', 'sa', 'nd') if part in words)
    if code in ('by-sa-nd', 'by-nc-sa-nd'):
        return ''
    version = next((w for w in words if re.match(r'^\d\.\d$', w)), '')
    return 'https://creativecommons.org/licenses/%s/%s' % (code, version + '/' if version else '')


def license_code(url):
    """'https://creativecommons.org/licenses/by-nc/4.0/' -> ('by-nc', '4.0'); arXiv's own -> ('nonexclusive-distrib', '1.0')."""
    match = re.search(r'/(?:licenses|publicdomain)/([a-z0-9\-]+)(?:/(\d\.\d))?', (url or '').lower())
    return (match.group(1), match.group(2) or '') if match else ('', '')


def license_label(url):
    """'CC BY-NC 4.0', 'NONEXCLUSIVE-DISTRIB 1.0' or '' for the license shown on the paper page."""
    code, version = license_code(url)
    if not code:
        return ''
    if 'creativecommons.org' in url.lower():
        name = 'CC0' if code == 'zero' else 'Public domain' if code == 'mark' else 'CC ' + code.upper()
        return ('%s %s' % (name, version)).strip() if code != 'mark' else name
    return ('%s %s' % (code.upper(), version)).strip()


def is_open_license(url):
    """Whether the license lets us build on the paper (summaries are derivatives): no ND, no publisher terms."""
    code, _ = license_code(url)
    return code in OPEN_LICENSES and 'creativecommons.org' in (url or '').lower()


# --- HTTP --------------------------------------------------------------------

def _get(url, params=None, timeout=TIMEOUT, **kwargs):
    try:
        response = _session.get(url, params=params, timeout=timeout, **kwargs)
    except requests.RequestException as e:
        raise SourceUnavailable('%s: %r' % (url, e))
    return response


def _json(url, params=None, missing=(404,)):
    """Decoded JSON, None when the source says the record does not exist."""
    response = _get(url, params)
    if response.status_code in missing:
        return None
    if response.status_code != 200:
        raise SourceUnavailable('%s: HTTP %s' % (url, response.status_code))
    try:
        return response.json()
    except ValueError:
        raise SourceUnavailable('%s: not JSON' % url)


def _openalex_params(params=None):
    params = dict(params or {})
    if getattr(settings, 'OPENALEX_API_KEY', ''):
        params['api_key'] = settings.OPENALEX_API_KEY
    else:
        params['mailto'] = 'contact@summarizepaper.com'
    return params


def _clean(text):
    return re.sub(r'\s+', ' ', text or '').strip()


def _strip_tags(text):
    return _clean(re.sub(r'<[^>]+>', ' ', text or ''))


def _date(value):
    """'2024-10-01', '2024-10', [2024, 10, 1] or a datetime string -> '2024-10-01' ('' when unknown)."""
    if isinstance(value, (list, tuple)):
        parts = [int(p) for p in value[:3]] + [1] * (3 - len(value[:3]))
        return '%04d-%02d-%02d' % tuple(parts) if value else ''
    match = re.match(r'(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?', str(value or ''))
    if not match:
        return ''
    year, month, day = match.groups()
    return '%s-%02d-%02d' % (year, int(month or 1), int(day or 1))


def _paper(**fields):
    """A paper as every source returns it; 'text' lists where its full text may be read, best first."""
    paper = {'key': '', 'source': '', 'source_name': '', 'title': '', 'abstract': '', 'authors': [],
             'affiliations': [], 'published': '', 'doi': '', 'url': '', 'journal': '', 'category': '',
             'license': '', 'text': [], 'arxiv_id': ''}
    paper.update(fields)
    paper['license'] = normalize_license(paper['license'])
    paper['title'] = _clean(paper['title'])
    paper['abstract'] = _clean(paper['abstract'])
    paper['affiliations'] = (paper['affiliations'] + [''] * len(paper['authors']))[:len(paper['authors'])]
    return paper


# --- bioRxiv and medRxiv -------------------------------------------------------

def _biorxiv_name(name):
    """'Fenstermacher, S.' -> 'S. Fenstermacher'."""
    last, _, first = name.partition(',')
    return _clean('%s %s' % (first, last)) if first.strip() else _clean(last)


def category_code(server, category):
    """('biorxiv', 'Cell Biology') -> 'biorxiv.cell-biology', the form digests store."""
    slug = re.sub(r'[^a-z]+', '-', (category or '').lower()).strip('-')
    return '%s.%s' % (server, slug) if slug else server


def biorxiv_paper(row, server):
    doi = row['doi'].lower()
    published = row.get('published') or ''
    return _paper(
        key=doi_key(doi), source=server, source_name='bioRxiv' if server == 'biorxiv' else 'medRxiv',
        title=row.get('title', ''), abstract=row.get('abstract', ''),
        authors=[_biorxiv_name(a) for a in (row.get('authors') or '').split(';') if a.strip()],
        published=_date(row.get('date')), doi=doi,
        url='https://www.%s.org/content/%sv%s' % (server, doi, row.get('version', '1')),
        journal='%s preprint%s' % ('bioRxiv' if server == 'biorxiv' else 'medRxiv',
                                   ', published as doi:%s' % published if published not in ('', 'NA') else ''),
        category=category_code(server, row.get('category')), license=_biorxiv_license(row.get('license')),
        text=[('jats', row['jatsxml'])] * bool(row.get('jatsxml')) +
             [('pdf', BIORXIV_PDF % (server, doi, row.get('version', '1')))])


def _biorxiv_license(value):
    """bioRxiv writes 'cc_by', 'cc_by_nc_nd', 'cc0', 'cc_no' (no reuse)."""
    value = (value or '').lower()
    if value in ('cc_no', 'none', ''):
        return ''
    if value.startswith('cc0'):
        return 'cc0'
    return 'cc ' + value[3:].replace('_', '-') + ' 4.0' if value.startswith('cc_') else value


def fetch_biorxiv(doi):
    for server in ('biorxiv', 'medrxiv'):
        body = _json(BIORXIV % (server, doi))
        rows = (body or {}).get('collection') or []
        if rows:
            latest = max(rows, key=lambda r: int(r.get('version') or 1))
            paper = biorxiv_paper(latest, server)
            first = min(rows, key=lambda r: int(r.get('version') or 1))
            paper['published'] = _date(first.get('date')) or paper['published']
            return paper
    return None


# --- OSF preprints -------------------------------------------------------------

def fetch_osf(osf_id):
    body = _json(OSF % osf_id, missing=(404, 410))
    if not body:
        return None
    data = body['data']
    attributes, relations = data['attributes'], data.get('relationships', {})

    def related(name):
        link = (relations.get(name) or {}).get('links', {}).get('related')
        return link.get('href') if isinstance(link, dict) else link

    provider = (related('provider') or '').rstrip('/').rsplit('/', 1)[-1]
    authors = []
    href = related('bibliographic_contributors')
    if href:
        contributors = _json(href, {'embed': 'users', 'page[size]': 100}) or {}
        for row in contributors.get('data', []):
            user = ((row.get('embeds') or {}).get('users') or {}).get('data') or {}
            name = (user.get('attributes') or {}).get('full_name') or (row.get('attributes') or {}).get('unregistered_contributor')
            if name:
                authors.append(_clean(name))
    license = ''
    if related('license'):
        lic = (_json(related('license')) or {}).get('data', {}).get('attributes', {})
        license = lic.get('url') or lic.get('name') or ''
    text = []
    if related('primary_file'):
        file = (_json(related('primary_file')) or {}).get('data') or {}
        name = (file.get('attributes') or {}).get('name') or ''
        if file.get('links', {}).get('download') and name.lower().endswith('.pdf'):
            text.append(('pdf', file['links']['download']))
    doi = (data.get('links', {}).get('preprint_doi') or '').split('doi.org/')[-1].lower()
    return _paper(
        key=doi_key(doi) if doi else '', source='osf', source_name=OSF_PROVIDERS.get(provider, provider or 'OSF'),
        title=attributes.get('title', ''), abstract=attributes.get('description', ''), authors=authors,
        published=_date(attributes.get('date_published') or attributes.get('date_created')), doi=doi,
        url=data.get('links', {}).get('html', ''), journal='%s preprint' % OSF_PROVIDERS.get(provider, 'OSF'),
        category='osf.%s' % provider if provider else 'osf', license=license, text=text)


# --- HAL -----------------------------------------------------------------------

def hal_paper(doc):
    hal_id = doc['halId_s'].lower()
    doi = (doc.get('doiIdentifier_s') or '').lower()
    first = lambda v: (v[0] if isinstance(v, list) and v else v) or ''  # noqa: E731
    return _paper(
        key=doi_key(doi) if doi else 'hal_' + hal_id, source='hal', source_name='HAL',
        title=first(doc.get('title_s')), abstract=first(doc.get('abstract_s')),
        authors=[_clean(a) for a in doc.get('authFullName_s') or []], published=_date(doc.get('producedDate_s')),
        doi=doi, url=doc.get('uri_s') or 'https://hal.science/%s' % hal_id,
        journal=doc.get('journalTitle_s') or 'HAL', category='hal.%s' % first(doc.get('domain_s')).split('.')[-1]
        if doc.get('domain_s') else 'hal', license=doc.get('licence_s') or '',
        text=[('pdf', doc['fileMain_s'])] if doc.get('fileMain_s') else [])


def fetch_hal(hal_id):
    body = _json(HAL, {'q': 'halId_s:"%s"' % hal_id.lower(), 'fl': HAL_FIELDS, 'rows': 1, 'wt': 'json'})
    docs = ((body or {}).get('response') or {}).get('docs') or []
    return hal_paper(docs[0]) if docs else None


# --- Europe PMC ----------------------------------------------------------------

def epmc_paper(row):
    pmcid = (row.get('pmcid') or '').lower()
    doi = (row.get('doi') or '').lower()
    authors = (row.get('authorList') or {}).get('author') or []
    affiliations = []
    for author in authors:
        details = ((author.get('authorAffiliationDetailsList') or {}).get('authorAffiliation') or [{}])[0]
        affiliations.append(_clean(details.get('affiliation', ''))[:300])
    open_text = pmcid and row.get('isOpenAccess') == 'Y' and row.get('inEPMC') == 'Y'
    return _paper(
        key=doi_key(doi) if doi else ('pmc_' + pmcid if pmcid else ''), source='europepmc', source_name='Europe PMC',
        title=_strip_tags(row.get('title', '')), abstract=_strip_tags(row.get('abstractText', '')),
        authors=[_clean(a.get('fullName', '')) for a in authors if a.get('fullName')], affiliations=affiliations,
        published=_date(row.get('firstPublicationDate')), doi=doi,
        url='https://europepmc.org/article/%s/%s' % (row.get('source', 'MED'), row.get('id', '')),
        journal=((row.get('journalInfo') or {}).get('journal') or {}).get('title') or row.get('bookOrReportDetails', {}).get('publisher', ''),
        category='europepmc', license=row.get('license') or '',
        text=[('epmc', pmcid.upper())] if open_text else [])


def epmc_search(query, page_size=1):
    body = _json(EPMC, {'query': query, 'format': 'json', 'resultType': 'core', 'pageSize': page_size})
    return ((body or {}).get('resultList') or {}).get('result') or []


def fetch_epmc(pmcid):
    rows = epmc_search('PMCID:%s' % pmcid.upper())
    return epmc_paper(rows[0]) if rows else None


# --- OpenAlex and Crossref -------------------------------------------------------

OPENALEX_FIELDS = ('id,doi,display_name,authorships,abstract_inverted_index,publication_date,primary_location,'
                   'best_oa_location,locations,primary_topic,ids,has_content,type')


def inverted_abstract(index):
    """OpenAlex gives abstracts as {word: [positions]}."""
    if not index:
        return ''
    words = sorted((pos, word) for word, positions in index.items() for pos in positions)
    return ' '.join(word for _, word in words)


def arxiv_of(work):
    """arXiv id of the arXiv version OpenAlex lists for a work ('' if none)."""
    for location in work.get('locations') or []:
        for url in (location.get('landing_page_url') or '', location.get('pdf_url') or ''):
            match = ARXIV_ABS_RE.search(url)
            if match:
                return match.group(1)
        match = ARXIV_DOI_RE.match((location.get('doi') or '').split('doi.org/')[-1])
        if match:
            return match.group(1)
    return ''


def openalex_paper(work):
    doi = (work.get('doi') or '').split('doi.org/')[-1].lower()
    primary = work.get('primary_location') or {}
    best = work.get('best_oa_location') or {}
    venue = (primary.get('source') or {}).get('display_name') or ''
    topic = work.get('primary_topic') or {}
    text = []
    pmcid = (work.get('ids') or {}).get('pmcid') or ''
    if pmcid:
        text.append(('epmc', pmcid.rstrip('/').rsplit('/', 1)[-1].upper()))
    # repositories first: publishers often refuse scripts
    locations = sorted(work.get('locations') or [], key=lambda l: ((l.get('source') or {}).get('type') != 'repository'))
    for location in locations:
        url = location.get('pdf_url')
        if url and 'arxiv.org' not in url and ('pdf', url) not in text:
            text.append(('pdf', url))
    if doi and not pmcid:
        text.append(('epmc_doi', doi))  # Europe PMC often has the open text OpenAlex does not link
    arxiv_id = arxiv_of(work)
    if arxiv_id:
        text.append(('arxiv', arxiv_id))  # publishers often refuse scripts; arXiv is free and has the same paper
    work_id = (work.get('id') or '').rsplit('/', 1)[-1]
    content = work.get('has_content') or {}
    if work_id and content.get('grobid_xml'):
        text.append(('openalex_tei', work_id))
    if work_id and content.get('pdf'):
        text.append(('openalex_pdf', work_id))
    authors, affiliations = [], []
    for authorship in work.get('authorships') or []:
        name = (authorship.get('author') or {}).get('display_name') or authorship.get('raw_author_name')
        if name:
            authors.append(name)
            affiliations.append(', '.join(i.get('display_name', '') for i in (authorship.get('institutions') or [])[:2]))
    is_preprint = work.get('type') == 'preprint'
    return _paper(
        key=doi_key(doi) if doi else '', source='openalex', source_name=venue or 'OpenAlex',
        title=work.get('display_name') or '', abstract=inverted_abstract(work.get('abstract_inverted_index')),
        authors=authors, affiliations=affiliations, published=_date(work.get('publication_date')), doi=doi,
        url=primary.get('landing_page_url') or ('https://doi.org/%s' % doi if doi else ''),
        journal=('%s preprint' % venue if is_preprint and venue else venue),
        category=((topic.get('field') or {}).get('display_name') or '')[:50],
        license=best.get('license') or '', text=text, arxiv_id=arxiv_id)


def fetch_openalex(doi):
    work = _json('%s/works/doi:%s' % (OPENALEX, urllib.parse.quote(doi, safe='/')),
                 _openalex_params({'select': OPENALEX_FIELDS}))
    return openalex_paper(work) if work else None


def fetch_crossref(doi):
    body = _json(CROSSREF % urllib.parse.quote(doi, safe='/'), {'mailto': 'contact@summarizepaper.com'})
    if not body:
        return None
    work = body['message']
    authors, affiliations = [], []
    for author in work.get('author') or []:
        name = _clean('%s %s' % (author.get('given', ''), author.get('family', ''))) or author.get('name', '')
        if name:
            authors.append(name)
            affiliations.append(', '.join(a.get('name', '') for a in (author.get('affiliation') or [])[:2]))
    licenses = [l for l in work.get('license') or [] if l.get('content-version') in ('vor', 'unspecified')] or work.get('license') or []
    pdfs = [l['URL'] for l in work.get('link') or [] if l.get('content-type') == 'application/pdf' and l.get('URL')]
    date = (work.get('published') or work.get('issued') or {}).get('date-parts') or [[]]
    return _paper(
        key=doi_key(doi), source='crossref', source_name=(work.get('container-title') or [''])[0] or 'Crossref',
        title=(work.get('title') or [''])[0], abstract=_strip_tags(work.get('abstract', '')),
        authors=authors, affiliations=affiliations, published=_date(date[0]), doi=doi,
        url=work.get('URL') or 'https://doi.org/%s' % doi, journal=(work.get('container-title') or [''])[0],
        category=((work.get('subject') or [''])[0])[:50], license=licenses[0]['URL'] if licenses else '',
        text=[('pdf', url) for url in pdfs])


def fetch_doi(doi):
    """The paper of a DOI from the source that knows it best; None if no source knows it."""
    if doi.startswith(BIORXIV_PREFIXES):
        paper = fetch_biorxiv(doi)
        if paper:
            return paper
    match = re.match(r'^10\.\d+/osf\.io/([a-z0-9_]+)$', doi)
    if match:
        paper = fetch_osf(match.group(1))
        if paper:
            paper['key'], paper['doi'] = doi_key(doi), doi
            return paper
    try:
        paper = fetch_openalex(doi)
    except SourceUnavailable:
        logger.warning('OpenAlex did not answer for %s, trying Crossref', doi, exc_info=True)
        paper = None
    # OpenAlex takes a few days to list a new DOI
    return paper or fetch_crossref(doi)


def fetch(key):
    if key.startswith('doi_'):
        doi = key_doi(key)
        return fetch_doi(doi) if doi else None
    if key.startswith('hal_'):
        return fetch_hal(key[4:])
    if key.startswith('pmc_'):
        return fetch_epmc(key[4:])
    return None


# --- stored metadata ------------------------------------------------------------

def metadata(key, refresh=False):
    """The paper of a key, from ExternalPaper when fetched less than a day ago; None if unknown.

    Raises SourceUnavailable when the source does not answer and nothing is stored.
    """
    from .models import ExternalPaper
    row = ExternalPaper.objects.filter(key=key).first()
    if row and row.metadata and not refresh and row.fetched and row.fetched > timezone.now() - REFRESH:
        return json.loads(row.metadata)
    try:
        paper = fetch(key)
    except SourceUnavailable:
        if row and row.metadata:
            logger.warning('Using the stored metadata of %s', key, exc_info=True)
            return json.loads(row.metadata)
        raise
    if paper is None:
        return None
    paper['key'] = key
    ExternalPaper.objects.update_or_create(key=key, defaults={
        'source': paper['source'], 'doi': paper['doi'][:300], 'metadata': json.dumps(paper), 'fetched': timezone.now()})
    return paper


def remember(paper):
    """Store a paper found by a search or a lookup, so its page opens without asking the source again."""
    from .models import ExternalPaper
    if paper.get('key'):
        ExternalPaper.objects.update_or_create(key=paper['key'], defaults={
            'source': paper['source'], 'doi': paper['doi'][:300], 'metadata': json.dumps(paper), 'fetched': timezone.now()})


def header(paper):
    """Text put before the paper's text for the language model (the arXiv API answer plays this role for arXiv)."""
    lines = ['Title: %s' % paper['title'], 'Authors: %s' % ', '.join(paper['authors'][:50])]
    for name, value in (('Published in', paper['journal']), ('Date', paper['published']), ('DOI', paper['doi'])):
        if value:
            lines.append('%s: %s' % (name, value))
    return '\n'.join(lines) + '\n'


def metadata_list(key):
    """The paper in the list form of utils.get_arxiv_metadata, so the summary pipeline needs no other branch.

    [exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref,
     comments, license, data]
    """
    paper = metadata(key)
    if paper is None:
        return [0] + [''] * 12 + ['']
    stamp = (paper['published'] or timezone.now().date().isoformat()) + 'T00:00:00Z'
    return [1, paper['authors'], paper['affiliations'], paper['url'] or ref_url(key), paper['title'],
            'https://doi.org/%s' % paper['doi'] if paper['doi'] else '', paper['abstract'] or '',
            paper['category'], stamp, stamp, paper['journal'][:200], '', paper['license'], header(paper)]


# --- full text -------------------------------------------------------------------

def _local(tag):
    return tag.rsplit('}', 1)[-1] if isinstance(tag, str) else ''


def _formula(element):
    """LaTeX of a JATS or TEI formula when it is given, else its text."""
    for child in element.iter():
        name = _local(child.tag)
        if name == 'tex-math' and child.text:
            return re.sub(r'\\documentclass.*?\\begin\{document\}|\\end\{document\}', '', child.text, flags=re.S).strip()
        if name == 'math' and child.get('alttext'):
            return child.get('alttext')
    return _clean(''.join(element.itertext()))


def _inline(element):
    """Text of a paragraph, formulas written in LaTeX."""
    parts = [element.text or '']
    for child in element:
        name = _local(child.tag)
        if name in ('inline-formula', 'disp-formula', 'formula'):
            parts.append(' $%s$ ' % _formula(child))
        elif name in ('fig', 'table-wrap', 'figure'):
            pass
        else:
            parts.append(_inline(child))
        parts.append(child.tail or '')
    return ''.join(parts)


BLOCKS = {'sec': 'section', 'div': 'section', 'app': 'section', 'boxed-text': 'section',
          'title': 'heading', 'head': 'heading',
          'p': 'paragraph', 'list': 'paragraph', 'disp-quote': 'paragraph',
          'fig': 'figure', 'table-wrap': 'figure', 'figure': 'figure',
          'disp-formula': 'formula', 'formula': 'formula'}
SKIP = {'ref-list', 'ack', 'listBibl', 'note', 'fn-group'}


def _walk(element, out):
    for child in element:
        name = _local(child.tag)
        kind = BLOCKS.get(name)
        if name in SKIP or child.get('type') in ('references', 'acknowledgement', 'annex'):
            continue
        if kind == 'section':
            _walk(child, out)
        elif kind == 'heading':
            out.append('## ' + _clean(_inline(child)))
        elif kind == 'paragraph':
            out.append(_clean(_inline(child)))
        elif kind == 'figure':
            caption = next((c for c in child.iter() if _local(c.tag) in ('caption', 'figDesc')), None)
            if caption is not None:
                out.append('Figure: ' + _clean(_inline(caption)))
        elif kind == 'formula':
            out.append('$$%s$$' % _formula(child))
        else:
            _walk(child, out)


def xml_to_text(xml):
    """Body of a JATS (bioRxiv, Europe PMC) or TEI (Grobid, OpenAlex) article, as headed paragraphs."""
    if isinstance(xml, str):
        xml = xml.encode('utf-8')
    root = ElementTree.fromstring(xml)
    body = next((e for e in root.iter() if _local(e.tag) == 'body'), None)
    if body is None:
        return ''
    out = []
    _walk(body, out)
    return '\n\n'.join(block for block in out if block.strip('# '))


def _download(url, params=None):
    """Bytes of a file, None when it cannot be had (refused, too big, not there)."""
    try:
        with _session.get(url, params=params, timeout=PDF_TIMEOUT, stream=True) as response:
            if response.status_code != 200:
                logger.info('Full text %s: HTTP %s', url, response.status_code)
                return None
            chunks, size = [], 0
            for chunk in response.iter_content(1 << 16):
                size += len(chunk)
                if size > MAX_PDF_BYTES:
                    logger.info('Full text %s is larger than %d bytes', url, MAX_PDF_BYTES)
                    return None
                chunks.append(chunk)
            return b''.join(chunks)
    except requests.RequestException:
        logger.info('Full text %s could not be read', url, exc_info=True)
        return None


def _pdf_text(content):
    from .utils import _pdf_bytes_to_text
    if not content or not content.lstrip()[:5].startswith(b'%PDF'):
        return ''
    try:
        return _pdf_bytes_to_text(content)
    except Exception:
        logger.warning('PDF text extraction failed', exc_info=True)
        return ''


def _xml_text(content):
    if not content:
        return ''
    try:
        return xml_to_text(content)
    except ElementTree.ParseError:
        logger.info('Full text XML could not be parsed', exc_info=True)
        return ''


def read_candidate(kind, where):
    """(text, source) from one of the places listed in paper['text']."""
    if kind == 'jats':
        return _xml_text(_download(where)), 'xml'
    if kind == 'epmc':
        return _xml_text(_download(EPMC_XML % where)), 'xml'
    if kind == 'epmc_doi':
        try:
            rows = epmc_search('DOI:"%s"' % where)
        except SourceUnavailable:
            return '', ''
        row = next((r for r in rows if r.get('pmcid') and r.get('isOpenAccess') == 'Y'), None)
        return (_xml_text(_download(EPMC_XML % row['pmcid'])), 'xml') if row else ('', '')
    if kind == 'pdf':
        return _pdf_text(_download(where)), 'pdf'
    if kind == 'arxiv':
        return arxiv_text(where)
    key = getattr(settings, 'OPENALEX_API_KEY', '')
    if kind in ('openalex_tei', 'openalex_pdf') and key and openalex_content_allowed():
        if kind == 'openalex_tei':
            return _xml_text(_download(OPENALEX_CONTENT % (where, 'grobid-xml'), {'api_key': key})), 'xml'
        return _pdf_text(_download(OPENALEX_CONTENT % (where, 'pdf'), {'api_key': key})), 'pdf'
    return '', ''


def arxiv_text(arxiv_id):
    """(text, source) of the arXiv version of a paper, when its own arXiv license lets us read it."""
    import asyncio
    from .paper_source import fetch_arxiv_html
    from .utils import get_arxiv_metadata
    try:
        row = asyncio.run(get_arxiv_metadata(arxiv_id))
    except Exception:
        logger.info('arXiv metadata of %s could not be read', arxiv_id, exc_info=True)
        return '', ''
    if not row or not row[0] or not is_open_license(row[12]):
        return '', ''
    text = fetch_arxiv_html(arxiv_id)
    if text:
        return text, 'html'
    return _pdf_text(_download('https://arxiv.org/pdf/%s' % arxiv_id)), 'pdf'


def full_text(key):
    """(text, source) of the paper: 'xml' or 'pdf', or its header and abstract with 'abstract' when no copy could be read.

    Only called for openly licensed papers (see is_open_license).
    """
    paper = metadata(key)
    if paper is None:
        return '', 'abstract'
    tei_read = False
    for kind, where in paper['text']:
        if kind == 'openalex_pdf' and tei_read:
            continue  # its TEI was made from this PDF: the PDF has no more text, and costs as much
        text, source = read_candidate(kind, where)
        if len(text) > MIN_TEXT:
            return text, source
        tei_read = tei_read or (kind == 'openalex_tei' and bool(text))
    logger.warning('No full text could be read for %s, summarizing its abstract', key)
    return 'Authors: %s. Title: %s. Abstract: %s.' % (paper['authors'], paper['title'], paper['abstract']), 'abstract'


# --- finding papers --------------------------------------------------------------

def resolve(text):
    """What a visitor typed or pasted -> ('arxiv', arxiv id) or ('key', key); None when nothing is recognized.

    A DOI of arXiv's own (10.48550/arXiv.*) goes to the arXiv page. Raises SourceUnavailable when the
    identifier needs a lookup (HAL, PMC, PMID, OSF link) and the source does not answer.
    """
    text = (text or '').strip()
    doi = clean_doi(text)
    if doi:
        match = ARXIV_DOI_RE.match(doi)
        if match:
            return 'arxiv', match.group(1)
        return 'key', doi_key(doi)
    match = HAL_ID_RE.search(text)
    if match and ('hal' in text.lower() or match.group(1).lower().startswith('hal')):
        paper = fetch_hal(match.group(1))
        if paper:
            remember(paper)
            return 'key', paper['key']
        return None
    match = PMC_RE.search(text)
    if match:
        rows = epmc_search('PMCID:%s' % match.group(1).upper())
        if rows:
            paper = epmc_paper(rows[0])
            remember(paper)
            return ('key', paper['key']) if paper['key'] else None
        return None
    match = PMID_RE.search(text)
    if match:
        rows = epmc_search('EXT_ID:%s AND SRC:MED' % match.group(1))
        if rows:
            paper = epmc_paper(rows[0])
            remember(paper)
            return ('key', paper['key']) if paper['key'] else None
        return None
    match = OSF_URL_RE.search(text)
    if match:
        paper = fetch_osf(match.group(1).lower())
        if paper and paper['key']:
            remember(paper)
            return 'key', paper['key']
    return None


SEARCH_TYPES = {'journals': 'type:article', 'preprints': 'type:preprint', 'reviews': 'type:review'}
SEARCH_FIELDS = 'id,doi,display_name,authorships,publication_date,primary_location,best_oa_location,locations,type'


SEARCH_SECONDS = 6 * 3600  # pages, reloads and the same words typed again reuse a search
# Searches cost $0.001 each on the same $1 a day as the full texts (OPENALEX_CONTENT_PER_DAY): past this many,
# the day's searches go to Crossref, which is free but does not know which papers are open.
OPENALEX_SEARCHES_PER_DAY = 450
OPENALEX_SEARCH_KEY = 'openalex-searches:%s'
TOOL_RESERVE = 150  # searches of the day the research tools leave to the paper search of the home page


def _spend(key, limit):
    """Count one call against a daily limit kept in the cache; False once it is reached."""
    from django.core.cache import cache
    key = key % timezone.now().date()
    cache.add(key, 0, 2 * 86400)
    try:
        used = cache.incr(key)
    except ValueError:
        return True
    if used == limit + 1:
        logger.warning('OpenAlex: the daily limit %s (%d) is reached', key, limit)
    return used <= limit


def tool_search_allowed():
    """Count one OpenAlex call of a research tool (team map, bibliography check); False once only the
    TOOL_RESERVE searches of the day kept for the paper search are left."""
    return _spend(OPENALEX_SEARCH_KEY, OPENALEX_SEARCHES_PER_DAY - TOOL_RESERVE)


def openalex_content_allowed():
    """Count one OpenAlex download against today's cap (OPENALEX_CONTENT_PER_DAY)."""
    return _spend(OPENALEX_CONTENT_KEY, OPENALEX_CONTENT_PER_DAY)


class SearchRows(list):
    """Rows plus the provider's continuation flag, preserved by the cache."""
    def __init__(self, rows=(), has_more=False):
        super().__init__(rows)
        self.has_more = has_more


def search_papers(query, kind='', per_page=25, abstracts=False, field='all', filters=None, page=1):
    """Open-access papers matching a query, as _paper dicts: from OpenAlex, or Crossref past the daily limit
    or when OpenAlex refuses. Cached SEARCH_SECONDS."""
    from django.core.cache import cache
    field = field if field in ('author', 'title', 'abstract') else 'all'
    options = filters or {}
    page = max(1, min(int(page), 400))
    cache_key = 'open-papers:v4:%s' % hashlib.sha256(json.dumps([query.lower(), kind, per_page, abstracts, field, options, page], sort_keys=True).encode()).hexdigest()
    papers = cache.get(cache_key)
    if papers is not None:
        return papers
    filters = ['is_oa:true', 'has_doi:true'] + ([SEARCH_TYPES[kind]] if kind in SEARCH_TYPES else [])
    params = {'search': query, 'filter': ','.join(filters), 'per_page': per_page,
              'select': OPENALEX_FIELDS if abstracts else SEARCH_FIELDS, 'page': page}
    if not query:
        params.pop('search')
    if field != 'all' and query:
        # Quoted author names match one byline, not words across several coauthors.
        scoped = ' '.join(re.sub(r'[,|:"\\]', ' ', query).split())
        filters.append(('%s.search:' % ({'author': 'raw_author_name', 'title': 'title', 'abstract': 'abstract'}[field])) +
                       json.dumps(scoped, ensure_ascii=False))
        params.pop('search')
        params['filter'] = ','.join(filters)
    if options.get('year_from'):
        filters.append('from_publication_date:%s-01-01' % options['year_from'])
    if options.get('year_to'):
        filters.append('to_publication_date:%s-12-31' % options['year_to'])
    if options.get('orcid'):
        filters.append('authorships.author.orcid:https://orcid.org/' + options['orcid'])
    if options.get('journals'):
        filters.append('primary_location.source.id:' + '|'.join(options['journals']))
    filters.extend('primary_location.source.id:!' + v for v in options.get('exclude_journals', []))
    params['filter'] = ','.join(filters)
    if options.get('sort') in ('newest', 'oldest'):
        params['sort'] = 'publication_date:' + ('desc' if options['sort'] == 'newest' else 'asc')
    papers = None
    if _spend(OPENALEX_SEARCH_KEY, OPENALEX_SEARCHES_PER_DAY):
        try:
            body = _json('%s/works' % OPENALEX, _openalex_params(params))
            works = (body or {}).get('results', [])
            count = ((body or {}).get('meta') or {}).get('count')
            more = page * per_page < count if isinstance(count, int) else len(works) >= per_page
            papers = SearchRows([openalex_paper(work) for work in works], has_more=more)
        except SourceUnavailable:
            logger.warning('OpenAlex search failed, trying Crossref', exc_info=True)
    if papers is None:
        if field == 'abstract' or kind or any(options.get(k) for k in ('journals', 'exclude_journals', 'orcid')):
            # Never silently broaden a filter the fallback cannot express reliably.
            raise SourceUnavailable('The filtered research index is unavailable')
        papers = (crossref_search(query, per_page) if field == 'all' and not options and page == 1 else
                  crossref_search(query, per_page, field=field, filters=options, page=page))
    papers = SearchRows([p for p in papers if p['key'] and p['title']],
                        has_more=getattr(papers, 'has_more', len(papers) >= per_page))
    for paper in papers:
        if paper['key'].startswith('doi_h'):
            remember(paper)  # a hashed key can only be read back from ExternalPaper (see key_doi)
    cache.set(cache_key, papers, SEARCH_SECONDS)
    return papers


def crossref_search(query, per_page, field='all', filters=None, page=1):
    query_key = {'author': 'query.author', 'title': 'query.title'}.get(field, 'query.bibliographic')
    options = filters or {}
    clauses = ['has-license:true']
    if options.get('year_from'):
        clauses.append('from-pub-date:%s-01-01' % options['year_from'])
    if options.get('year_to'):
        clauses.append('until-pub-date:%s-12-31' % options['year_to'])
    params = {query_key: query, 'rows': per_page, 'offset': (page - 1) * per_page, 'filter': ','.join(clauses),
              'select': 'DOI,title,author,published,container-title,license,abstract',
              'mailto': 'contact@summarizepaper.com'}
    if options.get('sort') in ('newest', 'oldest'):
        params.update(sort='published', order='desc' if options['sort'] == 'newest' else 'asc')
    body = _json('https://api.crossref.org/works', params)
    papers = []
    for work in ((body or {}).get('message') or {}).get('items', []):
        doi = work['DOI'].lower()
        date = (work.get('published') or {}).get('date-parts') or [[]]
        papers.append(_paper(
            key=doi_key(doi), source='crossref', doi=doi, title=(work.get('title') or [''])[0],
            abstract=_strip_tags(work.get('abstract', '')),
            authors=[_clean('%s %s' % (a.get('given', ''), a.get('family', ''))) for a in work.get('author') or []],
            published=_date(date[0]), journal=(work.get('container-title') or [''])[0],
            license=((work.get('license') or [{}])[0]).get('URL', '')))
    if field in ('author', 'title'):
        # Crossref ranks matches. Exclude records lacking the requested metadata.
        import unicodedata
        def words(value):
            value = unicodedata.normalize('NFKD', value.casefold())
            return set(re.findall(r'[^\W_]+', ''.join(c for c in value if not unicodedata.combining(c))))
        terms = words(query)
        papers = [p for p in papers if any(terms <= words(value) for value in
                  (p['authors'] if field == 'author' else [p['title']]))]
    count = ((body or {}).get('message') or {}).get('total-results')
    more = page * per_page < count if isinstance(count, int) else len(((body or {}).get('message') or {}).get('items', [])) >= per_page
    return SearchRows(papers, has_more=more)


def search(query, kind='', per_page=25, field='all', filters=None, page=1):
    """Search results as the search page shows them.

    [{'key', 'title', 'authors', 'venue', 'published', 'license', 'open', 'arxiv_id'}], open = we can read the full text.
    """
    kwargs = {'abstracts': True}
    if field != 'all':
        kwargs['field'] = field
    if filters:
        kwargs['filters'] = filters
    if page != 1:
        kwargs['page'] = page
    papers = search_papers(query, kind, per_page, **kwargs)
    return SearchRows([{
        'key': paper['key'], 'title': paper['title'], 'authors': paper['authors'][:6],
        'more_authors': len(paper['authors']) > 6, 'venue': paper['journal'] or paper['source_name'],
        'published': paper['published'], 'license': license_label(paper['license']),
        'abstract': paper.get('abstract', ''),
        'open': is_open_license(paper['license']), 'arxiv_id': paper['arxiv_id'].replace('/', '--'),
    } for paper in papers], has_more=getattr(papers, 'has_more', len(papers) >= per_page))


def atom_feed(papers, base_url='https://www.summarizepaper.com'):
    """Papers as an Atom feed shaped like arXiv's, for the page scripts that read arXiv answers
    (the paper picker of tools.js and the chat page): the id ends with /abs/<key>."""
    feed = ElementTree.Element('feed', xmlns='http://www.w3.org/2005/Atom')
    for paper in papers:
        entry = ElementTree.SubElement(feed, 'entry')
        ElementTree.SubElement(entry, 'id').text = '%s/abs/%s' % (base_url, paper['key'])
        ElementTree.SubElement(entry, 'title').text = paper['title']
        ElementTree.SubElement(entry, 'summary').text = paper['abstract']
        ElementTree.SubElement(entry, 'published').text = '%sT00:00:00Z' % (paper['published'] or '1970-01-01')
        ElementTree.SubElement(entry, 'label').text = label(paper['key'])
        for name in paper['authors'][:20] or ['']:
            ElementTree.SubElement(ElementTree.SubElement(entry, 'author'), 'name').text = name
    return b'<?xml version="1.0" encoding="UTF-8"?>' + ElementTree.tostring(feed, encoding='utf-8')
