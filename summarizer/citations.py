"""Citation alerts (Premium): who cites the papers I follow, and what they do with them.

NASA ADS gives the citing papers of an arXiv paper: it links preprints to
their journal version, so its counts are the ones astronomers and physicists
know. Semantic Scholar adds, when it has their full text, the sentences around
the citation, and is the source for papers ADS does not know. Once a day the hourly
digest run refreshes the papers somebody follows; a new citing paper is
queued for each follower, and a follower with queued citations gets one email
every day, week or month, as they chose (CitationPreference).

A new citing paper on arXiv whose citation sentences Semantic Scholar does not
have yet (most recent preprints) has them read from its arXiv HTML: the
bibliography entry of the followed paper, then the sentences that cite it
(CitationContext). The model reads the citation sentences, with the abstract of
the followed paper, to say whether the new paper questions it, supports it,
builds on it, uses it or only mentions it, and how sure it is (CitationVerdict,
judged once for every follower). A paper sharing an author with the followed
one is a self-citation, told apart and listed last. A paper that questions a
followed one with confidence is emailed at once, unless the user turned that off.

Following a paper records its current citations as known, so the first
email only has citations that are really new. A user can follow their own
papers from their ORCID iD or arXiv author identifier (CitationAuthorWatch),
and their new papers are then followed as they appear.
"""
import csv
import datetime
import hashlib
import json
import logging
import re
import time

import requests
from bs4 import BeautifulSoup
from django.conf import settings
from django.core.cache import cache
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.core.mail import EmailMultiAlternatives
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import get_language
from django.views.decorators.http import require_POST

from . import access, digest, generated, mailcheck, sources
from .models import (ArxivPaper, CitationAlertHistory, CitationAlertLog, CitationAuthorWatch, CitationCache,
                     CitationContext, CitationCorrection, CitationPreference, CitationVerdict, Collection,
                     FollowedPaper)

logger = logging.getLogger(__name__)

ADS = 'https://api.adsabs.harvard.edu/v1/search/query'
ADS_FIELDS = 'bibcode,title,author,pubdate,identifier,doi,citation_count'
ADS_PAGE = 2000
S2 = 'https://api.semanticscholar.org/graph/v1/paper/%s/citations'
S2_PAPER = 'https://api.semanticscholar.org/graph/v1/paper/%s'
FIELDS = 'title,externalIds,publicationDate,authors,citationCount,fieldsOfStudy,contexts,intents,isInfluential'
PAGE = 1000
MAX_PAGES = 10
REFRESH = datetime.timedelta(hours=20)
MAX_REFRESH_PER_RUN = 150
SECONDS_BETWEEN_CALLS = 1.1  # Semantic Scholar's public rate is about one request per second
EVERY = 7  # days between two emails, unless the user chose otherwise
EVERY_CHOICES = (1, 7, 30)
SLACK = datetime.timedelta(hours=1)  # the hourly run is not on the minute: a daily email does not drift an hour a day
MAX_FOLLOWED = 100
MAX_PER_EMAIL = 40
DAILY_KEY = 'citations-refreshed:%s'
# most telling first: the order of an email and of the filters
STANCES = ('questions', 'supports', 'builds_on', 'uses', 'mentions')
SCAN_AUTHORS = 100  # authors of a citing paper compared with the cited one's; the first ones are kept to show
SHOWN_AUTHORS = 6
# authors of the cited paper a self-citation shares: in a large collaboration, the 300th author sharing a name and an
# initial with somebody citing it is a coincidence more often than not, and being in the team is no self-citation
SELF_AUTHORS = 15
LATE = datetime.timedelta(days=180)  # a citing paper older than this when first seen was indexed late
ARXIV_HTML = 'https://arxiv.org/html/%s'
ARXIV_SECONDS = 3  # arXiv asks for one request every 3 seconds
MAX_CONTEXTS_PER_DAY = 60
MAX_CONTEXTS = 3
MAX_WATCHES_PER_DAY = 30
VERDICT_CHUNK = 30
ANALYZE_LIMIT = 60
ANALYZE_READS = 20
ANALYZE_KEY = 'citations-analyze:%s'
ANALYZE_PER_DAY = 5  # per user: each can mean a paid model call on ANALYZE_LIMIT citations
ANALYZE_COUNT_KEY = 'citations-analyses:%s:%s'
RETRY = datetime.timedelta(days=30)  # a citing paper with no arXiv HTML yet may get one with a new version
PAGE_SIZE = 50
TOP_AUTHORS = 8


def base_id(text):
    """'2101.00001v2', 'hep-th--9901001', or an arXiv link -> '2101.00001' / 'hep-th/9901001';
    a DOI or a doi_ key -> the doi_ key (Semantic Scholar and ADS know papers by DOI too)."""
    key = text.strip().lower()
    if sources.is_external(key):
        return key if key.startswith('doi_') and sources.valid_key(key) and sources.key_doi(key) else None
    doi = sources.clean_doi(text)
    if doi and not sources.ARXIV_DOI_RE.match(doi):
        return sources.doi_key(doi)
    match = digest.ARXIV_ID_RE.search(text.replace('--', '/'))
    return re.sub(r'v\d+$', '', match.group(1)) if match else None


def s2_headers():
    return {'x-api-key': settings.S2_API_KEY} if getattr(settings, 'S2_API_KEY', '') else {}


def fetch_s2(arxiv_id):
    """{paperId: citing paper} from Semantic Scholar; None when it does not know the paper."""
    citing, offset = {}, 0
    for _ in range(MAX_PAGES):
        for wait in (5, 20, None):
            response = requests.get(S2 % s2_ref(arxiv_id), params={'fields': FIELDS, 'limit': PAGE, 'offset': offset},
                                    headers=s2_headers(), timeout=60)
            if response.status_code != 429 or wait is None:
                break
            time.sleep(wait)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        for row in body.get('data') or []:
            paper = row.get('citingPaper') or {}
            if not paper.get('paperId') or not paper.get('title'):
                continue
            citing[paper['paperId']] = {
                'title': ' '.join(paper['title'].split()),
                'arxiv_id': (paper.get('externalIds') or {}).get('ArXiv') or '',
                'doi': (paper.get('externalIds') or {}).get('DOI') or '',
                'date': paper.get('publicationDate') or '',
                'authors': [a.get('name', '') for a in (paper.get('authors') or [])[:SCAN_AUTHORS]],
                'contexts': [' '.join(c.split()) for c in (row.get('contexts') or [])[:MAX_CONTEXTS]],
                'intents': row.get('intents') or [],
                'influential': bool(row.get('isInfluential')),
                'cited_by': paper.get('citationCount') or 0,
                'fields': paper.get('fieldsOfStudy') or [],
            }
        if body.get('next') is None:
            break
        offset = body['next']
        time.sleep(SECONDS_BETWEEN_CALLS)
    return citing


def s2_ref(paper_id):
    """How Semantic Scholar names a followed paper: 'arXiv:2101.00001' or 'DOI:10.1093/mnras/stad1234'."""
    if sources.is_external(paper_id):
        return 'DOI:%s' % sources.key_doi(paper_id)
    return 'arXiv:%s' % paper_id


def ads_get(params):
    response = requests.get(ADS, params=params, timeout=60,
                            headers={'Authorization': 'Bearer %s' % settings.ADS_TOKEN})
    response.raise_for_status()
    return response.json()['response']


def ads_name(name):
    """ADS writes 'Kral, Quentin': 'Quentin Kral', the order digest.name_key and the pages expect."""
    last, comma, first = name.partition(',')
    return ' '.join((first.strip() + ' ' + last.strip()).split()) if comma and first.strip() else name.strip()


def ads_target(arxiv_id):
    if sources.is_external(arxiv_id):
        return 'doi:"%s"' % sources.key_doi(arxiv_id)
    return 'identifier:"arXiv:%s"' % arxiv_id


def fetch_ads(arxiv_id):
    """{'ads:<bibcode>': citing paper} from ADS; None when it does not know the paper."""
    target = ads_target(arxiv_id)
    if not ads_get({'q': target, 'fl': 'bibcode', 'rows': 1})['numFound']:
        return None
    citing = {}
    for page in range(MAX_PAGES):
        body = ads_get({'q': 'citations(%s)' % target, 'fl': ADS_FIELDS, 'rows': ADS_PAGE,
                        'start': page * ADS_PAGE, 'sort': 'date desc'})
        for doc in body['docs']:
            if not doc.get('title'):
                continue
            arxiv = next((i[6:] for i in doc.get('identifier') or [] if i.startswith('arXiv:')), '')
            citing['ads:' + doc['bibcode']] = {
                'title': ' '.join(doc['title'][0].split()),
                'arxiv_id': arxiv,
                'doi': (doc.get('doi') or [''])[0],
                'date': (doc.get('pubdate') or '').replace('-00', ''),
                'authors': [ads_name(a) for a in (doc.get('author') or [])[:SCAN_AUTHORS]],
                'contexts': [],
                'intents': [],
                'influential': False,
                'cited_by': doc.get('citation_count') or 0,
                'fields': [],
            }
        if (page + 1) * ADS_PAGE >= body['numFound']:
            break
    return citing


def fetch_meta(paper_id):
    """(authors, abstract, DOI) of a followed paper from ADS or Semantic Scholar; None when neither knows it."""
    if getattr(settings, 'ADS_TOKEN', ''):
        docs = ads_get({'q': ads_target(paper_id), 'fl': 'author,abstract,doi', 'rows': 1})['docs']
        if docs:
            return ([ads_name(a) for a in docs[0].get('author') or []], ' '.join((docs[0].get('abstract') or '').split()),
                    (docs[0].get('doi') or [''])[0])
    time.sleep(SECONDS_BETWEEN_CALLS)  # right after this paper's citations, at Semantic Scholar's pace
    response = requests.get(S2_PAPER % s2_ref(paper_id), params={'fields': 'authors,abstract,externalIds'},
                            headers=s2_headers(), timeout=30)
    if response.status_code == 404:
        return None
    response.raise_for_status()
    body = response.json()
    return ([a.get('name', '') for a in body.get('authors') or []], ' '.join((body.get('abstract') or '').split()),
            (body.get('externalIds') or {}).get('DOI') or '')


def describe(cached):
    """Look up, once, the authors, abstract and journal DOI of the followed paper: from our tables, else ADS or
    Semantic Scholar. Left empty when the sources do not answer, to try again at the next check."""
    if cached.authors:
        return
    paper = None
    if not sources.is_external(cached.arxiv_id):
        paper = (ArxivPaper.objects.filter(arxiv_id__startswith=cached.arxiv_id.replace('/', '--') + 'v')
                 .prefetch_related('authors').order_by('-arxiv_id').first())
    if paper is not None and paper.authors.exists():
        authors, abstract = [a.name for a in paper.authors.all()], ' '.join((paper.abstract or '').split())
        doi = sources.clean_doi(paper.link_doi or '') or ''
    else:
        try:
            found = fetch_meta(cached.arxiv_id)
        except (requests.RequestException, ValueError, KeyError):
            logger.info('Authors of %s could not be looked up', cached.arxiv_id)
            return
        authors, abstract, doi = found or ([], '', '')
    if sources.is_external(cached.arxiv_id):
        doi = sources.key_doi(cached.arxiv_id) or doi
    cached.authors, cached.abstract, cached.doi = json.dumps(authors), abstract, (doi or '').lower()[:200]
    cached.save(update_fields=['authors', 'abstract', 'doi'])


class Citing(dict):
    """{id: citing paper}, and the source it came from ('ads' or 's2'), known even when it is empty; with ADS, the
    citing papers only Semantic Scholar has (None when it did not answer)."""

    def __init__(self, papers, source, unmatched=None):
        super().__init__(papers)
        self.source = source
        self.unmatched = unmatched


def same_paper(c):
    return [k for k in (c['arxiv_id'] and 'arxiv:' + re.sub(r'v\d+$', '', c['arxiv_id']).lower(),
                        c['doi'] and 'doi:' + c['doi'].lower()) if k]


def identity(pid, c):
    """What names a citing paper whatever the source: its arXiv id, else its DOI, else the source's id."""
    return (same_paper(c) or [pid])[0][:120]


def fetch(arxiv_id):
    """{id: citing paper}, from ADS when it knows the paper, with the citation sentences of Semantic
    Scholar; from Semantic Scholar alone otherwise. None when neither knows it."""
    citing = fetch_ads(arxiv_id) if getattr(settings, 'ADS_TOKEN', '') else None
    if citing is None:
        s2 = fetch_s2(arxiv_id)
        return None if s2 is None else Citing(s2, 's2')
    try:
        s2 = fetch_s2(arxiv_id)
    except (requests.RequestException, ValueError):
        s2 = None  # the counts do not depend on it; tomorrow's check adds the sentences
    by_key = {k: c for c in (s2 or {}).values() for k in same_paper(c)}
    for c in citing.values():
        match = next((by_key[k] for k in same_paper(c) if k in by_key), None)
        if match:
            c['contexts'], c['intents'] = match['contexts'], match['intents']
            c['influential'], c['fields'] = match.get('influential', False), match.get('fields', [])
    return Citing(citing, 'ads', None if s2 is None else unmatched(citing, s2))


def unmatched(citing, s2):
    """The citing papers of Semantic Scholar that ADS does not have, by arXiv id, DOI or title, newest first."""
    keys = {k for c in citing.values() for k in same_paper(c)} | {squash(c['title']) for c in citing.values()}
    extra = [dict(c, id=pid) for pid, c in s2.items()
             if not any(k in keys for k in same_paper(c)) and squash(c['title']) not in keys]
    extra.sort(key=lambda c: c['date'], reverse=True)
    return [{k: c[k] for k in ('id', 'title', 'arxiv_id', 'doi', 'date')} for c in extra[:200]]


def source_of(citing):
    """'ads' or 's2' from the ids of the citing papers, '' when there are none."""
    if not citing:
        return ''
    return 'ads' if any(k.startswith('ads:') for k in citing) else 's2'


def rekey(ids, old, new):
    """The ids of `new` for the papers `ids` of `old`, when the source of a paper's citations changed."""
    index = {k: pid for pid, c in new.items() for k in same_paper(c)}
    found = (next((index[k] for k in same_paper(old[pid]) if k in index), None) for pid in ids if pid in old)
    return [pid for pid in dict.fromkeys(found) if pid]


def name_keys(names):
    return [k for k in map(digest.name_key, names) if k]


def shares_author(keys, names):
    """Whether one of `names` is one of the authors whose name_keys are `keys`."""
    last = {k[0] for k in keys}
    for name in names:
        words = re.findall(r'[a-z]+', name.lower())
        if words and words[-1] not in last and not any(w in last for w in words):
            continue  # cheap test first: a large citing paper has hundreds of authors
        if any(digest.same_author(k, name) for k in keys):
            return True
    return False


def self_keys(cited_authors):
    return name_keys(cited_authors[:SELF_AUTHORS])


def mark_self(citing, before, cited_authors):
    """Tell the self-citations (an author in common with the first ones of the cited paper), then keep only the
    first authors to show. A citation already known keeps what was found for it."""
    keys = self_keys(cited_authors)
    for pid, c in citing.items():
        if 'self' in before.get(pid, {}):
            c['self'] = before[pid]['self']
        elif keys:
            c['self'] = shares_author(keys, c['authors'])
        c['authors'] = c['authors'][:SHOWN_AUTHORS]


def refresh(arxiv_id, now):
    """Update the cached citations of a paper and queue the new ones for its followers."""
    try:
        citing = fetch(arxiv_id)
    except (requests.RequestException, ValueError, KeyError):
        logger.warning('Citations failed for %s', arxiv_id)
        return False
    cached, _ = CitationCache.objects.get_or_create(arxiv_id=arxiv_id)
    cached.checked = now
    if citing is None:
        cached.save(update_fields=['checked'])
        return False
    describe(cached)
    before = json.loads(cached.citing or '{}')
    mark_self(citing, before, json.loads(cached.authors or '[]'))
    # ids are bibcodes ('ads:...') or Semantic Scholar ids: when the source changes, carry over what
    # followers knew and had queued, by arXiv id or DOI, instead of reporting every citation as new.
    # The source is stored, so a paper with no citation in the old source switches too.
    previous = cached.source or source_of(before)
    current = getattr(citing, 'source', '') or source_of(citing) or previous
    switched = bool(previous) and current != previous
    cached.citing, cached.total, cached.source = json.dumps(citing), len(citing), current
    if getattr(citing, 'unmatched', None) is not None or current != 'ads':
        cached.unmatched = json.dumps(citing.unmatched or []) if current == 'ads' else ''
    cached.save()
    for followed in FollowedPaper.objects.filter(arxiv_id=arxiv_id):
        if not followed.title:
            followed.title = titles([arxiv_id]).get(arxiv_id, '')
        if followed.known == '':
            # first check since the user followed it: what it already has is not news
            followed.known = json.dumps(list(citing))
            followed.save(update_fields=['known', 'title'])
            continue
        if switched:
            pending = rekey(json.loads(followed.pending or '[]'), before, citing)
            # a citation the new source has and the old one missed is not news either
            followed.known = json.dumps([pid for pid in citing if pid not in pending])
            followed.pending = json.dumps(pending) if pending else ''
            followed.save(update_fields=['known', 'pending', 'title'])
            continue
        known = set(json.loads(followed.known))
        pending = json.loads(followed.pending or '[]')
        new = [pid for pid in citing if pid not in known and pid not in pending]
        followed.pending = json.dumps(pending + new)
        followed.save(update_fields=['pending', 'title'])
    return True


def refresh_due(now, limit=MAX_REFRESH_PER_RUN):
    """Refresh the followed papers not checked for a day, the never checked then the oldest checked first (at most
    `limit`, spaced for the public API): past `limit` papers, the others wait a day, none waits for ever."""
    followed = set(FollowedPaper.objects.values_list('arxiv_id', flat=True))
    checked = dict(CitationCache.objects.filter(arxiv_id__in=followed, checked__isnull=False)
                   .values_list('arxiv_id', 'checked'))
    due = [a for a in followed if a not in checked or checked[a] < now - REFRESH]
    due.sort(key=lambda a: (a in checked, checked.get(a) or now, a))
    done = 0
    for arxiv_id in due[:limit]:
        if done:
            time.sleep(SECONDS_BETWEEN_CALLS)
        refresh(arxiv_id, now)
        done += 1
    return done


def titles(arxiv_ids):
    """{arxiv_id: title} from our tables, then one arXiv query for the rest; unknown ids are left out."""
    from .library import lookup_many
    found = lookup_many([a.replace('/', '--') for a in arxiv_ids])
    return {a: found[a.replace('/', '--')][0] for a in arxiv_ids if a.replace('/', '--') in found}


def follow(user, arxiv_id, title='', mine=False):
    """Start following. Nothing slow here: the first check, in the background, records the citations it
    already has as known, so they are not reported."""
    if sources.is_external(arxiv_id):
        paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    else:
        paper = ArxivPaper.objects.filter(arxiv_id__startswith=arxiv_id.replace('/', '--') + 'v').first()
    if paper and paper.title:
        title = ' '.join(paper.title.split())
    followed, created = FollowedPaper.objects.get_or_create(user=user, arxiv_id=arxiv_id,
                                                            defaults={'title': title, 'mine': mine})
    if created:
        cached = CitationCache.objects.filter(arxiv_id=arxiv_id, checked__isnull=False).first()
        if cached is not None:
            followed.known = json.dumps(list(json.loads(cached.citing or '{}')))
            followed.save(update_fields=['known'])
    elif mine and not followed.mine:
        followed.mine = True
        followed.save(update_fields=['mine'])
    return followed, created


def refresh_new(arxiv_ids):
    for index, arxiv_id in enumerate(arxiv_ids):
        if index:
            time.sleep(SECONDS_BETWEEN_CALLS)
        refresh(arxiv_id, timezone.now())


# --- the citation sentences, from the citing paper's arXiv HTML ----------------

SENTENCE_END = re.compile(r'(?<=[.!?])(?<!\bal\.)(?<!e\.g\.)(?<!i\.e\.)(?<!\bFig\.)(?<!\bEq\.)(?<!\bSec\.)\s+(?=[A-Z])')
MARK = '⁣'  # invisible: where a citation of the followed paper sits in a paragraph


def squash(text):
    return re.sub(r'[^a-z0-9]+', '', text.lower())


def cited_needles(cited_id, title, doi=''):
    """What tells the followed paper in a bibliography entry: (arXiv id pattern or None, DOI, squashed title). A
    paper followed by its arXiv id is often cited by its journal version, so its DOI counts too."""
    if sources.is_external(cited_id):
        arxiv, doi = None, (sources.key_doi(cited_id) or '').lower()
    else:
        arxiv, doi = re.compile(r'(?<![\d.])%s(?!\d)' % re.escape(cited_id.lower())), (doi or '').lower()
    title_key = squash(title)
    return arxiv, doi, title_key if len(title_key) >= 25 else ''


SUBSCRIPTS = str.maketrans('0123456789+-', '₀₁₂₃₄₅₆₇₈₉₊₋')
SYMBOLS = {'sim': '~', 'pm': '±', 'times': '×', 'newmu': 'μ', 'mu': 'μ'}


def plain_math(original):
    """LaTeX of a formula as text: 'CO₂' for chemistry and simple subscripts, '$...$' when it is real maths."""
    tex = re.sub(r'\\vphantom\{(?:[^{}]|\{[^{}]*\})*\}', '', original)
    for _ in range(3):
        tex = re.sub(r'\\(?:smash(?:\[\w\])?|text|mathrm|textrm|rm|mathit)\s*\{([^{}]*)\}', r'\1', tex)
    tex = re.sub(r'\{([^{}_^]*)\}', r'\1', tex)
    tex = re.sub(r'_\{?([0-9+-]+)\}?', lambda m: m.group(1).translate(SUBSCRIPTS), tex).strip()
    tex = re.sub(r'\\(sim|pm|times|newmu|mu)(?![A-Za-z])', lambda m: SYMBOLS[m.group(1)], tex)
    tex = re.sub(r'\\,|\\;', ' ', tex).strip()
    return tex if not re.search(r'[\\^_{}]', tex) else '$%s$' % original


def near_duplicate(sentence, found):
    return any(squash(sentence)[:80] == squash(f)[:80] for f in found)


def find_contexts(html, cited_id, title, doi=''):
    """The sentences of an arXiv HTML paper that cite the followed paper (at most MAX_CONTEXTS); a citation in a
    table gives its row, after the table's caption."""
    soup = BeautifulSoup(html, 'html.parser')
    arxiv, doi, title_key = cited_needles(cited_id, title, doi)
    wanted = set()
    for item in soup.select('li.ltx_bibitem'):
        text = (item.get_text(' ') + ' ' + ' '.join(a.get('href', '') for a in item.find_all('a'))).lower()
        if (arxiv and arxiv.search(text)) or (doi and doi in text) or (title_key and title_key in squash(text)):
            if item.get('id'):
                wanted.add(item['id'])
    if not wanted:
        return []
    for math in soup.find_all('math'):
        math.replace_with(plain_math(math['alttext']) if math.get('alttext') else math.get_text())
    paragraphs = []
    for link in soup.select('a.ltx_ref[href]'):
        if link['href'].rsplit('#', 1)[-1] not in wanted:
            continue
        cite = link.find_parent('cite') or link
        cite.insert_after(MARK)
        paragraph = cite.find_parent(['p', 'li', 'tr', 'figcaption'])
        if paragraph is not None and paragraph not in paragraphs:
            paragraphs.append(paragraph)
    found = []
    for paragraph in paragraphs:
        if paragraph.name == 'tr':
            caption = paragraph.find_parent('figure')
            caption = caption.find('figcaption') if caption is not None else None
            row = ' | '.join(' '.join(cell.get_text(' ').split()) for cell in paragraph.find_all(['td', 'th']))
            head = SENTENCE_END.split(' '.join(caption.get_text(' ').split()))[0] if caption is not None else ''
            sentences = [(head + ' — ' if head else '') + row]
        else:
            sentences = SENTENCE_END.split(' '.join(paragraph.get_text(' ').split()))
        for sentence in sentences:
            if MARK in sentence:
                sentence = re.sub(r'\s*%s' % MARK, '', sentence).strip()
                sentence = re.sub(r'\s+([,.;:)\]])', r'\1', re.sub(r'([(\[])\s+', r'\1', sentence))
                if sentence and not near_duplicate(sentence, found):
                    found.append(sentence[:600])
    return found[:MAX_CONTEXTS]


def fetch_html(arxiv_id):
    """The arXiv HTML of a paper, '' when it has none; raises RequestException when arXiv does not answer."""
    response = requests.get(ARXIV_HTML % arxiv_id, timeout=60, headers={'User-Agent': sources.USER_AGENT})
    if response.status_code == 404 or (response.status_code == 200 and '/html/' not in response.url):
        return ''  # no HTML rendering: it redirects to the abstract page
    response.raise_for_status()
    return response.text


def read_context(cited_id, title, c, now, doi=''):
    """Read and keep the citation sentences of one citing arXiv paper; None when arXiv did not answer."""
    citing_id = re.sub(r'v\d+$', '', c['arxiv_id'])
    try:
        html = fetch_html(citing_id)
    except requests.RequestException:
        logger.info('arXiv HTML of %s could not be read', citing_id)
        return None
    try:
        contexts = find_contexts(html, cited_id, title, doi) if html else []
    except Exception:
        logger.exception('Citation sentences of %s could not be found in %s', cited_id, citing_id)
        contexts = []
    CitationContext.objects.update_or_create(cited=cited_id, citing=identity('', c),
                                             defaults={'contexts': json.dumps(contexts), 'checked': now})
    return contexts


def fragmentary(contexts):
    """Whether Semantic Scholar's sentences are only cut pieces ('3 (Jiang et al., 2023); Gemma...'), as it often
    gives: the arXiv HTML gives whole sentences."""
    return bool(contexts) and all(len(x) < 60 or not re.match(r'[A-Z(\[]', x) for x in contexts)


def needs_reading(c):
    return bool(c.get('arxiv_id')) and (not c.get('contexts') or fragmentary(c['contexts']))


def tried_contexts(now, **filters):
    """(cited, citing) of the citing papers read already: found, or not found within RETRY."""
    return set(CitationContext.objects.filter(**filters).filter(~Q(contexts='[]') | Q(checked__gte=now - RETRY))
               .values_list('cited', 'citing'))


def read_contexts(now, limit=MAX_CONTEXTS_PER_DAY):
    """Read from arXiv the citation sentences of the queued citations that have none (at most `limit`)."""
    tried = tried_contexts(now)
    todo = {}
    for followed in FollowedPaper.objects.exclude(pending='').exclude(pending='[]'):
        cached = CitationCache.objects.filter(arxiv_id=followed.arxiv_id).first()
        citing = json.loads(cached.citing or '{}') if cached else {}
        for pid in json.loads(followed.pending):
            c = citing.get(pid)
            if c and needs_reading(c) and (followed.arxiv_id, identity(pid, c)) not in tried:
                todo.setdefault((followed.arxiv_id, identity(pid, c)), (followed.title, cached.doi, c))
    done = 0
    for (cited_id, _), (title, doi, c) in list(todo.items())[:limit]:
        if done:
            time.sleep(ARXIV_SECONDS)
        read_context(cited_id, title, c, now, doi)
        done += 1
    return done


def with_contexts(cited_id, citations):
    """The citations, those without whole sentences given the ones read from arXiv (marked context_source='arxiv')."""
    missing = {identity(pid, c): pid for pid, c in citations.items()
               if not c.get('contexts') or fragmentary(c['contexts'])}
    if not missing:
        return citations
    out = dict(citations)
    for key, contexts in CitationContext.objects.filter(cited=cited_id, citing__in=list(missing)).values_list('citing', 'contexts'):
        contexts = json.loads(contexts)
        if contexts:
            out[missing[key]] = dict(citations[missing[key]], contexts=contexts, context_source='arxiv')
    return out


# --- what the citing paper does with it ---------------------------------------

def verdict_of(row, lang):
    return {'stance': row.stance, 'confidence': row.confidence, 'note': row.notes.get(lang, '')}


def basis(c):
    """What a verdict was judged from: it is judged again when better sentences come (Semantic Scholar's after
    the ones read from arXiv, say)."""
    return hashlib.sha1(' | '.join(c.get('contexts') or []).encode()).hexdigest()[:16]


def still_good(row, c, lang):
    return row.source == 'reader' or (bool(row.notes.get(lang)) and row.basis == basis(c))


def stances(cited_id, title, citations, lang):
    """{pid: {'stance', 'confidence', 'note'}} of the citations with sentences, judged by the model once (and
    kept in CitationVerdict for every follower); judged again when the note is missing in `lang` or the sentences
    changed, never when an author of the cited paper corrected it."""
    with_context = {pid: c for pid, c in citations.items() if c.get('contexts')}
    if not with_context:
        return {}
    keys = {pid: identity(pid, c) for pid, c in with_context.items()}
    saved = {v.citing: v for v in CitationVerdict.objects.filter(cited=cited_id, citing__in=list(keys.values()))}
    found = {pid: verdict_of(saved[key], lang) for pid, key in keys.items()
             if key in saved and still_good(saved[key], with_context[pid], lang)}
    ask = [pid for pid in with_context if pid not in found]
    cached = CitationCache.objects.filter(arxiv_id=cited_id).first()
    abstract = cached.abstract if cached else ''
    for start in range(0, len(ask), VERDICT_CHUNK):
        chunk = ask[start:start + VERDICT_CHUNK]
        answer = judge(title, abstract, {pid: with_context[pid] for pid in chunk}, lang)
        for pid, verdict in answer.items():
            row = saved.get(keys[pid])
            if row is None:
                row = CitationVerdict(cited=cited_id, citing=keys[pid], notes={})
            if (row.stance != verdict['stance'] or row.confidence != verdict['confidence']
                    or row.basis != basis(with_context[pid])):
                row.notes = {}  # notes in other languages said something else
            row.stance, row.confidence, row.basis = verdict['stance'], verdict['confidence'], basis(with_context[pid])
            row.notes = dict(row.notes, **{lang: verdict['note']})
            row.save()
            saved[keys[pid]] = row
            found[pid] = verdict
    return found


def judge(title, abstract, citations, lang):
    """One model call: {pid: {'stance', 'confidence', 'note'}} of citations that all have sentences."""
    cited = 'Cited paper: %s' % title + ('\nIts abstract: %s' % abstract[:2500] if abstract else '')
    listing = '\n\n'.join('<citation id="%s">\nCiting paper: %s\n%sSentences citing it: %s\n</citation>'
                          % (pid, c['title'], 'Citation intent found by Semantic Scholar: %s\n' % ', '.join(c['intents'])
                             if c.get('intents') else '', ' | '.join(c['contexts'])) for pid, c in citations.items())
    try:
        answer = generated.chat_json(
            'Each citation above cites the cited paper. From the sentences citing it, say what the citing paper does '
            'with it:\n'
            '- "questions": its own results, data or analysis contradict it, fail to reproduce it, or show an '
            'error in it;\n'
            '- "supports": its own results or analysis agree with or confirm it;\n'
            '- "builds_on": it extends, generalises or improves on it;\n'
            '- "uses": it uses its method, code, data, catalogue or results as a tool or an input;\n'
            '- "mentions": a background or passing mention, possibly among other references.\n'
            'A limitation pointed out to motivate the citing paper\'s own work ("unlike X, we include Y", "X '
            'neglected Y") is "builds_on", not "questions"; so is an improvement on it. A different value found '
            'with a different method or sample, without saying it is wrong, is "questions" with low confidence. '
            'Confidence is "high" only when the sentences say it plainly, "low" when you infer it or the sentences '
            'are ambiguous; a comparison with mixed agreement is "questions" with low confidence. Write in %s one '
            'short sentence on what it says about the cited paper (what it uses, confirms or disputes), with no '
            'claim the sentences do not support. Answer with a JSON object {"items": [{"id": "...", "stance": '
            '"questions|supports|builds_on|uses|mentions", "confidence": "high|low", "note": "..."}]}.'
            % generated.language_name(lang), paid=True, max_tokens=4000, system=cited + '\n\n' + listing)
    except Exception:
        logger.exception('Citation stances failed')
        return {}
    return {i['id']: {'stance': i.get('stance') if i.get('stance') in STANCES else 'mentions',
                      'confidence': 'high' if i.get('confidence') == 'high' else 'low',
                      'note': str(i.get('note', '')).strip()}
            for i in answer.get('items', []) if isinstance(i, dict) and i.get('id') in citations}


def stance_rank(item):
    return STANCES.index(item['stance']) if item.get('stance') in STANCES else len(STANCES)


def is_late(date, now):
    """Whether a citing paper dated `date` ('2026-10-01', '2026-10' or '2026') was published well before it was
    seen: an old paper the sources only indexed now, not a new one."""
    return bool(date) and date < (now - LATE).strftime('%Y-%m-%d')[:len(date)]


def citing_url(pid, c, base_url, lang):
    if c['arxiv_id']:
        return '%s/%s/paper/%s/' % (base_url, lang, re.sub(r'v\d+$', '', c['arxiv_id']).replace('/', '--'))
    if c['doi']:
        return 'https://doi.org/%s' % c['doi']
    if pid.startswith('ads:'):
        return 'https://ui.adsabs.harvard.edu/abs/%s/abstract' % pid[4:]
    return 'https://www.semanticscholar.org/paper/%s' % pid


def judged_items(followed, citing, ids, base_url, lang, now, limit=MAX_PER_EMAIL):
    """The citations `ids` of a followed paper as shown items, most telling first: the citation sentences (read
    from arXiv when needed), the verdict of the first `limit` with sentences, self-citations last."""
    new = with_contexts(followed.arxiv_id, {pid: citing[pid] for pid in ids if pid in citing})
    order = sorted(new, key=lambda pid: new[pid]['date'], reverse=True)
    order.sort(key=lambda pid: not new[pid].get('contexts'))  # those with sentences first, the newest first
    notes = stances(followed.arxiv_id, followed.title, {pid: new[pid] for pid in order[:limit]}, lang)
    items = []
    for pid in order:
        c = new[pid]
        verdict = dict({'stance': '', 'confidence': '', 'note': ''}, **notes.get(pid, {}))
        items.append(dict(c, id=pid, key=identity(pid, c), url=citing_url(pid, c, base_url, lang),
                          self=bool(c.get('self')), late=is_late(c['date'], now),
                          no_excerpt=not c.get('contexts'), **verdict))
    corrected(followed, items)
    items.sort(key=lambda i: (i['self'], stance_rank(i), not i.get('influential')))
    return items


def corrected(followed, items):
    """The follower's own corrections of the verdicts, in place of the model's."""
    mine = dict(CitationCorrection.objects.filter(user_id=followed.user_id, cited=followed.arxiv_id)
                .values_list('citing', 'stance'))
    for i in items:
        if i['key'] in mine:
            i.update(stance=mine[i['key']], confidence='high', note='', corrected=True)
    return items


def correct(followed, key, stance):
    """A follower says what a citing paper really does with the followed paper. It is theirs alone, unless the
    paper is theirs: an author's correction becomes everybody's verdict. Back to the model's verdict when they
    choose it again."""
    row = CitationVerdict.objects.filter(cited=followed.arxiv_id, citing=key).first()
    model_stance = row.stance if row is not None and row.source != 'reader' else ''
    if stance == model_stance:
        CitationCorrection.objects.filter(user_id=followed.user_id, cited=followed.arxiv_id, citing=key).delete()
        return
    CitationCorrection.objects.update_or_create(user_id=followed.user_id, cited=followed.arxiv_id, citing=key,
                                                defaults={'stance': stance, 'model_stance': model_stance})
    if followed.mine:
        CitationVerdict.objects.update_or_create(cited=followed.arxiv_id, citing=key, defaults={
            'stance': stance, 'confidence': 'high', 'notes': {}, 'source': 'reader'})


def summary(items):
    """[(stance, count)] in the order of STANCES, then ('self', count) when there are self-citations."""
    counts = [(s, sum(1 for i in items if i.get('stance') == s)) for s in STANCES]
    found = [(s, n) for s, n in counts if n]
    selfs = sum(1 for i in items if i.get('self'))
    return found + ([('self', selfs)] if selfs else [])


# --- emails --------------------------------------------------------------------

def due_users(now):
    queued = FollowedPaper.objects.exclude(pending='').exclude(pending='[]').values_list('user_id', flat=True).distinct()
    prefs = dict(CitationPreference.objects.values_list('user_id', 'every'))
    last = dict(CitationAlertLog.objects.values_list('user_id', 'sent'))
    return [uid for uid in queued
            if uid not in last or last[uid] <= now - datetime.timedelta(days=prefs.get(uid, EVERY)) + SLACK]


def pending(user, base_url, lang, now=None):
    """(email context, commit(now)) of the new citations not emailed yet to this user, or None.

    A citing paper that cites several of the followed papers is shown once, under the first, with the others."""
    now = now or timezone.now()
    papers, archive, delivered = [], [], {}
    followed_rows = list(FollowedPaper.objects.filter(user=user).exclude(pending='').exclude(pending='[]')
                         .order_by('-mine', '-created'))
    shown = {}
    for followed in followed_rows:
        cached = CitationCache.objects.filter(arxiv_id=followed.arxiv_id).first()
        citing = json.loads(cached.citing or '{}') if cached else {}
        ids = [pid for pid in json.loads(followed.pending) if pid in citing]
        if not ids:
            continue
        delivered[followed.pk] = ids
        items = judged_items(followed, citing, ids, base_url, lang, now)
        archive.append({'arxiv_id': followed.arxiv_id, 'title': followed.title, 'mine': followed.mine, 'items': items})
        email_items = []
        for item in items[:MAX_PER_EMAIL]:
            first = shown.get(item['key'])
            if first is not None:
                first.setdefault('also', []).append({'title': followed.title or followed.arxiv_id,
                                                     'stance': item['stance'], 'confidence': item['confidence']})
                continue
            item = dict(item)
            shown[item['key']] = item
            email_items.append(item)
        papers.append({'followed': followed, 'items': email_items, 'more': max(0, len(items) - MAX_PER_EMAIL),
                       'summary': summary(items)})
    if not papers:
        return None
    context = {'papers': papers, 'count': sum(len(p['items']) for p in archive),
               'questioned': sum(1 for p in archive for i in p['items']
                                 if i['stance'] == 'questions' and i['confidence'] == 'high' and not i['self']),
               'manage_url': '%s/%s/citations/' % (base_url, lang), 'base_url': base_url, 'lang': lang}

    def commit(now, urgent=False):
        with transaction.atomic():
            CitationAlertHistory.objects.get_or_create(user=user, sent=now, defaults={
                'lang': lang, 'count': context['count'], 'papers': archive})
            for followed in followed_rows:
                current = FollowedPaper.objects.select_for_update().filter(pk=followed.pk, user=user).first()
                if current is None or followed.pk not in delivered:
                    continue
                ids = delivered[followed.pk]
                current.known = json.dumps(list(dict.fromkeys(json.loads(current.known or '[]') + ids)))
                pending = [pid for pid in json.loads(current.pending or '[]') if pid not in set(ids)]
                current.pending = json.dumps(pending) if pending else ''
                current.save(update_fields=['known', 'pending'])
            if not urgent:  # an urgent alert does not move the user's usual day
                CitationAlertLog.objects.update_or_create(user=user, defaults={'sent': now})
    return context, commit


STANCE_LABELS = {'questions': 'questions it', 'supports': 'supports it', 'builds_on': 'builds on it',
                 'uses': 'uses it', 'mentions': 'mentions it'}


def item_label(c):
    """'supports it (uncertain) · self-citation', in the active language."""
    parts = []
    if c.get('stance') in STANCE_LABELS:
        label = translation.gettext(STANCE_LABELS[c['stance']])
        parts.append(label + (' (%s)' % translation.gettext('uncertain') if c.get('confidence') == 'low' else ''))
    if c.get('self'):
        parts.append(translation.gettext('self-citation'))
    return ' · '.join(parts)


def bundle_section(user, base_url, lang, now):
    """The new citations as a section of the user's digest (see digest_bundle), in the active language."""
    if not access.has_tool(user, 'citations'):
        return None
    found = pending(user, base_url, lang, now)
    if found is None:
        return None
    context, commit = found
    return {'title': translation.gettext('New citations of the papers you follow'), 'manage_url': context['manage_url'],
            'more': sum(p['more'] for p in context['papers']), 'commit': lambda: commit(now),
            'items': [{'group': p['followed'].title or p['followed'].arxiv_id, 'title': c['title'], 'url': c['url'],
                       'label': item_label(c), 'note': c.get('note', '')} for p in context['papers'] for c in p['items']]}


def send_one(user, now, base_url, urgent=False):
    """Email this user their new citations; whether it was sent (and the citations marked as told)."""
    lang = user.digests.values_list('lang', flat=True).first() or 'en'
    found = pending(user, base_url, lang, now)
    if found is None:
        return False
    context, commit = found
    if urgent and not context['questioned']:
        return False
    context['urgent'] = urgent
    with translation.override(lang):
        subject = render_to_string('citations/subject.txt', context).strip()
        text = render_to_string('citations/email.txt', context)
        html = render_to_string('citations/email.html', context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
    message.attach_alternative(html, 'text/html')
    try:
        if not message.send():
            return False
    except Exception:
        logger.exception('Citation alert to %s failed', user.username)
        return False
    commit(now, urgent)
    return True


def can_email(user, suppressed):
    return bool(user.email) and user.email.strip().lower() not in suppressed and access.has_tool(user, 'citations')


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """Email up to `budget` users their new citations; returns how many emails were sent."""
    from django.contrib.auth.models import User
    from . import digest_bundle
    suppressed = mailcheck.suppressed()
    sent = 0
    for user in User.objects.filter(pk__in=due_users(now)).exclude(pk__in=digest_bundle.bundled_users()):
        if sent >= budget:
            break
        if can_email(user, suppressed) and send_one(user, now, base_url):
            sent += 1
    return sent


def send_urgent(now, budget, base_url=digest.BASE_URL):
    """Email at once, whatever their rhythm (and even when their digest carries their alerts), the users with a
    queued citation that questions a followed paper with confidence; returns how many emails were sent."""
    from django.contrib.auth.models import User
    queued = set(FollowedPaper.objects.exclude(pending='').exclude(pending='[]').values_list('user_id', flat=True))
    queued -= set(CitationPreference.objects.filter(urgent=False).values_list('user_id', flat=True))
    suppressed = mailcheck.suppressed()
    sent = 0
    for user in User.objects.filter(pk__in=queued):
        if sent >= budget:
            break
        if can_email(user, suppressed) and send_one(user, now, base_url, urgent=True):
            sent += 1
    return sent


def run(now, budget):
    """From the hourly digest run: once a day, follow the new papers of the watched authors, refresh the citations,
    read the missing citation sentences and send what is urgent; then send what is due within the email budget."""
    sent = 0
    if cache.add(DAILY_KEY % now.date(), True, 86400):
        watch_authors(now)
        refresh_due(now)
        read_contexts(now)
        if budget > 0:
            sent = send_urgent(now, budget)
    return sent + (send_alerts(now, budget - sent) if budget - sent > 0 else 0)


# --- my papers -----------------------------------------------------------------

def author_ident(text):
    """An ORCID iD or an arXiv author identifier as typed -> 'orcid:0000-0002-1825-0097' / 'arxiv:lee_a_1'."""
    from .digest_web import ARXIV_AUTHOR_RE, ORCID_RE
    text = (text or '').strip()
    orcid, author = ORCID_RE.search(text), ARXIV_AUTHOR_RE.match(text)
    if orcid:
        return 'orcid:' + orcid.group(1).upper()
    if author:
        return 'arxiv:' + author.group(1).lower()
    return None


def author_works(ident):
    """The ids to follow (base_id form) of the papers of an author, newest first; raises ValueError or
    RequestException when the source does not answer."""
    from .digest_web import orcid_works
    kind, value = ident.split(':', 1)
    if kind == 'orcid':
        arxiv_ids, dois = orcid_works(value)
        found = arxiv_ids + dois
    else:
        response = requests.get('https://arxiv.org/a/%s.atom' % value, timeout=20, headers={'User-Agent': sources.USER_AGENT})
        if response.status_code != 200:
            raise ValueError('arXiv author %s: HTTP %s' % (value, response.status_code))
        found = [p['arxiv_id'] for p in digest.parse_feed(response.text)]
    return list(dict.fromkeys(b for b in map(base_id, found) if b))


def follow_mine(user, ids):
    """Follow these papers as the user's own; (newly followed ids, how many did not fit)."""
    room = MAX_FOLLOWED - FollowedPaper.objects.filter(user=user).count()
    new, missed = [], 0
    for arxiv_id in ids:
        if FollowedPaper.objects.filter(user=user, arxiv_id=arxiv_id).exists():
            follow(user, arxiv_id, mine=True)
            continue
        if room <= 0:
            missed += 1
            continue
        follow(user, arxiv_id, mine=True)
        new.append(arxiv_id)
        room -= 1
    return new, missed


def watch_authors(now, limit=MAX_WATCHES_PER_DAY):
    """Follow the new papers of the watched authors (the oldest checked first, at most `limit`). Their first check
    is refresh_due's, which records their citations as known."""
    watches = CitationAuthorWatch.objects.filter(Q(checked__isnull=True) | Q(checked__lt=now - REFRESH))
    done = 0
    for watch in watches.select_related('user').order_by('checked')[:limit]:
        if done:
            time.sleep(ARXIV_SECONDS)
        done += 1
        if not access.has_tool(watch.user, 'citations'):
            continue
        try:
            ids = author_works(watch.ident)
        except (requests.RequestException, ValueError, KeyError):
            logger.info('Papers of %s could not be read', watch.ident)
            continue
        follow_mine(watch.user, ids)
        watch.checked = now
        watch.save(update_fields=['checked'])
    return done


# --- views -------------------------------------------------------------------

def preference(user):
    return CitationPreference.objects.filter(user=user).first() or CitationPreference(user=user)


def citations_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/citations.html', {'anonymous': True, 'max_followed': MAX_FOLLOWED})
    paid = access.has_tool(request.user, 'citations')
    error, notice = None, None
    lang = get_language()
    if request.method == 'POST' and paid:
        action = request.POST.get('action')
        if action == 'unfollow':
            FollowedPaper.objects.filter(user=request.user, pk=request.POST.get('pk')).delete()
            return redirect('/%s/citations/' % lang)
        if action == 'mine':
            FollowedPaper.objects.filter(user=request.user, pk=request.POST.get('pk')).update(
                mine=request.POST.get('mine') == '1')
            return redirect('/%s/citations/' % lang)
        if action == 'prefs':
            pref = preference(request.user)
            every = request.POST.get('every', '')
            pref.every = int(every) if every.isdigit() and int(every) in EVERY_CHOICES else EVERY
            pref.urgent = request.POST.get('urgent') == '1'
            pref.save()
            return redirect('/%s/citations/?saved=prefs#citation-settings' % lang)
        if action == 'unwatch':
            CitationAuthorWatch.objects.filter(user=request.user, pk=request.POST.get('pk')).delete()
            return redirect('/%s/citations/#my-papers' % lang)
        if action == 'import':
            ident = author_ident(request.POST.get('author', ''))
            if ident is None:
                error = 'author_format'
            else:
                try:
                    ids = author_works(ident)
                except (requests.RequestException, ValueError, KeyError):
                    logger.info('Papers of %s could not be read', ident)
                    ids = None
                if ids is None:
                    error = 'author_unavailable'
                elif not ids:
                    error = 'author_empty'
                else:
                    CitationAuthorWatch.objects.update_or_create(user=request.user, ident=ident,
                                                                 defaults={'checked': timezone.now()})
                    new, missed = follow_mine(request.user, ids)
                    fresh = [a for a in new if not CitationCache.objects.filter(arxiv_id=a, checked__isnull=False).exists()]
                    if fresh:
                        generated.background(refresh_new, fresh)
                    notice = {'found': len(ids), 'new': len(new), 'missed': missed}
        else:
            ids = []
            for text in re.split(r'[\s,;]+', request.POST.get('ids', '')):
                arxiv_id = base_id(text) if text else None
                if arxiv_id and arxiv_id not in ids:
                    ids.append(arxiv_id)
            room = MAX_FOLLOWED - FollowedPaper.objects.filter(user=request.user).count()
            if len(ids) > room:
                error = 'too_many'
            ids = ids[:max(0, room)]
            mine = request.POST.get('mine') == '1'
            known = titles([a for a in ids if not FollowedPaper.objects.filter(user=request.user, arxiv_id=a).exists()])
            new = [a for a in ids if follow(request.user, a, known.get(a, ''), mine=mine)[1] and not
                   CitationCache.objects.filter(arxiv_id=a, checked__isnull=False).exists()]
            if new:
                generated.background(refresh_new, new)
            if not error:
                return redirect('/%s/citations/' % lang)
    followed = list(FollowedPaper.objects.filter(user=request.user).order_by('-mine', '-created'))
    untitled = [f for f in followed if not f.title]
    if untitled:
        # followed before their title was looked up, or unknown to Semantic Scholar
        found = titles([f.arxiv_id for f in untitled])
        for f in untitled:
            if found.get(f.arxiv_id):
                f.title = found[f.arxiv_id]
                f.save(update_fields=['title'])
    totals = dict(CitationCache.objects.filter(arxiv_id__in=[f.arxiv_id for f in followed]).values_list('arxiv_id', 'total'))
    for f in followed:
        f.total = totals.get(f.arxiv_id)
        f.queued = len(json.loads(f.pending or '[]'))
    history = (Paginator(CitationAlertHistory.objects.filter(user=request.user), 10)
               .get_page(request.GET.get('alerts_page'))) if paid else None
    last_sent = (CitationAlertLog.objects.filter(user=request.user).values_list('sent', flat=True).first()
                 if paid and not history.paginator.count else None)
    from . import digest_bundle
    return render(request, 'summarizer/citations.html', {
        'paid': paid, 'followed': followed, 'error': error, 'notice': notice, 'max_followed': MAX_FOLLOWED,
        'alert_history': history, 'last_unarchived_alert': last_sent,
        'pref': preference(request.user) if paid else None, 'every_choices': EVERY_CHOICES,
        'bundled': paid and request.user.pk in digest_bundle.bundled_users(),
        'watches': list(CitationAuthorWatch.objects.filter(user=request.user)) if paid else [],
        'saved': request.GET.get('saved')})


def all_items(followed, lang, base_url=''):
    """Every known citation of a followed paper as shown items, newest first, with the sentences read from arXiv
    and the verdicts already judged (in any language when there is none in `lang`)."""
    cached = CitationCache.objects.filter(arxiv_id=followed.arxiv_id).first()
    citing = with_contexts(followed.arxiv_id, json.loads(cached.citing or '{}') if cached else {})
    verdicts = {v.citing: v for v in CitationVerdict.objects.filter(cited=followed.arxiv_id)}
    queued = set(json.loads(followed.pending or '[]'))
    items = []
    for pid, c in citing.items():
        key = identity(pid, c)
        v = verdicts.get(key)
        items.append(dict(c, id=pid, key=key, url=citing_url(pid, c, base_url, lang), self=bool(c.get('self')),
                          stance=v.stance if v else '', confidence=v.confidence if v else '',
                          note=(v.notes.get(lang) or '') if v else '', year=(c['date'] or '')[:4],
                          no_excerpt=not c.get('contexts'), queued=pid in queued,
                          by_author=bool(v) and v.source == 'reader'))
    items.sort(key=lambda i: i['date'], reverse=True)
    return cached, corrected(followed, items)


def filtered(items, params):
    stance, selfs, year = params.get('stance', ''), params.get('self', ''), params.get('year', '')
    if stance == 'none':
        items = [i for i in items if not i['stance']]
    elif stance in STANCES:
        items = [i for i in items if i['stance'] == stance]
    if selfs == 'hide':
        items = [i for i in items if not i['self']]
    elif selfs == 'only':
        items = [i for i in items if i['self']]
    if year.isdigit():
        items = [i for i in items if i['year'] == year]
    return items


def weight(item):
    """How much a citing paper weighs: influential for Semantic Scholar first, then how cited it is itself."""
    return (bool(item.get('influential')), item.get('cited_by') or 0)


def who_cites(items):
    """[(name, count)] of the authors who cite the paper most (in at least two papers), self-citations apart; and
    [(field, count)] of the citing papers' fields, as Semantic Scholar knows them."""
    names, counts = {}, {}
    for i in items:
        if i['self']:
            continue
        for name in dict.fromkeys(i['authors']):
            key = digest.name_key(name)
            if key:
                names.setdefault(key, name)
                counts[key] = counts.get(key, 0) + 1
    top = sorted((k for k in counts if counts[k] > 1), key=lambda k: (-counts[k], names[k]))[:TOP_AUTHORS]
    fields = {}
    for i in items:
        for field in i.get('fields') or []:
            fields[field] = fields.get(field, 0) + 1
    return [(names[k], counts[k]) for k in top], sorted(fields.items(), key=lambda f: -f[1])[:6]


def by_year(items):
    """[{'year', 'count', 'height'}] of the citations per year, oldest first, height in % of the busiest year."""
    counts = {}
    for i in items:
        if i['year'].isdigit():
            counts[i['year']] = counts.get(i['year'], 0) + 1
    if not counts:
        return []
    years = range(int(min(counts)), int(max(counts)) + 1)
    top = max(counts.values())
    return [{'year': str(y), 'count': counts.get(str(y), 0), 'height': round(100 * counts.get(str(y), 0) / top)}
            for y in years]


def analyze(followed_pk, lang):
    """In the background: read from arXiv the missing sentences of the newest citations, then judge the newest
    ones not judged yet (ANALYZE_LIMIT at most)."""
    try:
        followed = FollowedPaper.objects.filter(pk=followed_pk).first()
        if followed is None:
            return
        cached = CitationCache.objects.filter(arxiv_id=followed.arxiv_id).first()
        citing = json.loads(cached.citing or '{}') if cached else {}
        newest = sorted(citing, key=lambda pid: citing[pid]['date'], reverse=True)
        now, reads = timezone.now(), 0
        tried = {citing for _, citing in tried_contexts(now, cited=followed.arxiv_id)}
        for pid in newest:
            if reads >= ANALYZE_READS:
                break
            c = citing[pid]
            if needs_reading(c) and identity(pid, c) not in tried:
                if reads:
                    time.sleep(ARXIV_SECONDS)
                read_context(followed.arxiv_id, followed.title, c, now, cached.doi)
                reads += 1
        citing = with_contexts(followed.arxiv_id, citing)
        judged = {v.citing: v for v in CitationVerdict.objects.filter(cited=followed.arxiv_id)}
        todo = [pid for pid in newest if citing[pid].get('contexts') and not (
            identity(pid, citing[pid]) in judged and still_good(judged[identity(pid, citing[pid])], citing[pid], lang))]
        stances(followed.arxiv_id, followed.title, {pid: citing[pid] for pid in todo[:ANALYZE_LIMIT]}, lang)
    finally:
        cache.delete(ANALYZE_KEY % followed_pk)


def export_citations(followed, items, fmt):
    name = re.sub(r'[^A-Za-z0-9._-]+', '_', followed.arxiv_id) + '-citations'
    if fmt == 'csv':
        response = HttpResponse(content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = 'attachment; filename="%s.csv"' % name
        response.write('﻿')  # Excel reads it as UTF-8
        writer = csv.writer(response)
        writer.writerow(['title', 'authors', 'date', 'arxiv', 'doi', 'url', 'stance', 'confidence', 'self_citation',
                         'influential', 'cited_by', 'note', 'excerpts', 'excerpt_source'])
        for i in items:
            writer.writerow([i['title'], '; '.join(i['authors']), i['date'], i['arxiv_id'], i['doi'], i['url'],
                             i['stance'], i['confidence'], 'yes' if i['self'] else 'no',
                             'yes' if i.get('influential') else 'no', i.get('cited_by') or 0, i['note'],
                             ' | '.join(i.get('contexts') or []),
                             i.get('context_source') or ('semantic_scholar' if i.get('contexts') else '')])
        return response
    if fmt == 'bib':
        from .exports import bibtex_entry
        entries = []
        for i in items:
            arxiv = re.sub(r'v\d+$', '', i['arxiv_id'])
            key = digest.site_id(arxiv) if arxiv else (sources.doi_key(i['doi']) if i['doi'] else None)
            if not key:
                continue
            entries.append(bibtex_entry({'arxiv_id': key, 'title': i['title'], 'authors': i['authors'],
                                         'year': i['year'], 'category': '', 'doi': i['doi'], 'journal': ''}))
        response = HttpResponse('\n'.join(entries), content_type='application/x-bibtex; charset=utf-8')
        response['Content-Disposition'] = 'attachment; filename="%s.bib"' % name
        return response
    raise Http404


def add_to_collection(request, items):
    """The chosen citing papers added to a collection of the user (or a new one); a redirect to it, or an error."""
    from . import library
    chosen = set(request.POST.getlist('ids'))
    ids = []
    for i in items:
        if i['id'] in chosen:
            arxiv = re.sub(r'v\d+$', '', i['arxiv_id'])
            key = digest.site_id(arxiv) if arxiv else (sources.doi_key(i['doi']) if i['doi'] else None)
            if key:
                ids.append(key)
    if not ids:
        return None, 'none_chosen'
    paid = access.has_tool(request.user, 'collections')
    wanted = request.POST.get('collection', '')
    if wanted == 'new':
        if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
            return None, 'too_many_collections'
        name = request.POST.get('name', '').strip()[:120] or translation.gettext('Citing papers')
        collection = Collection.objects.create(user=request.user, name=name)
    elif wanted.isdigit():
        collection = library.editable(request, wanted)
    else:
        return None, 'none_chosen'
    library.add_many(collection, ids, access.has_tool(collection.user, 'collections'))
    return redirect('/%s/collections/%s/' % (get_language(), collection.pk)), None


def analyses_left(user):
    return ANALYZE_PER_DAY - (cache.get(ANALYZE_COUNT_KEY % (user.pk, timezone.now().date())) or 0)


def count_analysis(user):
    key = ANALYZE_COUNT_KEY % (user.pk, timezone.now().date())
    if not cache.add(key, 1, 86400):
        try:
            cache.incr(key)
        except ValueError:
            cache.set(key, 1, 86400)


def citation_detail(request, pk):
    """Every citation of one followed paper: per year, by verdict, filtered, exported or added to a collection."""
    if not request.user.is_authenticated or not access.has_tool(request.user, 'citations'):
        return redirect('/%s/citations/' % get_language())
    followed = get_object_or_404(FollowedPaper, pk=pk, user=request.user)
    lang = get_language()
    cached, items = all_items(followed, lang)
    error = None
    if request.method == 'POST':
        if request.POST.get('action') == 'analyze':
            if analyses_left(request.user) <= 0:
                return redirect('/%s/citations/%s/?limit=analyze' % (lang, followed.pk))
            if cache.add(ANALYZE_KEY % followed.pk, True, 1800):
                count_analysis(request.user)
                generated.background(analyze, followed.pk, lang)
            return redirect('/%s/citations/%s/' % (lang, followed.pk))
        if request.POST.get('correct'):
            key, keys = request.POST['correct'], {i['key'] for i in items}
            stance = request.POST.get('stance_' + key, '')
            if key in keys and stance in STANCES:
                correct(followed, key, stance)
            return redirect(request.get_full_path())
        if request.POST.get('action') == 'collect':
            response, error = add_to_collection(request, items)
            if response is not None:
                return response
    shown = filtered(items, request.GET)
    if request.GET.get('sort') == 'weight':
        shown = sorted(shown, key=weight, reverse=True)
    if request.GET.get('export') in ('csv', 'bib'):
        return export_citations(followed, shown, request.GET['export'])
    judged = [i for i in items if i['stance']]
    query = request.GET.copy()
    query.pop('page', None)
    query.pop('limit', None)
    year_query = query.copy()
    year_query.pop('year', None)
    authors, fields = who_cites(items)
    return render(request, 'summarizer/citation_detail.html', {
        'year_query': year_query.urlencode(), 'top_authors': authors, 'fields': fields,
        'unmatched': json.loads(cached.unmatched or '[]') if cached else [],
        'limit': request.GET.get('limit'), 'analyses_per_day': ANALYZE_PER_DAY,
        'followed': followed, 'cached': cached, 'total': len(items), 'shown_count': len(shown),
        'page': Paginator(shown, PAGE_SIZE).get_page(request.GET.get('page')),
        'years': by_year(items), 'summary': [s for s in summary(judged) if s[0] != 'self'], 'judged': len(judged),
        'with_excerpt': sum(1 for i in items if not i['no_excerpt']),
        'unjudged': sum(1 for i in items if not i['no_excerpt'] and not i['stance']),
        'self_count': sum(1 for i in items if i['self']),
        'analyzing': bool(cache.get(ANALYZE_KEY % followed.pk)), 'stances': [{'stance': s} for s in STANCES],
        'filters': {k: request.GET.get(k, '') for k in ('stance', 'self', 'year', 'sort')}, 'query': query.urlencode(),
        'collections': Collection.objects.filter(user=request.user).order_by('name'), 'error': error})


@require_POST
def follow_view(request):
    """From a paper page. JSON."""
    if not access.has_tool(request.user, 'citations'):
        return JsonResponse({'status': 'premium'}, status=403)
    arxiv_id = base_id(request.POST.get('arxiv_id', ''))
    if not arxiv_id:
        return JsonResponse({'status': 'unknown'}, status=400)
    if request.POST.get('action') == 'unfollow':
        FollowedPaper.objects.filter(user=request.user, arxiv_id=arxiv_id).delete()
        return JsonResponse({'status': 'unfollowed'})
    if FollowedPaper.objects.filter(user=request.user).count() >= MAX_FOLLOWED:
        return JsonResponse({'status': 'too_many'})
    followed, created = follow(request.user, arxiv_id)
    if created and not followed.title:
        followed.title = titles([arxiv_id]).get(arxiv_id, '')
        followed.save(update_fields=['title'])
    if created and followed.known == '':
        generated.background(refresh_new, [arxiv_id])
    return JsonResponse({'status': 'following'})


def is_following(user, arxiv_id):
    if not user.is_authenticated:
        return False
    return FollowedPaper.objects.filter(user=user, arxiv_id=base_id(arxiv_id) or '').exists()
