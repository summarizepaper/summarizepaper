"""Literature review from a research question (Premium+): an exploratory synthesis of 15 to 30 papers.

Not a systematic review: the searches are wide but not exhaustive, and the papers are read from their
abstracts and, when their license lets us, from the conclusions of their full text. What the page promises
instead is a search that can be checked: every query, source and count is kept and shown, and each statement
of the synthesis comes with the sentences of the papers that support it.

1. The model rewrites the question in English, lists its key concepts with their synonyms, and writes search
   queries that cover them (mechanisms, methods, settings) and a sentence that says which papers answer it.
   The researcher can rewrite the queries and search again.
2. OpenAlex (journals, preprints and conference papers, open or not) answers each query twice, by relevance
   and for the last RECENT_YEARS years; arXiv answers it too. One more OpenAlex search finds the reviews, one
   the most cited works. Retracted works are left out.
3. Citation snowballing, both ways: the works that the closest candidates cite most often (the founding
   papers a keyword search misses), then the recent works that cite those founding papers (the ones written
   with the field's newer words). The founding works OpenAlex has no abstract for get it from Semantic
   Scholar or arXiv; those still without one cannot be screened, and are listed apart.
4. The candidates are ranked by closeness to the question (embeddings); the closest are screened by the
   model on their abstract against the inclusion sentence, with their study design and a reason.
5. The included papers are chosen so the founding and the recent works are both there; deposits nobody
   screens (Zenodo, figshare) count less and never take a reserved place.
6. The conclusions and limitations of the openly licensed full texts are read.
7. The model groups the papers by approach and writes a comparative synthesis citing them. Every finding,
   agreement, side of a disagreement and gap keeps only the quotes found word for word in the paper they
   quote; a statement without one says so.

The candidates screened are kept in a second row (kind 'literature_pool'), so that the researcher can take
papers out or put others in and have the synthesis written again without searching again. The same
synthesis is also written from the papers of a collection.
"""
import datetime
import json
import logging
import math
import re
import uuid
from concurrent.futures import ThreadPoolExecutor, wait

import numpy as np
import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.http import Http404
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.html import escape
from django.utils.safestring import mark_safe
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import (access, assistant, citations, digest, evidence, exports, generated, library, mailcheck, result_share, similar,
               sources)
from .models import ArxivPaper, Collection, Generated, PaperText

logger = logging.getLogger(__name__)

RESULT_VERSION = 2
MIN_CHARS = 20
MAX_CHARS = 600
QUERIES = 6
QUERY_CHARS = 120
PER_QUERY = 25
RECENT_PER_QUERY = 15
ARXIV_PER_QUERY = 15
REVIEWS = 15
MOST_CITED = 25
RECENT_YEARS = 3
FOUNDING_YEARS = 5  # a founding paper is at least this old
SNOWBALL_FROM = 40  # closest candidates whose references are counted
SNOWBALL_MIN = 3  # a work cited by this many of them is fetched
SNOWBALL_MAX = 50
CITED_FROM = 8  # founding candidates whose recent citing works are fetched
CITING = 25
RECOVER_MAX = 20  # often cited works without an OpenAlex abstract, looked for elsewhere
UNREAD = 8  # often cited works still without an abstract, listed apart
SCREENED = 80
SCREEN_BATCH = 40
MIN_PAPERS = 15
MAX_PAPERS = 30
MAX_REVISED = 40  # papers a revision or a collection review may hold
RESERVED = 5  # places kept for founding and for recent papers
ALTERNATES = 30  # screened papers left out, shown so the researcher can put them in
ABSTRACT_CHARS = 2500
EXCERPT_CHARS = 3500
TEXT_THREADS = 4
TEXT_SECONDS = 60  # the full texts not read by then are left out: their host is often down
PER_DAY = 3
REVISIONS_PER_DAY = 10
QUOTA_KEY = 'literature-quota:%s:%s'
REVISION_KEY = 'literature-revisions:%s:%s'
POOL_KIND = 'literature_pool'
S2_BATCH = 'https://api.semanticscholar.org/graph/v1/paper/batch'
FIELDS = sources.OPENALEX_FIELDS + ',cited_by_count,referenced_works,is_retracted'
DEPOSIT_DOIS = ('10.5281/', '10.6084/', '10.13140/')  # Zenodo, figshare, ResearchGate: deposits nobody screens
LABELS = ('founding', 'recent', 'review')
DESIGNS = ('experiment', 'observation', 'simulation', 'theory', 'review', 'method', 'other')
EXPORTS = ('bib', 'ris', 'md', 'tex')
CLOSING_RE = re.compile(r'^(?:#+\s*)?(?:[\dIVX]+\.?[\d.]*\s+)?(?:[A-Za-z]+\s+){0,3}?(?:discussion|conclusions?|limitations?|'
                        r'outlook|future work|summary|concluding remarks|implications)\b[^\n]{0,60}$', re.I | re.M)
END_RE = re.compile(r'^(?:#+\s*)?(?:[\dIVX]+\.?\s+)?(?:references|bibliography|acknowledge?ments?|appendix)\b', re.I | re.M)
STEPS = [
    ('plan', _('Turning your question into searches')),
    ('search', _('Searching OpenAlex and arXiv')),
    ('snowball', _('Following the references of the closest papers, and the recent papers that cite them')),
    ('screen', _('Ranking the candidates and screening their abstracts')),
    ('read', _('Reading the conclusions of the open full texts')),
    ('write', _('An AI groups the papers by approach and writes the synthesis')),
    ('check', _('Checking each statement against the papers')),
]


def works(params, limited):
    """OpenAlex works for these parameters. Once today's budget is used up or when OpenAlex fails,
    [] and limited (a list) is told why."""
    if not sources.tool_search_allowed():
        limited.append('budget')
        return []
    if 'search' in params:
        # '?' and '*' are wildcards that OpenAlex refuses in a stemmed search; quotes and colons change its meaning
        params = dict(params, search=' '.join(re.sub(r'[?*"\\:|()]', ' ', params['search']).split()))
    try:
        body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params(dict(params, select=FIELDS)))
    except sources.SourceUnavailable:
        logger.warning('OpenAlex search failed for %s', params)
        limited.append('openalex')
        return []
    return (body or {}).get('results', [])


def status_of(doi, work_type, source_type, venue):
    """'published' (a journal, conference or book), 'preprint', or 'deposit' (Zenodo, figshare: nobody screens it)."""
    if doi.startswith(DEPOSIT_DOIS):
        return 'deposit'
    if work_type == 'preprint' or source_type == 'repository' or not venue or venue == 'arXiv':
        return 'preprint'
    return 'published'


def from_openalex(work):
    paper = sources.openalex_paper(work)
    arxiv_doi = sources.ARXIV_DOI_RE.match(paper['doi'])
    arxiv = (paper['arxiv_id'] or (arxiv_doi.group(1) if arxiv_doi else '')).replace('/', '--')
    # an arXiv preprint is known by its arXiv id, as everywhere on the site
    key = arxiv if arxiv and (not paper['doi'] or arxiv_doi) else paper['key']
    if not key or not paper['title']:
        return None
    if key.startswith('doi_h'):
        sources.remember(paper)  # a hashed key can only be read back from ExternalPaper
    source = (work.get('primary_location') or {}).get('source') or {}
    return {'key': key, 'title': paper['title'], 'abstract': paper['abstract'], 'authors': paper['authors'],
            'year': int(paper['published'][:4]) if paper['published'] else None, 'venue': paper['journal'],
            'doi': paper['doi'], 'cited_by': work.get('cited_by_count') or 0, 'type': work.get('type') or '',
            'status': status_of(paper['doi'], work.get('type'), source.get('type'), paper['journal']),
            'license': sources.normalize_license(paper['license']), 'retracted': bool(work.get('is_retracted')),
            'openalex': (work.get('id') or '').rsplit('/', 1)[-1],
            'references': [r.rsplit('/', 1)[-1] for r in work.get('referenced_works') or []]}


def from_arxiv(paper):
    return {'key': paper['arxiv_id'], 'title': ' '.join(paper['title'].split()), 'abstract': paper['abstract'],
            'authors': paper['authors'], 'year': paper['published'].year if paper.get('published') else None,
            'venue': 'arXiv', 'doi': '', 'cited_by': None, 'type': 'preprint', 'status': 'preprint', 'license': '',
            'retracted': False, 'openalex': '', 'references': []}


class Pool:
    """The candidates, one per paper (matched by key, OpenAlex id or title), with the searches that found them."""

    def __init__(self):
        self.papers, self.index = [], {}

    def add(self, paper, found_by):
        if paper is None:
            return
        names = [paper['key'], 'title:' + title_key(paper['title'])] + (['oa:' + paper['openalex']] if paper['openalex'] else [])
        known = next((self.index[n] for n in names if n in self.index), None)
        if known is None:
            known = dict(paper, found_by=[])
            self.papers.append(known)
        elif paper['cited_by'] is None:
            known['key'] = paper['key']  # found on arXiv too: its page reads the open arXiv text
        elif known['cited_by'] is None:
            # the journal record of an arXiv preprint: keep the arXiv key, take its citations and status
            known.update(cited_by=paper['cited_by'], references=paper['references'], openalex=paper['openalex'],
                         doi=known['doi'] or paper['doi'], venue=paper['venue'] or known['venue'],
                         status=paper['status'], type=paper['type'], license=paper['license'] or known['license'])
        known['retracted'] = known['retracted'] or paper['retracted']
        if not known['abstract'] and paper['abstract']:
            known['abstract'] = paper['abstract']
        for name in names:
            self.index.setdefault(name, known)
        if found_by not in known['found_by']:
            known['found_by'].append(found_by)

    def readable(self):
        """The candidates that can be screened: with an abstract, not retracted."""
        return [p for p in self.papers if p['abstract'] and not p['retracted']]


def title_key(title):
    """A title as two records of one paper share it, whatever their spaces and signs ('H 2-rich', 'H2 rich')."""
    return re.sub(r'[^a-z0-9]+', '', title.lower())[:150]


def clean_queries(values):
    return list(dict.fromkeys(' '.join(str(q).split())[:QUERY_CHARS] for q in values if str(q).strip()))[:QUERIES]


def plan_for(question, lang):
    plan = generated.chat_json(
        'A researcher asks the research question above and wants an exploratory review of the literature on it. '
        'Answer with a JSON object {"question": "the question in English, made precise without narrowing it", '
        '"title": "a short title for the review, in %s", '
        '"core": "the 3 to 5 English words that name the topic, as paper titles would", '
        '"concepts": [{"concept": "...", "synonyms": ["the other terms papers use for it"]}], '
        '"queries": ["%d literature search queries of 2 to 6 words each, in English: together they cover every '
        'concept and its synonyms, the mechanisms or explanations proposed, the methods used to study it and the '
        'settings where it is studied"], "include": "one sentence: which papers help answer the question"}.'
        % (generated.language_name(lang), QUERIES),
        paid=False, max_tokens=1200, system='<question>\n%s\n</question>' % question)
    english = str(plan.get('question') or question).strip()
    queries = clean_queries(plan.get('queries', [])) or [english[:QUERY_CHARS]]
    return {'question': english, 'title': str(plan.get('title') or '').strip(), 'queries': queries,
            'core': str(plan.get('core') or '').strip() or queries[0],
            'include': str(plan.get('include') or '').strip(),
            'concepts': [str(c.get('concept', '')).strip() for c in plan.get('concepts', []) if isinstance(c, dict)][:8]}


def gather(plan, limited, since):
    """The candidates of every search, and how many each search brought."""
    pool = Pool()
    for query in plan['queries']:
        for work in works({'search': query, 'filter': 'has_abstract:true', 'per_page': PER_QUERY}, limited):
            pool.add(from_openalex(work), 'relevance')
        for work in works({'search': query, 'filter': 'has_abstract:true,from_publication_date:%s' % since,
                           'per_page': RECENT_PER_QUERY}, limited):
            pool.add(from_openalex(work), 'recent')
        terms = ' AND '.join('all:%s' % w for w in re.findall(r'[\w\-]+', query)[:6])
        try:
            for paper in digest.arxiv_get({'search_query': terms, 'max_results': ARXIV_PER_QUERY, 'sortBy': 'relevance'}):
                pool.add(from_arxiv(paper), 'arxiv')
        except digest.ArxivUnavailable:
            logger.warning('arXiv search failed for %s', query)
            limited.append('arxiv')
    for work in works({'search': plan['core'], 'filter': 'has_abstract:true,type:review', 'per_page': REVIEWS}, limited):
        pool.add(from_openalex(work), 'reviews')
    for work in works({'search': plan['core'], 'filter': 'has_abstract:true', 'sort': 'cited_by_count:desc',
                       'per_page': MOST_CITED}, limited):
        pool.add(from_openalex(work), 'most_cited')
    return pool


def ranked(question, papers):
    """The papers that have an abstract, closest to the question first, with their similarity (computed once
    per paper)."""
    papers = [p for p in papers if p['abstract'] and not p.get('retracted')]
    new = [p for p in papers if 'similarity' not in p]
    if new:
        texts = [question] + ['%s\n\n%s' % (p['title'], p['abstract']) for p in new]
        vectors = np.concatenate([similar.embed([t[:similar.MAX_INPUT_CHARS] for t in texts[i:i + similar.BATCH]])
                                  for i in range(0, len(texts), similar.BATCH)])
        for paper, score in zip(new, vectors[1:] @ vectors[0]):
            paper['similarity'] = float(score)
    return sorted(papers, key=lambda p: -p['similarity'])


def snowball(pool, closest, limited):
    """Fetch the works the closest candidates cite most often and that no search found."""
    counts = {}
    for paper in closest[:SNOWBALL_FROM]:
        for ref in set(paper['references']):
            counts[ref] = counts.get(ref, 0) + 1
    known = {p['openalex'] for p in pool.papers if p['openalex']}
    wanted = sorted((r for r, n in counts.items() if n >= SNOWBALL_MIN and r not in known and re.match(r'^W\d+$', r)),
                    key=lambda r: -counts[r])[:SNOWBALL_MAX]
    if wanted:
        for work in works({'filter': 'openalex:%s' % '|'.join(wanted), 'per_page': len(wanted)}, limited):
            pool.add(from_openalex(work), 'cited')
    for paper in pool.papers:
        paper['cocited'] = counts.get(paper['openalex'], 0) if paper['openalex'] else 0
    return len(wanted)


def citing(pool, now_year, since, limited):
    """The recent works that cite the founding candidates most cited by the closest ones: a field renames its
    subject, and a keyword search misses what is written with the new words."""
    founding = sorted((p for p in pool.papers if p['openalex'] and not p['retracted'] and p['cocited'] >= SNOWBALL_MIN
                       and p['year'] and p['year'] <= now_year - FOUNDING_YEARS),
                      key=lambda p: (-p['cocited'], -(p['cited_by'] or 0)))[:CITED_FROM]
    if not founding:
        return 0
    before = len(pool.papers)
    for work in works({'filter': 'cites:%s,has_abstract:true,from_publication_date:%s'
                       % ('|'.join(p['openalex'] for p in founding), since),
                       'sort': 'cited_by_count:desc', 'per_page': CITING}, limited):
        pool.add(from_openalex(work), 'citing')
    return len(pool.papers) - before


def s2_batch(ids):
    """Semantic Scholar's abstract and identifiers of these papers, None when it does not answer."""
    try:
        response = citations.s2_call(lambda headers: requests.post(
            S2_BATCH, params={'fields': 'abstract,externalIds'}, headers=headers, json={'ids': ids}, timeout=60))
        response.raise_for_status()
        rows = response.json()
        return rows if isinstance(rows, list) else None
    except (requests.RequestException, ValueError):
        logger.warning('Semantic Scholar did not answer for %d abstracts', len(ids))
        return None


def recover(pool, limited):
    """The abstracts OpenAlex lacks for the works the closest candidates cite most, from Semantic Scholar, else
    from their arXiv version: the founding papers of journals that keep their abstracts from OpenAlex would
    otherwise stay unread."""
    missing = sorted((p for p in pool.papers if not p['abstract'] and not p['retracted'] and p['doi']
                      and p.get('cocited', 0) >= SNOWBALL_MIN), key=lambda p: -p['cocited'])[:RECOVER_MAX]
    if not missing:
        return 0
    rows = s2_batch(['DOI:%s' % p['doi'] for p in missing])
    if rows is None:
        limited.append('s2')
        return 0
    on_arxiv = {}
    for paper, row in zip(missing, rows):
        row = row or {}
        if row.get('abstract'):
            paper['abstract'] = ' '.join(row['abstract'].split())
        arxiv_id = ((row.get('externalIds') or {}).get('ArXiv') or '').strip()
        if arxiv_id and not paper['abstract']:
            on_arxiv[arxiv_id] = paper
    if on_arxiv:
        try:
            for entry in digest.arxiv_get({'id_list': ','.join(on_arxiv), 'max_results': len(on_arxiv)}):
                paper = on_arxiv.get(re.sub(r'v\d+$', '', entry['arxiv_id']).replace('--', '/'))
                if paper is not None and entry['abstract']:
                    paper['abstract'] = entry['abstract']
        except digest.ArxivUnavailable:
            limited.append('arxiv')
    for paper in missing:
        if paper['abstract']:
            paper['recovered'] = True
    return sum(1 for p in missing if p['abstract'])


SCREEN = """Research question: {question}
Papers that help answer it: {include}

Rate each paper above on its abstract:
- relevance: 3 = its main subject is the question itself, 2 = it brings evidence, a method or a mechanism that
  matters for the question, 1 = same field but it does not help answer it, 0 = off topic.
- approach: in 2 to 6 words of {language}, the approach it takes to the question (e.g. "field sampling of
  sediments", "laboratory degradation experiment", "transport model").
- design: one of experiment, observation (field data, surveys, observational or clinical studies), simulation
  (numerical models), theory, review, method (a new technique, instrument or dataset), other.
- reason: one short sentence of {language}: what it brings to the question, or why it does not help.
Answer with a JSON object {{"papers": [{{"id": "p0", "relevance": 0, "approach": "...", "design": "...", "reason": "..."}}]}}."""


def screen(plan, papers, lang):
    """Each paper's relevance to the question (0 to 3), approach, design and the reason, from the model reading
    the abstracts."""
    for start in range(0, len(papers), SCREEN_BATCH):
        batch = papers[start:start + SCREEN_BATCH]
        listing = '\n\n'.join('<paper id="p%d">\nTitle: %s\nAbstract: %s\n</paper>' % (i, p['title'], p['abstract'][:900])
                              for i, p in enumerate(batch))
        answer = generated.chat_json(
            SCREEN.format(question=plan['question'], include=plan['include'] or plan['question'],
                          language=generated.language_name(lang)),
            paid=False, max_tokens=6000, system=listing)
        by_id = {str(a.get('id')): a for a in answer.get('papers', []) if isinstance(a, dict)}
        for i, paper in enumerate(batch):
            note = by_id.get('p%d' % i, {})
            try:
                paper['relevance'] = max(0, min(3, int(note.get('relevance', 0))))
            except (TypeError, ValueError):
                paper['relevance'] = 0
            paper['approach'] = str(note.get('approach', '')).strip()[:80]
            design = str(note.get('design', '')).strip().lower()
            paper['design'] = design if design in DESIGNS else 'other'
            paper['reason'] = str(note.get('reason', '')).strip()[:300]


def labels(paper, now_year):
    found = []
    if paper['year'] and paper['year'] <= now_year - FOUNDING_YEARS and (
            paper.get('cocited', 0) >= SNOWBALL_MIN or 'most_cited' in paper['found_by'] or 'cited' in paper['found_by']):
        found.append('founding')
    if paper['year'] and paper['year'] >= now_year - RECENT_YEARS + 1:
        found.append('recent')
    if paper['type'] == 'review' or paper.get('design') == 'review':
        found.append('review')
    return found


def standing(paper, now_year):
    """0 to 1: how much the field cites the paper for its age; 0.3 when it is too recent to tell or we do not know."""
    age = now_year - paper['year'] if paper['year'] else 0
    if paper['cited_by'] is None or age <= 1:
        return 0.3
    return min(1.0, math.log10(1 + paper['cited_by'] / age) / 2)


def select(screened, now_year):
    """15 to 30 papers: the relevant ones, with places kept for the founding and the recent works; the
    peripheral ones (relevance 1) only to reach MIN_PAPERS. Between papers as relevant, the closest to the
    question and the most cited for their age come first; a deposit nobody screens counts one point less and
    never takes a reserved place."""
    for paper in screened:
        paper['labels'] = labels(paper, now_year)
        paper['score'] = (paper['relevance'] + paper['similarity'] + standing(paper, now_year) / 2
                          - (1 if paper['status'] == 'deposit' else 0))
    relevant = sorted((p for p in screened if p['relevance'] >= 2), key=lambda p: -p['score'])
    chosen = []
    vetted = [p for p in relevant if p['status'] != 'deposit']
    founding = sorted((p for p in vetted if 'founding' in p['labels']), key=lambda p: -(p['cited_by'] or 0))
    recent = [p for p in vetted if 'recent' in p['labels']]
    for paper in founding[:RESERVED] + recent[:RESERVED] + relevant:
        if paper not in chosen and len(chosen) < MAX_PAPERS:
            chosen.append(paper)
    for paper in sorted((p for p in screened if p['relevance'] == 1), key=lambda p: -p['score']):
        if len(chosen) >= MIN_PAPERS:
            break
        paper['peripheral'] = True
        chosen.append(paper)
    return by_year(chosen)


def by_year(papers):
    return sorted(papers, key=lambda p: (p['year'] or 0, p['title']))


def closing_sections(text):
    """The sections of a full text where its authors discuss, conclude and say what is left open ('' when
    none is found: the end of a text without headings is often its references)."""
    end = END_RE.search(text, len(text) // 3)
    body = text[:end.start()] if end else text
    parts, total = [], 0
    for match in CLOSING_RE.finditer(body):
        if parts and match.start() < parts[-1][1]:
            continue
        part = body[match.start():match.start() + EXCERPT_CHARS - total]
        parts.append((match.start(), match.start() + len(part)))
        total += len(part)
        if total >= EXCERPT_CHARS:
            break
    return '\n\n'.join(' '.join(body[a:b].split()) for a, b in parts)


def excerpt_of(paper):
    """The closing sections of a paper's full text, when its license lets us read it ('' otherwise). What the
    database knows is read first, and the connection given back before any download: the site has 20
    connections, and a download can take a minute."""
    key, text = paper['key'], ''
    if sources.is_external(key):
        if not sources.is_open_license(paper.get('license')):
            return ''
        known = sources.metadata(key)
        generated.release_connection()
        for kind, where in (known or {}).get('text', []):
            text, _source = sources.read_candidate(kind, where)
            if len(text) > sources.MIN_TEXT:
                break
    else:
        row = ArxivPaper.objects.filter(arxiv_id=key).first()
        stored = PaperText.objects.filter(arxiv_id=key).first() if row is not None and sources.is_open_license(row.license) else None
        generated.release_connection()
        if stored is not None and len(stored.text) > 3000:
            text = stored.text
        else:
            text, _source = sources.arxiv_text(key)  # its own license, read from arXiv (the cache, no database)
    return closing_sections(text) if text and len(text) > 3000 else ''


def read_texts(papers):
    """Each paper's 'excerpt': its closing sections, read in parallel; those not read in TEXT_SECONDS are left
    out."""
    todo = [p for p in papers if 'excerpt' not in p]
    for paper in todo:
        paper['excerpt'] = ''
    if not todo:
        return
    generated.release_connection()
    pool = ThreadPoolExecutor(TEXT_THREADS, thread_name_prefix='literature-text')
    futures = {pool.submit(sources.closing(excerpt_of), p): p for p in todo}
    done, _late = wait(futures, timeout=TEXT_SECONDS)
    pool.shutdown(wait=False, cancel_futures=True)
    for future in done:
        try:
            futures[future]['excerpt'] = future.result() or ''
        except Exception:
            logger.warning('Full text of %s could not be read', futures[future]['key'], exc_info=True)


SYNTHESIS = """Research question: {question}

The papers above were found for it and screened. Write, in {language}, an exploratory synthesis of what they
say about the question. Use only what the papers state (their abstract, and the conclusions of their full text
when given); cite the papers by their number. Weigh the evidence: say so when a statement rests on a single
study, or only on preprints or deposits nobody reviewed.
Answer with a JSON object:
{{"overview": [{{"text": "one sentence", "papers": [1, 2]}}],
 "themes": [{{"name": "the approach or explanation these papers share",
             "summary": "what this group of work does and finds, 2 or 3 sentences",
             "papers": [3, 5],
             "findings": [{{"text": "one finding, one sentence", "papers": [3, 5],
                           "quotes": [{{"paper": 3, "quote": "..."}}, {{"paper": 5, "quote": "..."}}]}}],
             "strengths": "what this approach is good at, one sentence",
             "limits": "what it cannot show or leaves out, one sentence"}}],
 "consensus": [{{"text": "...", "papers": [..], "quotes": [..]}}],
 "disagreements": [{{"text": "what they disagree on, one sentence",
                    "sides": [{{"text": "one position", "papers": [..], "quotes": [..]}},
                              {{"text": "the other position", "papers": [..], "quotes": [..]}}]}}],
 "gaps": [{{"text": "what the papers leave open or do not study", "papers": [..], "quotes": [..]}}]}}
- overview: 3 to 5 sentences that answer the question as far as these papers allow.
- themes: 3 to 6 groups by approach (mechanism proposed, method, setting); every paper in at least one group.
- findings: 2 to 4 per theme.
- quotes: for each paper cited, the sentence of that paper (its abstract or its conclusions above) that supports
  the statement, copied exactly, word for word, in the paper's language. Never paraphrase a quote; leave a paper
  out of quotes rather than stretch a sentence.
- disagreements: only real disagreements between papers: two or more sides, each with its own papers and a quote
  of each. Different settings or methods are not a disagreement.
- gaps: quote the papers where they state the limit or the open question themselves; a gap no paper states is
  allowed with "quotes": [].
- consensus, disagreements, gaps: 1 to 4 each; leave a list empty rather than stretch the papers.
- papers: the numbers of the papers that support the sentence, never a number that is not above."""

STATUS_TEXT = {'published': 'published in %s', 'preprint': 'preprint, not peer reviewed',
               'deposit': 'deposit on a repository nobody screens, not peer reviewed'}


def synthesize(question, chosen, lang):
    listing = '\n\n'.join(
        '<paper n="%d">\nTitle: %s\nAuthors: %s\nYear: %s\nStatus: %s\nDesign: %s\nCited by: %s\nAbstract: %s\n%s</paper>'
        % (n, p['title'], ', '.join(p['authors'][:4]), p['year'] or '?',
           STATUS_TEXT[p['status']] % p['venue'] if p['status'] == 'published' else STATUS_TEXT[p['status']],
           p.get('design') or '?', p['cited_by'] if p['cited_by'] is not None else '?', p['abstract'][:ABSTRACT_CHARS],
           'Conclusions, from the full text:\n%s\n' % p['excerpt'] if p.get('excerpt') else '')
        for n, p in enumerate(chosen, 1))
    return generated.chat_json(SYNTHESIS.format(question=question, language=generated.language_name(lang)),
                               paid=True, max_tokens=16000, system=listing)


def cited_numbers(value, count):
    numbers = []
    for n in value if isinstance(value, list) else []:
        try:
            n = int(n)
        except (TypeError, ValueError):
            continue
        if 1 <= n <= count and n not in numbers:
            numbers.append(n)
    return numbers


def statement(item, chosen, quoted=True):
    """A statement with valid paper numbers only and, when quoted, its 'support': the quotes found word for
    word in the abstract or the full-text excerpt of the paper they quote. None without a text."""
    if not isinstance(item, dict) or not str(item.get('text', '')).strip():
        return None
    count = len(chosen)
    out = {'text': str(item['text']).strip()[:1200], 'papers': cited_numbers(item.get('papers'), count)}
    if not quoted:
        return out
    support = []
    for raw in item.get('quotes') if isinstance(item.get('quotes'), list) else []:
        if not isinstance(raw, dict) or not str(raw.get('quote') or '').strip():
            continue
        try:
            n = int(raw.get('paper'))
        except (TypeError, ValueError):
            continue
        if not 1 <= n <= count or any(s['n'] == n for s in support):
            continue
        for where in ('abstract', 'excerpt'):
            match = evidence.find(str(raw['quote']), chosen[n - 1].get(where) or '')
            if match:
                support.append({'n': n, 'quote': ' '.join(match.group().split())[:1500],
                                'where': 'abstract' if where == 'abstract' else 'text'})
                if n not in out['papers']:
                    out['papers'].append(n)
                break
    out['support'] = support
    out['unquoted'] = [n for n in out['papers'] if not any(s['n'] == n for s in support)]
    return out


def statements(value, chosen, quoted=True):
    found = [statement(item, chosen, quoted) for item in value] if isinstance(value, list) else []
    return [s for s in found if s]


def checked(answer, chosen):
    """The synthesis with valid paper numbers only, and each quote kept only when it is in the paper it quotes."""
    themes = []
    for theme in answer.get('themes', []) if isinstance(answer.get('themes'), list) else []:
        if not isinstance(theme, dict) or not str(theme.get('name', '')).strip():
            continue
        themes.append({'name': str(theme['name']).strip()[:200], 'summary': str(theme.get('summary', '')).strip()[:1500],
                       'papers': cited_numbers(theme.get('papers'), len(chosen)),
                       'findings': statements(theme.get('findings'), chosen),
                       'strengths': str(theme.get('strengths', '')).strip()[:600],
                       'limits': str(theme.get('limits', '')).strip()[:600]})
    disagreements = []
    for item in answer.get('disagreements', []) if isinstance(answer.get('disagreements'), list) else []:
        head = statement(item, chosen, quoted=False)
        sides = [s for s in statements(item.get('sides') if head else None, chosen) if s['papers']]
        if head and len(sides) >= 2:
            head['sides'] = sides
            head['papers'] = list(dict.fromkeys(n for s in sides for n in s['papers']))
            disagreements.append(head)
    return {'overview': statements(answer.get('overview'), chosen, quoted=False), 'themes': themes,
            'consensus': statements(answer.get('consensus'), chosen), 'disagreements': disagreements,
            'gaps': statements(answer.get('gaps'), chosen)}


def short_authors(authors, n=4):
    return ', '.join(authors[:n]) + (' et al.' if len(authors) > n else '')


def shown_paper(n, p, previous=None):
    return {'n': n, 'key': p['key'], 'title': p['title'], 'authors': short_authors(p['authors']),
            'all_authors': p['authors'][:20], 'year': p['year'] or '', 'venue': p['venue'], 'doi': p['doi'],
            'cited_by': p['cited_by'], 'labels': p.get('labels', []), 'approach': p.get('approach', ''),
            'design': p.get('design', ''), 'reason': p.get('reason', ''), 'status': p['status'],
            'peripheral': bool(p.get('peripheral')), 'relevance': p.get('relevance'), 'found_by': p.get('found_by', []),
            'abstract': p['abstract'][:ABSTRACT_CHARS], 'full_text': bool(p.get('excerpt')),
            'new': previous is not None and p['key'] not in previous['seen'] and title_key(p['title']) not in previous['seen']}


def alternate(p):
    return {'key': p['key'], 'title': p['title'], 'authors': short_authors(p['authors'], 3), 'year': p['year'] or '',
            'venue': p['venue'], 'status': p['status'], 'relevance': p.get('relevance'), 'reason': p.get('reason', ''),
            'labels': p.get('labels', []), 'cited_by': p['cited_by']}


def written(result, question, chosen, lang, step, previous=None):
    """result, with the synthesis of the chosen papers and their references."""
    step('read')
    read_texts(chosen)
    step('write')
    answer = synthesize(question, chosen, lang)
    step('check')
    result.update(checked(answer, chosen))
    coverage = result['coverage']
    findings = [f for t in result['themes'] for f in t['findings']]
    others = result['consensus'] + [s for d in result['disagreements'] for s in d['sides']] + result['gaps']
    coverage.update(findings=len(findings), quoted=sum(1 for f in findings if f['support']),
                    statements=len(others), statements_quoted=sum(1 for s in others if s['support']),
                    full_text=sum(1 for p in chosen if p.get('excerpt')), included=len(chosen),
                    labels={name: sum(1 for p in chosen if name in p.get('labels', [])) for name in LABELS},
                    statuses={s: sum(1 for p in chosen if p['status'] == s) for s in ('published', 'preprint', 'deposit')})
    years = [p['year'] for p in chosen if p['year']]
    coverage['years'] = [min(years), max(years)] if years else []
    result['papers'] = [shown_paper(n, p, previous) for n, p in enumerate(chosen, 1)]
    return result


def stored(paper):
    """What the pool keeps of a candidate: all but its references."""
    return {k: v for k, v in paper.items() if k != 'references'}


def review(question, lang, step=lambda name: None, queries=None, previous=None, edited=None):
    """(the result shown on the page, the pool of screened candidates); step(name) reports progress.
    queries: the researcher's own; previous: the review this one runs again ({'key', 'searched', 'seen': the keys
    and titles of its papers})."""
    now_year = timezone.now().year
    since = (timezone.now() - datetime.timedelta(days=365 * RECENT_YEARS)).date().isoformat()
    limited = []
    plan = plan_for(question, lang)
    queries = clean_queries(queries or [])
    edited = (bool(queries) and queries != plan['queries']) if edited is None else edited
    plan['queries'] = queries or plan['queries']
    step('search')
    pool = gather(plan, limited, since)
    step('snowball')
    closest = ranked(plan['question'], pool.papers)
    fetched = snowball(pool, closest, limited)
    recovered = recover(pool, limited)
    cites = citing(pool, now_year, since, limited)
    step('screen')
    candidates = ranked(plan['question'], pool.papers)  # with the snowballed works
    screened = candidates[:SCREENED]
    screen(plan, screened, lang)
    chosen = select(screened, now_year)
    relevant = sum(1 for p in screened if p['relevance'] >= 2)
    by_source = {}
    for paper in pool.papers:
        for name in paper['found_by']:
            by_source[name] = by_source.get(name, 0) + 1
    unread = sorted((p for p in pool.papers if not p['abstract'] and not p['retracted'] and p.get('cocited', 0) >= SNOWBALL_MIN),
                    key=lambda p: (-p['cocited'], -(p['cited_by'] or 0)))[:UNREAD]
    alternates = sorted((p for p in screened if p not in chosen and p['relevance'] >= 1), key=lambda p: -p['score'])[:ALTERNATES]
    coverage = {
        'mode': 'search', 'queries': plan['queries'], 'edited': edited, 'concepts': plan['concepts'],
        'include': plan['include'], 'candidates': len(pool.papers), 'with_abstract': len(candidates), 'by_source': by_source,
        'snowballed': fetched, 'citing': cites, 'recovered': recovered,
        'retracted': [p['title'] for p in pool.papers if p['retracted']][:10],
        'screened': len(screened), 'relevant': relevant, 'included': len(chosen),
        'peripheral': sum(1 for p in chosen if p.get('peripheral')), 'years': [],
        'labels': {name: 0 for name in LABELS}, 'limited': sorted(set(limited)), 'thin': relevant < MIN_PAPERS,
        'unread': [{'key': p['key'], 'title': p['title'], 'year': p['year'] or '', 'doi': p['doi'],
                    'cited_by': p['cited_by'], 'cocited': p['cocited'], 'authors': short_authors(p['authors'], 3)}
                   for p in unread],
    }
    result = {'version': RESULT_VERSION, 'question': question, 'english': plan['question'],
              'title': plan['title'] or plan['question'], 'searched': timezone.now().date().isoformat(),
              'coverage': coverage, 'papers': [], 'alternates': [alternate(p) for p in alternates],
              'overview': [], 'themes': [], 'consensus': [], 'disagreements': [], 'gaps': []}
    if previous:
        result['previous'] = {'key': previous['key'], 'searched': previous['searched']}
    if chosen:
        written(result, plan['question'], chosen, lang, step, previous)
    seen = sorted({p['key'] for p in pool.papers} | {title_key(p['title']) for p in pool.papers})
    return result, {'english': plan['question'], 'papers': [stored(p) for p in chosen + alternates], 'seen': seen}


def revise(parent, pool, include, exclude, lang, step=lambda name: None):
    """The parent review written again without the papers of exclude and with those of include, taken from its
    pool of screened candidates: no new search."""
    papers = {p['key']: p for p in pool['papers']}
    chosen = [papers[p['key']] for p in parent['papers'] if p['key'] in papers and p['key'] not in exclude]
    for key in include:
        if key in papers and papers[key] not in chosen and len(chosen) < MAX_REVISED:
            papers[key].pop('peripheral', None)
            chosen.append(papers[key])
    chosen = by_year(chosen)
    kept = {p['key'] for p in chosen}
    coverage = dict(parent['coverage'], mode='revision', revised_from=parent.get('searched', ''),
                    added=sum(1 for k in kept if k not in {p['key'] for p in parent['papers']}),
                    removed=sum(1 for p in parent['papers'] if p['key'] not in kept),
                    peripheral=sum(1 for p in chosen if p.get('peripheral')))
    result = dict(parent, coverage=coverage, papers=[], overview=[], themes=[], consensus=[], disagreements=[], gaps=[],
                  version=RESULT_VERSION, alternates=[alternate(p) for p in pool['papers'] if p['key'] not in kept][:ALTERNATES])
    if chosen:
        written(result, pool.get('english') or parent['english'], chosen, lang, step)
    return result, {'english': pool.get('english') or parent['english'], 'papers': list(papers.values())}


def collection_paper(row, now_year):
    """A paper of a collection, as the review reads it (None without an abstract)."""
    meta = exports.paper_meta(row.arxiv_id, row)
    if not meta or not meta['abstract']:
        return None
    year = int(str(meta['year'])[:4]) if str(meta['year'])[:4].isdigit() else None
    doi = meta['doi'] or ''
    status = 'deposit' if doi.startswith(DEPOSIT_DOIS) else 'preprint' if exports.is_preprint(meta) else 'published'
    paper = {'key': row.arxiv_id, 'title': meta['title'], 'abstract': meta['abstract'], 'authors': meta['authors'],
             'year': year, 'venue': meta['journal'] or ('arXiv' if not sources.is_external(row.arxiv_id) else ''),
             'doi': doi, 'cited_by': None, 'type': '', 'status': status, 'license': '', 'found_by': ['collection']}
    if sources.is_external(row.arxiv_id):
        known = sources.metadata(row.arxiv_id) if sources.valid_key(row.arxiv_id) else None
        paper['license'] = sources.normalize_license((known or {}).get('license'))
    return paper


def from_collection(collection_pk, question, lang, step=lambda name: None):
    """A review of the selected papers of a collection, from the researcher's question (or the collection's name)."""
    now_year = timezone.now().year
    collection = Collection.objects.get(pk=collection_pk)
    rows = list(collection.papers.filter(selected=True)[:MAX_REVISED])
    papers, missing = [], []
    for row in rows:
        try:
            paper = collection_paper(row, now_year)
        except sources.SourceUnavailable:
            paper = None
        if paper is None:
            missing.append(row.title or row.arxiv_id)
        else:
            papers.append(paper)
    generated.release_connection()
    question = question or collection.name
    step('screen')
    plan = {'question': question, 'include': ''}
    screen(plan, papers, lang)
    for paper in papers:
        paper['labels'] = labels(paper, now_year)
    chosen = by_year(papers)
    coverage = {'mode': 'collection', 'collection': collection.name, 'collection_pk': collection.pk,
                'without_abstract': missing[:20], 'included': len(chosen), 'labels': {name: 0 for name in LABELS},
                'years': [], 'limited': [], 'queries': [], 'by_source': {}}
    result = {'version': RESULT_VERSION, 'question': question, 'english': question, 'title': collection.name,
              'searched': timezone.now().date().isoformat(), 'coverage': coverage, 'papers': [], 'alternates': [],
              'overview': [], 'themes': [], 'consensus': [], 'disagreements': [], 'gaps': []}
    if chosen:
        written(result, question, chosen, lang, step)
    return result, {'english': question, 'papers': [stored(p) for p in chosen]}


def pool_of(row):
    """The screened candidates kept for a review (None for a review made before they were kept)."""
    pool = Generated.objects.filter(kind=POOL_KIND, key=row.key, lang=row.lang).first()
    return pool.data() if pool is not None and pool.content else None


def ready_email(pk, result):
    """Tell the researcher their review is ready: it takes minutes, and few wait on the page."""
    row = Generated.objects.select_related('user').filter(pk=pk).first()
    user = row.user if row is not None else None
    if user is None or not user.email or mailcheck.undeliverable_reason(user.email):
        return
    with translation.override(row.lang):
        context = {'user': user, 'result': result, 'link': '%s/%s/literature-review/%s/' % (digest.BASE_URL, row.lang, row.key)}
        subject = render_to_string('summarizer/literature_email_subject.txt', context).strip()
        text = render_to_string('summarizer/literature_email.txt', context)
    try:
        EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email]).send()
    except Exception:
        logger.exception('Could not send the literature review %s to user %s', row.key, user.pk)


def finish(pk, made):
    """Save the review and its pool, then email its owner: the link works as soon as the email arrives."""
    result, pool = made
    row = Generated.objects.get(pk=pk)
    Generated.objects.update_or_create(kind=POOL_KIND, key=row.key, lang=row.lang,
                                       defaults={'user': row.user, 'status': 'done', 'content': json.dumps(pool)})
    Generated.objects.filter(pk=pk).update(status='done', content=json.dumps(result), updated=timezone.now())
    ready_email(pk, result)
    return result


def write(question, lang, queries=None, previous=None, edited=None):
    def work(pk):
        generated.progress(pk, 'plan')
        return finish(pk, review(question, lang, lambda name: generated.progress(pk, name), queries, previous, edited))
    return work


def write_revision(parent_pk, include, exclude, lang):
    def work(pk):
        generated.progress(pk, 'read')
        parent = Generated.objects.get(pk=parent_pk)
        made = revise(parent.data(), pool_of(parent), include, exclude, lang, lambda name: generated.progress(pk, name))
        made[0]['revision_of'] = {'key': parent.key, 'searched': made[0].get('searched', '')}
        made[0].pop('previous', None)
        return finish(pk, made)
    return work


def write_collection(collection_pk, question, lang):
    def work(pk):
        generated.progress(pk, 'screen')
        return finish(pk, from_collection(collection_pk, question, lang, lambda name: generated.progress(pk, name)))
    return work


def _spend(user, key=QUOTA_KEY, limit=PER_DAY):
    key = key % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= limit


def _start(work, user):
    row = generated.start('literature', uuid.uuid4().hex, get_language(), work, user=user)
    return redirect('/%s/literature-review/%s/' % (get_language(), row.key))


def literature_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/literature.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'literature')
    error = None
    question = ' '.join(request.GET.get('question', '').split())[:MAX_CHARS]  # from another tool's topic
    if request.method == 'POST' and paid and request.POST.get('collection', '').isdigit():
        return literature_from_collection(request, int(request.POST['collection']))
    if request.method == 'POST' and paid:
        question = ' '.join(request.POST.get('question', '').split())
        if not MIN_CHARS <= len(question) <= MAX_CHARS:
            error = 'length'
        elif not _spend(request.user):
            error = 'quota'
        else:
            return _start(write(question, get_language()), request.user)
    reviews = Generated.objects.filter(kind='literature', user=request.user).order_by('-created')[:20]
    collections = Collection.objects.filter(user=request.user).order_by('-updated')[:20] if paid else []
    return render(request, 'summarizer/literature.html', {
        'paid': paid, 'error': error, 'reviews': reviews, 'per_day': PER_DAY, 'question': question,
        'min_chars': MIN_CHARS, 'max_chars': MAX_CHARS, 'collections': collections})


def _row(request, key):
    if not access.has_tool(request.user, 'literature'):
        raise Http404
    row = Generated.objects.filter(kind='literature', key=key, user=request.user).first()
    if row is None:
        raise Http404
    return row


def marked(text, quotes):
    """The abstract as HTML, with the quoted sentences marked."""
    html = escape(text)
    for quote in quotes:
        words = [re.escape(escape(w)) for w in quote.split()]
        if words:
            html = re.sub(r'\s+'.join(words), lambda m: '<mark>%s</mark>' % m.group(), html, count=1)
    return mark_safe(html)


def _shown(row):
    """The review as its page shows it: each statement with its papers, each paper with its quoted sentences."""
    result = row.data() if row.status == 'done' else None
    if not result:
        return result
    numbers = {p['n']: p for p in result['papers']}
    quoted = {}

    def show(s, quoted_ok=True):
        if result.get('version', 1) < 2 and 'quote' in s:  # a finding of the first version: one quote
            s['support'] = [{'n': s['quoted'], 'quote': s['quote'], 'where': 'abstract'}] if s.get('quote') else []
        s['refs'] = [numbers[n] for n in s.get('papers', []) if n in numbers]
        for q in s.get('support') or []:
            q['paper'] = numbers.get(q['n'])
            if q['where'] == 'abstract':
                quoted.setdefault(q['n'], []).append(q['quote'])
        if not quoted_ok:
            s.pop('support', None)
    for s in result['overview']:
        show(s, quoted_ok=False)
    for theme in result['themes']:
        theme['entries'] = [numbers[n] for n in theme['papers'] if n in numbers]
        theme['refs'] = theme['entries']
        for finding in theme['findings']:
            show(finding)
    for s in result['consensus'] + result['gaps']:
        show(s, quoted_ok=result.get('version', 1) >= 2)
    for d in result['disagreements']:
        show(d, quoted_ok=False)
        for side in d.get('sides', []):
            show(side)
    for paper in result['papers']:
        paper['abstract_html'] = marked(paper.get('abstract', ''), quoted.get(paper['n'], []))
        paper['cite_title'] = '%s (%s) %s' % (paper['authors'], paper.get('year') or '', paper['title'])
    result['new'] = sum(1 for p in result['papers'] if p.get('new'))
    result['day'] = datetime.date.fromisoformat(result['searched']) if result.get('searched') else row.created
    if result.get('previous'):
        result['previous']['day'] = datetime.date.fromisoformat(result['previous']['searched'])
    return result


def links(request, row, result):
    """Where the papers and the question of the review go next, in the other tools."""
    keys = [p['key'] for p in result['papers']]
    english = result.get('english') or result.get('question') or ''
    overview = ' '.join(s['text'] for s in result.get('overview', []))
    from urllib.parse import urlencode
    return {'chat': ','.join(keys[:assistant.MAX_PAPERS]), 'chat_max': assistant.MAX_PAPERS,
            'teams': urlencode({'topic': english[:MAX_CHARS]}),
            'scoops': urlencode({'title': (result.get('title') or english)[:280], 'description': (english + '\n\n' + overview)[:5000]}),
            'trends': urlencode({'t': (result.get('title') or english)[:80]}),
            'again': '\n'.join(result.get('coverage', {}).get('queries', []))}


@login_required
def literature_result(request, key):
    row = _row(request, key)
    action = request.POST.get('action', 'save')
    if request.method == 'POST' and row.status == 'done':
        if result_share.act(row, action):
            return redirect('/%s/literature-review/%s/' % (get_language(), row.key))
        if action == 'again':
            return search_again(request, row)
        if action == 'revise':
            return revision(request, row)
        return save_collection(request, row)
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    result = _shown(row)
    return render(request, 'summarizer/literature_result.html', {
        'row': row, 'result': result, 'steps': STEPS, 'message': request.GET.get('m'),
        'share_url': result_share.shared_url(request, row, 'literature-review'),
        'links': links(request, row, result) if result else None,
        'revisable': bool(result and result.get('version', 1) >= 2 and pool_of(row) is not None),
        'max_chars': MAX_CHARS, 'max_revised': MAX_REVISED, 'per_day': PER_DAY,
        'collection': Collection.objects.filter(pk=request.GET.get('c'), user=request.user).first()
        if request.GET.get('c', '').isdigit() else None})


def literature_shared(request, token):
    """The read-only page of a shared review, for anyone with its link: never who asked it."""
    row = result_share.shared_row('literature', token)
    return render(request, 'summarizer/literature_result.html', {'row': row, 'result': _shown(row), 'shared': True,
                                                                 'token': token})


def search_again(request, row):
    """The same question searched again (with the queries the researcher rewrote): the papers this review's
    searches had not found at all are marked new, not those it found and left out (screening varies a little)."""
    question = ' '.join(request.POST.get('question', '').split()) or row.data()['question']
    if not MIN_CHARS <= len(question) <= MAX_CHARS:
        return redirect('/%s/literature-review/%s/?m=length' % (get_language(), row.key))
    if not _spend(request.user):
        return redirect('/%s/literature-review/%s/?m=quota' % (get_language(), row.key))
    data = row.data()
    previous = {'key': row.key, 'searched': data.get('searched') or row.created.date().isoformat(),
                'seen': {p['key'] for p in data['papers']} | {title_key(p['title']) for p in data['papers']}}
    pool = pool_of(row)
    if pool is not None:
        previous['seen'] |= set(pool.get('seen') or []) | {p['key'] for p in pool['papers']} | {title_key(p['title']) for p in pool['papers']}
    queries = clean_queries(request.POST.get('queries', '').splitlines())  # the same ones: the reviews compare
    edited = bool(queries) and (queries != data['coverage'].get('queries') or bool(data['coverage'].get('edited')))
    return _start(write(question, get_language(), queries, previous, edited), request.user)


def revision(request, row):
    """The synthesis written again with papers taken out or put in, from the candidates already screened."""
    pool = pool_of(row)
    if pool is None:
        raise Http404
    keys = {p['key'] for p in pool['papers']}
    data = row.data()
    exclude = {k for k in request.POST.getlist('exclude') if k in keys}
    include = [k for k in request.POST.getlist('include') if k in keys]
    if not exclude and not include:
        return redirect('/%s/literature-review/%s/?m=unchanged#adjust' % (get_language(), row.key))
    if len(data['papers']) - len(exclude) + len(include) < 2:
        return redirect('/%s/literature-review/%s/?m=too_few#adjust' % (get_language(), row.key))
    if not _spend(request.user, REVISION_KEY, REVISIONS_PER_DAY):
        return redirect('/%s/literature-review/%s/?m=revisions#adjust' % (get_language(), row.key))
    return _start(write_revision(row.pk, include, exclude, get_language()), request.user)


@login_required
def literature_from_collection(request, pk):
    """A review of the papers of a collection the researcher edits, from their question."""
    if request.method != 'POST':
        return redirect('/%s/collections/%s/' % (get_language(), pk))
    collection = library.editable(request, pk)
    if not access.has_tool(request.user, 'literature'):
        return redirect('/%s/literature-review/' % get_language())
    if collection.papers.filter(selected=True).count() < 3:
        return redirect('/%s/collections/%s/?review=too_few' % (get_language(), pk))
    if not _spend(request.user):
        return redirect('/%s/collections/%s/?review=quota' % (get_language(), pk))
    question = ' '.join(request.POST.get('question', '').split())[:MAX_CHARS]
    return _start(write_collection(collection.pk, question, get_language()), request.user)


def save_collection(request, row):
    """Put the papers of the review in a new collection, to compare them in a table or chat with them."""
    paid = access.has_tool(request.user, 'collections')
    if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
        return redirect('/%s/literature-review/%s/?m=too_many' % (get_language(), row.key))
    result = row.data()
    collection = Collection.objects.create(user=request.user, name=str(result.get('title') or _('Literature review'))[:120])
    papers = result.get('papers', [])
    known = {p['key']: (p['title'], p['authors']) for p in papers}
    for paper in papers:  # one per work, whatever its version; up to the plan's limit
        library.add_paper(collection, paper['key'], paid, known)
    return redirect('/%s/literature-review/%s/?m=saved&c=%s#next' % (get_language(), row.key, collection.pk))


# --- exports -------------------------------------------------------------------------------------------------

def metas(result):
    """The references of a review as the exports read them, from what the review keeps (no lookup)."""
    out = []
    for p in result['papers']:
        authors = p.get('all_authors') or [a for a in p['authors'].replace(' et al.', '').split(', ') if a]
        out.append({'arxiv_id': p['key'], 'title': p['title'], 'authors': authors, 'year': p.get('year') or '', 'date': '',
                    'category': '', 'abstract': p.get('abstract', ''), 'doi': p.get('doi') or '', 'journal': p.get('venue') or ''})
    return out


def cite_keys(entries):
    """Each BibTeX entry with a key no other entry has (smith2020a, smith2020b...), and the keys in order."""
    keys, seen, out = [], {}, []
    for entry in entries:
        head, rest = entry.split(',', 1)
        kind, key = head.split('{', 1)
        seen[key] = seen.get(key, 0) + 1
        keys.append(key)
        out.append((kind, key, rest))
    counts = {k: n for k, n in seen.items() if n > 1}
    taken = {}
    for i, (kind, key, rest) in enumerate(out):
        if key in counts:
            taken[key] = taken.get(key, 0) + 1
            keys[i] = key + 'abcdefghijklmnopqrstuvwxyz'[(taken[key] - 1) % 26]
        out[i] = '%s{%s,%s' % (kind, keys[i], rest)
    return out, keys


LATEX_SPECIAL = {'\\': r'\textbackslash{}', '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#', '_': r'\_', '{': r'\{',
                 '}': r'\}', '~': r'\textasciitilde{}', '^': r'\textasciicircum{}'}


def latex(text):
    return ''.join(LATEX_SPECIAL.get(c, c) for c in text)


def as_text(result, cite, para=lambda text: text, heading=lambda level, text: '#' * level + ' ' + text):
    """The synthesis as text: cite(numbers) writes the citations, heading() and para() the format."""
    def line(s):
        return para(s['text']) + cite(s['papers'])
    out = [heading(1, result['title']), para(result['question']), '']
    if result['overview']:
        out += [heading(2, str(_('In short'))), ' '.join(line(s) for s in result['overview']), '']
    for theme in result['themes']:
        out += [heading(2, theme['name']), para(theme['summary']) + cite(theme['papers']), '']
        out += ['- ' + line(f) for f in theme['findings']]
        if theme['strengths'] or theme['limits']:
            out += ['', para('%s %s' % (theme['strengths'], theme['limits'])).strip()]
        out.append('')
    for name, title in (('consensus', _('Where the papers agree')), ('disagreements', _('Where they disagree')),
                        ('gaps', _('What they leave open'))):
        if result[name]:
            out += [heading(2, str(title))] + ['- ' + line(s) for s in result[name]] + ['']
    return out


def export_text(result, fmt, lang):
    entries, keys = cite_keys([exports.bibtex_entry(m) for m in metas(result)])
    if fmt == 'tex':
        body = as_text(result, lambda ns: ('~\\cite{%s}' % ','.join(keys[n - 1] for n in ns)) if ns else '',
                       para=latex, heading=lambda level, text: ('\\section{%s}' if level == 1 else '\\subsection*{%s}') % latex(text))
        body = [b if not b.startswith('- ') else '\\begin{itemize}\\item %s\\end{itemize}' % b[2:] for b in body]
        return '\n'.join(['% ' + str(_('Exploratory literature review written with SummarizePaper; check each statement in the papers.')),
                          '\\begin{filecontents*}[overwrite]{review.bib}'] + entries + ['\\end{filecontents*}', '',
                          '\\documentclass{article}', '\\usepackage[utf8]{inputenc}', '\\begin{document}'] + body +
                         ['\\bibliographystyle{plain}', '\\bibliography{review}', '\\end{document}', ''])
    lines = as_text(result, lambda ns: (' ' + ''.join('[%d]' % n for n in ns)) if ns else '')
    lines += ['## %s' % _('References'), '']
    for p in result['papers']:
        ref = 'https://doi.org/%s' % p['doi'] if p.get('doi') else sources.ref_url(p['key']) if sources.is_external(p['key']) \
            else 'https://arxiv.org/abs/%s' % exports.bare_ref(p['key'])
        lines.append('%d. %s (%s). %s. %s%s' % (p['n'], p['authors'], p.get('year') or 'n.d.', p['title'],
                                               '*%s*. ' % p['venue'] if p.get('venue') else '', ref))
    lines += ['', '_%s %s_' % (_('Exploratory literature review written with SummarizePaper on'), result.get('searched', '')),
              '%s/%s/' % (digest.BASE_URL, lang)]
    return '\n'.join(lines) + '\n'


def export_result(row, fmt):
    result = row.data()
    if fmt not in EXPORTS or not result.get('papers'):
        raise Http404
    name = re.sub(r'[^A-Za-z0-9]+', '-', (result.get('english') or result.get('title') or 'review'))[:60].strip('-') or 'review'
    if fmt in ('bib', 'ris'):
        if fmt == 'bib':
            entries, _keys = cite_keys([exports.bibtex_entry(m) for m in metas(result)])
            return exports._download('\n'.join(entries), name + '.bib', 'application/x-bibtex; charset=utf-8')
        return exports.export_many(metas(result), 'ris', name, row.lang)
    content_type = 'text/markdown; charset=utf-8' if fmt == 'md' else 'application/x-tex; charset=utf-8'
    with translation.override(row.lang):
        return exports._download(export_text(result, fmt, row.lang), '%s.%s' % (name, fmt), content_type)


@login_required
def literature_export(request, key, fmt):
    row = _row(request, key)
    if row.status != 'done':
        raise Http404
    return export_result(row, fmt)


def literature_shared_export(request, token, fmt):
    return export_result(result_share.shared_row('literature', token), fmt)


def zotero_metas(user, key):
    """The references of one of the user's reviews, for 'Send to Zotero'."""
    row = Generated.objects.filter(kind='literature', key=key, user=user, status='done').first()
    return metas(row.data()) if row is not None else []

