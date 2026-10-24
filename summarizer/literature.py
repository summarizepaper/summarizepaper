"""Literature review from a research question (Premium): an exploratory synthesis of 15 to 30 papers.

Not a systematic review: the searches are wide but not exhaustive, and the papers are read from their
abstracts. What the page promises instead is a search that can be checked: every query, source and count is
kept and shown, and each finding of the synthesis comes with the sentence of an abstract that supports it.

1. The model rewrites the question in English, lists its key concepts with their synonyms, and writes search
   queries that cover them (mechanisms, methods, settings) and a sentence that says which papers answer it.
2. OpenAlex (journals, preprints and conference papers, open or not) answers each query twice, by relevance
   and for the last RECENT_YEARS years; arXiv answers it too. One more OpenAlex search finds the reviews, one
   the most cited works. Then the works that the closest candidates cite most often are fetched (citation
   snowballing): the founding papers a keyword search misses. Those OpenAlex has no abstract for cannot be
   screened nor synthesized; they are listed apart, so the reader knows they exist.
3. The candidates are ranked by closeness to the question (embeddings); the closest are screened by the
   model on their abstract against the inclusion sentence.
4. The included papers are chosen so the founding and the recent works are both there.
5. The model groups them by approach and writes a comparative synthesis citing them; a finding keeps its
   quote only when the quote is word for word in the abstract of a paper it cites.
"""
import datetime
import logging
import re
import uuid

import numpy as np
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, digest, evidence, generated, library, similar, sources
from .models import Collection, CollectionPaper, Generated

logger = logging.getLogger(__name__)

RESULT_VERSION = 1
MIN_CHARS = 20
MAX_CHARS = 600
QUERIES = 6
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
UNREAD = 8  # often cited works without an abstract, listed apart
SCREENED = 80
SCREEN_BATCH = 40
MIN_PAPERS = 15
MAX_PAPERS = 30
RESERVED = 5  # places kept for founding and for recent papers
ABSTRACT_CHARS = 2500
PER_DAY = 3
QUOTA_KEY = 'literature-quota:%s:%s'
FIELDS = sources.OPENALEX_FIELDS + ',cited_by_count,referenced_works'
LABELS = ('founding', 'recent', 'review')
STEPS = [
    ('plan', _('Turning your question into searches')),
    ('search', _('Searching OpenAlex and arXiv')),
    ('snowball', _('Following the references of the closest papers')),
    ('screen', _('Ranking the candidates and screening their abstracts')),
    ('write', _('An AI groups the papers by approach and writes the synthesis')),
    ('check', _('Checking each finding against the abstracts')),
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
    return {'key': key, 'title': paper['title'], 'abstract': paper['abstract'], 'authors': paper['authors'],
            'year': int(paper['published'][:4]) if paper['published'] else None, 'venue': paper['journal'],
            'doi': paper['doi'], 'cited_by': work.get('cited_by_count') or 0, 'type': work.get('type') or '',
            'openalex': (work.get('id') or '').rsplit('/', 1)[-1],
            'references': [r.rsplit('/', 1)[-1] for r in work.get('referenced_works') or []]}


def from_arxiv(paper):
    return {'key': paper['arxiv_id'], 'title': ' '.join(paper['title'].split()), 'abstract': paper['abstract'],
            'authors': paper['authors'], 'year': paper['published'].year if paper.get('published') else None,
            'venue': 'arXiv', 'doi': '', 'cited_by': None, 'type': 'preprint', 'openalex': '', 'references': []}


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
            # the journal record of an arXiv preprint: keep the arXiv key, take its citations
            known.update(cited_by=paper['cited_by'], references=paper['references'], openalex=paper['openalex'],
                         doi=known['doi'] or paper['doi'], venue=paper['venue'] or known['venue'])
        if not known['abstract'] and paper['abstract']:
            known['abstract'] = paper['abstract']
        for name in names:
            self.index.setdefault(name, known)
        if found_by not in known['found_by']:
            known['found_by'].append(found_by)


def title_key(title):
    return re.sub(r'[^a-z0-9]+', ' ', title.lower()).strip()[:150]


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
    queries = list(dict.fromkeys(str(q).strip() for q in plan.get('queries', []) if str(q).strip()))[:QUERIES]
    english = str(plan.get('question') or question).strip()
    queries = queries or [english[:120]]
    return {'question': english, 'title': str(plan.get('title') or '').strip(), 'queries': queries,
            'core': str(plan.get('core') or '').strip() or queries[0],
            'include': str(plan.get('include') or '').strip(),
            'concepts': [str(c.get('concept', '')).strip() for c in plan.get('concepts', []) if isinstance(c, dict)][:8]}


def gather(plan, limited, step=lambda name: None):
    """The candidates of every search, and how many each search brought."""
    pool = Pool()
    since = (timezone.now() - datetime.timedelta(days=365 * RECENT_YEARS)).date().isoformat()
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
    papers = [p for p in papers if p['abstract']]
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


def screen(plan, papers, lang):
    """Each paper's relevance to the question (0 to 3) and its approach, from the model reading the abstracts."""
    for start in range(0, len(papers), SCREEN_BATCH):
        batch = papers[start:start + SCREEN_BATCH]
        listing = '\n\n'.join('<paper id="p%d">\nTitle: %s\nAbstract: %s\n</paper>' % (i, p['title'], p['abstract'][:900])
                              for i, p in enumerate(batch))
        answer = generated.chat_json(
            'Research question: %s\nPapers that help answer it: %s\n\nRate each paper above on its abstract: '
            '3 = it studies the question directly, 2 = it brings evidence, a method or a mechanism that matters for '
            'it, 1 = same field but it does not help answer it, 0 = off topic. Say in 2 to 6 words of %s the '
            'approach it takes to the question (e.g. "field sampling of sediments", "laboratory degradation '
            'experiment", "transport model"). Answer with a JSON object {"papers": [{"id": "p0", "relevance": 0, '
            '"approach": "..."}]}.' % (plan['question'], plan['include'] or plan['question'], generated.language_name(lang)),
            paid=False, max_tokens=3000, system=listing)
        by_id = {str(a.get('id')): a for a in answer.get('papers', []) if isinstance(a, dict)}
        for i, paper in enumerate(batch):
            note = by_id.get('p%d' % i, {})
            try:
                paper['relevance'] = max(0, min(3, int(note.get('relevance', 0))))
            except (TypeError, ValueError):
                paper['relevance'] = 0
            paper['approach'] = str(note.get('approach', '')).strip()[:80]


def labels(paper, now_year):
    found = []
    if paper['year'] and paper['year'] <= now_year - FOUNDING_YEARS and (
            paper.get('cocited', 0) >= SNOWBALL_MIN or 'most_cited' in paper['found_by'] or 'cited' in paper['found_by']):
        found.append('founding')
    if paper['year'] and paper['year'] >= now_year - RECENT_YEARS + 1:
        found.append('recent')
    if paper['type'] == 'review':
        found.append('review')
    return found


def select(screened, now_year):
    """15 to 30 papers: the relevant ones, with places kept for the founding and the recent works; the
    peripheral ones (relevance 1) only to reach MIN_PAPERS."""
    for paper in screened:
        paper['labels'] = labels(paper, now_year)
        paper['score'] = paper['relevance'] + paper['similarity']
    relevant = sorted((p for p in screened if p['relevance'] >= 2), key=lambda p: -p['score'])
    chosen = []
    founding = sorted((p for p in relevant if 'founding' in p['labels']), key=lambda p: -(p['cited_by'] or 0))
    recent = [p for p in relevant if 'recent' in p['labels']]
    for paper in founding[:RESERVED] + recent[:RESERVED] + relevant:
        if paper not in chosen and len(chosen) < MAX_PAPERS:
            chosen.append(paper)
    for paper in sorted((p for p in screened if p['relevance'] == 1), key=lambda p: -p['score']):
        if len(chosen) >= MIN_PAPERS:
            break
        paper['peripheral'] = True
        chosen.append(paper)
    return sorted(chosen, key=lambda p: (p['year'] or 0, p['title']))


SYNTHESIS = """Research question: {question}

The papers above were found for it and screened on their abstracts. Write, in {language}, an exploratory
synthesis of what they say about the question. Use only what the abstracts state; cite the papers by their number.
Answer with a JSON object:
{{"overview": [{{"text": "one sentence", "papers": [1, 2]}}],
 "themes": [{{"name": "the approach or explanation these papers share",
             "summary": "what this group of work does and finds, 2 or 3 sentences",
             "papers": [3, 5],
             "findings": [{{"text": "one finding, one sentence", "papers": [3], "quote": "..."}}],
             "strengths": "what this approach is good at, one sentence",
             "limits": "what it cannot show or leaves out, one sentence"}}],
 "consensus": [{{"text": "...", "papers": [..]}}],
 "disagreements": [{{"text": "...", "papers": [..]}}],
 "gaps": [{{"text": "what the papers leave open or do not study", "papers": [..]}}]}}
- overview: 3 to 5 sentences that answer the question as far as these papers allow.
- themes: 3 to 6 groups by approach (mechanism proposed, method, setting); every paper in at least one group.
- findings: 2 to 4 per theme. quote: the sentence of the abstract of the first paper cited that supports the
  finding, copied exactly, word for word, in the abstract's language.
- consensus, disagreements, gaps: 1 to 4 each; leave a list empty rather than stretch the papers.
- papers: the numbers of the papers that support the sentence, never a number that is not above."""


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


def statements(value, count):
    out = []
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and str(item.get('text', '')).strip():
            out.append({'text': str(item['text']).strip()[:1200], 'papers': cited_numbers(item.get('papers'), count)})
    return out


def synthesize(plan, chosen, lang):
    listing = '\n\n'.join(
        '<paper n="%d">\nTitle: %s\nAuthors: %s\nYear: %s\nType: %s\nCited by: %s\nAbstract: %s\n</paper>'
        % (n, p['title'], ', '.join(p['authors'][:4]), p['year'] or '?', p['type'] or '?',
           p['cited_by'] if p['cited_by'] is not None else '?', p['abstract'][:ABSTRACT_CHARS])
        for n, p in enumerate(chosen, 1))
    return generated.chat_json(SYNTHESIS.format(question=plan['question'], language=generated.language_name(lang)),
                               paid=True, max_tokens=10000, system=listing)


def checked(answer, chosen):
    """The synthesis with valid paper numbers only, and each finding's quote kept only when it is in the
    abstract of a paper the finding cites."""
    count = len(chosen)
    themes = []
    for theme in answer.get('themes', []) if isinstance(answer.get('themes'), list) else []:
        if not isinstance(theme, dict) or not str(theme.get('name', '')).strip():
            continue
        findings = []
        for raw in theme.get('findings') if isinstance(theme.get('findings'), list) else []:
            if not isinstance(raw, dict) or not str(raw.get('text', '')).strip():
                continue
            finding = {'text': str(raw['text']).strip()[:1200], 'papers': cited_numbers(raw.get('papers'), count),
                       'quote': ''}
            quote = str(raw.get('quote') or '')
            for n in finding['papers'] if quote else []:
                match = evidence.find(quote, chosen[n - 1]['abstract'])
                if match:
                    finding.update(quote=' '.join(match.group().split()), quoted=n)
                    break
            findings.append(finding)
        themes.append({'name': str(theme['name']).strip()[:200], 'summary': str(theme.get('summary', '')).strip()[:1500],
                       'papers': cited_numbers(theme.get('papers'), count), 'findings': findings,
                       'strengths': str(theme.get('strengths', '')).strip()[:600],
                       'limits': str(theme.get('limits', '')).strip()[:600]})
    return {'overview': statements(answer.get('overview'), count), 'themes': themes,
            'consensus': statements(answer.get('consensus'), count),
            'disagreements': statements(answer.get('disagreements'), count),
            'gaps': statements(answer.get('gaps'), count)}


def review(question, lang, step=lambda name: None):
    """The result shown on the page; step(name) reports progress."""
    now_year = timezone.now().year
    limited = []
    plan = plan_for(question, lang)
    step('search')
    pool = gather(plan, limited)
    step('snowball')
    closest = ranked(plan['question'], pool.papers)
    fetched = snowball(pool, closest, limited)
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
    years = [p['year'] for p in chosen if p['year']]
    unread = sorted((p for p in pool.papers if not p['abstract'] and p.get('cocited', 0) >= SNOWBALL_MIN),
                    key=lambda p: (-p['cocited'], -(p['cited_by'] or 0)))[:UNREAD]
    coverage = {
        'queries': plan['queries'], 'concepts': plan['concepts'], 'include': plan['include'],
        'candidates': len(pool.papers), 'with_abstract': len(candidates), 'by_source': by_source,
        'snowballed': fetched, 'screened': len(screened), 'relevant': relevant, 'included': len(chosen),
        'peripheral': sum(1 for p in chosen if p.get('peripheral')),
        'years': [min(years), max(years)] if years else [],
        'labels': {name: sum(1 for p in chosen if name in p['labels']) for name in LABELS},
        'limited': sorted(set(limited)), 'thin': relevant < MIN_PAPERS,
        'unread': [{'key': p['key'], 'title': p['title'], 'year': p['year'] or '', 'doi': p['doi'],
                    'cited_by': p['cited_by'], 'cocited': p['cocited'],
                    'authors': ', '.join(p['authors'][:3]) + (' et al.' if len(p['authors']) > 3 else '')} for p in unread],
    }
    result = {'version': RESULT_VERSION, 'question': question, 'english': plan['question'],
              'title': plan['title'] or plan['question'], 'coverage': coverage, 'papers': [], 'overview': [],
              'themes': [], 'consensus': [], 'disagreements': [], 'gaps': []}
    if not chosen:
        return result
    step('write')
    answer = synthesize(plan, chosen, lang)
    step('check')
    result.update(checked(answer, chosen))
    findings = [f for t in result['themes'] for f in t['findings']]
    coverage['findings'] = len(findings)
    coverage['quoted'] = sum(1 for f in findings if f['quote'])
    result['papers'] = [{
        'n': n, 'key': p['key'], 'title': p['title'],
        'authors': ', '.join(p['authors'][:4]) + (' et al.' if len(p['authors']) > 4 else ''),
        'year': p['year'] or '', 'venue': p['venue'], 'doi': p['doi'], 'cited_by': p['cited_by'],
        'labels': p['labels'], 'approach': p['approach'], 'peripheral': bool(p.get('peripheral')),
        'relevance': p['relevance'], 'found_by': p['found_by'],
    } for n, p in enumerate(chosen, 1)]
    return result


def write(question, lang):
    def work(pk):
        generated.progress(pk, 'plan')
        return review(question, lang, lambda name: generated.progress(pk, name))
    return work


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def literature_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/literature.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'literature')
    error = None
    question = ''
    if request.method == 'POST' and paid:
        question = ' '.join(request.POST.get('question', '').split())
        if not MIN_CHARS <= len(question) <= MAX_CHARS:
            error = 'length'
        elif not _spend(request.user):
            error = 'quota'
        else:
            row = generated.start('literature', uuid.uuid4().hex, get_language(), write(question, get_language()),
                                  user=request.user)
            return redirect('/%s/literature-review/%s/' % (get_language(), row.key))
    reviews = Generated.objects.filter(kind='literature', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/literature.html', {
        'paid': paid, 'error': error, 'reviews': reviews, 'per_day': PER_DAY, 'question': question,
        'min_chars': MIN_CHARS, 'max_chars': MAX_CHARS})


def _row(request, key):
    if not access.has_tool(request.user, 'literature'):
        raise Http404
    row = Generated.objects.filter(kind='literature', key=key, user=request.user).first()
    if row is None:
        raise Http404
    return row


@login_required
def literature_result(request, key):
    row = _row(request, key)
    if request.method == 'POST' and row.status == 'done':
        return save_collection(request, row)
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    result = row.data() if row.status == 'done' else None
    if result:
        numbers = {p['n']: p for p in result['papers']}
        for theme in result['themes']:
            theme['entries'] = [numbers[n] for n in theme['papers'] if n in numbers]
    return render(request, 'summarizer/literature_result.html', {
        'row': row, 'result': result, 'steps': STEPS, 'message': request.GET.get('m'),
        'collection': Collection.objects.filter(pk=request.GET.get('c'), user=request.user).first()
        if request.GET.get('c', '').isdigit() else None})


def save_collection(request, row):
    """Put the papers of the review in a new collection, to compare them in a table or chat with them."""
    paid = access.has_tool(request.user, 'collections')
    if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
        return redirect('/%s/literature-review/%s/?m=too_many' % (get_language(), row.key))
    result = row.data()
    collection = Collection.objects.create(user=request.user, name=str(result.get('title') or _('Literature review'))[:120])
    room = library.limits(paid)['papers']
    for paper in result.get('papers', [])[:room]:
        CollectionPaper.objects.get_or_create(collection=collection, arxiv_id=paper['key'],
                                              defaults={'title': paper['title'], 'authors': paper['authors']})
    return redirect('/%s/literature-review/%s/?m=saved&c=%s' % (get_language(), row.key, collection.pk))
