"""Weekly letters: 'AI this week', 'Space this week'... what happens next door, for researchers of other fields
and curious readers.

Once a week, for each theme, every paper of the last seven announcements is read: the model screens them
all (title and start of the abstract) by chunks, then what can be counted orders the ones it kept (how many
research profiles of our digests they come close to, whether a rising topic cites them, how many categories
they were posted in), and the model picks the six of the letter from their abstracts, at most two from one
category and one per team. It then writes an article about them in each language, with a short sheet per
paper (type of study, what the evidence rests on, main limit, code, publication), and every number of the
text is checked against the abstracts: a section that keeps a number the abstracts do not have is dropped.
The letters are public pages, in the sitemap and in RSS feeds; a free account gets one by email, Premium
all of them, read aloud, with the papers close to its research profile or its watched projects marked.
"""
import contextvars
import datetime
import json
import logging
import math
import random
import re
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.contrib.syndication.views import Feed
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.db import connection
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation, formats
from django.utils.translation import get_language, gettext
from django.utils.translation import gettext_lazy as _
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import access, digest, generated, mailcheck, similar, sources
from .models import (AnnouncedPaper, ArxivBatch, ArxivPaper, Collection, DigestSubscription, Letter, LetterFeedback,
                     LetterSubscription, PaperAudio, PaperEmbedding, PaperHistory, RisingTopics, WatchedProject)

logger = logging.getLogger(__name__)

AI = {'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.RO', 'stat.ML'}
TOPICS = {
    'ai': {'name': _('Artificial intelligence'), 'categories': AI},
    'space': {'name': _('Astronomy and space'), 'archives': {'astro-ph'}},
    'quantum': {'name': _('Quantum physics'), 'categories': {'quant-ph'}},
    'matter': {'name': _('Physics of matter and light'), 'archives': {'cond-mat'},
               'categories': {'physics.optics', 'physics.app-ph', 'physics.flu-dyn', 'physics.chem-ph', 'physics.comp-ph',
                              'physics.plasm-ph', 'physics.atom-ph'}},
    'particles': {'name': _('Particles, nuclei and gravitation'),
                  'archives': {'hep-ex', 'hep-ph', 'hep-th', 'hep-lat', 'nucl-ex', 'nucl-th', 'gr-qc'}},
    'brain': {'name': _('Brain and behaviour'),
              'categories': {'q-bio.NC', 'biorxiv.neuroscience', 'biorxiv.animal-behavior-and-cognition', 'medrxiv.neurology',
                             'medrxiv.psychiatry-and-clinical-psychology'}},
    'life': {'name': _('Biology and medicine'), 'archives': {'q-bio', 'biorxiv', 'medrxiv'},
             'categories': {'physics.bio-ph', 'physics.med-ph'}},
    'climate': {'name': _('Climate and Earth'),
                'categories': {'physics.ao-ph', 'physics.geo-ph', 'astro-ph.EP', 'biorxiv.ecology'}},
    'math': {'name': _('Mathematics and statistics'), 'archives': {'math', 'stat'}, 'exclude': {'stat.ML'}},
    'computing': {'name': _('Computing beyond AI'), 'archives': {'cs', 'eess'}, 'exclude': AI},
    'economy': {'name': _('Economics and finance'), 'archives': {'econ', 'q-fin'}},
}
LANGS = ('en', 'fr')
WEEK = datetime.timedelta(days=7)
PICKED = 6
RANKED = 12  # the model ranks this many, so that the category and team rules still leave six
MIN_PAPERS = 20
LETTERS_PER_RUN = 4
SCREEN_CHUNK = 200  # papers the model screens in one call
SCREEN_RATE = 25  # it keeps about one paper in this many, at least SCREEN_MIN per chunk
SCREEN_MIN = 4
SHORTLIST_MIN = 25  # a small field's chunks keep more, so that the model still chooses among this many
SCREEN_THREADS = 4
SHORTLIST = 40  # papers whose abstracts the model reads to choose the six
SHORTLIST_PER_CATEGORY = 10
PER_CATEGORY = 2  # papers of one category in a letter, when the theme spans several
RELATED = 2  # earlier papers shown with each paper
RELATED_MIN = 0.6
RELATED_SEARCHED = 10  # OpenAlex works read per paper to find earlier ones
CLAIMS_PER_SECTION = 4
SUMMARIES_PER_RUN = 6  # papers of the week's letters summarized per hourly run, so that their pages are ready
PROFILE_CLOSE = digest.THRESHOLDS['medium']
PROJECT_CLOSE = 0.5
RISING_SHOWN = 3
WORDS_PER_MINUTE = 230
FEED_ITEMS = 20
FREE_SUBSCRIPTIONS = 1
LOCK = 'letters-run'
UNSUBSCRIBE_SALT = 'letter-unsubscribe'
AUDIO_LOCK = 'letter-audio:%s'
STUDY_TYPES = {
    'observation': _('Observation'), 'experiment': _('Experiment'), 'simulation': _('Simulation'), 'theory': _('Theory'),
    'clinical': _('Clinical study'), 'trial': _('Trial'), 'review': _('Review'), 'benchmark': _('Benchmark'),
    'method': _('New method'), 'dataset': _('Data set'), 'other': _('Other'),
}
CODE_URL = re.compile(r'https?://(?:www\.)?(?:github\.com|gitlab\.com|zenodo\.org|huggingface\.co|figshare\.com|osf\.io|'
                      r'bitbucket\.org|codeocean\.com)/[^\s,;)\]}>"\']+', re.I)
ACCEPTED = re.compile(r'\b(?:accepted|to appear|in press|published)(?: for publication)?\s+(?:in|at|by|to|for)\s+'
                      r'(?:the\s+)?([A-Z][^.;,()\n]{1,60})', re.I)
NUMBER = re.compile(r'(?<![\w.,])\d+(?:[.,   ]\d+)*')
NUMBER_WORDS = {w: i for i, w in enumerate(
    'zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen '
    'eighteen nineteen twenty'.split())}
NUMBER_WORDS.update({'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90,
                     'hundred': 100, 'thousand': 1000, 'million': 1000000, 'billion': 1000000000, 'dozen': 12, 'half': 2})


def in_topic(topic, categories):
    spec = TOPICS[topic]
    return any(c not in spec.get('exclude', ()) and
               (c in spec.get('categories', ()) or c.split('.')[0] in spec.get('archives', ())) for c in categories)


def scope_in_topic(topic, scope):
    """A rising-topics scope (a category or an archive) inside a theme; an archive only when the theme takes all of it."""
    if '.' in scope:
        return in_topic(topic, [scope])
    spec = TOPICS[topic]
    return scope in spec.get('archives', ()) and not spec.get('exclude')


def topic_of_scope(scope):
    for topic in TOPICS:
        if scope_in_topic(topic, scope):
            return topic
    return None


def bare(arxiv_id):
    return arxiv_id if sources.is_external(arxiv_id) else re.sub(r'v\d+$', '', arxiv_id)


def week_papers(topic, day):
    papers = []
    for row in AnnouncedPaper.objects.filter(announced__gt=day - WEEK, announced__lte=day).order_by('arxiv_id'):
        categories = [c for c in row.categories.split(',') if c]
        if in_topic(topic, categories):
            papers.append({'arxiv_id': row.arxiv_id, 'title': row.title, 'abstract': row.abstract,
                           'authors': json.loads(row.authors or '[]'), 'categories': categories,
                           'primary': in_topic(topic, categories[:1])})
    return papers


def short(text, length):
    text = ' '.join((text or '').split())
    return text if len(text) <= length else text[:length].rsplit(' ', 1)[0] + '…'


# --- choosing the papers ---------------------------------------------------------------------------------------

def _in_thread(work, *args):
    """Run in a worker thread with the caller's usage task name; the thread's database connection is closed after."""
    context = contextvars.copy_context()

    def run():
        try:
            return context.run(work, *args)
        finally:
            connection.close()
    return run


def _screen_chunk(name, chunk, rate):
    keep = max(SCREEN_MIN, math.ceil(len(chunk) * rate))
    try:
        answer = generated.chat_json(
            'Here are %d new papers in "%s" (ID | title | start of the abstract). Keep up to %d that report the most '
            'significant, solid and concrete advances (a result, a measurement, a discovery, a method shown to work on '
            'real cases) that researchers of neighbouring fields would want to hear about. Leave out incremental '
            'variations, surveys, position papers, and papers whose abstract claims more than it shows. Answer with a '
            'JSON object {"ids": ["ID", ...]}.\n\n%s'
            % (len(chunk), name, keep, '\n'.join('%s | %s | %s' % (p['arxiv_id'], p['title'], short(p['abstract'], 300))
                                                  for p in chunk)),
            paid=False, max_tokens=800)
    except (requests.RequestException, ValueError, KeyError, generated.GenerationError):
        logger.exception('Screening a chunk of %s failed', name)
        return []
    ids = {p['arxiv_id'] for p in chunk}
    return [a for a in dict.fromkeys(answer.get('ids', [])) if isinstance(a, str) and a in ids][:keep]


def screen(topic, papers):
    """The ids the model keeps after reading every paper, by chunks, a few chunks at a time."""
    name = str(TOPICS[topic]['name'])
    rate = max(1 / SCREEN_RATE, SHORTLIST_MIN / max(1, len(papers)))
    chunks = [papers[i:i + SCREEN_CHUNK] for i in range(0, len(papers), SCREEN_CHUNK)]
    if len(chunks) == 1:
        return _screen_chunk(name, chunks[0], rate)
    runners = [_in_thread(_screen_chunk, name, chunk, rate) for chunk in chunks]  # each with a copy of this thread's context
    with ThreadPoolExecutor(SCREEN_THREADS) as pool:
        found = list(pool.map(lambda run: run(), runners))
    return [a for ids in found for a in ids]


def profile_matrix():
    """The rows of every research profile of an active digest, stacked, with the profile each row belongs to."""
    rows, owners = [], []
    for i, (vector, labels) in enumerate(DigestSubscription.objects.filter(active=True).exclude(profile_labels='')
                                         .values_list('profile_vector', 'profile_labels')):
        count = len(json.loads(labels or '[]'))
        if vector and count:
            matrix = np.frombuffer(bytes(vector), dtype=np.float32).reshape(count, -1)
            if matrix.shape[1] == similar.DIMENSIONS:
                rows.append(matrix)
                owners += [i] * count
    if not rows:
        return None, None
    return np.vstack(rows), np.array(owners)


def rising_rows(topic, lang, day):
    """[(scope, week, topics)]: the latest rising topics written up to the letter's week, of every scope of the theme."""
    rows = (RisingTopics.objects.filter(lang=lang, day__gt=day - 4 * WEEK, day__lte=day).order_by('-day')
            .values_list('scope', 'day', 'content'))
    found, seen = [], set()
    for scope, week, content in rows:
        if scope in seen or not scope_in_topic(topic, scope):
            continue
        seen.add(scope)
        topics = json.loads(content).get('topics', [])
        if topics:
            found.append((scope, week, topics))
    return found


def readers(papers, day):
    """{bare id: distinct visitors who opened the paper's page on the site since its announcement week}."""
    keys = {bare(p['arxiv_id']) for p in papers}
    seen = {}
    for arxiv_id, visitor in PaperHistory.objects.filter(created__date__gt=day - WEEK).values_list('arxiv_id', 'ip_address'):
        key = bare((arxiv_id or '').replace('/', '--'))
        if key in keys:
            seen.setdefault(key, set()).add(visitor)
    return {k: len(v) for k, v in seen.items()}


def signals(topic, day, papers, vectors):
    """{arxiv_id: (interest, in a rising topic, categories)}: what can be counted about each paper. Interest is how
    many research profiles of our digests it comes close to plus how many visitors opened it; a paper is in a rising
    topic when it uses the main term of one of the theme's last weeks (its own week is not counted yet when the
    letter is written: a week of submissions is complete the Saturday after it)."""
    from . import trends
    rising_ids, rising_terms = set(), set()
    for _scope, _week, topics in rising_rows(topic, 'en', day):
        for t in topics:
            rising_ids |= {key for key in t.get('ids') or [p.get('arxiv_id') for p in t.get('papers', [])]}
            if t.get('main'):
                rising_terms.add(t['main'])
    matrix, owners = profile_matrix()
    visitors = readers(papers, day)
    found = {}
    for paper in papers:
        vector = vectors.get(paper['arxiv_id'])
        interest = visitors.get(bare(paper['arxiv_id']), 0)
        if matrix is not None and vector is not None and vector.shape[0] == matrix.shape[1]:
            interest += len(set(owners[(matrix @ vector) >= PROFILE_CLOSE]))
        rising = bare(paper['arxiv_id']) in rising_ids or bool(
            rising_terms and rising_terms & trends.terms('%s. %s' % (paper['title'], paper['abstract'])))
        found[paper['arxiv_id']] = (interest, rising, len(paper['categories']))
    return found


def shortlist(topic, day, papers, screened, scores):
    """The papers whose abstracts the model reads: the screened ones, then the ones the signals put first, ordered
    by the signals, no more than SHORTLIST_PER_CATEGORY of one category while others are left; then the rest, when
    a field lives mostly in one category (bioRxiv neuroscience for the brain)."""
    noise = random.Random('%s%s' % (topic, day))
    tie = {p['arxiv_id']: noise.random() for p in papers}  # so that equal papers are not ordered by their ID
    by_signal = sorted(papers, key=lambda p: (scores[p['arxiv_id']], tie[p['arxiv_id']]), reverse=True)
    kept = set(screened) | {p['arxiv_id'] for p in by_signal[:10] if scores[p['arxiv_id']][:2] != (0, False)}
    found, per_category = [], {}
    for paper in by_signal:
        category = paper['categories'][0] if paper['categories'] else ''
        if paper['arxiv_id'] not in kept or per_category.get(category, 0) >= SHORTLIST_PER_CATEGORY:
            continue
        per_category[category] = per_category.get(category, 0) + 1
        found.append(paper)
        if len(found) == SHORTLIST:
            break
    chosen = {p['arxiv_id'] for p in found}
    found += [p for p in by_signal if p['arxiv_id'] in kept and p['arxiv_id'] not in chosen][:SHORTLIST - len(found)]
    return found


def diverse(ranked, by_id):
    """The first PICKED papers in the model's order, at most PER_CATEGORY of a category when the theme spans
    several, and never two that share an author."""
    categories = {by_id[a]['categories'][0] for a in ranked if by_id[a]['categories']}
    chosen, authors, per_category = [], set(), {}
    for arxiv_id in ranked:
        paper = by_id[arxiv_id]
        category = paper['categories'][0] if paper['categories'] else ''
        names = {n for n in (' '.join(re.sub(r'[^\w\s-]', ' ', a.lower()).split()) for a in paper['authors']) if n}
        if names & authors or (len(categories) >= 3 and per_category.get(category, 0) >= PER_CATEGORY):
            continue
        chosen.append(paper)
        authors |= names
        per_category[category] = per_category.get(category, 0) + 1
        if len(chosen) == PICKED:
            break
    return chosen


def select(topic, day, papers):
    """(chosen papers, selection notes, vectors) of a letter, from the theme's papers of the week."""
    taken = set()
    for content in Letter.objects.filter(day=day).exclude(topic=topic).values_list('content', flat=True):
        taken |= {bare(a) for a in json.loads(content).get('papers', {})}
    listed = [p for p in (p for p in papers if p['primary']) if bare(p['arxiv_id']) not in taken] or \
        [p for p in papers if bare(p['arxiv_id']) not in taken]
    by_id = {p['arxiv_id']: p for p in listed}
    vectors = digest.ensure_vectors(listed)
    screened = screen(topic, listed)
    scores = signals(topic, day, listed, vectors)
    candidates = shortlist(topic, day, listed, screened, scores)
    answer = generated.chat_json(
        'Here are %d papers of the week in "%s", kept from %d (ID | title | categories | abstract). Rank the %d that '
        'researchers of other fields and curious, educated readers would most want to know about: solid, concrete '
        'findings first, then new methods shown to work; varied subjects; no two papers on the same question. '
        'Answer with a JSON object {"ids": ["ID", ...]}, best first.\n\n%s'
        % (len(candidates), TOPICS[topic]['name'], len(listed), RANKED, '\n\n'.join(
            '%s | %s | %s\n%s' % (p['arxiv_id'], p['title'], ', '.join(p['categories'][:3]), short(p['abstract'], 900))
            for p in candidates)),
        paid=True, max_tokens=800)
    allowed = {p['arxiv_id'] for p in candidates}
    ranked = [a for a in dict.fromkeys(answer.get('ids', [])) if isinstance(a, str) and a in allowed]
    chosen = diverse(ranked, by_id)
    notes = {'screened': len(listed), 'kept': len(set(screened)), 'shortlist': len(candidates),
             'rising': sum(1 for p in chosen if scores[p['arxiv_id']][1]),
             'interest': sum(1 for p in chosen if scores[p['arxiv_id']][0])}
    return chosen, notes, vectors


def pick(topic, day, papers, lang, reuse=True):
    """(chosen papers, selection notes, vectors): the papers already chosen for this week in another language, or new ones."""
    other = Letter.objects.filter(topic=topic, day=day).exclude(lang=lang).first() if reuse else None
    by_id = {p['arxiv_id']: p for p in papers}
    if other is not None:
        content = json.loads(other.content)
        chosen = [by_id[a] for a in content.get('order') or content.get('papers', {}) if a in by_id]
        if chosen:
            return chosen, content.get('selection', {}), digest.stored_vectors([p['arxiv_id'] for p in chosen])
    return select(topic, day, papers)


# --- what each paper rests on ----------------------------------------------------------------------------------

def links_of(arxiv_id, categories):
    """(source name, page URL, PDF URL) of a paper."""
    if sources.is_external(arxiv_id):
        doi = arxiv_id[4:].replace('--', '/') if arxiv_id.startswith('doi_') else ''
        server = categories[0].split('.')[0] if categories else ''
        return digest.SERVER_NAMES.get(server, 'DOI'), 'https://doi.org/%s' % doi if doi else '', ''
    ident = arxiv_id.replace('--', '/')
    return 'arXiv', 'https://arxiv.org/abs/%s' % ident, 'https://arxiv.org/pdf/%s' % ident


def arxiv_details(papers):
    """{arxiv_id: entry of the arXiv API} (comments, journal reference, DOI, first version) of the arXiv papers."""
    ids = [p['arxiv_id'].replace('--', '/') for p in papers if not sources.is_external(p['arxiv_id'])]
    try:
        found = digest.fetch_ids(ids, timeout=30, retry=False)
    except digest.ArxivUnavailable:
        logger.warning('arXiv details of a letter could not be read')
        return {}
    return {bare(e['arxiv_id']): e for e in found}


def facts_of(papers):
    """{arxiv_id: what can be read from the paper's record}: authors, links, code, publication."""
    details = arxiv_details(papers)
    found = {}
    for paper in papers:
        entry = details.get(bare(paper['arxiv_id']), {})
        authors = paper['authors'] or entry.get('authors') or []
        comment, journal = entry.get('comment', ''), entry.get('journal_ref', '')
        paper['comment'] = comment  # its numbers count as stated (pages, figures, a conference year)
        source, url, pdf = links_of(paper['arxiv_id'], paper['categories'])
        code = CODE_URL.search('%s %s' % (paper['abstract'], comment))
        accepted = ACCEPTED.search(comment)
        published = entry.get('published')
        found[paper['arxiv_id']] = {
            'authors': [authors[0], authors[-1]] if len(authors) > 1 else authors, 'n_authors': len(authors),
            'source': source, 'url': url, 'pdf': pdf, 'category': paper['categories'][0] if paper['categories'] else '',
            'code': code.group(0).rstrip('.') if code else '',
            'journal': plain_math(journal or (accepted.group(1).strip() if accepted else '')),
            'status': 'published' if journal or entry.get('doi') else 'accepted' if accepted else 'preprint',
            'doi': entry.get('doi', ''), 'first': published.date().isoformat() if published else '',
        }
    return found


def site_candidates(paper, vector, by_category):
    """[(similarity, arxiv_id)]: the papers summarized on the site, of the paper's category, closest to it."""
    category = paper['categories'][0] if paper['categories'] else ''
    if not category or sources.is_external(paper['arxiv_id']):
        return []
    if category not in by_category:
        rows = list(PaperEmbedding.objects.filter(model=similar.MODEL, category=category).values_list('arxiv_id', 'vector'))
        by_category[category] = ([a for a, _ in rows], np.stack([np.frombuffer(bytes(v), dtype=np.float32) for _, v in rows])
                                 if rows else None)
    ids, matrix = by_category[category]
    if matrix is None or matrix.shape[1] != vector.shape[0]:
        return []
    scores = matrix @ vector
    return [(float(scores[i]), ids[i]) for i in np.argsort(-scores)[:20]]


def openalex_candidates(query, vector, day):
    """[(similarity, paper)]: works OpenAlex finds for the subject, published a month or more before the letter,
    compared with the paper by their title and abstract. Counted in the research tools' daily OpenAlex budget."""
    from . import literature
    limited = []
    works = literature.works({'search': query, 'per_page': RELATED_SEARCHED, 'filter': 'to_publication_date:%s,has_abstract:true'
                              % (day - 30 * datetime.timedelta(days=1)).isoformat()}, limited)
    found = [p for p in (literature.from_openalex(w) for w in works) if p and p['abstract']]
    if not found:
        return []
    try:
        vectors = similar.embed([short('%s. %s' % (p['title'], p['abstract']), similar.MAX_INPUT_CHARS) for p in found])
    except (requests.RequestException, KeyError, ValueError):
        logger.exception('Embedding earlier works failed')
        return []
    return [(float(v @ vector), p) for p, v in zip(found, vectors) if v.shape == vector.shape]


def earlier(chosen, vectors, day, queries):
    """{arxiv_id: [{arxiv_id, title, year, venue}]}: the closest earlier papers, among those summarized on the site
    (same category) and those OpenAlex finds for the subject of the paper's section (queries: {arxiv_id: query})."""
    week = {bare(p['arxiv_id']) for p in chosen}
    found, by_category = {}, {}
    for paper in chosen:
        vector = vectors.get(paper['arxiv_id'])
        if vector is None:
            continue
        candidates = []
        site = [(score, a) for score, a in site_candidates(paper, vector, by_category) if score >= RELATED_MIN and bare(a) not in week]
        known = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=[a for _, a in site], published_arxiv__lt=day - WEEK)}
        for score, arxiv_id in site:
            if arxiv_id in known and known[arxiv_id].title:
                candidates.append((score, {'arxiv_id': arxiv_id, 'title': ' '.join(known[arxiv_id].title.split()),
                                           'year': known[arxiv_id].published_arxiv.year, 'venue': 'arXiv'}))
        query = queries.get(paper['arxiv_id']) or paper['title']
        for score, work in openalex_candidates(query, vector, day):
            if score >= RELATED_MIN and bare(work['key']) not in week:
                candidates.append((score, {'arxiv_id': work['key'], 'title': ' '.join(work['title'].split()),
                                           'year': work['year'], 'venue': work['venue']}))
        related, seen = [], set()
        for _score, item in sorted(candidates, key=lambda c: -c[0]):
            title = re.sub(r'\W+', ' ', item['title'].lower()).strip()
            if bare(item['arxiv_id']) in seen or title in seen:
                continue
            seen |= {bare(item['arxiv_id']), title}
            related.append(item)
            if len(related) == RELATED:
                break
        if related:
            found[paper['arxiv_id']] = related
    return found


# --- plain text ------------------------------------------------------------------------------------------------

SUPERSCRIPT = str.maketrans('0123456789+-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻')
SYMBOLS = {'\\left': '', '\\right': '', '\\to': '→', '\\rightarrow': '→', '\\pm': '±', '\\sim': '~', '\\simeq': '≃',
           '\\approx': '≈', '\\times': '×', '\\le': '≤', '\\leq': '≤', '\\ge': '≥', '\\geq': '≥', '\\alpha': 'α',
           '\\beta': 'β', '\\gamma': 'γ', '\\delta': 'δ', '\\Delta': 'Δ', '\\epsilon': 'ε', '\\mu': 'μ', '\\nu': 'ν',
           '\\pi': 'π', '\\sigma': 'σ', '\\tau': 'τ', '\\lambda': 'λ', '\\Omega': 'Ω', '\\omega': 'ω', '\\odot': '☉',
           '\\infty': '∞', '\\%': '%', '\\,': ' ', '\\;': ' ', '\\!': '', '\\ ': ' '}


def _plain_formula(match):
    text = match.group(1)
    text = re.sub(r'\^\{([0-9+\-]+)\}|\^([0-9+\-])', lambda m: (m.group(1) or m.group(2)).translate(SUPERSCRIPT), text)
    for command in sorted(SYMBOLS, key=len, reverse=True):
        text = text.replace(command, SYMBOLS[command])
    text = re.sub(r'\\(?:mathcal|mathrm|mathbf|mathit|mathsf|text|textrm|rm|it|bf)\s*\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'_\{([^{}]*)\}|_(\w)', lambda m: m.group(1) or m.group(2), text)
    text = re.sub(r'\\[A-Za-z]+', '', text)
    return ' '.join(text.replace('{', '').replace('}', '').split())


def plain_math(text):
    """'$10^6$ s and $\\mathcal{F}t$ values' -> '10⁶ s and Ft values': formulas as text, for emails and feeds."""
    return re.sub(r'\$([^$]+)\$', _plain_formula, text or '')


# --- checking the numbers --------------------------------------------------------------------------------------

def _number(value):
    try:
        number = float(value)
    except ValueError:
        return None
    return ('%f' % number).rstrip('0').rstrip('.')


def number_values(token):
    """The numbers a written token can be: '3,706' or '3 706' is 3706, '2,7' or '2.7' is 2.7, '1.234,5' is 1234.5."""
    parts = re.split(r'[.,   ]', token)
    separators = re.findall(r'[.,   ]', token)
    values = {''.join(parts)}
    if separators and separators[-1] in '.,':
        values.add(''.join(parts[:-1]) + '.' + parts[-1])
    return {v for v in (_number(v) for v in values) if v is not None}


def numbers_in(text):
    """[(token, its possible values)] of the numbers written with digits in a text."""
    return [(m.group(0), number_values(m.group(0))) for m in NUMBER.finditer(text or '')]


def known_numbers(papers, extra=()):
    """Every number the abstracts, titles and comments of these papers state, in digits or in English words."""
    known = {_number(str(v)) for v in extra}
    for paper in papers:
        text = '%s %s %s' % (paper['title'], paper['abstract'], paper.get('comment', ''))
        for _token, values in numbers_in(text):
            known |= values
        known |= {_number(str(NUMBER_WORDS[w])) for w in re.findall(r'[a-z]+', text.lower()) if w in NUMBER_WORDS}
    return known


def unsupported(text, known):
    """The numbers of a text the papers do not state."""
    return [token for token, values in numbers_in(text) if values and not values & known]


# --- writing ---------------------------------------------------------------------------------------------------

def localized_name(topic, lang):
    with translation.override(lang):
        return str(TOPICS[topic]['name']), gettext('%(name)s this week') % {'name': str(TOPICS[topic]['name'])}



def normalized_title(topic, lang, title):
    """Repair only the letter heading, preserving the editorial title that follows."""
    title = str(title).strip()[:300]
    if lang == 'fr':
        name, heading = localized_name(topic, lang)
        english, _heading = localized_name(topic, 'en')
        title = re.sub(r'^(?:%s|%s)\s+this\s+week\b' % (re.escape(name), re.escape(english)),
                       lambda match: heading, title, count=1, flags=re.I)
        title = re.sub(r'\s*:\s*', '\u00a0: ', title, count=1)
    return title

def write(topic, day, lang, save=True, reuse=True):
    """The letter of a theme, week and language, stored unless save=False; reuse=False chooses the papers again
    even when another language already has them."""
    papers = week_papers(topic, day)
    if len(papers) < MIN_PAPERS:
        return None
    chosen, selection, vectors = pick(topic, day, papers, lang, reuse)
    if len(chosen) < 3:
        return None
    name, heading = localized_name(topic, lang)
    facts = facts_of(chosen)
    listing = '\n\n'.join('<paper id="%s">\nTitle: %s\nCategories: %s\nAbstract: %s\n</paper>'
                          % (p['arxiv_id'], p['title'], ', '.join(p['categories'][:3]), p['abstract']) for p in chosen)
    answer = generated.chat_json(
        'Write, in %s, the weekly letter "%s" from the new papers above (%d papers were announced in this field this '
        'week). It is read by researchers of other fields, who want to know what happens next door, and by curious, '
        'educated readers. Write it like a good science magazine article: a title that starts with "%s: ", a '
        'one-sentence subtitle, then one short section per paper or group of related papers, each with a heading and 1 '
        'or 2 paragraphs saying what was found, how, and why it matters beyond its subfield, and a closing paragraph on '
        'what links them. Stay faithful to the abstracts: every number you write must be in an abstract, as it is '
        'written there; say "preprint" or "not yet peer reviewed" where useful; no hype, no LaTeX, no markdown, never '
        'write IDs in the texts. For each section also give "query": its subject as a researcher would type it in a '
        'search engine, in English, 4 to 12 words. For each paper give "study": one of %s; "evidence": what the result '
        'rests on in a few words (a sample, a data set, a number of runs, an instrument), copied from the abstract, '
        'or "" if it does not say or for a proof or a theory; "limit": in one sentence, what the result does not show yet or the main caveat a referee would raise (a small sample, simulations only, one data set, no independent test, a proxy measure...), from the abstract, not a restatement of its scope. Answer with a JSON object {"title": "...", '
        '"dek": "...", "sections": [{"heading": "...", "text": "...", "papers": ["ID"], "query": "..."}], "closing": '
        '"...", "papers": {"ID": {"study": "...", "evidence": "...", "limit": "..."}}}.'
        % (generated.language_name(lang), heading, len(papers), heading, ', '.join(STUDY_TYPES)),
        paid=True, max_tokens=6000, system=listing)
    titles = {p['arxiv_id']: p['title'] for p in chosen}
    sections = [{'heading': str(s.get('heading', '')).strip(), 'text': str(s.get('text', '')).strip(),
                 'papers': [a for a in dict.fromkeys(s.get('papers') or []) if a in titles],
                 'query': ' '.join(str(s.get('query') or s.get('heading') or '').split())[:200]}
                for s in answer.get('sections', []) if isinstance(s, dict) and s.get('text')]
    if not sections or not answer.get('title'):
        return None
    texts = {'dek': str(answer.get('dek', '')).strip(), 'closing': str(answer.get('closing', '')).strip()}
    sections, texts, check = verify(sections, texts, chosen, len(papers), lang, listing)
    if len(sections) < 3:
        logger.warning('Letter %s %s %s: only %d sections hold against the abstracts', topic, day, lang, len(sections))
        return None
    known = known_numbers(chosen)
    written_facts = answer.get('papers') if isinstance(answer.get('papers'), dict) else {}
    for arxiv_id, written in written_facts.items():
        if arxiv_id in facts and isinstance(written, dict):
            evidence = plain_math(str(written.get('evidence') or '').strip())[:200]
            facts[arxiv_id].update({
                'study': written.get('study') if written.get('study') in STUDY_TYPES else 'other',
                'evidence': '' if unsupported(evidence, known) else evidence,
                'limit': plain_math(str(written.get('limit') or '').strip())[:400]})
    title = normalized_title(topic, lang, answer['title'])
    selection = dict(selection, sources=sorted({facts[p['arxiv_id']]['source'] for p in chosen}), **check)
    order = [p['arxiv_id'] for p in chosen]
    queries = {}
    for section in sections:
        for arxiv_id in section['papers']:
            queries.setdefault(arxiv_id, section['query'])
    content = {'sections': sections, 'closing': texts['closing'], 'papers': titles, 'count': len(papers),
               'order': order, 'facts': facts, 'earlier': earlier(chosen, vectors, day, queries), 'selection': selection}
    stacked = [vectors[a] for a in order if a in vectors]
    fields = {'title': title, 'dek': texts['dek'], 'content': json.dumps(content), 'papers': ',%s,' % ','.join(bare(a) for a in order),
              'vectors': np.stack(stacked).astype(np.float32).tobytes() if len(stacked) == len(order) else None}
    if not save:
        return Letter(topic=topic, lang=lang, day=day, **fields)
    letter, _created = Letter.objects.update_or_create(topic=topic, lang=lang, day=day, defaults=fields)
    return letter


def claim_problems(sections, chosen, lang, listing, keys):
    """({key: [statement]}, statements checked): the statements of these sections ('s0', 's1'...) about what their
    papers found or did that no passage of the abstracts supports. The model names the passage of each statement;
    it is accepted only when it is in the abstract word for word (evidence.find), and 'unsupported' is kept apart
    from a passage that was not found."""
    from . import evidence
    by_id = {p['arxiv_id']: p for p in chosen}
    if not keys:
        return {}, {}, 0
    try:
        answer = generated.chat_json(
            'Check these sections of a letter written in %s from the papers above. For each section, list its statements '
            'about what the papers found, measured or did, %d at most; leave out background explanations and why it '
            'matters. For each statement give "claim": the statement in English, one sentence; "paper": the ID of the '
            'paper it is about; "basis": "abstract" when the abstract states it, "unsupported" when no passage of the '
            'abstract supports it as stated (never stretch a passage to fit); "quote": for "abstract", the sentence of '
            'the abstract that supports it, copied word for word, with no ellipsis. Answer with a JSON object '
            '{"sections": [{"key": "s0", "claims": [{"claim": "...", "paper": "ID", "basis": "...", "quote": "..."}]}]}.\n\n%s'
            % (generated.language_name(lang), CLAIMS_PER_SECTION, '\n\n'.join(
                '[%s]\n%s\n%s' % (k, sections[int(k[1:])]['heading'], sections[int(k[1:])]['text']) for k in keys)),
            paid=True, max_tokens=5000, system=listing)
    except (requests.RequestException, ValueError, KeyError, generated.GenerationError):
        logger.exception('Checking the statements of a letter failed')
        return {}, {}, 0
    unsupported_claims, unfound, checked = {}, {}, 0
    for item in answer.get('sections') or []:
        key = item.get('key') if isinstance(item, dict) else None
        if key not in keys:
            continue
        own = [by_id[a] for a in sections[int(key[1:])]['papers'] if a in by_id] or chosen
        for claim in (item.get('claims') or [])[:CLAIMS_PER_SECTION]:
            if not isinstance(claim, dict) or not claim.get('claim'):
                continue
            checked += 1
            statement = str(claim['claim'])[:300]
            if claim.get('basis') == 'unsupported':
                unsupported_claims.setdefault(key, []).append(statement)
            elif not any(evidence.find(str(claim.get('quote') or ''), p['abstract']) for p in own):
                unfound.setdefault(key, []).append(statement)
    return unsupported_claims, unfound, checked


def verify(sections, texts, chosen, count, lang, listing):
    """(sections, texts, check): every number of a section must be in the abstracts of its own papers (those of the
    subtitle and closing in any), and every statement of a section on what its papers found must be in their
    abstracts. The model rewrites the parts that fail once; the sections still holding a number the abstracts do
    not state, or a statement the model itself calls unsupported, are dropped, the subtitle or closing emptied.
    A statement whose passage is still not found is only counted ('unverified'): the model may misquote."""
    by_id = {p['arxiv_id']: p for p in chosen}
    extra = (count, len(chosen))
    every = known_numbers(chosen, extra=extra)

    def known(i):
        own = [by_id[a] for a in sections[i]['papers'] if a in by_id]
        return known_numbers(own, extra=extra) if own else every
    numbers = sum(len(numbers_in(s['text']) + numbers_in(s['heading'])) for s in sections) + \
        sum(len(numbers_in(t)) for t in texts.values())

    def number_problems(keys=None):
        found = {'s%d' % i: unsupported(s['heading'] + ' ' + s['text'], known(i)) for i, s in enumerate(sections)}
        found.update({k: unsupported(v, every) for k, v in texts.items()})
        return {k: v for k, v in found.items() if v and (keys is None or k in keys)}
    section_keys = ['s%d' % i for i in range(len(sections))]
    bad_numbers = number_problems()
    bad_claims, unfound, claims = claim_problems(sections, chosen, lang, listing, section_keys)
    rewritten = set()
    flagged = set(bad_numbers) | set(bad_claims) | set(unfound)
    if flagged:
        parts = {k: (sections[int(k[1:])]['heading'] + '\n' + sections[int(k[1:])]['text']) if k.startswith('s') else texts[k]
                 for k in flagged}
        problems = []
        for k in sorted(flagged):
            if bad_numbers.get(k):
                problems.append('%s: numbers that are not in the abstracts of its papers: %s' % (k, ', '.join(bad_numbers[k])))
            for statement in bad_claims.get(k, []) + unfound.get(k, []):
                problems.append('%s: statement the abstracts do not support as written: %s' % (k, statement))
        try:
            answer = generated.chat_json(
                'Some parts of a letter written in %s from the papers above do not hold against the abstracts:\n%s\n\n'
                'Rewrite each of these parts, in the same language and style, so that every number and every statement on '
                'what the papers found is in their abstracts: drop or soften what is not, use the exact figures of the '
                'abstracts; change nothing else. For a section, the first line is its heading. Answer with a JSON object '
                '{"parts": {"key": "text"}}.\n\n%s'
                % (generated.language_name(lang), '\n'.join('- ' + p for p in problems),
                   '\n\n'.join('[%s]\n%s' % (k, v) for k, v in parts.items())),
                paid=True, max_tokens=4000, system=listing)
            for key, text in (answer.get('parts') or {}).items():
                text = str(text or '').strip()
                if key in flagged and text:
                    rewritten.add(key)
                    if key.startswith('s'):
                        heading, _sep, body = text.partition('\n')
                        section = sections[int(key[1:])]
                        section['heading'], section['text'] = (heading.strip(), body.strip()) if body.strip() else (section['heading'], text)
                    else:
                        texts[key] = text
        except (requests.RequestException, ValueError, KeyError, generated.GenerationError):
            logger.exception('Rewriting a letter against its abstracts failed')
        bad_numbers = number_problems()
        again = sorted(k for k in rewritten if k.startswith('s'))
        if again:
            claims_again, unfound_again, more = claim_problems(sections, chosen, lang, listing, again)
            claims += more
            for key in again:
                bad_claims[key], unfound[key] = claims_again.get(key, []), unfound_again.get(key, [])
    dropped = sorted({int(k[1:]) for k in bad_numbers if k.startswith('s')} |
                     {int(k[1:]) for k, v in bad_claims.items() if v})
    for key in bad_numbers:
        if not key.startswith('s'):
            texts[key] = ''
    kept = [s for i, s in enumerate(sections) if i not in dropped]
    unverified = sum(len(v) for k, v in unfound.items() if int(k[1:]) not in dropped)
    return kept, texts, {'numbers': numbers, 'claims': claims, 'rewritten': len(rewritten), 'dropped': len(dropped),
                         'unverified': unverified}


def update(now=None):
    """From the hourly digest run: write the letters of the week, a few per run."""
    if not cache.add(LOCK, True, 3600):
        return 0
    try:
        day = ArxivBatch.objects.order_by('-day').values_list('day', flat=True).first()
        if day is None:
            return 0
        last = Letter.objects.order_by('-day').values_list('day', flat=True).first()
        target = day if last is None or day - last >= WEEK else last
        done = set(Letter.objects.filter(day=target).values_list('topic', 'lang'))
        tried = cache.get('letters-tried:%s' % target) or set()
        written = 0
        for topic in TOPICS:
            for lang in LANGS:
                if (topic, lang) in done or (topic, lang) in tried or written >= LETTERS_PER_RUN:
                    continue
                try:
                    if write(topic, target, lang):
                        written += 1
                    else:
                        tried.add((topic, lang))  # too few papers this week: not tried again
                except Exception:
                    logger.exception('Letter %s %s failed', topic, lang)
        cache.set('letters-tried:%s' % target, tried, 8 * 86400)
        return written
    finally:
        cache.delete(LOCK)


def presummarize_due(budget=SUMMARIES_PER_RUN):
    """From the hourly digest run: write the summary of a few papers of the latest letters that have none yet in
    the letter's language, so that their pages open on a summary and show which letter presents them. How many."""
    day = Letter.objects.order_by('-day').values_list('day', flat=True).first()
    if day is None:
        return 0
    left = [budget]
    for letter in Letter.objects.filter(day=day).order_by('topic', 'lang').only('lang', 'content'):
        if left[0] <= 0:
            break
        from .presskit import paper_id
        ids = [paper_id(a) or a for a in json.loads(letter.content).get('order') or []]
        digest.presummarize(ids, letter.lang, left)
    return budget - left[0]


# --- email -----------------------------------------------------------------------------------------------------

def tracked(url, content):
    return '%s%sutm_source=letter&utm_medium=email&utm_content=%s' % (url, '&' if '?' in url else '?', content)


def allowed_subscriptions(user):
    """The active subscriptions of a user that get emails: all with Premium, the oldest FREE_SUBSCRIPTIONS without."""
    rows = list(LetterSubscription.objects.filter(user=user, active=True).order_by('created', 'pk'))
    return rows if access.has_tool(user, 'letters') else rows[:FREE_SUBSCRIPTIONS]


def send(now, budget, base_url=digest.BASE_URL):
    """Email the new letters to their subscribers, within the email budget."""
    from django.core import signing
    suppressed = mailcheck.suppressed()
    sent = 0
    allowed = {}
    for subscription in LetterSubscription.objects.filter(active=True).select_related('user').order_by('created', 'pk'):
        if sent >= budget:
            break
        letter = Letter.objects.filter(topic=subscription.topic, lang=subscription.lang).first()
        if letter is None or (subscription.last_letter and letter.day <= subscription.last_letter):
            continue
        user = subscription.user
        if not user.email or user.email.strip().lower() in suppressed:
            continue
        if user.pk not in allowed:
            allowed[user.pk] = {s.pk for s in allowed_subscriptions(user)}
        if subscription.pk not in allowed[user.pk]:
            continue
        token = signing.dumps(subscription.pk, salt=UNSUBSCRIBE_SALT)
        url = '%s/%s/letters/%s/%s/' % (base_url, letter.lang, letter.topic, letter.day.isoformat())
        context = dict(letter_context(letter), base_url=base_url, url=tracked(url, 'web'),
                       tools_url=tracked(url + '#papers', 'tools'), paid=access.has_tool(user, 'letters'),
                       manage_url=tracked('%s/%s/letters/' % (base_url, letter.lang), 'manage'),
                       unsubscribe_url='%s/%s/letters/unsubscribe/%s/' % (base_url, letter.lang, token))
        for section in context['sections']:
            for link in section['links']:
                link['tracked'] = tracked('%s/%s/paper/%s/' % (base_url, letter.lang, link['arxiv_id']), 'paper')
                link['title'] = plain_math(link['title'])
        context['feedback_url'] = tracked(url + '#feedback', 'feedback')
        with translation.override(letter.lang):
            text = render_to_string('letters/email.txt', context)
            html = render_to_string('letters/email.html', context)
        message = EmailMultiAlternatives(letter.title, text, settings.DEFAULT_FROM_EMAIL, [user.email],
                                         headers={'List-Unsubscribe': '<%s>' % context['unsubscribe_url'],
                                                  'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click'})
        message.attach_alternative(html, 'text/html')
        try:
            message.send()
        except Exception:
            logger.exception('Letter email to %s failed', user.username)
            continue
        subscription.last_letter = letter.day
        subscription.save(update_fields=['last_letter'])
        sent += 1
    return sent


# --- reading the letters ---------------------------------------------------------------------------------------

def letter_vectors(letter):
    """{arxiv_id: vector} of the letter's papers, as stored with it."""
    content = json.loads(letter.content)
    order = content.get('order') or []
    if not letter.vectors or not order:
        return {}
    matrix = np.frombuffer(bytes(letter.vectors), dtype=np.float32).reshape(len(order), -1)
    return dict(zip(order, matrix))


def profile_of(user):
    """The stacked rows of the research profiles of a user's digests, or None."""
    rows = []
    for vector, labels in DigestSubscription.objects.filter(user=user, active=True).exclude(profile_labels='') \
            .values_list('profile_vector', 'profile_labels'):
        count = len(json.loads(labels or '[]'))
        if vector and count:
            rows.append(np.frombuffer(bytes(vector), dtype=np.float32).reshape(count, -1))
    return np.vstack(rows) if rows else None


def personal_marks(user, letter):
    """{arxiv_id: [why it is close to the user]}: close to their research profile, or to a watched project."""
    vectors = letter_vectors(letter)
    if not vectors:
        return {}
    marks = {}
    profile = profile_of(user)
    projects = [(p, np.frombuffer(bytes(p.vector), dtype=np.float32)) for p in WatchedProject.objects.filter(user=user)
                if p.vector]
    with translation.override(letter.lang):
        for arxiv_id, vector in vectors.items():
            if profile is not None and profile.shape[1] == vector.shape[0] and float(np.max(profile @ vector)) >= PROFILE_CLOSE:
                marks.setdefault(arxiv_id, []).append({'text': gettext('Close to your research profile'), 'url': 'digest/'})
            for project, project_vector in projects:
                if project_vector.shape == vector.shape and float(project_vector @ vector) >= PROJECT_CLOSE:
                    marks.setdefault(arxiv_id, []).append({
                        'text': gettext('Close to your project "%(title)s"') % {'title': project.title or gettext('Your draft')},
                        'url': 'scoops/#project-%d' % project.pk})
    return marks


def letter_context(letter):
    from .presskit import paper_id
    content = json.loads(letter.content)
    titles = content.get('papers', {})
    facts = content.get('facts', {})
    related = content.get('earlier', {})
    words = 0
    for section in content['sections']:
        section['links'] = []
        for arxiv_id in section['papers']:
            fact = dict(facts.get(arxiv_id) or {})
            if fact:
                fact['study_label'] = STUDY_TYPES.get(fact.get('study'), '')
            section['links'].append({'arxiv_id': arxiv_id, 'title': titles.get(arxiv_id, arxiv_id), 'presskit_id': paper_id(arxiv_id),
                                     'facts': fact, 'earlier': related.get(arxiv_id, [])})
        words += len(section['text'].split()) + len(section['heading'].split())
    words += len(content.get('closing', '').split()) + len((letter.dek or '').split())
    with translation.override(letter.lang):
        name = str(TOPICS[letter.topic]['name'])
    order = content.get('order') or list(titles)
    return {'letter': letter, 'sections': content['sections'], 'closing': content.get('closing', ''),
            'count': content.get('count'), 'topic_name': name, 'selection': content.get('selection') or {},
            'minutes': max(1, math.ceil(words / WORDS_PER_MINUTE)), 'chat_ids': ','.join(order[:10])}


def rising_for(letter):
    """The first rising topics of the theme, as of the letter's week."""
    found, seen = [], set()
    for scope, week, topics in rising_rows(letter.topic, letter.lang, letter.day):
        for topic in topics[:2]:
            if topic.get('main') in seen:
                continue
            seen.add(topic.get('main'))
            llr = topic['stats'][0]['llr'] if topic.get('stats') else 0
            found.append((llr, dict(topic, scope=scope, week=week)))
    from . import trends
    return [dict(t, scope_name=trends.scope_name(t['scope'])) for _llr, t in sorted(found, key=lambda f: -f[0])[:RISING_SHOWN]]


def latest_letters(lang):
    latest = {}
    for letter in Letter.objects.filter(lang=lang).order_by('-day').defer('content', 'vectors')[:120]:
        latest.setdefault(letter.topic, letter)
    return latest


def letters_page(request):
    lang = get_language()
    paid = access.has_tool(request.user, 'letters')
    subscribed = set()
    if request.user.is_authenticated:
        if request.method == 'POST':
            wanted = [t for t in request.POST.getlist('topics') if t in TOPICS]
            if not paid:
                wanted = wanted[:FREE_SUBSCRIPTIONS]
            for topic in TOPICS:
                LetterSubscription.objects.update_or_create(user=request.user, topic=topic, defaults={
                    'active': topic in wanted, 'lang': lang})
            return redirect('/%s/letters/?saved=1' % lang)
        subscribed = {s.topic for s in (LetterSubscription.objects.filter(user=request.user, active=True) if paid
                                        else allowed_subscriptions(request.user))}
    editions = list(Letter.objects.filter(lang=lang, topic__in=TOPICS).order_by('-day')
                    .values_list('day', flat=True).distinct())
    latest_day = editions[0] if editions else None
    selected = request.GET.get('edition', '')
    selected_day = latest_day
    if selected:
        try:
            selected_day = datetime.date.fromisoformat(selected)
        except ValueError:
            raise Http404
        if selected_day not in editions:
            raise Http404
        latest = {letter.topic: letter for letter in Letter.objects.filter(lang=lang, day=selected_day,
                    topic__in=TOPICS).defer('content', 'vectors')}
    else:
        latest = latest_letters(lang)
    index = editions.index(selected_day) if selected_day in editions else 0
    next_day = latest_day + WEEK if latest_day else None
    next_date = formats.date_format(next_day, 'l j F Y') if next_day else ''
    if lang.startswith('fr'):
        next_date = next_date[:1].lower() + next_date[1:]
    topics = [{'slug': slug, 'name': spec['name'], 'letter': latest.get(slug), 'subscribed': slug in subscribed}
              for slug, spec in TOPICS.items()]
    return render(request, 'summarizer/letters.html', {
        'topics': topics, 'paid': paid, 'free_count': FREE_SUBSCRIPTIONS,
        'saved': request.GET.get('saved') == '1', 'editions': editions, 'selected_day': selected_day,
        'archive_selected': bool(selected), 'latest_day': latest_day, 'next_date': next_date,
        'edition_pending': bool(next_day and timezone.localdate() >= next_day),
        'edition_count': len(latest),
        'older_edition': editions[index + 1] if index + 1 < len(editions) else None,
        'newer_edition': editions[index - 1] if index > 0 else None,
    })


def topic_page(request, topic):
    if topic not in TOPICS:
        raise Http404
    letters = Letter.objects.filter(topic=topic, lang=get_language()).order_by('-day').defer('content', 'vectors')[:52]
    return render(request, 'summarizer/letter_topic.html', {'topic': topic, 'name': TOPICS[topic]['name'], 'letters': letters})


def _letter(topic, day, lang):
    try:
        day = datetime.date.fromisoformat(day)
    except ValueError:
        raise Http404
    letter = Letter.objects.filter(topic=topic, day=day, lang=lang).first()
    if letter is None or topic not in TOPICS:
        raise Http404
    return letter


def letter_page(request, topic, day):
    letter = _letter(topic, day, get_language())
    other = Letter.objects.filter(topic=topic, day=letter.day).exclude(lang=letter.lang).values_list('lang', flat=True).first()
    same = Letter.objects.filter(topic=topic, lang=letter.lang)
    paid = access.has_tool(request.user, 'letters')
    context = dict(letter_context(letter), other_lang=other, paid=paid,
                   older=same.filter(day__lt=letter.day).order_by('-day').values_list('day', flat=True).first(),
                   newer=same.filter(day__gt=letter.day).order_by('day').values_list('day', flat=True).first(),
                   rising=rising_for(letter), audio=paid)
    if paid and request.user.is_authenticated:
        marks = personal_marks(request.user, letter)
        for section in context['sections']:
            for link in section['links']:
                link['marks'] = marks.get(link['arxiv_id'], [])
    return render(request, 'summarizer/letter.html', context)


@require_POST
@login_required
def letter_collection(request, topic, day):
    """The papers of a letter in a new collection of the user."""
    from . import library
    from .presskit import paper_id
    letter = _letter(topic, day, get_language())
    paid = access.has_tool(request.user, 'collections')
    if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
        return redirect('/%s/collections/?full=1' % get_language())
    content = json.loads(letter.content)
    keys = [paper_id(a) or a for a in content.get('order') or content.get('papers', {})]
    collection = Collection.objects.create(user=request.user, name=letter.title[:120])
    library.add_many(collection, keys, paid)
    return redirect('/%s/collections/%s/' % (get_language(), collection.pk))


@require_POST
def letter_feedback(request, topic, day):
    """'Useful' or 'not useful' on the letter (no arxiv_id) or one of its papers; a reader may change their vote."""
    letter = _letter(topic, day, get_language())
    try:
        value = int(request.POST.get('value', ''))
    except ValueError:
        value = 0
    arxiv_id = request.POST.get('arxiv_id', '')
    order = json.loads(letter.content).get('order') or list(json.loads(letter.content).get('papers', {}))
    if value not in (1, -1) or (arxiv_id and arxiv_id not in order):
        return JsonResponse({'status': 'invalid'}, status=400)
    visitor = 'u%d' % request.user.pk if request.user.is_authenticated else 'ip' + access.hash_ip(access.client_ip(request))[:64]
    LetterFeedback.objects.update_or_create(letter=letter, arxiv_id=arxiv_id, visitor=visitor, defaults={'value': value})
    return JsonResponse({'status': 'saved'})


def review(weeks=2):
    """For the staff activity page: the letters of the last weeks, with how their papers were chosen and checked,
    and what readers said of them."""
    from django.db.models import Count, Q
    day = Letter.objects.order_by('-day').values_list('day', flat=True).first()
    if day is None:
        return []
    rows = []
    letters = Letter.objects.filter(day__gt=day - weeks * WEEK).order_by('-day', 'topic', 'lang').annotate(
        up=Count('feedback', filter=Q(feedback__value=1, feedback__arxiv_id='')),
        down=Count('feedback', filter=Q(feedback__value=-1, feedback__arxiv_id='')),
        paper_up=Count('feedback', filter=Q(feedback__value=1) & ~Q(feedback__arxiv_id='')),
        paper_down=Count('feedback', filter=Q(feedback__value=-1) & ~Q(feedback__arxiv_id='')))
    for letter in letters:
        selection = json.loads(letter.content).get('selection') or {}
        rows.append({'letter': letter, 'url': '/%s/letters/%s/%s/' % (letter.lang, letter.topic, letter.day.isoformat()),
                     'selection': selection, 'up': letter.up, 'down': letter.down, 'paper_up': letter.paper_up,
                     'paper_down': letter.paper_down,
                     'flag': bool(selection.get('dropped') or selection.get('unverified') or letter.down > letter.up)})
    return rows


# --- audio -----------------------------------------------------------------------------------------------------

def audio_parts(letter):
    """The texts read aloud, one clip each: the title and subtitle, each section, the closing."""
    context = letter_context(letter)
    parts = ['%s.\n\n%s' % (letter.title.rstrip('.'), letter.dek)]
    parts += ['%s.\n\n%s' % (s['heading'].rstrip('.'), s['text']) for s in context['sections']]
    if context['closing']:
        parts.append(context['closing'])
    return parts


def audio_key(letter, index):
    return 'letter-%d-%d' % (letter.pk, index)


def make_audio(letter_pk):
    from . import audio
    letter = Letter.objects.filter(pk=letter_pk).first()
    try:
        if letter is None:
            return
        for index, text in enumerate(audio_parts(letter)):
            key = audio_key(letter, index)
            if PaperAudio.objects.filter(arxiv_id=key, lang=letter.lang).exists():
                continue
            try:
                data = audio.speak(text)
            except requests.RequestException:
                logger.exception('Text to speech of letter %s failed', letter_pk)
                return
            PaperAudio.objects.get_or_create(arxiv_id=key, lang=letter.lang, defaults={'mp3': data})
    finally:
        cache.delete(AUDIO_LOCK % letter_pk)
        generated.release_connection()


def letter_audio(request, topic, day):
    """The letter read aloud (Premium): the MP3 once every clip is made, else {'status': 'pending'} while they are.
    The check answers with the second each clip starts at, for the chapters under the player."""
    from . import audio
    letter = _letter(topic, day, get_language())
    if not access.has_tool(request.user, 'letters'):
        raise Http404
    keys = [audio_key(letter, i) for i in range(len(audio_parts(letter)))]
    clips = {a.arxiv_id: a for a in PaperAudio.objects.filter(arxiv_id__in=keys, lang=letter.lang)}
    if len(clips) == len(keys):
        if request.GET.get('check'):
            starts, at = [], 0.0
            for key in keys:
                starts.append(round(at, 1))
                at += audio.mp3_seconds(clips[key].mp3)
            return JsonResponse({'status': 'ready', 'chapters': starts})
        return audio.mp3_response(request, b''.join(bytes(clips[k].mp3) for k in keys))
    if cache.add(AUDIO_LOCK % letter.pk, True, 600):
        generated.background(make_audio, letter.pk)
    return JsonResponse({'status': 'pending', 'ready': len(clips), 'total': len(keys)}, status=202)


# --- feeds -----------------------------------------------------------------------------------------------------

class LetterFeed(Feed):
    """RSS of the letters of one theme, or of every theme, in the language of the URL."""

    def get_object(self, request, topic=None):
        if topic is not None and topic not in TOPICS:
            raise Http404
        return {'topic': topic, 'lang': get_language()}

    def title(self, obj):
        if obj['topic']:
            return '%s · SummarizePaper' % localized_name(obj['topic'], obj['lang'])[1]
        return '%s · SummarizePaper' % gettext('Weekly letters')

    def link(self, obj):
        return '/%s/letters/%s' % (obj['lang'], '%s/' % obj['topic'] if obj['topic'] else '')

    def description(self, obj):
        return gettext('Every week, what is new in research, told like a magazine article from the latest papers.')

    def items(self, obj):
        rows = Letter.objects.filter(lang=obj['lang']).defer('vectors')
        if obj['topic']:
            rows = rows.filter(topic=obj['topic'])
        return rows.order_by('-day', 'topic')[:FEED_ITEMS]

    def item_title(self, item):
        return item.title

    def item_description(self, item):
        context = letter_context(item)
        paragraphs = ['<p><em>%s</em></p>' % escape(item.dek)]
        for section in context['sections']:
            paragraphs.append('<h3>%s</h3>' % escape(section['heading']))
            paragraphs += ['<p>%s</p>' % escape(p) for p in section['text'].split('\n') if p.strip()]
            paragraphs += ['<p>→ <a href="https://www.summarizepaper.com/%s/paper/%s/">%s</a></p>' % (
                item.lang, link['arxiv_id'], escape(plain_math(link['title']))) for link in section['links']]
        return ''.join(paragraphs)

    def item_link(self, item):
        return '/%s/letters/%s/%s/' % (item.lang, item.topic, item.day.isoformat())

    def item_pubdate(self, item):
        return item.created


def escape(text):
    from django.utils.html import escape as html_escape
    return html_escape(text or '')


@csrf_exempt  # mail clients send the one-click unsubscribe POST without a CSRF token
def unsubscribe(request, token):
    from django.core import signing
    try:
        pk = signing.loads(token, salt=UNSUBSCRIBE_SALT)
    except signing.BadSignature:
        raise Http404
    subscription = LetterSubscription.objects.filter(pk=pk).first()
    done = False
    if subscription is not None and request.method == 'POST':
        subscription.active = False
        subscription.save(update_fields=['active'])
        done = True
    return render(request, 'summarizer/letter_unsubscribe.html', {'subscription': subscription, 'done': done,
                                                                  'name': TOPICS.get(getattr(subscription, 'topic', ''), {}).get('name', '')})


# --- links from the other tools --------------------------------------------------------------------------------

def letter_of_paper(arxiv_id, lang):
    """The latest letter that presents this paper, or None."""
    key = bare(arxiv_id.replace('/', '--'))
    if not key:
        return None
    return (Letter.objects.filter(lang=lang, papers__contains=',%s,' % key).order_by('-day')
            .only('topic', 'day', 'title', 'lang').first())


def latest_for_scope(scope, lang):
    """The latest letter of the theme of a rising-topics scope, or None (all of arXiv has no theme)."""
    topic = topic_of_scope(scope) if scope and scope != '*' else None
    if topic is None:
        return None
    return Letter.objects.filter(topic=topic, lang=lang).order_by('-day').only('topic', 'day', 'title', 'lang').first()


def for_digest(user, categories, lang, today):
    """For a Monday or weekly digest: up to two letters of the week from other fields than the digest's,
    the closest to the user's research profile first; without a profile, none."""
    own = {t for t in TOPICS if in_topic(t, categories)}
    profile = profile_of(user)
    if profile is None:
        return []
    found = []
    for letter in Letter.objects.filter(lang=lang, day__gt=today - WEEK).exclude(topic__in=own).defer('content'):
        vectors = [v for v in letter_vectors_raw(letter) if v.shape[0] == profile.shape[1]]
        if vectors:
            score = float(np.max(profile @ np.stack(vectors).T))
            if score >= digest.THRESHOLDS['low']:
                found.append((score, letter))
    found.sort(key=lambda f: -f[0])
    return [letter for _score, letter in found[:2]]


def letter_vectors_raw(letter):
    if not letter.vectors:
        return []
    papers = [p for p in letter.papers.split(',') if p]
    if not papers:
        return []
    return list(np.frombuffer(bytes(letter.vectors), dtype=np.float32).reshape(len(papers), -1))
