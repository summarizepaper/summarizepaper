"""The neighbourhood of a paper: the papers a researcher reading it should know, and why.

Three groups, every paper with its similarity to this one (the cosine of the title-and-abstract embeddings of
similar.py, so that all are on one scale):
- close papers: the closest of our whole index, every category together (the nearest paper of an astro-ph.IM
  paper may be in cs.LG), and those Semantic Scholar recommends from its own embeddings (all of computer science
  for a cs paper, recent papers otherwise), which reaches the many fields where few papers are summarized here;
- prior work: the papers it cites, the closest first: what it builds on rather than the software it uses;
- follow-ups: the papers that cite it, the closest first.
A paper both close and cited stays among the close papers, with both facts. Among the prior work and the follow-ups,
closeness is raised a little when Semantic Scholar says one paper uses the other in depth (isInfluential) or for its
method, and for much-cited follow-ups: Semantic Scholar gives the newest citations of a much-cited paper, so its most
cited ones come from OpenAlex. Each paper keeps its citation count and whether it is a preprint, so that a reader
can weigh it; a paper without an abstract is compared by its title alone, and says so.

The map groups the papers alike into a few sub-topics (clusters), named with the sentences. A reader can mark a
paper not relevant: it leaves their list, and everyone's once HIDDEN_FOR_ALL readers agree.

Computed when a paper is summarized, when a visitor asks to search again, and in the hourly run for the papers
summarized before this existed or whose neighbourhood is older than REFRESH_DAYS (the index grows, new papers
cite it). Never on a page view: crawlers read every page.

Stored as a Generated row (kind KIND, lang ''): the items and the similarities between them (edges, which the
map of the tree page draws). One sentence per shown item says what links it to the paper (kind WHY_KIND, one
row per language), written from the two abstracts the first time a visitor scrolls to the list.
"""
import datetime
import json
import logging
import time

import numpy as np
import requests
from django.http import Http404, JsonResponse
from django.utils import timezone
from django.utils.formats import number_format
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _
from django.utils.translation import pgettext_lazy
from django.views.decorators.http import require_POST

from . import arxiv_taxonomy, citations, generated, insight, similar, sources
from .models import ArxivPaper, Generated, NeighbourFeedback, PaperEmbedding, PaperScore, SummaryPaper

logger = logging.getLogger(__name__)

KIND = 'neighbours'
WHY_KIND = 'neighbours_why'
S2_GRAPH = 'https://api.semanticscholar.org/graph/v1/paper/%s/%s'
S2_RECOMMEND = 'https://api.semanticscholar.org/recommendations/v1/papers/forpaper/%s'
S2_BATCH = 'https://api.semanticscholar.org/graph/v1/paper/batch'
S2_FIELDS = 'title,abstract,year,publicationDate,authors,externalIds,venue,journal,citationCount'
S2_LINK_FIELDS = 'isInfluential,intents,' + S2_FIELDS  # how the citing paper uses the cited one
RECOMMENDED = 40
LINKED = 300  # references, and citing papers, read for each paper
MOST_CITED = 25  # citing papers of a much-cited paper read from OpenAlex, the most cited first
OPENALEX_PER_DAY = 600
OPENALEX_KEY = 'openalex-neighbours:%s'
INDEX_K = 60
MAX_EMBED = 450
# prior work and follow-ups: closeness first, then how the papers use each other and how much a follow-up is cited
INFLUENTIAL_BONUS = 0.05  # Semantic Scholar: the citation weighs on the citing paper (not a passing mention)
METHOD_BONUS = 0.03  # a reference whose method the paper uses
CITED_BONUS = 0.08  # a follow-up cited 10 000 times (log scale)
CLUSTER_MINS = (0.70, 0.74, 0.78, 0.82)  # average similarity within a sub-topic of the map: the lowest that...
CLUSTER_SHARE = 0.4  # ...leaves no sub-topic with more than this share of the papers (a dense field is one blob at 0.70)
CLUSTER_SIZE, MAX_CLUSTERS = 3, 5
PREPRINT_SERVERS = {'arxiv', 'arxiv.org', 'biorxiv', 'medrxiv', 'chemrxiv', 'ssrn', 'research square', 'preprints',
                    'preprints.org', 'osf preprints', 'psyarxiv', 'socarxiv', 'techrxiv', 'authorea'}
CLOSE_KEPT, PRIOR_KEPT, FOLLOWUP_KEPT = 15, 8, 8
CLOSE_SHOWN, PRIOR_SHOWN, FOLLOWUP_SHOWN = 8, 5, 5
EDGE_MIN = 0.65  # two neighbours closer than this are linked on the map...
EDGES_PER_ITEM = 3  # ...each to its few closest
ABSTRACT_KEPT = 700
REFRESH_DAYS = 60
RUN_SECONDS = 900  # the hourly run's share: 15 to 25 seconds a paper, Semantic Scholar's 429s included, so 40 papers
WHY_PER_DAY = 600
WHY_KEY = 'neighbours-why:%s'

# Levels of the cosine similarity of text-embedding-3-small, from the production index (October 2026): the best
# neighbour of half the papers is above 0.67, of a quarter above 0.73; half of all listed neighbours are above 0.59.
VERY_CLOSE = 0.72
CLOSE = 0.60
LEVELS = [('very_close', VERY_CLOSE, pgettext_lazy('closeness of two papers', 'Very close')),
          ('close', CLOSE, pgettext_lazy('closeness of two papers', 'Close')),
          ('related', -1, pgettext_lazy('closeness of two papers', 'Related'))]

GROUPS = {
    'close': (_('Closest papers'), CLOSE_SHOWN),
    'prior': (_('Prior work it builds on'), PRIOR_SHOWN),
    'followup': (_('Papers that cite it'), FOLLOWUP_SHOWN),
}
RELATIONS = {
    'same_question': _('Same question'),
    'other_approach': _('Another approach to the problem'),
    'same_method': _('Same method, another problem'),
    'groundwork': _('Groundwork for this paper'),
    'builds_on_it': _('Builds on this paper'),
    'different_results': _('Different results'),
    'review': pgettext_lazy('a kind of paper', 'Review'),
    'resource': _('Shared data, tool or method'),
    'context': _('Broader context'),
}


def level(similarity):
    return next((code, label) for code, floor, label in LEVELS if similarity >= floor)


def identity(ref):
    from .library import identity as library_identity
    return library_identity(ref)


# --- gathering -------------------------------------------------------------------------

def short_authors(names):
    names = [n for n in names if n]
    return ', '.join(names[:3]) + (' et al.' if len(names) > 3 else '')


def venue_of(paper):
    if paper.journal_ref:
        return ' '.join(paper.journal_ref.split())[:120]
    from .templatetags.dash_slash import category_label
    return str(arxiv_taxonomy.NAMES.get(paper.category or '') or category_label(paper.category or ''))


def index_item(paper, similarity):
    return {'ref': paper.arxiv_id, 'title': ' '.join((paper.title or '').split()),
            'authors': short_authors([a.name for a in paper.authors.all()[:4]]),
            'date': paper.published_arxiv.isoformat()[:10] if paper.published_arxiv else '',
            'venue': venue_of(paper), 'abstract': ' '.join((paper.abstract or '').split())[:ABSTRACT_KEPT],
            'similarity': round(similarity, 3), 'citations': None, 'status': 'published' if paper.journal_ref else '',
            'influential': False, 'method': False, 'cited': False, 'cites': False, 'source': 'index'}


def s2_ref(paper_id):
    """How Semantic Scholar names a paper, None for the sources it cannot be asked about (HAL, PMC keys)."""
    if sources.is_external(paper_id) and not (paper_id.startswith('doi_') and sources.key_doi(paper_id)):
        return None
    return citations.s2_ref(citations.base_id(paper_id) or paper_id)


_last_s2 = [0.0]


def s2_request(method, url, **kwargs):
    """A Semantic Scholar call, SECONDS_BETWEEN_CALLS after the previous one of this process (the key allows one a
    second, shared with the citation alerts and the chat), tried again after a 429; None when unreachable."""
    for wait in (1.5, 4, 10, None):  # a 429 usually clears within a second
        pause = _last_s2[0] + citations.SECONDS_BETWEEN_CALLS - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_s2[0] = time.monotonic()
        try:
            response = requests.request(method, url, headers=citations.s2_headers(), timeout=30, **kwargs)
        except requests.RequestException:
            logger.warning('Semantic Scholar unreachable: %s', url)
            return None
        if response.status_code != 429 or wait is None:
            return response
        time.sleep(wait)


def s2_get(url, params):
    """The JSON answer of Semantic Scholar, None when it does not know the paper or does not answer."""
    response = s2_request('GET', url, params=params)
    if response is None:
        return None
    if response.status_code != 200:
        if response.status_code != 404:
            logger.warning('Semantic Scholar answered %s: %s', response.status_code, url)
        return None
    try:
        return response.json()
    except ValueError:
        return None


def status(venue, journal=''):
    """'published', 'preprint' or '' (unknown) from the venue names a source gives."""
    names = [n.strip().lower() for n in (venue, journal) if n and n.strip()]
    if not names:
        return ''
    if all(n in PREPRINT_SERVERS or n.endswith(' preprint') or 'rxiv' in n for n in names):
        return 'preprint'
    return 'published'


def s2_item(paper, row=None):
    """The item of a paper as Semantic Scholar gives it, None without a title or an identifier we can link.
    row: the reference or citation it comes from, which says how one paper uses the other."""
    ids = paper.get('externalIds') or {}
    title = ' '.join((paper.get('title') or '').split())
    if ids.get('ArXiv'):
        from .digest import site_id
        ref = site_id(ids['ArXiv'])
    elif ids.get('DOI'):
        ref = sources.doi_key(ids['DOI'])
    else:
        return None
    if not title or ref.startswith('doi_h'):  # a DOI too odd for our URLs: its page could not find it
        return None
    journal = (paper.get('journal') or {}).get('name') or ''
    year = paper.get('year')
    row = row or {}
    return {'ref': ref, 'title': title[:300], 'authors': short_authors([a.get('name', '') for a in paper.get('authors') or []]),
            'date': paper.get('publicationDate') or (str(year) if year else ''),
            'venue': (paper.get('venue') or journal or '')[:120],
            'abstract': ' '.join((paper.get('abstract') or '').split())[:ABSTRACT_KEPT],
            'citations': paper.get('citationCount'), 'status': status(paper.get('venue') or '', journal),
            'influential': bool(row.get('isInfluential')), 'method': 'methodology' in (row.get('intents') or []),
            'cited': False, 'cites': False, 'source': 's2'}


def s2_candidates(paper):
    """{'recommended', 'cited', 'cites': [items], 'more_citing': bool} from Semantic Scholar; a list is empty when
    its call fails. more_citing: the paper has more citations than one page holds (they come newest first)."""
    found = {'recommended': [], 'cited': [], 'cites': [], 'more_citing': False}
    ref = s2_ref(paper.arxiv_id)
    if not ref:
        return found
    pool = 'all-cs' if (paper.category or '').startswith('cs.') else 'recent'
    body = s2_get(S2_RECOMMEND % ref, {'fields': S2_FIELDS, 'limit': RECOMMENDED, 'from': pool}) or {}
    found['recommended'] = [i for i in map(s2_item, body.get('recommendedPapers') or []) if i]
    for kind, path, field in (('cited', 'references', 'citedPaper'), ('cites', 'citations', 'citingPaper')):
        body = s2_get(S2_GRAPH % (ref, path), {'fields': S2_LINK_FIELDS, 'limit': LINKED}) or {}
        rows = body.get('data') or []
        found[kind] = [i for i in (s2_item(row.get(field) or {}, row) for row in rows) if i]
        if kind == 'cites':
            found['more_citing'] = len(rows) >= LINKED
    return found


def s2_facts(items):
    """Citation counts and venues, in one call, for the items of our index Semantic Scholar has not described."""
    wanted = [i for i in items if i.get('citations') is None]
    ids = [s2_ref(i['ref']) for i in wanted]
    pairs = [(i, r) for i, r in zip(wanted, ids) if r]
    if not pairs:
        return
    response = s2_request('POST', S2_BATCH, params={'fields': 'citationCount,venue,journal'}, json={'ids': [r for _, r in pairs]})
    try:
        rows = response.json() if response is not None and response.status_code == 200 else []
    except ValueError:
        rows = []
    for (item, _ref), row in zip(pairs, rows if isinstance(rows, list) else []):
        if row:
            item['citations'] = row.get('citationCount')
            item['status'] = item.get('status') or status(row.get('venue') or '', (row.get('journal') or {}).get('name') or '')


def openalex_most_cited(paper):
    """The citing papers of a much-cited paper, the most cited first (Semantic Scholar gives the newest): the
    follow-ups everyone knows. Two OpenAlex calls, within OPENALEX_PER_DAY; [] when out of budget or unavailable."""
    if sources.is_external(paper.arxiv_id):
        doi = sources.key_doi(paper.arxiv_id) if paper.arxiv_id.startswith('doi_') else ''
    else:
        doi = '10.48550/arxiv.%s' % (citations.base_id(paper.arxiv_id) or '')
    if not doi or not sources._spend(OPENALEX_KEY, OPENALEX_PER_DAY):
        return []
    try:
        work = sources._json('%s/works/doi:%s' % (sources.OPENALEX, doi), sources._openalex_params({'select': 'id'}))
        if not work:
            return []
        body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
            'filter': 'cites:%s' % work['id'].rsplit('/', 1)[-1], 'sort': 'cited_by_count:desc', 'per-page': MOST_CITED,
            'select': sources.OPENALEX_FIELDS + ',cited_by_count'})) or {}
    except sources.SourceUnavailable:
        return []
    found = []
    from .digest import site_id
    for work in body.get('results') or []:
        meta = sources.openalex_paper(work)
        ref = site_id(meta['arxiv_id']) if meta.get('arxiv_id') else meta.get('key') or ''
        if not ref or ref.startswith('doi_h') or not meta.get('title'):
            continue
        found.append({'ref': ref, 'title': ' '.join(meta['title'].split())[:300], 'authors': short_authors(meta.get('authors') or []),
                      'date': meta.get('published') or '', 'venue': (meta.get('journal') or '')[:120],
                      'abstract': ' '.join((meta.get('abstract') or '').split())[:ABSTRACT_KEPT],
                      'citations': work.get('cited_by_count'),
                      'status': 'preprint' if work.get('type') == 'preprint' else status(meta.get('journal') or ''),
                      'influential': False, 'method': False, 'cited': False, 'cites': True, 'source': 'openalex'})
    return found


def known_versions(items):
    """{identity: the arXiv id or key we store} for the items whose paper is in our tables."""
    wanted = {}
    for item in items:
        if sources.is_external(item['ref']):
            wanted[item['ref']] = item['ref']
        else:
            for v in range(1, 10):
                wanted['%sv%d' % (item['ref'], v)] = item['ref']
            wanted[item['ref']] = item['ref']
    stored = {}
    for aid in ArxivPaper.objects.filter(arxiv_id__in=list(wanted)).values_list('arxiv_id', flat=True):
        key = identity(aid)
        if key and aid > stored.get(key, ''):
            stored[key] = aid  # the latest version
    return stored


def weight(item, group):
    """How a candidate ranks among the prior work or the follow-ups: closeness, raised by how the papers use each
    other and, for a follow-up, by how much it is cited."""
    value = item['similarity'] + (INFLUENTIAL_BONUS if item.get('influential') else 0)
    if group == 'prior' and item.get('method'):
        value += METHOD_BONUS
    if group == 'followup' and item.get('citations'):
        value += CITED_BONUS * min(1.0, np.log10(1 + item['citations']) / 4)
    return value


def merge(target, item, kind):
    """The facts a second source gives on a paper already in the pool."""
    if kind in ('cited', 'cites'):
        target[kind] = True
    target['influential'] = target.get('influential') or item.get('influential', False)
    target['method'] = target.get('method') or item.get('method', False)
    if target.get('citations') is None and item.get('citations') is not None:
        target['citations'] = item['citations']
    if not target.get('status'):
        target['status'] = item.get('status', '')


def compute(paper):
    """The neighbourhood of a paper, as stored: {'items', 'edges', 'clusters', 'computed', 'counts'}."""
    from .library import work_keys
    similar.ensure(paper)
    query = np.frombuffer(PaperEmbedding.objects.get(arxiv_id=paper.arxiv_id).vector, dtype=np.float32)
    own = work_keys([paper.arxiv_id])[paper.arxiv_id] | {identity(paper.arxiv_id)}
    pool, vectors = {}, {}

    best = similar.scan(query, INDEX_K, skip=lambda aid: identity(aid) in own)
    rows = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=[a for _, a in best]).prefetch_related('authors')}
    for value, aid in best:
        key = identity(aid)
        if key and key not in pool and aid in rows and rows[aid].title:
            pool[key] = index_item(rows[aid], value)

    found = s2_candidates(paper)
    found['most_cited'] = openalex_most_cited(paper) if found['more_citing'] else []
    pending = {}
    # the most cited follow-ups before the others: past MAX_EMBED, the last ones are left out
    for kind in ('recommended', 'cited', 'most_cited', 'cites'):
        for item in found[kind]:
            key = identity(item['ref'])
            if not key or key in own:
                continue
            target = pool.get(key) or pending.get(key)
            if target is None:
                pending[key] = target = item
            merge(target, item, 'cites' if kind == 'most_cited' else kind)
    pending = dict(list(pending.items())[:MAX_EMBED])
    if pending:
        texts = ['%s\n\n%s' % (i['title'], i['abstract']) for i in pending.values()]
        embedded = np.concatenate([similar.embed(texts[start:start + similar.BATCH])
                                   for start in range(0, len(texts), similar.BATCH)])
        for (key, item), vector in zip(pending.items(), embedded):
            item['similarity'] = round(float(vector @ query), 3)
            vectors[key] = vector
            pool[key] = item
    stored = known_versions([i for i in pool.values() if i['source'] != 'index'])
    for key, item in pool.items():
        if item['source'] != 'index':
            item['ref'] = stored.get(key) or (item['ref'] if sources.is_external(item['ref']) else item['ref'] + 'v1')
        item['title_only'] = not item['abstract']  # its closeness comes from its title alone

    chosen = []
    for group, flag, kept in (('close', None, CLOSE_KEPT), ('prior', 'cited', PRIOR_KEPT), ('followup', 'cites', FOLLOWUP_KEPT)):
        taken = {k for k, _ in chosen}
        ranked = sorted(pool.items(), key=lambda kv: -(kv[1]['similarity'] if flag is None else weight(kv[1], group)))
        picked = [(k, i) for k, i in ranked if k not in taken and (flag is None or i[flag])][:kept]
        for _key, item in picked:
            item['group'] = group
        chosen += picked
    s2_facts([i for _k, i in chosen if i['source'] == 'index'])

    missing = [i['ref'] for k, i in chosen if k not in vectors]
    for aid, vec in PaperEmbedding.objects.filter(arxiv_id__in=missing, model=similar.MODEL).values_list('arxiv_id', 'vector'):
        vectors[identity(aid)] = np.frombuffer(vec, dtype=np.float32)
    items = [i for _k, i in chosen]
    chosen_vectors = [vectors.get(k) for k, _i in chosen]
    return {'items': items, 'edges': edges(chosen_vectors), 'clusters': clusters(chosen_vectors),
            'computed': timezone.now().isoformat(),
            'counts': {kind: len(found[kind]) for kind in ('recommended', 'cited', 'cites', 'most_cited')} | {'index': len(best)}}


def clusters(vectors):
    """[[item indexes]] of the sub-topics of the map: average-linkage groups whose papers are, on average, alike
    above the lowest of CLUSTER_MINS that leaves no group with more than CLUSTER_SHARE of the papers; the
    MAX_CLUSTERS largest of CLUSTER_SIZE papers or more."""
    present = [i for i, v in enumerate(vectors) if v is not None]
    if len(present) < CLUSTER_SIZE:
        return []
    sims = np.stack([vectors[i] for i in present]) @ np.stack([vectors[i] for i in present]).T
    groups = [[n] for n in range(len(present))]
    largest = max(CLUSTER_SIZE * 2, int(CLUSTER_SHARE * len(present)))
    for floor in sorted(CLUSTER_MINS, reverse=True):  # merging from the strictest level down, stopping in time
        merged = [list(g) for g in groups]
        while len(merged) > 1:
            best, pair = -1.0, None
            for a in range(len(merged)):
                for b in range(a + 1, len(merged)):
                    value = float(sims[np.ix_(merged[a], merged[b])].mean())
                    if value > best:
                        best, pair = value, (a, b)
            if best < floor:
                break
            a, b = pair
            merged[a] += merged.pop(b)
        if max(len(g) for g in merged) > largest:
            break
        groups = merged
    kept = sorted((g for g in groups if len(g) >= CLUSTER_SIZE), key=len, reverse=True)[:MAX_CLUSTERS]
    return [sorted(present[n] for n in g) for g in kept]


def edges(vectors):
    """[[i, j, similarity]] linking each item to its EDGES_PER_ITEM closest items above EDGE_MIN."""
    present = [i for i, v in enumerate(vectors) if v is not None]
    if len(present) < 2:
        return []
    matrix = np.stack([vectors[i] for i in present])
    sims = matrix @ matrix.T
    found = set()
    for a in range(len(present)):
        sims[a, a] = -1
        for b in np.argsort(-sims[a])[:EDGES_PER_ITEM]:
            if sims[a, b] >= EDGE_MIN:
                found.add((min(present[a], present[b]), max(present[a], present[b]), round(float(sims[a, b]), 3)))
    return [list(e) for e in sorted(found)]


def store(arxiv_id):
    """Compute and keep the neighbourhood of a paper; None when the paper is unknown."""
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None or not (paper.title or paper.abstract):
        return None
    data = compute(paper)
    Generated.objects.update_or_create(kind=KIND, key=arxiv_id, lang='', defaults={
        'status': 'done', 'content': json.dumps(data), 'error': ''})
    return data


def refresh(arxiv_id, lang=None):
    """A visitor asked to search again and waits: the index neighbours (PaperScore, for the press kit), the
    neighbourhood, then the sentences of the visitor's language in the background. Never fatal."""
    try:
        similar.find_and_store(arxiv_id)
    except Exception:
        logger.exception('Closest papers of %s failed', arxiv_id)
    complete(arxiv_id, lang)


def complete(arxiv_id, lang=None):
    try:
        store(arxiv_id)
    except Exception:
        logger.exception('Neighbourhood of %s failed', arxiv_id)
        return
    if lang:
        start_why(arxiv_id, lang)


def after_synthesis(arxiv_id, lang):
    """At the end of a synthesis the visitor waits for the page: the index neighbours now (a second), the rest
    (Semantic Scholar, ten seconds or more) in the background, shown at the next visit."""
    try:
        similar.find_and_store(arxiv_id)
    except Exception:
        logger.exception('Closest papers of %s failed', arxiv_id)
    generated.background(complete, arxiv_id, lang)


def stored(arxiv_id):
    row = Generated.objects.filter(kind=KIND, key=arxiv_id, lang='', status='done').first()
    data = row.data() if row else None
    return data if isinstance(data, dict) and data.get('items') is not None else None


def legacy(paper):
    """The index neighbours computed before neighbourhoods existed (PaperScore), as close items."""
    scores = (PaperScore.objects.filter(from_paper=paper, active=True).order_by('score')
              .select_related('to_paper').prefetch_related('to_paper__authors')[:CLOSE_KEPT])
    items = [index_item(s.to_paper, 1 - s.score) for s in scores if s.to_paper.title]
    for item in items:
        item['group'] = 'close'
    return items


def refresh_due(now, seconds=RUN_SECONDS):
    """The hourly run: the summarized papers without a neighbourhood, the newest first, then the oldest
    neighbourhoods, for at most `seconds`. Returns how many were computed."""
    started, done = time.monotonic(), 0
    have = Generated.objects.filter(kind=KIND, lang='').values('key')
    missing = (SummaryPaper.objects.exclude(paper__arxiv_id__in=have).order_by('-created')
               .values_list('paper__arxiv_id', flat=True))
    stale = (Generated.objects.filter(kind=KIND, lang='', updated__lt=now - datetime.timedelta(days=REFRESH_DAYS))
             .order_by('updated').values_list('key', flat=True))
    seen = set()
    for queue in (missing, stale):
        for arxiv_id in queue.iterator():
            if time.monotonic() - started > seconds:
                return done
            if arxiv_id in seen:
                continue
            seen.add(arxiv_id)
            try:
                store(arxiv_id)
                done += 1
            except Exception:
                logger.exception('Neighbourhood of %s failed', arxiv_id)
                Generated.objects.filter(kind=KIND, key=arxiv_id, lang='').update(updated=timezone.now())
    return done


# --- why each paper is there -----------------------------------------------------------

WHY_SYSTEM = """You help a researcher see how research papers relate to the paper they are reading.
For each numbered neighbour, give:
- "relation": the one code that fits best:
  same_question (studies the same question or object), other_approach (another method for the same problem),
  same_method (the same method or model applied to another problem), groundwork (earlier work this paper
  relies on), builds_on_it (later work that extends or uses this paper), different_results (its results differ
  from or contradict this paper's: only when both texts state results that conflict), review (a review or
  survey of the field), resource (shares a dataset, code, instrument or generic method), context (broader or
  neighbouring topic);
- "why": one sentence of at most 30 words, in {language}, that says concretely what they share or how they
  differ (object studied, question, method, data, result), using only the two texts. Start with what the
  neighbour does (its name if it has one), never with "Both papers" or "Les deux études"; no number that is not
  in the texts; do not praise.
"link" says when the neighbour is cited by the paper or cites it.
Then name each numbered group of titles: the sub-topic its papers share, in 2 to 5 words, in {language}, specific
enough to tell the groups apart (never just the field of the paper).
Answer with a JSON object {{"neighbours": [{{"n": 1, "relation": "...", "why": "..."}}],
"groups": [{{"g": 1, "name": "..."}}]}}."""


def shown(data):
    """The items the paper page shows, in its order: close, prior, follow-ups."""
    return [i for group, (_title, count) in GROUPS.items() for i in [x for x in data['items'] if x.get('group') == group][:count]]


WHY_ROUNDS = 3  # the model sometimes answers for a few neighbours only: the others are asked again


def why_work(arxiv_id, lang):
    def work(pk):
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        data = stored(arxiv_id) or {'items': legacy(paper)}
        items = shown(data)
        if not items:
            return {'refs': [], 'by_ref': {}, 'clusters': []}
        groups = [[data['items'][n] for n in members if n < len(data['items'])] for members in data.get('clusters') or []]
        reading = insight.for_page(arxiv_id, lang) or insight.for_page(arxiv_id, 'en') or {}
        about = {'title': paper.title, 'question': reading.get('question', ''), 'method': reading.get('method', ''),
                 'abstract': ' '.join((paper.abstract or '').split())[:2000]}
        generated.release_connection()
        by_ref, names = {}, {}
        for _round in range(WHY_ROUNDS):
            asked = [(n, i) for n, i in enumerate(items, 1) if i['ref'] not in by_ref]
            unnamed = [(g, members) for g, members in enumerate(groups, 1) if g not in names]
            if not asked and not unnamed:
                break
            context = {'paper': about,
                       'neighbours': [{'n': n, 'title': i['title'], 'abstract': i['abstract'] or '(no abstract)',
                                       'link': 'cited by the paper' if i.get('cited') else 'cites the paper' if i.get('cites') else ''}
                                      for n, i in asked],
                       'groups': [{'g': g, 'titles': [i['title'] for i in members]} for g, members in unnamed]}
            answer = generated.chat_json(json.dumps(context, ensure_ascii=False), paid=False, max_tokens=2800, timeout=120,
                                         system=WHY_SYSTEM.format(language=generated.language_name(lang)))
            wanted = dict(asked)
            for row in answer.get('neighbours') or []:
                if not isinstance(row, dict) or not str(row.get('n', '')).isdigit() or int(row['n']) not in wanted:
                    continue
                why = insight.clean(row.get('why'), 300)
                if why:
                    relation = row.get('relation') if row.get('relation') in RELATIONS else ''
                    by_ref[wanted[int(row['n'])]['ref']] = {'relation': relation, 'why': why}
            for row in answer.get('groups') or []:
                if isinstance(row, dict) and str(row.get('g', '')).isdigit() and 1 <= int(row['g']) <= len(groups):
                    name = insight.clean(row.get('name'), 60)
                    if name:
                        names[int(row['g'])] = name
        return {'refs': [i['ref'] for i in items], 'by_ref': by_ref,
                'clusters': [{'refs': [i['ref'] for i in members], 'name': names[g]}
                             for g, members in enumerate(groups, 1) if names.get(g)]}
    return work


def why_row(arxiv_id, lang):
    return Generated.objects.filter(kind=WHY_KIND, key=arxiv_id, lang=lang).first()


WHY_COVERED = 0.8  # share of the shown papers that must have their sentence, or the sentences are written again...
WHY_RETRY = datetime.timedelta(days=1)  # ...once a day at most


def covers(row, items):
    if row is None or row.status != 'done':
        return False
    data = row.data()
    if not set(i['ref'] for i in items) <= set(data.get('refs') or []):
        return False
    written = sum(1 for i in items if i['ref'] in (data.get('by_ref') or {}))
    return written >= WHY_COVERED * len(items) or row.updated > timezone.now() - WHY_RETRY


def start_why(arxiv_id, lang):
    """Start writing the sentences when the shown items have none yet; the Generated row, or None past the budget."""
    data = stored(arxiv_id)
    row = why_row(arxiv_id, lang)
    if data is not None and covers(row, shown(data)):
        return row
    if not sources._spend(WHY_KEY, WHY_PER_DAY):
        return None
    return generated.start(WHY_KIND, arxiv_id, lang, why_work(arxiv_id, lang), force=True)


@require_POST
def why_view(request, arxiv_id):
    """The paper page asks for the sentences once its list is in view: {'status', 'id'} to follow, or the data."""
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None:
        raise Http404
    lang = get_language()
    row = why_row(arxiv_id, lang)
    items = shown(stored(arxiv_id) or {'items': legacy(paper)})
    if covers(row, items):
        return JsonResponse({'status': 'done', 'data': row.data()})
    if row is not None and row.status == 'pending':
        return JsonResponse({'status': 'pending', 'id': row.pk})
    if row is not None and row.status == 'failed' and row.updated > timezone.now() - datetime.timedelta(hours=1):
        return JsonResponse({'status': 'busy'})
    row = start_why(arxiv_id, lang)
    if row is None:
        return JsonResponse({'status': 'busy'})
    return JsonResponse({'status': row.status, 'id': row.pk, 'data': row.data() if row.status == 'done' else None})


# --- readers' judgments ------------------------------------------------------------------

HIDDEN_FOR_ALL = 3  # distinct readers who must mark a paper not relevant before it leaves everyone's list
MARKS_PER_DAY = 50  # by a visitor without an account
MARKS_KEY = 'neighbour-marks:%s:%%s'


def reader(request):
    """(user, visitor): who marks, a visitor without an account by their hashed address."""
    from . import access
    if request.user.is_authenticated:
        return request.user, ''
    return None, access.hash_ip(access.client_ip(request))


def hidden_refs(arxiv_id, request=None):
    """The papers left out of this paper's list: those this reader marked, and those HIDDEN_FOR_ALL readers marked."""
    user, visitor = reader(request) if request is not None else (None, '')
    people, mine = {}, set()
    for ref, user_id, who in NeighbourFeedback.objects.filter(arxiv_id=arxiv_id).values_list('ref', 'user_id', 'visitor'):
        people.setdefault(ref, set()).add(user_id or who)
        if (user is not None and user_id == user.pk) or (visitor and who == visitor):
            mine.add(ref)
    return mine | {ref for ref, who in people.items() if len(who) >= HIDDEN_FOR_ALL}


@require_POST
def mark_view(request, arxiv_id):
    """A reader marks a paper of the list as not relevant (or takes it back with undo=1)."""
    ref = request.POST.get('ref', '')[:100]
    if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists() or not ref:
        raise Http404
    user, visitor = reader(request)
    mine = NeighbourFeedback.objects.filter(arxiv_id=arxiv_id, ref=ref, user=user, visitor=visitor)
    if request.POST.get('undo') == '1':
        mine.delete()
        return JsonResponse({'status': 'shown'})
    if not mine.exists():
        if visitor and not sources._spend(MARKS_KEY % visitor, MARKS_PER_DAY):
            return JsonResponse({'status': 'busy'}, status=429)
        NeighbourFeedback.objects.create(arxiv_id=arxiv_id, ref=ref, user=user, visitor=visitor)
    return JsonResponse({'status': 'hidden'})


# --- what the pages show ---------------------------------------------------------------

def year(item):
    return (item.get('date') or '')[:4]


def for_page(paper, lang, with_map=False, request=None):
    """{'groups': [{'key', 'title', 'items'}], 'computed', 'why_ready', 'complete', 'relations_json'} for the paper
    and tree pages, with 'map' (see map_data) when asked; None when the paper has no neighbour yet.
    complete: Semantic Scholar's neighbours are there too. The papers this reader, or enough readers, marked
    not relevant are left out (the next ones take their place)."""
    data = stored(paper.arxiv_id)
    items = data['items'] if data else legacy(paper)
    if not items:
        return None
    hidden = hidden_refs(paper.arxiv_id, request)
    row = why_row(paper.arxiv_id, lang)
    why_data = row.data() if row is not None and row.status == 'done' else {}
    why = why_data.get('by_ref', {})
    refs = [i['ref'] for i in items]
    summarized = set(SummaryPaper.objects.filter(paper__arxiv_id__in=refs).values_list('paper__arxiv_id', flat=True))
    tldrs = {}
    for g in Generated.objects.filter(kind=insight.KIND, key__in=refs, lang=lang, status='done'):
        tldrs[g.key] = (g.data() or {}).get('tldr', '')
    groups = []
    for key, (title, count) in GROUPS.items():
        members = []
        for item in [i for i in items if i.get('group') == key and i['ref'] not in hidden][:count]:
            code, label = level(item['similarity'])
            reason = why.get(item['ref']) or {}
            members.append(dict(item, level=code, level_label=label, percent=round(100 * item['similarity']),
                                year=year(item), summarized=item['ref'] in summarized, tldr=tldrs.get(item['ref'], ''),
                                citations_shown=number_format(item['citations'], force_grouping=True) if item.get('citations') else '',
                                relation=reason.get('relation', ''), relation_label=RELATIONS.get(reason.get('relation'), ''),
                                why=reason.get('why', '')))
        if members:
            groups.append({'key': key, 'title': title, 'items': members})
    computed = datetime.datetime.fromisoformat(data['computed']) if data and data.get('computed') else None
    result = {'groups': groups, 'computed': computed, 'why_ready': bool(why), 'complete': data is not None,
              'relations_json': json.dumps({code: str(label) for code, label in RELATIONS.items()}, ensure_ascii=False)}
    if with_map:
        result['map'] = map_data(paper, items, data or {}, why_data, summarized, hidden)
    return result


def map_data(paper, items, data, why_data, summarized, hidden=()):
    """JSON for the map of the tree page: the paper, every stored neighbour (hidden: marked not relevant), the
    edges between them and their named sub-topics."""
    why = why_data.get('by_ref', {})
    nodes = []
    for item in items:
        code, label = level(item['similarity'])
        reason = why.get(item['ref']) or {}
        nodes.append({k: item.get(k, '') for k in ('ref', 'title', 'authors', 'date', 'venue', 'abstract', 'group', 'status')}
                     | {'similarity': item['similarity'], 'cited': item.get('cited', False), 'cites': item.get('cites', False),
                        'citations': item.get('citations'), 'influential': item.get('influential', False),
                        'level': code, 'level_label': str(label), 'summarized': item['ref'] in summarized,
                        'hidden': item['ref'] in hidden,
                        'relation_label': str(RELATIONS.get(reason.get('relation'), '')), 'why': reason.get('why', '')})
    names = {tuple(sorted(c['refs'])): c['name'] for c in why_data.get('clusters') or []}
    topics = []
    for members in data.get('clusters') or []:
        name = names.get(tuple(sorted(items[n]['ref'] for n in members if n < len(items))))
        if name:
            topics.append({'members': members, 'name': name})
    main = {'ref': paper.arxiv_id, 'title': paper.title or '',
            'date': paper.published_arxiv.isoformat()[:10] if paper.published_arxiv else ''}
    return {'main': main, 'nodes': nodes, 'edges': data.get('edges') or [], 'topics': topics}
