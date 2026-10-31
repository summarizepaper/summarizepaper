"""The neighbourhood of a paper: the papers a researcher reading it should know, and why.

Three groups, every paper with its similarity to this one (the cosine of the title-and-abstract embeddings of
similar.py, so that all are on one scale):
- close papers: the closest of our whole index, every category together (the nearest paper of an astro-ph.IM
  paper may be in cs.LG), and those Semantic Scholar recommends from its own embeddings (all of computer science
  for a cs paper, recent papers otherwise), which reaches the many fields where few papers are summarized here;
- prior work: the papers it cites, the closest first: what it builds on rather than the software it uses;
- follow-ups: the papers that cite it, the closest first.
A paper both close and cited stays among the close papers, with both facts.

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
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _
from django.utils.translation import pgettext_lazy
from django.views.decorators.http import require_POST

from . import arxiv_taxonomy, citations, generated, insight, similar, sources
from .models import ArxivPaper, Generated, PaperEmbedding, PaperScore, SummaryPaper

logger = logging.getLogger(__name__)

KIND = 'neighbours'
WHY_KIND = 'neighbours_why'
S2_GRAPH = 'https://api.semanticscholar.org/graph/v1/paper/%s/%s'
S2_RECOMMEND = 'https://api.semanticscholar.org/recommendations/v1/papers/forpaper/%s'
S2_FIELDS = 'title,abstract,year,publicationDate,authors,externalIds,venue,journal'
RECOMMENDED = 40
LINKED = 300  # references, and citing papers, read for each paper
INDEX_K = 60
MAX_EMBED = 400
CLOSE_KEPT, PRIOR_KEPT, FOLLOWUP_KEPT = 15, 8, 8
CLOSE_SHOWN, PRIOR_SHOWN, FOLLOWUP_SHOWN = 8, 5, 5
EDGE_MIN = 0.65  # two neighbours closer than this are linked on the map...
EDGES_PER_ITEM = 3  # ...each to its few closest
ABSTRACT_KEPT = 700
REFRESH_DAYS = 60
RUN_SECONDS = 600  # the hourly run's share: about 12 seconds a paper, so 50 papers
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
            'similarity': round(similarity, 3), 'cited': False, 'cites': False, 'source': 'index'}


def s2_ref(paper_id):
    """How Semantic Scholar names a paper, None for the sources it cannot be asked about (HAL, PMC keys)."""
    if sources.is_external(paper_id) and not (paper_id.startswith('doi_') and sources.key_doi(paper_id)):
        return None
    return citations.s2_ref(citations.base_id(paper_id) or paper_id)


def s2_get(url, params):
    """The JSON answer of Semantic Scholar, None when it does not know the paper or does not answer."""
    for wait in (2, 6, None):  # one call a second with a key, shared with the citation alerts
        try:
            response = requests.get(url, params=params, headers=citations.s2_headers(), timeout=30)
        except requests.RequestException:
            logger.warning('Semantic Scholar unreachable: %s', url)
            return None
        if response.status_code != 429 or wait is None:
            break
        time.sleep(wait)
    if response.status_code != 200:
        if response.status_code != 404:
            logger.warning('Semantic Scholar answered %s: %s', response.status_code, url)
        return None
    try:
        return response.json()
    except ValueError:
        return None


def s2_item(paper):
    """The item of a paper as Semantic Scholar gives it, None without a title or an identifier we can link."""
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
    return {'ref': ref, 'title': title[:300], 'authors': short_authors([a.get('name', '') for a in paper.get('authors') or []]),
            'date': paper.get('publicationDate') or (str(year) if year else ''),
            'venue': (paper.get('venue') or journal or '')[:120],
            'abstract': ' '.join((paper.get('abstract') or '').split())[:ABSTRACT_KEPT],
            'cited': False, 'cites': False, 'source': 's2'}


def s2_candidates(paper):
    """{'recommended', 'cited', 'cites': [items]} from Semantic Scholar; a list is empty when its call fails."""
    found = {'recommended': [], 'cited': [], 'cites': []}
    ref = s2_ref(paper.arxiv_id)
    if not ref:
        return found
    pool = 'all-cs' if (paper.category or '').startswith('cs.') else 'recent'
    body = s2_get(S2_RECOMMEND % ref, {'fields': S2_FIELDS, 'limit': RECOMMENDED, 'from': pool}) or {}
    found['recommended'] = [i for i in map(s2_item, body.get('recommendedPapers') or []) if i]
    for kind, path, field in (('cited', 'references', 'citedPaper'), ('cites', 'citations', 'citingPaper')):
        time.sleep(citations.SECONDS_BETWEEN_CALLS)
        body = s2_get(S2_GRAPH % (ref, path), {'fields': S2_FIELDS, 'limit': LINKED}) or {}
        found[kind] = [i for i in (s2_item(row.get(field) or {}) for row in body.get('data') or []) if i]
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


def compute(paper):
    """The neighbourhood of a paper, as stored: {'items', 'edges', 'computed', 'counts'}."""
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
    pending = {}
    for kind in ('recommended', 'cited', 'cites'):
        for item in found[kind]:
            key = identity(item['ref'])
            if not key or key in own:
                continue
            target = pool.get(key) or pending.setdefault(key, item)
            if kind != 'recommended':
                target[kind] = True
    pending = dict(list(pending.items())[:MAX_EMBED])
    if pending:
        texts = ['%s\n\n%s' % (i['title'], i['abstract']) for i in pending.values()]
        embedded = np.concatenate([similar.embed(texts[start:start + similar.BATCH])
                                   for start in range(0, len(texts), similar.BATCH)])
        for (key, item), vector in zip(pending.items(), embedded):
            item['similarity'] = round(float(vector @ query), 3)
            vectors[key] = vector
            pool[key] = item
    stored = known_versions([i for i in pool.values() if i['source'] == 's2'])
    for key, item in pool.items():
        if item['source'] == 's2':
            item['ref'] = stored.get(key) or (item['ref'] if sources.is_external(item['ref']) else item['ref'] + 'v1')

    ranked = sorted(pool.items(), key=lambda kv: -kv[1]['similarity'])
    chosen = []
    for group, flag, kept in (('close', None, CLOSE_KEPT), ('prior', 'cited', PRIOR_KEPT), ('followup', 'cites', FOLLOWUP_KEPT)):
        taken = {k for k, _ in chosen}
        picked = [(k, i) for k, i in ranked if k not in taken and (flag is None or i[flag])][:kept]
        for _key, item in picked:
            item['group'] = group
        chosen += picked

    missing = [i['ref'] for k, i in chosen if k not in vectors]
    for aid, vec in PaperEmbedding.objects.filter(arxiv_id__in=missing, model=similar.MODEL).values_list('arxiv_id', 'vector'):
        vectors[identity(aid)] = np.frombuffer(vec, dtype=np.float32)
    items = [i for _k, i in chosen]
    return {'items': items, 'edges': edges([vectors.get(k) for k, _i in chosen]),
            'computed': timezone.now().isoformat(),
            'counts': {kind: len(found[kind]) for kind in found} | {'index': len(best)}}


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
Answer with a JSON object {{"neighbours": [{{"n": 1, "relation": "...", "why": "..."}}]}}."""


def shown(data):
    """The items the paper page shows, in its order: close, prior, follow-ups."""
    return [i for group, (_title, count) in GROUPS.items() for i in [x for x in data['items'] if x.get('group') == group][:count]]


def why_work(arxiv_id, lang):
    def work(pk):
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        data = stored(arxiv_id) or {'items': legacy(paper)}
        items = shown(data)
        if not items:
            return {'refs': [], 'by_ref': {}}
        reading = insight.for_page(arxiv_id, lang) or insight.for_page(arxiv_id, 'en') or {}
        context = {'paper': {'title': paper.title, 'question': reading.get('question', ''), 'method': reading.get('method', ''),
                             'abstract': ' '.join((paper.abstract or '').split())[:2000]},
                   'neighbours': [{'n': n, 'title': i['title'], 'abstract': i['abstract'] or '(no abstract)',
                                   'link': 'cited by the paper' if i.get('cited') else 'cites the paper' if i.get('cites') else ''}
                                  for n, i in enumerate(items, 1)]}
        generated.release_connection()
        answer = generated.chat_json(json.dumps(context, ensure_ascii=False), paid=False, max_tokens=2500, timeout=120,
                                     system=WHY_SYSTEM.format(language=generated.language_name(lang)))
        by_ref = {}
        for row in answer.get('neighbours') or []:
            if not isinstance(row, dict) or not str(row.get('n', '')).isdigit():
                continue
            n = int(row['n'])
            if 1 <= n <= len(items):
                relation = row.get('relation') if row.get('relation') in RELATIONS else ''
                by_ref[items[n - 1]['ref']] = {'relation': relation, 'why': insight.clean(row.get('why'), 300)}
        return {'refs': [i['ref'] for i in items], 'by_ref': by_ref}
    return work


def why_row(arxiv_id, lang):
    return Generated.objects.filter(kind=WHY_KIND, key=arxiv_id, lang=lang).first()


def covers(row, items):
    return row is not None and row.status == 'done' and set(i['ref'] for i in items) <= set(row.data().get('refs') or [])


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


# --- what the pages show ---------------------------------------------------------------

def year(item):
    return (item.get('date') or '')[:4]


def for_page(paper, lang, with_map=False):
    """{'groups': [{'key', 'title', 'items'}], 'computed', 'why_ready', 'complete', 'relations_json'} for the paper
    and tree pages, with 'map' (see map_data) when asked; None when the paper has no neighbour yet.
    complete: Semantic Scholar's neighbours are there too."""
    data = stored(paper.arxiv_id)
    items = data['items'] if data else legacy(paper)
    if not items:
        return None
    row = why_row(paper.arxiv_id, lang)
    why = row.data().get('by_ref', {}) if row is not None and row.status == 'done' else {}
    refs = [i['ref'] for i in items]
    summarized = set(SummaryPaper.objects.filter(paper__arxiv_id__in=refs).values_list('paper__arxiv_id', flat=True))
    tldrs = {}
    for g in Generated.objects.filter(kind=insight.KIND, key__in=refs, lang=lang, status='done'):
        tldrs[g.key] = (g.data() or {}).get('tldr', '')
    groups = []
    for key, (title, count) in GROUPS.items():
        members = []
        for item in [i for i in items if i.get('group') == key][:count]:
            code, label = level(item['similarity'])
            reason = why.get(item['ref']) or {}
            members.append(dict(item, level=code, level_label=label, percent=round(100 * item['similarity']),
                                year=year(item), summarized=item['ref'] in summarized, tldr=tldrs.get(item['ref'], ''),
                                relation=reason.get('relation', ''), relation_label=RELATIONS.get(reason.get('relation'), ''),
                                why=reason.get('why', '')))
        if members:
            groups.append({'key': key, 'title': title, 'items': members})
    computed = datetime.datetime.fromisoformat(data['computed']) if data and data.get('computed') else None
    result = {'groups': groups, 'computed': computed, 'why_ready': bool(why), 'complete': data is not None,
              'relations_json': json.dumps({code: str(label) for code, label in RELATIONS.items()}, ensure_ascii=False)}
    if with_map:
        result['map'] = map_data(paper, items, (data or {}).get('edges') or [], why, summarized)
    return result


def map_data(paper, items, links, why, summarized):
    """JSON for the map of the tree page: the paper, every stored neighbour and the edges between them."""
    nodes = []
    for item in items:
        code, label = level(item['similarity'])
        reason = why.get(item['ref']) or {}
        nodes.append({k: item.get(k, '') for k in ('ref', 'title', 'authors', 'date', 'venue', 'abstract', 'group')}
                     | {'similarity': item['similarity'], 'cited': item.get('cited', False), 'cites': item.get('cites', False),
                        'level': code, 'level_label': str(label), 'summarized': item['ref'] in summarized,
                        'relation_label': str(RELATIONS.get(reason.get('relation'), '')), 'why': reason.get('why', '')})
    main = {'ref': paper.arxiv_id, 'title': paper.title or '',
            'date': paper.published_arxiv.isoformat()[:10] if paper.published_arxiv else ''}
    return {'main': main, 'nodes': nodes, 'edges': links}
