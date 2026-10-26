"""What a collection knows of its papers besides their titles, and the slow work done for it in the background.

- facts(): year, venue, editorial status (with retractions and corrections Crossref links to the article),
  citation count and newer arXiv version of each paper, from what is stored: a page never waits for an API.
- refresh() (Premium): checks again the editorial status, citations and latest arXiv version of the papers.
  The latest versions are kept in the result, never in PaperVersion: that table is what the version alerts
  compare with, and writing it here would swallow an alert.
- suggestions() (Premium): what the collection may be missing. The papers cited by at least two of its papers
  (their references, from Semantic Scholar or OpenAlex), then the papers closest to the collection as a whole
  (the mean of their title-and-abstract embeddings, see similar.py).
- import_ids() and zotero_import(): adding many papers at once, a lookup per DOI, out of the request.
"""
import json
import logging
import re
import time
import uuid

import numpy as np
import requests
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import JsonResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, changes, chat_page, citations, comparison, digest, exports, generated, publication_status, similar, sources
from .models import (ArxivPaper, CitationCache, Collection, CollectionPaper, ExternalPaper, Generated, PaperEmbedding,
                     PaperVersion, ZoteroAccount)

logger = logging.getLogger(__name__)

META_MAX = 60  # papers checked by one refresh
SUGGEST_MAX = 15
ZOTERO_MAX = 500
META_STEPS = [
    ('status', _('Checking where each paper was published, and any retraction or correction')),
    ('citations', _('Counting the citations')),
    ('versions', _('Looking for newer arXiv versions')),
]
SUGGEST_STEPS = [
    ('references', _('Reading the references of each paper')),
    ('close', _('Finding the papers closest to the collection')),
]
IMPORT_STEPS = [('read', _('Looking up each paper')), ('add', _('Adding them to the collection'))]
GOOD, BAD, WARN, INFO = 'good', 'high', 'medium', 'info'
NOTICE_LABELS = {
    'retraction': (_('Retracted'), BAD), 'withdrawal': (_('Withdrawn'), BAD), 'removal': (_('Removed'), BAD),
    'expression_of_concern': (_('Expression of concern'), WARN),
    'correction': (_('Correction published'), INFO), 'erratum': (_('Correction published'), INFO),
    'corrigendum': (_('Correction published'), INFO),
}


def _mine(request, pk):
    return get_object_or_404(Collection, pk=pk, user=request.user)


# --- what the page shows of each paper --------------------------------------------

def facts(papers, versions=None):
    """{arxiv_id: {year, venue, status, label, tone, notices, citations, newer}} from stored data only.
    versions: {arXiv base id: latest version} found by the last refresh."""
    ids = [p.arxiv_id for p in papers]
    arxiv = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=ids)}
    external = {}
    for row in ExternalPaper.objects.filter(key__in=ids):
        try:
            external[row.key] = json.loads(row.metadata or '{}')
        except ValueError:
            pass
    reports = {g.key: g.data() for g in Generated.objects.filter(kind=publication_status.KIND, key__in=ids, lang='en')
               .exclude(content='')}
    bases = {aid: citations.base_id(aid) for aid in ids}
    counts = dict(CitationCache.objects.filter(arxiv_id__in=[b for b in bases.values() if b], checked__isnull=False)
                  .values_list('arxiv_id', 'total'))
    arxiv_bases = {aid: changes.base_id(aid) for aid in ids if not sources.is_external(aid)}
    latest = dict(PaperVersion.objects.filter(arxiv_id__in=[b for b, _v in arxiv_bases.values()]).values_list('arxiv_id', 'version'))
    for base, version in (versions or {}).items():
        latest[base] = max(latest.get(base, 0), version)
    stored_next = set(ArxivPaper.objects.filter(arxiv_id__in=['%sv%d' % (b.replace('/', '--'), v + 1)
                                                            for b, v in arxiv_bases.values() if v])
                      .values_list('arxiv_id', flat=True))
    found = {}
    for aid in ids:
        paper, meta, report = arxiv.get(aid), external.get(aid) or {}, reports.get(aid) or {}
        year = paper.published_arxiv.year if paper is not None and paper.published_arxiv else (meta.get('published') or '')[:4]
        status = report.get('status', '')
        venue = report.get('journal') if status in ('published', 'forthcoming') else ''
        venue = venue or (paper.journal_ref if paper is not None else '') or meta.get('journal', '')
        label, tone = '', INFO
        if status in ('published', 'forthcoming'):
            label, tone = publication_status.description(report), GOOD
        elif status == 'withdrawn_declared':
            label, tone = publication_status.description(report), BAD
        elif status in ('accepted_declared', 'submitted_declared', 'journal_reference', 'preprint'):
            label = publication_status.description(report)
        notices = []
        for notice in report.get('updates') or []:
            text, level = NOTICE_LABELS.get(notice.get('type'), ('', INFO))
            if text and text not in [n['label'] for n in notices]:
                notices.append({'label': text, 'tone': level, 'doi': notice.get('doi', '')})
        newer = None
        if aid in arxiv_bases:
            base, version = arxiv_bases[aid]
            if version and latest.get(base, 0) > version:
                newer = latest[base]
            elif version and '%sv%d' % (base.replace('/', '--'), version + 1) in stored_next:
                newer = version + 1
        found[aid] = {'year': str(year or ''), 'venue': (venue or '')[:160], 'status': status, 'label': label, 'tone': tone,
                      'notices': notices, 'citations': counts.get(bases.get(aid)), 'newer': newer,
                      'checked': bool(report)}
    return found


# --- checking the papers again (Premium) ---------------------------------------------

def refresh(collection_pk):
    def work(pk):
        ids = list(CollectionPaper.objects.filter(collection__pk=collection_pk).values_list('arxiv_id', flat=True))
        generated.progress(pk, 'status')
        papers = list(ArxivPaper.objects.filter(arxiv_id__in=ids))
        papers.sort(key=lambda p: (publication_status.latest(p.arxiv_id) or {}).get('checked_at', ''))  # oldest check first
        for index, paper in enumerate(papers[:META_MAX]):
            if publication_status.stale(publication_status.latest(paper.arxiv_id)):
                try:
                    publication_status.for_generation(paper)
                except Exception:
                    logger.warning('Publication status failed for %s', paper.arxiv_id)
            if index % 10 == 9:
                generated.progress(pk, 'status')  # still alive
        generated.progress(pk, 'citations')
        now = timezone.now()
        bases = [b for b in dict.fromkeys(citations.base_id(a) for a in ids) if b]
        fresh = set(CitationCache.objects.filter(arxiv_id__in=bases, checked__gte=now - citations.REFRESH)
                    .values_list('arxiv_id', flat=True))
        for index, base in enumerate([b for b in bases if b not in fresh][:META_MAX]):
            if index:
                time.sleep(citations.SECONDS_BETWEEN_CALLS)
            citations.refresh(base, now)
            if index % 10 == 9:
                generated.progress(pk, 'citations')
        generated.progress(pk, 'versions')
        versions = {}
        arxiv_bases = sorted({changes.base_id(a)[0] for a in ids if not sources.is_external(a)})
        for start in range(0, len(arxiv_bases), changes.CHECK_BATCH):
            try:
                for paper in digest.fetch_ids(arxiv_bases[start:start + changes.CHECK_BATCH]):
                    base, version = changes.base_id(paper['arxiv_id'])
                    if version:
                        versions[base] = version
            except digest.ArxivUnavailable:
                logger.warning('arXiv versions of a collection failed')
        return {'versions': versions, 'checked_at': now.isoformat(), 'papers': len(ids)}
    return work


def meta_key(collection):
    return '%s:meta' % collection.pk


@login_required
@require_POST
def refresh_start(request, pk):
    collection = _mine(request, pk)
    if not access.has_tool(request.user, 'collections'):
        return JsonResponse({'status': 'premium'}, status=403)
    row = generated.start('collection_meta', meta_key(collection), 'en', refresh(collection.pk), user=request.user,
                          force=True)
    return JsonResponse({'status': row.status, 'id': row.pk})


# --- what the collection may be missing (Premium) ------------------------------------

def same_ref(text):
    """One form for an arXiv id (without version) or a DOI, whatever the source wrote."""
    from .library import canonical_id
    key = canonical_id(str(text or ''))
    return re.sub(r'v\d+$', '', key) if key and not sources.is_external(key) else key


def common_references(papers, own):
    """[{ref, title, authors, year, count, by}] of the papers cited by two papers of the collection or more."""
    counts, by, info = {}, {}, {}
    asked = 0
    for row in papers:
        found = cache.get('chat-refs:%s' % row.arxiv_id)
        if found is None:
            if asked:
                time.sleep(citations.SECONDS_BETWEEN_CALLS)
            asked += 1
            found = chat_page.fetch_references(row.arxiv_id) or []
        for ref in found:
            key = same_ref(ref.get('ref'))
            if not key or key in own:
                continue
            if row.arxiv_id not in by.setdefault(key, []):
                by[key].append(row.arxiv_id)
                counts[key] = counts.get(key, 0) + 1
            info.setdefault(key, ref)
    ranked = sorted((k for k in counts if counts[k] >= 2), key=lambda k: (-counts[k], k))[:SUGGEST_MAX]
    titles = {p.arxiv_id: p.title or p.arxiv_id for p in papers}
    return [{'ref': key, 'title': info[key].get('title', ''), 'authors': info[key].get('authors', ''),
             'year': info[key].get('year', ''), 'count': counts[key], 'by': [titles[a] for a in by[key]][:6]}
            for key in ranked]


def closest_to(papers, own):
    """[{ref, title, authors, year, similarity}] of the indexed papers closest to the mean of the collection."""
    rows = list(ArxivPaper.objects.filter(arxiv_id__in=[p.arxiv_id for p in papers]))
    if not rows:
        return []
    try:
        similar.store([r for r in rows if not PaperEmbedding.objects.filter(arxiv_id=r.arxiv_id, model=similar.MODEL).exists()])
    except (requests.RequestException, ValueError, KeyError):
        logger.warning('Embeddings failed for a collection')
    vectors = [np.frombuffer(v, dtype=np.float32) for v in PaperEmbedding.objects.filter(
        arxiv_id__in=[r.arxiv_id for r in rows], model=similar.MODEL).values_list('vector', flat=True)]
    if not vectors:
        return []
    centre = np.mean(vectors, axis=0)
    centre = centre / (np.linalg.norm(centre) or 1)
    candidates = PaperEmbedding.objects.filter(model=similar.MODEL)
    categories = {c for c in (similar.candidate_category(r) for r in rows) if c}
    if categories:
        candidates = candidates.filter(category__in=categories)
    pairs = [(aid, np.frombuffer(vec, dtype=np.float32)) for aid, vec in candidates.values_list('arxiv_id', 'vector')]
    pairs = [(aid, vec) for aid, vec in pairs if same_ref(aid) not in own]
    if not pairs:
        return []
    scores = np.stack([vec for _a, vec in pairs]) @ centre
    chosen, seen = [], set()
    for i in np.argsort(-scores):
        key = same_ref(pairs[i][0])
        if key in seen:
            continue
        seen.add(key)
        chosen.append((pairs[i][0], float(scores[i])))
        if len(chosen) == SUGGEST_MAX:
            break
    found = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=[a for a, _s in chosen]).prefetch_related('authors')}
    result = []
    for aid, score in chosen:
        paper = found.get(aid)
        if paper is None or not paper.title:
            continue
        names = [a.name for a in paper.authors.all()[:4]]
        result.append({'ref': aid, 'title': ' '.join(paper.title.split()),
                       'authors': ', '.join(names[:3]) + (' et al.' if len(names) > 3 else ''),
                       'year': paper.published_arxiv.year if paper.published_arxiv else '',
                       'similarity': round(100 * score, 1)})
    return result


def suggestions(collection_pk):
    def work(pk):
        collection = Collection.objects.get(pk=collection_pk)
        papers = comparison.analysed(collection)
        own = {same_ref(a) for a in collection.papers.values_list('arxiv_id', flat=True)}
        generated.progress(pk, 'references')
        cited = common_references(papers, own)
        generated.progress(pk, 'close')
        close = closest_to(papers, own | {c['ref'] for c in cited})
        return {'cited': cited, 'close': close, 'ids': [p.arxiv_id for p in papers]}
    return work


def suggest_key(collection):
    return '%s:suggest' % collection.pk


@login_required
@require_POST
def suggest_start(request, pk):
    collection = _mine(request, pk)
    if not access.has_tool(request.user, 'collections'):
        return JsonResponse({'status': 'premium'}, status=403)
    if not comparison.analysed(collection):
        return JsonResponse({'status': 'too_few'})
    row = generated.start('suggest', suggest_key(collection), get_language(), suggestions(collection.pk),
                          user=request.user, force=True)
    return JsonResponse({'status': row.status, 'id': row.pk})


# --- adding many papers --------------------------------------------------------------

def import_ids(collection_pk, ids, paid, skipped=0):
    def work(pk):
        from .library import add_many
        collection = Collection.objects.get(pk=collection_pk)
        generated.progress(pk, 'read')
        counts = add_many(collection, ids, paid, step=lambda: generated.progress(pk, 'add'))
        counts['skipped'] = counts.get('skipped', 0) + skipped
        return counts
    return work


def start_import(request, collection, ids, paid, skipped=0):
    return generated.start('collection_import', uuid.uuid4().hex, get_language(),
                           import_ids(collection.pk, ids, paid, skipped), user=request.user)


ARXIV_IN_ZOTERO = re.compile(r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?)', re.I)


def zotero_ref(data):
    """The arXiv id or DOI of a Zotero item, '' when it has neither."""
    for field in ('archiveID', 'url', 'extra'):
        match = ARXIV_IN_ZOTERO.search(str(data.get(field) or ''))
        if match:
            return match.group(1)
    doi = sources.clean_doi(str(data.get('DOI') or '')) or sources.clean_doi(
        (re.search(r'^DOI:\s*(\S+)', str(data.get('extra') or ''), re.I | re.M) or [None, ''])[1])
    return doi or ''


def zotero_get(account, path, params=None):
    response = requests.get('%s/users/%s/%s' % (exports.ZOTERO_API, account.library_id, path), params=params or {},
                            headers={'Zotero-API-Key': account.api_key, 'Zotero-API-Version': '3'}, timeout=30)
    response.raise_for_status()
    return response.json()


def zotero_items(account, key):
    """The references of the top items of a Zotero collection: (arXiv ids or DOIs, items without either)."""
    refs, skipped = [], 0
    for start in range(0, ZOTERO_MAX, 100):
        items = zotero_get(account, 'collections/%s/items/top' % key, {'format': 'json', 'limit': 100, 'start': start})
        for item in items:
            data = item.get('data') or {}
            if data.get('itemType') in ('attachment', 'note'):
                continue
            ref = zotero_ref(data)
            if ref:
                refs.append(ref)
            else:
                skipped += 1
        if len(items) < 100:
            break
    return refs, skipped


def zotero_access(request):
    if not access.has_tool(request.user, 'zotero'):
        return None, JsonResponse({'status': 'premium'}, status=403)
    account = ZoteroAccount.objects.filter(user=request.user).first()
    if account is None:
        return None, JsonResponse({'status': 'setup', 'url': '/%s/integrations/' % get_language()})
    return account, None


@login_required
def zotero_collections(request):
    """The user's Zotero collections, to choose the one to import. JSON."""
    account, refused = zotero_access(request)
    if refused:
        return refused
    try:
        found = []
        for start in range(0, 300, 100):
            page = zotero_get(account, 'collections', {'limit': 100, 'start': start})
            found += [{'key': c.get('key'), 'name': (c.get('data') or {}).get('name', ''),
                       'items': (c.get('meta') or {}).get('numItems', 0)} for c in page]
            if len(page) < 100:
                break
    except (requests.RequestException, ValueError):
        return JsonResponse({'status': 'unavailable'})
    return JsonResponse({'status': 'done', 'collections': sorted(found, key=lambda c: c['name'].lower())})


def zotero_import(collection_pk, account, key, paid):
    def work(pk):
        from .library import add_many, canonical_id
        generated.progress(pk, 'read')
        refs, skipped = zotero_items(account, key)
        ids = [i for i in dict.fromkeys(canonical_id(r) for r in refs) if i]
        skipped += len(refs) - len(ids)
        generated.progress(pk, 'add')
        counts = add_many(Collection.objects.get(pk=collection_pk), ids, paid)
        counts['skipped'] = counts.get('skipped', 0) + skipped
        return counts
    return work


@login_required
@require_POST
def zotero_import_start(request, pk):
    collection = _mine(request, pk)
    account, refused = zotero_access(request)
    if refused:
        return refused
    key = request.POST.get('key', '')
    if not re.fullmatch(r'[A-Z0-9]{8}', key):
        return JsonResponse({'status': 'unknown'}, status=400)
    row = generated.start('zotero_import', uuid.uuid4().hex, get_language(),
                          zotero_import(collection.pk, account, key, access.has_tool(request.user, 'collections')),
                          user=request.user)
    return JsonResponse({'status': row.status, 'id': row.pk})
