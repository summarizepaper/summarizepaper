"""Collections: a researcher's folders of papers, to sort, annotate, compare, chat with, share and export.

Everyone with an account gets a few collections, with notes, tags, a reading status, imports (identifiers,
links, BibTeX, RIS), exports and a read-only link to share. Premium gets many more, the comparison table
(comparison.py), the open questions (questions.py), what the collection may be missing, checks of where each
paper was published (collection_tools.py), imports from Zotero, and alerts on new citations and versions of
the papers of a collection it watches (citations.py, changes.py).

The analyses and the chat read the selected papers of a collection (comparison.analysed). A paper is kept
once, whatever its version: adding a newer version replaces the older one, and a preprint and its published
version are recognized as one work (work_keys) when arXiv, Crossref or the source say so.

comparison and collection_tools are imported where they are used: presskit imports this module, and evidence,
which comparison uses, imports presskit.
"""
import csv
import datetime
import io
import json
import re
import secrets

import requests
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.utils.http import urlencode
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, citations, digest, exports, sources
from django.db.models import Max

from .models import (AnnouncedPaper, ArxivPaper, Collection, CollectionPaper, ExternalPaper, FollowedPaper, Generated,
                     ZoteroAccount)

FREE = {'collections': 3, 'papers': 20}
PREMIUM = {'collections': 100, 'papers': 200}
CHAT_MAX = 10  # papers the chat page takes at once
MAX_ADD = 200  # ids added at once (one arXiv query)
SYNC_ADD = 10  # more than this are added in the background: each DOI is one lookup
MAX_UPLOAD = 2 * 1024 * 1024
SHARE_DAYS = 90  # a shared link works this long after its last sharing
READING = [('to_read', _('To read')), ('read', _('Read')), ('key', _('Key paper'))]
ADD_RESULTS = ('added', 'updated', 'already', 'duplicate', 'unknown', 'full', 'skipped')


def limits(paid):
    return PREMIUM if paid else FREE


def canonical_id(text):
    """An arXiv id or link as typed -> the id form of our URLs ('2101.00001v1', 'hep-th--9901001v1');
    a DOI or the key of a paper from another source -> its key (see sources.py)."""
    key = text.strip().lower()
    if sources.is_external(key):
        return key if sources.valid_key(key) else None
    doi = sources.clean_doi(text)
    if doi and not sources.ARXIV_DOI_RE.match(doi):
        return sources.doi_key(doi)
    match = digest.ARXIV_ID_RE.search(text.replace('--', '/'))
    return digest.site_id(match.group(1)) if match else None


def version_of(arxiv_id):
    """('2101.00001', 2) for '2101.00001v2'; (key, 0) for a paper without version or from another source."""
    if sources.is_external(arxiv_id):
        return arxiv_id, 0
    match = re.match(r'^(.*?)(?:v(\d+))?$', arxiv_id)
    return match.group(1), int(match.group(2) or 0)


def identity(text):
    """One form for any identifier of a paper: 'arxiv:2101.00001' (no version; an arXiv DOI too),
    'doi:10.1093/mnras/stad1234', or the key of another source; None when it is none of them."""
    key = canonical_id(str(text or '')) if text else None
    if not key:
        return None
    if sources.is_external(key):
        doi = sources.key_doi(key) if key.startswith('doi_') else ''
        return 'doi:' + doi.lower() if doi else key
    return 'arxiv:' + version_of(key)[0].lower()


def work_keys(arxiv_ids):
    """{id: the identities of its work}: its own, the journal DOI arXiv gives, the published version Crossref
    links (publication_status), and the arXiv id or DOI another source gives. A preprint and its published
    version share one of them."""
    keys = {a: {k for k in [identity(a)] if k} for a in arxiv_ids}

    def add(aid, value):
        key = identity(value)
        if key:
            keys[aid].add(key)
    for aid, link in ArxivPaper.objects.filter(arxiv_id__in=arxiv_ids).values_list('arxiv_id', 'link_doi'):
        add(aid, sources.clean_doi(link or '') or '')
    for row in Generated.objects.filter(kind='presskit_status', key__in=arxiv_ids, lang='en').exclude(content=''):
        report = row.data()
        if report.get('status') in ('published', 'forthcoming'):
            add(row.key, report.get('doi', ''))
    for row in ExternalPaper.objects.filter(key__in=arxiv_ids):
        try:
            meta = json.loads(row.metadata or '{}')
        except ValueError:
            continue
        for value in (meta.get('arxiv_id'), meta.get('doi')):
            add(row.key, value or '')
    return keys


def duplicates(papers):
    """[[paper, paper...]] of the papers of a collection that are one work (a preprint and its published version)."""
    keys = work_keys([p.arxiv_id for p in papers])
    groups, owner = [], {}
    for paper in papers:
        joined = [g for g in groups if keys[paper.arxiv_id] & g['keys']]
        group = {'keys': set(keys[paper.arxiv_id]), 'papers': [paper]}
        for other in joined:
            groups.remove(other)
            group['keys'] |= other['keys']
            group['papers'] = other['papers'] + group['papers']
        groups.append(group)
    return [g['papers'] for g in groups if len(g['papers']) > 1]


# --- reading pasted lists, BibTeX and RIS -------------------------------------------

BIBTEX_ENTRY = re.compile(r'@\w+\s*[{(]')
ARXIV_IN_TEXT = re.compile(r'(?:arxiv\.org/(?:abs|pdf)/|\barxiv\s*:\s*|eprint\s*=\s*[{"]\s*(?:arxiv:)?\s*)'
                           r'((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?)', re.I)
DOI_IN_TEXT = re.compile(r'(?:\bdoi\s*=\s*[{"]\s*|^DO\s+-\s+|doi\.org/|\bdoi:\s*)(10\.\d{4,9}/[^\s"{}]+)', re.I | re.M)


def entry_ref(entry):
    """The arXiv id, else the DOI, of one BibTeX or RIS entry; None when it has neither."""
    match = ARXIV_IN_TEXT.search(entry)
    if match:
        return match.group(1)
    match = DOI_IN_TEXT.search(entry)
    return match.group(1).rstrip('.,;') if match else None


def references_in(text):
    """(the identifiers of a pasted or uploaded list, how many entries had none): BibTeX and RIS entries give
    their arXiv id or DOI; anything else is read as ids and links separated by spaces, commas or new lines."""
    if BIBTEX_ENTRY.search(text):
        entries = [e for e in re.split(r'(?=@\w+\s*[{(])', text) if BIBTEX_ENTRY.match(e.strip())]
        entries = [e for e in entries if not re.match(r'@(?:comment|string|preamble)\b', e.strip(), re.I)]
    elif re.search(r'^TY  - ', text, re.M):
        entries = [e for e in re.split(r'^ER  -.*$', text, flags=re.M) if e.strip()]
    else:
        return [t for t in re.split(r'[\s,;]+', text) if t], 0
    refs = [entry_ref(e) for e in entries]
    return [r for r in refs if r], sum(1 for r in refs if not r)


def wanted_ids(texts):
    """(canonical ids, MAX_ADD at most, without repeats; entries or words that are not papers)."""
    ids, skipped = [], 0
    for text in texts:
        refs, missed = references_in(text)
        skipped += missed
        for ref in refs:
            arxiv_id = canonical_id(ref)
            if arxiv_id and arxiv_id not in ids:
                ids.append(arxiv_id)
            elif not arxiv_id:
                skipped += 1
    return ids[:MAX_ADD], skipped


# --- adding papers -----------------------------------------------------------------

def lookup_many(arxiv_ids):
    """{arxiv_id: (title, authors)} from our tables, then one arXiv query for the rest; unknown ids are left out."""
    found = {}
    for paper in ArxivPaper.objects.filter(arxiv_id__in=arxiv_ids).prefetch_related('authors'):
        if paper.title:
            found[paper.arxiv_id] = (' '.join(paper.title.split()), ', '.join(a.name for a in paper.authors.all()[:8]))
    for announced in AnnouncedPaper.objects.filter(arxiv_id__in=[a for a in arxiv_ids if a not in found]):
        found[announced.arxiv_id] = (announced.title, ', '.join(json.loads(announced.authors or '[]')[:8]))
    for key in [a for a in arxiv_ids if a not in found and sources.is_external(a)]:
        try:
            paper = sources.metadata(key)
        except sources.SourceUnavailable:
            paper = None
        if paper and paper['title']:
            found[key] = (paper['title'], ', '.join(paper['authors'][:8]))
    missing = [a for a in arxiv_ids if a not in found and not sources.is_external(a)]
    if missing:
        try:
            response = requests.get(digest.ARXIV_API, timeout=10, params={
                'id_list': ','.join(a.replace('--', '/') for a in missing), 'max_results': len(missing)})
            papers = digest.parse_feed(response.text) if response.status_code == 200 else []
        except Exception:
            papers = []
        for paper in papers:
            for asked in missing:
                # arXiv answers with the latest version when the id had none
                if paper['title'] and (paper['arxiv_id'] == asked or re.sub(r'v\d+$', '', paper['arxiv_id']) == asked):
                    found[asked] = (paper['title'], ', '.join(paper['authors'][:8]))
    return found


def lookup(arxiv_id):
    """(title, authors) of a paper, from our tables or from arXiv; None if arXiv does not know it."""
    return lookup_many([arxiv_id]).get(arxiv_id)


def kept_version(collection, arxiv_id):
    """The paper of the collection that is this one in any version, or None."""
    base = version_of(arxiv_id)[0]
    return next((p for p in collection.papers.filter(arxiv_id__startswith=base) if version_of(p.arxiv_id)[0] == base), None)


def add_paper(collection, arxiv_id, paid, found=None, works=None):
    """'added', 'updated' (a newer version replaced the one kept), 'already', 'duplicate' (its preprint or
    published version is there), 'full' or 'unknown'. found: what lookup_many already knows;
    works: the identities of the works already in the collection (see work_keys), to share between calls."""
    kept = kept_version(collection, arxiv_id)
    if kept is not None:
        if version_of(arxiv_id)[1] > version_of(kept.arxiv_id)[1] > 0:
            newer = (found or {}).get(arxiv_id) if found is not None else lookup(arxiv_id)
            kept.arxiv_id = arxiv_id
            if newer:
                kept.title, kept.authors = newer
            kept.save(update_fields=['arxiv_id', 'title', 'authors'])
            collection.save(update_fields=['updated'])
            return 'updated'
        return 'already'
    if works is None:
        works = set().union(*work_keys(list(collection.papers.values_list('arxiv_id', flat=True))).values())
    if work_keys([arxiv_id])[arxiv_id] & works:
        return 'duplicate'
    if collection.papers.count() >= limits(paid)['papers']:
        return 'full'
    found = found.get(arxiv_id) if found is not None else lookup(arxiv_id)
    if found is None:
        return 'unknown'
    last = collection.papers.aggregate(last=Max('position'))['last'] or 0
    CollectionPaper.objects.get_or_create(collection=collection, arxiv_id=arxiv_id,
                                          defaults={'title': found[0], 'authors': found[1], 'position': last + 1})
    works |= work_keys([arxiv_id])[arxiv_id]
    collection.save(update_fields=['updated'])
    return 'added'


def add_many(collection, ids, paid, step=None):
    """{result: count} once these canonical ids are added (see add_paper); a watched collection follows them."""
    known = lookup_many(ids)
    if step:
        step()
    counts, added = {}, []
    works = set().union(*work_keys(list(collection.papers.values_list('arxiv_id', flat=True))).values())
    for arxiv_id in ids:
        status = add_paper(collection, arxiv_id, paid, known, works)
        counts[status] = counts.get(status, 0) + 1
        if status in ('added', 'updated'):
            added.append(arxiv_id)
    if collection.watch and added and access.has_tool(collection.user, 'citations'):
        counts['unfollowed'] = follow_papers(collection.user, added)
    return counts


def follow_papers(user, arxiv_ids):
    """Follow these papers (citation alerts, and version alerts for arXiv papers); returns how many could not be
    followed, past the most a user follows. Their first check is the hourly run's (citations.refresh_due), which
    records the citations they already have as known: a whole collection is not read here, a call a second."""
    room = citations.MAX_FOLLOWED - FollowedPaper.objects.filter(user=user).count()
    missed = 0
    for arxiv_id in arxiv_ids:
        base = citations.base_id(arxiv_id)
        if not base or FollowedPaper.objects.filter(user=user, arxiv_id=base).exists():
            continue
        if room <= 0:
            missed += 1
            continue
        citations.follow(user, base)
        room -= 1
    return missed


def followed_count(user, collection):
    bases = {citations.base_id(a) for a in collection.papers.values_list('arxiv_id', flat=True)}
    return FollowedPaper.objects.filter(user=user, arxiv_id__in=[b for b in bases if b]).count()


# --- views -------------------------------------------------------------------------

def _mine(request, pk):
    return get_object_or_404(Collection, pk=pk, user=request.user)


def _page(collection, tab='', **params):
    query = urlencode({k: v for k, v in params.items() if v})
    return redirect('/%s/collections/%s/%s%s' % (get_language(), collection.pk, '?' + query if query else '',
                                                  '#' + tab if tab else ''))


def collections_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/collections.html', {'anonymous': True, 'limits': FREE, 'premium': PREMIUM})
    paid = access.has_tool(request.user, 'collections')
    collections = list(Collection.objects.filter(user=request.user).prefetch_related('papers'))
    can_create = len(collections) < limits(paid)['collections']
    if request.method == 'POST' and can_create:
        name = request.POST.get('name', '').strip()[:120]
        if name:
            collection = Collection.objects.create(user=request.user, name=name)
            return redirect('/%s/collections/%s/' % (get_language(), collection.pk))
    now = timezone.now()
    for c in collections:
        c.shared = bool(c.share_token and (c.share_expires is None or c.share_expires > now))
    return render(request, 'summarizer/collections.html', {
        'collections': collections, 'paid': paid, 'can_create': can_create, 'limits': limits(paid), 'premium': PREMIUM})


def clean_tags(text):
    tags = []
    for tag in str(text or '').split(','):
        tag = ' '.join(tag.split())[:40]
        if tag and tag.lower() not in [t.lower() for t in tags]:
            tags.append(tag)
    return ', '.join(tags)[:300]


def update_paper(collection, post):
    """Change what the request sends of one paper: note, tags, reading status, selection. The paper, or None."""
    paper = collection.papers.filter(arxiv_id=post.get('arxiv_id', '')).first()
    if paper is None:
        return None
    fields = []
    if 'note' in post:
        paper.note = post['note'][:5000]
        fields.append('note')
    if 'tags' in post:
        paper.tags = clean_tags(post['tags'])
        fields.append('tags')
    if 'reading' in post and post['reading'] in ('',) + tuple(code for code, _l in READING):
        paper.reading = post['reading']
        fields.append('reading')
    if 'selected' in post:
        paper.selected = post['selected'] in ('1', 'true', 'on')
        fields.append('selected')
    if fields:
        paper.save(update_fields=fields)
    return paper


def select(collection, value):
    """Select all papers, none, or those with a tag or a reading status."""
    papers = collection.papers.all()
    if value == 'all':
        papers.update(selected=True)
    elif value == 'none':
        papers.update(selected=False)
    elif value.startswith('tag:'):
        tag = value[4:].strip().lower()
        chosen = [p.pk for p in papers if tag in [t.strip().lower() for t in p.tags.split(',')]]
        papers.update(selected=False)
        papers.filter(pk__in=chosen).update(selected=True)
    elif value.startswith('reading:'):
        papers.update(selected=False)
        papers.filter(reading=value[8:]).update(selected=True)


def reorder(collection, ids):
    """Put the papers in this order (ids of the collection; the others keep theirs, after them)."""
    papers = list(collection.papers.all())
    rank = {aid: i for i, aid in enumerate(ids)}
    papers.sort(key=lambda p: (rank.get(p.arxiv_id, len(rank)), p.position, p.added, p.pk))
    for position, paper in enumerate(papers, 1):
        if paper.position != position:
            paper.position = position
            paper.save(update_fields=['position'])


def move(collection, arxiv_id, step):
    ids = list(collection.papers.values_list('arxiv_id', flat=True))
    if arxiv_id in ids:
        i = ids.index(arxiv_id)
        j = max(0, min(len(ids) - 1, i + step))
        ids[i], ids[j] = ids[j], ids[i]
        reorder(collection, ids)


def share(collection, notes):
    if not collection.share_token:
        collection.share_token = secrets.token_urlsafe(16)
    collection.share_expires = timezone.now() + datetime.timedelta(days=SHARE_DAYS)
    collection.share_notes = notes
    collection.save(update_fields=['share_token', 'share_expires', 'share_notes', 'updated'])


def shared_url(request, collection):
    if not collection.share_token or (collection.share_expires and collection.share_expires < timezone.now()):
        return ''
    return request.build_absolute_uri('/%s/collections/shared/%s/' % (get_language(), collection.share_token))


@login_required
def collection_page(request, pk):
    from . import collection_tools, comparison
    collection = _mine(request, pk)
    paid = access.has_tool(request.user, 'collections')
    if request.method == 'POST':
        action = request.POST.get('action')
        fetch = request.headers.get('X-Requested-With') == 'fetch'
        if action == 'delete':
            collection.delete()
            return redirect('/%s/collections/' % get_language())
        if action == 'rename':
            collection.name = request.POST.get('name', '').strip()[:120] or collection.name
            collection.save(update_fields=['name', 'updated'])
            return _page(collection, 'settings')
        if action == 'remove':
            collection.papers.filter(arxiv_id=request.POST.get('arxiv_id')).delete()
            if fetch:
                return JsonResponse({'status': 'removed'})
            return _page(collection, 'papers')
        if action in ('paper', 'note'):
            paper = update_paper(collection, request.POST)
            if fetch:
                if paper is None:
                    return JsonResponse({'status': 'unknown'}, status=404)
                return JsonResponse({'status': 'saved', 'tags': paper.tags, 'selected': paper.selected,
                                     'analysed': len(comparison.analysed(collection))})
            return _page(collection, 'papers')
        if action in ('up', 'down'):
            move(collection, request.POST.get('arxiv_id', ''), -1 if action == 'up' else 1)
            if fetch:
                return JsonResponse({'status': 'moved'})
            return _page(collection, 'papers')
        if action == 'order':
            reorder(collection, [i for i in request.POST.get('ids', '').split(',') if i])
            return JsonResponse({'status': 'moved'}) if fetch else _page(collection, 'papers')
        if action == 'select':
            select(collection, request.POST.get('value', ''))
            return _page(collection, request.POST.get('tab', 'papers'))
        if action == 'columns':
            template = request.POST.get('template', 'general')
            collection.template = template if template in dict(comparison.TEMPLATE_CHOICES) else 'general'
            collection.custom_columns = json.dumps(comparison.parse_custom(request.POST.get('custom', '')))
            collection.save(update_fields=['template', 'custom_columns', 'updated'])
            return _page(collection, 'table')
        if action == 'watch':
            if not access.has_tool(request.user, 'citations'):
                return _page(collection, 'settings')
            collection.watch = request.POST.get('value') == 'on'
            collection.save(update_fields=['watch', 'updated'])
            missed = follow_papers(request.user, list(collection.papers.values_list('arxiv_id', flat=True))) if collection.watch else 0
            return _page(collection, 'settings', unfollowed=missed)
        if action == 'share':
            share(collection, request.POST.get('notes') == 'on')
            return _page(collection, 'settings')
        if action == 'unshare':
            collection.share_token, collection.share_expires = '', None
            collection.save(update_fields=['share_token', 'share_expires', 'updated'])
            return _page(collection, 'settings')
        if action == 'add':
            texts = [request.POST.get('ids', '')]
            upload = request.FILES.get('file')
            if upload is not None and upload.size <= MAX_UPLOAD:
                texts.append(upload.read().decode('utf-8', 'replace'))
            ids, skipped = wanted_ids(texts)
            if len(ids) > SYNC_ADD:
                row = collection_tools.start_import(request, collection, ids, paid, skipped)
                return _page(collection, 'papers', job=row.pk)
            counts = add_many(collection, ids, paid)
            counts['skipped'] = counts.get('skipped', 0) + skipped
            if not ids and not skipped:
                counts['unknown'] = 1
            tab = request.POST.get('tab') if request.POST.get('tab') in ('papers', 'missing') else 'papers'
            return _page(collection, tab, **counts)
        return _page(collection)

    lang = get_language()
    papers = list(collection.papers.all())
    chosen = comparison.analysed(collection)
    meta_row = Generated.objects.filter(kind='collection_meta', key=collection_tools.meta_key(collection)).first()
    meta = meta_row.data() if meta_row is not None and meta_row.content else {}
    try:
        meta_checked = datetime.datetime.fromisoformat(meta['checked_at'])
    except (KeyError, TypeError, ValueError):
        meta_checked = None
    facts = collection_tools.facts(papers, meta.get('versions'))
    for index, p in enumerate(papers):
        p.order, p.facts = index, facts.get(p.arxiv_id, {})
        p.tag_list = [t.strip() for t in p.tags.split(',') if t.strip()]
    tags = sorted({t for p in papers for t in p.tag_list}, key=str.lower)

    context = {
        'collection': collection, 'papers': papers, 'paid': paid, 'limits': limits(paid), 'tags': tags,
        'duplicates': duplicates(papers),
        'reading_choices': READING, 'analysed': chosen, 'compare_max': comparison.COMPARE_MAX,
        'chat_ids': ','.join(p.arxiv_id for p in chosen[:CHAT_MAX]), 'chat_max': CHAT_MAX,
        'added': {k: int(request.GET[k]) for k in ADD_RESULTS + ('unfollowed',) if request.GET.get(k, '').isdigit()},
        'job': request.GET.get('job', ''), 'import_steps': collection_tools.IMPORT_STEPS,
        'meta_row': meta_row, 'meta_checked': meta_checked, 'meta_steps': collection_tools.META_STEPS,
        'can_follow': access.has_tool(request.user, 'citations'), 'max_followed': citations.MAX_FOLLOWED,
        'followed': followed_count(request.user, collection) if collection.watch else 0,
        'zotero': ZoteroAccount.objects.filter(user=request.user).exists(),
        'can_zotero': access.has_tool(request.user, 'zotero'),
        'share_url': shared_url(request, collection),
        'templates': comparison.TEMPLATE_CHOICES,
        'template_columns': [(code, [label for _c, label, _h in cols]) for code, cols in comparison.TEMPLATES.items()],
        'custom_text': '\n'.join('%s: %s' % (c['label'], c['hint']) if c['hint'] else c['label']
                                 for c in comparison.custom_columns(collection)),
        'columns_now': [label for _c, label, _h in comparison.columns(collection)],
        'compare_steps': comparison.STEPS, 'suggest_steps': collection_tools.SUGGEST_STEPS,
        'questions_steps': [('read', _('Reading the limitations, discussions and future work of each paper')),
                            ('write', _('An AI gathers the open questions and groups them'))],
    }
    if paid:
        exact, shown = comparison.shown_result(collection, 'compare', lang)
        context.update(comparison=exact, compare_shown=shown, editable=True,
                       table=comparison.display(shown.data(), lang) if shown is not None else None,
                       compare_stale=comparison.staleness(collection, 'compare', shown, chosen))
        exact, shown = comparison.shown_result(collection, 'questions', lang)
        context.update(questions=exact, questions_shown=shown, open=shown.data() if shown is not None else None,
                       questions_stale=comparison.staleness(collection, 'questions', shown, chosen))
        suggest = Generated.objects.filter(kind='suggest', key=collection_tools.suggest_key(collection), lang=lang).first()
        context.update(suggest=suggest, suggested=suggest.data() if suggest is not None and suggest.content else None)
    return render(request, 'summarizer/collection.html', context)


@require_POST
def add_to_collection(request):
    """From a paper page: add the paper to a collection, or to a new one. JSON."""
    if not request.user.is_authenticated:
        return JsonResponse({'status': 'login'}, status=403)
    paid = access.has_tool(request.user, 'collections')
    arxiv_id = request.POST.get('arxiv_id', '')
    if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        raise Http404
    wanted = request.POST.get('collection')
    if wanted == 'new':
        if Collection.objects.filter(user=request.user).count() >= limits(paid)['collections']:
            return JsonResponse({'status': 'too_many'})
        name = request.POST.get('name', '').strip()[:120] or arxiv_id
        collection = Collection.objects.create(user=request.user, name=name)
    else:
        collection = _mine(request, wanted)
    status = add_many(collection, [arxiv_id], paid)
    status = next(iter(k for k in status if k in ADD_RESULTS), 'unknown')
    return JsonResponse({'status': 'added' if status == 'updated' else status, 'collection': collection.pk,
                         'name': collection.name, 'url': '/%s/collections/%s/' % (get_language(), collection.pk)})


def export_name(collection):
    return re.sub(r'[^A-Za-z0-9_-]+', '-', collection.name).strip('-')[:60] or 'collection'


@login_required
def collection_export(request, pk, fmt):
    """bib, ris, md: the papers (with a tag only, ?tag=), with their notes and tags; list: the papers as a
    spreadsheet; csv: the comparison table."""
    from . import comparison
    collection = _mine(request, pk)
    lang = get_language()
    filename = export_name(collection)
    if fmt == 'csv':
        _exact, shown = comparison.shown_result(collection, 'compare', lang)
        if shown is None or not access.has_tool(request.user, 'collections'):
            raise Http404
        return comparison.csv_response(shown.data(), lang, filename)
    rows = list(collection.papers.all())
    tag = request.GET.get('tag', '').strip().lower()
    if tag:
        rows = [r for r in rows if tag in [t.strip().lower() for t in r.tags.split(',')]]
    if fmt == 'list':
        return list_csv(rows, filename)
    metas = [m for m in (exports.paper_meta(p.arxiv_id, p) for p in rows) if m]
    notes = {p.arxiv_id: (p.note, p.tags) for p in rows}
    return exports.export_many(metas, fmt, filename, lang, notes=notes)


def list_csv(papers, filename):
    """The papers as a spreadsheet: what the page shows of each, with the reader's notes, tags and status."""
    from . import collection_tools
    facts = collection_tools.facts(papers)
    reading = dict(READING)
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(['Identifier', 'Title', 'Authors', 'Year', 'Venue', 'Editorial status', 'Notices', 'Citations',
                     'Newer version', 'Reading status', 'Tags', 'Note', 'Selected', 'Link'])
    for p in papers:
        f = facts.get(p.arxiv_id, {})
        writer.writerow([exports.bare_ref(p.arxiv_id), p.title, p.authors, f.get('year', ''), f.get('venue', ''),
                         f.get('label', ''), '; '.join(str(n['label']) for n in f.get('notices', [])),
                         '' if f.get('citations') is None else f['citations'], f.get('newer') or '',
                         str(reading.get(p.reading, '')), p.tags, p.note, 'yes' if p.selected else 'no',
                         sources.ref_url(p.arxiv_id) if sources.is_external(p.arxiv_id)
                         else 'https://arxiv.org/abs/%s' % exports.bare_ref(p.arxiv_id)])
    response = HttpResponse('\ufeff' + out.getvalue(), content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="%s-papers.csv"' % filename
    return response


def paper_collections(user, arxiv_id):
    """For the paper page: the user's collections, and whether each holds this paper (in any version)."""
    if not user.is_authenticated:
        return []
    base = version_of(arxiv_id)[0]
    inside = {c for c, aid in CollectionPaper.objects.filter(collection__user=user, arxiv_id__startswith=base)
              .values_list('collection_id', 'arxiv_id') if version_of(aid)[0] == base}
    return [{'pk': c.pk, 'name': c.name, 'has': c.pk in inside} for c in Collection.objects.filter(user=user)]


# --- the shared page ---------------------------------------------------------------

def _shared(token):
    collection = get_object_or_404(Collection, share_token=token) if token else None
    if collection is None or (collection.share_expires and collection.share_expires < timezone.now()):
        raise Http404
    return collection


def shared_page(request, token):
    """The read-only page of a shared collection: its papers (the owner's notes when they chose to show them),
    its comparison table and its open questions, never who made it."""
    from . import collection_tools, comparison
    collection = _shared(token)
    lang = get_language()
    papers = list(collection.papers.all())
    facts = collection_tools.facts(papers)
    for p in papers:
        p.facts = facts.get(p.arxiv_id, {})
    _exact, table = comparison.shown_result(collection, 'compare', lang)
    _exact, questions = comparison.shown_result(collection, 'questions', lang)
    return render(request, 'summarizer/collection_shared.html', {
        'collection': collection, 'papers': papers, 'token': token,
        'table': comparison.display(table.data(), lang) if table is not None else None,
        'open': questions.data() if questions is not None else None,
    })


@login_required
@require_POST
def shared_copy(request, token):
    """Make a collection of one's own with the papers of a shared one (not the owner's notes)."""
    source = _shared(token)
    paid = access.has_tool(request.user, 'collections')
    if Collection.objects.filter(user=request.user).count() >= limits(paid)['collections']:
        return redirect('/%s/collections/' % get_language())
    collection = Collection.objects.create(user=request.user, name=source.name, template=source.template,
                                           custom_columns=source.custom_columns)
    for position, paper in enumerate(source.papers.all()[:limits(paid)['papers']], 1):
        CollectionPaper.objects.create(collection=collection, arxiv_id=paper.arxiv_id, title=paper.title,
                                       authors=paper.authors, selected=paper.selected, position=position)
    return redirect('/%s/collections/%s/' % (get_language(), collection.pk))
