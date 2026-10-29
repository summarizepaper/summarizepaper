"""The chat page: conversations with up to ten papers at once.

A conversation is a room (a random id the page keeps in its URL, ?c=) holding the questions, answers and
comparison tables of one reader; opening the URL again shows it to that reader only (account, else IP), with
the papers it was last about. Its owner can name it, delete it, or share it: a public, read-only page of the
conversation as it was when shared, for SHARE_DAYS days. Besides the page, the endpoints the page script calls: what the assistant will read of each
selected paper, the references of a paper to add one in a click, and the BibTeX, RIS or Markdown of the
selection.
"""
import datetime
import logging
import re
import secrets
import time

import requests
from django.conf import settings
from django.core.cache import cache
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, render
from django.utils import timezone
from django.views.decorators.http import require_POST
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import getting_started, access, assistant, citations, exports, result_share, sources
from .models import AIassistant, ArxivPaper, ChatConversation, ChatShare

logger = logging.getLogger(__name__)

ROOM_RE = re.compile(r'^[A-Za-z0-9_-]{12,64}$')
ID_RE = re.compile(r'^[\w.\-/]{1,100}$')
RECENT = 8  # conversations listed on the page
S2_REFERENCES = 'https://api.semanticscholar.org/graph/v1/paper/%s/references'
MAX_REFERENCES = 100
REFERENCES_SECONDS = 24 * 3600
S2_RETRY_SECONDS = 1.5
SHARE_DAYS = 90  # a shared link works this long after its last sharing
MAX_NAME_CHARS = 120
TABLE_COLUMNS = [('question', _('Question')), ('data', _('Data')), ('method', _('Method')),
                 ('sample', _('Sample size and scale')), ('results', _('Main results')),
                 ('limitations', _('Limitations'))]


def owned(rows, request):
    """The rows of this visitor: their account, else their IP without an account."""
    if request.user.is_authenticated:
        return rows.filter(user=request.user)
    return rows.filter(user=None, ip_address=access.hash_ip(access.client_ip(request)))


def recent_conversations(request, current=''):
    """The visitor's last chat page conversations: [{room, question, date, papers, title}], newest first.

    A chat page room is not a paper: paper pages store their conversations under the paper's id."""
    found, order = {}, []
    rows = owned(AIassistant.objects.filter(active=True, papers__isnull=False), request).exclude(arxiv_id=current)
    for row in rows.order_by('-created').only('arxiv_id', 'query', 'created', 'papers', 'kind')[:300]:
        if row.arxiv_id == row.papers:
            continue
        if row.arxiv_id not in found:
            if len(order) == RECENT:
                continue
            found[row.arxiv_id] = {'room': row.arxiv_id, 'date': row.created, 'papers': row.papers.split(',')}
            order.append(row.arxiv_id)
        if row.kind == 'answer':
            found[row.arxiv_id]['question'] = row.query  # rows come newest first: this ends as the first question
    titles = dict(ArxivPaper.objects.filter(arxiv_id__in={c['papers'][0] for c in found.values()})
                  .values_list('arxiv_id', 'title'))
    names = dict(owned(ChatConversation.objects.filter(room__in=found), request).values_list('room', 'title'))
    for item in found.values():
        item['name'] = names.get(item['room'], '')
        item['title'] = ' '.join((titles.get(item['papers'][0]) or item['papers'][0]).split())
        item['ids'] = ','.join(item['papers'])
    return [found[room] for room in order]


def preset_papers(ids):
    """The papers the page opens with, as its script shows them; a paper we do not know yet is only its id
    (the script looks it up)."""
    known = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=ids).prefetch_related('authors')}
    papers = []
    for arxiv_id in ids:
        paper = known.get(arxiv_id)
        if paper is None or not paper.title:
            papers.append({'id': arxiv_id})
            continue
        authors = [a.name for a in paper.authors.all()[:2]]
        papers.append({'id': arxiv_id, 'title': ' '.join(paper.title.split()), 'label': sources.label(arxiv_id),
                       'author': authors[0] + (' et al.' if len(authors) > 1 else '') if authors else '',
                       'published': paper.published_arxiv.isoformat() if paper.published_arxiv else ''})
    return papers


def page(request):
    room = request.GET.get('c', '')
    room = room if ROOM_RE.match(room) else ''
    history, papers = [], []
    if room:
        history = list(owned(AIassistant.objects.filter(arxiv_id=room, active=True), request).order_by('created'))
        last = next((row.papers for row in reversed(history) if row.papers), '')
        papers = last.split(',') if last else []
    asked = [i.strip().replace('/', '--') for i in request.GET.get('ids', '').split(',') if i.strip()]
    asked = [i for i in asked if ID_RE.match(i)][:assistant.MAX_PAPERS]
    return render(request, 'summarizer/chat.html', {
        'guided': request.GET.get('tour') == '1',
        'guided_domain': getting_started.domain(request.GET.get('tour_domain', '')),
        'room': room, 'chathistory': history, 'preset': preset_papers(list(dict.fromkeys(asked or papers))),
        'conversations': recent_conversations(request, room), 'max_nb_paper': assistant.MAX_PAPERS,
        'table_columns': [str(label) for _, label in TABLE_COLUMNS],
        'table_allowed': access.has_tool(request.user, 'collections'),  # Premium, like the collections' tables
        'quota_left': quota_left(request),
        'share': current_share(request, room),
    })


def quota_left(request):
    """Questions left today, None when there is no limit to show (staff)."""
    if request.user.is_superuser:
        return None
    return max(access.chat_quota_left(request.user, access.hash_ip(access.client_ip(request))), 0)


def quota(request):
    """What the page shows under the question box after each answer."""
    return JsonResponse({'left': quota_left(request)})


def selected_ids(request, limit=assistant.MAX_PAPERS):
    ids = [i.strip().replace('/', '--') for i in request.GET.get('ids', '').split(',') if i.strip()]
    if not ids or len(ids) > limit or not all(ID_RE.match(i) for i in ids):
        raise Http404
    return list(dict.fromkeys(ids))


def supplied(request):
    """What the assistant reads of each selected paper, before the first question: the page shows it."""
    if access.throttled('chat-sources', request, 60):
        return JsonResponse({'status': 'busy'}, status=429)
    ids = selected_ids(request)
    try:
        _, papers = assistant.build_sources(ids)
    except assistant.AssistantError:
        papers = []
    return JsonResponse(assistant.supplied_summary(papers, ids))


def export_selection(request, fmt):
    """BibTeX, RIS or Markdown of the selected papers."""
    if fmt not in ('bib', 'ris', 'md'):
        raise Http404
    metas = [m for m in (exports.paper_meta(i) for i in selected_ids(request)) if m]
    if not metas:
        raise Http404
    return exports.export_many(metas, fmt, 'papers', get_language())


def reference(title, year, authors, ref, also=()):
    """also: the paper's other identifiers (its DOI when ref is its arXiv id...), so that one work cited under two
    of them is counted once (collection_tools.common_references)."""
    return {'title': ' '.join(title.split())[:250], 'year': year or '',
            'authors': ', '.join(authors[:3]) + (' et al.' if len(authors) > 3 else ''), 'ref': ref,
            'also': [i for i in also if i and i != ref]}


def s2_references(arxiv_id):
    """The references Semantic Scholar knows, None when it does not answer (logged: a 403 is a wrong key,
    a 429 the limit of calls without one)."""
    headers = {'x-api-key': settings.S2_API_KEY} if getattr(settings, 'S2_API_KEY', '') else {}
    for wait in (S2_RETRY_SECONDS, None):  # a key allows about one call a second, shared with the citation alerts
        try:
            response = requests.get(S2_REFERENCES % citations.s2_ref(citations.base_id(arxiv_id) or arxiv_id),
                                    params={'fields': 'title,year,authors,externalIds', 'limit': 500},
                                    headers=headers, timeout=20)
        except requests.RequestException:
            logger.warning('Semantic Scholar unreachable for the references of %s', arxiv_id)
            return None
        if response.status_code != 429 or wait is None:
            break
        time.sleep(wait)
    if response.status_code != 200:
        logger.warning('Semantic Scholar answered %s for the references of %s (key set: %s)',
                       response.status_code, arxiv_id, bool(headers))
        return None
    found = []
    for row in response.json().get('data') or []:
        paper = row.get('citedPaper') or {}
        ids = paper.get('externalIds') or {}
        ref = ids.get('ArXiv') or ids.get('DOI') or ''
        if ref and paper.get('title'):
            found.append(reference(paper['title'], paper.get('year'), [a.get('name', '') for a in paper.get('authors') or []], ref,
                                   (ids.get('ArXiv') or '', ids.get('DOI') or '')))
    return found


def openalex_references(arxiv_id):
    """The references OpenAlex knows (fewer for preprints than Semantic Scholar), None when it does not answer."""
    if sources.is_external(arxiv_id):
        doi = sources.key_doi(arxiv_id) if arxiv_id.startswith('doi_') else ''
    else:
        doi = '10.48550/arXiv.%s' % re.sub(r'v\d+$', '', arxiv_id.replace('--', '/'))
    if not doi:
        return None
    try:
        work = sources._json('%s/works/doi:%s' % (sources.OPENALEX, doi), sources._openalex_params({'select': 'referenced_works'}))
        cited = [w.rsplit('/', 1)[-1] for w in (work or {}).get('referenced_works') or []][:MAX_REFERENCES]
        works = []
        for start in range(0, len(cited), 50):
            body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
                'filter': 'openalex:' + '|'.join(cited[start:start + 50]), 'per-page': 50,
                'select': 'doi,title,publication_year,authorships,locations'}))
            works += (body or {}).get('results') or []
    except sources.SourceUnavailable:
        return None
    found = []
    for work in works:
        arxiv = next((re.sub(r'^https?://arxiv\.org/(?:abs|pdf)/', '', l.get('landing_page_url') or '')
                      for l in work.get('locations') or [] if 'arxiv.org/' in (l.get('landing_page_url') or '')), '')
        ref = arxiv or (work.get('doi') or '').replace('https://doi.org/', '')
        if ref and work.get('title'):
            found.append(reference(work['title'], work.get('publication_year'),
                                   [(a.get('author') or {}).get('display_name', '') for a in work.get('authorships') or []], ref,
                                   (arxiv, (work.get('doi') or '').replace('https://doi.org/', ''))))
    return found


def fetch_references(arxiv_id):
    """[{title, year, authors, ref}] of the papers a paper cites that have an arXiv id or a DOI: Semantic Scholar,
    else OpenAlex; None when neither answers."""
    found = s2_references(arxiv_id)
    if not found:
        fallback = openalex_references(arxiv_id)
        found = fallback if fallback is not None else found  # an empty list: 'none found', not 'unavailable'
    if found is None:
        return None
    # a reference cannot be younger than the paper citing it (OpenAlex sometimes merges records)
    published = ArxivPaper.objects.filter(arxiv_id=arxiv_id).values_list('published_arxiv', flat=True).first()
    if published:
        found = [r for r in found if not r['year'] or int(r['year']) <= published.year + 1]
    return found[:MAX_REFERENCES]


def references(request, arxiv_id):
    """The references of a selected paper that the reader can add to the conversation in one click."""
    if not ID_RE.match(arxiv_id):
        raise Http404
    key = 'chat-refs:%s' % arxiv_id
    found = cache.get(key)
    if found is None:
        if access.throttled('chat-references', request, 10):
            return JsonResponse({'status': 'busy'}, status=429)
        found = fetch_references(arxiv_id)
        if found is None:
            return JsonResponse({'status': 'unavailable'})
        cache.set(key, found, REFERENCES_SECONDS)
    return JsonResponse({'status': 'done', 'references': found})



@require_POST
def share(request):
    """Share the visitor's conversation as it is now: the link of its public page (the same link when shared
    before, now showing the exchanges since)."""
    room = request.POST.get('room', '')  # a chat page room, or the paper of a paper page's conversation
    if not (ROOM_RE.match(room) or ID_RE.match(room)) or not owned(AIassistant.objects.filter(arxiv_id=room, active=True), request).exists():
        raise Http404
    link = owned(ChatShare.objects.filter(room=room, active=True), request).first()
    if link is None:
        link = ChatShare(room=room, token=secrets.token_urlsafe(16),
                         user=request.user if request.user.is_authenticated else None,
                         ip_address=None if request.user.is_authenticated else access.hash_ip(access.client_ip(request)))
    link.upto = timezone.now()
    link.expires = link.upto + datetime.timedelta(days=SHARE_DAYS)
    link.save()
    return JsonResponse({'url': request.build_absolute_uri('/%s/chat/shared/%s/' % (get_language(), link.token)),
                         'expires': link.expires.date().isoformat(), 'timing': result_share.timing(link.expires)})


@require_POST
def unshare(request, token):
    link = get_object_or_404(owned(ChatShare.objects.filter(active=True), request), token=token)
    link.active = False
    link.save(update_fields=['active'])
    return JsonResponse({'status': 'done'})


def shared(request, token):
    """The public page of a shared conversation: its questions (never who asked), answers with their sources,
    tables, and the papers to go on with."""
    link = get_object_or_404(ChatShare, token=token, active=True)
    if link.expires and link.expires < timezone.now():
        raise Http404
    rows = AIassistant.objects.filter(arxiv_id=link.room, active=True, created__lte=link.upto)
    rows = rows.filter(user=link.user) if link.user_id else rows.filter(user=None, ip_address=link.ip_address)
    rows = list(rows.order_by('created'))
    if not rows:
        raise Http404
    ids = []
    for row in rows:
        ids += [i for i in (row.papers or '').split(',') if i and i not in ids]
    first = next((row for row in rows if row.kind == 'answer' and row.response), None)
    preview = re.sub(r'[*_`#>|$\\\[\]]', '', first.response if first else '')
    return render(request, 'summarizer/chat_shared.html', {
        'link': link, 'chathistory': rows, 'papers': preset_papers(ids[:assistant.MAX_PAPERS * 3]),
        'continue_ids': ','.join(ids[-assistant.MAX_PAPERS:]), 'first_question': rows[0].query,
        'preview': ' '.join(preview.split())[:300],  # what a link preview (Open Graph) shows
    })


def owned_room(request, room):
    """The visitor's exchanges of a chat page conversation (404 when they have none there)."""
    if not ROOM_RE.match(room):
        raise Http404
    rows = owned(AIassistant.objects.filter(arxiv_id=room, active=True), request)
    if not rows.exists():
        raise Http404
    return rows


@require_POST
def delete_conversation(request, room):
    """Delete a conversation of the visitor: its exchanges leave their history and its shared link stops."""
    owned_room(request, room).update(active=False)
    owned(ChatShare.objects.filter(room=room, active=True), request).update(active=False)
    owned(ChatConversation.objects.filter(room=room), request).delete()
    return JsonResponse({'status': 'done'})


@require_POST
def rename_conversation(request, room):
    """Name a conversation of the visitor; an empty name goes back to its first question."""
    owned_room(request, room)
    title = ' '.join(request.POST.get('title', '').split())[:MAX_NAME_CHARS]
    names = owned(ChatConversation.objects.filter(room=room), request)
    if not title:
        names.delete()
    elif not names.update(title=title):
        ChatConversation.objects.create(room=room, title=title,
                                        user=request.user if request.user.is_authenticated else None,
                                        ip_address=None if request.user.is_authenticated else access.hash_ip(access.client_ip(request)))
    return JsonResponse({'status': 'done', 'title': title})


def current_share(request, room):
    """The owner's last active link, including an expired one they can renew."""
    return owned(ChatShare.objects.filter(room=room, active=True), request).first() if room else None
