"""The chat page: conversations with up to ten papers at once.

A conversation is a room (a random id the page keeps in its URL, ?c=) holding the questions, answers and
comparison tables of one reader; opening the URL again shows it to that reader only (account, else IP), with
the papers it was last about. Besides the page, the endpoints the page script calls: what the assistant will
read of each selected paper, the references of a paper to add one in a click, and the BibTeX, RIS or
Markdown of the selection.
"""
import re

import requests
from django.conf import settings
from django.core.cache import cache
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, assistant, citations, exports, sources
from .models import AIassistant, ArxivPaper

ROOM_RE = re.compile(r'^[A-Za-z0-9_-]{12,64}$')
ID_RE = re.compile(r'^[\w.\-/]{1,100}$')
RECENT = 8  # conversations listed on the page
S2_REFERENCES = 'https://api.semanticscholar.org/graph/v1/paper/%s/references'
MAX_REFERENCES = 100
REFERENCES_SECONDS = 24 * 3600
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
    for item in found.values():
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
        'room': room, 'chathistory': history, 'preset': preset_papers(list(dict.fromkeys(asked or papers))),
        'conversations': recent_conversations(request, room), 'max_nb_paper': assistant.MAX_PAPERS,
        'table_columns': [str(label) for _, label in TABLE_COLUMNS],
    })


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


def fetch_references(arxiv_id):
    """[{title, year, authors, ref}] of the papers a paper cites that have an arXiv id or a DOI, from Semantic
    Scholar; None when it is unreachable or does not know the paper."""
    headers = {'x-api-key': settings.S2_API_KEY} if getattr(settings, 'S2_API_KEY', '') else {}
    try:
        response = requests.get(S2_REFERENCES % citations.s2_ref(citations.base_id(arxiv_id) or arxiv_id),
                                params={'fields': 'title,year,authors,externalIds', 'limit': 500},
                                headers=headers, timeout=20)
    except requests.RequestException:
        return None
    if response.status_code != 200:
        return None
    found = []
    for row in response.json().get('data') or []:
        paper = row.get('citedPaper') or {}
        ids = paper.get('externalIds') or {}
        ref = ids.get('ArXiv') or ids.get('DOI') or ''
        if not ref or not paper.get('title'):
            continue
        names = [a.get('name', '') for a in (paper.get('authors') or [])[:3]]
        found.append({'title': ' '.join(paper['title'].split())[:250], 'year': paper.get('year') or '',
                      'authors': ', '.join(names) + (' et al.' if len(paper.get('authors') or []) > 3 else ''),
                      'ref': ref})
        if len(found) == MAX_REFERENCES:
            break
    return found


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

