"""Journal club slides of a paper (Premium): a short deck to present it, its strengths and limits, and questions.

Written once per paper and language from what the license lets us read (see
assistant.build_context: the full text, or only the abstract). For an arXiv
paper under an open license (CC BY...), the figures of its HTML version are
offered to the model by number and caption, and shown from arXiv with their
credit; it picks the ones worth a slide. The deck is a page: presented full
screen with the arrow keys, or printed to PDF one slide per page.
"""
import urllib.parse
import copy
import hashlib
import json

import requests
from bs4 import BeautifulSoup
from django.http import Http404, JsonResponse, HttpResponse
from django.contrib.auth.models import User
from django.db import transaction
from django.views.decorators.cache import never_cache
from django.utils.translation import gettext
from django.shortcuts import render
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, generated, paper_source, sources
from .assistant import build_context
from .models import ArxivPaper, Generated

MAX_FIGURES = 12
MAX_SLIDES = 12
STEPS = [
    ('read', _('Reading the paper and its figures')),
    ('write', _('Writing the slides, the strengths and limits, and the questions')),
]

PROMPT = """You prepare a journal club presentation of the paper above for researchers of the field. Write in {language}.
Make 8 to 10 slides: context and question, what was done (method, data), the main results with their key numbers,
how it compares with earlier work, and the take-home message. Keep each bullet under 20 words; no LaTeX, no markdown.
{figures}Answer with a JSON object:
{{"slides": [{{"title": "...", "bullets": ["3 to 5 bullets"], "figure": null or a figure number, "notes": "2 or 3 sentences
to say while showing it"}}],
  "strengths": ["3 or 4 strengths"], "limitations": ["3 or 4 limitations or open doubts, including the ones the authors
do not state"], "questions": ["4 to 6 questions to start the discussion"]}}"""


def figures(arxiv_id):
    """[{'number', 'url', 'caption'}] of the figures of the arXiv HTML, [] when there is none."""
    try:
        response = requests.get(paper_source.ARXIV_HTML % paper_source.arxiv_ref(arxiv_id), timeout=60,
                                headers={'User-Agent': 'SummarizePaper (https://www.summarizepaper.com)'})
    except requests.RequestException:
        return []
    if response.status_code != 200 or '/html/' not in response.url:
        return []
    soup = BeautifulSoup(response.text, 'html.parser')
    found = []
    for figure in soup.select('figure.ltx_figure'):
        if figure.find_parent('figure'):
            continue  # a panel of a larger figure
        image = figure.find(['img', 'object'])
        source = image and (image.get('src') or image.get('data'))
        caption = figure.find('figcaption')
        if not source or caption is None:
            continue
        found.append({'number': len(found) + 1, 'url': urllib.parse.urljoin(response.url, source),
                      'caption': ' '.join(caption.get_text(' ').split())[:600]})
        if len(found) == MAX_FIGURES:
            break
    return found


def write(arxiv_id, lang):
    def work(pk):
        generated.progress(pk, 'read')
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        context = build_context([arxiv_id])
        shown = figures(arxiv_id) if not sources.is_external(arxiv_id) and sources.is_open_license(paper.license) else []
        generated.progress(pk, 'write')
        figure_list = ''
        if shown:
            figure_list = ('The paper has these figures; put one on a slide when it helps, by its number, at most once '
                           'each:\n%s\n' % '\n'.join('Figure %d: %s' % (f['number'], f['caption'][:300]) for f in shown))
        answer = generated.chat_json(PROMPT.format(language=generated.language_name(lang), figures=figure_list),
                                     paid=True, max_tokens=5000, system=context)
        by_number = {f['number']: f for f in shown}
        slides, used = [], set()
        for s in answer.get('slides', [])[:MAX_SLIDES]:
            if not isinstance(s, dict) or not str(s.get('title', '')).strip():
                continue
            number = s.get('figure') if isinstance(s.get('figure'), int) else None
            figure = by_number.get(number) if number not in used else None
            used.add(number)
            slides.append({'title': str(s['title']).strip(), 'figure': figure,
                           'bullets': [str(b).strip() for b in s.get('bullets', []) if str(b).strip()][:6],
                           'notes': str(s.get('notes', '')).strip()})
        if not slides:
            raise generated.GenerationError('no slides')
        lists = {k: [str(x).strip() for x in answer.get(k, []) if str(x).strip()][:6]
                 for k in ('strengths', 'limitations', 'questions')}
        return dict(lists, slides=slides, license=sources.license_label(paper.license or ''),
                    abstract_only='Only the title, authors and abstract are available' in context)
    return work


@never_cache
def slides_page(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if paper is None:
        raise Http404
    paid = access.is_paid(request.user)
    row = Generated.objects.filter(kind='slides', key=arxiv_id, lang=request.LANGUAGE_CODE).first() if paid else None
    deck = row.data() if row is not None and row.status == 'done' else None
    pages, revision = editable_pages(request.user, arxiv_id, request.LANGUAGE_CODE, paper, deck) if deck else ([], '')
    return render(request, 'summarizer/slides.html', {
        'paper': paper, 'arxiv_id': arxiv_id, 'paid': paid, 'row': row, 'steps': STEPS,
        'authors': ', '.join(a.name for a in paper.authors.all()[:6]) + (' et al.' if paper.authors.count() > 6 else ''),
        'deck': deck, 'pages': pages, 'deck_revision': revision})


@require_POST
def slides_start(request, arxiv_id):
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        raise Http404
    lang = request.LANGUAGE_CODE
    row = generated.start('slides', arxiv_id, lang, write(arxiv_id, lang), force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})


def edit_key(user, arxiv_id):
    return '%s:%s' % (user.pk, hashlib.sha256(arxiv_id.encode()).hexdigest())


def original_pages(paper, deck):
    authors = ', '.join(a.name for a in paper.authors.all()[:6]) + (' et al.' if paper.authors.count() > 6 else '')
    pages = [dict(kind='title', title=paper.title, bullets=[authors], notes='', figure=None)]
    pages += [dict(s, kind='content') for s in deck['slides']]
    pages += [dict(kind='columns', title=gettext('Strengths and limits'), bullets=deck['strengths'],
                   secondary=deck['limitations'], notes='', figure=None),
              dict(kind='discussion', title=gettext('For the discussion'), bullets=deck['questions'], notes='', figure=None)]
    for page in pages:
        page.setdefault('notes', '')
        page.update(layout='right', figure_size=50, text_size=30)
    return pages


def editable_pages(user, arxiv_id, lang, paper, deck):
    pages = original_pages(paper, deck)
    saved = Generated.objects.filter(kind='slides_edit', key=edit_key(user, arxiv_id), lang=lang, user=user).first()
    if saved:
        # Merge edits with trusted source figures. URLs, licensing and slide kind are never client-controlled.
        try:
            pages = validate_pages(saved.data()['pages'], pages)
        except (ValueError, KeyError, TypeError):
            pass  # An updated generated deck may have a different slide count.
    revision = hashlib.sha256(json.dumps(pages, sort_keys=True).encode()).hexdigest()
    return pages, revision


def validate_pages(value, originals):
    if not isinstance(value, list) or len(value) != len(originals):
        raise ValueError
    pages = copy.deepcopy(originals)
    def text(value, limit):
        if not isinstance(value, str) or len(value) > limit or '\x00' in value:
            raise ValueError
        return value.strip()
    for edit, page in zip(value, pages):
        if not isinstance(edit, dict):
            raise ValueError
        page['title'] = text(edit.get('title'), 300)
        if not page['title']:
            raise ValueError
        for field in ('bullets', 'secondary'):
            if field == 'secondary' and page['kind'] != 'columns':
                continue
            lines = edit.get(field, [])
            if not isinstance(lines, list) or len(lines) > 8:
                raise ValueError
            page[field] = [text(line, 600) for line in lines if text(line, 600)]
        page['notes'] = text(edit.get('notes', ''), 3000)
        page['layout'] = edit.get('layout', 'right')
        page['figure_size'] = edit.get('figure_size', 50)
        page['text_size'] = edit.get('text_size', 30)
        if (page['layout'] not in ('left', 'right', 'top', 'bottom') or
            type(page['figure_size']) is not int or not 25 <= page['figure_size'] <= 70 or
            page['text_size'] not in (24, 30, 36)):
            raise ValueError
    return pages


@never_cache
@require_POST
def slides_save(request, arxiv_id):
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    if len(request.body) > 120000:
        return JsonResponse({'status': 'invalid'}, status=400)
    try:
        payload = json.loads(request.body)
        with transaction.atomic():
            User.objects.select_for_update().get(pk=request.user.pk)
            paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
            row = Generated.objects.filter(kind='slides', key=arxiv_id, lang=request.LANGUAGE_CODE, status='done').first()
            if paper is None or row is None:
                raise Http404
            current, revision = editable_pages(request.user, arxiv_id, request.LANGUAGE_CODE, paper, row.data())
            if payload.get('revision') != revision:
                return JsonResponse({'status': 'conflict'}, status=409)
            pages = validate_pages(payload.get('pages'), original_pages(paper, row.data()))
            # Store only editable text/layout, without retaining client-supplied URLs or HTML.
            edits = [{k: v for k, v in p.items() if k in ('title', 'bullets', 'secondary', 'notes', 'layout', 'figure_size', 'text_size')} for p in pages]
            Generated.objects.update_or_create(kind='slides_edit', key=edit_key(request.user, arxiv_id), lang=request.LANGUAGE_CODE,
                defaults={'user': request.user, 'status': 'done', 'content': json.dumps({'pages': edits})})
            revision = hashlib.sha256(json.dumps(pages, sort_keys=True).encode()).hexdigest()
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({'status': 'invalid'}, status=400)
    return JsonResponse({'status': 'saved', 'revision': revision})


@never_cache
@require_POST
def slides_powerpoint(request, arxiv_id):
    from . import slides_pptx
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    if len(request.body) > 120000:
        return JsonResponse({'status': 'invalid'}, status=400)
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    row = Generated.objects.filter(kind='slides', key=arxiv_id, lang=request.LANGUAGE_CODE, status='done').first()
    if paper is None or row is None:
        raise Http404
    try:
        payload = json.loads(request.body)
        _, revision = editable_pages(request.user, arxiv_id, request.LANGUAGE_CODE, paper, row.data())
        if payload.get('revision') != revision:
            return JsonResponse({'status': 'conflict'}, status=409)
        pages = validate_pages(payload.get('pages'), original_pages(paper, row.data()))
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({'status': 'invalid'}, status=400)
    try:
        data = slides_pptx.render_pptx(pages, arxiv_id, row.data().get('license', ''), paper.title)
    except slides_pptx.ExportError as error:
        return JsonResponse({'status': 'unavailable', 'message': str(error)}, status=503)
    import re
    name = re.sub(r'[^a-zA-Z0-9._-]', '-', arxiv_id)[:80]
    response = HttpResponse(data, content_type=slides_pptx.MIME)
    response['Content-Disposition'] = 'attachment; filename="SummarizePaper-JournalClub-%s.pptx"' % name
    return response
