"""Journal club slides of a paper (Premium): a short deck to present it, its strengths and limits, and questions.

Written once per paper and language from what the license lets us read (see
assistant.build_context: the full text, or only the abstract). For an arXiv
paper under an open license (CC BY...), the figures of its HTML version are
offered to the model by number and caption, and shown from arXiv with their
credit; it picks the ones worth a slide. The deck is a page: presented full
screen with the arrow keys, or printed to PDF one slide per page.
"""
import urllib.parse

import requests
from bs4 import BeautifulSoup
from django.http import Http404, JsonResponse
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


def slides_page(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if paper is None:
        raise Http404
    paid = access.is_paid(request.user)
    row = Generated.objects.filter(kind='slides', key=arxiv_id, lang=request.LANGUAGE_CODE).first() if paid else None
    return render(request, 'summarizer/slides.html', {
        'paper': paper, 'arxiv_id': arxiv_id, 'paid': paid, 'row': row, 'steps': STEPS,
        'authors': ', '.join(a.name for a in paper.authors.all()[:6]) + (' et al.' if paper.authors.count() > 6 else ''),
        'deck': row.data() if row is not None and row.status == 'done' else None})


@require_POST
def slides_start(request, arxiv_id):
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        raise Http404
    lang = request.LANGUAGE_CODE
    row = generated.start('slides', arxiv_id, lang, write(arxiv_id, lang), force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})
