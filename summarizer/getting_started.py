"""A ready, openly licensed example. Visiting never starts generation or changes an account."""
from django.shortcuts import render
from django.urls import reverse
from django.utils.translation import get_language
from django.views.decorators.http import require_GET

from . import sources, summary_text
from .models import ArxivPaper, PaperText, SummaryPaper

EXAMPLE = 'doi_10.1038--s41550-026-03004-3'


@require_GET
def page(request):
    lang = 'fr' if get_language() == 'fr' else 'en'
    paper = ArxivPaper.objects.filter(arxiv_id=EXAMPLE).prefetch_related('authors').first()
    summary = SummaryPaper.objects.filter(paper=paper, lang=lang).order_by('-updated').first() if paper else None
    text = PaperText.objects.filter(arxiv_id=EXAMPLE, source__in=['html', 'xml', 'pdf']).first()
    ready = bool(paper and sources.is_open_license(paper.license) and text and text.text.strip()
                 and summary and (summary.lay_summary or '').strip())
    questions = ([
        'Quel est le résultat principal de cet article ? Explique-le sans jargon et indique les passages qui le soutiennent.',
        'Explique la figure 1 : que représentent les axes et quelles conclusions peut-on en tirer ?',
        'Quelles sont les limites de cette étude ? Distingue les observations des interprétations des auteurs.',
    ] if lang == 'fr' else [
        'What is the main result of this paper? Explain it without jargon and show the passages that support it.',
        'Explain Figure 1: what do the axes represent and what conclusions can we draw?',
        'What are the limitations of this study? Distinguish observations from the authors’ interpretations.',
    ])
    return render(request, 'summarizer/getting_started.html', {
        'example_ready': ready, 'paper': paper, 'questions': questions,
        'summary_html': summary_text.lay_html(summary.lay_summary) if ready else '',
        'figure_paper': EXAMPLE, 'figure_chat_url': reverse('chat') + '?ids=' + EXAMPLE + '&tour=1',
    })
