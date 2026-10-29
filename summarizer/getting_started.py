"""Curated openly licensed examples. A visit never generates content or changes an account."""
from django.shortcuts import render
from django.urls import reverse
from django.utils.translation import get_language
from django.utils.text import Truncator
from django.views.decorators.http import require_GET

from . import insight, keypoints, sources, summary_text
from .models import ArxivPaper, PaperText, SummaryPaper

EXAMPLES = {
    'astronomy': {
        'id': 'doi_10.1038--s41550-026-03004-3', 'fr': 'Astronomie', 'en': 'Astronomy', 'figure': 1,
        'question_fr': 'Explique la figure 1 : que représentent les axes et comment les profils des raies soutiennent-ils l’existence de deux anneaux de gaz ?',
        'question_en': 'Explain Figure 1: what do the axes represent and how do the emission-line profiles support two gas rings?',
        'task_fr': 'Dans la figure 1, comparez les profils à quatre pics et à deux pics. Que permettent-ils de déduire sur la structure du gaz ?',
        'task_en': 'In Figure 1, compare the quadruple-peaked and double-peaked profiles. What do they tell us about the gas structure?',
    },
    'biology': {
        'id': '2610.25061v1', 'fr': 'Biologie', 'en': 'Biology', 'figure': 2,
        'question_fr': 'Explique la figure 2 : comment l’activité cérébrale avant un signal est-elle reliée à la réponse comportementale ? Distingue association et causalité.',
        'question_en': 'Explain Figure 2: how is brain activity before a cue related to the behavioral response? Distinguish association from causation.',
        'task_fr': 'Dans la figure 2, repérez les essais avec et sans réponse comportementale. Comment les auteurs les relient-ils à l’activité cérébrale ?',
        'task_en': 'In Figure 2, identify trials with and without a behavioral response. How do the authors relate them to brain activity?',
    },
    'computing': {
        'id': '2610.27930v1', 'fr': 'Informatique', 'en': 'Computer science', 'figure': 1,
        'question_fr': 'Explique la figure 1 : les modèles d’apprentissage automatique sont-ils toujours plus rapides que les solveurs CPU ou GPU ? Précise les conditions du benchmark.',
        'question_en': 'Explain Figure 1: are machine-learning models always faster than CPU or GPU solvers? Specify the benchmark conditions.',
        'task_fr': 'Dans la figure 1, comparez les solveurs CPU, GPU et le modèle d’apprentissage automatique. Le résultat est-il le même à toutes les tailles de réseau ?',
        'task_en': 'In Figure 1, compare CPU and GPU solvers with the machine-learning model. Does the result hold at every grid size?',
    },
}
EXAMPLE = EXAMPLES['astronomy']['id']  # Original entry point remains the default.


def domain(value):
    return value if value in EXAMPLES else 'astronomy'


def reading_formats(paper, summary, lang):
    """The same stored texts and safe presentation helpers as the paper page."""
    from .utils import clean_blog_html
    labels = {
        'fr': {
            'summary': ('Résumé complet', 'La synthèse détaillée : contexte, démarche, résultats et limites.'),
            'plain': ('Résumé vulgarisé', 'Les idées du papier expliquées dans un langage plus accessible.'),
            'points': ('Points clés', 'Les points clés générés par l’IA pour une lecture rapide.'),
            'blog': ('Article de blog', 'Une présentation narrative générée par l’IA à partir du papier.'),
            'abstract': ('Abstract original', 'Le résumé rédigé par les auteurs, dans la langue de publication.'),
        },
        'en': {
            'summary': ('Full summary', 'The detailed synthesis: context, approach, findings and limitations.'),
            'plain': ('Plain-language summary', 'The paper’s ideas explained in more accessible language.'),
            'points': ('Key points', 'AI-generated key points for a quick read.'),
            'blog': ('Blog article', 'An AI-generated narrative introduction to the paper.'),
            'abstract': ('Original abstract', 'The authors’ own abstract, in the publication’s original language.'),
        },
    }[lang]
    blog = summary_text.without_invented_references(clean_blog_html(summary.blog),
        [a.name for a in paper.authors.all()], paper.journal_ref, paper.comments) if summary.blog else ''
    points = keypoints.stored_points(summary.notes)
    bodies = [
        ('summary', summary_text.lay_html(summary.summary)),
        ('plain', summary_text.lay_html(summary.lay_summary)),
        ('points', '<ul>' + ''.join('<li>%s</li>' % keypoints.point_html(p) for p in points) + '</ul>' if points else ''),
        ('blog', blog),
        ('abstract', summary_text.lay_html(paper.abstract)),
    ]
    return [{'key': key, 'label': labels[key][0], 'description': labels[key][1], 'html': body}
            for key, body in bodies if body.strip()]


@require_GET
def page(request):
    lang = 'fr' if get_language() == 'fr' else 'en'
    selected = domain(request.GET.get('domain', ''))
    example = EXAMPLES[selected]
    key = example['id']
    paper = ArxivPaper.objects.filter(arxiv_id=key).prefetch_related('authors').first()
    summary = SummaryPaper.objects.filter(paper=paper, lang=lang).order_by('-updated').first() if paper else None
    text = PaperText.objects.filter(arxiv_id=key, source__in=['html', 'xml', 'pdf']).first()
    ready = bool(paper and sources.is_open_license(paper.license) and text and text.text.strip()
                 and summary and (summary.summary or summary.lay_summary or '').strip())
    reading = insight.for_page(key, lang) if ready else None
    exercise = None
    if reading and not reading.get('abstract_only'):
        # Check against the current full text as well: cached findings can predate a source refresh.
        reading = dict(reading, findings=[dict(item) for item in reading['findings']],
                       limitations=[dict(item) for item in reading.get('limitations', [])])
        for item in reading['findings']:
            item['quote'] = insight.evidence(item.get('quote'), text.text)
        for item in reading['limitations']:
            item['quote'] = insight.evidence(item.get('quote'), text.text)
        exercise = next((item for item in reading['findings'][:3] if item.get('quote')), None)
        if exercise:
            exercise['exercise'] = True
    else:
        reading = None
    questions = ([
        'Quel est le résultat principal de cet article ? Explique-le sans jargon et indique les passages qui le soutiennent.',
        example['question_fr'],
        'Quelles sont les limites de cette étude ? Distingue les observations des interprétations des auteurs.',
    ] if lang == 'fr' else [
        'What is the main result of this paper? Explain it without jargon and show the passages that support it.',
        example['question_en'],
        'What are the limitations of this study? Distinguish observations from the authors’ interpretations.',
    ])
    source_label = ('Prépublication arXiv' if lang == 'fr' else 'arXiv preprint') if not sources.is_external(key) else 'DOI'
    return render(request, 'summarizer/getting_started.html', {
        'example_ready': ready, 'paper': paper, 'questions': questions, 'insight': reading,
        'evidence_exercise': exercise, 'domain': selected, 'source_label': source_label, 'is_preprint': not sources.is_external(key),
        'domains': [{'slug': slug, 'label': row[lang]} for slug, row in EXAMPLES.items()],
        'figure_number': example['figure'], 'figure_task': example['task_' + lang],
        'summary_html': summary_text.lay_html(Truncator(summary.summary or summary.lay_summary).words(180)) if ready else '',
        'reading_formats': reading_formats(paper, summary, lang) if ready else [],
        'figure_paper': key, 'figure_chat_url': reverse('chat') + '?ids=' + key + '&tour=1&tour_domain=' + selected,
    })
