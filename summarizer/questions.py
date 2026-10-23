"""Open questions of a collection (Premium): what the papers leave open, gathered and grouped, with who works on it.

For each paper, the model gets the sections where authors say what is left to
do (limitations, discussion, future work, conclusions) when the license lets us
read the paper, and its abstract and our summary otherwise. It gathers the open
questions of the whole collection, groups the ones several papers share, says
which papers raise each one, and suggests projects that would answer them.
Written once per set of papers and language, like the comparison table.
"""
import re

from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from django.utils.translation import get_language
from django.views.decorators.http import require_POST

from . import access, generated, library, paper_source, sources
from .models import ArxivPaper, CollectionPaper

SECTION_RE = re.compile(r'^##\s*(?:[\dIVX.]+\s*)?.*(limitation|future|outlook|discussion|conclusion|open (?:question|problem)|'
                        r'caveat|perspective|summary)', re.I | re.M)
SECTION_CHARS = 2500
PAPER_CHARS = 5000
KINDS = ('open_problem', 'limitation', 'disagreement', 'untested')


def closing_sections(text):
    """The sections of a paper's text where authors say what is left open, or its last part when none is found."""
    found = []
    for match in SECTION_RE.finditer(text):
        found.append(text[match.start():match.start() + SECTION_CHARS])
    joined = '\n\n'.join(found)
    return (joined or text[-PAPER_CHARS:])[:PAPER_CHARS]


def paper_block(row):
    """What the model reads of one paper of the collection."""
    paper = ArxivPaper.objects.filter(arxiv_id=row.arxiv_id).first()
    if paper is not None and sources.is_open_license(paper.license):
        text = paper_source.paper_text(row.arxiv_id)
        if text and len(text) > 3000:
            return '<paper id="%s">\nTitle: %s\nAbstract: %s\nSections on what is left open:\n%s\n</paper>' % (
                row.arxiv_id, row.title, paper.abstract or '', closing_sections(text))
    return library.paper_block(row)


def write(collection_pk, lang):
    def work(pk):
        generated.progress(pk, 'read')
        rows = list(CollectionPaper.objects.filter(collection__pk=collection_pk)[:library.COMPARE_MAX])
        blocks = '\n\n'.join(paper_block(r) for r in rows)
        generated.release_connection()
        generated.progress(pk, 'write')
        answer = generated.chat_json(
            'From the %d papers above, gather the open questions of this field for a researcher looking for what to work '
            'on: problems the authors say remain open, limitations of their methods or data, results that disagree '
            'between papers, and assumptions nobody tested. Group the ones several papers share. Write in %s; no '
            'LaTeX, no markdown, never write paper IDs in the texts. Then suggest 3 research projects that would '
            'answer some of them. Answer with a JSON object {"questions": [{"question": "the open question in one '
            'sentence", "kind": "open_problem|limitation|disagreement|untested", "detail": "why it is open, 1 or 2 '
            'sentences", "papers": ["ids of the papers that raise it"], "next_step": "a concrete first step to address '
            'it"}], "projects": [{"title": "...", "idea": "2 or 3 sentences", "papers": ["ids it builds on"]}]}.'
            % (len(rows), generated.language_name(lang)), paid=True, max_tokens=6000, system=blocks)
        known = {r.arxiv_id: {'arxiv_id': r.arxiv_id, 'title': r.title, 'authors': r.authors} for r in rows}

        def papers(ids):
            return [known[i] for i in dict.fromkeys(ids if isinstance(ids, list) else []) if i in known]
        questions = [{'question': str(q.get('question', '')).strip(),
                      'kind': q.get('kind') if q.get('kind') in KINDS else 'open_problem',
                      'detail': str(q.get('detail', '')).strip(), 'next_step': str(q.get('next_step', '')).strip(),
                      'papers': papers(q.get('papers'))}
                     for q in answer.get('questions', []) if isinstance(q, dict) and str(q.get('question', '')).strip()]
        if not questions:
            raise generated.GenerationError('no open question')
        questions.sort(key=lambda q: -len(q['papers']))  # the questions several papers share first
        projects = [{'title': str(p.get('title', '')).strip(), 'idea': str(p.get('idea', '')).strip(),
                     'papers': papers(p.get('papers'))}
                    for p in answer.get('projects', []) if isinstance(p, dict) and str(p.get('title', '')).strip()][:5]
        return {'questions': questions[:20], 'projects': projects}
    return work


@login_required
@require_POST
def questions_start(request, pk):
    collection = library._mine(request, pk)
    if not access.has_tool(request.user, 'collections'):
        return JsonResponse({'status': 'premium'}, status=403)
    if collection.papers.count() < 2:
        return JsonResponse({'status': 'too_few'})
    lang = get_language()
    row = generated.start('questions', library.comparison_key(collection), lang, write(collection.pk, lang),
                          user=request.user, force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})
