"""Open questions of a collection (Premium): what the papers leave open, gathered and grouped, with who works on it.

For each paper, the model gets the sections where authors say what is left to
do (limitations, discussion, future work, conclusions) when the license lets us
read the paper, and its abstract and our summary otherwise. It gathers the open
questions of the whole collection, groups the ones several papers share, says
which papers raise each one, and suggests projects that would answer them.
Written once per set of papers and language, like the comparison table.
"""
import json
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
        prompt = (
            'From the %d papers above, gather the open questions of this field for a researcher looking for what to work '
            'on: problems the authors say remain open, limitations of their methods or data, results that disagree '
            'between papers, and assumptions nobody tested. Group the ones several papers share. Return at most '
            '12 questions and 3 projects. Only include questions supported by the supplied papers; if no supported '
            'question can be identified, return {"questions": [], "projects": []}. Never invent one just to fill '
            'the list. Keep the JSON field names exactly as specified, in English. Write the values in %s; no '
            'LaTeX, no markdown, never write paper IDs in the texts. Then suggest 3 research projects that would '
            'answer some of them. Answer with a JSON object {"questions": [{"question": "the open question in one '
            'sentence", "kind": "open_problem|limitation|disagreement|untested", "detail": "why it is open, 1 or 2 '
            'sentences", "papers": ["ids of the papers that raise it"], "next_step": "a concrete first step to address '
            'it"}], "projects": [{"title": "...", "idea": "2 or 3 sentences", "papers": ["ids it builds on"]}]}.'
            % (len(rows), generated.language_name(lang)))
        # Retry malformed output once; an honest empty list is a valid outcome.
        for attempt in range(2):
            try:
                instruction = prompt + (' Your previous response had an invalid structure. Return exactly an object '
                    'with arrays named questions and projects; use empty arrays when nothing is supported.' if attempt else '')
                answer = generated.chat_json(instruction, paid=True, max_tokens=6000, system=blocks)
                if (not isinstance(answer.get('questions'), list)
                        or not isinstance(answer.get('projects', []), list)):
                    raise ValueError('invalid questions structure')
                break
            except (json.JSONDecodeError, ValueError):
                if attempt:
                    raise generated.GenerationError('invalid open questions response') from None
        known = {r.arxiv_id: {'arxiv_id': r.arxiv_id, 'title': r.title, 'authors': r.authors} for r in rows}

        def papers(ids):
            safe_ids = [i for i in ids if isinstance(i, str)] if isinstance(ids, list) else []
            return [known[i] for i in dict.fromkeys(safe_ids) if i in known]

        def text(value):
            return value.strip() if isinstance(value, str) else ''

        questions = [{'question': text(q.get('question')),
                      'kind': q.get('kind') if q.get('kind') in KINDS else 'open_problem',
                      'detail': text(q.get('detail')), 'next_step': text(q.get('next_step')),
                      'papers': papers(q.get('papers'))}
                     for q in answer['questions'] if isinstance(q, dict) and text(q.get('question'))]
        questions = [q for q in questions if q['papers']]
        questions.sort(key=lambda q: -len(q['papers']))  # the questions several papers share first
        projects = [{'title': text(p.get('title')), 'idea': text(p.get('idea')),
                     'papers': papers(p.get('papers'))}
                    for p in answer.get('projects', []) if isinstance(p, dict) and text(p.get('title'))][:5]
        projects = [p for p in projects if p['papers']] if questions else []
        return {'questions': questions[:20], 'projects': projects, 'empty': not questions,
                'papers_read': len(rows)}
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
