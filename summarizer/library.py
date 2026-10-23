"""Collections: folders of papers to compare, chat with and export.

Everyone with an account gets a few small collections; Premium gets many,
and the comparison table: one model call reads every paper of the collection
(abstract and our summary) and fills the same columns for each, then says
where they agree, where they disagree and what nobody has done yet.
"""
import csv
import hashlib
import io
import json
import re

import requests
from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils.translation import get_language, override
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, digest, exports, generated, sources
from .models import AnnouncedPaper, ArxivPaper, Collection, CollectionPaper, Generated, SummaryPaper

FREE = {'collections': 3, 'papers': 20}
PREMIUM = {'collections': 100, 'papers': 200}
COMPARE_MAX = 25  # papers read by one comparison
CHAT_MAX = 10  # papers the chat page takes at once
MAX_ADD = 50  # ids added at once (one arXiv query)
SUMMARY_CHARS = 4000
COMPARE_STEPS = [
    ('read', _('Reading the papers and our summaries')),
    ('compare', _('Filling the table and comparing the results')),
]
QUESTIONS_STEPS = [  # the steps of questions.write, kept here so this module does not import it
    ('read', _('Reading the limitations, discussions and future work of each paper')),
    ('write', _('An AI gathers the open questions and groups them')),
]
COLUMNS = [
    ('question', _('Question')),
    ('method', _('Method')),
    ('data', _('Data')),
    ('results', _('Main results')),
    ('limitations', _('Limitations')),
]


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


def add_paper(collection, arxiv_id, paid, found=None):
    """'added', 'already', 'full' or 'unknown'. found: what lookup_many already knows."""
    if collection.papers.filter(arxiv_id=arxiv_id).exists():
        return 'already'
    if collection.papers.count() >= limits(paid)['papers']:
        return 'full'
    found = found.get(arxiv_id) if found is not None else lookup(arxiv_id)
    if found is None:
        return 'unknown'
    CollectionPaper.objects.get_or_create(collection=collection, arxiv_id=arxiv_id,
                                          defaults={'title': found[0], 'authors': found[1]})
    collection.save(update_fields=['updated'])
    return 'added'


def comparison_key(collection):
    ids = sorted(collection.papers.values_list('arxiv_id', flat=True))[:COMPARE_MAX]
    return '%s:%s' % (collection.pk, hashlib.sha256(','.join(ids).encode()).hexdigest()[:16])


def paper_block(row):
    meta = exports.paper_meta(row.arxiv_id, row)
    summaries = SummaryPaper.objects.filter(paper__arxiv_id=row.arxiv_id).exclude(summary__isnull=True) \
        .exclude(summary='').exclude(summary__startswith='Error').values_list('summary', flat=True)
    summary = max(summaries, key=len, default='')[:SUMMARY_CHARS]
    return '<paper id="%s">\nTitle: %s\nAuthors: %s\nAbstract: %s\n%s</paper>' % (
        row.arxiv_id, meta['title'], ', '.join(meta['authors'][:6]), meta['abstract'],
        'Summary: %s\n' % summary if summary else '')


def comparison_missing_label(lang):
    with override(lang):
        return str(_('Not stated'))


def comparison_data(table, lang):
    """Translate exact missing-value markers without rewriting cached scientific content."""
    label = comparison_missing_label(lang)

    def cell(value):
        if isinstance(value, str) and re.fullmatch(r'not\s+stated[.!]?', value.strip(), re.I):
            return label
        return value

    result = dict(table)
    result['rows'] = [dict(row, **{code: cell(row.get(code, '')) for code, _ in COLUMNS})
                      for row in table.get('rows', [])]
    for field in ('agreements', 'disagreements', 'gaps'):
        if field in table:
            result[field] = [cell(value) for value in table[field]]
    return result


def write_comparison(collection_pk, lang):
    def work(pk):
        generated.progress(pk, 'read')
        rows = list(CollectionPaper.objects.filter(collection__pk=collection_pk)[:COMPARE_MAX])
        blocks = '\n\n'.join(paper_block(r) for r in rows)
        titles = {r.arxiv_id: r.title for r in rows}
        generated.progress(pk, 'compare')
        answer = generated.chat_json(
            'Compare the %d papers above for a researcher writing a literature review. Write in %s. For every paper, '
            'fill each column in one or two short sentences, with key numbers when the paper gives them; write '
            '%s when it does not. Then say where the papers agree, where they disagree or report '
            'different results, and which questions none of them addresses. No LaTeX, no markdown, never write paper '
            'IDs in the texts. Answer with a JSON object {"rows": [{"id": "paper id", "question": "...", '
            '"method": "...", "data": "...", "results": "...", "limitations": "..."}], "agreements": ["..."], '
            '"disagreements": ["..."], "gaps": ["..."]}.' % (len(rows), generated.language_name(lang), json.dumps(comparison_missing_label(lang), ensure_ascii=False)),
            paid=True, max_tokens=8000, system=blocks)
        table = []
        for item in answer.get('rows', []):
            if isinstance(item, dict) and item.get('id') in titles:
                table.append(dict({c: str(item.get(c, '')).strip() for c, _ in COLUMNS},
                                  arxiv_id=item['id'], title=titles[item['id']]))
        if not table:
            raise generated.GenerationError('empty comparison')
        lists = {k: [str(x).strip() for x in answer.get(k, []) if str(x).strip()][:8]
                 for k in ('agreements', 'disagreements', 'gaps')}
        return comparison_data(dict(lists, rows=table), lang)
    return work


# --- views -------------------------------------------------------------------

def _mine(request, pk):
    return get_object_or_404(Collection, pk=pk, user=request.user)


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
    return render(request, 'summarizer/collections.html', {
        'collections': collections, 'paid': paid, 'can_create': can_create, 'limits': limits(paid), 'premium': PREMIUM})


@login_required
def collection_page(request, pk):
    collection = _mine(request, pk)
    paid = access.has_tool(request.user, 'collections')
    message = None
    if request.method == 'POST':
        action = request.POST.get('action')
        if action == 'delete':
            collection.delete()
            return redirect('/%s/collections/' % get_language())
        if action == 'rename':
            collection.name = request.POST.get('name', '').strip()[:120] or collection.name
            collection.save(update_fields=['name', 'updated'])
        elif action == 'remove':
            collection.papers.filter(arxiv_id=request.POST.get('arxiv_id')).delete()
        elif action == 'note':
            collection.papers.filter(arxiv_id=request.POST.get('arxiv_id')).update(note=request.POST.get('note', '')[:2000])
        elif action == 'add':
            ids = []
            for text in re.split(r'[\s,;]+', request.POST.get('ids', '')):
                arxiv_id = canonical_id(text) if text else None
                if arxiv_id and arxiv_id not in ids:
                    ids.append(arxiv_id)
            ids = ids[:MAX_ADD]
            known = lookup_many(ids)
            added = [add_paper(collection, arxiv_id, paid, known) for arxiv_id in ids]
            message = 'full' if 'full' in added else 'unknown' if 'unknown' in added else 'added' if added else 'unknown'
            return redirect('/%s/collections/%s/?m=%s' % (get_language(), collection.pk, message))
        return redirect('/%s/collections/%s/' % (get_language(), collection.pk))
    papers = list(collection.papers.all())
    lang = get_language()
    comparison = Generated.objects.filter(kind='compare', key=comparison_key(collection), lang=lang).first() if paid else None
    questions = Generated.objects.filter(kind='questions', key=comparison_key(collection), lang=lang).first() if paid else None
    return render(request, 'summarizer/collection.html', {
        'collection': collection, 'papers': papers, 'paid': paid, 'limits': limits(paid),
        'comparison': comparison, 'table': comparison_data(comparison.data(), lang) if comparison and comparison.status == 'done' else None,
        'columns': COLUMNS, 'compare_max': COMPARE_MAX, 'compare_steps': COMPARE_STEPS, 'message': request.GET.get('m'),
        'chat_ids': ','.join(p.arxiv_id for p in papers[:CHAT_MAX]), 'chat_max': CHAT_MAX,
        'questions': questions, 'open': questions.data() if questions and questions.status == 'done' else None,
        'questions_steps': QUESTIONS_STEPS,
    })


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
    status = add_paper(collection, arxiv_id, paid)
    return JsonResponse({'status': status, 'collection': collection.pk, 'name': collection.name,
                         'url': '/%s/collections/%s/' % (get_language(), collection.pk)})


@login_required
@require_POST
def compare_start(request, pk):
    collection = _mine(request, pk)
    if not access.has_tool(request.user, 'collections'):
        return JsonResponse({'status': 'premium'}, status=403)
    if collection.papers.count() < 2:
        return JsonResponse({'status': 'too_few'})
    lang = get_language()
    row = generated.start('compare', comparison_key(collection), lang, write_comparison(collection.pk, lang),
                          user=request.user, force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})


@login_required
def collection_export(request, pk, fmt):
    collection = _mine(request, pk)
    lang = get_language()
    filename = re.sub(r'[^A-Za-z0-9_-]+', '-', collection.name).strip('-')[:60] or 'collection'
    if fmt == 'csv':
        row = Generated.objects.filter(kind='compare', key=comparison_key(collection), lang=lang, status='done').first()
        if row is None or not access.has_tool(request.user, 'collections'):
            raise Http404
        out = io.StringIO()
        writer = csv.writer(out)
        writer.writerow(['Identifier', 'Title'] + [str(label) for _, label in COLUMNS])
        for line in comparison_data(row.data(), lang)['rows']:
            writer.writerow([exports.bare_ref(line['arxiv_id']), line['title']] + [line[c] for c, _ in COLUMNS])
        response = HttpResponse('﻿' + out.getvalue(), content_type='text/csv; charset=utf-8')
        response['Content-Disposition'] = 'attachment; filename="%s.csv"' % filename
        return response
    metas = [m for m in (exports.paper_meta(p.arxiv_id, p) for p in collection.papers.all()) if m]
    return exports.export_many(metas, fmt, filename, lang)


def paper_collections(user, arxiv_id):
    """For the paper page: the user's collections, and whether each holds this paper."""
    if not user.is_authenticated:
        return []
    inside = set(CollectionPaper.objects.filter(collection__user=user, arxiv_id=arxiv_id).values_list('collection_id', flat=True))
    return [{'pk': c.pk, 'name': c.name, 'has': c.pk in inside} for c in Collection.objects.filter(user=user)]
