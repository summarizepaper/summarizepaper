"""The comparison table of a collection (Premium): the same columns filled for every paper, each cell with the
passage of the paper that supports it, then where the papers agree, where they disagree and what none addresses.

What the model reads of a paper: when its license lets us read it, its abstract and the sections that give the
method, data and results (paper_source); otherwise its abstract, with our summary as context only. A cell keeps
its quote only when the quote is word for word in what the model read of the paper, never in our summary
(evidence.find); the page shows it with its section and, for arXiv HTML, a link that opens the passage. A cell
without a quote is shown as not checked against the paper, and numbers of a cell missing from its quote are named.
The owner can correct a cell by hand; the correction is marked and survives updates and rewrites. Each
agreement, disagreement or gap names the papers and columns it rests on, so the page shows the passages behind
it, or says that it rests on cells without a source.

The columns are a template's (general, clinical, machine learning, observation, theory) followed by the user's
own. The papers read are the collection's selected ones, COMPARE_MAX at most, in the collection's order; the
open questions (questions.py) and the chat read the same ones. Papers are read EXTRACT_BATCH at a time and a
table keeps its rows: when papers are added, only the new ones are read before the comparison is written again
from the rows. A table written for other papers or other columns stays on the page, marked out of date.
"""
import csv
import hashlib
import io
import json
import re

from django.contrib.auth.decorators import login_required
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect
from django.utils.translation import get_language, override
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, evidence, exports, generated, library, paper_source, sources
from .models import ArxivPaper, Collection, Generated, PaperText, SummaryPaper

COMPARE_MAX = 25  # papers read by one comparison (and by the open questions)
EXTRACT_BATCH = 5  # papers per model call
SUMMARY_CHARS = 4000
SECTION_CHARS = 1800
PAPER_CHARS = 9000
MAX_COLUMNS = 8
MAX_ITEMS = 8
SECTION_RE = re.compile(r'^#{1,6}\s*(?:[\dIVX.]+\s*)?[^\n]*(method|approach|data\b|dataset|sample|observation|'
                        r'experiment|setup|model|simulation|benchmark|evaluation|result|finding|discussion|conclusion|'
                        r'limitation|analysis|measurement|participant|design|outcome)', re.I | re.M)
STEPS = [
    ('read', _('Reading the key sections of each paper when its license allows it, its abstract otherwise')),
    ('extract', _('Filling the columns, with the passage of the paper behind each cell')),
    ('compare', _('Comparing the papers: agreements, disagreements, gaps')),
]
# (code, label, what the model looks for, in English)
TEMPLATES = {
    'general': [
        ('question', _('Question'), 'the research question or aim'),
        ('method', _('Method'), 'the method or approach'),
        ('data', _('Data'), 'the data, sample or material used'),
        ('results', _('Main results'), 'the main results, with key numbers'),
        ('limitations', _('Limitations'), 'the limitations the authors state'),
    ],
    'clinical': [
        ('population', _('Population'), 'the population or participants, inclusion criteria'),
        ('intervention', _('Intervention and comparator'), 'the intervention or exposure and what it is compared with'),
        ('design', _('Design and sample size'), 'the study design (randomized, cohort...) and the sample size'),
        ('outcomes', _('Outcomes and effect sizes'), 'the primary outcomes with effect sizes and confidence intervals'),
        ('bias', _('Risk of bias and limitations'), 'blinding, attrition, confounders and other limitations'),
    ],
    'ml': [
        ('task', _('Task'), 'the task addressed'),
        ('datasets', _('Datasets and benchmarks'), 'the datasets and benchmarks used'),
        ('model', _('Model or method'), 'the model or method proposed'),
        ('scores', _('Metrics and scores'), 'the metrics and the scores reached'),
        ('baselines', _('Baselines'), 'the baselines compared and by how much they are beaten'),
        ('limitations', _('Limitations and compute'), 'the limitations, compute and data needs'),
    ],
    'observation': [
        ('target', _('Object or system studied'), 'the object, system or phenomenon studied'),
        ('instrument', _('Instrument and data'), 'the instrument, facility and data set'),
        ('method', _('Analysis method'), 'the analysis method'),
        ('measurement', _('Key measurements'), 'the key measured values with their uncertainties'),
        ('systematics', _('Systematics and limitations'), 'the systematic errors and limitations'),
    ],
    'theory': [
        ('question', _('Question'), 'the question addressed'),
        ('assumptions', _('Framework and assumptions'), 'the theoretical framework and the assumptions made'),
        ('approach', _('Approach'), 'the analytical or numerical approach'),
        ('result', _('Main result'), 'the main result or prediction'),
        ('validity', _('Domain of validity'), 'where the result holds and what it leaves open'),
    ],
}
TEMPLATE_CHOICES = [
    ('general', _('General')), ('clinical', _('Clinical or biomedical study')), ('ml', _('Machine learning')),
    ('observation', _('Observation or experiment')), ('theory', _('Theory or modelling')), ('custom', _('Only my columns')),
]
BLANK_CELL = {'text': '', 'quote': '', 'section': '', 'link': '', 'numbers': []}


# --- columns -------------------------------------------------------------------

def custom_columns(collection):
    """The user's own columns: [{'label', 'hint'}]."""
    try:
        items = json.loads(collection.custom_columns or '[]')
    except ValueError:
        items = []
    return [{'label': str(i.get('label') or '')[:80], 'hint': str(i.get('hint') or '')[:200]}
            for i in items if isinstance(i, dict) and str(i.get('label') or '').strip()] if isinstance(items, list) else []


def parse_custom(text):
    """'Label: what to look for' lines -> [{'label', 'hint'}]."""
    found = []
    for line in str(text or '').splitlines():
        label, _sep, hint = line.partition(':')
        if label.strip():
            found.append({'label': ' '.join(label.split())[:80], 'hint': ' '.join(hint.split())[:200]})
    return found[:MAX_COLUMNS]


def columns(collection):
    """[(code, label, what the model looks for)]: the template's columns, then the user's; MAX_COLUMNS at most."""
    found = [] if collection.template == 'custom' else list(TEMPLATES.get(collection.template, TEMPLATES['general']))
    found += [('c%d' % i, c['label'], c['hint'] or c['label']) for i, c in enumerate(custom_columns(collection), 1)]
    return found[:MAX_COLUMNS] or list(TEMPLATES['general'])


def signature(collection):
    with override('en'):
        cols = [(code, str(label), hint) for code, label, hint in columns(collection)]
    return hashlib.sha256(json.dumps(cols).encode()).hexdigest()[:12]


# --- which papers, which result --------------------------------------------------

def analysed(collection):
    """The papers the table, the open questions and the chat read: the selected ones, in the collection's order."""
    return list(collection.papers.filter(selected=True)[:COMPARE_MAX])


def result_key(collection, kind):
    """A table depends on its columns; the open questions do not. The papers read are kept in the result."""
    if kind == 'compare':
        return '%s:v2:%s' % (collection.pk, signature(collection))
    return '%s:v2' % collection.pk


def shown_result(collection, kind, lang):
    """(the row of the current key or None, the row whose content the page shows or None). While the current
    result is not written, the latest one of the collection is shown, marked out of date."""
    exact = Generated.objects.filter(kind=kind, key=result_key(collection, kind), lang=lang).first()
    if exact is not None and exact.content:
        return exact, exact
    latest = (Generated.objects.filter(kind=kind, key__startswith='%s:' % collection.pk, lang=lang)
              .exclude(content='').order_by('-updated').first())
    return exact, latest


def read_ids(data):
    """The papers a result read: kept since the second version; the rows' papers for an older table."""
    if 'ids' in data:
        return list(data['ids'])
    return [r.get('arxiv_id') for r in data.get('rows', []) if r.get('arxiv_id')] or None


def staleness(collection, kind, row, papers):
    """None when the result shown read the papers selected now (and, for a table, with today's columns);
    else {'added', 'removed', 'columns'}."""
    if row is None:
        return None
    data = row.data()
    ids = read_ids(data)
    columns_changed = kind == 'compare' and row.key != result_key(collection, kind)
    if ids is None:
        return {'added': 0, 'removed': 0, 'columns': columns_changed} if columns_changed else None
    now = [p.arxiv_id for p in papers]
    added, removed = len(set(now) - set(ids)), len(set(ids) - set(now))
    if not (added or removed or columns_changed):
        return None
    return {'added': added, 'removed': removed, 'columns': columns_changed and not (added or removed)}


# --- what the model reads ----------------------------------------------------------

def best_summary(arxiv_id):
    summaries = SummaryPaper.objects.filter(paper__arxiv_id=arxiv_id).exclude(summary__isnull=True) \
        .exclude(summary='').exclude(summary__startswith='Error').values_list('summary', flat=True)
    return max(summaries, key=len, default='')[:SUMMARY_CHARS]


def summary_block(row):
    """Abstract and our summary of a paper of the collection (the open questions read this when they cannot
    read the paper)."""
    meta = exports.paper_meta(row.arxiv_id, row)
    summary = best_summary(row.arxiv_id)
    return '<paper id="%s">\nTitle: %s\nAuthors: %s\nAbstract: %s\n%s</paper>' % (
        row.arxiv_id, meta['title'], ', '.join(meta['authors'][:6]), meta['abstract'],
        'Summary: %s\n' % summary if summary else '')


def key_sections(text):
    """The sections of a paper's text that give its method, data and results (the start and end when none is found)."""
    found, end, total = [], 0, 0
    for match in SECTION_RE.finditer(text):
        if match.start() < end:
            continue  # a subsection of the part already taken
        part = text[match.start():match.start() + SECTION_CHARS]
        found.append(part)
        end, total = match.start() + len(part), total + len(part)
        if total >= PAPER_CHARS:
            break
    if not found:
        half = PAPER_CHARS // 2
        found = [text[:half], text[-half:]] if len(text) > PAPER_CHARS else [text]
    return '\n\n'.join(found)[:PAPER_CHARS]


def reading(row):
    """What the model reads of a paper: {arxiv_id, title, authors, text (what a quote may come from), source,
    summary (context, never quoted)}."""
    meta = exports.paper_meta(row.arxiv_id, row) or {'title': row.title, 'authors': [], 'abstract': ''}
    abstract = meta.get('abstract') or ''
    body, source = '', 'abstract'
    paper = ArxivPaper.objects.filter(arxiv_id=row.arxiv_id).first()
    if paper is not None and sources.is_open_license(paper.license):
        full = paper_source.paper_text(row.arxiv_id)
        if full and len(full) > 3000:
            body = key_sections(full)
            source = PaperText.objects.filter(arxiv_id=row.arxiv_id).values_list('source', flat=True).first() or 'text'
    return {
        'arxiv_id': row.arxiv_id, 'title': meta.get('title') or row.title,
        'authors': ', '.join(meta.get('authors', [])[:6]) or row.authors,
        'text': '## Abstract\n\n%s%s' % (abstract, '\n\n' + body if body else ''),
        'source': source, 'summary': '' if body else best_summary(row.arxiv_id),
    }


def block(paper):
    return '<paper id="%s">\nTitle: %s\nAuthors: %s\n<paper_text>\n%s\n</paper_text>\n%s</paper>' % (
        paper['arxiv_id'], paper['title'], paper['authors'], paper['text'],
        '<our_summary>\n%s\n</our_summary>\n' % paper['summary'] if paper['summary'] else '')


# --- writing the table --------------------------------------------------------------

def missing_label(lang):
    with override(lang):
        return str(_('Not stated'))


def is_missing(text):
    """An empty cell, or one saying the paper does not state it (in English or French)."""
    return not str(text or '').strip() or bool(re.fullmatch(r'not\s+stated[.!]?|non\s+(?:pr[ée]cis[ée]|indiqu[ée])[.!]?',
                                                            str(text).strip(), re.I))


def cell(raw, paper, missing):
    """A cell as the model wrote it, its quote kept only when it is in what the model read of the paper.
    missing: what an empty cell says, in the table's language."""
    if isinstance(raw, str):
        raw = {'text': raw}
    raw = raw if isinstance(raw, dict) else {}
    text = ' '.join(str(raw.get('text') or '').split())[:700]
    if is_missing(text):
        return dict(BLANK_CELL, text=missing)
    found = dict(BLANK_CELL, text=text)
    quote = str(raw.get('quote') or '')
    if quote:
        match = evidence.find(quote, paper['text'])
        if match:
            passage = ' '.join(paper['text'][match.start():match.end()].split())
            where = evidence.location(match, paper)
            section = where['caption'] or where['section']
            if where['paragraph']:
                section = '%s, ¶%d' % (section, where['paragraph']) if section else '¶%d' % where['paragraph']
            stated = evidence.numbers(text) - evidence.with_roundings(evidence.numbers(passage))
            found.update(quote=passage[:900], section=section, link=evidence.passage_link(passage, paper),
                         numbers=sorted(stated)[:5])
    return found


def extract(papers, cols, lang):
    """{arxiv_id: row} of one batch of papers: each column with its text and checked quote."""
    with override('en'):
        lines = '\n'.join('- "%s" (%s): %s' % (code, label, hint) for code, label, hint in cols)
    prompt = (
        'For each of the %d papers above, fill these columns for a researcher writing a literature review. Write the '
        'texts in %s.\n%s\nFor each cell give "text": one or two short sentences, with the key numbers and their '
        'uncertainties when the paper gives them, or %s when what you were given of the paper does not say it; and '
        '"quote": the passage of <paper_text> that states what the cell says (not a sentence about context, motivation '
        'or future work), copied word for word in the paper\'s language, 1 or 2 consecutive sentences, no ellipsis, '
        'or "" when the cell rests on <our_summary> or on nothing. Never quote <our_summary>. No LaTeX, no markdown, never write paper IDs in the texts. Answer with a JSON object '
        '{"rows": [{"id": "paper id", "cells": {"<column code>": {"text": "...", "quote": "..."}}}]}.'
        % (len(papers), generated.language_name(lang), lines, json.dumps(missing_label(lang), ensure_ascii=False)))
    answer = generated.chat_json(prompt, paid=True, max_tokens=8000, system='\n\n'.join(block(p) for p in papers))
    by_id = {p['arxiv_id']: p for p in papers}
    missing = missing_label(lang)
    rows = {}
    for item in answer.get('rows', []) if isinstance(answer.get('rows'), list) else []:
        if not isinstance(item, dict) or item.get('id') not in by_id:
            continue
        paper = by_id[item['id']]
        raw = item.get('cells') if isinstance(item.get('cells'), dict) else item
        rows[item['id']] = {'arxiv_id': paper['arxiv_id'], 'title': paper['title'], 'source': paper['source'],
                            'cells': {code: cell(raw.get(code), paper, missing) for code, _l, _h in cols}}
    return rows


def synthesize(rows, cols, lang):
    """Agreements, disagreements and gaps, from the table alone, each with the papers it rests on."""
    with override('en'):
        table = [dict({'id': r['arxiv_id'], 'title': r['title']},
                      **{str(label): r['cells'].get(code, {}).get('text', '') for code, label, _h in cols}) for r in rows]
    answer = generated.chat_json(
        'Above is a comparison table of %d papers, one JSON object per paper. For a researcher writing a literature '
        'review, say in %s where the papers agree, where they disagree or report different results (give the '
        'numbers), and which questions none of them addresses. Rest only on the table. Each item lists the ids of '
        'the papers it rests on (an empty list for a question nobody addresses) and the codes of the columns it '
        'rests on (%s). At most %d items per list. No LaTeX, no markdown, never write paper IDs in the texts. '
        'Answer with a JSON object {"agreements": [{"text": "...", "papers": ["id"], "columns": ["code"]}], '
        '"disagreements": [...], "gaps": [...]}.'
        % (len(rows), generated.language_name(lang), ', '.join(code for code, _l, _h in cols), MAX_ITEMS),
        paid=True, max_tokens=3000, system=json.dumps(table, ensure_ascii=False))
    known = {r['arxiv_id'] for r in rows}
    codes = [code for code, _l, _h in cols]
    lists = {}
    for name in ('agreements', 'disagreements', 'gaps'):
        items = []
        for item in answer.get(name, []) if isinstance(answer.get(name), list) else []:
            if isinstance(item, str):
                item = {'text': item}
            if not isinstance(item, dict) or not str(item.get('text') or '').strip():
                continue
            papers = item.get('papers') if isinstance(item.get('papers'), list) else []
            columns = item.get('columns') if isinstance(item.get('columns'), list) else []
            items.append({'text': ' '.join(str(item['text']).split()),
                          'papers': [p for p in dict.fromkeys(papers) if isinstance(p, str) and p in known],
                          'columns': [c for c in dict.fromkeys(columns) if c in codes]})
        lists[name] = items[:MAX_ITEMS]
    return lists


def write(collection_pk, lang, full=False):
    """The work of the table: read the papers it has no row for (all of them when full), then compare."""
    def work(pk):
        collection = Collection.objects.get(pk=collection_pk)
        papers = analysed(collection)
        cols = columns(collection)
        sig = signature(collection)
        previous = Generated.objects.get(pk=pk).data()
        same = previous.get('sig') == sig
        kept = {} if full else {r['arxiv_id']: r for r in previous.get('rows', []) if same and 'cells' in r}
        generated.progress(pk, 'read')
        todo = [reading(p) for p in papers if p.arxiv_id not in kept]
        generated.progress(pk, 'extract')
        new = {}
        for start in range(0, len(todo), EXTRACT_BATCH):
            new.update(extract(todo[start:start + EXTRACT_BATCH], cols, lang))
        rows = [r for r in (kept.get(p.arxiv_id) or new.get(p.arxiv_id) for p in papers) if r]
        if not rows or (todo and not new):
            raise generated.GenerationError('empty comparison')
        generated.progress(pk, 'compare')
        lists = synthesize(rows, cols, lang) if len(rows) > 1 else {'agreements': [], 'disagreements': [], 'gaps': []}
        # what the owner corrected stays corrected, read last: they may correct a cell while this runs
        latest = Generated.objects.get(pk=pk).data()
        if latest.get('sig') == sig:
            edited = {(r['arxiv_id'], code): c for r in latest.get('rows', [])
                      for code, c in (r.get('cells') or {}).items() if isinstance(c, dict) and c.get('edited')}
            for row in rows:
                for code in row['cells']:
                    row['cells'][code] = edited.get((row['arxiv_id'], code), row['cells'][code])
        return dict(lists, version=2, sig=sig, template=collection.template, rows=rows,
                    columns=[[code, label if code.startswith('c') else ''] for code, label, _h in cols],
                    ids=[p.arxiv_id for p in papers],
                    missing=[p.title or p.arxiv_id for p in papers if p.arxiv_id not in kept and p.arxiv_id not in new])
    return work


# --- showing it -------------------------------------------------------------------

def display(data, lang):
    """The stored table as the page, the shared page and the CSV show it; older tables (one text per column,
    lists of plain sentences) included."""
    missing = missing_label(lang)
    template = data.get('template') or 'general'
    labels = {code: label for code, label, _h in TEMPLATES.get(template, TEMPLATES['general'])}
    if data.get('columns'):
        cols = [(code, custom or labels.get(code, code)) for code, custom in data['columns']]
    else:
        cols = [(code, label) for code, label, _h in TEMPLATES['general']]
    rows, numbers = [], {}
    for index, row in enumerate(data.get('rows', []), 1):
        stored = row.get('cells') or {code: {'text': row.get(code, '')} for code, _l in cols}
        cells = []
        for code, _l in cols:
            value = dict(dict(BLANK_CELL, edited=False), **(stored.get(code) or {}))
            value['code'] = code
            value['missing'] = is_missing(value['text'])
            if value['missing']:
                value.update(text=missing, quote='')
            cells.append(value)
        numbers[row.get('arxiv_id')] = index
        rows.append({'n': index, 'arxiv_id': row.get('arxiv_id', ''), 'title': row.get('title', ''),
                     'source': row.get('source', ''), 'cells': cells})
    by_paper = {r['arxiv_id']: r for r in rows}
    labels_now = dict(cols)

    def items(name):
        """Each finding with its rows, and the passages behind it: the quoted cells of its papers in its columns.
        backed: every paper it names has one; None when the finding does not name columns (older tables)."""
        found = []
        for item in data.get(name, []):
            item = item if isinstance(item, dict) else {'text': str(item), 'papers': []}
            text = missing if is_missing(item.get('text', '')) else item.get('text', '')
            papers = [p for p in item.get('papers', []) if p in numbers]
            passages, backed = [], None
            if 'columns' in item and papers:
                quoted = set()
                for aid in papers:
                    for c in by_paper[aid]['cells']:
                        if c['code'] in item['columns'] and c['quote']:
                            passages.append({'n': numbers[aid], 'label': labels_now.get(c['code'], c['code']),
                                             'quote': c['quote'], 'section': c['section']})
                            quoted.add(aid)
                backed = len(quoted) == len(papers)
            found.append({'text': text, 'papers': sorted(numbers[p] for p in papers), 'passages': passages,
                          'backed': backed})
        return found

    filled = [c for r in rows for c in r['cells'] if not c['missing']]
    return {'columns': cols, 'rows': rows, 'agreements': items('agreements'), 'disagreements': items('disagreements'),
            'gaps': items('gaps'), 'missing': data.get('missing', []), 'version': data.get('version', 1),
            'quoted': sum(1 for c in filled if c['quote']), 'filled': len(filled),
            'full_text': sum(1 for r in rows if r['source'] and r['source'] != 'abstract')}


def csv_response(data, lang, filename):
    table = display(data, lang)
    out = io.StringIO()
    writer = csv.writer(out)
    header = ['Identifier', 'Title']
    for _code, label in table['columns']:
        header += [str(label), '%s (source)' % label]
    writer.writerow(header)
    for row in table['rows']:
        line = [exports.bare_ref(row['arxiv_id']), row['title']]
        for c in row['cells']:
            line += [c['text'], ('"%s" (%s)' % (c['quote'], c['section']) if c['section'] else '"%s"' % c['quote'])
                     if c['quote'] else '']
        writer.writerow(line)
    response = HttpResponse('﻿' + out.getvalue(), content_type='text/csv; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="%s.csv"' % filename
    return response


@login_required
@require_POST
def compare_start(request, pk):
    """Start (or follow) the table; mode 'update' reads only the papers it has no row for, 'again' all of them."""
    collection = library.editable(request, pk)
    if not access.has_tool(request.user, 'collections'):
        return JsonResponse({'status': 'premium'}, status=403)
    if len(analysed(collection)) < 2:
        return JsonResponse({'status': 'too_few'})
    mode = 'again' if request.POST.get('again') == '1' else request.POST.get('mode', '')
    lang = get_language()
    row = generated.start('compare', result_key(collection, 'compare'), lang, write(collection.pk, lang, full=mode == 'again'),
                          user=request.user, force=mode in ('update', 'again'))
    return JsonResponse({'status': row.status, 'id': row.pk})


@login_required
@require_POST
def edit_cell(request, pk):
    """The owner or an editor corrects a cell of a table of this collection. The correction is marked as such."""
    collection = library.editable(request, pk)
    if not access.has_tool(request.user, 'collections'):
        raise Http404
    row = get_object_or_404(Generated, pk=request.POST.get('row'), kind='compare', key__startswith='%s:' % collection.pk)
    data = row.data()
    arxiv_id, code = request.POST.get('arxiv_id', ''), request.POST.get('code', '')
    target = next((r for r in data.get('rows', []) if r.get('arxiv_id') == arxiv_id), None)
    if target is None:
        raise Http404
    if 'cells' not in target:  # an older table: one text per column
        target['cells'] = {c: {'text': target.pop(c, '')} for c in list(target) if c not in ('arxiv_id', 'title', 'id', 'source')}
    text = ' '.join(request.POST.get('text', '').split())[:700]
    previous = dict(BLANK_CELL, **(target['cells'].get(code) or {}))
    target['cells'][code] = dict(previous, text=text or missing_label(row.lang), edited=True, numbers=[])
    row.content = json.dumps(data)
    row.save(update_fields=['content'])
    number = data['rows'].index(target) + 1
    return redirect('/%s/collections/%s/#row-%d' % (get_language(), collection.pk, number))
