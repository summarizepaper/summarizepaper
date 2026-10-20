"""Getting papers out of SummarizePaper: BibTeX, RIS, Markdown, and 'Send to Zotero'.

BibTeX goes to LaTeX, RIS to Zotero, Mendeley and EndNote, Markdown to Notion
and Obsidian (paste or import the file). With Premium, 'Send to Zotero' writes
the paper and a note with its summary and key points straight into the user's
Zotero library through the Zotero web API.
"""
import json
import re

import requests
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.utils.html import escape
from django.utils.translation import get_language
from django.views.decorators.http import require_POST

from . import access, generated, keypoints, sources
from .models import AnnouncedPaper, ArxivPaper, CollectionPaper, SummaryPaper, ZoteroAccount

ZOTERO_API = 'https://api.zotero.org'


def bare_ref(arxiv_id):
    """'hep-th--9901001v2' -> 'hep-th/9901001'; for other sources the DOI (or HAL / PMC id)."""
    if sources.is_external(arxiv_id):
        return sources.label(arxiv_id).split(': ', 1)[-1]
    return re.sub(r'v\d+$', '', arxiv_id.replace('--', '/'))


def file_name(arxiv_id):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', bare_ref(arxiv_id))


def paper_meta(arxiv_id, fallback=None):
    """What the exports need about a paper, from our tables (None if we know nothing of it)."""
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if paper is not None:
        return {
            'arxiv_id': arxiv_id, 'title': ' '.join((paper.title or '').split()),
            'authors': [a.name for a in paper.authors.all()],
            'year': paper.published_arxiv.year if paper.published_arxiv else '',
            'date': paper.published_arxiv.isoformat() if paper.published_arxiv else '',
            'category': paper.category or '', 'abstract': ' '.join((paper.abstract or '').split()),
            'doi': paper.link_doi.split('doi.org/')[-1] if paper.link_doi and 'doi.org' in paper.link_doi else '',
            'journal': paper.journal_ref or '',
        }
    announced = AnnouncedPaper.objects.filter(arxiv_id=arxiv_id).first()
    if announced is not None:
        return {
            'arxiv_id': arxiv_id, 'title': announced.title, 'authors': json.loads(announced.authors or '[]'),
            'year': announced.announced.year, 'date': announced.announced.isoformat(),
            'category': announced.categories.split(',')[0], 'abstract': announced.abstract, 'doi': '', 'journal': '',
        }
    if fallback is not None:
        return {'arxiv_id': arxiv_id, 'title': fallback.title, 'authors': [a for a in fallback.authors.split(', ') if a],
                'year': '', 'date': '', 'category': '', 'abstract': '', 'doi': '', 'journal': ''}
    return None


def is_preprint(meta):
    return 'preprint' in (meta['journal'] or '').lower() or not meta['journal']


def bibtex_entry(meta):
    ref = bare_ref(meta['arxiv_id'])
    first_word = next((w for w in re.findall(r'[A-Za-z]+', meta['title']) if len(w) > 3), 'paper')
    last_name = re.sub(r'[^A-Za-z]', '', meta['authors'][0].split()[-1]) if meta['authors'] else 'paper'
    cite_key = '%s%s%s' % (last_name.lower(), meta['year'], first_word.lower())
    if sources.is_external(meta['arxiv_id']):
        lines = ['@%s{%s,' % ('misc' if is_preprint(meta) else 'article', cite_key),
                 '  title={%s},' % meta['title'], '  author={%s},' % ' and '.join(meta['authors']),
                 '  year={%s},' % meta['year']]
        if meta['journal'] and not is_preprint(meta):
            lines.append('  journal={%s},' % meta['journal'])
        elif meta['journal']:
            lines.append('  howpublished={%s},' % meta['journal'])
        if meta['doi']:
            lines.append('  doi={%s},' % meta['doi'])
        lines.append('  url={%s}' % sources.ref_url(meta['arxiv_id']))
        lines.append('}')
        return '\n'.join(lines) + '\n'
    lines = [
        '@misc{%s,' % cite_key,
        '  title={%s},' % meta['title'],
        '  author={%s},' % ' and '.join(meta['authors']),
        '  year={%s},' % meta['year'],
        '  eprint={%s},' % ref,
        '  archivePrefix={arXiv},',
    ]
    if meta['category']:
        lines.append('  primaryClass={%s},' % meta['category'])
    if meta['doi']:
        lines.append('  doi={%s},' % meta['doi'])
    lines.append('  url={https://arxiv.org/abs/%s}' % ref)
    lines.append('}')
    return '\n'.join(lines) + '\n'


def ris_entry(meta):
    ref = bare_ref(meta['arxiv_id'])
    external = sources.is_external(meta['arxiv_id'])
    lines = ['TY  - %s' % ('JOUR' if external and not is_preprint(meta) else 'PREP'), 'TI  - %s' % meta['title']]
    lines += ['AU  - %s' % a for a in meta['authors']]
    if meta['year']:
        lines.append('PY  - %s' % meta['year'])
    if meta['date']:
        lines.append('DA  - %s' % meta['date'].replace('-', '/'))
    if meta['abstract']:
        lines.append('AB  - %s' % meta['abstract'])
    if meta['doi']:
        lines.append('DO  - %s' % meta['doi'])
    if external:
        if meta['journal']:
            lines.append('%s  - %s' % ('T2' if not is_preprint(meta) else 'PB', meta['journal']))
        lines += ['UR  - %s' % sources.ref_url(meta['arxiv_id']), 'ER  - ']
    else:
        lines += ['PB  - arXiv', 'AN  - arXiv:%s' % ref, 'UR  - https://arxiv.org/abs/%s' % ref, 'ER  - ']
    return '\n'.join(lines) + '\n'


def summary_of(arxiv_id, lang):
    row = SummaryPaper.objects.filter(paper__arxiv_id=arxiv_id, lang=lang).first()
    if row is None or not row.summary or row.summary.startswith('Error'):
        return None
    return row


def markdown_entry(meta, lang):
    ref = bare_ref(meta['arxiv_id'])
    lines = ['# %s' % meta['title'], '']
    if meta['authors']:
        lines.append('**Authors:** %s  ' % ', '.join(meta['authors'][:20]))
    if sources.is_external(meta['arxiv_id']):
        if meta['journal']:
            lines.append('**Published in:** %s  ' % meta['journal'])
        lines.append('**%s:** [%s](%s)  ' % (sources.label(meta['arxiv_id']).split(':')[0], ref, sources.ref_url(meta['arxiv_id'])))
    else:
        lines.append('**arXiv:** [%s](https://arxiv.org/abs/%s)  ' % (ref, ref))
    if meta['date']:
        lines.append('**Date:** %s  ' % meta['date'])
    lines.append('**Summary page:** https://www.summarizepaper.com/%s/arxiv-id/%s/' % (lang, meta['arxiv_id']))
    lines.append('')
    row = summary_of(meta['arxiv_id'], lang)
    if row is not None and row.notes:
        lines += ['## Key points', ''] + ['- %s' % p for p in keypoints.stored_points(row.notes)] + ['']
    if row is not None:
        lines += ['## Summary', '', row.summary.strip(), '']
    elif meta['abstract']:
        lines += ['## Abstract', '', meta['abstract'], '']
    return '\n'.join(lines)


def _download(text, filename, content_type):
    response = HttpResponse(text, content_type=content_type)
    response['Content-Disposition'] = 'attachment; filename="%s"' % filename
    return response


def export_paper(request, arxiv_id, fmt):
    meta = paper_meta(arxiv_id)
    if meta is None:
        raise Http404
    name = file_name(arxiv_id)
    if fmt == 'ris':
        return _download(ris_entry(meta), name + '.ris', 'application/x-research-info-systems; charset=utf-8')
    if fmt == 'md':
        return _download(markdown_entry(meta, get_language()), name + '.md', 'text/markdown; charset=utf-8')
    raise Http404


def export_many(metas, fmt, filename, lang):
    if fmt == 'bib':
        return _download('\n'.join(bibtex_entry(m) for m in metas), filename + '.bib', 'application/x-bibtex; charset=utf-8')
    if fmt == 'ris':
        return _download('\n'.join(ris_entry(m) for m in metas), filename + '.ris',
                         'application/x-research-info-systems; charset=utf-8')
    if fmt == 'md':
        return _download('\n\n---\n\n'.join(markdown_entry(m, lang) for m in metas), filename + '.md',
                         'text/markdown; charset=utf-8')
    raise Http404


# --- Zotero ------------------------------------------------------------------

def zotero_item(meta, lang):
    """The Zotero item and its child note (summary and key points)."""
    ref = bare_ref(meta['arxiv_id'])
    item = {
        'itemType': 'preprint', 'title': meta['title'],
        'creators': [_creator(a) for a in meta['authors'][:100]],
        'abstractNote': meta['abstract'], 'repository': 'arXiv', 'archiveID': 'arXiv:%s' % ref,
        'date': meta['date'], 'url': 'https://arxiv.org/abs/%s' % ref, 'DOI': meta['doi'],
        'tags': [{'tag': 'SummarizePaper'}],
    }
    if sources.is_external(meta['arxiv_id']):
        item.pop('archiveID')
        item['url'] = sources.ref_url(meta['arxiv_id'])
        if is_preprint(meta):
            item['repository'] = (meta['journal'] or '').replace(' preprint', '').split(',')[0]
        else:
            item.pop('repository')
            item.update({'itemType': 'journalArticle', 'publicationTitle': meta['journal']})
    row = summary_of(meta['arxiv_id'], lang)
    note = None
    if row is not None:
        points = ''.join('<li>%s</li>' % escape(p) for p in keypoints.stored_points(row.notes))
        note = ('<h1>SummarizePaper</h1>%s<h2>Summary</h2><p>%s</p>'
                '<p><a href="https://www.summarizepaper.com/%s/arxiv-id/%s/">summarizepaper.com</a></p>') % (
            '<h2>Key points</h2><ul>%s</ul>' % points if points else '', escape(row.summary), lang, meta['arxiv_id'])
    return item, note


def _creator(name):
    parts = name.rsplit(' ', 1)
    if len(parts) == 2:
        return {'creatorType': 'author', 'firstName': parts[0], 'lastName': parts[1]}
    return {'creatorType': 'author', 'name': name}


def zotero_post(account, items):
    response = requests.post('%s/users/%s/items' % (ZOTERO_API, account.library_id), json=items, timeout=30,
                             headers={'Zotero-API-Key': account.api_key, 'Zotero-API-Version': '3'})
    response.raise_for_status()
    return response.json()


def send_to_zotero(account, pairs):
    """Write the (item, note) pairs; returns how many papers were saved."""
    saved = 0
    for start in range(0, len(pairs), 25):  # the API takes 50 objects per call: 25 papers and their notes
        chunk = pairs[start:start + 25]
        result = zotero_post(account, [item for item, _ in chunk])
        notes = []
        for index, (_, note) in enumerate(chunk):
            key = (result.get('successful', {}).get(str(index)) or {}).get('key')
            if key:
                saved += 1
                if note:
                    notes.append({'itemType': 'note', 'parentItem': key, 'note': note, 'tags': [{'tag': 'SummarizePaper'}]})
        if notes:
            zotero_post(account, notes)
    return saved


def check_zotero(library_id, api_key):
    """True when the key exists, belongs to this library and may write to it."""
    try:
        response = requests.get('%s/keys/%s' % (ZOTERO_API, api_key), timeout=15)
        if response.status_code != 200:
            return False
        info = response.json()
    except (requests.RequestException, ValueError):
        return False
    return str(info.get('userID')) == str(library_id) and bool(info.get('access', {}).get('user', {}).get('write'))


def zotero_settings(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/integrations.html', {'anonymous': True})
    paid = access.is_paid(request.user)
    account = ZoteroAccount.objects.filter(user=request.user).first()
    error = False
    if request.method == 'POST' and paid:
        if request.POST.get('action') == 'disconnect':
            ZoteroAccount.objects.filter(user=request.user).delete()
            return redirect('/%s/integrations/' % get_language())
        library_id = re.sub(r'\D', '', request.POST.get('library_id', ''))[:40]
        api_key = re.sub(r'[^A-Za-z0-9]', '', request.POST.get('api_key', ''))[:100]
        if library_id and api_key and check_zotero(library_id, api_key):
            ZoteroAccount.objects.update_or_create(user=request.user, defaults={'library_id': library_id, 'api_key': api_key})
            return redirect('/%s/integrations/?saved=1' % get_language())
        error = True
    return render(request, 'summarizer/integrations.html', {
        'paid': paid, 'account': account, 'error': error, 'saved': request.GET.get('saved') == '1'})


@require_POST
def zotero_send(request):
    """Send one paper (arxiv_id) or a collection (collection) to Zotero, in the background."""
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    account = ZoteroAccount.objects.filter(user=request.user).first()
    if account is None:
        return JsonResponse({'status': 'setup', 'url': '/%s/integrations/' % get_language()})
    if request.POST.get('collection'):
        rows = CollectionPaper.objects.filter(collection__pk=request.POST['collection'], collection__user=request.user)
        ids = [(r.arxiv_id, r) for r in rows]
    else:
        ids = [(request.POST.get('arxiv_id', ''), None)]
    metas = [m for m in (paper_meta(aid, fallback) for aid, fallback in ids if aid) if m]
    if not metas:
        raise Http404
    lang = get_language()
    import uuid

    def work(pk):
        pairs = [zotero_item(m, lang) for m in metas]
        generated.release_connection()
        return {'saved': send_to_zotero(account, pairs)}
    row = generated.start('zotero', uuid.uuid4().hex, lang, work, user=request.user)
    return JsonResponse({'status': row.status, 'id': row.pk})
