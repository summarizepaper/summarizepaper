"""Getting papers out of SummarizePaper as files: BibTeX, RIS, Markdown (Obsidian, Notion) and an Obsidian vault.

BibTeX goes to LaTeX, RIS to Zotero, Mendeley and EndNote, Markdown to Obsidian (with its properties) and Notion.
A preprint published since is exported as its published version (journal and DOI, its arXiv id kept), so that
our own files pass the bibliography check. Every summary exported says it was written by a model, from which
version of the paper and when. The Zotero web API (sending, syncing, writing back) is in zotero.py.
"""
import io
import json
import re
import unicodedata
import zipfile

from django.http import Http404, HttpResponse
from django.utils.http import content_disposition_header
from django.utils.translation import get_language
from django.utils.translation import gettext as _

from . import keypoints, publication_status, sources
from .models import AnnouncedPaper, ArxivPaper, SummaryPaper

MAX_AUTHORS_SHOWN = 20


def bare_ref(arxiv_id):
    """'hep-th--9901001v2' -> 'hep-th/9901001'; for other sources the DOI (or HAL / PMC id)."""
    if sources.is_external(arxiv_id):
        return sources.label(arxiv_id).split(': ', 1)[-1]
    return re.sub(r'v\d+$', '', arxiv_id.replace('--', '/'))


def file_name(arxiv_id):
    return re.sub(r'[^A-Za-z0-9._-]+', '_', bare_ref(arxiv_id))


def published_version(paper):
    """{journal, doi, year} of the journal version of an arXiv paper: Crossref's when the editorial check found it,
    otherwise the DOI and journal reference its authors declared on arXiv. None for a preprint."""
    report = publication_status.latest(paper.arxiv_id) or {}
    if report.get('status') == 'published' and report.get('doi') and report.get('journal'):
        return {'journal': report['journal'], 'doi': report['doi'], 'year': (report.get('date') or '')[:4]}
    doi = sources.clean_doi(paper.link_doi or '')
    if doi and not sources.ARXIV_DOI_RE.match(doi) and paper.journal_ref:
        year = re.search(r'\b(?:19|20)\d{2}\b', paper.journal_ref)
        return {'journal': ' '.join(paper.journal_ref.split()), 'doi': doi, 'year': year.group() if year else ''}
    return None


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
            'published': None if sources.is_external(arxiv_id) else published_version(paper),
        }
    announced = AnnouncedPaper.objects.filter(arxiv_id=arxiv_id).first()
    if announced is not None:
        return {
            'arxiv_id': arxiv_id, 'title': announced.title, 'authors': json.loads(announced.authors or '[]'),
            'year': announced.announced.year, 'date': announced.announced.isoformat(),
            'category': announced.categories.split(',')[0], 'abstract': announced.abstract, 'doi': '', 'journal': '',
        }
    if sources.is_external(arxiv_id):
        try:
            paper = sources.metadata(arxiv_id)
        except sources.SourceUnavailable:
            paper = None
        if paper:
            return {
                'arxiv_id': arxiv_id, 'title': paper['title'], 'authors': paper['authors'],
                'year': paper['published'][:4], 'date': paper['published'], 'category': paper['category'],
                'abstract': paper['abstract'], 'doi': paper['doi'], 'journal': paper['journal'],
            }
    if fallback is not None:
        return {'arxiv_id': arxiv_id, 'title': fallback.title, 'authors': [a for a in fallback.authors.split(', ') if a],
                'year': '', 'date': '', 'category': '', 'abstract': '', 'doi': '', 'journal': ''}
    return None


def is_preprint(meta):
    return 'preprint' in (meta['journal'] or '').lower() or not meta['journal']


def journal_of(meta):
    """(journal, DOI, year) of the version to cite: the published one when there is one, else ('', DOI, year)."""
    published = meta.get('published')
    if published:
        return published['journal'], published['doi'], published['year'] or meta['year']
    if sources.is_external(meta['arxiv_id']) and not is_preprint(meta):
        return meta['journal'], meta['doi'], meta['year']
    return '', meta['doi'], meta['year']


def paper_url(meta):
    if sources.is_external(meta['arxiv_id']):
        return sources.ref_url(meta['arxiv_id'])
    return 'https://arxiv.org/abs/%s' % bare_ref(meta['arxiv_id'])


# --- BibTeX -------------------------------------------------------------------------------------------------

def cite_key(meta):
    """lee2021clouds: the first author's family name (Beránek: beranek), the year of the version cited (that of the
    journal for a preprint published since, as in its year field), the first word of more than 3 letters."""
    first_word = next((w for w in re.findall(r'[A-Za-z]+', meta['title']) if len(w) > 3), 'paper')
    family = split_name(meta['authors'][0])[1] if meta['authors'] else ''
    last_name = re.sub(r'[^A-Za-z]', '', unicodedata.normalize('NFKD', family).encode('ascii', 'ignore').decode())
    year = (meta.get('published') or {}).get('year') or meta['year']
    return '%s%s%s' % ((last_name or 'paper').lower(), year, first_word.lower())


def unique_keys(keys):
    """The keys with a letter added to those several papers share (smith2020a, smith2020b...)."""
    counts = {}
    for key in keys:
        counts[key] = counts.get(key, 0) + 1
    taken, out = {}, []
    for key in keys:
        if counts[key] > 1:
            taken[key] = taken.get(key, 0) + 1
            key += 'abcdefghijklmnopqrstuvwxyz'[(taken[key] - 1) % 26]
        out.append(key)
    return out


def cite_keys(entries):
    """Each BibTeX entry with a key no other entry has, and the keys in order."""
    parts = [entry.split(',', 1) for entry in entries]
    keys = unique_keys([head.split('{', 1)[1] for head, _rest in parts])
    return ['%s{%s,%s' % (head.split('{', 1)[0], key, rest) for (head, rest), key in zip(parts, keys)], keys


def _protected(word):
    """A word whose capitals a bibliography style must keep (GPT, JWST, LaTeX, McDonald, 3D) in braces."""
    text = word.group()
    return '{%s}' % text if any(c.isupper() for c in text[1:]) else text


def bib_text(text):
    """A title or a journal for BibTeX: LaTeX's special characters escaped outside math, acronyms protected.
    A text that already holds LaTeX (braces or commands) only gets its bare &, % and # escaped."""
    text = ' '.join((text or '').split())
    parts = re.split(r'(\$[^$]*\$)', text)
    if any('{' in p or '\\' in p for p in parts[::2]):
        return re.sub(r'(?<!\\)([&%#])', r'\\\1', text)
    for i in range(0, len(parts), 2):
        escaped = re.sub(r'([&%#_])', r'\\\1', parts[i])
        parts[i] = re.sub(r"[A-Za-z0-9][A-Za-z0-9']*", _protected, escaped)
    return ''.join(parts)


def bibtex_entry(meta):
    ref = bare_ref(meta['arxiv_id'])
    journal, doi, year = journal_of(meta)
    authors = '  author={%s},' % ' and '.join(re.sub(r'(?<!\\)&', r'\\&', a) for a in meta['authors'])
    if sources.is_external(meta['arxiv_id']):
        lines = ['@%s{%s,' % ('misc' if is_preprint(meta) else 'article', cite_key(meta)),
                 '  title={%s},' % bib_text(meta['title']), authors, '  year={%s},' % meta['year']]
        if journal:
            lines.append('  journal={%s},' % bib_text(journal))
        elif meta['journal']:
            lines.append('  howpublished={%s},' % bib_text(meta['journal']))
        if doi:
            lines.append('  doi={%s},' % doi)
        lines.append('  url={%s}' % sources.ref_url(meta['arxiv_id']))
        lines.append('}')
        return '\n'.join(lines) + '\n'
    lines = ['@%s{%s,' % ('article' if journal else 'misc', cite_key(meta)),
             '  title={%s},' % bib_text(meta['title']), authors, '  year={%s},' % year]
    if journal:
        lines.append('  journal={%s},' % bib_text(journal))
    lines += ['  eprint={%s},' % ref, '  archivePrefix={arXiv},']
    if meta['category']:
        lines.append('  primaryClass={%s},' % meta['category'])
    if doi:
        lines.append('  doi={%s},' % doi)
    lines.append('  url={%s}' % ('https://doi.org/%s' % doi if journal else 'https://arxiv.org/abs/%s' % ref))
    lines.append('}')
    return '\n'.join(lines) + '\n'


# --- RIS ----------------------------------------------------------------------------------------------------

def ris_entry(meta):
    ref = bare_ref(meta['arxiv_id'])
    external = sources.is_external(meta['arxiv_id'])
    journal, doi, year = journal_of(meta)
    lines = ['TY  - %s' % ('JOUR' if journal else 'PREP'), 'TI  - %s' % meta['title']]
    lines += ['AU  - %s' % a for a in meta['authors']]
    if year:
        lines.append('PY  - %s' % year)
    if meta['date'] and not meta.get('published'):
        lines.append('DA  - %s' % meta['date'].replace('-', '/'))
    if meta['abstract']:
        lines.append('AB  - %s' % meta['abstract'])
    if doi:
        lines.append('DO  - %s' % doi)
    if journal:
        lines.append('T2  - %s' % journal)
    elif external and meta['journal']:
        lines.append('PB  - %s' % meta['journal'])
    if external:
        lines += ['UR  - %s' % sources.ref_url(meta['arxiv_id']), 'ER  - ']
    else:
        if not journal:
            lines.append('PB  - arXiv')
        lines += ['AN  - arXiv:%s' % ref, 'UR  - https://arxiv.org/abs/%s' % ref, 'ER  - ']
    return '\n'.join(lines) + '\n'


# --- Markdown -----------------------------------------------------------------------------------------------

def summary_of(arxiv_id, lang):
    row = SummaryPaper.objects.filter(paper__arxiv_id=arxiv_id, lang=lang).first()
    if row is None or not row.summary or row.summary.startswith('Error'):
        return None
    return row


def provenance(arxiv_id, row):
    """What a summary is: written by a model, from which version of the paper, when."""
    source = sources.label(arxiv_id) if sources.is_external(arxiv_id) else 'arXiv:%s' % arxiv_id.replace('--', '/')
    return _('Summary written by an AI model (SummarizePaper) from %(source)s on %(date)s. Check it against the paper.') % {
        'source': source, 'date': (row.updated or row.created).date().isoformat()}


def authors_line(authors):
    shown = ', '.join(authors[:MAX_AUTHORS_SHOWN])
    return shown + (' et al.' if len(authors) > MAX_AUTHORS_SHOWN else '')


def page_url(meta, lang):
    return 'https://www.summarizepaper.com/%s/paper/%s/' % (lang, meta['arxiv_id'])


def _yaml(value):
    return json.dumps(str(value), ensure_ascii=False)


def obsidian_tag(tag):
    """An Obsidian tag: no spaces, not only digits."""
    tag = re.sub(r'[^\w/-]+', '-', tag.strip(), flags=re.UNICODE).strip('-')
    return tag if tag and not tag.isdigit() else ''


def frontmatter(meta, lang, key, tags=(), status=''):
    """The properties of an Obsidian note: YAML that Dataview and the Zotero plugins read."""
    journal, doi, year = journal_of(meta)
    lines = ['---', 'title: %s' % _yaml(meta['title'])]
    if meta['authors']:
        lines += ['authors:'] + ['  - %s' % _yaml(a) for a in meta['authors'][:100]]
    if year:
        lines.append('year: %s' % year)
    if journal:
        lines.append('journal: %s' % _yaml(journal))
    if doi:
        lines.append('doi: %s' % _yaml(doi))
    if not sources.is_external(meta['arxiv_id']):
        lines.append('arxiv: %s' % _yaml(bare_ref(meta['arxiv_id'])))
    lines += ['citekey: %s' % _yaml(key), 'url: %s' % _yaml('https://doi.org/%s' % doi if journal else paper_url(meta)),
              'summarizepaper: %s' % _yaml(page_url(meta, lang))]
    tags = [t for t in (obsidian_tag(t) for t in tags) if t]
    if tags:
        lines += ['tags:'] + ['  - %s' % _yaml(t) for t in tags]
    if status:
        lines.append('status: %s' % _yaml(status))
    return '\n'.join(lines + ['---', ''])


def field(name, lang, value):
    """A bold label and its value on a line of its own, with the French space before the colon."""
    return '**%s%s** %s  ' % (name, ' :' if lang.startswith('fr') else ':', value)


def markdown_entry(meta, lang, key=None, tags=(), status='', heading=True):
    """The note of a paper; with key, Obsidian properties first (citekey, DOI, tags...); heading: its title first."""
    ref = bare_ref(meta['arxiv_id'])
    journal, doi, _year = journal_of(meta)
    lines = [frontmatter(meta, lang, key, tags, status)] if key else []
    lines += ['# %s' % meta['title'], ''] if heading else []
    if meta['authors']:
        lines.append(field(_('Authors'), lang, authors_line(meta['authors'])))
    if journal:
        lines.append(field(_('Published in'), lang, journal))
    if doi and journal:
        lines.append(field('DOI', lang, '[%s](https://doi.org/%s)' % (doi, doi)))
    if sources.is_external(meta['arxiv_id']):
        if meta['journal'] and not journal:
            lines.append(field(_('Published in'), lang, meta['journal']))
        if not (doi and journal):
            lines.append(field(sources.label(meta['arxiv_id']).split(':')[0], lang, '[%s](%s)' % (ref, sources.ref_url(meta['arxiv_id']))))
    else:
        lines.append(field('arXiv', lang, '[%s](https://arxiv.org/abs/%s)' % (ref, ref)))
    if meta['date']:
        lines.append(field(_('Date'), lang, meta['date']))
    lines.append(field(_('Summary page'), lang, page_url(meta, lang)).rstrip())
    lines.append('')
    row = summary_of(meta['arxiv_id'], lang)
    if row is not None and row.notes:
        lines += ['## %s' % _('Key points'), ''] + ['- %s' % p for p in keypoints.stored_points(row.notes)] + ['']
    if row is not None:
        lines += ['## %s' % _('Summary'), '', row.summary.strip(), '', '_%s_' % provenance(meta['arxiv_id'], row), '']
    elif meta['abstract']:
        lines += ['## %s' % _('Abstract'), '', meta['abstract'], '']
    return '\n'.join(lines)


# --- downloads ----------------------------------------------------------------------------------------------

def _download(text, filename, content_type):
    response = HttpResponse(text, content_type=content_type)
    response['Content-Disposition'] = content_disposition_header(True, filename)
    return response


def export_paper(request, arxiv_id, fmt):
    """ris; md: the note for Obsidian, with its properties; notion: the same note without them."""
    meta = paper_meta(arxiv_id)
    if meta is None:
        raise Http404
    name = file_name(arxiv_id)
    if fmt == 'ris':
        return _download(ris_entry(meta), name + '.ris', 'application/x-research-info-systems; charset=utf-8')
    if fmt == 'md':
        key = cite_key(meta)
        return _download(markdown_entry(meta, get_language(), key), key + '.md', 'text/markdown; charset=utf-8')
    if fmt == 'notion':  # Notion names the page after the file, so the title is the file's name, not a heading
        title = ' '.join(re.sub(r'[\\/:*?"<>|]+', ' ', meta['title']).split())[:150] or name
        return _download(markdown_entry(meta, get_language(), heading=False), title + '.md', 'text/markdown; charset=utf-8')
    raise Http404


def with_note(entry, fmt, note, tags):
    """A BibTeX, RIS or Markdown entry with the reader's note and tags (a collection's). BibTeX keeps them in
    annote and keywords, which bibliography styles do not print."""
    tags = [t.strip() for t in (tags or '').split(',') if t.strip()]
    note = (note or '').strip()
    if not (note or tags):
        return entry
    if fmt == 'bib':
        fields = ([',\n  keywords={%s}' % ', '.join(tags)] if tags else []) + \
                 ([',\n  annote={%s}' % note.replace('{', '(').replace('}', ')')] if note else [])
        body = entry.rstrip()
        return body[:-1].rstrip() + ''.join(fields) + '\n}\n'
    if fmt == 'ris':
        lines = ['KW  - %s' % t for t in tags] + (['N1  - %s' % ' '.join(note.split())] if note else [])
        return entry.replace('ER  - ', '\n'.join(lines + ['ER  - ']), 1)
    extra = ['**Tags:** %s  ' % ', '.join(tags)] if tags else []
    return entry.rstrip() + '\n\n' + '\n'.join(extra + (['## %s' % _('My note'), '', note] if note else [])) + '\n'


def export_many(metas, fmt, filename, lang, notes=None, status=None):
    """notes: {arxiv_id: (note, tags)} to add to each entry; status: {arxiv_id: reading status}, for Obsidian."""
    notes = notes or {}

    def entries(write):
        return [with_note(write(m), fmt, *notes.get(m['arxiv_id'], ('', ''))) for m in metas]
    if fmt == 'bib':
        return _download('\n'.join(cite_keys(entries(bibtex_entry))[0]), filename + '.bib',
                         'application/x-bibtex; charset=utf-8')
    if fmt == 'ris':
        return _download('\n'.join(entries(ris_entry)), filename + '.ris',
                         'application/x-research-info-systems; charset=utf-8')
    if fmt == 'md':
        return _download('\n\n---\n\n'.join(entries(lambda m: markdown_entry(m, lang))), filename + '.md',
                         'text/markdown; charset=utf-8')
    if fmt == 'obsidian':
        return _download(obsidian_vault(metas, filename, lang, notes, status or {}), filename + '.zip', 'application/zip')
    raise Http404


def obsidian_vault(metas, name, lang, notes, status):
    """A folder for an Obsidian vault: one note per paper, named by its citekey, with its properties, and a note
    for the collection that links them all."""
    keys = unique_keys([cite_key(m) for m in metas])
    out = io.BytesIO()
    with zipfile.ZipFile(out, 'w', zipfile.ZIP_DEFLATED) as archive:
        index = ['# %s' % name, '']
        for meta, key in zip(metas, keys):
            note, tags = notes.get(meta['arxiv_id'], ('', ''))
            tag_list = [t.strip() for t in (tags or '').split(',') if t.strip()]
            body = markdown_entry(meta, lang, key, tags=tag_list, status=status.get(meta['arxiv_id'], ''))
            if note.strip():
                body = body.rstrip() + '\n\n## %s\n\n%s\n' % (_('My note'), note.strip())
            body = body.rstrip() + '\n\n%s: [[%s]]\n' % (_('Collection'), name)
            archive.writestr('%s/%s.md' % (name, key), body)
            _journal, _doi, year = journal_of(meta)
            index.append('- [[%s]] %s%s' % (key, meta['title'], ' (%s)' % year if year else ''))
        archive.writestr('%s/%s.md' % (name, name), '\n'.join(index) + '\n')
    return out.getvalue()


# --- authors ------------------------------------------------------------------------------------------------

PARTICLES = {'van', 'von', 'der', 'den', 'de', 'del', 'della', 'di', 'da', 'dos', 'das', 'du', 'la', 'le', 'ter', 'ten',
             'bin', 'ibn', 'al', 'el', 'zu', 'y', 'st', 'st.', 'mac'}


def split_name(name):
    """(given names, family name): 'Jean-Pierre van der Waals' -> ('Jean-Pierre', 'van der Waals'),
    'Lee, Ann' -> ('Ann', 'Lee'). The family name alone for a one-word name."""
    name = ' '.join((name or '').split())
    if ',' in name:
        family, given = [p.strip() for p in name.split(',', 1)]
        return given, family
    words = name.split(' ')
    if len(words) < 2:
        return '', name
    start = len(words) - 1
    while start > 1 and words[start - 1].lower() in PARTICLES:
        start -= 1
    if words[-1].rstrip('.').lower() in ('jr', 'sr', 'ii', 'iii') and len(words) > 2:
        return ' '.join(words[:-2]), '%s %s' % (words[-2], words[-1])
    return ' '.join(words[:start]), ' '.join(words[start:])
