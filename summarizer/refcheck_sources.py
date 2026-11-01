"""What the bibliography check reads (see refcheck.py): a .bib, the LaTeX of the manuscript, an Overleaf archive,
the source of an arXiv paper, a Zotero collection or a PDF.

Archives are read in memory, never extracted: only their .tex, .bib and .bbl files, MAX_FILES of them and
MAX_UNPACKED bytes at most. The main LaTeX file is the one with \\begin{document}; the files it \\input or
\\include follow it in the order the manuscript reads them, and each keeps its name, so line numbers say where
a sentence is.

Without a .bib (an arXiv source often has only its .bbl, a PDF only its printed list), the references are turned
into BibTeX entries: a biblatex .bbl gives its fields as they are; a \\bibitem list and a PDF are read by the
model, entry by entry, with the DOI and arXiv id found in the text preferred to what the model copies. Such
entries are checked but never 'corrected': there is no .bib of the author's to correct.
"""
import io
import posixpath
import re
import tarfile
import zipfile
import zlib

import requests

from . import generated, sources

MAX_FILES = 300
MAX_UNPACKED = 30 * 1024 * 1024
MAX_DOWNLOAD = 60 * 1024 * 1024
BBL_BATCH = 40
ARXIV_ID_RE = re.compile(r'^\s*(?:arxiv:\s*|https?://arxiv\.org/(?:abs|pdf)/)?(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?(?:\.pdf)?\s*$', re.I)
INPUT_RE = re.compile(r'\\(?:input|include|subfile)\s*\{([^}]+)\}')
BIB_RE = re.compile(r'\\(?:bibliography|addbibresource)\s*(?:\[[^\]]*\])?\{([^}]+)\}')
BIBITEM_RE = re.compile(r'\\bibitem\s*(?:\[(?:[^\[\]]|\[[^\]]*\])*\])?\s*\{([^}]+)\}')
DOI_IN_TEXT = re.compile(r'\b(10\.\d{4,9}/[^\s{}"\\,]+)')
ARXIV_IN_TEXT = re.compile(r'(?:arxiv[:\s/]*(?:abs/)?|eprint\s*[={]\s*|corr\W+(?:vol\.?\s*)?abs/)(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})', re.I)


class Unreadable(Exception):
    """The input cannot be read; the code says why (shown on the page)."""


def decode(data):
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        return data.decode('latin-1')


def strip_comments(tex):
    return re.sub(r'(?<!\\)%.*', '', tex)


# --- archives ----------------------------------------------------------------

def gunzip(data):
    inflate = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        out = inflate.decompress(data, MAX_UNPACKED + 1)
    except zlib.error:
        raise Unreadable('archive')
    if len(out) > MAX_UNPACKED:
        raise Unreadable('too_large')
    return out


def archive_files(data, name='main'):
    """{path: text} of the .tex, .bib and .bbl files of a .zip, a .tar(.gz), or one gzipped LaTeX file (arXiv's
    form for a one-file source)."""
    files, total = {}, [0]

    def keep(path, size, read):
        if not path.lower().endswith(('.tex', '.bib', '.bbl')) or '__MACOSX' in path or path.startswith('/'):
            return
        total[0] += size
        if len(files) >= MAX_FILES or total[0] > MAX_UNPACKED:
            raise Unreadable('too_large')
        files[posixpath.normpath(path)] = decode(read())

    if zipfile.is_zipfile(io.BytesIO(data)):
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    if not info.is_dir():
                        keep(info.filename, info.file_size, lambda info=info: archive.read(info)[:MAX_UNPACKED])
        except (zipfile.BadZipFile, RuntimeError, NotImplementedError):
            raise Unreadable('archive')
        return files
    raw = gunzip(data) if data[:2] == b'\x1f\x8b' else data
    try:
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive:
                if member.isfile():
                    keep(member.name, member.size, lambda member=member: archive.extractfile(member).read(MAX_UNPACKED))
        return files
    except tarfile.TarError:
        pass
    if raw is not data and b'\\begin{document}' in raw:
        files[name + '.tex'] = decode(raw)
        return files
    raise Unreadable('archive')


def locate(name, base, tex):
    """The path of the file \\input{name} reads, among the archive's LaTeX files."""
    name = name.strip()
    for candidate in (name, name + '.tex'):
        for path in (posixpath.normpath(posixpath.join(base, candidate)), posixpath.normpath(candidate)):
            if path in tex:
                return path
    return next((p for p in tex if p.endswith('/' + name) or p.endswith('/' + name + '.tex')), None)


def manuscript(files):
    """[(name, text)] of the LaTeX files in the order the manuscript reads them, the main file first."""
    tex = {p: t for p, t in files.items() if p.lower().endswith('.tex')}
    mains = [p for p, t in tex.items() if re.search(r'^[^%\n]*\\begin\{document\}', t, re.M)]
    if not mains:
        return sorted(tex.items())
    main = min(mains, key=lambda p: (p.count('/'), -len(tex[p])))
    base, ordered, seen = posixpath.dirname(main), [], set()

    def visit(path, depth):
        if path in seen or depth > 12:
            return
        seen.add(path)
        ordered.append((path, tex[path]))
        for name in INPUT_RE.findall(strip_comments(tex[path])):
            found = locate(name, base, tex)
            if found:
                visit(found, depth + 1)
    visit(main, 0)
    return ordered


def references_of(files, texts):
    """(BibTeX text, origin) of an archive: its .bib files ('bib'), else its .bbl or the thebibliography of its
    LaTeX ('bbl'); ('', '') without."""
    bibs = {p: t for p, t in files.items() if p.lower().endswith('.bib')}
    named = [n.strip() for group in BIB_RE.findall(' '.join(strip_comments(t) for _, t in texts))
             for n in group.split(',') if n.strip()]
    wanted = [p for p in bibs if any(posixpath.basename(p) in (posixpath.basename(n), posixpath.basename(n) + '.bib')
                                     for n in named)]
    chosen = wanted or list(bibs)
    if chosen:
        return '\n\n'.join(bibs[p] for p in sorted(chosen)), 'bib'
    bbls = [t for p, t in sorted(files.items()) if p.lower().endswith('.bbl')]
    bbls = bbls or [found for _name, tex in texts for found in
                    re.findall(r'\\begin\{thebibliography\}.*?(?:\\end\{thebibliography\}|$)', strip_comments(tex), re.S)]
    if bbls:
        return bbl_bibtex('\n'.join(bbls)), 'bbl'
    return '', ''


# --- .bbl ----------------------------------------------------------------------

def braced(text, i):
    """The content of the {...} group starting at text[i], and the index after it."""
    depth, j = 0, i
    while j < len(text):
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        if depth == 0:
            return text[i + 1:j], j + 1
        j += 1
    return text[i + 1:], len(text)


def biblatex_entries(bbl):
    """BibTeX entries of a biblatex .bbl, from its \\entry blocks: their fields are given as they are."""
    entries = []
    for block in re.split(r'\\entry\{', bbl)[1:]:
        head = re.match(r'([^}]+)\}\{(\w+)\}', block)
        if not head:
            continue
        body = block.split('\\endentry', 1)[0]
        fields = {}
        for name in ('title', 'journaltitle', 'booktitle', 'volume', 'pages', 'year', 'eprint', 'publisher'):
            match = re.search(r'\\field\{%s\}\{' % name, body)
            if match:
                fields[name] = braced(body, match.end() - 1)[0]
        if 'year' not in fields:
            date = re.search(r'\\field\{(?:year|labelyear)\}\{(\d{4})', body)
            if date:
                fields['year'] = date.group(1)
        for name in ('doi', 'eprint', 'url'):
            match = re.search(r'\\verb\{%s\}\s*\\verb (.+?)\s*\\endverb' % name, body, re.S)
            if match:
                fields[name] = match.group(1).strip()
        authors = re.split(r'\\(?:name|list|strng|field|range|verb)\{', body.split('\\name{author}', 1)[1], 1)[0] \
            if '\\name{author}' in body else ''
        names = re.findall(r'family=\{([^}]*)\}', authors)
        if names:
            fields['author'] = ' and '.join(names)
        if 'journaltitle' in fields:
            fields['journal'] = fields.pop('journaltitle')
        entries.append((head.group(1), head.group(2).lower(), fields))
    return entries


def bibitems(bbl):
    """[(key, text)] of a \\bibitem list."""
    found = list(BIBITEM_RE.finditer(bbl))
    items = []
    for match, following in zip(found, found[1:] + [None]):
        text = bbl[match.end():following.start() if following else len(bbl)]
        text = text.split('\\end{thebibliography}', 1)[0]
        items.append((match.group(1).strip(), ' '.join(text.replace('\\newblock', ' ').split())[:1500]))
    return items


def as_bibtex(key, kind, fields):
    lines = ['@%s{%s,' % (kind, key)] + ['  %s = {%s},' % (k, v) for k, v in fields.items() if v]
    return '\n'.join(lines) + '\n}'


def read_items(items):
    """{key: [family, year, title, journal, volume, page, doi, arxiv]} of reference texts, as the model copies them."""
    from .missed import at_once

    def ask(batch):
        listing = '\n'.join('<ref id="%s">%s</ref>' % (i, text) for i, (_key, text) in enumerate(batch))
        answer = generated.chat_json(
            'Above are references of a research paper, as LaTeX prints them. Answer with a JSON object {"references": '
            '{"ref id": ["family name of the first author", "year", "title, or empty when the reference does not give '
            'it", "journal or book", "volume", "first page or article number", "DOI", "arXiv id"]}}. Copy the values as '
            'they are written, without LaTeX commands; leave empty what the reference does not give.',
            paid=False, max_tokens=8000, system='<references>\n%s\n</references>' % listing)
        rows = answer.get('references') if isinstance(answer.get('references'), dict) else {}
        found = {}
        for i, (key, _text) in enumerate(batch):
            row = rows.get(str(i))
            if isinstance(row, list):
                found[key] = ([str(v or '').strip() for v in row] + [''] * 8)[:8]
        return found
    batches = [items[i:i + BBL_BATCH] for i in range(0, len(items), BBL_BATCH)]
    read = {}
    for found in at_once([lambda batch=batch: ask(batch) for batch in batches]):
        read.update(found)
    return read


def entry_of(key, row, text=''):
    """A BibTeX entry from what the model copied of a reference; identifiers in its text win."""
    family, year, title, journal, volume, page, doi, arxiv = row
    doi = (DOI_IN_TEXT.search(text).group(1).rstrip('.,;)') if DOI_IN_TEXT.search(text) else '') or sources.clean_doi(doi) or ''
    found = ARXIV_IN_TEXT.search(text)
    arxiv = found.group(1) if found else re.sub(r'v\d+$', '', arxiv.lower().replace('arxiv:', '').strip())
    year = (re.search(r'(1[89]|20)\d\d', year) or [''])[0] if year else ''
    fields = {'author': family, 'title': title, 'journal': journal if journal.lower() != 'arxiv' else '',
              'volume': volume, 'pages': page, 'year': year, 'doi': doi,
              'eprint': arxiv if re.fullmatch(r'\d{4}\.\d{4,5}|[a-z\-]+(?:\.[a-z]{2})?/\d{7}', arxiv or '') else ''}
    if fields['eprint']:
        fields['archiveprefix'] = 'arXiv'
    return as_bibtex(key, 'article' if fields['journal'] else 'misc', fields)


def bbl_bibtex(bbl):
    """BibTeX text of a .bbl: biblatex fields as they are, \\bibitem texts read by the model."""
    entries = biblatex_entries(bbl)
    if entries:
        return '\n\n'.join(as_bibtex(key, kind, fields) for key, kind, fields in entries)
    items = bibitems(bbl)
    if not items:
        return ''
    read = read_items(items)
    return '\n\n'.join(entry_of(key, read[key], text) for key, text in items if key in read)


# --- PDF -----------------------------------------------------------------------

def pdf_bibtex(data):
    """BibTeX text of the reference list of a PDF, read by the model (no LaTeX: no citation is checked)."""
    from . import missed
    try:
        text = missed.pdf_text(data)
    except Exception:
        raise Unreadable('pdf')
    listing = missed.split_references(text)[1]
    if not listing:
        raise Unreadable('no_references')
    items = [('ref%d' % (i + 1), line) for i, line in enumerate(reference_lines(listing))]
    read = read_items(items)
    return '\n\n'.join(entry_of(key, read[key], text) for key, text in items if key in read)


def reference_lines(listing):
    """The references of a printed list: split at their numbers ([12], 12.) or, without numbers, at blank lines."""
    listing = listing[:120000]
    parts = re.split(r'\n\s*(?=\[\d{1,3}\]|\d{1,3}\.\s+[A-Z])', '\n' + listing)
    if len(parts) < 5:
        parts = re.split(r'\n\s*\n', listing)
    return [' '.join(p.split())[:1500] for p in parts if len(p.strip()) > 20][:400]


# --- arXiv and Zotero ----------------------------------------------------------

def arxiv_id(text):
    """'2101.00001v2' or 'hep-th/9901001' from what was typed, '' when it is no arXiv identifier."""
    match = ARXIV_ID_RE.match(text or '')
    return (match.group(1) + (match.group(2) or '')) if match else ''


def arxiv_source(paper_id):
    """{path: text} of the LaTeX source arXiv keeps for a paper."""
    try:
        response = requests.get('https://arxiv.org/e-print/%s' % paper_id, stream=True, timeout=60,
                                headers={'User-Agent': 'SummarizePaper bibliography check (contact@summarizepaper.com)'})
    except requests.RequestException:
        raise generated.SourceUnavailable('arxiv')
    if response.status_code == 404:
        raise Unreadable('arxiv_missing')
    if response.status_code != 200:
        raise generated.SourceUnavailable('arxiv')
    if 'pdf' in response.headers.get('Content-Type', ''):
        raise Unreadable('arxiv_pdf_only')
    data = b''
    for chunk in response.iter_content(1 << 16):
        data += chunk
        if len(data) > MAX_DOWNLOAD:
            raise Unreadable('too_large')
    return archive_files(data, name=paper_id.replace('/', '_'))


def zotero_bibtex(account, key, limit=500):
    """BibTeX of the top items of a Zotero collection, as Zotero exports them."""
    from .exports import ZOTERO_API
    parts = []
    for start in range(0, limit, 100):
        try:
            response = requests.get('%s/users/%s/collections/%s/items/top' % (ZOTERO_API, account.library_id, key),
                                    params={'format': 'bibtex', 'limit': 100, 'start': start}, timeout=30,
                                    headers={'Zotero-API-Key': account.api_key, 'Zotero-API-Version': '3'})
            response.raise_for_status()
        except requests.RequestException:
            raise generated.SourceUnavailable('zotero')
        parts.append(response.text)
        if int(response.headers.get('Total-Results') or 0) <= start + 100:
            break
    return '\n\n'.join(parts)
