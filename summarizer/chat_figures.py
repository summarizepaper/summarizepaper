"""Public figure galleries. Only known, openly licensed papers and trusted hosts.

No model-generated image URLs or implicit paper generation. Publisher sources are explicitly supported.
"""
import asyncio
import copy
import hashlib
import time
import json
import re
from urllib.parse import unquote, urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from . import sources, unified_search
from .models import ArxivPaper, ExternalPaper

MAX_HTML = 6_000_000
MAX_FIGURES = 40
IMAGE_HOSTS = {'arxiv.org', 'www.arxiv.org', 'pmc.ncbi.nlm.nih.gov', 'cdn.ncbi.nlm.nih.gov', 'media.springernature.com'}


def safe_image(url):
    try:
        parsed = urlsplit(url)
        return (parsed.scheme == 'https' and parsed.hostname in IMAGE_HOSTS and
                not parsed.username and not parsed.password and parsed.port in (None, 443))
    except ValueError:
        return False


def html_source(paper):
    key = paper.arxiv_id
    if not sources.is_external(key):
        ref = key.replace('--', '/')
        if unified_search.arxiv_identifier(ref) == key:
            return 'https://arxiv.org/html/' + ref, 'arxiv'
        return None
    pmcid = key[4:].upper() if key.startswith('pmc_') else ''
    row = ExternalPaper.objects.filter(key=key).first()
    if not pmcid and row:
        try:
            pmcid = next((where for kind, where in json.loads(row.metadata).get('text', []) if kind == 'epmc'), '')
        except (ValueError, TypeError, AttributeError):
            pass
    if isinstance(pmcid, str) and re.fullmatch(r'PMC\d+', pmcid.upper()):
        return 'https://pmc.ncbi.nlm.nih.gov/articles/%s/' % pmcid.upper(), 'pmc'
    doi = sources.key_doi(key) or sources.clean_doi(paper.link_doi or '')
    if re.fullmatch(r'10\.1038/[a-z0-9.-]+', doi):
        return 'https://www.nature.com/articles/' + doi.split('/', 1)[1], 'nature'
    return None


def arxiv_fallback(paper):
    """Only an explicit repository mapping; check the preprint's own license."""
    row = ExternalPaper.objects.filter(key=paper.arxiv_id).first()
    try:
        meta = json.loads(row.metadata) if row else {}
        refs = [meta.get('arxiv_id', '')] + [entry[1] for entry in meta.get('text', [])
                if isinstance(entry, (list, tuple)) and len(entry) == 2 and entry[0] == 'arxiv']
    except (ValueError, TypeError, AttributeError):
        return None
    for ref in refs:
        if not isinstance(ref, str):
            continue
        key = unified_search.arxiv_identifier(ref)
        if not key:
            continue
        license_key = 'figure-license:v1:' + key
        license_url = cache.get(license_key)
        if license_url is None:
            known = ArxivPaper.objects.filter(arxiv_id=key).first()
            if known:
                license_url = known.license or ''
            else:
                from .utils import get_arxiv_metadata
                try:
                    metadata = asyncio.run(get_arxiv_metadata(key))
                    license_url = metadata[12] if metadata and metadata[0] else ''
                except Exception:
                    license_url = ''
            cache.set(license_key, license_url, 86400 if license_url else 600)
        if sources.is_open_license(license_url):
            return ('https://arxiv.org/html/' + key.replace('--', '/'), 'arxiv', key, license_url)
    return None


def caption_text(caption):
    """Keep one representation of each formula, never MathML plus its annotation.

    Return plain text with TeX delimiters; the browser inserts it with textContent.
    Publisher HTML and MathML are never copied into the page.
    """
    for node in caption.select('script, style'):
        node.decompose()

    def formula(node):
        math = node if node.name == 'math' else node.find('math')
        annotation = node.find('annotation', encoding=re.compile(r'(?:x-tex|latex)', re.I))
        tex_node = node.find('tex-math')
        tex = (math.get('alttext') if math else '') or (
            tex_node.get_text() if tex_node else '') or (annotation.get_text() if annotation else '')
        if tex and tex.strip():
            return r'\(' + tex.strip() + r'\)'
        # Without TeX, keep only the visible representation, not hidden alternatives.
        for hidden in node.find_all(['annotation', 'annotation-xml', 'tex-math']):
            hidden.decompose()
        return node.get_text(' ', strip=True)

    # PMC may wrap TeX and MathML together in alternatives inside inline-formula.
    for node in list(caption.find_all(['inline-formula', 'disp-formula'])):
        node.replace_with(formula(node))
    for node in list(caption.find_all('math')):
        node.replace_with(formula(node))
    for node in caption.find_all(['sub', 'sup']):
        text = node.get_text(' ', strip=True)
        text = re.sub(r'([\\{}%$#&_])', r'\\\1', text)
        node.replace_with(r'\({}' + ('_' if node.name == 'sub' else '^') + r'{\text{' + text + r'}}\)')
    for block in caption.find_all(['p', 'div', 'br']):
        block.insert_after(' ')
    return ' '.join(caption.get_text().split())[:5000]


def extract(html, base, kind):
    soup = BeautifulSoup(html, 'html.parser')
    found = []
    selector = 'figure.ltx_figure' if kind == 'arxiv' else 'figure'
    for index, figure in enumerate(soup.select(selector), 1):
        if figure.find_parent('figure'):
            continue
        caption = figure.find('figcaption')
        if caption is None:
            continue
        if kind == 'nature':
            caption = copy.copy(caption)
            description = figure.select_one('.c-article-section__figure-description')
            if description:
                caption.append(copy.copy(description))
            # The article license does not override a different third-party credit.
            if re.search(r'©|copyright|reproduced (?:from|with)|adapted (?:from|with)|not (?:included|covered).*licen', caption.get_text(' ', strip=True), re.I):
                continue
        images = []
        for node in figure.find_all(['img', 'object']):
            if node.find_parent('figcaption'):
                continue
            source = node.get('src') or node.get('data-src') or node.get('data')
            if not source:
                continue
            url = urljoin(base, source)
            own_nature_image = kind != 'nature' or (
                urlsplit(url).hostname == 'media.springernature.com' and
                '/art:10.1038/' + base.rsplit('/', 1)[-1] + '/' in unquote(urlsplit(url).path))
            if safe_image(url) and own_nature_image and url not in images:
                images.append(url)
            if len(images) == 8:
                break
        if not images:
            continue
        text = caption_text(caption)
        tag = figure.select_one('.ltx_tag_figure')
        match = re.match(r'\s*((?:Extended Data )?(?:Figure|Fig\.)\s*[A-Za-z]?\d+(?:[.\-]\d+)?)', text, flags=re.I)
        label = tag.get_text(' ', strip=True) if tag else (match.group(1) if match else 'Figure %d' % index)
        anchor = figure.get('id', '')
        if kind == 'nature':
            tag = figure.select_one('figcaption [id]')
            anchor = tag.get('id', '') if tag else anchor
        found.append({'number': len(found) + 1, 'label': label[:80], 'images': images,
                      'caption': text, 'source_url': base + ('#' + anchor if re.fullmatch(r'[\w.:-]+', anchor) else '')})
        if len(found) == MAX_FIGURES:
            break
    return found


def fetch_html(url, kind):
    """Bounded public HTML. Nature's anonymous cookie redirects stay on fixed hosts."""
    deadline = time.monotonic() + 30
    session = requests.Session() if kind == 'nature' else None
    get = session.get if session else requests.get
    try:
        for _ in range(5 if session else 1):
            if time.monotonic() > deadline:
                raise ValueError('HTML deadline exceeded')
            with get(url, timeout=(4, 10), allow_redirects=False, stream=True,
                     headers={'User-Agent': sources.USER_AGENT}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    if not session:
                        return b''
                    next_url = urljoin(url, response.headers.get('Location', ''))
                    target = urlsplit(next_url)
                    if (target.scheme != 'https' or target.hostname not in {'www.nature.com', 'idp.nature.com'}
                            or target.username or target.password or target.port not in (None, 443)):
                        return b''
                    url = next_url
                    continue
                if response.status_code == 404:
                    return b''
                response.raise_for_status()
                # An identity page is never scientific content.
                if session and urlsplit(url).hostname != 'www.nature.com':
                    return b''
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_HTML or time.monotonic() > deadline:
                        raise ValueError('HTML too large or slow')
                    chunks.append(chunk)
                return b''.join(chunks)
        return b''
    finally:
        if session:
            session.close()


def paper_gallery(paper):
    """Shared by the public gallery and the assistant's validated image inputs."""
    data = {'paper_id': paper.arxiv_id, 'title': paper.title or paper.arxiv_id,
            'source_url': sources.ref_url(paper.arxiv_id), 'figures': []}
    if not sources.is_open_license(paper.license):
        return dict(data, status='license')
    primary = html_source(paper)
    candidates = [(*primary, paper.arxiv_id, paper.license)] if primary else []
    # Resolve the arXiv fallback only if the primary source fails or is unsupported.
    attempted, unavailable, fallback_checked = False, False, False
    while candidates or not fallback_checked:
        if not candidates:
            fallback_checked = True
            fallback = arxiv_fallback(paper) if sources.is_external(paper.arxiv_id) else None
            if not fallback:
                break
            candidates.append(fallback)
        url, kind, source_id, license_url = candidates.pop(0)
        attempted = True
        cache_key = 'chat-figures:v3:' + hashlib.sha256(url.encode()).hexdigest()
        figures = cache.get(cache_key)
        if figures is None:
            try:
                figures = extract(fetch_html(url, kind), url, kind)
                cache.set(cache_key, figures, 86400 if figures else 600)
            except (requests.RequestException, ValueError):
                unavailable = True
                continue
        if figures:
            figures = [dict(f, source_paper_id=source_id, license_url=license_url) for f in figures]
            return dict(data, status='ready', source_url=url, figures=figures)
    return dict(data, status='unavailable' if unavailable else 'empty' if attempted else 'unsupported')


@require_GET
def gallery(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None:
        return JsonResponse({'status': 'not_ready'}, status=404)
    data = paper_gallery(paper)
    return JsonResponse(data, status=503 if data['status'] == 'unavailable' else 200)
