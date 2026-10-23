"""Public figure galleries. Only known, openly licensed papers and trusted hosts.

No model-generated image URLs, publisher scraping, or implicit paper generation.
"""
import hashlib
import json
import re
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup
from django.core.cache import cache
from django.http import JsonResponse
from django.views.decorators.http import require_GET

from . import sources, unified_search
from .models import ArxivPaper, ExternalPaper

MAX_HTML = 6_000_000
MAX_FIGURES = 40
IMAGE_HOSTS = {'arxiv.org', 'www.arxiv.org', 'pmc.ncbi.nlm.nih.gov', 'cdn.ncbi.nlm.nih.gov'}


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
    return None


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
        images = []
        for node in figure.find_all(['img', 'object']):
            source = node.get('src') or node.get('data-src') or node.get('data')
            if not source:
                continue
            url = urljoin(base, source)
            if safe_image(url) and url not in images:
                images.append(url)
            if len(images) == 8:
                break
        if not images:
            continue
        text = ' '.join(caption.get_text(' ', strip=True).split())[:5000]
        tag = figure.select_one('.ltx_tag_figure')
        match = re.match(r'\s*((?:Figure|Fig\.)\s*[A-Za-z]?\d+(?:[.\-]\d+)?)', text, flags=re.I)
        label = tag.get_text(' ', strip=True) if tag else (match.group(1) if match else 'Figure %d' % index)
        anchor = figure.get('id', '')
        found.append({'number': len(found) + 1, 'label': label[:80], 'images': images,
                      'caption': text, 'source_url': base + ('#' + anchor if re.fullmatch(r'[\w.:-]+', anchor) else '')})
        if len(found) == MAX_FIGURES:
            break
    return found


@require_GET
def gallery(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None:
        return JsonResponse({'status': 'not_ready'}, status=404)
    data = {'paper_id': paper.arxiv_id, 'title': paper.title or paper.arxiv_id,
            'source_url': sources.ref_url(paper.arxiv_id), 'figures': []}
    if not sources.is_open_license(paper.license):
        return JsonResponse(dict(data, status='license'))
    source = html_source(paper)
    if source is None:
        return JsonResponse(dict(data, status='unsupported'))
    url, kind = source
    cache_key = 'chat-figures:v1:' + hashlib.sha256(url.encode()).hexdigest()
    figures = cache.get(cache_key)
    if figures is None:
        try:
            # Never follow a redirect supplied by external content, and bound downloaded HTML.
            with requests.get(url, timeout=(4, 10), allow_redirects=False, stream=True,
                              headers={'User-Agent': sources.USER_AGENT}) as response:
                if response.status_code in (301, 302, 303, 307, 308, 404):
                    figures = []
                else:
                    response.raise_for_status()
                    chunks, size = [], 0
                    for chunk in response.iter_content(65536):
                        size += len(chunk)
                        if size > MAX_HTML:
                            raise ValueError('HTML too large')
                        chunks.append(chunk)
                    figures = extract(b''.join(chunks), url, kind)
            cache.set(cache_key, figures, 86400 if figures else 600)
        except (requests.RequestException, ValueError):
            return JsonResponse(dict(data, status='unavailable'), status=503)
    return JsonResponse(dict(data, status='ready' if figures else 'empty', figures=figures))
