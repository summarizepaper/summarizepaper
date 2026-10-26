"""Bounded visual evidence from figures belonging to the current papers only.

The client/model cannot supply an image URL. Images are fetched without redirects,
decoded and re-encoded before they become API inputs. No image persists in history.
"""
import base64
import hashlib
import io
import json
import re
import time

import requests
from django.core.cache import cache
from PIL import Image, ImageOps
from lxml import etree

from . import chat_figures, sources
from .models import ArxivPaper
from .slides_pptx import ExportError, svg_png

MAX_FIGURES = 2
MAX_IMAGES = 4
MAX_BYTES = 4 * 1024 * 1024
MAX_PIXELS = 12_000_000
VISUAL = re.compile(r'\b(?:figures?|figs?\.?|plots?|graphs?|graphiques?|courbes?|images?|panels?|panneaux|diagram\w*)\b', re.I)
NUMBER = re.compile(r'\b(?:figures?|figs?\.?)\s*([A-Za-z]?\d+(?:[.\-]\d+)?)', re.I)
NUMBER_LIST = re.compile(r'\b(?:figures?|figs?\.?)\s*([A-Za-z]?\d+(?:[.\-]\d+)?(?:\s*(?:,|and|et|&)\s*[A-Za-z]?\d+(?:[.\-]\d+)?)*)', re.I)


def image_data(url):
    if not chat_figures.safe_image(url):
        return None
    key = 'chat-vision:image:v1:' + hashlib.sha256(url.encode()).hexdigest()
    result = cache.get(key)
    if result is not None:
        return result
    try:
        deadline = time.monotonic() + 10
        with requests.get(url, timeout=(2, 5), allow_redirects=False, stream=True,
                          headers={'User-Agent': sources.USER_AGENT}) as response:
            if response.status_code != 200:
                return None
            data = bytearray()
            for chunk in response.iter_content(65536):
                data.extend(chunk)
                if len(data) > MAX_BYTES or time.monotonic() > deadline:
                    return None
        data = bytes(data)
        if data.lstrip().startswith(b'<'):
            root = etree.fromstring(data, parser=etree.XMLParser(resolve_entities=False, no_network=True))
            if root.getroottree().docinfo.doctype:
                return None
            # arXiv often uses pt units; the rasterizer needs numeric pixel dimensions.
            box = (root.get('viewBox') or '').replace(',', ' ').split()
            if len(box) == 4:
                root.set('width', box[2])
                root.set('height', box[3])
            data = svg_png(etree.tostring(root), width_limit=2560, height_limit=2560)
        with Image.open(io.BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS:
                return None
            image = ImageOps.exif_transpose(image)
            image.thumbnail((2560, 2560))
            output = io.BytesIO()
            rgba = image.convert('RGBA')
            background = Image.new('RGBA', rgba.size, 'white')
            Image.alpha_composite(background, rgba).convert('RGB').save(output, format='PNG')
            result = output.getvalue()
        if len(result) > MAX_BYTES:
            return None
        # Avoid large cache entries for unusually complex plots.
        if len(result) < 256 * 1024:
            cache.set(key, result, 3600)
        return result
    except (requests.RequestException, OSError, ValueError, RuntimeError, ExportError,
            Image.DecompressionBombError, etree.XMLSyntaxError):
        return None


def prepare(arxiv_ids, question):
    """Return image parts plus the exact public figures actually sent to the model.

    Explicit figure numbers take precedence; otherwise rank captions for a visual
    question. Ordinary text questions never trigger extra downloads or model calls.
    """
    if not VISUAL.search(question):
        return [], [], 'not_requested'
    known = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=arxiv_ids)}
    papers = [known[key] for key in dict.fromkeys(arxiv_ids) if key in known]
    # The gallery's Ask button includes the paper ID, making multi-paper references unambiguous.
    named = [p for p in papers if p.arxiv_id in question or p.arxiv_id.replace('--', '/') in question]
    papers = named or papers
    numbers = {n.lower() for group in NUMBER_LIST.findall(question)
               for n in re.findall(r'[A-Za-z]?\d+(?:[.\-]\d+)?', group)}
    words = set(re.findall(r'\w{4,}', question.lower()))
    candidates = []
    for paper in papers:
        if not sources.is_open_license(paper.license):
            continue
        data = chat_figures.paper_gallery(paper)
        for figure in data.get('figures', []):
            label = NUMBER.search(figure['label'])
            if numbers and (not label or label[1].lower() not in numbers):
                continue
            score = len(words.intersection(re.findall(r'\w{4,}', figure['caption'].lower())))
            candidates.append((score, {'paper_id': paper.arxiv_id, 'title': data['title'], 'figure': figure}))
    candidates.sort(key=lambda item: item[0], reverse=True)
    chosen = [item for _, item in candidates[:MAX_FIGURES]]
    jobs = []
    for index, item in enumerate(chosen):
        for url in item['figure']['images'][:MAX_IMAGES // len(chosen)]:
            if len(jobs) >= MAX_IMAGES:
                break
            jobs.append((index, url))
    # Decode one image at a time to keep peak memory bounded on the web dyno.
    images = [image_data(url) for _, url in jobs]
    parts, used = [], []
    for index, item in enumerate(chosen):
        successful = [(url, image) for (owner, url), image in zip(jobs, images) if owner == index and image]
        if not successful:
            continue
        figure = dict(item['figure'], images=[url for url, _ in successful])
        used.append(dict(item, figure=figure))
        # Captions remain source data, never application/system instructions.
        parts.append({'type': 'text', 'text': json.dumps({
            'figure_source': {'paper_id': item['paper_id'], 'title': item['title'],
                              'label': figure['label'], 'caption': figure['caption'],
                              'source_url': figure['source_url']},
            'panels_supplied': len(successful), 'panels_total': len(item['figure']['images'])}, ensure_ascii=False)})
        parts.extend({'type': 'image_url', 'image_url': {
            'url': 'data:image/png;base64,' + base64.b64encode(image).decode('ascii'), 'detail': 'high'}}
            for _, image in successful)
    return parts, used, 'ready' if used else 'unavailable'
