"""Full text of a paper, as the language model will read it.

arXiv renders most LaTeX submissions since late 2023 as HTML
(arxiv.org/html/<id>). That HTML keeps the section structure and gives the
LaTeX source of every equation, which the PDF extraction mangles, so it is
tried first and the PDF is the fallback.

PaperText stores the text that was read, so answering a question never
needs the FAISS index. Papers summarized before PaperText existed get their
text rebuilt once from the chunks stored in their FAISS docstore.
"""
import logging
import pickle
import re

import requests
from bs4 import BeautifulSoup

from .models import ArxivPaper, PaperText, PickledData

logger = logging.getLogger(__name__)

ARXIV_HTML = 'https://arxiv.org/html/%s'


def arxiv_ref(arxiv_id):
    """Our URL form 'hep-th--9901001v1' -> arXiv's 'hep-th/9901001v1'."""
    return arxiv_id.replace('--', '/')


def html_to_text(html):
    soup = BeautifulSoup(html, 'html.parser')
    article = soup.find('article') or soup.body or soup
    for selector in ('.ltx_bibliography', '.ltx_page_footer', '.ltx_authors', 'nav', 'script', 'style',
                     '.ltx_role_footnote .ltx_tag', 'figure.ltx_figure img', '.ltx_note_mark'):
        for node in article.select(selector):
            node.decompose()
    # equations: keep their LaTeX source
    for math in article.find_all('math'):
        tex = math.get('alttext')
        display = math.get('display') == 'block'
        math.replace_with(('$$%s$$' if display else '$%s$') % tex if tex else math.get_text())
    for heading in article.find_all(re.compile(r'^h[1-6]$')):
        heading.insert_before('\n\n## ')
        heading.insert_after('\n')
    for block in article.find_all(['p', 'div', 'figcaption', 'li', 'table']):
        block.insert_after('\n')
    text = article.get_text()
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n\s*(\n\s*)+', '\n\n', text)
    return text.strip()


def fetch_arxiv_html(arxiv_id):
    """Text of the arXiv HTML rendering, or None when the paper has none."""
    try:
        response = requests.get(ARXIV_HTML % arxiv_ref(arxiv_id), timeout=60,
                                headers={'User-Agent': 'SummarizePaper (https://www.summarizepaper.com)'})
    except requests.RequestException:
        logger.exception('arXiv HTML fetch failed for %s', arxiv_id)
        return None
    # papers without an HTML rendering redirect to the abstract page or return 404
    if response.status_code != 200 or '/html/' not in response.url:
        return None
    text = html_to_text(response.text)
    return text if len(text) > 2000 else None


def strip_api_answer(text):
    """The indexed text starts with the raw arXiv API answer; the paper header replaces it."""
    if text.lstrip().startswith('<?xml') and '</feed>' in text:
        text = text.split('</feed>', 1)[1]
    return text.strip()


def store_text(arxiv_id, text, source):
    PaperText.objects.update_or_create(arxiv_id=arxiv_id, defaults={'text': strip_api_answer(text), 'source': source})


def _merge_chunks(chunks):
    """Join index chunks, dropping the overlap each chunk shares with the previous one."""
    merged = ''
    for chunk in chunks:
        if not merged:
            merged = chunk
            continue
        # chunks were split on blank lines, so the overlap starts with a whole paragraph
        probe = chunk.split('\n\n', 1)[0][:200]
        start = merged.find(probe, max(0, len(merged) - len(chunk)))
        if probe and start >= 0 and chunk.startswith(merged[start:]):
            merged += chunk[len(merged) - start:]
        else:
            merged += '\n\n' + chunk
    return merged


def text_from_index(arxiv_id):
    stored = PickledData.objects.filter(arxiv_id=arxiv_id).first()
    if stored is None:
        return None
    try:
        docstore = pickle.loads(stored.docstore_pickle)
        order = pickle.loads(stored.index_to_docstore_id_pickle)
        chunks = [docstore._dict[order[i]].page_content for i in sorted(order)]
    except Exception:
        logger.exception('Could not read the stored index of %s', arxiv_id)
        return None
    return strip_api_answer(_merge_chunks(chunks))


def paper_text(arxiv_id):
    """Best available text of a paper: stored text, rebuilt index text, or the abstract."""
    row = PaperText.objects.filter(arxiv_id=arxiv_id).first()
    if row is not None:
        return row.text
    text = text_from_index(arxiv_id)
    if text:
        store_text(arxiv_id, text, 'index')
        return text
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    return (paper.abstract or '') if paper else ''
