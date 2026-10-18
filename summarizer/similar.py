"""Closest papers, from one embedding of each paper's title and abstract.

Every summarized paper gets a single normalized vector. The closest papers are
the ones of the same arXiv category with the highest cosine similarity, which
works for every paper, whatever its license. PaperScore.score keeps a distance
(lower is closer): 1 - cosine similarity.
"""
import logging
import re

import numpy as np
import requests
from django.conf import settings

from .models import ArxivPaper, PaperEmbedding, PaperScore

logger = logging.getLogger(__name__)

ENDPOINT = 'https://api.openai.com/v1/embeddings'
MODEL = 'text-embedding-3-small'
DIMENSIONS = 512
BATCH = 100
MAX_INPUT_CHARS = 8000
CLOSEST = 15


def paper_input(paper):
    text = '%s\n\n%s' % (paper.title or '', paper.abstract or '')
    return re.sub(r'[ \t\r\f\v]+', ' ', text).strip()[:MAX_INPUT_CHARS]


def embed(texts):
    """Normalized float32 vectors for texts, in order."""
    response = requests.post(ENDPOINT, timeout=60,
                             headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
                             json={'model': MODEL, 'input': texts, 'dimensions': DIMENSIONS})
    response.raise_for_status()
    rows = sorted(response.json()['data'], key=lambda row: row['index'])
    vectors = np.array([row['embedding'] for row in rows], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def store(papers):
    """Embed and save these papers (one API call per BATCH papers)."""
    papers = [p for p in papers if (p.title or p.abstract)]
    for start in range(0, len(papers), BATCH):
        chunk = papers[start:start + BATCH]
        vectors = embed([paper_input(p) for p in chunk])
        for paper, vector in zip(chunk, vectors):
            PaperEmbedding.objects.update_or_create(arxiv_id=paper.arxiv_id, defaults={
                'category': paper.category or '', 'model': MODEL, 'vector': vector.tobytes()})
    return len(papers)


def ensure(paper):
    if not PaperEmbedding.objects.filter(arxiv_id=paper.arxiv_id, model=MODEL).exists():
        store([paper])


def _base(arxiv_id):
    return re.sub(r'v\d+$', '', arxiv_id)


def closest(paper, k=CLOSEST, candidates=None):
    """[(arxiv_id, distance)] of the k closest papers of the same category, one version per paper."""
    ensure(paper)
    own = PaperEmbedding.objects.get(arxiv_id=paper.arxiv_id)
    query = np.frombuffer(own.vector, dtype=np.float32)
    if candidates is None:
        rows = PaperEmbedding.objects.filter(category=own.category, model=MODEL).values_list('arxiv_id', 'vector')
        candidates = [(aid, np.frombuffer(vec, dtype=np.float32)) for aid, vec in rows]
    own_base = _base(paper.arxiv_id)
    candidates = [(aid, vec) for aid, vec in candidates if _base(aid) != own_base]
    if not candidates:
        return []
    similarity = np.stack([vec for _, vec in candidates]) @ query
    found, seen = [], set()
    for i in np.argsort(-similarity):
        aid = candidates[i][0]
        if _base(aid) in seen:
            continue
        seen.add(_base(aid))
        found.append((aid, float(1 - similarity[i])))
        if len(found) == k:
            break
    return found


def save_closest(paper, found):
    PaperScore.objects.filter(from_paper=paper).delete()
    targets = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=[aid for aid, _ in found])}
    PaperScore.objects.bulk_create([PaperScore(from_paper=paper, to_paper=targets[aid], score=distance)
                                    for aid, distance in found if aid in targets])


def find_and_store(arxiv_id, k=CLOSEST):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None:
        return []
    found = closest(paper, k)
    save_closest(paper, found)
    return found


def percent(score):
    """Similarity shown to visitors, from a stored distance."""
    return '%.1f' % (100 * (1 - score))
