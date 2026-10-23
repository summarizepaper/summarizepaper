"""'What did I miss?': papers close to a draft that the draft does not cite (Premium).

1. The model reads the start of the draft and writes search queries.
2. arXiv and the open-access journals and preprints (OpenAlex, see sources.py)
   answer them; the candidates are ranked by closeness to the draft
   (embeddings) and the papers the draft already cites are dropped, matched
   by arXiv id, DOI or title in the draft's text.
3. The model reads the closest ones and says, for each, how it relates to
   the draft and how to position the draft against it.
"""
import datetime
import logging
import re
import uuid
from io import BytesIO, StringIO

import numpy as np
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, digest, generated, similar, sources
from .models import Generated

logger = logging.getLogger(__name__)

MAX_UPLOAD = 20 * 1024 * 1024
MAX_TEXT_CHARS = 300_000
QUERIES = 6
RESULTS_PER_QUERY = 30
OPEN_RESULTS_PER_QUERY = 15  # OpenAlex: each search costs, see sources.OPENALEX_SEARCHES_PER_DAY
SHOWN = 15
PER_DAY = 10
QUOTA_KEY = 'missed-quota:%s:%s'
STEPS = [
    ('read', _('Reading your draft')),
    ('search', _('Searching arXiv and the journals for close papers')),
    ('rank', _('Ranking them and removing the papers you already cite')),
    ('review', _('An AI reads the closest papers and explains each one')),
]


def pdf_text(data):
    """All the text of a PDF, references included (they tell us what the draft cites)."""
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    manager = PDFResourceManager(caching=False)
    out = StringIO()
    converter = TextConverter(manager, out, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    for page in PDFPage.get_pages(BytesIO(data)):
        interpreter.process_page(page)
    converter.close()
    return out.getvalue().replace('\x00', '')


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def cited(candidate, draft_ids, draft_words, draft_dois=frozenset()):
    """True if the draft already cites this paper (its arXiv id, DOI or title appears in the draft)."""
    base = re.sub(r'v\d+$', '', candidate['arxiv_id'].replace('--', '/'))
    if base in draft_ids or (candidate.get('doi') and candidate['doi'] in draft_dois):
        return True
    title = normalized(candidate['title'])
    return len(title) > 20 and title in draft_words


def search(queries):
    """Candidates from arXiv and the open-access journals and preprints for these queries, one per paper."""
    found = {}
    for query in queries[:QUERIES]:
        terms = ' AND '.join('all:%s' % w for w in re.findall(r'[\w\-]+', query)[:6])
        if not terms:
            continue
        try:
            papers = digest.arxiv_get({'search_query': terms, 'max_results': RESULTS_PER_QUERY, 'sortBy': 'relevance'})
        except digest.ArxivUnavailable:
            logger.warning('arXiv search failed for %s', query)
            continue
        for paper in papers:
            base = re.sub(r'v\d+$', '', paper['arxiv_id'])
            found.setdefault(base, paper)
        try:
            others = sources.search_papers(query, per_page=OPEN_RESULTS_PER_QUERY, abstracts=True)
        except sources.SourceUnavailable:
            logger.warning('Open-access search failed for %s', query)
            others = []
        for paper in others:
            if not paper['abstract'] or (paper['arxiv_id'] and paper['arxiv_id'].replace('/', '--') in found):
                continue  # nothing to rank it by, or arXiv already gave it
            found.setdefault(paper['key'], {
                'arxiv_id': paper['key'], 'title': paper['title'], 'abstract': paper['abstract'],
                'authors': paper['authors'], 'doi': paper['doi'],
                'published': datetime.datetime.fromisoformat(paper['published']) if paper['published'] else None})
    return list(found.values())


def find(text, lang, step=lambda name: None):
    """The result shown on the page, from the draft's text; step(name) reports progress."""
    text = text[:MAX_TEXT_CHARS]
    head = text[:12000]
    plan = generated.chat_json(
        'Above is the beginning of a research draft. Answer with a JSON object {"title": "its title or a short '
        'description", "summary": "what it does, in 3 sentences, in English", "queries": ["%d literature search queries of '
        '2 to 5 keywords each, in English, covering its topic, its method and its closest problems"]}.' % QUERIES,
        paid=False, max_tokens=800, system='<draft>\n%s\n</draft>' % head)
    queries = [str(q) for q in plan.get('queries', []) if str(q).strip()]
    step('search')
    candidates = search(queries)
    step('rank')
    draft_ids = {m.replace('--', '/') for m in digest.ARXIV_ID_RE.findall(text)}
    draft_ids |= {re.sub(r'v\d+$', '', m) for m in draft_ids}
    draft_words = normalized(text)
    draft_dois = {m.group(1).rstrip('.,;)]').lower() for m in sources.DOI_RE.finditer(text)}
    fresh = [c for c in candidates if not cited(c, draft_ids, draft_words, draft_dois)]
    if not fresh:
        return {'title': plan.get('title', ''), 'items': [], 'searched': len(candidates), 'cited': len(candidates)}
    vectors = similar.embed(['%s\n\n%s' % (plan.get('title', ''), plan.get('summary', '') or head[:4000])] +
                            ['%s\n\n%s' % (c['title'], c['abstract']) for c in fresh])
    scores = vectors[1:] @ vectors[0]
    order = list(np.argsort(-scores))[:SHOWN]
    closest = [dict(fresh[i], similarity=float(scores[i])) for i in order]
    listing = '\n\n'.join('<paper id="%s">\nTitle: %s\nAbstract: %s\n</paper>' % (c['arxiv_id'], c['title'], c['abstract'])
                          for c in closest)
    step('review')
    notes = generated.chat_json(
        'The draft is summarized here: %s\n\nThe papers above are close to it and the draft does not cite them. '
        'In %s, for each paper, say in one or two sentences how it relates to the draft (same problem, competing '
        'method, earlier result, data set to use...) and how the draft could position itself against it. Rate how '
        'important it is for the draft to cite it: "high", "medium" or "low". Answer with a JSON object '
        '{"items": [{"id": "paper id", "relation": "...", "importance": "high|medium|low"}]}.'
        % (plan.get('summary', ''), generated.language_name(lang)),
        paid=True, max_tokens=4000, system=listing)
    by_id = {n['id']: n for n in notes.get('items', []) if isinstance(n, dict) and n.get('id')}
    rank = {'high': 0, 'medium': 1, 'low': 2}
    items = []
    for c in closest:
        note = by_id.get(c['arxiv_id'], {})
        items.append({
            'arxiv_id': c['arxiv_id'], 'title': c['title'],
            'authors': ', '.join(c['authors'][:5]) + (' et al.' if len(c['authors']) > 5 else ''),
            'year': c['published'].year if c.get('published') else '',
            'similarity': round(100 * c['similarity']),
            'relation': str(note.get('relation', '')).strip(),
            'importance': note.get('importance') if note.get('importance') in rank else 'low',
        })
    items.sort(key=lambda i: (rank[i['importance']], -i['similarity']))
    return {'title': str(plan.get('title', '')).strip(), 'items': items, 'searched': len(candidates),
            'cited': len(candidates) - len(fresh)}


def write(data, text, lang):
    def work(pk):
        generated.progress(pk, 'read')
        draft = pdf_text(data) if data is not None else text
        if len(draft.strip()) < 200:
            raise generated.GenerationError('no text found in the draft')
        return find(draft, lang, lambda name: generated.progress(pk, name))
    return work


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def missed_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/missed.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'missed')
    error = None
    if request.method == 'POST' and paid:
        upload = request.FILES.get('draft')
        text = request.POST.get('text', '').strip()
        if upload is not None and (upload.size > MAX_UPLOAD or not upload.name.lower().endswith('.pdf')):
            error = 'file'
        elif upload is None and len(text) < 200:
            error = 'empty'
        elif not _spend(request.user):
            error = 'quota'
        else:
            data = upload.read() if upload is not None else None
            row = generated.start('missed', uuid.uuid4().hex, get_language(), write(data, text, get_language()),
                                  user=request.user)
            return redirect('/%s/missed/%s/' % (get_language(), row.key))
    searches = Generated.objects.filter(kind='missed', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/missed.html', {'paid': paid, 'error': error, 'searches': searches,
                                                      'per_day': PER_DAY, 'upload_max_bytes': MAX_UPLOAD, 'upload_max_mb': MAX_UPLOAD // (1024 * 1024)})


@login_required
def missed_result(request, key):
    if not access.has_tool(request.user, 'missed'):
        raise Http404
    row = Generated.objects.filter(kind='missed', key=key, user=request.user).first()
    if row is None:
        raise Http404
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    return render(request, 'summarizer/missed_result.html', {
        'row': row, 'result': row.data() if row.status == 'done' else None, 'paid': access.has_tool(request.user, 'missed'),
        'steps': STEPS})
