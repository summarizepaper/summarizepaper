"""The card shown on arXiv's own abstract pages by the SummarizePaper toggle of arXivLabs (arxiv-browse).

arXiv's page asks /api/arxiv-labs/<arXiv id> from the reader's browser once the reader turns the toggle on:
a public JSON answer, readable from any origin, with no cookie and no session, that never starts a generation
and records nothing. It gives the essentials already written for the paper in English: the structured reading
(insight.py: one sentence, question, findings, limitations) when the paper has one, else the opening of the
summary and its key points, else only the link to the paper's page, where a summary can be made.
"""
import re
from urllib.parse import urlencode

from django.http import JsonResponse
from django.urls import reverse
from django.utils import translation
from django.views.decorators.http import require_GET

from . import digest, insight, keypoints, loadshed, sources, unified_search
from .models import ArxivPaper, Generated, SummaryPaper

LANG = 'en'  # arXiv's pages are in English
SITE = 'https://www.summarizepaper.com'
MAX_FINDINGS = 4
MAX_POINTS = 4
SUMMARY_CHARS = 500
MAX_AGE = 3600  # seconds: a new summary shows up within the hour
VERSION = re.compile(r'v(\d+)$')
SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z])')


def versions(reference):
    """Our ids of the paper ('hep-th--9901001v2' form), the asked version first, then the newest ones."""
    asked = digest.site_id(reference)
    base = VERSION.sub('', asked)
    known = ArxivPaper.objects.filter(arxiv_id__startswith=base + 'v').values_list('arxiv_id', flat=True)
    known = sorted((k for k in known if VERSION.sub('', k) == base), key=lambda k: -int(VERSION.search(k)[1]))
    return ([asked] if asked in known else []) + [k for k in known if k != asked], asked if asked != base else base + 'v1'


def plain(text):
    """arXiv's page does not typeset what a Labs script adds: accents and formulas as text."""
    return sources.plain_math(sources.latex_accents(str(text or '')))


def opening(text, limit=SUMMARY_CHARS):
    """The first sentences of a summary, whole, within `limit` characters."""
    text = plain(' '.join(keypoints.clean_point(line) for line in str(text or '').splitlines() if line.strip()))
    if len(text) <= limit:
        return text
    kept = ''
    for sentence in SENTENCE_END.split(text):
        if len(kept) + len(sentence) + 1 > limit:
            break
        kept = (kept + ' ' + sentence).strip()
    return kept or text[:limit].rsplit(' ', 1)[0] + '…'


def repeats(point, text):
    """Whether a key point says again a sentence of the opening shown above it (older summaries open with them)."""
    words = re.findall(r'\w+', point.lower())
    if len(words) < 6:
        return False
    for sentence in SENTENCE_END.split(text):
        said = set(re.findall(r'\w+', sentence.lower()))
        if sum(word in said for word in words) >= 0.85 * len(words):
            return True
    return False


def page_url(arxiv_id):
    """The paper's page, counted as a visit from arXivLabs (utm_source), with the pass that spares a reader
    coming from arXiv the page asking new visitors for our cookie (loadshed.first_visit)."""
    with translation.override(LANG):
        path = reverse('arxividpage', kwargs={'arxiv_id': arxiv_id})
    return '%s%s?%s' % (SITE, path, urlencode({'utm_source': 'arxiv-labs', 'labs': loadshed.labs_token(path)}))


def reading(arxiv_id):
    row = Generated.objects.filter(kind=insight.KIND, key=arxiv_id, lang=LANG, status='done', user=None).first()
    data = row.data() if row else None
    if not (isinstance(data, dict) and data.get('tldr') and data.get('findings')):
        return None
    return {'kind': 'reading', 'tldr': plain(data['tldr']), 'question': plain(data.get('question')),
            'findings': [{'text': plain(f['text']), 'kind': f.get('kind', 'interpretation')} for f in data['findings'][:MAX_FINDINGS]],
            'limitations': [plain(item['text']) for item in data.get('limitations') or []][:1],
            'scope': 'abstract' if data.get('abstract_only') else 'full_text'}


def summary(arxiv_id):
    row = SummaryPaper.objects.filter(paper__arxiv_id=arxiv_id, lang=LANG).exclude(summary__isnull=True).exclude(summary='').first()
    if not row or row.summary.startswith('Error'):
        return None
    text = opening(row.summary)
    points = [p for p in map(plain, keypoints.stored_points(row.notes)) if not repeats(p, text)]
    return {'kind': 'summary', 'summary': text, 'key_points': points[:MAX_POINTS]}


def card(arxiv_id_candidates, fallback):
    for arxiv_id in arxiv_id_candidates:
        found = reading(arxiv_id)
        if found:
            break
    else:
        found = None
        for arxiv_id in arxiv_id_candidates:
            found = summary(arxiv_id)
            if found:
                break
    if not found:
        return {'available': False, 'arxiv_id': fallback.replace('--', '/'), 'url': page_url(fallback)}
    return dict(found, available=True, arxiv_id=arxiv_id.replace('--', '/'), version=int(VERSION.search(arxiv_id)[1]),
                url=page_url(arxiv_id))


@require_GET
def labs_card(request, reference):
    found = unified_search.arxiv_reference(reference)
    if found:
        candidates, fallback = versions(found)
        response = JsonResponse(card(candidates, fallback))
    else:
        response = JsonResponse({'error': 'Not an arXiv identifier.'}, status=404)
    response['Access-Control-Allow-Origin'] = '*'  # public data, asked without credentials
    response['Cache-Control'] = 'public, max-age=%d' % MAX_AGE
    return response
