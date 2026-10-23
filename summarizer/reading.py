"""A paper's summary rewritten for another reader: another level, another language.

Each version is written once from the stored summary and abstract (not the
full text, so it is quick) and shared by every reader. Without Premium a
visitor may ask for a few new versions a day; versions already written are
free to read.
"""
import re

from django.core.cache import cache
from django.http import Http404, JsonResponse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, generated
from .models import ArxivPaper, Generated, SiteSettings, SummaryPaper

LEVELS = [
    ('child', _('For a 12-year-old')),
    ('highschool', _('High school')),
    ('undergrad', _('Undergraduate')),
    ('expert', _('Expert (dense, technical)')),
]
LEVEL_PROMPTS = {
    'child': 'a curious 12-year-old: short sentences, everyday comparisons, no technical words left unexplained',
    'highschool': 'a high school student who likes science: plain words, one or two key terms explained',
    'undergrad': 'an undergraduate student in a neighbouring field: the main concepts named and explained briefly',
    'expert': 'a researcher of the field: dense and technical, methods, numbers and caveats, no popularisation',
}
OUTPUT_LANGUAGES = [
    ('en', 'English'), ('fr', 'Français'), ('es', 'Español'), ('de', 'Deutsch'), ('it', 'Italiano'),
    ('pt', 'Português'), ('nl', 'Nederlands'), ('pl', 'Polski'), ('tr', 'Türkçe'), ('ru', 'Русский'),
    ('uk', 'Українська'), ('ar', 'العربية'), ('hi', 'हिन्दी'), ('zh-hans', '中文'), ('ja', '日本語'), ('ko', '한국어'),
]
LANGUAGE_NAMES = {
    'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German', 'it': 'Italian', 'pt': 'Portuguese',
    'nl': 'Dutch', 'pl': 'Polish', 'tr': 'Turkish', 'ru': 'Russian', 'uk': 'Ukrainian', 'ar': 'Arabic', 'hi': 'Hindi',
    'zh-hans': 'Simplified Chinese', 'ja': 'Japanese', 'ko': 'Korean',
}
QUOTA_KEY = 'reading-quota:%s:%s'


def source_text(paper):
    """The longest summary we have of the paper, in any language, and its abstract."""
    summaries = SummaryPaper.objects.filter(paper=paper).exclude(summary__isnull=True).exclude(summary='') \
        .exclude(summary__startswith='Error').values_list('summary', flat=True)
    summary = max(summaries, key=len, default='')
    return 'Title: %s\n\nAbstract: %s\n\nSummary: %s' % (
        ' '.join((paper.title or '').split()), ' '.join((paper.abstract or '').split()), summary)


def write(arxiv_id, level, out_lang):
    def work(pk):
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        text = source_text(paper)
        generated.release_connection()
        answer = generated.chat_json(
            'Rewrite what this paper found for %s. Write in %s, 150 to 300 words, in 2 to 4 short paragraphs '
            'separated by blank lines. Stay faithful to the paper: no claim it does not make. No LaTeX, no '
            'markdown. Answer with a JSON object {"title": "a short title", "text": "..."}.\n\n%s'
            % (LEVEL_PROMPTS[level], LANGUAGE_NAMES[out_lang], text), paid=False, max_tokens=1500)
        title, body = str(answer.get('title', '')).strip(), str(answer.get('text', '')).strip()
        if not body:
            raise generated.GenerationError('empty answer')
        return {'title': title, 'paragraphs': [p.strip() for p in re.split(r'\n\s*\n', body) if p.strip()]}
    return work


def quota_left(request):
    """New versions this visitor may still ask for today (None: unlimited)."""
    if access.has_tool(request.user, 'reading') or request.user.is_superuser:
        return None
    who = request.user.pk if request.user.is_authenticated else access.hash_ip(access.client_ip(request))
    used = cache.get(QUOTA_KEY % (who, timezone.now().date()), 0)
    return max(0, SiteSettings.current().free_readings_per_day - used)


def _spend(request):
    who = request.user.pk if request.user.is_authenticated else access.hash_ip(access.client_ip(request))
    key = QUOTA_KEY % (who, timezone.now().date())
    cache.add(key, 0, 86400)
    cache.incr(key)


@require_POST
def reading_view(request, arxiv_id):
    level, out_lang = request.POST.get('level'), request.POST.get('lang')
    if level not in LEVEL_PROMPTS or out_lang not in LANGUAGE_NAMES:
        return JsonResponse({'error': 'bad request'}, status=400)
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None:
        raise Http404
    key = '%s:%s' % (arxiv_id, level)
    row = Generated.objects.filter(kind='reading', key=key, lang=out_lang).first()
    if row is None or row.status == 'failed':
        left = quota_left(request)
        if left is not None and left <= 0:
            return JsonResponse({'status': 'quota'}, status=429)
        if left is not None:
            _spend(request)
    if row is None or row.status != 'done':
        row = generated.start('reading', key, out_lang, write(arxiv_id, level, out_lang))
    return JsonResponse({'status': row.status, 'id': row.pk, 'data': row.data() if row.status == 'done' else None})
