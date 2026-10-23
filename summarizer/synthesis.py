"""Public progress for a public paper; no user/session data or generation on GET."""
from django.core.cache import cache
from django.http import Http404, JsonResponse
from django.views.decorators.http import require_GET

from . import sources, utils
from .models import ArxivPaper, SummaryPaper, PaperText


def cache_key(group):
    return 'synthesis:v1:' + group


def remember(group, field, value):
    state = cache.get(cache_key(group)) or {}
    state[field] = value
    cache.set(cache_key(group), state, 1200)


def snapshot(arxiv_id, lang):
    group = utils.run_group(arxiv_id, lang)
    data = dict(cache.get(cache_key(group)) or {})
    running = bool(cache.get(group) or cache.get(group + '-close'))
    progress = data.get('progress', {})
    if progress.get('progress') == 0:
        status = 'failed'
    elif progress.get('progress') == 100:
        status = 'done'
    else:
        status = 'running' if running else 'idle'
    return dict(data, status=status)


@require_GET
def state(request, arxiv_id):
    data = snapshot(arxiv_id, request.LANGUAGE_CODE)
    if data == {'status': 'idle'} and not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        raise Http404
    return JsonResponse(data, headers={'Cache-Control': 'no-store'})


@require_GET
def fragment(request, arxiv_id):
    if not SummaryPaper.objects.filter(paper__arxiv_id=arxiv_id, lang=request.LANGUAGE_CODE).exists():
        raise Http404
    from .views import arxividpage
    response = arxividpage(request, arxiv_id, _fragment=True)
    response['Cache-Control'] = 'private, no-store'
    return response


def license_scope(paper):
    if paper is None:
        return 'checking'
    if not sources.is_open_license(paper.license):
        return 'abstract_license'
    source = PaperText.objects.filter(arxiv_id=paper.arxiv_id).values_list('source', flat=True).first()
    return ('abstract_unavailable' if source == 'abstract' else 'full_text') if source else 'full_text_allowed'
