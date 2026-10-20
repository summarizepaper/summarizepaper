"""Model-written results made in a background thread, and the model call they share.

A view starts the work and answers at once; the page then polls the status
endpoint until the result is ready. Heroku cuts requests after 30 seconds and
the database pool is small, so nothing slow runs inside a request: the thread
reads what it needs, gives its connection back to the pool during the model
call, and saves the result.

Results are rows of Generated, keyed by kind, key and language, so the same
press kit or reading level is written once and shared by every reader.
"""
import datetime
import json
import logging
import threading

import requests
from django.conf import settings
from django.db import IntegrityError, connections
from django.http import Http404, JsonResponse
from django.utils import timezone

from . import usage
from .models import Generated
from .utils import ai_models, chat_params

logger = logging.getLogger(__name__)

STALE = datetime.timedelta(minutes=10)  # a 'pending' row older than this was lost (dyno restart): start again
MAX_THREADS = 3  # each one holds a database connection now and then; the pool has 8
_slots = threading.BoundedSemaphore(MAX_THREADS)
LANGUAGES = {'en': 'English', 'fr': 'French'}
PREMIUM_KINDS = {'presskit', 'compare', 'missed', 'zotero', 'refcheck', 'changes', 'teams', 'slides', 'conference', 'questions'}  # their results are only shown to Premium accounts


class GenerationError(Exception):
    pass


def language_name(lang):
    from django.utils.translation import get_language_info
    try:
        return get_language_info(lang)['name']
    except KeyError:
        return LANGUAGES.get(lang, 'English')


def chat_json(prompt, paid=True, max_tokens=3000, system=None):
    """One chat completion that answers with a JSON object, returned as a dict."""
    model_free, model_paid = ai_models()
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': prompt}]
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
        json={'model': model_paid if paid else model_free, 'messages': messages,
              'response_format': {'type': 'json_object'}, **chat_params(paid, max_tokens)},
        timeout=300,
    )
    response.raise_for_status()
    content = usage.record(response.json())['choices'][0]['message']['content']
    answer = json.loads(content)
    if not isinstance(answer, dict):
        raise GenerationError('the model did not answer with a JSON object')
    return answer


def background(work, *args):
    """Run work(*args) in a thread that leaves the database pool as it found it."""
    def run():
        with _slots:
            try:
                work(*args)
            except Exception:
                logger.exception('Background work %s failed', getattr(work, '__name__', work))
            finally:
                connections.close_all()
    threading.Thread(target=run, daemon=True).start()


def release_connection():
    """Give the thread's database connection back to the pool before a slow call."""
    connections.close_all()


def progress(pk, step):
    """Tell the waiting page what the work is doing now (and that it is still alive)."""
    Generated.objects.filter(pk=pk).update(step=step, updated=timezone.now())
    release_connection()


def start(kind, key, lang, work, user=None, force=False):
    """The Generated row for (kind, key, lang), with work(row_pk) started in the background unless it is
    already done or already running. work returns the content (a dict) or raises."""
    row = Generated.objects.filter(kind=kind, key=key, lang=lang).first()
    if row is not None:
        if row.status == 'pending' and row.updated > timezone.now() - STALE:
            return row
        if row.status == 'done' and not force:
            return row
    if row is None:
        try:
            row = Generated.objects.create(kind=kind, key=key, lang=lang, user=user)
        except IntegrityError:  # started by another request in the meantime
            return Generated.objects.get(kind=kind, key=key, lang=lang)
    else:
        row.status, row.error, row.step = 'pending', '', ''
        row.save(update_fields=['status', 'error', 'step', 'updated'])
    background(_run, row.pk, work)
    return row


def _run(pk, work):
    kind, user_id = Generated.objects.filter(pk=pk).values_list('kind', 'user_id').first() or ('other', None)
    try:
        with usage.task(kind, user_id):
            content = work(pk)
    except Exception as e:
        logger.exception('Generation %s failed', pk)
        Generated.objects.filter(pk=pk).update(status='failed', error=repr(e)[:2000], updated=timezone.now())
        return
    Generated.objects.filter(pk=pk).update(status='done', content=json.dumps(content), updated=timezone.now())


def run_now(kind, key, lang, work, user=None):
    """Same as start(), in the calling thread (management commands and tests)."""
    row, _ = Generated.objects.get_or_create(kind=kind, key=key, lang=lang, defaults={'user': user})
    if row.status != 'done':
        _run(row.pk, work)
        row.refresh_from_db()
    return row


def status_view(request, pk):
    """{'status', 'data'} of a result; private results only for their owner."""
    from .access import is_paid
    row = Generated.objects.filter(pk=pk).first()
    if row is None or (row.user_id and row.user_id != request.user.id):
        raise Http404
    if row.kind in PREMIUM_KINDS and not is_paid(request.user):
        raise Http404
    if row.status == 'pending' and row.updated < timezone.now() - STALE:
        row.status = 'failed'
    return JsonResponse({'status': row.status, 'step': row.step,
                         'data': row.data() if row.status == 'done' else None})
