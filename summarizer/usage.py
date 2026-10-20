"""What the AI costs, by task, model, day and user: tokens counted from every OpenAI answer.

Each OpenAI answer gives the tokens it used; record() adds them to the day's
AIUsage row of its task (summary, assistant, refcheck, scoops...) and user. The
task and user come from the caller, or from the surrounding `with task(...)`
(generated.py sets it for every background result, the hourly command for each
of its steps). Prices per million tokens are set in the admin (AIPrice), so the
admin shows what each tool costs and who uses it most.

Counting never breaks the call it counts: errors are logged and dropped.
"""
import asyncio
import contextlib
import contextvars
import logging
import threading

from django.db import connections
from django.db.models import F
from django.utils import timezone

logger = logging.getLogger(__name__)

_current = contextvars.ContextVar('ai_task', default=('other', None))


@contextlib.contextmanager
def task(name, user_id=None):
    """Count the AI calls made inside as `name`, for this user."""
    token = _current.set((name, user_id))
    try:
        yield
    finally:
        _current.reset(token)


def _save(day, name, user_id, model, calls, prompt, completion, close=False):
    from .models import AIUsage
    try:
        updated = AIUsage.objects.filter(day=day, task=name, user_id=user_id, model=model).update(
            calls=F('calls') + calls, prompt_tokens=F('prompt_tokens') + prompt,
            completion_tokens=F('completion_tokens') + completion)
        if not updated:
            AIUsage.objects.create(day=day, task=name, user_id=user_id, model=model, calls=calls,
                                   prompt_tokens=prompt, completion_tokens=completion)
    except Exception:
        logger.exception('AI usage could not be saved')
    finally:
        if close:
            connections.close_all()


def add(model, prompt, completion, name=None, user_id=None):
    """Count one call. From async code, the row is written in a thread (the ORM is sync only)."""
    current, current_user = _current.get()
    args = (timezone.now().date(), name or current, user_id if user_id is not None else current_user,
            (model or 'unknown')[:100], 1, int(prompt or 0), int(completion or 0))
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        _save(*args)
        return
    threading.Thread(target=_save, args=args + (True,), daemon=True).start()


def record(body, name=None, user_id=None):
    """Count the tokens of an OpenAI answer (its JSON body as a dict); returns the body unchanged."""
    if not isinstance(body, dict):
        return body
    try:
        used = body.get('usage') or {}
        add(body.get('model'), used.get('prompt_tokens') or used.get('input_tokens'),
            used.get('completion_tokens') or used.get('output_tokens'), name, user_id)
    except Exception:
        logger.exception('AI usage could not be counted')
    return body


def summary(days=30):
    """{'tasks': [{task, calls, prompt, completion, cost}], 'users': [...], 'total', 'unpriced'} of the last days,
    the costliest first. A model without a price counts no dollars and is listed in 'unpriced'."""
    from .models import AIPrice, AIUsage
    since = timezone.now().date() - timezone.timedelta(days=days - 1)
    prices = {p.model: p for p in AIPrice.objects.all()}
    tasks, users, unpriced = {}, {}, set()
    for row in AIUsage.objects.filter(day__gte=since).select_related('user'):
        cost = row.cost(prices)
        if cost is None:
            unpriced.add(row.model)
        for key, table in ((row.task, tasks), (row.user.username if row.user else None, users)):
            if key is None:
                continue
            entry = table.setdefault(key, {'name': key, 'calls': 0, 'prompt': 0, 'completion': 0, 'cost': 0.0})
            entry['calls'] += row.calls
            entry['prompt'] += row.prompt_tokens
            entry['completion'] += row.completion_tokens
            entry['cost'] += cost or 0
    order = lambda t: sorted(t.values(), key=lambda e: (-e['cost'], -e['prompt']))
    return {'tasks': order(tasks), 'users': order(users)[:15], 'days': days,
            'total': sum(e['cost'] for e in tasks.values()), 'unpriced': sorted(unpriced)}
