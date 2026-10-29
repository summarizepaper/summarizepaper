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
    """Count one call. From async code, the row is written in the loop's thread pool (the ORM is sync only)."""
    current, current_user = _current.get()
    args = (timezone.now().date(), name or current, user_id if user_id is not None else current_user,
            (model or 'unknown')[:100], 1, int(prompt or 0), int(completion or 0))
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        _save(*args)
        return
    loop.run_in_executor(None, _save, *args, True)  # the loop's bounded pool, not a new thread per call


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


TASK_LABELS = {
    'summary': 'Résumés', 'assistant': 'Assistant', 'chat': 'Chat multi-articles',
    'presskit': 'Kit presse', 'journal_club': 'Journal club', 'slides': 'Diapositives',
    'digest': 'Digest quotidien', 'audio': 'Synthèse vocale', 'trends': 'Sujets qui montent',
    'letters': 'Lettres hebdomadaires', 'scoops': 'Veille anti-scoop', 'students': 'Stages de recherche',
    'teams': 'Qui travaille dessus ?', 'refcheck': 'Vérification bibliographique',
    'reading': 'Niveaux de lecture', 'scoop_alerts': 'Alertes anti-scoop',
    'citation_alerts': 'Alertes de citations', 'version_alerts': 'Alertes de versions',
    'publication_alerts': 'Alertes de publication', 'search_watches': 'Recherches enregistrées',
    'letter_emails': 'Envoi des lettres', 'other': 'Autres appels non attribués',
    'insight': 'Fiches L’essentiel', 'evidence': 'Analyse des preuves',
    'table': 'Tableaux comparatifs', 'compare': 'Comparaisons d’articles',
    'questions': 'Questions ouvertes', 'missing': 'Ce qui manque',
    'literature': 'Revue de littérature', 'changes': 'Comparaisons de versions',
    'letter_summaries': 'Sélection des lettres', 'presskit_experts': 'Kit presse · spécialistes',
    'suggest': 'Suggestions de recherche', 'conference': 'Compagnon de conférence',
}


TOOL_GROUPS = {
    'summary': 'summaries', 'insight': 'summaries', 'evidence': 'summaries',
    'presskit': 'presskit', 'presskit_experts': 'presskit',
    'table': 'collections', 'compare': 'collections', 'questions': 'collections', 'missing': 'collections',
    'letters': 'letters', 'letter_summaries': 'letters', 'letter_emails': 'letters',
    'slides': 'slides', 'slides_v2': 'slides',
    'search_watches': 'saved_searches', 'saved_searches': 'saved_searches',
}
TOOL_LABELS = dict(TASK_LABELS, summaries='Résumés et fiches de lecture',
                   collections='Collections', audio='Audio · tous outils confondus')


def price_for(model, prices):
    """Resolve exact models and dated snapshots with the same rule as AIUsage.cost."""
    if model in prices:
        return prices[model]
    return next((prices[name] for name in sorted(prices, key=len, reverse=True)
                 if model.startswith(name + '-')), None)


def costs(row, prices):
    price = price_for(row.model, prices)
    if price is None:
        return None
    return (row.prompt_tokens * float(price.input_per_million) / 1e6,
            row.completion_tokens * float(price.output_per_million) / 1e6)


def summary(days=30, queryset=None, prices=None):
    """Estimated costs for the selected usage, by task, model, task/model, user and day.

    Unknown prices never turn into a known zero: calls and units stay visible,
    with their missing-price count. Totals include only priced usage.
    """
    from .models import AIPrice, AIUsage
    today = timezone.localdate()
    if queryset is None:
        queryset = AIUsage.objects.filter(day__gte=today - timezone.timedelta(days=days - 1), day__lte=today)
    if prices is None:
        prices = {p.model: p for p in AIPrice.objects.all()}
    tables = {name: {} for name in ('tools', 'tasks', 'models', 'combinations', 'users', 'daily')}
    unpriced = set()
    total = dict(calls=0, prompt=0, completion=0, input_cost=0.0, output_cost=0.0,
                 cost=0.0, unpriced_calls=0)
    first = last = None
    for row in queryset.select_related('user').iterator():
        first = min(first, row.day) if first else row.day
        last = max(last, row.day) if last else row.day
        split = costs(row, prices)
        if split is None:
            unpriced.add(row.model)
        values = dict(calls=row.calls, prompt=row.prompt_tokens, completion=row.completion_tokens,
                      input_cost=split[0] if split else 0.0, output_cost=split[1] if split else 0.0,
                      cost=sum(split) if split else 0.0, unpriced_calls=row.calls if split is None else 0)
        keys = dict(tools=TOOL_GROUPS.get(row.task, row.task), tasks=row.task, models=row.model, combinations=(row.task, row.model),
                    users=(row.user_id, row.user.username if row.user else 'Sans utilisateur / traitement automatique'),
                    daily=row.day)
        for key, value in values.items():
            total[key] += value
        for name, key in keys.items():
            entry = tables[name].setdefault(key, dict(name=key, **{k: 0 for k in values}))
            for field, value in values.items():
                entry[field] += value
    for name, table in tables.items():
        for key, entry in table.items():
            entry['average'] = entry['cost'] / entry['calls'] if entry['calls'] and not entry['unpriced_calls'] else None
            entry['share'] = entry['cost'] * 100 / total['cost'] if total['cost'] else 0
            if name == 'combinations':
                entry['task'], entry['model'] = key
                entry['label'] = TASK_LABELS.get(key[0], key[0])
            elif name == 'tools':
                entry['label'] = TOOL_LABELS.get(key, key)
            elif name == 'tasks':
                entry['label'] = TASK_LABELS.get(key, key)
            elif name == 'users':
                entry['user_id'], entry['name'] = key
        tables[name] = sorted(table.values(), key=lambda e: (-e['cost'], -e['prompt'], str(e['name'])))
    tables['daily'].sort(key=lambda e: e['name'], reverse=True)
    return dict(tables, days=days, total=total['cost'], totals=total, unpriced=sorted(unpriced),
                first=first, last=last, users=tables['users'][:15])
