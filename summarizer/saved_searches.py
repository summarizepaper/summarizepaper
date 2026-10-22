"""Private saved searches and opt-in email watches using the shared hourly budget."""
import datetime
import hashlib
import json
import logging
from urllib.parse import urlencode

from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core import signing
from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.db.models import Q, F
from django.http import Http404, HttpResponseBadRequest
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import timezone, translation
from django.utils.translation import get_language, gettext as _
from django.views.decorators.http import require_POST
from django.views.decorators.cache import never_cache

from . import access, mailcheck, search_filters, unified_search
from .models import SavedSearch

logger = logging.getLogger(__name__)
STATE_SALT = 'save-research-search'
UNSUBSCRIBE_SALT = 'pause-research-watch'
MAX_SAVED = 50
MAX_CHECKS_PER_RUN = 10
MAX_EMAIL_ITEMS = 10


def watch_limit(user):
    return 5 if access.is_paid(user) else 1


def fingerprint(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


def action_context(request, state):
    params = {key: state.getlist(key) if key in ('journals', 'exclude_journals') else state[key] for key in state}
    saved = SavedSearch.objects.filter(user=request.user, fingerprint=fingerprint(params)).first() if request.user.is_authenticated else None
    return {'save_token': signing.dumps(params, salt=STATE_SALT, compress=True), 'saved_search': saved,
            'search_return': reverse('search_results') + '?' + state.urlencode()}


def load_params(token):
    try:
        params = signing.loads(token, salt=STATE_SALT, max_age=86400)
        from django.http import QueryDict
        state = QueryDict(urlencode(params, doseq=True))
        form = search_filters.SearchFilters(state)
        if not form.is_valid() or not (state.get('q', '').strip() or state.get('orcid') or state.get('journals')):
            raise ValueError
        if state.get('src', 'all') not in ('all', 'arxiv', 'open') or state.get('field', 'all') not in ('all', 'author', 'title', 'abstract'):
            raise ValueError
        if len(state.get('q', '')) > 2000:
            raise ValueError
        return params
    except (signing.BadSignature, ValueError, TypeError, AttributeError):
        return None


def latest(saved):
    params = saved.params
    form = search_filters.SearchFilters(params)
    if not form.is_valid():
        raise ValueError('Invalid saved search')
    options = form.options()
    options['sort'] = 'newest'
    return unified_search.search(params.get('q', ''), field=params.get('field', 'all'),
                                 source=params.get('src', 'all'), kind=params.get('type', ''), filters=options)


@never_cache
@login_required
@require_POST
def save(request):
    params = load_params(request.POST.get('state', ''))
    if params is None:
        return HttpResponseBadRequest(_('This search link has expired. Run the search again before saving it.'))
    with transaction.atomic():
        User.objects.select_for_update().get(pk=request.user.pk)
        existing = SavedSearch.objects.filter(user=request.user, fingerprint=fingerprint(params)).first()
        if not existing and request.user.saved_searches.count() >= MAX_SAVED:
            messages.error(request, _('You can keep up to 50 saved searches. Remove one before adding another.'))
            return redirect('saved_searches')
        saved, created = SavedSearch.objects.get_or_create(user=request.user, fingerprint=fingerprint(params), defaults={
            'params': params, 'name': ' '.join((params.get('q') or params.get('orcid') or _('Journal search')).split())[:120],
            'lang': get_language() if get_language() in ('en', 'fr') else 'en'})
    if created:
        request.session['saved_search_created'] = True
    messages.success(request, _('Search saved. You can reopen it or enable an email watch below.'))
    return redirect('saved_searches')


@never_cache
@login_required
def manage(request):
    if request.method == 'POST':
        try:
            pk = int(request.POST.get('search', ''))
        except (ValueError, TypeError):
            raise Http404
        saved = get_object_or_404(SavedSearch, user=request.user, pk=pk)
        action = request.POST.get('action')
        if action == 'delete':
            saved.delete()
            messages.success(request, _('Saved search removed. Its watch is stopped.'))
        elif action == 'pause':
            SavedSearch.objects.filter(pk=saved.pk, user=request.user).update(watch_enabled=False)
            messages.success(request, _('Search watch paused.'))
        elif action == 'rename':
            name = ' '.join(request.POST.get('name', '').split())[:120]
            if name:
                saved.name = name
                saved.save(update_fields=['name'])
        elif action == 'enable':
            if not request.user.email or request.user.email.strip().lower() in mailcheck.suppressed():
                messages.error(request, _('A deliverable account email is required to enable a watch.'))
                return redirect('saved_searches')
            if request.user.saved_searches.filter(watch_enabled=True).exclude(pk=saved.pk).count() >= watch_limit(request.user):
                messages.error(request, _('Watch limit reached. Pause another watch or see Premium for more watches.'))
                return redirect('saved_searches')
            # Persist the opt-in immediately. The hourly worker establishes the baseline
            # after a complete source response, retrying outages without blocking the web request.
            with transaction.atomic():
                User.objects.select_for_update().get(pk=request.user.pk)
                saved = get_object_or_404(SavedSearch.objects.select_for_update(), pk=saved.pk, user=request.user)
                count = request.user.saved_searches.filter(watch_enabled=True).exclude(pk=saved.pk).count()
                if count >= watch_limit(request.user):
                    messages.error(request, _('Watch limit reached. Pause another watch or see Premium for more watches.'))
                else:
                    saved.watch_enabled = True
                    saved.frequency = 'daily' if request.POST.get('frequency') == 'daily' else 'weekly'
                    saved.lang = request.POST.get('lang') if request.POST.get('lang') in ('en', 'fr') else get_language()
                    saved.seen = []
                    saved.last_checked = None
                    saved.last_attempt = None
                    saved.save(update_fields=['watch_enabled', 'frequency', 'lang', 'seen', 'last_checked', 'last_attempt'])
                    request.session['search_watch_enabled'] = True
                    messages.success(request, _('Watch enabled. Initialization will run automatically; no existing results will be emailed.'))
        return redirect('saved_searches')
    rows = list(request.user.saved_searches.all())
    eligible = set(request.user.saved_searches.filter(watch_enabled=True).order_by('pk').values_list('pk', flat=True)[:watch_limit(request.user)])
    for row in rows:
        row.watch_available = row.pk in eligible
    return render(request, 'summarizer/saved_searches.html', {'searches': rows, 'watch_limit': watch_limit(request.user),
        'search_saved_event': request.session.pop('saved_search_created', False),
        'watch_enabled_event': request.session.pop('search_watch_enabled', False)})


@never_cache
def unsubscribe(request):
    try:
        pk = signing.loads(request.GET.get('token', ''), salt=UNSUBSCRIBE_SALT, max_age=365 * 86400)
        saved = SavedSearch.objects.get(pk=pk)
    except (signing.BadSignature, SavedSearch.DoesNotExist, ValueError, TypeError):
        raise Http404
    paused = False
    if request.method == 'POST':
        SavedSearch.objects.filter(pk=saved.pk).update(watch_enabled=False)
        paused = True
    return render(request, 'summarizer/search_unsubscribe.html', {'paused': paused})


def run(now, budget, base_url='https://www.summarizepaper.com'):
    """Check bounded latest-match windows, send only unseen records; never modify state after a failed send."""
    if budget <= 0:
        return 0
    sent, checked, eligible = 0, 0, {}
    suppressed = mailcheck.suppressed()
    rows = SavedSearch.objects.filter(watch_enabled=True, user__is_active=True).select_related('user').order_by(F('last_attempt').asc(nulls_first=True), 'last_checked', 'pk')
    for saved in rows.iterator():
        if sent >= budget or checked >= MAX_CHECKS_PER_RUN:
            break
        user = saved.user
        if not user.email or user.email.strip().lower() in suppressed:
            continue
        if user.pk not in eligible:
            eligible[user.pk] = set(user.saved_searches.filter(watch_enabled=True).order_by('pk').values_list('pk', flat=True)[:watch_limit(user)])
        if saved.pk not in eligible[user.pk]:
            continue
        delay = datetime.timedelta(days=1 if saved.frequency == 'daily' else 7)
        if saved.last_checked and saved.last_checked > now - delay:
            continue
        lease = now + datetime.timedelta(minutes=5)
        claimed = SavedSearch.objects.filter(pk=saved.pk, watch_enabled=True).filter(
            Q(watch_claim_until__isnull=True) | Q(watch_claim_until__lte=now)).filter(
            Q(last_checked__isnull=True) | Q(last_checked__lte=now - delay)).update(watch_claim_until=lease, last_attempt=now)
        if not claimed:
            continue
        checked += 1
        try:
            saved.refresh_from_db()
            matches, failed = latest(saved)
            if failed:
                continue
            if saved.last_checked is None:
                # First complete check after activation/resumption: never email historical matches.
                saved.seen = sorted({a for row in matches for a in unified_search.paper_aliases(row)})
                saved.last_checked = now
                saved.save(update_fields=['seen', 'last_checked'])
                continue
            seen = set(saved.seen)
            new = [p for p in matches if not (unified_search.paper_aliases(p) & seen)][:MAX_EMAIL_ITEMS]
            if not new:
                SavedSearch.objects.filter(pk=saved.pk, watch_enabled=True).update(last_checked=now)
                continue
            if not SavedSearch.objects.filter(pk=saved.pk, watch_enabled=True).exists():
                continue
            token = signing.dumps(saved.pk, salt=UNSUBSCRIBE_SALT)
            context = {'saved': saved, 'papers': new, 'base_url': base_url,
                       'search_url': base_url.rstrip('/') + '/%s/search-results/?' % saved.lang + urlencode(saved.params, doseq=True),
                       'manage_url': base_url.rstrip('/') + '/%s/saved-searches/' % saved.lang,
                       'unsubscribe_url': base_url.rstrip('/') + '/%s/saved-searches/unsubscribe/?' % saved.lang + urlencode({'token': token})}
            with translation.override(saved.lang):
                text = render_to_string('search_watches/email.txt', context)
                html = render_to_string('search_watches/email.html', context)
                subject = _('New matches for your saved search: %(name)s') % {'name': saved.name}
            message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email],
                                            headers={'List-Unsubscribe': '<%s>' % context['unsubscribe_url']})
            message.attach_alternative(html, 'text/html')
            if message.send() != 1:
                continue
            sent += 1
            known = list(dict.fromkeys(saved.seen + [a for p in new for a in sorted(unified_search.paper_aliases(p))]))[-5000:]
            SavedSearch.objects.filter(pk=saved.pk).update(seen=known, last_checked=now, last_sent=now)
        except Exception:
            logger.exception('Saved search watch %s failed', saved.pk)
        finally:
            SavedSearch.objects.filter(pk=saved.pk, watch_claim_until=lease).update(watch_claim_until=None)
    return sent
