"""Two emails, each sent once, at the moments a free account or a trial decides whether to pay.

- 'trial_result': the day after a free account ran its free try of a tool (teasers.TRIAL_TOOLS), what the result
  holds beyond the first results it saw, with a link back to it and to the Premium page.
- 'trial_ending': two days before the free days of a first subscription end (billing.TRIAL_DAYS), what the
  subscriber made during them, the date and amount of the first payment, and where to cancel.

From the hourly digest run, within its email budget. Every email has a link that stops them ('optout'); an
address that bounced or complained (mailcheck) gets none. Sent emails are kept (ConversionEmail) and counted on
the staff conversions page (conversions.py).
"""
import datetime
import logging

from django.conf import settings
from django.contrib.auth.models import User
from django.core import signing
from django.core.mail import EmailMultiAlternatives
from django.http import Http404
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import NoReverseMatch, reverse
from django.utils import formats, timezone, translation

from . import access, mailcheck, teasers
from .models import ConversionEmail, Generated, ToolUse, UserPlan

logger = logging.getLogger(__name__)

BASE_URL = 'https://www.summarizepaper.com'
OPTOUT_SALT = 'summarizer.conversion-optout'
RESULT_DELAY = datetime.timedelta(hours=20)  # the next day, about when the result was read
RESULT_TOO_OLD = datetime.timedelta(days=4)  # a try older than this is not written about any more
ENDING_AHEAD = datetime.timedelta(days=2)  # before the end of the trial: time to decide, to cancel without a charge
ENDING_TOO_LATE = datetime.timedelta(hours=12)  # closer to the end, an email would come too late
RESULT_URLS = {'teams': 'teams_result', 'missed': 'missed_result', 'refcheck': 'refcheck_result'}
UTM = 'utm_source=email&utm_medium=email&utm_campaign=%s'


def optout_token(user):
    return signing.dumps(user.pk, salt=OPTOUT_SALT)


def optout(request, token):
    """The link at the bottom of the emails: no more of them (the digests and alerts the user chose are not touched)."""
    try:
        pk = signing.loads(token, salt=OPTOUT_SALT)
    except signing.BadSignature:
        raise Http404
    user = User.objects.filter(pk=pk).first()
    if user is None:
        raise Http404
    ConversionEmail.objects.get_or_create(user=user, kind='optout', key='')
    return render(request, 'conversion/optout.html')


def _lang(user, default='en'):
    lang = user.digests.values_list('lang', flat=True).first() or default
    return lang if lang in ('en', 'fr') else 'en'


def _reachable(user, suppressed):
    return (user.is_active and user.email and user.email.strip().lower() not in suppressed
            and not ConversionEmail.objects.filter(user=user, kind='optout').exists())


def _send(user, kind, key, template, context, lang, base_url):
    context.update(base_url=base_url, lang=lang, optout_url='%s/%s/emails/stop/%s/' % (base_url, lang, optout_token(user)))
    with translation.override(lang):
        subject = render_to_string('conversion/%s_subject.txt' % template, context).strip()
        text = render_to_string('conversion/%s.txt' % template, context)
        html = render_to_string('conversion/%s.html' % template, context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email],
                                     headers={'List-Unsubscribe': '<%s>' % context['optout_url']})
    message.attach_alternative(html, 'text/html')
    try:
        message.send()
    except Exception:
        logger.exception('Conversion email %s to %s failed', kind, user.username)
        return False
    ConversionEmail.objects.get_or_create(user=user, kind=kind, key=key)
    return True


# --- the day after a free try -------------------------------------------------------------

def due_results(now):
    """The free tries finished between RESULT_TOO_OLD and RESULT_DELAY ago whose owner has not been written to."""
    rows = Generated.objects.filter(trial=True, status='done', user__isnull=False,
                                    updated__lte=now - RESULT_DELAY, updated__gte=now - RESULT_TOO_OLD).select_related('user')
    done = set(ConversionEmail.objects.filter(kind='trial_result').values_list('key', flat=True))
    return [r for r in rows.order_by('updated') if str(r.pk) not in done]


def result_context(row, base_url, lang):
    """What the email says of a free try: its teaser, its page and the Premium page (None when it has nothing to show)."""
    with translation.override(lang):
        t = teasers.teaser(row)
        if t is None:
            return None
        try:
            url = base_url + reverse(RESULT_URLS[row.kind], args=[row.key])
        except (KeyError, NoReverseMatch):
            return None
        return {'t': t, 'total': len(t['items']) + t['hidden'], 'result_url': url + '?' + UTM % 'trial_result',
                'premium_url': '%s/%s/premium/?%s' % (base_url, lang, UTM % 'trial_result')}


def send_results(now, budget, base_url=BASE_URL):
    sent = 0
    suppressed = mailcheck.suppressed()
    for row in due_results(now):
        if sent >= budget:
            break
        user = row.user
        if not _reachable(user, suppressed) or access.has_tool(user, row.kind):
            ConversionEmail.objects.get_or_create(user=user, kind='trial_result', key=str(row.pk))  # never later
            continue
        lang = row.lang if row.lang in ('en', 'fr') else _lang(user)
        context = result_context(row, base_url, lang)
        if context is None:
            ConversionEmail.objects.get_or_create(user=user, kind='trial_result', key=str(row.pk))
            continue
        from .billing import trial_days
        context['trial_days'] = trial_days(user)
        sent += _send(user, 'trial_result', str(row.pk), 'trial_result', context, lang, base_url)
    return sent


# --- two days before the end of a Stripe trial ---------------------------------------------

def due_endings(now):
    plans = (UserPlan.objects.filter(paid=True, canceling=False, trial_end__gt=now + ENDING_TOO_LATE,
                                     trial_end__lte=now + ENDING_AHEAD).exclude(stripe_subscription_id='').select_related('user'))
    done = set(ConversionEmail.objects.filter(kind='trial_ending').values_list('key', flat=True))
    return [p for p in plans if p.stripe_subscription_id not in done and p.user is not None]


def made_during(user, since, base_url, lang, limit=6):
    """What the subscriber made since the trial began: [{'tool', 'title', 'url'}], newest first."""
    from .premium_tools import GENERATED_TOOLS, TOOL_CHOICES
    names = dict(TOOL_CHOICES)
    made, seen = [], set()
    with translation.override(lang):
        for use in ToolUse.objects.filter(user=user, updated__gte=since).select_related('row').order_by('-updated')[:limit]:
            tool = GENERATED_TOOLS.get(use.row.kind)
            if tool and use.url not in seen:
                seen.add(use.url)
                made.append({'tool': str(names[tool]), 'title': use.title, 'url': base_url + use.url})
        for row in Generated.objects.filter(user=user, created__gte=since, status='done', kind__in=list(RESULT_URLS) + ['literature']).order_by('-created')[:limit]:
            data = row.data()
            name = 'literature_result' if row.kind == 'literature' else RESULT_URLS[row.kind]
            try:
                url = base_url + reverse(name, args=[row.key])
            except NoReverseMatch:
                continue
            if url not in seen:
                seen.add(url)
                made.append({'tool': str(names[row.kind]), 'url': url,
                             'title': str(data.get('title') or data.get('topic') or data.get('question') or '')[:120]})
    return made[:limit]


def _first_payment(subscription):
    """('9.00 EUR', 'month') of the subscription's price, from Stripe; None when unknown."""
    items = (subscription.get('items') or {}).get('data') or []
    price = items[0].get('price') if items else None
    if not price or price.get('unit_amount') is None:
        return None
    return '%.2f %s' % (price['unit_amount'] / 100, price['currency'].upper()), (price.get('recurring') or {}).get('interval', '')


def send_endings(now, budget, base_url=BASE_URL):
    from . import billing
    sent = 0
    if not billing.billing_enabled():
        return 0
    suppressed = mailcheck.suppressed()
    for plan in due_endings(now):
        if sent >= budget:
            break
        user = plan.user
        key = plan.stripe_subscription_id
        if not _reachable(user, suppressed):
            ConversionEmail.objects.get_or_create(user=user, kind='trial_ending', key=key)
            continue
        try:
            subscription = billing._stripe().Subscription.retrieve(key)
        except Exception:
            logger.exception('Stripe subscription %s for the trial email', key)
            continue  # next hour
        if subscription.get('status') != 'trialing' or subscription.get('cancel_at_period_end') or subscription.get('cancel_at'):
            ConversionEmail.objects.get_or_create(user=user, kind='trial_ending', key=key)
            continue
        lang = _lang(user, Generated.objects.filter(user=user).values_list('lang', flat=True).last() or 'en')
        payment = _first_payment(subscription)
        with translation.override(lang):
            end = formats.date_format(timezone.localtime(plan.trial_end), 'DATE_FORMAT')
        context = {
            'plan': 'Premium+' if plan.tier == 'plus' else 'Premium', 'end': end,
            'amount': payment[0] if payment else '', 'interval': payment[1] if payment else '',
            'made': made_during(user, plan.trial_end - datetime.timedelta(days=billing.TRIAL_DAYS + 1), base_url, lang),
            'premium_url': '%s/%s/premium/?%s' % (base_url, lang, UTM % 'trial_ending'),
            'tools_url': '%s/%s/tools/?%s' % (base_url, lang, UTM % 'trial_ending'),
        }
        sent += _send(user, 'trial_ending', key, 'trial_ending', context, lang, base_url)
    return sent


def run(now, budget, base_url=BASE_URL):
    """From the hourly digest run: the emails that are due, within the budget. Returns the emails sent."""
    sent = send_results(now, budget, base_url)
    return sent + send_endings(now, budget - sent, base_url)
