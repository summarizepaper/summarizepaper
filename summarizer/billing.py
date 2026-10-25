"""Premium subscriptions with Stripe Checkout.

Stripe owns the payment pages, the invoices and the cancellation flow
(Customer Portal). We only keep, per user, a UserPlan row that says whether
the subscription is active and until when; access.is_paid reads it.

Configuration (Heroku config vars):
  STRIPE_SECRET_KEY        sk_live_... / sk_test_...
  STRIPE_WEBHOOK_SECRET    whsec_... of the endpoint /stripe/webhook/
  STRIPE_PRICE_MONTHLY     price_... of the monthly price
  STRIPE_PRICE_YEARLY      price_... of the yearly price (optional)
  STRIPE_PRICE_MONTHLY_USD price_... of the fixed USD monthly price (optional)
  STRIPE_PRICE_YEARLY_USD  price_... of the fixed USD yearly price (optional)
  STRIPE_PRICE_PLUS_...    the same four for Premium+ (premium_tools.PLUS_TOOLS), twice the price;
                           without STRIPE_PRICE_PLUS_MONTHLY only Premium is sold

A subscription's tier (UserPlan.tier) is read from its price at each webhook. A
subscriber switches between Premium and Premium+ on the premium page (change_plan):
an upgrade is invoiced at once for the rest of the period, a downgrade is credited.
"""
import datetime
import logging
from urllib.parse import urlencode

import stripe
from django.conf import settings
from django.core.cache import cache
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import get_language, gettext as _
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.views.decorators.cache import never_cache

from . import access
from .models import SiteSettings, UserPlan

logger = logging.getLogger(__name__)


def billing_enabled():
    return bool(settings.STRIPE_SECRET_KEY and settings.STRIPE_PRICE_MONTHLY)


TIERS = ('premium', 'plus')


def price_id(currency, period='monthly', tier='premium'):
    name = 'STRIPE_PRICE_' + ('PLUS_' if tier == 'plus' else '') + period.upper() + ('_USD' if currency == 'usd' else '')
    return getattr(settings, name, '')


def plus_enabled():
    return billing_enabled() and bool(price_id('eur', tier='plus'))


def tier_of_price(price):
    """'plus' for a Premium+ price id, else 'premium'."""
    plus = {price_id(c, p, 'plus') for c in ('eur', 'usd') for p in ('monthly', 'yearly')} - {''}
    return 'plus' if price in plus else 'premium'


def currencies():
    return ['eur', 'usd'] if price_id('usd') else ['eur']


def selected_currency(request):
    """An explicit choice wins over geography; language never determines currency."""
    available = currencies()
    explicit = request.GET.get('currency', '').lower()
    if explicit in available:
        request.session['billing_currency'] = explicit
        return explicit
    saved = request.session.get('billing_currency')
    if saved in available:
        return saved
    # Heroku appends the connecting proxy as the final forwarded address.
    hops = request.META.get('HTTP_X_FORWARDED_FOR', '').split(',')
    edge = hops[-1].strip() if hops[-1].strip() else request.META.get('REMOTE_ADDR', '')
    if (access._is_cloudflare(edge) and request.META.get('HTTP_CF_IPCOUNTRY', '').upper() == 'US'
            and 'usd' in available):
        return 'usd'
    return 'eur'


def _stripe():
    stripe.api_key = settings.STRIPE_SECRET_KEY
    return stripe


def _absolute(request, path):
    """Absolute URL of a page in the current language, e.g. 'premium/'."""
    #behind Cloudflare the request can look like plain http, so don't derive the scheme from it
    base = settings.SITE_URL or request.build_absolute_uri('/').rstrip('/')
    return '%s/%s/%s' % (base, get_language(), path)


def _latest_plan(user):
    return UserPlan.objects.filter(user=user).order_by('created').last()


def price_label(price_id):
    """'9.00 EUR / month' for a Stripe price, cached for an hour; None if unavailable."""
    if not price_id or not settings.STRIPE_SECRET_KEY:
        return None
    key = 'stripe_price_label_' + price_id
    label = cache.get(key)
    if label is None:
        try:
            price = _stripe().Price.retrieve(price_id)
        except stripe.error.StripeError:
            logger.exception('Could not load Stripe price %s', price_id)
            # Avoid retrying Stripe on every homepage visit during an outage.
            cache.set(key, '', 60)
            return None
        interval = (price.get('recurring') or {}).get('interval', '')
        label = '%.2f %s / %s' % (price['unit_amount'] / 100, price['currency'].upper(), interval)
        cache.set(key, label, 3600)
    return label or None


def features():
    """Every tool and what each offer gives of it, by step of research: [(group, [(name, path, free, premium,
    plus)])], a value being True, False or a short text. Paths are relative to /<lang>/."""
    readings = SiteSettings.current().free_readings_per_day
    chat = access.chat_messages_per_day()
    runs = access.runs_per_day()
    unlimited = _('Unlimited')
    return [
        (_('Read papers faster'), [
            (_('Summaries, key points, lay summary and blog article'), '', _('%d new per day') % runs, unlimited, unlimited),
            (_('AI model'), '', _('Standard'), _('Most capable'), _('Most capable')),
            (_('Chat with papers'), 'chat', _('%d questions per day') % chat['user'], _('%d questions per day') % chat['paid'],
             _('%d questions per day') % chat['paid']),
            (_('Rewrites for another reader or language'), '', _('%d per day') % readings, unlimited, unlimited),
            (_('Collections'), 'collections/', _('3 of 20 papers'), _('100 of 200 papers'), _('100 of 200 papers')),
            (_('Comparison tables and open questions of a collection'), 'collections/', False, True, True),
            (_('BibTeX, RIS and Markdown exports'), 'integrations/', True, True, True),
            (_('Papers sent to Zotero with their summary'), 'integrations/', False, True, True),
        ]),
        (_('Follow your field'), [
            (_('Daily digest'), 'digest/', _('One digest'), _('Up to 5, ranked for you, read aloud'),
             _('Up to 5, ranked for you, read aloud')),
            (_('Weekly letters'), 'letters/', _('On the web'), _('On the web and by email'), _('On the web and by email')),
            (_('Rising topics'), 'trends/', _('The top ones'), _('All, field by field'), _('All, field by field')),
            (_('Saved searches with email alerts'), 'saved-searches/', '1', '5', '5'),
            (_('Citation and version alerts'), 'citations/', False, True, True),
            (_('Scoop watch'), 'scoops/', False, True, True),
            (_('Who works on it?'), 'teams/', False, True, True),
        ]),
        (_('Write and submit'), [
            (_('What did I miss?'), 'missed/', False, True, True),
            (_('Bibliography check'), 'refcheck/', False, True, True),
            (_('Literature review from a research question'), 'literature-review/', False, False, True),
        ]),
        (_('Present, meet and apply'), [
            (_('Journal club slides'), 'journal-club/', False, False, True),
            (_('Press kit and hype check'), 'press-kit/', False, False, True),
            (_('Conference companion'), 'conference/', False, False, True),
            (_('Find a research placement'), 'students/', False, False, True),
        ]),
    ]


@never_cache
def premium(request):
    currency = selected_currency(request)
    plan = _latest_plan(request.user) if request.user.is_authenticated else None
    paid = access.is_paid(request.user)
    plus = access.is_plus(request.user)
    # This marker requires server-side Premium access, never just ?success=1.
    activated = bool(paid and request.session.pop('premium_checkout_pending', False))
    return render(request, 'summarizer/premium.html', {
        'billing_enabled': billing_enabled(),
        'monthly_price': price_label(price_id(currency)),
        'yearly_price': price_label(price_id(currency, 'yearly')),
        'plus_enabled': plus_enabled(),
        'plus_monthly_price': price_label(price_id(currency, tier='plus')),
        'plus_yearly_price': price_label(price_id(currency, 'yearly', 'plus')),
        'billing_currency': currency, 'billing_currencies': currencies(),
        'is_paid': paid, 'is_plus': plus,
        'features': features(),
        'premium_activated': activated,
        'plan': plan,
        'tool_grants': list(access.active_grants(request.user)),
        'subscription_active': bool(plan and plan.paid and (plan.expires is None or plan.expires >= timezone.now())),
        # a running Stripe subscription can move between Premium and Premium+ (change_plan)
        'can_switch': bool(plus_enabled() and plan and plan.stripe_subscription_id and access.current_plan(request.user) == plan),
        'can_manage': bool(plan and plan.stripe_customer_id),
        'runs_per_day': access.runs_per_day(),
        'chat_free': access.chat_messages_per_day()['user'],
        'chat_paid': access.chat_messages_per_day()['paid'],
        'readings_free': SiteSettings.current().free_readings_per_day,
        'success': request.GET.get('success') == '1',
        'changed': request.GET.get('changed') if request.GET.get('changed') in TIERS else None,
        'checkout_cancelled': request.GET.get('cancelled') == '1',
    })


@require_POST
def checkout(request):
    if not request.user.is_authenticated:
        return redirect(reverse("login") + "?" + urlencode({"next": reverse("premium")}))
    if not billing_enabled():
        return redirect('premium')
    if access.is_paid(request.user):
        return redirect('premium')
    period = request.POST.get('period', 'monthly')
    tier = request.POST.get('tier', 'premium')
    currency = request.POST.get('currency', selected_currency(request)).lower()
    if currency not in currencies():
        return HttpResponseBadRequest('invalid billing currency')
    if tier not in TIERS or period not in ('monthly', 'yearly') or not price_id(currency, period, tier):
        return HttpResponseBadRequest('invalid billing period')
    price = price_id(currency, period, tier)
    request.session['billing_currency'] = currency
    plan = _latest_plan(request.user)
    params = {
        'mode': 'subscription',
        'currency': currency,
        'adaptive_pricing': {'enabled': False},
        'line_items': [{'price': price, 'quantity': 1}],
        'client_reference_id': str(request.user.id),
        'success_url': _absolute(request, 'premium/') + '?success=1',
        'cancel_url': _absolute(request, 'premium/') + '?cancelled=1',
        'allow_promotion_codes': True,
        'subscription_data': {'metadata': {'user_id': str(request.user.id)}},
    }
    if plan and plan.stripe_customer_id:
        params['customer'] = plan.stripe_customer_id
    elif request.user.email:
        params['customer_email'] = request.user.email
    try:
        session = _stripe().checkout.Session.create(**params)
    except stripe.error.StripeError:
        logger.exception('Could not create Stripe Checkout session')
        messages.error(request, _('Payment is temporarily unavailable. Please try again.'))
        return redirect('premium')
    request.session['premium_checkout_pending'] = True
    return redirect(session.url, permanent=False)


@login_required
@require_POST
def portal(request):
    plan = _latest_plan(request.user)
    if not billing_enabled() or not plan or not plan.stripe_customer_id:
        return redirect('premium')
    session = _stripe().billing_portal.Session.create(
        customer=plan.stripe_customer_id,
        return_url=_absolute(request, 'premium/'),
    )
    return redirect(session.url, permanent=False)


@login_required
@require_POST
def change_plan(request):
    """Move a running subscription to Premium+ or back to Premium, same currency and period."""
    tier = request.POST.get('tier')
    plan = access.current_plan(request.user)
    if tier not in TIERS or not plus_enabled() or plan is None or not plan.stripe_subscription_id or plan.tier == tier:
        return redirect('premium')
    try:
        subscription = _stripe().Subscription.retrieve(plan.stripe_subscription_id)
        item = subscription['items']['data'][0]
        period = 'yearly' if item['price']['recurring']['interval'] == 'year' else 'monthly'
        new = price_id(item['price']['currency'], period, tier)
        if not new:
            return redirect('premium')
        updated = _stripe().Subscription.modify(
            subscription['id'], items=[{'id': item['id'], 'price': new}],
            # an upgrade is paid now for the rest of the period; a downgrade is credited on the next invoice
            proration_behavior='always_invoice' if tier == 'plus' else 'create_prorations',
            payment_behavior='error_if_incomplete')
    except stripe.error.StripeError:
        logger.exception('Could not change the plan of %s', plan.stripe_subscription_id)
        messages.error(request, _('Your plan could not be changed. Your card was not charged; please try again.'))
        return redirect('premium')
    sync_subscription(updated)
    return redirect(reverse('premium') + '?changed=' + tier)


def _period_end(subscription):
    # newer Stripe API versions moved the period from the subscription to its items
    end = subscription.get('current_period_end')
    if end is None:
        items = (subscription.get('items') or {}).get('data') or []
        end = items[0].get('current_period_end') if items else None
    if end is None:
        return None
    return datetime.datetime.fromtimestamp(end, tz=datetime.timezone.utc)


def sync_subscription(subscription, user=None):
    """Mirror a Stripe subscription object onto the user's UserPlan."""
    plan = UserPlan.objects.filter(stripe_subscription_id=subscription['id']).order_by('created').last()
    if plan is None:
        if user is None:
            user_id = (subscription.get('metadata') or {}).get('user_id')
            user = User.objects.filter(pk=user_id).first() if user_id else None
        if user is None:
            logger.warning('Stripe subscription %s has no matching user', subscription['id'])
            return None
        plan = UserPlan(user=user)
    plan.stripe_subscription_id = subscription['id']
    plan.stripe_customer_id = subscription['customer']
    # past_due keeps access while Stripe retries the card; expires still bounds it
    plan.paid = subscription['status'] in ('active', 'trialing', 'past_due')
    items = (subscription.get('items') or {}).get('data') or []
    plan.tier = tier_of_price((items[0].get('price') or {}).get('id')) if items else 'premium'
    plan.expires = _period_end(subscription)
    plan.canceling = bool(subscription.get('cancel_at_period_end') or subscription.get('cancel_at'))
    plan.save()
    return plan


@csrf_exempt
@require_POST
def webhook(request):
    try:
        event = stripe.Webhook.construct_event(
            request.body, request.headers.get('Stripe-Signature', ''), settings.STRIPE_WEBHOOK_SECRET)
    except (ValueError, stripe.error.SignatureVerificationError):
        return HttpResponseBadRequest('invalid signature')

    obj = event['data']['object']
    if event['type'] == 'checkout.session.completed' and obj.get('mode') == 'subscription':
        user = User.objects.filter(pk=obj.get('client_reference_id')).first()
        subscription = _stripe().Subscription.retrieve(obj['subscription'])
        sync_subscription(subscription, user=user)
    elif event['type'] in ('customer.subscription.created', 'customer.subscription.updated',
                           'customer.subscription.deleted'):
        sync_subscription(obj)
    return HttpResponse('ok')
