"""Premium subscriptions with Stripe Checkout.

Stripe owns the payment pages, the invoices and the cancellation flow
(Customer Portal). We only keep, per user, a UserPlan row that says whether
the subscription is active and until when; access.is_paid reads it.

Configuration (Heroku config vars):
  STRIPE_SECRET_KEY        sk_live_... / sk_test_...
  STRIPE_WEBHOOK_SECRET    whsec_... of the endpoint /stripe/webhook/
  STRIPE_PRICE_MONTHLY     price_... of the monthly price
  STRIPE_PRICE_YEARLY      price_... of the yearly price (optional)
"""
import datetime
import logging

import stripe
from django.conf import settings
from django.core.cache import cache
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.http import HttpResponse, HttpResponseBadRequest
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import get_language
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import access
from .models import UserPlan

logger = logging.getLogger(__name__)


def billing_enabled():
    return bool(settings.STRIPE_SECRET_KEY and settings.STRIPE_PRICE_MONTHLY)


def _stripe():
    stripe.api_key = settings.STRIPE_SECRET_KEY
    return stripe


def _absolute(request, name):
    return request.build_absolute_uri('/%s%s' % (get_language(), reverse(name)))


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
            return None
        interval = (price.get('recurring') or {}).get('interval', '')
        label = '%.2f %s / %s' % (price['unit_amount'] / 100, price['currency'].upper(), interval)
        cache.set(key, label, 3600)
    return label


def premium(request):
    plan = _latest_plan(request.user) if request.user.is_authenticated else None
    return render(request, 'summarizer/premium.html', {
        'billing_enabled': billing_enabled(),
        'monthly_price': price_label(settings.STRIPE_PRICE_MONTHLY),
        'yearly_price': price_label(settings.STRIPE_PRICE_YEARLY),
        'is_paid': access.is_paid(request.user),
        'can_manage': bool(plan and plan.stripe_customer_id),
        'runs_per_day': access.RUNS_PER_DAY,
        'chat_free': access.CHAT_MESSAGES_PER_DAY['user'],
        'chat_paid': access.CHAT_MESSAGES_PER_DAY['paid'],
        'success': request.GET.get('success') == '1',
    })


@login_required
@require_POST
def checkout(request):
    if not billing_enabled():
        return redirect('premium')
    price = settings.STRIPE_PRICE_YEARLY if request.POST.get('period') == 'yearly' else settings.STRIPE_PRICE_MONTHLY
    plan = _latest_plan(request.user)
    params = {
        'mode': 'subscription',
        'line_items': [{'price': price, 'quantity': 1}],
        'client_reference_id': str(request.user.id),
        'success_url': _absolute(request, 'premium') + '?success=1',
        'cancel_url': _absolute(request, 'premium'),
        'allow_promotion_codes': True,
        'subscription_data': {'metadata': {'user_id': str(request.user.id)}},
    }
    if plan and plan.stripe_customer_id:
        params['customer'] = plan.stripe_customer_id
    elif request.user.email:
        params['customer_email'] = request.user.email
    session = _stripe().checkout.Session.create(**params)
    return redirect(session.url, permanent=False)


@login_required
@require_POST
def portal(request):
    plan = _latest_plan(request.user)
    if not billing_enabled() or not plan or not plan.stripe_customer_id:
        return redirect('premium')
    session = _stripe().billing_portal.Session.create(
        customer=plan.stripe_customer_id,
        return_url=_absolute(request, 'premium'),
    )
    return redirect(session.url, permanent=False)


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
    plan.expires = _period_end(subscription)
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
