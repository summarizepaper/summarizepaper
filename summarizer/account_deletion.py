"""A signed-in user deletes their own account and everything attached to it.

Rows linked to the user go with it (CASCADE); the few kept by email address
or with a nulled user are removed here, except the suppression list, which
keeps us from writing again to an address that bounced or complained. A
running Stripe subscription is cancelled first, so nobody is charged for an
account that no longer exists; Stripe keeps the invoices, which the law
requires us to keep.
"""
import logging

from django.contrib.auth import logout
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import render

from . import billing
from .models import EmailEvent, OutreachContact, ReactivationEmail, UserPlan

logger = logging.getLogger(__name__)


def _running_subscriptions(user):
    return [plan.stripe_subscription_id for plan in
            UserPlan.objects.filter(user=user, paid=True).exclude(stripe_subscription_id='')]


def _cancel_subscriptions(ids):
    """Whether every subscription is cancelled (or already gone) at Stripe."""
    for subscription_id in ids:
        try:
            billing._stripe().Subscription.cancel(subscription_id)
        except billing.stripe.error.InvalidRequestError as error:
            # already cancelled or deleted at Stripe: nothing left to charge
            if getattr(error, 'code', '') != 'resource_missing' and 'canceled' not in str(error):
                logger.exception('Could not cancel subscription %s', subscription_id)
                return False
        except Exception:
            logger.exception('Could not cancel subscription %s', subscription_id)
            return False
    return True


def erase(user):
    email = (user.email or '').strip()
    with transaction.atomic():
        if email:
            OutreachContact.objects.filter(email__iexact=email).delete()
            ReactivationEmail.objects.filter(email__iexact=email).delete()
            EmailEvent.objects.filter(recipient__iexact=email).delete()
        user.delete()


@login_required
def page(request):
    user = request.user
    subscriptions = _running_subscriptions(user)
    context = {'subscribed': bool(subscriptions), 'staff': user.is_staff or user.is_superuser}
    if request.method != 'POST' or context['staff']:
        return render(request, 'summarizer/account_delete.html', context)
    if request.POST.get('confirm') != 'yes':
        context['error'] = 'confirm'
        return render(request, 'summarizer/account_delete.html', context, status=400)
    if not _cancel_subscriptions(subscriptions):
        context['error'] = 'billing'
        return render(request, 'summarizer/account_delete.html', context, status=503)
    logout(request)
    erase(user)
    return render(request, 'summarizer/account_delete.html', {'deleted': True})
