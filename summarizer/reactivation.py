"""One-off email to people whose account was never activated.

Activation emails were not delivered for months, so real signups never got
their link. Most inactive accounts are not real signups though: from 2024-09
to 2025-09 bots registered thousands of random usernames with other people's
addresses (subscription bombing). Those addresses never asked for anything
from us, so they are left out: an account is a candidate only when its
username does not look like such a random string.

Each address gets at most one email (ReactivationEmail), with a 30-day
activation link and a link that deletes the account.
"""
import datetime
import logging
import re

from django.conf import settings
from django.contrib.auth.models import User
from django.core import signing
from django.core.mail import EmailMultiAlternatives
from django.http import Http404
from django.shortcuts import render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import translation
from django.views.decorators.csrf import csrf_exempt

from . import accounts, mailcheck
from .models import ReactivationEmail

logger = logging.getLogger(__name__)

DELETE_SALT = 'summarizer.reactivation-delete'
# signups from this date on got a working activation email (accounts.py)
CUTOFF = datetime.datetime(2026, 10, 18, tzinfo=datetime.timezone.utc)
LANG = 'en'


def _letters(text):
    return re.sub(r'[^a-z]', '', text.lower())


def looks_random(username, email):
    """Mixed-case letters only, unrelated to the address: the bot pattern."""
    if not re.fullmatch(r'[A-Za-z]{7,20}', username):
        return False
    if not any(c.isupper() for c in username[1:]):
        return False
    name, local = _letters(username), _letters(email.split('@')[0])
    return not (name in local or (len(local) >= 4 and local in name))


def candidates():
    """Inactive, never used accounts worth one email, newest signup first, one per address."""
    taken = {e.lower() for e in User.objects.filter(is_active=True).values_list('email', flat=True)}
    already = {e.lower() for e in ReactivationEmail.objects.values_list('email', flat=True)}
    suppressed = mailcheck.suppressed()
    seen, chosen = set(), []
    for user in User.objects.filter(is_active=False, last_login__isnull=True, date_joined__lt=CUTOFF).exclude(email='').order_by('-date_joined'):
        email = user.email.strip().lower()
        if email in seen or email in taken or email in already or looks_random(user.username, user.email):
            continue
        seen.add(email)
        # bounces hurt the sender score of every email we send (CloudMailin alert, 2026-10-18)
        if mailcheck.undeliverable_reason(email, suppressed):
            continue
        chosen.append(user)
    return chosen


def delete_token(user):
    return signing.dumps(user.pk, salt=DELETE_SALT)


def send_one(user, base_url, to=None):
    activate_url = base_url + '/%s/login/link/%s/' % (LANG, accounts.make_link_token(user, long_lived=True))
    delete_url = base_url + '/%s/account/delete/%s/' % (LANG, delete_token(user))
    context = {'user': user, 'activate_url': activate_url, 'delete_url': delete_url, 'base_url': base_url}
    with translation.override(LANG):
        subject = render_to_string('reactivation/subject.txt', context).strip()
        text = render_to_string('reactivation/email.txt', context)
        html = render_to_string('reactivation/email.html', context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [to or user.email],
                                     headers={'List-Unsubscribe': '<%s>' % delete_url,
                                              'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click'})
    message.attach_alternative(html, 'text/html')
    message.send()
    if to is None:
        ReactivationEmail.objects.create(user=user, email=user.email.strip().lower())


def run(base_url, limit, dry_run=False, stdout=None):
    chosen = candidates()
    if stdout:
        stdout.write('%d accounts still to email' % len(chosen))
    sent = 0
    for user in chosen[:limit]:
        if dry_run:
            continue
        try:
            send_one(user, base_url)
            sent += 1
        except Exception:
            logger.exception('Reactivation email to user %s failed', user.pk)
    return sent


@csrf_exempt  # mail clients send the one-click List-Unsubscribe POST without a token
def delete_account(request, token):
    try:
        user = User.objects.get(pk=signing.loads(token, salt=DELETE_SALT))
    except (signing.BadSignature, User.DoesNotExist):
        raise Http404
    # only accounts nobody ever used can go this way
    if user.is_active or user.last_login is not None:
        return render(request, 'reactivation/delete_done.html', {'kept': True})
    if request.method == 'POST':
        user.delete()
        return render(request, 'reactivation/delete_done.html', {'kept': False})
    return render(request, 'reactivation/delete_confirm.html', {'account': user})
