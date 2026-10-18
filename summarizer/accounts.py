"""Email login links, which also activate new accounts, Google sign-in, and signup bot checks.

A login link carries a signed user id and the user's last_login, so it stops
working as soon as it has been used (logging in updates last_login) and after
LINK_MAX_AGE. Opening the link shows a button and only the POST logs in, so
mail scanners that prefetch links cannot spend it.
"""
import json
import logging
import re
import secrets
import time
from urllib.parse import urlencode

import requests

from django import forms
from django.conf import settings
from django.contrib.auth import login
from django.contrib.auth.models import User
from django.core import signing
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils.http import url_has_allowed_host_and_scheme
from django.utils import translation
from django.utils.translation import get_language

from . import access

logger = logging.getLogger(__name__)

LINK_MAX_AGE = 3 * 24 * 3600
LINK_SALT = 'summarizer.login-link'
STAMP_SALT = 'summarizer.form-stamp'
HONEYPOT_FIELD = 'website'
STAMP_FIELD = 'form_stamp'
MIN_FORM_SECONDS = 3
LINKS_PER_EMAIL_PER_HOUR = 3
LINKS_PER_IP_PER_HOUR = 10
SIGNUPS_PER_IP_PER_DAY = 10


# --- bot checks -------------------------------------------------------------

def form_stamp():
    return signing.dumps(time.time(), salt=STAMP_SALT)


def auth_context():
    """Context of the login and signup pages: bot check fields and the Google button."""
    return {'antibot_stamp': form_stamp(), 'antibot_field': HONEYPOT_FIELD,
            'google_login': bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)}


def looks_like_bot(data):
    """A filled honeypot, or a form sent back faster than a person can type, or without our stamp."""
    if data.get(HONEYPOT_FIELD):
        return True
    try:
        started = signing.loads(data.get(STAMP_FIELD, ''), salt=STAMP_SALT, max_age=24 * 3600)
    except signing.BadSignature:
        return True
    return time.time() - float(started) < MIN_FORM_SECONDS


def over_limit(key, limit, seconds):
    """Count one more hit on key and say whether it went over limit in the window."""
    cache.add(key, 0, seconds)
    try:
        count = cache.incr(key)
    except ValueError:
        cache.set(key, 1, seconds)
        count = 1
    return count > limit


def signups_exhausted(hashed_ip):
    """Only created accounts count, so retrying after a form error never locks a person out."""
    return (cache.get('signup:ip:' + hashed_ip) or 0) >= SIGNUPS_PER_IP_PER_DAY


def count_signup(hashed_ip):
    over_limit('signup:ip:' + hashed_ip, SIGNUPS_PER_IP_PER_DAY, 24 * 3600)


# --- login links --------------------------------------------------------------

def _last_login_stamp(user):
    return user.last_login.isoformat() if user.last_login else ''


def make_link_token(user):
    return signing.dumps({'u': user.pk, 'l': _last_login_stamp(user)}, salt=LINK_SALT)


def user_from_link_token(token):
    try:
        data = signing.loads(token, salt=LINK_SALT, max_age=LINK_MAX_AGE)
        user = User.objects.get(pk=data['u'])
    except (signing.BadSignature, User.DoesNotExist, KeyError, TypeError):
        return None
    if data['l'] != _last_login_stamp(user):
        return None
    return user


def _base_url(request):
    return settings.SITE_URL or '%s://%s' % (request.scheme, request.get_host())


def _safe_next(request, next_url):
    if next_url and url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()},
                                                    require_https=request.is_secure()):
        return next_url
    return ''


def send_login_link(request, user, next_url=''):
    lang = get_language()
    link = '%s%s' % (_base_url(request), reverse('login_link', args=[make_link_token(user)]))
    if next_url:
        link += '?next=' + next_url
    context = {'user': user, 'link': link, 'activation': not user.is_active, 'hours': LINK_MAX_AGE // 3600}
    subject = render_to_string('accounts/login_link_subject.txt', context).strip()
    text = render_to_string('accounts/login_link_email.txt', context)
    html = render_to_string('accounts/login_link_email.html', context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
    message.attach_alternative(html, 'text/html')
    try:
        message.send()
    except Exception:
        logger.exception('Could not send the login link to user %s (%s)', user.pk, lang)
        return False
    return True


def account_for_email(email):
    """The account a login link for this address should open: an active one if any, else the newest."""
    return (User.objects.filter(email__iexact=email.strip())
            .order_by('-is_active', '-last_login', '-date_joined').first())


class EmailLinkForm(forms.Form):
    email = forms.EmailField(max_length=254)


def email_login(request):
    next_url = _safe_next(request, request.POST.get('next') or request.GET.get('next'))
    if request.method == 'POST':
        form = EmailLinkForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email'].lower()
            ip = access.hash_ip(access.client_ip(request))
            blocked = (looks_like_bot(request.POST)
                       or over_limit('loginlink:ip:' + ip, LINKS_PER_IP_PER_HOUR, 3600)
                       or over_limit('loginlink:email:' + access.hash_ip(email), LINKS_PER_EMAIL_PER_HOUR, 3600))
            user = None if blocked else account_for_email(email)
            if user is not None:
                send_login_link(request, user, next_url)
            # same answer whether or not the address has an account
            return render(request, 'accounts/email_login_sent.html', {'email': email})
    else:
        form = EmailLinkForm(initial={'email': request.GET.get('email', '')})
    return render(request, 'accounts/email_login.html', {'form': form, 'next': next_url, **auth_context()})


def login_link(request, token):
    user = user_from_link_token(token)
    if user is None:
        return render(request, 'accounts/login_link_invalid.html', status=400)
    next_url = _safe_next(request, request.GET.get('next'))
    if request.method != 'POST':
        return render(request, 'accounts/login_link_confirm.html', {'link_user': user, 'next': next_url})
    activated = not user.is_active
    if activated:
        user.is_active = True
        user.save(update_fields=['is_active'])
    login(request, user, backend=settings.AUTHENTICATION_BACKENDS[0])
    if next_url:
        return redirect(next_url)
    response = redirect('summarize')
    if activated:
        response['Location'] += '?activated=True'
    return response


# --- Sign in with Google ------------------------------------------------------

GOOGLE_AUTH = 'https://accounts.google.com/o/oauth2/v2/auth'
GOOGLE_TOKEN = 'https://oauth2.googleapis.com/token'
GOOGLE_USERINFO = 'https://openidconnect.googleapis.com/v1/userinfo'
GOOGLE_SESSION_KEY = 'google_oauth'


def _google_redirect_uri(request):
    # one URL outside the language prefix, as registered in Google Cloud Console
    return _base_url(request) + '/google/callback/'


def google_start(request):
    if not auth_context()['google_login']:
        return redirect('login')
    state = secrets.token_urlsafe(24)
    request.session[GOOGLE_SESSION_KEY] = {
        'state': state, 'lang': get_language(),
        'next': _safe_next(request, request.GET.get('next')),
    }
    params = {
        'client_id': settings.GOOGLE_CLIENT_ID, 'redirect_uri': _google_redirect_uri(request),
        'response_type': 'code', 'scope': 'openid email profile', 'state': state,
        'prompt': 'select_account',
    }
    return redirect('%s?%s' % (GOOGLE_AUTH, urlencode(params)))


def _google_profile(request, code):
    token = requests.post(GOOGLE_TOKEN, timeout=20, data={
        'code': code, 'client_id': settings.GOOGLE_CLIENT_ID, 'client_secret': settings.GOOGLE_CLIENT_SECRET,
        'redirect_uri': _google_redirect_uri(request), 'grant_type': 'authorization_code',
    })
    token.raise_for_status()
    profile = requests.get(GOOGLE_USERINFO, timeout=20,
                           headers={'Authorization': 'Bearer %s' % token.json()['access_token']})
    profile.raise_for_status()
    return profile.json()


def _new_username(email):
    base = re.sub(r'[^\w.@+-]', '', email.split('@')[0])[:24] or 'user'
    username, n = base, 1
    while User.objects.filter(username__iexact=username).exists():
        n += 1
        username = '%s%d' % (base, n)
    return username


def google_user(email):
    """(user, created_or_activated) for a Google-verified address."""
    user = account_for_email(email)
    if user is None:
        user = User(username=_new_username(email), email=email, is_active=True)
        user.set_unusable_password()
        user.save()
        return user, True
    if not user.is_active:
        user.is_active = True
        user.save(update_fields=['is_active'])
        return user, True
    return user, False


def google_callback(request):
    pending = request.session.pop(GOOGLE_SESSION_KEY, None) or {}
    lang = pending.get('lang') or settings.LANGUAGE_CODE
    with translation.override(lang):
        if (not pending.get('state') or request.GET.get('state') != pending['state']
                or 'code' not in request.GET):
            return render(request, 'accounts/google_error.html', status=400)
        try:
            profile = _google_profile(request, request.GET['code'])
        except (requests.RequestException, KeyError, ValueError):
            logger.exception('Google sign-in failed')
            return render(request, 'accounts/google_error.html', status=502)
        email = (profile.get('email') or '').strip().lower()
        if not email or not profile.get('email_verified'):
            return render(request, 'accounts/google_error.html', status=400)
        user, fresh = google_user(email)
        login(request, user, backend=settings.AUTHENTICATION_BACKENDS[0])
        target = pending.get('next') or '/%s/%s' % (lang, '?activated=True' if fresh else '')
        # Not a redirect: browsers may leave the new session cookie off the next hop of a
        # redirect chain that started on accounts.google.com. A page that moves on by itself
        # starts a same-site navigation, which carries the cookie.
        return render(request, 'accounts/google_done.html',
                      {'target': target, 'target_json': json.dumps(target).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026'), 'lang': lang})
