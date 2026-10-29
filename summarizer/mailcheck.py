"""Checks before emailing an address, to keep bounces (and our sender score) low.

`undeliverable_reason(email)` returns why an address should not be written to,
or '' when it looks fine: suppressed after a bounce, invalid syntax, throwaway
domain, Gmail dot-padding (a bot signature), or a domain without mail servers.
"""
import functools
import re

import dns.exception
import dns.resolver
from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_email

from .models import SuppressedEmail

# Throwaway inbox providers seen in our signups, plus the most common ones.
DISPOSABLE_DOMAINS = {
    'ixunbo.com', 'typingsquirrel.com', 'mailinator.com', 'guerrillamail.com', 'guerrillamail.info',
    'sharklasers.com', 'grr.la', '10minutemail.com', 'tempmail.com', 'temp-mail.org', 'yopmail.com',
    'yopmail.fr', 'trashmail.com', 'getnada.com', 'dispostable.com', 'maildrop.cc', 'throwawaymail.com',
    'fakeinbox.com', 'mohmal.com', 'emailondeck.com', 'mintemail.com', 'spamgourmet.com', 'tempail.com',
    'tmpmail.org', 'tmpmail.net', 'minuteinbox.com', 'mailnesia.com', 'mail.tm', 'emailfake.com',
}
GMAIL_DOMAINS = {'gmail.com', 'googlemail.com'}
MAX_GMAIL_DOTS = 2  # 'o.k.or.ol.a.d.ed.ep21@gmail.com': bots pad one inbox with dots


def suppressed():
    return {e.lower() for e in SuppressedEmail.objects.values_list('email', flat=True)}


@functools.lru_cache(maxsize=4096)
def domain_accepts_mail(domain):
    """True when the domain has MX records (or, per RFC 5321, an A record). DNS trouble counts as yes."""
    resolver = dns.resolver.Resolver()
    resolver.lifetime = 5
    for rtype in ('MX', 'A'):
        try:
            answers = resolver.resolve(domain, rtype)
        except (dns.resolver.NXDOMAIN, dns.resolver.NoNameservers):
            return False
        except dns.resolver.NoAnswer:
            continue
        except dns.exception.DNSException:
            return True
        if rtype == 'MX' and all(str(r.exchange) == '.' for r in answers):
            return False  # null MX: the domain explicitly takes no mail
        return True
    return False


def undeliverable_reason(email, suppressed_set=None, *, check_dns=None):
    email = (email or '').strip().lower()
    if suppressed_set is None:
        suppressed_set = suppressed()
    if email in suppressed_set:
        return 'suppressed'
    try:
        validate_email(email)
    except ValidationError:
        return 'invalid'
    local, domain = email.rsplit('@', 1)
    if domain in DISPOSABLE_DOMAINS:
        return 'disposable'
    if domain in GMAIL_DOMAINS and local.split('+')[0].count('.') > MAX_GMAIL_DOTS:
        return 'gmail dots'
    if domain in GMAIL_DOMAINS and not re.fullmatch(r'[a-z0-9.+]+', local):
        return 'invalid'
    if (settings.MAILCHECK_DNS if check_dns is None else check_dns) and not domain_accepts_mail(domain):
        return 'no mail server'
    return ''


def suppress(email, reason):
    obj, created = SuppressedEmail.objects.get_or_create(email=email.strip().lower(), defaults={'reason': reason[:200]})
    return created
