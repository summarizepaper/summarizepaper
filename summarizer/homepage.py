"""Homepage presentation data; never generate summaries or send email on a visit."""
import re
from decimal import Decimal

from django.contrib.auth.models import User
from django.core.cache import cache
from django.db.models import Q
from django.urls import reverse
from django.utils.translation import get_language, gettext_lazy as _

from . import billing
from .models import SummaryPaper


# Free tools shown first on the homepage: (url name or anchor, icon, name, pitch, needs an account, call to action).
FREE_TOOLS = [
    ('#arxiv-id', 'file-alt', _('Summaries'), _('Key ideas, results and limits of any paper, in 16 languages.'), False, _('Summarize a paper')),
    ('chat', 'comments', _('Chat with papers'), _('Ask a question to one paper or several, answered from the text.'), False, _('Ask a question')),
    ('digest', 'envelope', _('Daily digest'), _('New papers in your fields every morning, with short summaries.'), True, _('Set up my free digest')),
    ('letters', 'feather-alt', _('Weekly letters'), _('Each week, the papers that matter in your field, put into perspective.'), False, _('Read this week’s letters')),
    ('trends', 'chart-line', _('Rising topics'), _('The topics rising on arXiv, bioRxiv and medRxiv, field by field.'), False, _('See rising topics')),
    ('saved_searches', 'bookmark', _('Saved searches'), _('Save a search and get its new papers by email or RSS.'), True, _('Save a search')),
]


def free_tools():
    return [{'url': url if url.startswith('#') else reverse(url), 'slug': url.lstrip('#'), 'icon': icon, 'name': name,
             'pitch': pitch, 'account': account, 'cta': cta} for url, icon, name, pitch, account, cta in FREE_TOOLS]


READERS_SHOWN_FROM = 100  # a smaller count would not reassure anyone


def readers():
    """Researchers receiving a digest or a weekly letter, rounded down for display; None while too few."""
    count = cache.get('home-readers')
    if count is None:
        count = (User.objects.filter(Q(digests__active=True) | Q(letter_subscriptions__active=True), is_active=True)
                 .distinct().count())
        cache.set('home-readers', count, 6 * 3600)
    if count < READERS_SHOWN_FROM:
        return None
    step = 100 if count >= 1000 else 10 if count >= 100 else 1
    shown = '{:,}'.format(count // step * step)
    return shown.replace(',', '\u202f') if get_language() == 'fr' else shown


def monthly_price(currency='eur'):
    """Use the same cached Stripe quote as Premium, with a compact local label."""
    if not billing.billing_enabled():
        return None
    label = billing.price_label(billing.price_id(currency))
    if not label:
        return None
    match = re.fullmatch(r'(\d+\.\d{2}) (EUR|USD) / month', label)
    if not match:
        return label
    amount = format(Decimal(match[1]).normalize(), 'f')
    if match[2] == 'USD':
        return '%s $ US / mois' % amount.replace('.', ',') if get_language() == 'fr' else 'US$%s / month' % amount
    if get_language() == 'fr':
        return '%s € / mois' % amount.replace('.', ',')
    return '€%s / month' % amount


def context(request=None):
    from .email_updates import form_context
    currency = billing.selected_currency(request) if request is not None else 'eur'
    ready = ((Q(summary__isnull=False) & ~Q(summary='')) |
             (Q(lay_summary__isnull=False) & ~Q(lay_summary='')))
    papers = list(SummaryPaper.objects.filter(ready, lang=get_language())
                  .select_related('paper').order_by('-updated')[:7])
    papers = [p for p in papers if (p.summary or '').strip() or (p.lay_summary or '').strip()]
    return {**form_context(), 'free_tools': free_tools(), 'home_readers': readers(), 'latestpapers': papers[:3], 'example_paper': papers[0] if papers else None,
            'home_price': monthly_price(currency), 'billing_currency': currency, 'billing_currencies': billing.currencies()}
