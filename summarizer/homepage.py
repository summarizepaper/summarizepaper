"""Homepage presentation data; never generate summaries or send email on a visit."""
import re
from decimal import Decimal

from django.conf import settings
from django.db.models import Q
from django.utils.translation import get_language

from . import billing
from .models import SummaryPaper


def monthly_price():
    """Use the same cached Stripe quote as Premium, with a compact local label."""
    if not billing.billing_enabled():
        return None
    label = billing.price_label(settings.STRIPE_PRICE_MONTHLY)
    if not label:
        return None
    match = re.fullmatch(r'(\d+\.\d{2}) EUR / month', label)
    if not match:
        return label
    amount = format(Decimal(match[1]).normalize(), 'f')
    if get_language() == 'fr':
        return '%s € / mois' % amount.replace('.', ',')
    return '€%s / month' % amount


def context():
    ready = ((Q(summary__isnull=False) & ~Q(summary='')) |
             (Q(lay_summary__isnull=False) & ~Q(lay_summary='')))
    papers = list(SummaryPaper.objects.filter(ready, lang=get_language())
                  .select_related('paper').order_by('-updated')[:7])
    papers = [p for p in papers if (p.summary or '').strip() or (p.lay_summary or '').strip()]
    return {'latestpapers': papers[:3], 'example_paper': papers[0] if papers else None,
            'home_price': monthly_price()}
