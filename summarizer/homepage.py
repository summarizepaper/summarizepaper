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


# Free tools shown first on the homepage: (funnel slug, url name or anchor, icon, name, pitch, needs an account, call to action).
# The digest card opens the two-click signup on the page rather than the full settings.
FREE_TOOLS = [
    ('paper', '#arxiv-id', 'file-alt', _('Summaries'), _('Key ideas, results and limits of any paper, in 16 languages.'), False, _('Summarize a paper')),
    ('chat', 'chat', 'comments', _('Chat with papers'), _('Ask a question to one paper or several, answered from the text.'), False, _('Ask a question')),
    ('digest', '#email-updates', 'envelope', _('Daily paper alert'), _('New papers in your fields every morning, with short summaries.'), True, _('Set up my free paper alert')),
    ('letters', 'letters', 'feather-alt', _('Weekly letters'), _('Each week, the papers that matter in your field, put into perspective.'), False, _('Read this week’s letters')),
    ('trends', 'trends', 'chart-line', _('Rising topics'), _('The topics rising on arXiv, bioRxiv and medRxiv, field by field.'), False, _('See rising topics')),
    ('saved_searches', 'saved_searches', 'bookmark', _('Saved searches'), _('Save a search and get its new papers by email or RSS.'), True, _('Save a search')),
]


def free_tools():
    return [{'url': url if url.startswith('#') else reverse(url), 'slug': slug, 'icon': icon, 'name': name,
             'pitch': pitch, 'account': account, 'cta': cta} for slug, url, icon, name, pitch, account, cta in FREE_TOOLS]


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


SAMPLE_TOPIC = 'ai'  # the field the email samples open on, when the visitor has not chosen one
SAMPLE_PAPERS = 3
PREPARED_PAPERS = 6  # per field: the homepage shows SAMPLE_PAPERS of them, the example digest all (digest_web.EXAMPLE_PAPERS)
PREPARED_KEY = 'home-samples-prepared:%s'


def prepare_samples():
    """Once per announcement, the digest summaries of PREPARED_PAPERS papers of each field, in each language, so
    the homepage shows a digest of whichever field a visitor picks (14 calls of the free model). Hourly job."""
    from . import digest, email_updates
    from .models import ArxivBatch
    day = ArxivBatch.objects.order_by('-day').values_list('day', flat=True).first()
    if day is None or cache.get(PREPARED_KEY % day):
        return 0
    papers = sorted(digest.load_papers([day]).values(), key=lambda p: p['arxiv_id'])
    chosen = []
    for categories in email_updates.STARTERS.values():
        field = categories.split(',')
        chosen += [p for p in papers if p['categories'][0] in field and p['abstract'].strip()][:PREPARED_PAPERS]
    chosen = list({p['arxiv_id']: p for p in chosen}.values())  # astro-ph.EP is in two fields
    for lang in ('en', 'fr'):
        digest.ensure_items(chosen, lang)  # only the missing ones are written
    cache.set(PREPARED_KEY % day, True, 4 * 86400)
    return len(chosen)


def email_samples():
    """A real recent digest and weekly letter of each field, to show visitors what an account sends them."""
    lang = get_language()
    key = 'home-samples5:%s' % lang
    samples = cache.get(key)
    if samples is None:
        samples = _email_samples(lang)
        cache.set(key, samples, 3600)
    return samples


def _main_in(field):
    """Digest summaries whose main category is one of these."""
    query = Q()
    for category in field:
        query |= Q(categories=category) | Q(categories__startswith=category + ',')
    return query


def _email_samples(lang):
    """{'fields': a sample per field with a digest or a letter, 'shown': the field shown first}."""
    from . import email_updates, letters
    from .models import DigestItem
    latest = letters.latest_letters(lang)
    recent = DigestItem.objects.filter(lang=lang).exclude(blurb='').order_by('-published', '-created')
    fields = []
    for topic, categories in email_updates.STARTERS.items():
        field = categories.split(',')
        items = [{'arxiv_id': i.arxiv_id, 'title': i.title, 'authors': i.authors,
                  'category': next((c for c in i.categories.split(',') if c in field), i.categories.split(',')[0]),
                  'blurb': i.blurb} for i in recent.filter(_main_in(field))[:SAMPLE_PAPERS]]
        letter = latest.get(topic)
        if not items and not letter:
            continue
        sample = {'topic': topic, 'topic_name': str(letters.TOPICS[topic]['name']), 'digest': items, 'letter': None}
        if letter:
            shown = letters.letter_context(letter)
            sample['letter'] = {'topic': letter.topic, 'day': letter.day, 'title': letter.title, 'dek': letter.dek,
                                'minutes': shown['minutes'], 'sections': [s['heading'] for s in shown['sections']],
                                'papers': sum(len(s['papers']) for s in shown['sections'])}
        fields.append(sample)
    # the preferred field once it has both, else the most complete one
    complete = lambda f: (len(f['digest']) >= SAMPLE_PAPERS, bool(f['letter']), f['topic'] == SAMPLE_TOPIC, len(f['digest']))
    shown = max(fields, key=complete)['topic'] if fields else None
    return {'fields': fields, 'shown': shown, 'digest': any(f['digest'] for f in fields)}


def context(request=None):
    from .email_updates import form_context
    currency = billing.selected_currency(request) if request is not None else 'eur'
    ready = ((Q(summary__isnull=False) & ~Q(summary='')) |
             (Q(lay_summary__isnull=False) & ~Q(lay_summary='')))
    papers = list(SummaryPaper.objects.filter(ready, lang=get_language())
                  .select_related('paper').order_by('-updated')[:7])
    papers = [p for p in papers if (p.summary or '').strip() or (p.lay_summary or '').strip()]
    return {**form_context(), 'free_tools': free_tools(), 'home_readers': readers(), 'latestpapers': papers[:3], 'example_paper': papers[0] if papers else None,
            'email_samples': email_samples(),
            'home_price': monthly_price(currency), 'billing_currency': currency, 'billing_currencies': billing.currencies()}
