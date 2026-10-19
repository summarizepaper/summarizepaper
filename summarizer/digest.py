"""Daily digest of new arXiv papers, by category and optional keywords.

Run once a day (Heroku Scheduler): python manage.py send_digest
"""
import json
import logging
import re
import time
import urllib.parse
import zoneinfo
from xml.etree import ElementTree

import requests
from django.conf import settings
from django.core import signing
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.dateparse import parse_datetime

from . import mailcheck
from .models import DigestDelivery, DigestItem, DigestSubscription
from .utils import ai_models, chat_params

logger = logging.getLogger(__name__)

ARXIV_API = 'https://export.arxiv.org/api/query'
ATOM = '{http://www.w3.org/2005/Atom}'
ARXIV_NS = '{http://arxiv.org/schemas/atom}'
MAX_CATEGORIES = 5
PAPERS_PER_EMAIL = 10
# CloudMailin sends at most 20 emails/hour on our plan; keep room for account emails.
# Subscribers left over are sent in the next hourly runs (see DigestSubscription.CATCH_UP_HOURS).
MAX_EMAILS_PER_RUN = 15
LOOKBACK_DAYS = 4  # arXiv announces papers 1-3 days after submission
UNSUBSCRIBE_SALT = 'digest-unsubscribe'

# Categories offered in the form; any valid arXiv category code is accepted too.
CATEGORIES = [
    ('astro-ph.CO', 'Cosmology and Nongalactic Astrophysics'),
    ('astro-ph.EP', 'Earth and Planetary Astrophysics'),
    ('astro-ph.GA', 'Astrophysics of Galaxies'),
    ('astro-ph.HE', 'High Energy Astrophysical Phenomena'),
    ('astro-ph.IM', 'Instrumentation and Methods for Astrophysics'),
    ('astro-ph.SR', 'Solar and Stellar Astrophysics'),
    ('cond-mat.mtrl-sci', 'Materials Science'),
    ('cond-mat.stat-mech', 'Statistical Mechanics'),
    ('cs.AI', 'Artificial Intelligence'),
    ('cs.CL', 'Computation and Language'),
    ('cs.CR', 'Cryptography and Security'),
    ('cs.CV', 'Computer Vision'),
    ('cs.LG', 'Machine Learning'),
    ('cs.RO', 'Robotics'),
    ('cs.SE', 'Software Engineering'),
    ('econ.GN', 'General Economics'),
    ('eess.SP', 'Signal Processing'),
    ('gr-qc', 'General Relativity and Quantum Cosmology'),
    ('hep-ph', 'High Energy Physics - Phenomenology'),
    ('hep-th', 'High Energy Physics - Theory'),
    ('math.NA', 'Numerical Analysis'),
    ('math.PR', 'Probability'),
    ('physics.ao-ph', 'Atmospheric and Oceanic Physics'),
    ('physics.bio-ph', 'Biological Physics'),
    ('physics.optics', 'Optics'),
    ('q-bio.NC', 'Neurons and Cognition'),
    ('q-bio.QM', 'Quantitative Methods'),
    ('q-fin.ST', 'Statistical Finance'),
    ('quant-ph', 'Quantum Physics'),
    ('stat.ML', 'Machine Learning (Statistics)'),
]
TIMEZONES = {tz for tz in zoneinfo.available_timezones() if '/' in tz and not tz.startswith(('Etc/', 'SystemV/', 'US/'))} | {'UTC'}
CATEGORY_RE = re.compile(r'^[a-z\-]+(\.[A-Za-z\-]+)?$')


def clean_categories(values):
    seen = []
    for value in values:
        value = value.strip()
        if value and CATEGORY_RE.match(value) and value not in seen:
            seen.append(value)
    return seen[:MAX_CATEGORIES]


def site_id(arxiv_abs_id):
    """'hep-th/9901001v1' -> 'hep-th--9901001v1', the form used in our URLs."""
    return arxiv_abs_id.replace('/', '--')


def parse_feed(xml_text):
    papers = []
    root = ElementTree.fromstring(xml_text)
    for entry in root.findall(ATOM + 'entry'):
        abs_url = entry.findtext(ATOM + 'id', '')
        if '/abs/' not in abs_url:
            continue
        papers.append({
            'arxiv_id': site_id(abs_url.split('/abs/')[-1]),
            'title': re.sub(r'\s+', ' ', entry.findtext(ATOM + 'title', '')).strip(),
            'abstract': re.sub(r'\s+', ' ', entry.findtext(ATOM + 'summary', '')).strip(),
            'authors': [a.findtext(ATOM + 'name', '') for a in entry.findall(ATOM + 'author')],
            'categories': [c.get('term') for c in entry.findall(ATOM + 'category')],
            'published': parse_datetime(entry.findtext(ATOM + 'published', '')),
        })
    return papers


def fetch_recent(category, since, max_results=200):
    query = urllib.parse.urlencode({
        'search_query': 'cat:' + category,
        'sortBy': 'submittedDate',
        'sortOrder': 'descending',
        'max_results': max_results,
    })
    response = requests.get(ARXIV_API + '?' + query, timeout=60)
    response.raise_for_status()
    return [p for p in parse_feed(response.text) if p['published'] and p['published'] >= since]


def fetch_all(categories, since):
    papers = {}
    for i, category in enumerate(sorted(categories)):
        if i:
            time.sleep(3)  # arXiv API terms: at most one request every 3 seconds
        try:
            for paper in fetch_recent(category, since):
                papers.setdefault(paper['arxiv_id'], paper)
        except (requests.RequestException, ElementTree.ParseError):
            logger.exception('arXiv fetch failed for %s', category)
    return papers


def select_papers(subscription, papers, already_sent):
    wanted = set(subscription.category_list())
    keywords = subscription.keyword_list()
    scored = []
    for paper in papers.values():
        if paper['arxiv_id'] in already_sent or not wanted.intersection(paper['categories']):
            continue
        score = 0
        if keywords:
            text = (paper['title'] + ' ' + paper['abstract']).lower()
            score = sum(text.count(k) for k in keywords) + 3 * sum(k in paper['title'].lower() for k in keywords)
            if score == 0:
                continue
        scored.append((score, paper['published'], paper))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    return [paper for _, _, paper in scored[:PAPERS_PER_EMAIL]]


def write_blurbs(papers, lang):
    """One short plain-language summary per paper, a single model call per batch of 10."""
    language = 'French' if lang == 'fr' else 'English'
    listing = '\n\n'.join('ID: %s\nTitle: %s\nAbstract: %s' % (p['arxiv_id'], p['title'], p['abstract']) for p in papers)
    prompt = (
        'For each paper below, write a 2-sentence summary in %s that a curious non-specialist can '
        'understand: what was done and why it matters. No jargon, no LaTeX. '
        'Answer with a JSON object {"summaries": [{"id": "...", "summary": "..."}]}.\n\n%s' % (language, listing))
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
        json={'model': ai_models()[0], 'messages': [{'role': 'user', 'content': prompt}],
              'response_format': {'type': 'json_object'}, **chat_params(False, 2500)},
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()['choices'][0]['message']['content']
    return {item['id']: item['summary'].strip() for item in json.loads(content).get('summaries', [])
            if item.get('id') and item.get('summary')}


def ensure_items(papers, lang):
    """Return {arxiv_id: DigestItem}, generating the missing blurbs."""
    ids = [p['arxiv_id'] for p in papers]
    items = {i.arxiv_id: i for i in DigestItem.objects.filter(arxiv_id__in=ids, lang=lang)}
    missing = [p for p in papers if p['arxiv_id'] not in items]
    for start in range(0, len(missing), 10):
        batch = missing[start:start + 10]
        try:
            blurbs = write_blurbs(batch, lang)
        except (requests.RequestException, ValueError, KeyError):
            logger.exception('Blurb generation failed, falling back to abstracts')
            blurbs = {}
        for paper in batch:
            blurb = blurbs.get(paper['arxiv_id']) or (paper['abstract'][:400].rsplit(' ', 1)[0] + '…')
            items[paper['arxiv_id']] = DigestItem.objects.create(
                arxiv_id=paper['arxiv_id'], lang=lang, title=paper['title'],
                authors=', '.join(paper['authors'][:6]) + (' et al.' if len(paper['authors']) > 6 else ''),
                categories=','.join(paper['categories']), published=paper['published'], blurb=blurb)
    return items


def unsubscribe_token(subscription):
    return signing.dumps(subscription.pk, salt=UNSUBSCRIBE_SALT)


def subscription_from_token(token):
    try:
        pk = signing.loads(token, salt=UNSUBSCRIBE_SALT)
    except signing.BadSignature:
        return None
    return DigestSubscription.objects.filter(pk=pk).first()


def send_one(subscription, items, base_url):
    unsubscribe_url = '%s/%s/digest/unsubscribe/%s/' % (base_url, subscription.lang, unsubscribe_token(subscription))
    context = {
        'subscription': subscription,
        'items': items,
        'base_url': base_url,
        'manage_url': '%s/%s/digest/' % (base_url, subscription.lang),
        'unsubscribe_url': unsubscribe_url,
        'date': timezone.now(),
    }
    with translation.override(subscription.lang):
        subject = render_to_string('digest/subject.txt', context).strip()
        text = render_to_string('digest/email.txt', context)
        html = render_to_string('digest/email.html', context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [subscription.user.email],
                                     headers={'List-Unsubscribe': '<%s>' % unsubscribe_url,
                                              'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click'})
    message.attach_alternative(html, 'text/html')
    message.send()


def run(base_url='https://www.summarizepaper.com', dry_run=False, only_user=None, stdout=None, now=None):
    """Send the digests that are due (meant to run every hour); --user ignores the schedule."""
    now = now or timezone.now()
    subscriptions = DigestSubscription.objects.filter(active=True).select_related('user').exclude(user__email='')
    if only_user:
        subscriptions = list(subscriptions.filter(user__username=only_user))
    else:
        suppressed = mailcheck.suppressed()
        subscriptions = [s for s in subscriptions if s.is_due(now) and s.user.email.strip().lower() not in suppressed]
        # longest-waiting first, and no more than the mail provider allows per hour
        subscriptions.sort(key=lambda s: s.last_sent or s.created)
        if len(subscriptions) > MAX_EMAILS_PER_RUN:
            logger.warning('%d digests due, sending %d this hour', len(subscriptions), MAX_EMAILS_PER_RUN)
            subscriptions = subscriptions[:MAX_EMAILS_PER_RUN]
    if not subscriptions:
        return 0

    since = timezone.now() - timezone.timedelta(days=LOOKBACK_DAYS)
    papers = fetch_all({c for s in subscriptions for c in s.category_list()}, since)
    sent = 0
    for subscription in subscriptions:
        already = set(DigestDelivery.objects.filter(subscription=subscription).values_list('arxiv_id', flat=True))
        chosen = select_papers(subscription, papers, already)
        if not chosen:
            continue
        items = ensure_items(chosen, subscription.lang)
        ordered = [items[p['arxiv_id']] for p in chosen]
        if stdout:
            stdout.write('%s: %d papers' % (subscription.user.username, len(ordered)))
        if dry_run:
            continue
        try:
            send_one(subscription, ordered, base_url)
        except Exception:
            logger.exception('Digest email to %s failed', subscription.user.username)
            continue
        DigestDelivery.objects.bulk_create(
            [DigestDelivery(subscription=subscription, arxiv_id=i.arxiv_id) for i in ordered], ignore_conflicts=True)
        subscription.last_sent = now
        subscription.save(update_fields=['last_sent'])
        sent += 1
    return sent


# --- views -------------------------------------------------------------------

from django.shortcuts import redirect, render  # noqa: E402
from django.utils.translation import get_language  # noqa: E402
from django.views.decorators.csrf import csrf_exempt  # noqa: E402


def digest_page(request):
    subscription = None
    if request.user.is_authenticated:
        subscription = DigestSubscription.objects.filter(user=request.user).first()
        if request.method == 'POST':
            categories = clean_categories(request.POST.getlist('categories') + request.POST.get('other', '').split(','))
            if not categories:
                return render(request, 'summarizer/digest.html', _digest_context(request, subscription, error=True))
            if subscription is None:
                subscription = DigestSubscription(user=request.user)
            subscription.categories = ','.join(categories)
            subscription.keywords = ', '.join(k.strip() for k in request.POST.get('keywords', '').split(',') if k.strip())[:500]
            subscription.lang = request.POST.get('lang') if request.POST.get('lang') in ('en', 'fr') else get_language()
            try:
                subscription.send_hour = min(23, max(0, int(request.POST.get('send_hour', 7))))
            except ValueError:
                subscription.send_hour = 7
            if request.POST.get('timezone') in TIMEZONES:
                subscription.timezone = request.POST['timezone']
            subscription.active = True
            subscription.save()
            return redirect('/%s/digest/?saved=1' % get_language())
    return render(request, 'summarizer/digest.html', _digest_context(request, subscription))


def _digest_context(request, subscription, error=False):
    chosen = subscription.category_list() if subscription else []
    known = {code for code, _ in CATEGORIES}
    return {
        'subscription': subscription,
        'categories': CATEGORIES,
        'chosen': chosen,
        'other': ', '.join(c for c in chosen if c not in known),
        'max_categories': MAX_CATEGORIES,
        'papers_per_email': PAPERS_PER_EMAIL,
        'saved': request.GET.get('saved') == '1',
        'hours': range(24),
        'timezones': sorted(TIMEZONES),
        'send_hour': subscription.send_hour if subscription else 7,
        'timezone': subscription.timezone if subscription else '',
        'error': error,
    }


@csrf_exempt  # mail clients send the one-click unsubscribe POST without a CSRF token
def unsubscribe(request, token):
    subscription = subscription_from_token(token)
    done = False
    if subscription is not None and request.method == 'POST':
        subscription.active = False
        subscription.save(update_fields=['active'])
        done = True
    return render(request, 'summarizer/digest_unsubscribe.html', {'subscription': subscription, 'done': done})
