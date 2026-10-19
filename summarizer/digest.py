"""Email digests of new arXiv papers, by category, keywords and, with Premium, a research profile.

Run every hour (Heroku Scheduler): python manage.py send_digest

Whatever costs a model call is done once and shared by every subscriber: the
2-sentence blurb (per paper and language), the embedding (per paper), the full
summary (per paper and language) and the weekly synthesis (per category,
language and day). Only the ranking is computed per subscriber.
"""
import datetime
import hashlib
import json
import logging
import re
import time
import unicodedata
import zoneinfo
from xml.etree import ElementTree

import numpy as np
import requests
from django.conf import settings
from django.core import signing
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.dateparse import parse_datetime

from . import access, mailcheck, similar
from .models import CategorySynthesis, DigestDelivery, DigestItem, DigestSubscription, DigestVector, SummaryPaper
from .utils import ai_models, chat_params

logger = logging.getLogger(__name__)

ARXIV_API = 'https://export.arxiv.org/api/query'
ATOM = '{http://www.w3.org/2005/Atom}'
ARXIV_NS = '{http://arxiv.org/schemas/atom}'

FREE = {'categories': 5, 'papers': 10, 'digests': 1}
PREMIUM = {'categories': 15, 'papers': 25, 'digests': 5}
MAX_CATEGORIES = FREE['categories']
PAPERS_PER_EMAIL = FREE['papers']
MAX_AUTHORS = 10
MAX_REFERENCES = 5
PRESUMMARIZED = 3  # the top papers of a Premium digest get their full summary written before the email
MAX_PRESUMMARIES_PER_RUN = 9  # each takes about a minute
MAX_SYNTHESES_PER_EMAIL = 5
SYNTHESIS_MIN_PAPERS = 5
SYNTHESIS_MAX_PAPERS = 600

# CloudMailin sends at most 20 emails/hour on our plan; keep room for account emails.
# Subscribers left over are sent in the next hourly runs (see DigestSubscription.CATCH_UP_HOURS).
MAX_EMAILS_PER_RUN = 15
LOOKBACK_DAYS = 4  # arXiv announces papers 1-3 days after submission
WEEKLY_LOOKBACK_DAYS = 8
PAGE_SIZE = 500
MAX_PAGES = 12  # a category with more papers than this in the window loses its oldest ones
MAX_AUTHOR_PAGES = 3
SECONDS_BETWEEN_CALLS = 3  # arXiv API terms: at most one request every 3 seconds
RETRY_WAITS = (10, 30, 90)  # arXiv answers 429 "Rate exceeded" when busy
EMBED_BATCH = 256
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
ARXIV_ID_RE = re.compile(r'(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)')


class ArxivUnavailable(Exception):
    pass


def limits(paid):
    return PREMIUM if paid else FREE


def clean_categories(values, limit=MAX_CATEGORIES):
    seen = []
    for value in values:
        value = value.strip()
        if value and CATEGORY_RE.match(value) and value not in seen:
            seen.append(value)
    return seen[:limit]


def clean_references(text):
    """arXiv ids found in what the user typed (ids or arXiv links), in arXiv form."""
    found = []
    for match in ARXIV_ID_RE.findall(text.replace('--', '/')):
        if match not in found:
            found.append(match)
    return found[:MAX_REFERENCES]


def clean_authors(text):
    names = []
    for name in text.split(','):
        name = re.sub(r'\s+', ' ', name).strip()[:80]
        if name_key(name) and name not in names:
            names.append(name)
    return names[:MAX_AUTHORS]


def name_key(name):
    """'Yann LeCun', 'Y. LeCun' and 'Y LeCún' -> ('lecun', 'y'): last name and first initial."""
    plain = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    parts = re.findall(r'[a-z]+', plain)
    if len(parts) < 2:
        return None
    return parts[-1], parts[0][0]


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


_last_call = [0.0]


def arxiv_get(params):
    """One arXiv API query, spaced from the previous one and retried when arXiv asks us to slow down."""
    error = None
    for wait in RETRY_WAITS + (None,):
        pause = _last_call[0] + SECONDS_BETWEEN_CALLS - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_call[0] = time.monotonic()
        try:
            response = requests.get(ARXIV_API, params=params, timeout=60)
            if response.status_code == 200:
                return parse_feed(response.text)
            error = 'HTTP %s: %s' % (response.status_code, response.text[:80])
        except (requests.RequestException, ElementTree.ParseError) as e:
            error = repr(e)
        if wait is not None:
            logger.info('arXiv query %s failed (%s), retrying in %ds', params.get('search_query'), error, wait)
            time.sleep(wait)
    raise ArxivUnavailable(error)


def fetch_recent(query, since, max_pages=MAX_PAGES):
    """Papers matching an arXiv query, submitted since `since`, newest first, page after page."""
    papers = []
    for page in range(max_pages):
        batch = arxiv_get({'search_query': query, 'sortBy': 'submittedDate', 'sortOrder': 'descending',
                           'start': page * PAGE_SIZE, 'max_results': PAGE_SIZE})
        papers += [p for p in batch if p['published'] and p['published'] >= since]
        if len(batch) < PAGE_SIZE or any(p['published'] and p['published'] < since for p in batch):
            return papers
    logger.warning('%s: more than %d papers since %s, the oldest are left out', query, len(papers), since)
    return papers


def fetch_all(windows):
    """{category: since} -> ({arxiv_id: paper}, categories that could not be fetched)."""
    papers, failed = {}, set()
    for category in sorted(windows):
        try:
            for paper in fetch_recent('cat:' + category, windows[category]):
                papers.setdefault(paper['arxiv_id'], paper)
        except ArxivUnavailable:
            logger.exception('arXiv fetch failed for %s', category)
            failed.add(category)
    return papers, failed


def fetch_authors(windows):
    """{name_key: (name, since)} -> {name_key: [paper]}, recent papers by each followed author."""
    found = {}
    for key, (name, since) in sorted(windows.items()):
        try:
            papers = fetch_recent('au:"%s"' % key[0], since, max_pages=MAX_AUTHOR_PAGES)
        except ArxivUnavailable:
            logger.exception('arXiv fetch failed for author %s', name)
            continue
        found[key] = [p for p in papers if key in {name_key(a) for a in p['authors']}]
    return found


def fetch_ids(ids):
    if not ids:
        return []
    return arxiv_get({'id_list': ','.join(ids), 'max_results': len(ids)})


# --- ranking -----------------------------------------------------------------

def paper_text(paper):
    return re.sub(r'[ \t\r\f\v]+', ' ', '%s\n\n%s' % (paper['title'], paper['abstract'])).strip()[:similar.MAX_INPUT_CHARS]


def ensure_vectors(papers):
    """{arxiv_id: vector} for these papers, embedding only the ones never seen before."""
    ids = [p['arxiv_id'] for p in papers]
    vectors = {}
    for start in range(0, len(ids), 1000):
        for aid, vec in DigestVector.objects.filter(arxiv_id__in=ids[start:start + 1000]).values_list('arxiv_id', 'vector'):
            vectors[aid] = np.frombuffer(vec, dtype=np.float32)
    missing = [p for p in papers if p['arxiv_id'] not in vectors]
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start:start + EMBED_BATCH]
        try:
            embedded = similar.embed([paper_text(p) for p in batch])
        except (requests.RequestException, KeyError, ValueError):
            logger.exception('Embedding of %d digest papers failed', len(batch))
            continue
        DigestVector.objects.bulk_create([DigestVector(arxiv_id=p['arxiv_id'], vector=v.tobytes())
                                          for p, v in zip(batch, embedded)], ignore_conflicts=True)
        vectors.update((p['arxiv_id'], v) for p, v in zip(batch, embedded))
    return vectors


def profile_vector(subscription):
    """Embedding of the research profile and reference papers, recomputed only when they change."""
    text = subscription.profile.strip()
    refs = subscription.reference_list()[:MAX_REFERENCES]
    if not text and not refs:
        return None
    key = hashlib.sha256(json.dumps([similar.MODEL, text, refs]).encode()).hexdigest()
    if subscription.profile_key == key and subscription.profile_vector:
        return np.frombuffer(bytes(subscription.profile_vector), dtype=np.float32)
    try:
        inputs = ([text] if text else []) + [paper_text(p) for p in fetch_ids(refs)]
        if not inputs:
            return None
        vector = similar.embed(inputs).mean(axis=0)
    except (ArxivUnavailable, requests.RequestException, KeyError, ValueError):
        logger.exception('Profile embedding failed for digest %s', subscription.pk)
        return None
    vector = (vector / np.linalg.norm(vector)).astype(np.float32)
    subscription.profile_vector, subscription.profile_key = vector.tobytes(), key
    subscription.save(update_fields=['profile_vector', 'profile_key'])
    return vector


def candidates(subscription, papers, already_sent, since=None, categories=None, exclude=()):
    wanted = set(categories or subscription.category_list())
    return [p for p in papers.values()
            if p['arxiv_id'] not in already_sent and p['arxiv_id'] not in exclude
            and wanted.intersection(p['categories']) and (since is None or p['published'] >= since)]


def select_papers(subscription, papers, already_sent, since=None, limit=PAPERS_PER_EMAIL, categories=None,
                  profile=None, vectors=None, exclude=()):
    """The best new papers for this digest: keyword matches, ranked by closeness to the profile if there is one."""
    keywords = subscription.keyword_list()
    scored = []
    for paper in candidates(subscription, papers, already_sent, since, categories, exclude):
        score = 0
        if keywords:
            text = (paper['title'] + ' ' + paper['abstract']).lower()
            score = sum(text.count(k) for k in keywords) + 3 * sum(k in paper['title'].lower() for k in keywords)
            if score == 0:
                continue
        closeness = 0.0
        if profile is not None:
            vector = (vectors or {}).get(paper['arxiv_id'])
            closeness = float(vector @ profile) if vector is not None else -1.0
        scored.append((closeness, score, paper['published'], paper))
    scored.sort(key=lambda s: s[:3], reverse=True)
    return [paper for *_, paper in scored[:limit]]


def author_papers(subscription, by_author, already_sent, since):
    keys = {name_key(a) for a in subscription.author_list()[:MAX_AUTHORS]}
    found = {}
    for key in keys:
        for paper in by_author.get(key, []):
            if paper['arxiv_id'] not in already_sent and paper['published'] >= since:
                found.setdefault(paper['arxiv_id'], paper)
    return sorted(found.values(), key=lambda p: p['published'], reverse=True)


# --- shared content ------------------------------------------------------------

def write_blurbs(papers, lang):
    """One short plain-language summary per paper, a single model call per batch of 10."""
    language = 'French' if lang == 'fr' else 'English'
    listing = '\n\n'.join('ID: %s\nTitle: %s\nAbstract: %s' % (p['arxiv_id'], p['title'], p['abstract']) for p in papers)
    prompt = (
        'For each paper below, write a 2-sentence summary in %s that a curious non-specialist can '
        'understand: what was done and why it matters. No jargon, no LaTeX. '
        'Answer with a JSON object {"summaries": [{"id": "...", "summary": "..."}]}.\n\n%s' % (language, listing))
    content = _chat(prompt, ai_models()[0], False, 2500)
    return {item['id']: item['summary'].strip() for item in json.loads(content).get('summaries', [])
            if isinstance(item, dict) and item.get('id') and item.get('summary')}


def _chat(prompt, model, paid, max_tokens):
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
        json={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
              'response_format': {'type': 'json_object'}, **chat_params(paid, max_tokens)},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()['choices'][0]['message']['content']


def ensure_items(papers, lang):
    """Return {arxiv_id: DigestItem}, generating the missing blurbs."""
    ids = [p['arxiv_id'] for p in papers]
    items = {i.arxiv_id: i for i in DigestItem.objects.filter(arxiv_id__in=ids, lang=lang)}
    missing = [p for p in papers if p['arxiv_id'] not in items]
    for start in range(0, len(missing), 10):
        batch = missing[start:start + 10]
        try:
            blurbs = write_blurbs(batch, lang)
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
            logger.exception('Blurb generation failed, falling back to abstracts')
            blurbs = {}
        for paper in batch:
            item = DigestItem(
                arxiv_id=paper['arxiv_id'], lang=lang, title=paper['title'],
                authors=', '.join(paper['authors'][:6]) + (' et al.' if len(paper['authors']) > 6 else ''),
                categories=','.join(paper['categories']), published=paper['published'],
                blurb=blurbs.get(paper['arxiv_id'], ''))
            if item.blurb:
                # stored once and reused by every subscriber; get_or_create in case two runs overlap
                item, _ = DigestItem.objects.get_or_create(arxiv_id=item.arxiv_id, lang=lang, defaults={
                    f: getattr(item, f) for f in ('title', 'authors', 'categories', 'published', 'blurb')})
            else:
                # not stored, so the next subscriber or run tries the model again
                item.blurb = paper['abstract'][:400].rsplit(' ', 1)[0] + '…'
            items[paper['arxiv_id']] = item
    return items


def summarized(arxiv_ids, lang):
    return set(SummaryPaper.objects.filter(paper__arxiv_id__in=arxiv_ids, lang=lang)
               .exclude(summary__isnull=True).exclude(summary='').exclude(summary__startswith='Error')
               .values_list('paper__arxiv_id', flat=True))


def presummarize(arxiv_ids, lang, budget):
    """Write the full summary of these papers now, so the links in the email open on a finished page.

    budget is a one-element list: how many summaries this run may still write.
    """
    from .consumers import summarize_now
    done = summarized(arxiv_ids, lang)
    for arxiv_id in arxiv_ids:
        if arxiv_id in done or budget[0] <= 0:
            continue
        budget[0] -= 1
        try:
            summarize_now(arxiv_id, lang)
        except Exception:
            logger.exception('Summary of %s for the digest failed', arxiv_id)


def category_synthesis(category, lang, day, papers):
    """{'trends': [...], 'notable': [...]} for the last 7 days of a category, written once per language and day."""
    existing = CategorySynthesis.objects.filter(category=category, lang=lang, day=day).first()
    if existing:
        return json.loads(existing.content)
    start = datetime.datetime.combine(day, datetime.time(), tzinfo=datetime.timezone.utc) - datetime.timedelta(days=7)
    week = sorted((p for p in papers.values() if category in p['categories'] and p['published'] >= start),
                  key=lambda p: p['published'], reverse=True)[:SYNTHESIS_MAX_PAPERS]
    if len(week) < SYNTHESIS_MIN_PAPERS:
        return None
    by_id = {p['arxiv_id']: p for p in week}
    language = 'French' if lang == 'fr' else 'English'
    prompt = (
        'Here are the titles of the %d papers posted on arXiv in the category %s over the last 7 days. '
        'In %s, write for a researcher of this field: the 3 to 5 main trends or topics of the week, each with a '
        'short title and 2 sentences, and the IDs of the 3 papers most likely to matter. Be concrete, no hype, no LaTeX. '
        'Answer with a JSON object {"trends": [{"title": "...", "text": "..."}], "notable": ["ID", ...]}.\n\n%s'
        % (len(week), category, language, '\n'.join('%s | %s' % (p['arxiv_id'], p['title']) for p in week)))
    try:
        answer = json.loads(_chat(prompt, ai_models()[1], True, 3000))
        content = {
            'category': category,
            'trends': [{'title': str(t['title']).strip(), 'text': str(t['text']).strip()}
                       for t in answer.get('trends', [])[:5] if isinstance(t, dict) and t.get('title') and t.get('text')],
            'notable': [{'arxiv_id': aid, 'title': by_id[aid]['title']}
                        for aid in answer.get('notable', [])[:3] if isinstance(aid, str) and aid in by_id],
        }
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        logger.exception('Synthesis of %s failed', category)
        return None
    if not content['trends']:
        return None
    CategorySynthesis.objects.get_or_create(category=category, lang=lang, day=day, defaults={
        'content': json.dumps(content), 'papers': len(week)})
    return content


# --- sending -----------------------------------------------------------------

def unsubscribe_token(subscription):
    return signing.dumps(subscription.pk, salt=UNSUBSCRIBE_SALT)


def subscription_from_token(token):
    try:
        pk = signing.loads(token, salt=UNSUBSCRIBE_SALT)
    except signing.BadSignature:
        return None
    return DigestSubscription.objects.filter(pk=pk).first()


def send_one(subscription, items, base_url, from_authors=(), syntheses=(), ready=(), now=None):
    unsubscribe_url = '%s/%s/digest/unsubscribe/%s/' % (base_url, subscription.lang, unsubscribe_token(subscription))
    context = {
        'subscription': subscription,
        'items': items,
        'from_authors': from_authors,
        'count': len(items) + len(from_authors),
        'syntheses': syntheses,
        'ready': set(ready),
        'weekly': subscription.frequency == 'weekly',
        'base_url': base_url,
        'manage_url': '%s/%s/digest/?d=%s' % (base_url, subscription.lang, subscription.pk),
        'unsubscribe_url': unsubscribe_url,
        'date': now or timezone.now(),
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


class Plan:
    """What one due digest needs: its time window, its limits and its Premium extras."""

    def __init__(self, subscription, paid, now):
        self.subscription = subscription
        self.paid = paid
        self.limits = limits(paid)
        self.categories = subscription.category_list()[:self.limits['categories']]
        weekly = subscription.frequency == 'weekly'
        self.since = now - timezone.timedelta(days=WEEKLY_LOOKBACK_DAYS if weekly else LOOKBACK_DAYS)
        try:
            local = now.astimezone(zoneinfo.ZoneInfo(subscription.timezone))
        except Exception:
            local = now
        # Premium: what happened this week, in the weekly digest or in Monday's daily digest
        self.synthesis = paid and (weekly or local.weekday() == 0)
        self.window = now - timezone.timedelta(days=WEEKLY_LOOKBACK_DAYS) if self.synthesis else self.since
        self.authors = subscription.author_list()[:MAX_AUTHORS] if paid else []
        self.ranked = paid and bool(subscription.profile.strip() or subscription.reference_papers.strip())


def due_plans(now, only_user=None):
    subscriptions = DigestSubscription.objects.filter(active=True).select_related('user').exclude(user__email='')
    if only_user:
        subscriptions = list(subscriptions.filter(user__username=only_user))
    else:
        suppressed = mailcheck.suppressed()
        subscriptions = [s for s in subscriptions if s.is_due(now) and s.user.email.strip().lower() not in suppressed]
    paid, first = {}, {}
    for s in subscriptions:
        if s.user_id not in paid:
            paid[s.user_id] = access.is_paid(s.user)
            first[s.user_id] = DigestSubscription.objects.filter(user_id=s.user_id).values_list('pk', flat=True).first()
    # without Premium (or after it ended) only the first digest of an account is sent
    plans = [Plan(s, paid[s.user_id], now) for s in subscriptions if paid[s.user_id] or s.pk == first[s.user_id]]
    if not only_user:
        # Premium first, then longest-waiting, and no more than the mail provider allows per hour
        plans.sort(key=lambda p: (not p.paid, p.subscription.last_sent or p.subscription.created))
        if len(plans) > MAX_EMAILS_PER_RUN:
            logger.warning('%d digests due, sending %d this hour', len(plans), MAX_EMAILS_PER_RUN)
            plans = plans[:MAX_EMAILS_PER_RUN]
    return plans


def run(base_url='https://www.summarizepaper.com', dry_run=False, only_user=None, stdout=None, now=None):
    """Send the digests that are due (meant to run every hour); --user ignores the schedule."""
    now = now or timezone.now()
    plans = due_plans(now, only_user)
    if not plans:
        return 0

    windows = {}
    for plan in plans:
        for category in plan.categories:
            windows[category] = min(windows.get(category, plan.window), plan.window)
    papers, failed = fetch_all(windows)
    author_windows = {}
    for plan in plans:
        for name in plan.authors:
            key = name_key(name)
            author_windows[key] = (name, min(author_windows.get(key, (name, plan.since))[1], plan.since))
    by_author = fetch_authors(author_windows) if author_windows else {}

    budget = [MAX_PRESUMMARIES_PER_RUN]
    sent = 0
    for plan in plans:
        subscription = plan.subscription
        if failed.intersection(plan.categories):
            logger.warning('Digest %s postponed: arXiv did not answer for %s', subscription.pk, failed)
            continue
        already = set(DigestDelivery.objects.filter(subscription=subscription).values_list('arxiv_id', flat=True))
        from_authors = author_papers(subscription, by_author, already, plan.since)[:plan.limits['papers']] if plan.authors else []
        profile = vectors = None
        if plan.ranked:
            profile = profile_vector(subscription)
            if profile is not None:
                vectors = ensure_vectors(candidates(subscription, papers, already, plan.since, plan.categories))
        chosen = select_papers(subscription, papers, already, plan.since, plan.limits['papers'] - len(from_authors),
                               plan.categories, profile, vectors, exclude={p['arxiv_id'] for p in from_authors})
        if not chosen and not from_authors:
            continue
        items = ensure_items(from_authors + chosen, subscription.lang)
        author_items = [items[p['arxiv_id']] for p in from_authors]
        ordered = [items[p['arxiv_id']] for p in chosen]
        if stdout:
            stdout.write('%s: %d papers%s' % (subscription.user.username, len(author_items) + len(ordered),
                                               ' (Premium)' if plan.paid else ''))
        if dry_run:
            continue
        syntheses = []
        if plan.synthesis:
            for category in plan.categories[:MAX_SYNTHESES_PER_EMAIL]:
                content = category_synthesis(category, subscription.lang, now.date(), papers)
                if content:
                    syntheses.append(content)
        all_ids = [i.arxiv_id for i in author_items + ordered]
        if plan.paid:
            presummarize(all_ids[:PRESUMMARIZED], subscription.lang, budget)
        try:
            send_one(subscription, ordered, base_url, author_items, syntheses, summarized(all_ids, subscription.lang), now)
        except Exception:
            logger.exception('Digest email to %s failed', subscription.user.username)
            continue
        DigestDelivery.objects.bulk_create(
            [DigestDelivery(subscription=subscription, arxiv_id=aid) for aid in all_ids], ignore_conflicts=True)
        subscription.last_sent = now
        subscription.save(update_fields=['last_sent'])
        sent += 1
    return sent


def cleanup(days=30):
    """Forget the embeddings of papers too old to appear in a digest again."""
    return DigestVector.objects.filter(created__lt=timezone.now() - timezone.timedelta(days=days)).delete()[0]


# --- views -------------------------------------------------------------------

from django.http import Http404  # noqa: E402
from django.shortcuts import redirect, render  # noqa: E402
from django.utils.translation import get_language  # noqa: E402
from django.views.decorators.csrf import csrf_exempt  # noqa: E402


def digest_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/digest.html', _digest_context(request, None, [], False))
    paid = access.is_paid(request.user)
    lim = limits(paid)
    digests = list(request.user.digests.all())
    wanted = request.POST.get('digest') if request.method == 'POST' else request.GET.get('d')
    if wanted == 'new' or (not wanted and not digests):
        subscription = None
    elif wanted:
        subscription = next((d for d in digests if str(d.pk) == wanted), None)
        if subscription is None:
            raise Http404
    else:
        subscription = digests[0]
    can_create = len(digests) < lim['digests']
    if subscription is None and not can_create:
        return redirect('/%s/digest/' % get_language())

    if request.method == 'POST':
        if request.POST.get('action') == 'delete' and subscription is not None:
            subscription.delete()
            return redirect('/%s/digest/' % get_language())
        categories = clean_categories(request.POST.getlist('categories') + request.POST.get('other', '').split(','),
                                      lim['categories'])
        if not categories:
            return render(request, 'summarizer/digest.html', _digest_context(request, subscription, digests, paid, error=True))
        if subscription is None:
            subscription = DigestSubscription(user=request.user)
        subscription.categories = ','.join(categories)
        subscription.keywords = ', '.join(k.strip() for k in request.POST.get('keywords', '').split(',') if k.strip())[:500]
        subscription.lang = request.POST.get('lang') if request.POST.get('lang') in ('en', 'fr') else get_language()
        subscription.frequency = 'weekly' if request.POST.get('frequency') == 'weekly' else 'daily'
        subscription.weekday = _int(request.POST.get('weekday'), 0, 0, 4)
        subscription.send_hour = _int(request.POST.get('send_hour'), 7, 0, 23)
        if request.POST.get('timezone') in TIMEZONES:
            subscription.timezone = request.POST['timezone']
        if paid:  # kept as they are when Premium has ended, and used again if it comes back
            subscription.name = request.POST.get('name', '').strip()[:80]
            subscription.profile = request.POST.get('profile', '').strip()[:2000]
            subscription.reference_papers = ', '.join(clean_references(request.POST.get('reference_papers', '')))
            subscription.authors = ', '.join(clean_authors(request.POST.get('authors', '')))
        subscription.active = True
        subscription.save()
        return redirect('/%s/digest/?d=%s&saved=1' % (get_language(), subscription.pk))
    return render(request, 'summarizer/digest.html', _digest_context(request, subscription, digests, paid))


def _int(value, default, low, high):
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return default


def _digest_context(request, subscription, digests, paid, error=False):
    lim = limits(paid)
    chosen = subscription.category_list() if subscription else []
    known = {code for code, _ in CATEGORIES}
    return {
        'subscription': subscription,
        'digests': digests,
        'paid': paid,
        'can_create': len(digests) < lim['digests'],
        'inactive_digests': [] if paid else digests[1:],
        'categories': CATEGORIES,
        'chosen': chosen,
        'other': ', '.join(c for c in chosen if c not in known),
        'max_categories': lim['categories'],
        'papers_per_email': lim['papers'],
        'premium': PREMIUM,
        'max_authors': MAX_AUTHORS,
        'max_references': MAX_REFERENCES,
        'saved': request.GET.get('saved') == '1',
        'hours': range(24),
        'weekdays': range(5),
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
