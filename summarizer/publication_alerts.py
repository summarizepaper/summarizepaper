"""An email when a paper someone follows is published in a journal, retracted or corrected (Premium).

The papers are the ones followed for citation alerts (citations.py), which include the papers of a watched
collection. Once a day the hourly run checks again the editorial status of the followed arXiv papers
(publication_status.lookup: arXiv's journal reference, Crossref's explicit links and strict title-and-author
matches, and the retractions and corrections Crossref links to the article), MAX_CHECKED_PER_DAY at most, the
oldest checks first and at arXiv's pace. Papers not published yet are checked every day, published ones once a
month, for notices. A paper whose status becomes published or forthcoming, or that gets a new notice, queues an
alert for each follower; the first check of a paper only records its status, since what was already true when
someone followed it is not news. Each user gets one email with what was queued.
"""
import datetime
import json
import logging
import re
import time

from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.template.loader import render_to_string
from django.utils import translation

from . import access, digest, mailcheck, publication_status, sources
from .models import ArxivPaper, FollowedPaper, Generated, PublicationAlert

logger = logging.getLogger(__name__)

MAX_CHECKED_PER_DAY = 60  # about five minutes of the hourly run, once a day
ARXIV_SECONDS = 3
PUBLISHED = ('published', 'forthcoming')
PUBLISHED_EVERY = datetime.timedelta(days=30)
DAILY_KEY = 'publications-checked:%s'


def followed_papers():
    """{followed id: (latest ArxivPaper we have of it, set of user ids)} for the followed arXiv papers."""
    users = {}
    for user_id, arxiv_id in FollowedPaper.objects.values_list('user_id', 'arxiv_id'):
        if not sources.is_external(arxiv_id):
            users.setdefault(arxiv_id, set()).add(user_id)
    found = {}
    for base, followers in users.items():
        prefix = base.replace('/', '--') + 'v'
        versions = [p for p in ArxivPaper.objects.filter(arxiv_id__startswith=prefix) if re.fullmatch(r'v\d+', p.arxiv_id[len(prefix) - 1:])]
        if versions:
            found[base] = (max(versions, key=lambda p: int(p.arxiv_id[len(prefix):])), followers)
    return found


def due(report, now):
    """Whether a paper's status should be checked again now."""
    if not report:
        return True
    every = PUBLISHED_EVERY if report.get('status') in PUBLISHED else publication_status.TTL
    return publication_status.stale(report, every)


def news(before, after):
    """[(kind, journal, doi)] of what a new check found that the previous one did not."""
    if not before:
        return []
    found = []
    if after.get('status') in PUBLISHED and before.get('status') not in PUBLISHED:
        found.append(('published', after.get('journal', ''), after.get('doi', '')))
    seen = {(n.get('type'), n.get('doi')) for n in before.get('updates') or []}
    for notice in after.get('updates') or []:
        if (notice.get('type'), notice.get('doi')) not in seen:
            found.append((notice.get('type', ''), after.get('journal', ''), notice.get('doi', '')))
    return found


def check(now, limit=MAX_CHECKED_PER_DAY):
    """Check the papers due, oldest checks first; queue the alerts. Returns how many alerts were queued."""
    papers = followed_papers()
    reports = {base: publication_status.latest(paper.arxiv_id) for base, (paper, _users) in papers.items()}
    order = sorted((b for b in papers if due(reports[b], now)), key=lambda b: (reports[b] or {}).get('checked_at', ''))
    queued = 0
    for index, base in enumerate(order[:limit]):
        paper, followers = papers[base]
        if index:
            time.sleep(ARXIV_SECONDS)
        try:
            report = publication_status.lookup(paper)
        except Exception:
            logger.warning('Publication check failed for %s', paper.arxiv_id)
            continue
        Generated.objects.update_or_create(kind=publication_status.KIND, key=paper.arxiv_id, lang='en',
                                           defaults={'status': 'done', 'content': json.dumps(report)})
        for kind, journal, doi in news(reports[base], report):
            for user_id in followers:
                _alert, created = PublicationAlert.objects.get_or_create(
                    user_id=user_id, arxiv_id=base, kind=kind, doi=doi or '',
                    defaults={'paper': paper.arxiv_id, 'title': ' '.join((paper.title or '').split()), 'journal': journal or ''})
                queued += created
    return queued


def items_of(alerts, base_url, lang):
    """The queued alerts as email items, in the active language."""
    from .collection_tools import NOTICE_LABELS
    return [{'title': a.title or a.arxiv_id, 'kind': a.kind, 'journal': a.journal, 'doi': a.doi,
             'label': str(NOTICE_LABELS.get(a.kind, ('', ''))[0]),
             'url': '%s/%s/paper/%s/' % (base_url, lang, a.paper or a.arxiv_id.replace('/', '--'))} for a in alerts]


def bundle_section(user, base_url, lang):
    """The queued alerts as a section of the user's digest (see digest_bundle), in the active language."""
    alerts = list(PublicationAlert.objects.filter(user=user, emailed=False))
    if not alerts or not access.has_tool(user, 'citations'):
        return None
    items = []
    for item in items_of(alerts, base_url, lang):
        if item['kind'] == 'published':
            label = (translation.gettext('Published in %(journal)s') % {'journal': item['journal']} if item['journal']
                     else translation.gettext('Published in a journal'))
        else:
            label = item['label']
        items.append({'title': item['title'], 'url': item['url'], 'label': label, 'note': ''})
    return {'title': translation.gettext('Published, retracted or corrected'), 'items': items, 'more': 0,
            'manage_url': '%s/%s/citations/' % (base_url, lang),
            'commit': lambda: PublicationAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True)}


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """One email per user with the alerts queued. Returns the emails sent."""
    from django.contrib.auth.models import User
    from . import digest_bundle
    suppressed = mailcheck.suppressed()
    sent = 0
    for user in User.objects.filter(pk__in=list(PublicationAlert.objects.filter(emailed=False)
                                                 .values_list('user_id', flat=True).distinct())).exclude(pk__in=digest_bundle.bundled_users()):
        if sent >= budget:
            break
        alerts = list(PublicationAlert.objects.filter(user=user, emailed=False))
        if not user.email or user.email.strip().lower() in suppressed or not access.has_tool(user, 'citations'):
            PublicationAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True)
            continue
        lang = user.digests.values_list('lang', flat=True).first() or 'en'
        with translation.override(lang):
            items = items_of(alerts, base_url, lang)
            context = {'items': items, 'count': len(items), 'base_url': base_url,
                       'manage_url': '%s/%s/citations/' % (base_url, lang)}
            subject = render_to_string('publications/subject.txt', context).strip()
            text = render_to_string('publications/email.txt', context)
            html = render_to_string('publications/email.html', context)
        message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
        message.attach_alternative(html, 'text/html')
        try:
            message.send()
        except Exception:
            logger.exception('Publication alert to %s failed', user.username)
            continue
        PublicationAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True)
        sent += 1
    return sent


def run(now, budget):
    """From the hourly digest run: check once a day, then send within the email budget."""
    if cache.add(DAILY_KEY % now.date(), True, 86400):
        check(now)
    return send_alerts(now, budget) if budget > 0 else 0
