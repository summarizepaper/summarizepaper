"""Emails to the authors of the papers a weekly letter picked, to let them know the site exists: one email per
paper, to the first address its arXiv HTML gives, written here (no language model) and sent only once a staff
member approved it on /admin/prospection/.

Who is written to: the papers of the English letters of the last days (letters.py), except the astronomy theme and
any paper listed in astro-ph (EXCLUDED_ARCHIVES), papers without an arXiv HTML rendering (bioRxiv, medRxiv, PDF
only), addresses that already have an account, that bounced or asked for nothing more (SuppressedEmail), or that
were already written to. Each address hears from us once; the email says where its address was found and has a
link that stops any other.

The hourly run prepares at most PREPARE_PER_RUN papers (one arXiv request each) and sends at most PER_DAY approved
emails a day, from settings.OUTREACH_FROM_EMAIL. Links carry utm_source=outreach-author (conversions page).
"""
import datetime
import json
import logging
import re

import requests
from bs4 import BeautifulSoup
from django.conf import settings
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.auth.models import User
from django.core import signing
from django.core.mail import EmailMultiAlternatives
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.html import escape, urlize
from django.views.decorators.http import require_POST

from . import letters, mailcheck, social
from .models import AnnouncedPaper, ArxivPaper, Letter, Prospect

logger = logging.getLogger(__name__)

SITE = 'https://www.summarizepaper.com'
PER_DAY = 20
PREPARE_PER_RUN = 10
LETTER_AGE = datetime.timedelta(days=8)
PENDING_DAYS = 14  # an email not approved within this many days is about an old letter: it expires
EXCLUDED_TOPICS = {'space'}
EXCLUDED_ARCHIVES = ('astro-ph',)
OPTOUT_SALT = 'summarizer.outreach-optout'
UTM = 'utm_source=outreach-author&utm_medium=email&utm_campaign=letter-%s'
EMAIL = re.compile(r'[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}')
GENERIC = re.compile(r'^(?:info|contact|admin|office|secretariat|noreply|no-reply|webmaster|help|support)@', re.I)


# --- who ------------------------------------------------------------------------------------------------------

def excluded_paper(arxiv_id):
    """Astronomy is left out, even cross-listed."""
    base = re.sub(r'v\d+$', '', arxiv_id)
    categories = set(filter(None, (AnnouncedPaper.objects.filter(arxiv_id__in=[arxiv_id, base])
                                   .values_list('categories', flat=True).first() or '').split(',')))
    primary = ArxivPaper.objects.filter(arxiv_id=arxiv_id).values_list('category', flat=True).first()
    if primary:
        categories.add(primary)
    return any(c.strip().startswith(EXCLUDED_ARCHIVES) for c in categories)


def author_emails(html):
    """The addresses in the authors block of an arXiv HTML page, in order."""
    soup = BeautifulSoup(html, 'html.parser')
    block = soup.select_one('.ltx_authors')
    if block is None:
        return []
    found = [a['href'][7:].split('?')[0] for a in block.select('a[href^="mailto:"]')]
    found += EMAIL.findall(block.get_text(' '))
    clean = []
    for email in found:
        email = email.strip().strip('.').lower()
        if email and email not in clean and not GENERIC.match(email):
            clean.append(email)
    return clean


def fetch_emails(arxiv_id):
    from .paper_source import ARXIV_HTML, arxiv_ref
    from .sources import USER_AGENT
    try:
        response = requests.get(ARXIV_HTML % arxiv_ref(arxiv_id), timeout=60, headers={'User-Agent': USER_AGENT})
    except requests.RequestException:
        logger.exception('arXiv HTML fetch failed for %s', arxiv_id)
        return None  # try again later
    if response.status_code != 200 or '/html/' not in response.url:
        return []
    return author_emails(response.text)


def name_for(email, names):
    """The author whose last name is in the address, else ''."""
    local = re.sub(r'[^a-z]', '', email.split('@')[0].lower())
    for name in names:
        last = re.sub(r'[^a-z]', '', (name.split()[-1] if name.split() else '').lower())
        if len(last) >= 3 and last in local:
            return name
    return ''


def known_addresses():
    known = {e.strip().lower() for e in User.objects.values_list('email', flat=True) if e}
    known |= {e.lower() for e in Prospect.objects.exclude(email='').values_list('email', flat=True)}
    return known | mailcheck.suppressed()


# --- what ------------------------------------------------------------------------------------------------------

def tagged(url, topic):
    return '%s%s%s' % (url, '&' if '?' in url else '?', UTM % topic)


def write(paper, letter, name):
    """(subject, body) of the email about one paper of one letter."""
    _theme, letter_name = letters.localized_name(letter.topic, 'en')
    title = social.plain(paper.title)
    # One link only, to the paper's own page and its free citation alert: the first emails also linked the letter,
    # and most readers stopped there (2026-11-18: 22 visitors from 29 emails, no sign-up).
    paper_url = tagged('%s/en/paper/%s/' % (SITE, paper.arxiv_id), letter.topic) + '#own-paper'
    greeting = 'Dear %s,' % name if name else 'Hello,'
    subject = 'Your paper “%s” in “%s”' % (short_title(title), letter_name)
    body = '\n\n'.join([
        greeting,
        '“%s” is one of the six papers we picked this week for “%s”, our free weekly letter, and it now has a '
        'plain-language summary on SummarizePaper.com.' % (title, letter_name),
        'We can also email you each time a new paper cites it, with the sentence where it does. It is free for your '
        'own paper and takes one click with Google:\n%s' % paper_url,
        'Best regards,\n%s' % settings.OUTREACH_SIGNATURE,
    ])
    return subject, body


def short_title(title, limit=60):
    if len(title) <= limit:
        return title
    return title[:limit].rsplit(' ', 1)[0].rstrip(' ,:;-') + '…'


def footer(prospect):
    url = '%s/en/outreach/stop/%s/' % (SITE, signing.dumps(prospect.email, salt=OPTOUT_SALT))
    return ('We found your address on the arXiv version of your paper. This is the only email we send you unless '
            'you create an account. Never email me: %s' % url), url


# --- the hourly steps -----------------------------------------------------------------------------------------

def candidates(today):
    """(paper id, letter) of the recent English letters, outside the excluded themes, not considered yet."""
    from .presskit import paper_id
    seen = set(Prospect.objects.exclude(arxiv_id='').values_list('arxiv_id', flat=True))
    for letter in (Letter.objects.filter(lang='en', day__gte=today - LETTER_AGE)
                   .exclude(topic__in=EXCLUDED_TOPICS).order_by('-day', 'topic')):
        content = json.loads(letter.content)
        for raw in content.get('order') or list(content.get('papers', {})):
            arxiv_id = paper_id(raw) or raw
            if arxiv_id not in seen:
                seen.add(arxiv_id)
                yield arxiv_id, letter


def prepare(now, limit=PREPARE_PER_RUN, fetch=fetch_emails):
    """Write the emails of new letter papers into the queue (pending). The number of papers looked at."""
    from . import sources
    Prospect.objects.filter(status='pending', created__lt=now - datetime.timedelta(days=PENDING_DAYS)).update(status='expired')
    known = known_addresses()
    looked = 0
    for arxiv_id, letter in candidates(now.date()):
        if looked >= limit:
            break
        common = dict(kind='author', arxiv_id=arxiv_id, topic=letter.topic, letter_day=letter.day)
        paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
        if paper is None:
            continue  # its page is not written yet (the letter came first): a later run (2026-11-16 lost 37 papers)
        if sources.is_external(arxiv_id) or excluded_paper(arxiv_id):
            Prospect.objects.create(status='rejected', **common)
            continue
        looked += 1
        emails = fetch(arxiv_id)
        if emails is None:
            break  # arXiv did not answer: next run
        fresh = [e for e in emails if e not in known and not mailcheck.undeliverable_reason(e, frozenset())]
        if not fresh:
            Prospect.objects.create(status='no_email', **common)
            continue
        names = [a.name for a in paper.authors.all()]
        name = name_for(fresh[0], names)
        subject, body = write(paper, letter, name)
        Prospect.objects.create(status='pending', email=fresh[0], name=name, subject=subject, body=body, **common)
        known.add(fresh[0])
    return looked


def send_one(prospect):
    text_footer, stop_url = footer(prospect)
    text = '%s\n\n--\n%s\n' % (prospect.body, text_footer)
    html = '<div style="font-family:sans-serif;font-size:15px;line-height:1.5">%s<p style="color:#666;font-size:12px">%s</p></div>' % (
        ''.join('<p>%s</p>' % urlize(escape(p)).replace('\n', '<br>') for p in prospect.body.split('\n\n')),
        urlize(escape(text_footer)))
    message = EmailMultiAlternatives(prospect.subject, text, settings.OUTREACH_FROM_EMAIL, [prospect.email],
                                     reply_to=[settings.OUTREACH_REPLY_TO],
                                     headers={'List-Unsubscribe': '<%s>' % stop_url})
    message.attach_alternative(html, 'text/html')
    message.send()


def send_approved(now, budget):
    """Send the approved emails, at most PER_DAY a day and `budget` in this run. The number sent."""
    sent_today = Prospect.objects.filter(status='sent', sent__date=now.date()).count()
    room = min(budget, PER_DAY - sent_today)
    suppressed = mailcheck.suppressed()
    done = 0
    for prospect in Prospect.objects.filter(status='approved').order_by('decided')[:max(room, 0)]:
        if prospect.email in suppressed or User.objects.filter(email__iexact=prospect.email).exists():
            prospect.status = 'rejected'
            prospect.save(update_fields=['status'])
            continue
        try:
            send_one(prospect)
        except Exception:
            logger.exception('Outreach email %s failed', prospect.pk)
            prospect.status = 'failed'
            prospect.save(update_fields=['status'])
            continue
        prospect.status, prospect.sent = 'sent', now
        prospect.save(update_fields=['status', 'sent'])
        done += 1
    return done


def run(now, budget):
    prepare(now)
    return send_approved(now, budget)


# --- pages ----------------------------------------------------------------------------------------------------

def stop(request, token):
    """The link at the bottom of the email: the address is never written to again."""
    try:
        email = signing.loads(token, salt=OPTOUT_SALT)
    except signing.BadSignature:
        raise Http404
    mailcheck.suppress(email, 'outreach opt-out')
    return render(request, 'conversion/optout.html')


@staff_member_required
def queue_view(request):
    """/admin/prospection/: the emails written, to approve or set aside; then what was sent."""
    from django.contrib import admin
    if request.method == 'POST':
        ids = [int(i) for i in request.POST.getlist('ids') if i.isdigit()]
        action = request.POST.get('action')
        rows = Prospect.objects.filter(pk__in=ids, status='pending')
        if action == 'approve':
            rows.update(status='approved', decided=timezone.now())
        elif action == 'reject':
            rows.update(status='rejected', decided=timezone.now())
        return redirect(request.path)
    counts = {key: Prospect.objects.filter(status=key).count() for key, _label in Prospect.STATUSES}
    context = dict(admin.site.each_context(request), title='Prospection',
                   pending=Prospect.objects.filter(status='pending').order_by('created'),
                   approved=Prospect.objects.filter(status='approved').order_by('decided'),
                   sent=Prospect.objects.filter(status='sent').order_by('-sent')[:50],
                   counts=counts, labels=dict(Prospect.STATUSES), per_day=PER_DAY,
                   sender=settings.OUTREACH_FROM_EMAIL)
    return render(request, 'admin/summarizer/prospection.html', context)
