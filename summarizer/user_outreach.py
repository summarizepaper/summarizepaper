"""Small, explicit feedback batches. No scheduled sending, SMTP mailbox probing or account deletion."""
import datetime
import smtplib
from contextlib import contextmanager
from email.utils import parseaddr

from django.conf import settings
from django.contrib.auth.models import User
from django.core import signing
from django.core.cache import cache
from django.core.mail import EmailMessage
from django.db import connection, IntegrityError
from django.db.models import Max
from django.http import Http404
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone, translation
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from . import mailcheck, reactivation
from .models import AIassistant, AnnouncementSent, OutreachContact, PaperHistory, Search

BASE = 'https://www.summarizepaper.com'
SALT = 'summarizer.feedback-optout.v1'
MAX_BATCH = 6
COOLDOWN = datetime.timedelta(hours=24)


class BatchStopped(Exception):
    pass


def candidates(segment='recent', now=None):
    if segment not in ('recent', 'older'): raise BatchStopped('Segment inconnu.')
    now = now or timezone.now()
    contacted = {canonical(e) for e in OutreachContact.objects.values_list('email', flat=True)}
    suppressed = mailcheck.suppressed()
    suppressed_keys = {canonical(e) for e in suppressed}
    staff_mail = {canonical(e) for e in User.objects.filter(is_staff=True).values_list('email', flat=True)}
    announced = set(AnnouncementSent.objects.values_list('user_id', flat=True))
    activity = {}
    for model in (PaperHistory, AIassistant, Search):
        for row in model.objects.exclude(user=None).values('user_id').annotate(latest=Max('created')):
            uid, date = row['user_id'], row['latest']
            if date and (uid not in activity or date > activity[uid]): activity[uid] = date
    chosen, seen = [], set()
    users = User.objects.filter(is_active=True, is_staff=False).exclude(email='').order_by('-last_login', '-date_joined')
    for user in users:
        email = user.email.strip().lower()
        key = canonical(email)
        if key in seen or key in staff_mail or key in contacted or user.pk in announced:
            continue
        if key in suppressed_keys or reactivation.looks_random(user.username, email):
            continue
        # DNS is checked only for the small batch at send time, never for the whole list.
        if mailcheck.undeliverable_reason(email, suppressed, check_dns=False): continue
        dates = [d for d in (user.last_login, activity.get(user.pk)) if d]
        latest = max(dates) if dates else None
        recent = latest and latest >= now - datetime.timedelta(days=90)
        if not latest or (segment == 'recent' and not recent) or (segment == 'older' and recent): continue
        seen.add(key)
        user.outreach_activity = latest
        chosen.append(user)
    return sorted(chosen, key=lambda u:u.outreach_activity, reverse=True)


def canonical(email):
    local, _, domain = (email or '').strip().lower().partition('@')
    if domain in ('gmail.com', 'googlemail.com'):
        return local.split('+')[0].replace('.', '') + '@gmail.com'
    return (email or '').strip().lower()


def language(user):
    latest = []
    for model in (PaperHistory, AIassistant, Search):
        row = model.objects.filter(user=user).order_by('-created').values('lang','created').first()
        if row: latest.append(row)
    if latest:
        return 'fr' if max(latest,key=lambda r:r['created'])['lang'] == 'fr' else 'en'
    return 'fr' if user.digests.order_by('-created').values_list('lang',flat=True).first() == 'fr' else 'en'


def message(lang, unsubscribe_url):
    example = BASE + '/' + lang + '/paper/doi_10.1038--s41550-026-03004-3/'
    if lang == 'fr':
        return ('SummarizePaper a évolué — votre avis ?',
            'Bonjour,\n\nVous avez un compte sur SummarizePaper. Nous avons récemment amélioré les résumés, la discussion avec les articles et les outils de veille scientifique.\n\n'
            'Voici un exemple de résumé fondé sur le texte intégral, lorsque la licence le permet :\n' + example + '\n\n'
            'Que devrions-nous améliorer en priorité pour vous être utile dans votre travail ? Vous pouvez simplement répondre à cet email.\n\n'
            'Merci pour votre retour,\nQuentin — SummarizePaper\n\nNe plus recevoir ces messages :\n' + unsubscribe_url + '\n')
    return ('SummarizePaper has changed — what do you think?',
        'Hello,\n\nYou have an account on SummarizePaper. We have recently improved paper summaries, chat with papers and research monitoring tools.\n\n'
        'Here is an example of a summary based on the full text, when the licence permits it:\n' + example + '\n\n'
        'What should we improve first to make SummarizePaper useful for your work? You can simply reply to this email.\n\n'
        'Thank you for your feedback,\nQuentin — SummarizePaper\n\nStop receiving these messages:\n' + unsubscribe_url + '\n')


@contextmanager
def batch_lock():
    key = 67203195
    if connection.vendor == 'postgresql':
        with connection.cursor() as cursor:
            cursor.execute('SELECT pg_try_advisory_lock(%s)', [key])
            acquired = cursor.fetchone()[0]
        if not acquired: raise BatchStopped('Un autre lot est en cours.')
        try: yield
        finally:
            with connection.cursor() as cursor: cursor.execute('SELECT pg_advisory_unlock(%s)', [key])
    else:
        if not cache.add('outreach-batch-lock', True, 600): raise BatchStopped('Un autre lot est en cours.')
        try: yield
        finally: cache.delete('outreach-batch-lock')


def send_batch(limit=MAX_BATCH, segment='recent'):
    if not 1 <= limit <= MAX_BATCH: raise BatchStopped('Un lot contient de 1 à 6 messages.')
    with batch_lock():
        latest = OutreachContact.objects.order_by('-created').first()
        if latest and latest.created > timezone.now()-COOLDOWN:
            raise BatchStopped('Attendre au moins 24 heures et vérifier les retours du prestataire avant le lot suivant.')
        if segment == 'older' and latest is None:
            raise BatchStopped('Commencer par un lot récent avant de contacter les anciens comptes.')
        sent = []
        for user in candidates(segment)[:limit]:
            email = user.email.strip().lower()
            reason = mailcheck.undeliverable_reason(email)
            if reason: continue
            try:
                row = OutreachContact.objects.create(user=user, email=email, lang=language(user))
            except IntegrityError: continue
            with translation.override(row.lang):
                url = BASE + reverse('outreach_unsubscribe', args=[signing.dumps(row.pk, salt=SALT)])
            row.subject, row.body = message(row.lang, url)
            row.save(update_fields=['subject','body'])
            mail = EmailMessage(row.subject, row.body, settings.DEFAULT_FROM_EMAIL, [email],
                reply_to=[parseaddr(settings.DEFAULT_FROM_EMAIL)[1]],
                headers={'List-Unsubscribe':'<%s>' % url, 'List-Unsubscribe-Post':'List-Unsubscribe=One-Click'})
            try:
                if mail.send() != 1: raise RuntimeError('SMTP did not accept the message')
            except Exception as exc:
                row.status, row.error = 'failed', str(exc)[:1000]
                if isinstance(exc, smtplib.SMTPRecipientsRefused):
                    detail = exc.recipients.get(email, (0,b''))[1]
                    detail = detail.decode(errors='replace') if isinstance(detail,bytes) else str(detail)
                    if '5.1.1' in detail:
                        row.status = 'bounced'
                        mailcheck.suppress(email, 'outreach: SMTP 5.1.1 unknown mailbox')
                row.save(update_fields=['status','error'])
                raise BatchStopped('Lot arrêté après une erreur SMTP. Vérifier les retours avant toute reprise.') from exc
            OutreachContact.objects.filter(pk=row.pk,status='sending').update(status='sent',sent=timezone.now())
            sent.append(row)
        return sent


@csrf_exempt
@require_http_methods(['GET','POST'])
def unsubscribe(request, token):
    try:
        pk = signing.loads(token, salt=SALT)
        row = OutreachContact.objects.get(pk=pk)
    except (signing.BadSignature, OutreachContact.DoesNotExist, ValueError, TypeError): raise Http404
    if request.method == 'POST':
        row.status = 'unsubscribed'
        row.save(update_fields=['status'])
    response = render(request, 'summarizer/outreach_unsubscribe.html', {'stopped':row.status=='unsubscribed'})
    response['Cache-Control'] = 'private, no-store'
    response['X-Robots-Tag'] = 'noindex, nofollow'
    return response
