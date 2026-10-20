from django.conf import settings
from django.contrib.auth.models import User
from django.core.mail import EmailMultiAlternatives
from django.core.management.base import BaseCommand
from django.template.loader import render_to_string
from django.utils import translation

from summarizer import access, digest, mailcheck
from summarizer.models import AnnouncementSent

NAME = 'tools-2026-10'


class Command(BaseCommand):
    help = ('Email the Premium subscribers about the new tools for researchers, once each. Without --send, only '
            'says who would get it. --limit keeps each run within the hourly email budget: run it again until done.')

    def add_arguments(self, parser):
        parser.add_argument('--send', action='store_true', help='Really send the emails.')
        parser.add_argument('--limit', type=int, default=10)
        parser.add_argument('--base-url', default=digest.BASE_URL)

    def handle(self, *args, **options):
        suppressed = mailcheck.suppressed()
        done = set(AnnouncementSent.objects.filter(name=NAME).values_list('user_id', flat=True))
        waiting = [u for u in User.objects.exclude(email='').exclude(pk__in=done).order_by('pk')
                   if u.email.strip().lower() not in suppressed and access.is_paid(u)]
        self.stdout.write('%d Premium subscriber(s) have not had it yet' % len(waiting))
        if not options['send']:
            return
        sent = 0
        for user in waiting[:options['limit']]:
            lang = user.digests.values_list('lang', flat=True).first() or 'en'
            context = {'base_url': options['base_url'], 'lang': lang}
            with translation.override(lang):
                subject = render_to_string('announce/tools_subject.txt', context).strip()
                text = render_to_string('announce/tools_email.txt', context)
                html = render_to_string('announce/tools_email.html', context)
            message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
            message.attach_alternative(html, 'text/html')
            try:
                message.send()
            except Exception as e:
                self.stderr.write('%s: %r' % (user.username, e))
                continue
            AnnouncementSent.objects.create(user=user, name=NAME)
            sent += 1
        self.stdout.write(self.style.SUCCESS('%d sent' % sent))
