from django.conf import settings
from django.contrib.auth.models import User
from django.core.management.base import BaseCommand

from summarizer import reactivation


class Command(BaseCommand):
    help = 'Email up to --limit never-activated accounts an activation link (once per address); meant to run hourly.'

    def add_arguments(self, parser):
        parser.add_argument('--limit', type=int, default=10)
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--preview-to', help='send the email of the next candidate to this address instead, without recording it')
        parser.add_argument('--base-url', default=settings.SITE_URL or 'https://www.summarizepaper.com')

    def handle(self, limit, dry_run, preview_to, base_url, **options):
        if preview_to:
            chosen = reactivation.candidates()
            reactivation.send_one(chosen[0], base_url, to=preview_to)
            self.stdout.write('preview of %s sent to %s' % (chosen[0].username, preview_to))
            return
        sent = reactivation.run(base_url, limit, dry_run=dry_run, stdout=self.stdout)
        self.stdout.write('%d email(s) sent' % sent)
