import logging

from django.core.management.base import BaseCommand
from django.utils import timezone

from summarizer import audio, citations, digest, letters, trends

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = ('Email the digests that are due at this hour (run every hour), then the citation alerts and letters '
            'within what is left of the hourly email budget. --user sends now, ignoring the schedule.')

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Select papers and write blurbs, send nothing.')
        parser.add_argument('--user', help='Only this username.')
        parser.add_argument('--base-url', default='https://www.summarizepaper.com')

    def handle(self, *args, **options):
        sent = digest.run(base_url=options['base_url'], dry_run=options['dry_run'],
                          only_user=options['user'], stdout=self.stdout)
        self.stdout.write(self.style.SUCCESS('%d digest(s) sent' % sent))
        if options['user'] or options['dry_run']:
            return
        digest.cleanup()
        audio.cleanup()
        now = timezone.now()
        budget = digest.MAX_EMAILS_PER_RUN - sent
        # each step is independent: one failing must not stop the others
        for name, step in (('trends', lambda: trends.update(now)),
                           ('letters', lambda: letters.update(now)),
                           ('citation alerts', lambda: citations.run(now, budget)),
                           ('letter emails', lambda: letters.send(now, budget))):
            try:
                done = step()
            except Exception:
                logger.exception('Hourly step %s failed', name)
                continue
            if name in ('citation alerts', 'letter emails'):
                budget -= done
            self.stdout.write('%s: %s' % (name, done))
