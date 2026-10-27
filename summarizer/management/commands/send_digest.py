import logging

from django.core.management.base import BaseCommand
from django.utils import timezone

from summarizer import audio, changes, citations, digest, letters, publication_alerts, scoops, students, teams, trends, usage, saved_searches

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = ('Email the digests that are due at this hour (run every hour), then the scoop, citation, version, publication, saved-search and topic map alerts and letters '
            'within what is left of the hourly email budget. --user sends now, ignoring the schedule.')

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Select papers and write blurbs, send nothing.')
        parser.add_argument('--user', help='Only this username.')
        parser.add_argument('--base-url', default='https://www.summarizepaper.com')

    def handle(self, *args, **options):
        with usage.task('digest'):
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
        for name, step in (('letters', lambda: letters.update(now)),
                           ('letter summaries', lambda: letters.presummarize_due()),
                           ('scoop alerts', lambda: scoops.run(now, budget)),
                           ('citation alerts', lambda: citations.run(now, budget)),
                           ('version alerts', lambda: changes.run(now, budget)),
                           ('letter emails', lambda: letters.send(now, budget)),
                           ('search watches', lambda: saved_searches.run(now, budget, base_url=options['base_url'])),
                           ('publication alerts', lambda: publication_alerts.run(now, budget)),
                           ('placement reminders', lambda: students.send_reminders(now, budget, base_url=options['base_url'])),
                           ('placement new papers', lambda: students.watch_new_papers(now, budget, base_url=options['base_url'])),
                           ('topic maps', lambda: teams.watch_maps(now, budget, base_url=options['base_url'])),
                           # last: once a week it reads two weeks of arXiv and writes topics for a few minutes
                           ('trends', lambda: trends.update(now))):
            try:
                with usage.task(name.replace(' ', '_')):
                    done = step()
            except Exception:
                logger.exception('Hourly step %s failed', name)
                continue
            if name in ('scoop alerts', 'citation alerts', 'version alerts', 'letter emails', 'search watches', 'publication alerts',
                        'placement reminders', 'placement new papers', 'topic maps'):
                budget -= done
            self.stdout.write('%s: %s' % (name, done))
