"""What the rising topics find in the stored weeks, to check their thresholds against what one knows of a field.

python manage.py trends_calibrate                               # the last counted week, all of arXiv and the largest scopes
python manage.py trends_calibrate --scope cs.CL --scope astro-ph.EP --week 2026-10-12
python manage.py trends_calibrate --llr 20 --growth 1.5         # what looser thresholds would let through

For each scope it prints the papers of the week, the rising terms with their count, expected count, growth,
log-likelihood ratio, teams and kind of rise, then the terms left out because a single team signs them.
It reads only stored counts and papers: no model call, nothing stored.
"""
import datetime

from django.core.management.base import BaseCommand

from summarizer import trends
from summarizer.models import TermWeek


class Command(BaseCommand):
    help = 'Print the rising terms of stored weeks with their statistics, to check the thresholds.'

    def add_arguments(self, parser):
        parser.add_argument('--scope', action='append')
        parser.add_argument('--week', help='Its Monday; the last counted week otherwise.')
        parser.add_argument('--llr', type=float, default=trends.MIN_LLR)
        parser.add_argument('--growth', type=float, default=trends.MIN_GROWTH)
        parser.add_argument('--top', type=int, default=6, help='Scopes when none is given: all of arXiv and the largest.')

    def handle(self, *args, **options):
        week = datetime.date.fromisoformat(options['week']) if options['week'] else \
            TermWeek.objects.filter(scope='*').order_by('-week').values_list('week', flat=True).first()
        if week is None:
            self.stdout.write('No week counted yet: run trends_backfill.')
            return
        scopes = options['scope'] or ['*'] + [s for s, _ in TermWeek.objects.filter(week=week, part=0).exclude(scope='*')
                                              .order_by('-papers').values_list('scope', 'papers')[:options['top'] - 1]]
        trends.MIN_LLR, trends.MIN_GROWTH = options['llr'], options['growth']
        for scope in scopes:
            found = trends.rising_terms(scope, week, verbose=True)
            papers = TermWeek.objects.filter(scope=scope, week=week, part=0).values_list('papers', flat=True).first()
            self.stdout.write(self.style.MIGRATE_HEADING('\n== %s, week of %s: %s papers' % (scope, week, papers)))
            if found is None:
                self.stdout.write('Not enough history or papers.')
                continue
            rising, left = found
            history = trends.history(scope, [t['term'] for t in rising], weeks=[week - trends.WEEK * i for i in range(60)][::-1])
            for item in rising:
                rise = trends.kind(history[item['term']])
                self.stdout.write('%-40s %4d  exp %6.1f  x%-5.1f llr %6.1f  teams %-4s %s%s' % (
                    item['term'][:40], item['n'], item['expected'], item['growth'], item['llr'], item.get('teams', '?'),
                    rise['kind'], ' (%d weeks)' % rise['weeks'] if rise['kind'] == 'growing' else ''))
            for item in left:
                self.stdout.write(self.style.WARNING('  left out: %-30s %4d  x%-5.1f %s' % (
                    item['term'][:30], item['n'], item['growth'], item['reason'])))
