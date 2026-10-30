"""Rebuild one stored scope/week without harvesting or changing its statistical counts."""
import datetime

from django.core.cache import cache
from django.core.management.base import BaseCommand, CommandError

from summarizer import trends, usage
from summarizer.models import TermWeek


class Command(BaseCommand):
    help = 'Rebuild one week of rising topics from its existing counts and papers.'

    def add_arguments(self, parser):
        parser.add_argument('--scope', default='*')
        parser.add_argument('--week', required=True, help='Monday, YYYY-MM-DD')

    def handle(self, *args, **options):
        try:
            week = datetime.date.fromisoformat(options['week'])
        except ValueError:
            raise CommandError('Invalid date; use YYYY-MM-DD.')
        if trends.week_of(week) != week:
            raise CommandError('Use the Monday of the submission week.')
        scope = options['scope']
        if not TermWeek.objects.filter(scope=scope, week=week).exists():
            raise CommandError('No stored counts for this scope/week.')
        if not cache.add(trends.LOCK, True, 3 * 3600):
            raise CommandError('Another trends calculation is running; retry later.')
        try:
            with usage.task('trends'):
                trends.write(scope, week)
            result = trends.result_state(scope, 'en', week)
            self.stdout.write('scope=%s week=%s status=%s candidates=%s topics=%d attempts=%s' % (
                scope, week, result.get('status'), result.get('candidate_count'),
                len(result.get('topics', [])), result.get('attempts')))
        finally:
            cache.delete(trends.LOCK)
