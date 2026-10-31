from django.core.management.base import BaseCommand
from django.utils import timezone

from summarizer import neighbours


class Command(BaseCommand):
    help = ('Compute the neighbourhoods of the summarized papers that have none, then of the oldest ones, '
            'for at most --minutes (the hourly run does the same for a few minutes); or of the given papers.')

    def add_arguments(self, parser):
        parser.add_argument('arxiv_ids', nargs='*')
        parser.add_argument('--minutes', type=float, default=30)

    def handle(self, arxiv_ids, minutes, **options):
        if arxiv_ids:
            for arxiv_id in arxiv_ids:
                data = neighbours.store(arxiv_id)
                self.stdout.write('%s: %s' % (arxiv_id, data['counts'] if data else 'unknown paper'))
            return
        done = neighbours.refresh_due(timezone.now(), seconds=60 * minutes)
        self.stdout.write('%d neighbourhoods computed' % done)
