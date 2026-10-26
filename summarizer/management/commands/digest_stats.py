"""How good the digest ranking is: clicks by rank, votes, unsubscriptions.

python manage.py digest_stats --days 30
"""
import datetime
import json

from django.core.management.base import BaseCommand
from django.db.models import Count
from django.utils import timezone

from summarizer.models import DigestClick, DigestEdition, DigestFeedback, DigestSubscription

RANKS = 10


class Command(BaseCommand):
    help = 'Click-through rate by rank in the digest emails, votes and unsubscriptions over the last days.'

    def add_arguments(self, parser):
        parser.add_argument('--days', type=int, default=30)

    def handle(self, *args, **options):
        since = timezone.now() - datetime.timedelta(days=options['days'])
        editions = DigestEdition.objects.filter(sent__gte=since)
        # how many emails had a paper at each rank
        shown = [0] * (RANKS + 1)
        for papers in editions.values_list('papers', flat=True):
            for rank in range(1, min(len(json.loads(papers or '[]')), RANKS) + 1):
                shown[rank] += 1
        clicks = DigestClick.objects.filter(created__gte=since, burst=False)
        # one click per email and paper: a reader opening the summary then the PDF counts once
        opened = dict(clicks.filter(rank__gte=1, rank__lte=RANKS).values('rank')
                      .annotate(n=Count('edition_id', distinct=True)).values_list('rank', 'n'))
        out = self.stdout.write
        out('Digest emails: %d, clicks: %d (%d from mail scanners, left out)' % (
            editions.count(), clicks.count(), DigestClick.objects.filter(created__gte=since, burst=True).count()))
        out('\nRank  emails  opened  rate')
        for rank in range(1, RANKS + 1):
            rate = opened.get(rank, 0) / shown[rank] if shown[rank] else 0
            out('%4d  %6d  %6d  %5.1f%%' % (rank, shown[rank], opened.get(rank, 0), 100 * rate))
        targets = dict(clicks.values('target').annotate(n=Count('id')).values_list('target', 'n'))
        out('\nTargets: %s' % ', '.join('%s %d' % item for item in sorted(targets.items())))
        votes = dict(DigestFeedback.objects.filter(created__gte=since, source='vote').values('value')
                     .annotate(n=Count('id')).values_list('value', 'n'))
        out('Votes: %d more like this, %d less like this' % (votes.get(1, 0), votes.get(-1, 0)))
        stopped = DigestSubscription.objects.filter(active=False, updated__gte=since)
        out('Digests stopped: %d (%d with a research profile), of %d active' % (
            stopped.count(), stopped.exclude(profile='').count(), DigestSubscription.objects.filter(active=True).count()))
        by_level = dict(DigestSubscription.objects.filter(active=True).values('level').annotate(n=Count('id'))
                        .values_list('level', 'n'))
        out('Active digests by summary level: %s' % ', '.join('%s %d' % item for item in sorted(by_level.items())))
