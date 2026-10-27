"""Count the weeks of submissions of the last months again, so the rising topics have their history.

python manage.py trends_backfill --weeks 80                # arXiv (OAI-PMH) and bioRxiv/medRxiv, then count
python manage.py trends_backfill --weeks 80 --source bio   # only bioRxiv and medRxiv (slow: run it on its own dyno)
python manage.py trends_backfill --weeks 80 --store /tmp/t.sqlite3 --no-harvest   # count what was read already

arXiv's OAI-PMH lists every record changed since the first week with the date of its version 1: the papers of each
week are kept in a local SQLite file (about 1 GB for 80 weeks), then every week is counted and stored (TermWeek)
and the papers of the last weeks are kept for the examples (TrendPaper). A run reads a few hundred OAI pages
(about two hours) and asks nothing of the model; it can be run again, each week is replaced.
"""
import datetime
import json
import os
import sqlite3
import tempfile

from django.core.cache import cache
from django.core.management.base import BaseCommand
from django.utils import timezone

from summarizer import trend_sources, trends


class Command(BaseCommand):
    help = 'Read and count the weeks of submissions of the last months for the rising topics.'

    def add_arguments(self, parser):
        parser.add_argument('--weeks', type=int, default=80)
        parser.add_argument('--store', default=os.path.join(tempfile.gettempdir(), 'trends-backfill.sqlite3'))
        parser.add_argument('--no-harvest', action='store_true', help='Count the papers already in --store.')
        parser.add_argument('--source', choices=['arxiv', 'bio'], help='Read and count only arXiv, or only bioRxiv and medRxiv.')

    def handle(self, *args, **options):
        cache.set(trends.LOCK, True, 24 * 3600)  # the hourly run leaves the weeks alone meanwhile
        try:
            self.backfill(options)
        finally:
            cache.delete(trends.LOCK)

    def backfill(self, options):
        today = timezone.now().date()
        last = trends.closed_week(today)
        first = last - trends.WEEK * (options['weeks'] - 1)
        db = sqlite3.connect(options['store'])
        db.execute('CREATE TABLE IF NOT EXISTS papers (key TEXT PRIMARY KEY, week TEXT, body TEXT)')
        db.execute('CREATE INDEX IF NOT EXISTS papers_week ON papers (week)')

        def keep(papers):
            db.executemany('INSERT OR REPLACE INTO papers VALUES (?, ?, ?)', [
                (p['key'], trends.week_of(p['day']).isoformat(), json.dumps(dict(p, day=p['day'].isoformat()))) for p in papers])
            db.commit()

        if not options['no_harvest']:
            batch, read = [], 0
            for paper in trend_sources.arxiv_papers(first) if options['source'] != 'bio' else []:
                batch.append(paper)
                read += 1
                if len(batch) == 2000:
                    keep(batch)
                    batch = []
                    self.stdout.write('arXiv: %d papers, now at %s' % (read, paper['day']))
            keep(batch)
            if options['source'] != 'arxiv':
                week = first
                while week <= last:
                    keep(list(trend_sources.bio_papers(week, week + datetime.timedelta(days=6))))
                    self.stdout.write('bioRxiv and medRxiv: week of %s' % week)
                    week += trends.WEEK
        week = first
        while week <= last:
            papers = [json.loads(body) for (body,) in db.execute('SELECT body FROM papers WHERE week = ?', (week.isoformat(),))]
            for paper in papers:
                paper['day'] = datetime.date.fromisoformat(paper['day'])
            rows = trends.count_week(week, papers, options['source'])
            if week > last - trends.WEEK * trends.KEEP_PAPERS:
                trends.keep_papers(papers)
            self.stdout.write('Week of %s: %d papers, %d rows' % (week, len(papers), rows))
            week += trends.WEEK
        db.close()
