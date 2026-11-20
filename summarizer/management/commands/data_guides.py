"""Write the statistics guides of data_guides.py.

    manage.py data_guides topics [--dry-run DIR]      # production: TermWeek -> Guide rows (also monthly in send_digest)
    manage.py data_guides harvest --file F [--since 2022-01-01]   # laptop: arXiv OAI-PMH metadata, resumable
    manage.py data_guides metadata --file F [--year 2025]         # laptop: authors and journals guides -> content/guides
"""
import os

from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from summarizer import arxiv_metadata, data_guides


class Command(BaseCommand):
    help = 'Write the statistics guides (fastest-growing topics, authors per paper, arXiv papers in journals).'

    def add_arguments(self, parser):
        parser.add_argument('what', choices=['topics', 'harvest', 'metadata'])
        parser.add_argument('--file', help='the harvested metadata (.jsonl.gz)')
        parser.add_argument('--since', default='2022-01-01')
        parser.add_argument('--year', type=int, help='the last complete year (default: last year)')
        parser.add_argument('--dry-run', metavar='DIR', help='write the Markdown pages into DIR instead of saving them')

    def handle(self, what, **options):
        today = timezone.now().date()
        if what == 'topics':
            if options['dry_run']:
                data = data_guides.topics_data(today)
                if data is None:
                    raise CommandError('Not enough weeks of term counts')
                for lang in data_guides.LANGS:
                    self._write(options['dry_run'], lang, data_guides.TOPICS_SLUG, *data_guides.topics_page(data, lang))
            elif not data_guides.write_topics(today):
                raise CommandError('Not enough weeks of term counts')
            self.stdout.write('topics guide written')
            return
        if not options['file']:
            raise CommandError('--file is required')
        if what == 'harvest':
            arxiv_metadata.harvest(options['file'], options['since'], log=self.stdout.write)
            return
        year = options['year'] or today.year - 1
        records = arxiv_metadata.load(options['file'])
        self.stdout.write('%d records' % len(records))
        authors = data_guides.authors_data(records, year, year - 3)
        journals = data_guides.journal_data(records, list(range(year - 3, year + 1)))
        for lang in data_guides.LANGS:
            for slug, page in ((data_guides.AUTHORS_SLUG, data_guides.authors_page(authors, lang, today)),
                               (data_guides.JOURNALS_SLUG, data_guides.journals_page(journals, lang, today))):
                if options['dry_run']:
                    self._write(options['dry_run'], lang, slug, *page)
                else:
                    self.stdout.write(data_guides.write_file(lang, slug, *page))

    def _write(self, folder, lang, slug, meta, body):
        os.makedirs(folder, exist_ok=True)
        path = os.path.join(folder, '%s-%s.md' % (slug, lang))
        with open(path, 'w', encoding='utf-8') as f:
            f.write(data_guides.document(meta, body))
        self.stdout.write(path)
