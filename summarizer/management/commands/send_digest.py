from django.core.management.base import BaseCommand

from summarizer import digest


class Command(BaseCommand):
    help = 'Email each digest subscriber the new arXiv papers in their categories (run daily).'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true', help='Select papers and write blurbs, send nothing.')
        parser.add_argument('--user', help='Only this username.')
        parser.add_argument('--base-url', default='https://www.summarizepaper.com')

    def handle(self, *args, **options):
        sent = digest.run(base_url=options['base_url'], dry_run=options['dry_run'],
                          only_user=options['user'], stdout=self.stdout)
        self.stdout.write(self.style.SUCCESS('%d digest(s) sent' % sent))
