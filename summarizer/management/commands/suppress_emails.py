import re

from django.core.management.base import BaseCommand

from summarizer import mailcheck


class Command(BaseCommand):
    help = ('Never email these addresses again (hard bounces, complaints). Takes addresses, '
            'or --file with any text containing them, such as a CloudMailin export.')

    def add_arguments(self, parser):
        parser.add_argument('emails', nargs='*')
        parser.add_argument('--file')
        parser.add_argument('--reason', default='bounce')

    def handle(self, emails, file, reason, **options):
        if file:
            with open(file) as f:
                emails += re.findall(r'[\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+', f.read())
        added = sum(mailcheck.suppress(e, reason) for e in {e.lower() for e in emails})
        self.stdout.write('%d address(es) added, %d suppressed in total' % (added, len(mailcheck.suppressed())))
