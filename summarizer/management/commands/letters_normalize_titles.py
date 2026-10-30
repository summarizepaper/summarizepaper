"""Correct stored French letter headings without regenerating or sending letters."""
from django.core.management.base import BaseCommand
from summarizer import letters
from summarizer.models import Letter


class Command(BaseCommand):
    help = 'Replace English weekly headings in existing French letter titles.'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')

    def handle(self, *args, **options):
        changed = 0
        for row in Letter.objects.filter(lang='fr', topic__in=letters.TOPICS).only('id', 'topic', 'lang', 'title'):
            corrected = letters.normalized_title(row.topic, row.lang, row.title)
            if corrected != row.title:
                changed += 1
                if not options['dry_run']:
                    Letter.objects.filter(pk=row.pk, title=row.title).update(title=corrected)
        self.stdout.write('%d titles %s' % (changed, 'to correct' if options['dry_run'] else 'corrected'))
