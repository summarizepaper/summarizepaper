from django.core.management.base import BaseCommand

from summarizer import guides
from summarizer.models import Guide


class Command(BaseCommand):
    help = ('Copy the Markdown guides of content/guides into the database so they can be edited in the admin. '
            'Guides already in the database are kept unless --overwrite.')

    def add_arguments(self, parser):
        parser.add_argument('--overwrite', action='store_true')

    def handle(self, overwrite, **options):
        created = skipped = 0
        for lang, slug in guides.repository_guides():
            if not overwrite and Guide.objects.filter(lang=lang, slug=slug).exists():
                skipped += 1
                continue
            guides.import_file(lang, slug)
            created += 1
            self.stdout.write('%s/%s' % (lang, slug))
        self.stdout.write('%d guide(s) imported, %d already in the database' % (created, skipped))
