import os

import yaml
from django.core.management.base import BaseCommand

from summarizer import guides
from summarizer.models import Guide

FIELDS = ('title', 'description', 'date', 'updated', 'draft', 'product')


class Command(BaseCommand):
    help = ('Copy the Markdown guides of content/guides into the database so they can be edited in the admin. '
            'Guides already in the database are kept unless --overwrite.')

    def add_arguments(self, parser):
        parser.add_argument('--overwrite', action='store_true')

    def handle(self, overwrite, **options):
        created = skipped = 0
        for lang in sorted(os.listdir(guides.CONTENT_DIR)):
            folder = os.path.join(guides.CONTENT_DIR, lang)
            for name in sorted(os.listdir(folder)):
                if not name.endswith('.md'):
                    continue
                slug = name[:-3]
                if not overwrite and Guide.objects.filter(lang=lang, slug=slug).exists():
                    skipped += 1
                    continue
                with open(os.path.join(folder, name), encoding='utf-8') as f:
                    text = f.read()
                meta = {}
                if text.startswith('---'):
                    _, front, text = text.split('---', 2)
                    meta = yaml.safe_load(front) or {}
                values = {k: meta.pop(k) for k in FIELDS if k in meta}
                values['draft'] = bool(values.get('draft'))
                values['product'] = bool(values.get('product'))
                values['body'] = text.strip() + '\n'
                values['extra'] = yaml.safe_dump(meta, allow_unicode=True, sort_keys=False) if meta else ''
                Guide.objects.update_or_create(lang=lang, slug=slug, defaults=values)
                created += 1
                self.stdout.write('%s/%s' % (lang, slug))
        self.stdout.write('%d guide(s) imported, %d already in the database' % (created, skipped))
