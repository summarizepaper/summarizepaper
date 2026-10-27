"""Write the weekly letters of a week now, or see what they would hold, to check them on real weeks.

python manage.py letters_write --dry-run --topic ai             # the choice and the text, nothing stored
python manage.py letters_write --topic space --topic life       # write the letters still missing this week
python manage.py letters_write --force                          # write the whole week again, in every language

The week is the one of the latest letters (or of the latest announcement). For each letter it prints how many
papers were read, kept and shortlisted, the six chosen with their sheet, and the numbers checked, rewritten
or dropped. --force deletes the week's letters of these themes first, so that the papers are chosen again.
"""
import datetime
import json

from django.core.management.base import BaseCommand

from summarizer import letters
from summarizer.models import ArxivBatch, Letter


class Command(BaseCommand):
    help = "Write the weekly letters of a week now, or print what they would hold."

    def add_arguments(self, parser):
        parser.add_argument('--topic', action='append', choices=list(letters.TOPICS))
        parser.add_argument('--lang', action='append', choices=list(letters.LANGS))
        parser.add_argument('--day', help='The announcement day the letters are written after; the latest otherwise.')
        parser.add_argument('--dry-run', action='store_true', help='Print the letters without storing them.')
        parser.add_argument('--force', action='store_true', help="Delete the week's letters of these themes and write them again.")

    def handle(self, *args, **options):
        day = datetime.date.fromisoformat(options['day']) if options['day'] else (
            Letter.objects.order_by('-day').values_list('day', flat=True).first()
            or ArxivBatch.objects.order_by('-day').values_list('day', flat=True).first())
        if day is None:
            self.stdout.write('No announcement yet.')
            return
        topics = options['topic'] or list(letters.TOPICS)
        langs = options['lang'] or list(letters.LANGS)
        if options['force'] and not options['dry_run']:
            deleted = Letter.objects.filter(day=day, topic__in=topics, lang__in=langs).delete()[0]
            self.stdout.write('%d letters of %s deleted' % (deleted, day))
        for topic in topics:
            for lang in langs:
                if not options['force'] and not options['dry_run'] and Letter.objects.filter(topic=topic, day=day, lang=lang).exists():
                    self.stdout.write('%s %s %s: already written' % (topic, lang, day))
                    continue
                letter = letters.write(topic, day, lang, save=not options['dry_run'])
                if letter is None:
                    self.stdout.write('%s %s %s: no letter (too few papers, or too few sections holding their numbers)' % (topic, lang, day))
                    continue
                self.report(letter)

    def report(self, letter):
        content = json.loads(letter.content)
        selection = content.get('selection', {})
        self.stdout.write('\n=== %s %s %s: %s' % (letter.topic, letter.lang, letter.day, letter.title))
        self.stdout.write('%s papers in the theme, %s read, %s kept by the screening, %s shortlisted; %s chosen in a rising '
                          'topic, %s close to a research profile' % (
                              content.get('count'), selection.get('screened'), selection.get('kept'), selection.get('shortlist'),
                              selection.get('rising'), selection.get('interest')))
        self.stdout.write('numbers: %s checked, %s parts rewritten, %s sections dropped' % (
            selection.get('numbers'), selection.get('rewritten'), selection.get('dropped')))
        for arxiv_id in content.get('order', []):
            facts = content.get('facts', {}).get(arxiv_id, {})
            self.stdout.write('- %s %s\n  %s · %s · %s %s · code %s · rests on: %s\n  limit: %s\n  earlier: %s' % (
                arxiv_id, content['papers'].get(arxiv_id, ''), facts.get('category'), facts.get('study'), facts.get('status'),
                facts.get('journal', ''), facts.get('code') or '-', facts.get('evidence') or '-', facts.get('limit') or '-',
                ' | '.join(e['title'] for e in content.get('earlier', {}).get(arxiv_id, [])) or '-'))
        for section in content['sections']:
            self.stdout.write('\n## %s  [%s]\n%s' % (section['heading'], section.get('query', ''), section['text']))
