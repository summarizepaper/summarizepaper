"""How well the conference companion reads a program and ranks its talks, on a real program annotated by hand.
Writes nothing (everything happens in a transaction rolled back).

python manage.py conference_benchmark                 # reading (web page and PDF layouts) and ranking
python manage.py conference_benchmark --only rank     # ranking only; --only read: reading only

The program is ACAT 2024 (fixtures/conference_benchmark.json), 175 contributions over five days, as its Indico
timetable gives them: the truth of each talk's day, times, room and presenter. Reading: the program is written
out as a researcher would paste it, from the conference's website (day headers with the year, times, speakers,
abstracts) and from a PDF booklet (no year but on the cover, 12-hour times, no abstracts), and read by the model;
it prints the talks found and missed, and how often their day, start, end, room and presenter are right.
Ranking: for each research profile, the talks annotated as core (to be chosen, 'must' or 'should') and off topic
(never to be), and the preprints found for the chosen talks. It costs about 20 model calls and 80 OpenAlex calls.
"""
import datetime
import json
import time
from pathlib import Path

from django.contrib.auth.models import User
from django.core.management.base import BaseCommand
from django.db import transaction

from summarizer import conference, generated
from summarizer.models import ConferenceProgram

FIXTURE = Path(__file__).resolve().parents[2] / 'fixtures' / 'conference_benchmark.json'


class Rollback(Exception):
    pass


def web_text(data):
    """The program as copied from the conference's timetable page."""
    out, day, session = [data['intro'], 'Timetable', ''], None, None
    for t in data['talks']:
        if t['day'] != day:
            day, session = t['day'], None
            out.append('\n%s\n' % datetime.date.fromisoformat(day).strftime('%A %-d %B %Y'))
        if t['session'] != session:
            session = t['session']
            out.append('%s\n%s' % (session or 'Plenary', t['room']))
        out.append('%s → %s  %s\nSpeaker: %s%s\n%s\n' % (t['start'], t['end'], t['title'], t['speakers'] or '—',
                                                        ' (%s)' % t['affiliation'] if t['affiliation'] else '', t['abstract']))
    return '\n'.join(out)


def pdf_text(data):
    """The program as a PDF booklet gives it: the year on the cover only, 12-hour times, posters without times."""
    def twelve(time_):
        hour, minute = int(time_[:2]), time_[3:]
        return '%d:%s %s' % (hour % 12 or 12, minute, 'PM' if hour >= 12 else 'AM')
    out, day, session = [data['intro'].upper(), 'PROGRAM BOOKLET'], None, None
    for t in data['talks']:
        if t['day'] != day:
            day, session = t['day'], None
            out.append('\n%s\n' % datetime.date.fromisoformat(day).strftime('%A, %B %-d').upper())
        if t['session'] != session:
            session = t['session']
            first = '%s (%s)' % (session or 'Plenary', t['room'])
            out.append('%s  %s' % (twelve(t['start']), first) if t['kind'] == 'poster' else first)
        speaker = '%s — %s' % (t['speakers'], t['affiliation']) if t['affiliation'] else t['speakers']
        if t['kind'] == 'poster':
            out.append('  • %s\n    %s' % (t['title'], speaker))
        else:
            out.append('%s - %s   %s\n            %s' % (twelve(t['start']), twelve(t['end']), t['title'], speaker))
    return '\n'.join(out)


def matcher(truth):
    keys = {conference.title_key(t['title']): t for t in truth}

    def match(title):
        found = keys.get(conference.title_key(title))
        if found:
            return found
        words = conference.words(title)
        return next((t for t in truth if words and len(words & conference.words(t['title'])) /
                     len(words | conference.words(t['title'])) >= 0.8), None)
    return match


class Command(BaseCommand):
    help = 'Read and rank a real conference program annotated by hand, and report what is missed or wrong.'

    def add_arguments(self, parser):
        parser.add_argument('--only', default='', choices=['', 'read', 'rank'])
        parser.add_argument('--lang', default='en')

    def handle(self, *args, **options):
        data = json.loads(FIXTURE.read_text())
        if options['only'] != 'rank':
            for layout, text in (('web page', web_text(data)), ('PDF booklet', pdf_text(data))):
                self.read(layout, text, data['talks'])
        if options['only'] != 'read':
            self.rank(data, options['lang'])

    def read(self, layout, text, truth):
        out = self.stdout.write
        started = time.monotonic()
        talks, read = conference.read_text(text)
        match = matcher(truth)
        found = [(t, match(t['title'])) for t in talks]
        matched = {id(m): (t, m) for t, m in found if m}
        out('\n== Reading the %s (%d characters, %.0f s)' % (layout, len(text), time.monotonic() - started))
        out('%d talks listed, %d of the %d of the program (%.0f%%), %d not in it%s' % (
            len(talks), len(matched), len(truth), 100 * len(matched) / len(truth), sum(1 for _, m in found if not m),
            ', reading cut: %s' % read if read else ''))
        for field in ('day', 'start', 'end', 'room', 'presenter'):
            right = sum(1 for t, m in matched.values() if (t[field] or '').lower() == (m[field] or '').lower())
            out('  %-9s right for %d of %d (%.0f%%)' % (field, right, len(matched), 100 * right / max(1, len(matched))))
        missed = [m['title'] for m in truth if id(m) not in matched]
        for title in missed[:15]:
            out('  missed: %s' % title[:100])
        for t, m in found:
            if not m:
                out('  not in the program: %s' % t['title'][:100])
        wrong_day = [(t['title'], t['day'], m['day']) for t, m in matched.values() if t['day'] != m['day']]
        for title, got, want in wrong_day[:10]:
            out('  day %s instead of %s: %s' % (got or '(none)', want, title[:80]))

    def rank(self, data, lang):
        out = self.stdout.write
        talks = conference.settle([conference.clean_talk(t) for t in data['talks']])
        try:
            with transaction.atomic():
                user = User.objects.create_user('conference-benchmark-%s' % time.time_ns())
                program = ConferenceProgram.objects.create(user=user, name=data['name'], talks=json.dumps(talks),
                                                           timezone=data['timezone'])
                for profile in data['profiles']:
                    started = time.monotonic()
                    result = conference.rank(program, profile['interests'], lang, user)
                    chosen = {t['title']: t for t in result['talks']}
                    strong = {title for title, t in chosen.items() if t['priority'] in ('must', 'should')}
                    out('\n== Ranking for "%s" (%.0f s): %d talks chosen of %d read closely; %d must, %d should, %d maybe' % (
                        profile['id'], time.monotonic() - started, len(chosen), result['judged'],
                        *(sum(1 for t in result['talks'] if t['priority'] == p) for p in conference.PRIORITIES)))
                    core_strong = [c for c in profile['core'] if c in strong]
                    out('core talks chosen as must or should: %d of %d; chosen at all: %d' % (
                        len(core_strong), len(profile['core']), sum(1 for c in profile['core'] if c in chosen)))
                    for title in profile['core']:
                        if title not in strong:
                            out('  core %s: %s' % (chosen[title]['priority'] if title in chosen else 'left out', title[:90]))
                    off = [o for o in profile['off'] if o in strong]
                    out('off-topic talks chosen as must or should: %d of %d' % (len(off), len(profile['off'])))
                    for title in off:
                        out('  off topic but %s: %s (%s)' % (chosen[title]['priority'], title[:80], chosen[title]['why'][:100]))
                    papers = [t for t in result['talks'] if t.get('paper')]
                    out('preprints found: %d of %d chosen talks looked for' % (len(papers), min(len(chosen), conference.PAPERS)))
                    for t in papers[:8]:
                        out('  %s\n    -> %s (%s)' % (t['title'][:90], t['paper']['title'][:90], t['paper']['key']))
                    out('people to meet: %s' % ', '.join(p['name'] for p in result['people'][:10]))
                    out('themes: %s' % '; '.join('%s (%d)' % (t['name'], t['count']) for t in result['themes']))
                    for t in result['talks'][:3]:
                        out('  e.g. [%s] %s\n     why: %s\n     ask: %s' % (t['priority'], t['title'][:80], t['why'], t['ask']))
                raise Rollback
        except Rollback:
            pass
