"""Tests of the conference companion: reading a program (text, Indico, calendar feed), ranking it, the program
day by day with its clashes, marks, notes, the team, the calendar and notes exports, and the follow-up."""
import datetime
import json
from unittest import mock

import numpy as np
from django.core import mail
from django.core.cache import cache
from django.test import Client
from django.utils import timezone
from django.utils.translation import override as translation_override

from summarizer import conference, conference_import
from summarizer.models import (CitationCache, Collection, ConferenceMark, ConferenceMember, ConferenceProgram,
                               DigestSubscription, FollowedPaper, Generated, WatchedProject)
from summarizer.test_tools import ToolsTestCase

FIND_PAPER = conference.find_paper  # patched in every test but the one of preprints

TALKS = [
    {'title': 'Clouds on hot Jupiters', 'speakers': 'Ann Lee, Bob Ray', 'session': 'Exoplanets', 'day': '2026-07-14',
     'start': '09:00', 'end': '09:20', 'room': 'A', 'abstract': 'We observe clouds on hot Jupiters with JWST.'},
    {'title': 'Exomoons with Gaia', 'speakers': 'Zoe Quinn', 'session': 'Moons', 'day': '2026-07-14',
     'start': '09:10', 'end': '09:30', 'room': 'B', 'abstract': 'Gaia astrometry of exomoons.'},
    {'title': 'Protein folding at scale', 'speakers': 'Max Power', 'session': 'Biology', 'day': '2026-07-15',
     'start': '10:00', 'room': 'C', 'abstract': ''},
    {'title': 'My own clouds talk', 'speakers': 'Quentin Kral', 'session': 'Exoplanets', 'day': '2026-07-15',
     'start': '11:00', 'room': 'A', 'abstract': 'Clouds again.'},
]


def embed(texts):
    """Two dimensions: clouds and planets on one, proteins on the other."""
    return np.array([[0.0, 1.0] if 'Protein' in t else [0.9, 0.436] if 'Exomoons' in t else [1.0, 0.0]
                     for t in texts], dtype=np.float32)


def chat(prompt, **kwargs):
    if 'piece of a conference program' in prompt:
        if 'Clouds on hot' in kwargs['system'].split('<program>')[1]:
            return {'talks': [dict(TALKS[0], day='Tuesday 14 July', start='9:00', end='9.20')]}
        return {'talks': [dict(TALKS[1], day='Tuesday', start='9h10', end='9:30 am'),
                          dict(TALKS[2], day='', start='10 am'), {'title': 'clouds on hot jupiters', 'speakers': 'Ann Lee'}]}
    if 'A researcher goes to the conference' in prompt:
        found = {part.split('">')[0]: part for part in kwargs['system'].split('<talk id="')[1:]}
        notes = []
        for talk_id, text in found.items():
            if 'Clouds on hot' in text:
                notes.append({'id': talk_id, 'priority': 'must', 'why': 'Same planets.', 'ask': 'Which cloud species?'})
            elif 'Exomoons' in text:
                notes.append({'id': talk_id, 'priority': 'should', 'why': 'Close.', 'ask': ''})
            else:
                notes.append({'id': talk_id, 'priority': 'skip'})
        return {'talks': notes}
    if 'Each pair above' in prompt:
        return {'pairs': [{'id': '0', 'overlap': True, 'why': 'Same clouds, same telescope.'}]}
    if 'Each theme above' in prompt:
        return {'themes': [{'id': '0', 'name': 'Planets'}, {'id': '1', 'name': 'Biology'}]}
    if 'first email' in prompt:
        return {'subject': 'Your clouds talk', 'body': 'Dear Ann, [your name]'}
    raise AssertionError(prompt[:80])


class ConferenceCase(ToolsTestCase):
    def setUp(self):
        super().setUp()
        for target, effect in (('summarizer.generated.chat_json', chat), ('summarizer.similar.embed', embed),
                               ('summarizer.conference.find_paper', lambda talk, since: None)):
            patch = mock.patch(target, side_effect=effect)
            patch.start()
            self.addCleanup(patch.stop)

    def program(self, user, talks=TALKS, **fields):
        talks = conference.settle([conference.clean_talk(t) for t in talks])
        return ConferenceProgram.objects.create(user=user, name='EAS 2026', talks=json.dumps(talks), **fields)

    def ranked(self, user, program, **options):
        result = conference.rank(program, 'Clouds of exoplanets', 'en', user, **options)
        return Generated.objects.create(kind='conference', key='k%s' % Generated.objects.count(), user=user, lang='en',
                                        status='done', content=json.dumps(dict(result, name=program.name)))


class ConferenceTests(ConferenceCase):
    def test_times_and_days_are_read_whatever_their_form(self):
        self.assertEqual([conference.clock(v) for v in ('9:05', '09h05', '2.30 pm', '14:30:00', '12 am', 'noon', '25:00')],
                         ['09:05', '09:05', '14:30', '14:30', '00:00', '', ''])
        talks = conference.settle([conference.clean_talk(t) for t in (
            {'title': 'A', 'day': '2026-07-14', 'start': '9:00', 'session': 'S'},
            {'title': 'B', 'day': 'Tue', 'start': '10:00'},
            {'title': 'C', 'start': '11:00'},  # follows B: same day
            {'title': 'D', 'day': 'Wednesday', 'start': '9:00'},  # no Wednesday dated: kept as written
            {'title': 'a', 'speakers': 'Ann Lee', 'abstract': 'Longer abstract'})])
        self.assertEqual([(t['id'], t['title'], t['day']) for t in talks],
                         [('t0', 'A', '2026-07-14'), ('t1', 'B', '2026-07-14'), ('t2', 'C', '2026-07-14'),
                          ('t3', 'D', 'Wednesday')])
        self.assertEqual((talks[0]['speakers'], talks[0]['abstract'], talks[0]['presenter']), ('Ann Lee', 'Longer abstract', 'Ann Lee'))
        twice = conference.settle([conference.clean_talk(t) for t in (
            {'title': 'Updates from the organizers', 'day': '2024-03-12', 'start': '09:00'},
            {'title': 'Updates from the organizers', 'day': '2024-03-13', 'start': '09:00'})])
        self.assertEqual([t['day'] for t in twice], ['2024-03-12', '2024-03-13'])  # two talks, not one listed twice
        self.assertEqual(conference.day_heading('MONDAY, MARCH 11\nTUESDAY, MARCH 12\nPlenary\n9:00 AM   Mardi gras\n'),
                         'TUESDAY, MARCH 12')

    def test_a_long_program_is_read_in_pieces_and_says_where_it_stopped(self):
        text = 'Tuesday 14 July 2026\n\nClouds on hot Jupiters, by Ann Lee\n\n' + 'x ' * 7000 + '\n\nExomoons with Gaia\n\nProtein'
        talks, read = conference.read_text(text)
        self.assertEqual([(t['title'], t['day'], t['start'], t['end']) for t in talks],
                         [('Clouds on hot Jupiters', '2026-07-14', '09:00', '09:20'),
                          ('Exomoons with Gaia', '2026-07-14', '09:10', '09:30'),
                          ('Protein folding at scale', '2026-07-14', '10:00', '')])
        self.assertEqual(read, {})
        with mock.patch.object(conference, 'MAX_PROGRAM', 14000):
            talks, read = conference.read_text(text + 'y' * 20000)
        self.assertEqual(read['total'], len(text) + 20000)
        self.assertIn('Protein folding at scale', read['last'])

    def test_a_piece_starting_within_a_session_takes_its_session_and_room(self):
        pieces = [[{'title': 'A', 'session': 'Tracking', 'room': 'Hall 1', 'start': '9:00'}],
                  [{'title': 'B', 'start': '9:20'}, {'title': 'C poster', 'session': 'Posters', 'start': '16:00'},
                   {'title': 'D', 'start': '16:00', 'speakers': '—'}]]
        with mock.patch('summarizer.conference.chunks', return_value=[(0, 'x'), (1, 'y')]), \
                mock.patch('summarizer.conference.list_talks', side_effect=lambda *a: [conference.clean_talk(t) for t in pieces.pop(0)]):
            talks, _ = conference.read_text('x' * 400)
        self.assertEqual([(t['title'], t['session'], t['room'], t['speakers']) for t in talks],
                         [('A', 'Tracking', 'Hall 1', ''), ('B', 'Tracking', 'Hall 1', ''), ('C poster', 'Posters', '', ''),
                          ('D', '', '', '')])

    def test_indico_timetables_and_calendar_feeds_are_read_whole(self):
        timetable = {'results': {'99': {'20260714': {
            's1': {'entryType': 'Session', 'slotTitle': 'Posters', 'isPoster': True, 'room': 'Hall',
                   'startDate': {'date': '2026-07-14', 'time': '18:00:00', 'tz': 'Europe/Paris'},
                   'entries': {'c2': {'entryType': 'Contribution', 'title': 'Dusty discs', 'contributionId': 7,
                                      'startDate': {'date': '2026-07-14', 'time': '18:00:00', 'tz': 'Europe/Paris'},
                                      'endDate': {'date': '2026-07-14', 'time': '19:30:00'}, 'room': '',
                                      'presenters': [{'name': 'Eve Moss', 'affiliation': 'OCA'}],
                                      'description': '<p>Dust <b>rings</b></p>'}}},
            'c1': {'entryType': 'Contribution', 'title': 'Opening talk', 'contributionId': 3, 'room': 'Amphi',
                   'startDate': {'date': '2026-07-14', 'time': '09:00:00', 'tz': 'Europe/Paris'},
                   'endDate': {'date': '2026-07-14', 'time': '09:30:00'}, 'presenters': []},
            'b1': {'entryType': 'Break', 'title': 'Coffee', 'startDate': {'time': '10:00:00'}}}}}}
        talks, zone = conference_import.indico_talks(timetable, 'https://indico.example.org', '99')
        self.assertEqual(zone, 'Europe/Paris')
        self.assertEqual([(t['title'], t['kind'], t['start'], t['room'], t['presenter']) for t in talks],
                         [('Opening talk', 'talk', '09:00', 'Amphi', ''), ('Dusty discs', 'poster', '18:00', 'Hall', 'Eve Moss')])
        self.assertEqual((talks[1]['abstract'], talks[1]['affiliation'], talks[1]['url']),
                         ('Dust rings', 'OCA', 'https://indico.example.org/event/99/contributions/7/'))

        def fetch(url):
            if 'export/timetable' in url:
                return url, 'application/json', json.dumps(timetable).encode()
            return url, 'application/json', json.dumps({'results': [{'title': 'Discs 2026'}]}).encode()
        with mock.patch('summarizer.conference_import.fetch', side_effect=fetch):
            found = conference_import.read_link('indico.example.org/event/99/timetable/', None)
        self.assertEqual((found['name'], len(found['talks']), found['timezone']), ('Discs 2026', 2, 'Europe/Paris'))

        feed = ('BEGIN:VCALENDAR\r\nX-WR-CALNAME:Sched conf\r\nBEGIN:VEVENT\r\nSUMMARY:Clouds\\, again\r\n'
                'DTSTART;TZID=Europe/Paris:20260714T100000\r\nDTEND;TZID=Europe/Paris:20260714T103000\r\n'
                'LOCATION:Room 1\r\nDESCRIPTION:Speakers: Ann Lee\\nA long \r\n abstract\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n')
        name, talks, zone = conference_import.ics_talks(feed)
        self.assertEqual((name, zone), ('Sched conf', 'Europe/Paris'))
        self.assertEqual((talks[0]['title'], talks[0]['speakers'], talks[0]['start'], talks[0]['end'], talks[0]['abstract']),
                         ('Clouds, again', 'Ann Lee', '10:00', '10:30', 'Speakers: Ann Lee A long abstract'))
        _, talks, _ = conference_import.ics_talks('PRODID:-//pretalx//x\r\nBEGIN:VEVENT\r\nSUMMARY:Clouds - and more - Ann Lee\, Bo Ray\r\n'
                                                  'DTSTART:20260714T080000Z\r\nEND:VEVENT\r\n')
        self.assertEqual((talks[0]['title'], talks[0]['speakers'], talks[0]['start']), ('Clouds - and more', 'Ann Lee, Bo Ray', '08:00'))

    def test_a_page_leads_to_its_calendar_feed(self):
        feed = b'BEGIN:VCALENDAR\r\nBEGIN:VEVENT\r\nSUMMARY:Clouds\r\nDTSTART:20260714T080000Z\r\nEND:VEVENT\r\nEND:VCALENDAR'
        pages = {'https://conf.example.org/program': ('text/html', b'<title>Conf 26</title><a href="/agenda.ics">iCal</a>'),
                 'https://conf.example.org/agenda.ics': ('text/calendar', feed),
                 'https://pretalx.com/conf-26/schedule/': ('text/plain', b'\x1b[1mOpening\x1b[0m drawing'),
                 'https://pretalx.com/conf-26/schedule/export/schedule.ics': ('text/calendar', feed)}
        with mock.patch('summarizer.conference_import.fetch', side_effect=lambda url: (url,) + pages[url]):
            found = conference_import.read_link('https://conf.example.org/program', None)
            self.assertEqual((found['name'], found['talks'][0]['title'], found['source']),
                             ('Conf 26', 'Clouds', 'https://conf.example.org/agenda.ics'))
            found = conference_import.read_link('https://pretalx.com/conf-26/schedule/', None)
            self.assertEqual(found['source'], 'https://pretalx.com/conf-26/schedule/export/schedule.ics')

    def test_a_server_error_is_tried_again_then_told(self):
        answers = [conference_import.LinkError('answered 502'), ('https://x.org/', 'text/html', b'<p>ok</p>')]

        def once(url):
            answer = answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            return answer
        with mock.patch('summarizer.conference_import.fetch_once', side_effect=once), \
                mock.patch('summarizer.conference_import.time.sleep') as sleep:
            self.assertEqual(conference_import.fetch('https://x.org/')[2], b'<p>ok</p>')
        sleep.assert_called_once()
        user = self.user('pro', paid=True)
        client = self.client_for(user)
        with mock.patch('summarizer.conference_import.read_link', side_effect=conference_import.LinkError('answered 503')):
            r = client.post('/en/conference/', {'url': 'https://example.org/program', 'interests': 'clouds of hot planets'})
        self.assertContains(client.get(r['Location']), 'The conference website did not answer')

    def test_only_public_links_are_fetched(self):
        for url in ('http://127.0.0.1/admin', 'http://localhost:8000/', 'file:///etc/passwd', 'http://10.0.0.1/',
                    'http://user:pw@example.org/', 'ftp://example.org/x'):
            with self.subTest(url=url), self.assertRaises(conference_import.LinkError):
                conference_import.fetch(url)
        self.assertEqual(conference_import.utc_stamp('2026-07-14', '10:00', 'Europe/Paris'), '20260714T080000Z')
        self.assertEqual(conference_import.utc_stamp('2026-01-14', '10:00', ''), '20260114T100000')

    def test_ranking_sets_own_talks_apart_flags_overlaps_and_finds_people(self):
        user = self.user('pro', paid=True)
        user.first_name, user.last_name = 'Quentin', 'Kral'
        user.save()
        vector = np.array([1.0, 0.0], dtype=np.float32)
        WatchedProject.objects.create(user=user, title='Cloud survey', description='Clouds of hot Jupiters with JWST',
                                      vector=vector.tobytes(), facet_vectors=np.vstack([vector, vector]).tobytes())
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='My clouds paper', mine=True)
        CitationCache.objects.create(arxiv_id='2101.00001', citing=json.dumps({'p1': {'title': 'X', 'authors': ['Zoe Quinn']}}))
        program = self.program(user)
        result = conference.rank(program, 'Clouds of exoplanets', 'en', user, names=['Quentin Kral'])
        self.assertEqual(result['yours'], ['t3'])
        self.assertEqual([(t['title'], t['priority']) for t in result['talks']],
                         [('Clouds on hot Jupiters', 'must'), ('Exomoons with Gaia', 'should')])
        self.assertEqual(result['talks'][0]['scoop'], {'project': 'Cloud survey', 'why': 'Same clouds, same telescope.'})
        self.assertEqual(result['talks'][0]['ask'], 'Which cloud species?')
        self.assertEqual([(p['name'], p['cites']) for p in result['people']], [('Zoe Quinn', 'My clouds paper'), ('Ann Lee', '')])
        self.assertTrue(program.vectors)
        with mock.patch('summarizer.similar.embed', side_effect=embed) as again:
            conference.rank(program, 'Clouds of exoplanets', 'en', user)
        self.assertEqual(len(again.call_args_list), 1)  # only the interests: the talks' embeddings are kept

    def test_preprints_are_matched_by_title_and_speaker(self):
        work = {'display_name': 'Clouds on hot Jupiters', 'doi': 'https://doi.org/10.48550/arxiv.2601.00001',
                'publication_date': '2026-01-02', 'locations': [{'landing_page_url': 'https://arxiv.org/abs/2601.00001v1'}],
                'authorships': [{'author': {'id': 'https://openalex.org/A1', 'display_name': 'Ann Lee'},
                                 'institutions': [{'display_name': 'MIT'}]}]}
        talk = conference.clean_talk(TALKS[0])
        with mock.patch('summarizer.sources._json', return_value={'results': [dict(work, display_name='Unrelated title here'), work]}):
            paper = FIND_PAPER(talk, 2023)
        self.assertEqual((paper['key'], paper['authors'][0]['id'], paper['authors'][0]['institution']), ('2601.00001', 'A1', 'MIT'))
        with mock.patch('summarizer.sources._json', return_value={'results': [dict(work, authorships=[])]}):
            self.assertIsNone(FIND_PAPER(talk, 2023))  # same title, but no speaker signs it

    def test_the_form_reads_the_program_ranks_it_and_gives_the_quota_back_on_failure(self):
        self.assertContains(self.client_for(self.user()).get('/en/conference/'), 'Discover Premium')
        user = self.user('pro', paid=True)
        DigestSubscription.objects.create(user=user, categories='astro-ph.EP', profile='I study clouds on hot planets.')
        client = self.client_for(user)
        page = client.get('/en/conference/')
        self.assertContains(page, 'I study clouds on hot planets.')
        self.assertContains(client.post('/en/conference/', {'program': 'short', 'interests': 'clouds of planets'}),
                            'Give the link of the program, paste it')
        program = 'Tuesday 14 July 2026\n\nClouds on hot Jupiters, by Ann Lee\n\n' + 'talk ' * 100
        r = client.post('/en/conference/', {'program': program, 'interests': 'clouds of hot planets', 'name': 'EAS 2026'})
        row = Generated.objects.get(kind='conference')
        self.assertEqual(row.status, 'done', row.error)
        page = client.get(r['Location'])
        self.assertContains(page, 'EAS 2026')
        self.assertContains(page, 'Clouds on hot Jupiters')
        self.assertContains(page, 'Tuesday 14 July 2026')
        self.assertContains(client.get('/en/conference/'), 'EAS 2026')
        key = conference.QUOTA_KEY % (user.pk, timezone.now().date())
        self.assertEqual(cache.get(key), 1)
        with mock.patch('summarizer.conference.rank', side_effect=RuntimeError):
            client.post('/en/conference/', {'program': program, 'interests': 'clouds of hot planets'})
        self.assertEqual(cache.get(key), 1)  # the failed reading was given back
        with mock.patch('summarizer.conference_import.read_link', side_effect=conference_import.LinkError('private')):
            r = client.post('/en/conference/', {'url': 'https://example.org/program', 'interests': 'clouds of hot planets'})
        self.assertContains(client.get(r['Location']), 'We could not read this link')

    def test_a_program_day_by_day_marked_annotated_exported_and_shared_with_a_team(self):
        owner = self.user('pro', paid=True)
        program = self.program(owner, timezone='Europe/Paris')
        row = self.ranked(owner, program)
        client = self.client_for(owner)
        url = '/en/conference/%s/' % row.key
        page = client.get(url)
        self.assertContains(page, 'At the same time as')  # the two talks at 9:00 and 9:10
        self.assertContains(page, 'Which cloud species?')
        answer = client.post(url, {'action': 'mark', 'talk': 't0', 'state': 'going'}, HTTP_X_REQUESTED_WITH='fetch')
        self.assertEqual(answer.json()['state'], 'going')
        self.assertEqual(client.post(url, {'action': 'note', 'talk': 't0', 'note': 'Ask about silicates', 'view': 'day'})['Location'],
                         url + '?view=day#talk-t0')
        client.post(url, {'action': 'mark', 'talk': 't2', 'state': 'going'})
        calendar = client.get(url + 'calendar.ics').content.decode().replace('\r\n ', '')
        self.assertIn('DTSTART:20260714T070000Z', calendar)  # 9:00 in Paris
        self.assertIn('SUMMARY:Protein folding at scale', calendar)
        self.assertNotIn('Exomoons', calendar)  # not marked going
        self.assertIn('Ask about silicates', calendar)
        notes = client.get(url + 'notes.md').content.decode()
        self.assertIn('**Notes**: Ask about silicates', notes)
        self.assertContains(client.get(url + '?view=all&q=protein'), 'Protein folding at scale')
        self.assertNotContains(client.get(url + '?view=all&q=protein'), 'Exomoons with Gaia')

        client.post(url, {'action': 'team'})
        program.refresh_from_db()
        colleague = self.user('bob')
        other = self.client_for(colleague)
        self.assertEqual(other.get(url).status_code, 404)
        self.assertContains(other.get('/en/conference/team/%s/' % program.team_token), 'Join the team')
        self.assertEqual(other.post('/en/conference/team/%s/' % program.team_token)['Location'], url)
        other.post(url, {'action': 'mark', 'talk': 't1', 'state': 'going'})
        self.assertContains(client.get(url), 'Also going: bob')
        page = other.get(url)
        self.assertContains(page, 'Also going: pro')
        self.assertNotContains(page, 'Ask about silicates')  # notes stay private
        self.assertNotContains(page, 'Clouds of exoplanets')  # so do the owner's interests
        self.assertEqual(other.post(url, {'action': 'rank', 'interests': 'proteins of many kinds'}).status_code, 404)
        client.post(url, {'action': 'unteam'})
        program.refresh_from_db()
        self.assertEqual(other.get('/en/conference/team/x/').status_code, 404)

    def test_rank_again_collection_and_follow_up_email(self):
        owner = self.user('pro', paid=True)
        program = self.program(owner)
        row = self.ranked(owner, program)
        client = self.client_for(owner)
        url = '/en/conference/%s/' % row.key
        with mock.patch('summarizer.conference.read_source') as read:
            r = client.post(url, {'action': 'rank', 'interests': 'Exomoons and their detection with astrometry'})
        read.assert_not_called()
        again = Generated.objects.filter(kind='conference').exclude(pk=row.pk).get()
        self.assertEqual((r['Location'], again.data()['program']), ('/en/conference/%s/' % again.key, program.pk))
        data = row.data()
        data['talks'][0]['paper'] = {'key': '2601.00001', 'title': 'Clouds', 'authors': []}
        row.content = json.dumps(data)
        row.save()
        with mock.patch('summarizer.library.add_many', return_value={'added': 1}) as add:
            r = client.post(url, {'action': 'collection', 'collection': 'new'})
        collection = Collection.objects.get(user=owner)
        self.assertEqual((add.call_args[0][1], collection.name), (['2601.00001'], 'EAS 2026'))
        self.assertEqual(r['Location'], url + '?view=after&papers=1#papers')  # looked for in the background
        self.assertContains(client.get(url + '?view=after&papers=1'), '1 preprints added.')
        client.post(url, {'action': 'mark', 'talk': 't1', 'state': 'seen'})
        paper = {'key': '2602.00002', 'title': 'Exomoons', 'authors': []}
        with mock.patch('summarizer.library.add_many', return_value={'added': 1}) as add, \
                mock.patch('summarizer.conference.find_paper', return_value=paper) as find:
            client.post(url, {'action': 'collection', 'collection': str(collection.pk)})
        self.assertEqual((add.call_args[0][1], find.call_args[0][0]['id']), (['2602.00002'], 't1'))  # the talk seen
        client.post(url, {'action': 'note', 'talk': 't0', 'note': 'Silicate clouds at 1500 K'})
        client.post(url, {'action': 'mail', 'person': 'Ann Lee'})
        draft = Generated.objects.get(kind='conference_mail')
        self.assertEqual(draft.data()['subject'], 'Your clouds talk')
        self.assertContains(client.get(url + '?view=people'), 'Dear Ann, [your name]')
        self.assertEqual(Generated.objects.filter(kind='conference_mail').count(), 1)
        client.post(url, {'action': 'mail', 'person': 'Somebody Else'})  # not one of the people to meet
        self.assertEqual(Generated.objects.filter(kind='conference_mail').count(), 1)


class ConferenceChangesTests(ConferenceCase):
    """The program read again (changes, emails), the view of now, the team, the time zone, names and preprints."""

    def test_a_program_read_again_keeps_ids_and_tells_what_changed(self):
        old = conference.settle([conference.clean_talk(t) for t in TALKS])
        new = conference.settle([conference.clean_talk(dict(TALKS[1], room='C', start='11:00', end='11:20')),
                                 conference.clean_talk(TALKS[0]), conference.clean_talk(TALKS[3]),
                                 conference.clean_talk({'title': 'A late talk', 'day': '2026-07-15', 'start': '16:00'})])
        talks, changes = conference.merge_talks(old, new)
        by_title = {t['title']: t for t in talks}
        self.assertEqual((by_title['Exomoons with Gaia']['id'], by_title['Clouds on hot Jupiters']['id'],
                          by_title['A late talk']['id']), ('t1', 't0', 't4'))
        self.assertTrue(by_title['Protein folding at scale']['cancelled'])
        self.assertEqual({c['title']: c.get('what') or ('added' if c.get('added') else 'cancelled') for c in changes},
                         {'Exomoons with Gaia': {'start': ['09:10', '11:00'], 'end': ['09:30', '11:20'], 'room': ['B', 'C']},
                          'Protein folding at scale': 'cancelled', 'A late talk': 'added'})
        with translation_override('en'):
            self.assertEqual(conference.change_lines(changes[0], None), ['09:10 → 11:00', 'Room: B → C'])

    def test_changes_are_read_hourly_emailed_and_shown(self):
        owner = self.user('pro', paid=True)
        owner.email = 'pro@example.org'
        owner.save()
        program = self.program(owner, source='https://indico.example.org/event/9/', source_kind='indico',
                               first_day=timezone.localdate() + datetime.timedelta(days=3),
                               last_day=timezone.localdate() + datetime.timedelta(days=4))
        row = self.ranked(owner, program)
        mate = self.user('bob')
        mate.email = 'bob@example.org'
        mate.save()
        ConferenceMember.objects.create(program=program, user=mate)
        ConferenceMark.objects.create(program=program, user=mate, talk='t2', state='going')
        moved = [dict(TALKS[0], room='Amphi'), TALKS[1], TALKS[3]]  # Protein folding is cancelled
        with mock.patch('summarizer.conference_import.read_link', return_value={'talks': moved, 'kind': 'indico'}):
            sent = conference.refresh_due(timezone.now(), 10)
            again = conference.refresh_due(timezone.now(), 10)  # read a moment ago: not again
        self.assertEqual((sent, again), (2, 0))
        mails = {m.to[0]: m for m in mail.outbox}
        self.assertIn('Room: A → Amphi', mails['pro@example.org'].body)  # a talk not to miss of the owner's ranking
        self.assertIn('Cancelled', mails['bob@example.org'].body)  # the talk the colleague goes to
        self.assertNotIn('Amphi', mails['bob@example.org'].body)
        page = self.client_for(owner).get('/en/conference/%s/' % row.key)
        self.assertContains(page, 'Changed: Room: A → Amphi')
        self.assertContains(page, 'Changed in the program this week')
        client = self.client_for(mate)
        client.post('/en/conference/%s/' % row.key, {'action': 'quiet'})
        program.refresh_from_db()
        self.assertEqual(program.quiet_ids(), [mate.pk])
        self.assertNotIn('Protein', client.get('/en/conference/%s/calendar.ics' % row.key).content.decode())

    def test_now_shows_the_talk_on_and_the_next_ones(self):
        owner = self.user('pro', paid=True)
        program = self.program(owner, timezone='Europe/Paris', first_day=datetime.date(2026, 7, 14),
                               last_day=datetime.date(2026, 7, 15))
        row = self.ranked(owner, program)
        moment = datetime.datetime(2026, 7, 14, 7, 5, tzinfo=datetime.timezone.utc)  # 9:05 in Paris
        with mock.patch('summarizer.conference.timezone.now', return_value=moment):
            page = self.client_for(owner).get('/en/conference/%s/' % row.key).content.decode()
        self.assertIn('aria-current="page">Now', page)
        now, upcoming = page.split('>Next<') if '>Next<' in page else (page, '')
        self.assertIn('Clouds on hot Jupiters', now.split('<h2>Now</h2>')[1])
        self.assertIn('Exomoons with Gaia', upcoming)
        with mock.patch('summarizer.conference.timezone.now', return_value=moment + datetime.timedelta(days=5)):
            self.assertContains(self.client_for(owner).get('/en/conference/%s/?view=now' % row.key), 'The conference is over')
        sw = self.client_for(owner).get('/en/conference/sw.js')
        self.assertEqual(sw['Content-Type'], 'application/javascript')

    def test_a_colleague_ranks_the_shared_program_and_sees_what_nobody_covers(self):
        owner = self.user('pro', paid=True)
        program = self.program(owner, team_token='team1')
        row = self.ranked(owner, program)
        mate = self.user('bob', paid=True)
        client = self.client_for(mate)
        client.post('/en/conference/team/team1/')
        self.assertContains(client.get('/en/conference/%s/?view=day' % row.key), 'Nobody from the team yet')
        r = client.post('/en/conference/%s/' % row.key, {'action': 'rank', 'interests': 'Exomoons and their detection'})
        own = Generated.objects.filter(kind='conference', user=mate).get()
        self.assertEqual((r['Location'], own.data()['program']), ('/en/conference/%s/' % own.key, program.pk))
        self.assertEqual(client.get('/en/conference/team/team1/')['Location'], '/en/conference/%s/' % own.key)
        self.client_for(owner).post('/en/conference/%s/' % row.key, {'action': 'mark', 'talk': 't0', 'state': 'going'})
        page = client.get('/en/conference/%s/?view=day' % own.key)
        self.assertContains(page, 'Also going: pro')
        unpaid = self.user('eve')
        ConferenceMember.objects.create(program=program, user=unpaid)
        self.assertEqual(self.client_for(unpaid).post('/en/conference/%s/' % row.key, {
            'action': 'rank', 'interests': 'Exomoons and their detection'}).status_code, 404)

    def test_scoop_alerts_stay_apart_from_the_ranking_and_posters_never_clash(self):
        user = self.user('pro', paid=True)
        vector = np.array([0.0, 1.0], dtype=np.float32)  # a project on proteins: not the interests
        WatchedProject.objects.create(user=user, title='Folding', description='Protein folding at scale',
                                      vector=vector.tobytes(), facet_vectors=np.vstack([vector, vector]).tobytes())
        program = self.program(user, talks=TALKS + [{'title': 'Poster on clouds', 'kind': 'poster', 'day': '2026-07-14',
                                                      'start': '09:00', 'end': '10:30', 'room': 'Hall'}])
        result = conference.rank(program, 'Clouds of exoplanets', 'en', user)
        self.assertEqual([s['id'] for s in result['scoops']], ['t2'])
        self.assertNotIn('Protein folding at scale', [t['title'] for t in result['talks']])
        row = Generated.objects.create(kind='conference', key='s', user=user, lang='en', status='done',
                                       content=json.dumps(dict(result, name='EAS')))
        page = self.client_for(user).get('/en/conference/s/?view=priority')
        self.assertContains(page, 'Close to your projects in progress')
        items = [{'talk': t, 'role': 'should'} for t in program.talk_list() if t['start'] == '09:00']
        conference.clashes(items)
        self.assertEqual([i['clash'] for i in items], [[], []])  # a talk and a poster at 9:00: the poster is seen any time

    def test_times_in_utc_move_to_the_conference_zone(self):
        feed = ('BEGIN:VCALENDAR\r\nX-WR-TIMEZONE:Europe/Paris\r\nBEGIN:VEVENT\r\nSUMMARY:Clouds\r\n'
                'DTSTART:20260714T080000Z\r\nDTEND:20260714T083000Z\r\nEND:VEVENT\r\nEND:VCALENDAR\r\n')
        _, talks, zone = conference_import.ics_talks(feed)
        self.assertEqual((talks[0]['start'], talks[0]['end'], zone), ('10:00', '10:30', 'Europe/Paris'))
        owner = self.user('pro', paid=True)
        program = self.program(owner, timezone='UTC')
        row = self.ranked(owner, program)
        client = self.client_for(owner)
        self.assertContains(client.get('/en/conference/%s/?view=after' % row.key), 'gives its times in UTC')
        client.post('/en/conference/%s/' % row.key, {'action': 'zone', 'zone': 'America/New_York'})
        program.refresh_from_db()
        self.assertEqual((program.timezone, program.talk_list()[0]['day'], program.talk_list()[0]['start']),
                         ('America/New_York', '2026-07-14', '05:00'))

    def test_names_and_a_close_paper_by_the_speaker(self):
        self.assertEqual([conference.person_name(n) for n in ('Dr Jerome LAURET', 'Prof. Ann SEXTON-KENNEDY', 'ATLAS')],
                         ['Jerome Lauret', 'Ann Sexton-Kennedy', 'ATLAS'])
        self.assertEqual(conference.clean_talk({'title': 'x', 'speakers': 'Dr Ann Lee, Prof. Bob Ray'})['speakers'], 'Ann Lee, Bob Ray')
        talk = conference.clean_talk(TALKS[0])
        work = {'display_name': 'Silicate clouds of WASP-17b', 'doi': 'https://doi.org/10.1/x', 'publication_date': '2025-01-01',
                'authorships': [{'author': {'id': 'https://openalex.org/A1', 'display_name': 'Ann Lee'}}],
                'abstract_inverted_index': {'clouds': [0]}}
        answers = [{'results': []}, {'results': [work, dict(work, authorships=[])]}]
        with mock.patch('summarizer.sources._json', side_effect=lambda *a, **k: answers.pop(0)), \
                mock.patch('summarizer.similar.embed', return_value=np.array([[1.0, 0.0], [0.9, 0.436]], dtype=np.float32)):
            paper = FIND_PAPER(talk, 2023)
        self.assertEqual((paper['match'], paper['key'][:4], paper['title'], paper['closeness']),
                         ('speaker', 'doi_', 'Silicate clouds of WASP-17b', 0.9))


class ConferenceFollowsTests(ConferenceCase):
    """A program read again without false changes, a renamed talk, cancelled talks left out of a ranking, the emails
    of a reading by hand and of the talks added, the calendar feed, and a ranking that loses a part only."""

    def test_a_program_in_utc_moved_to_its_zone_reads_again_without_false_changes(self):
        owner = self.user('pro', paid=True)
        utc = [dict(TALKS[0], start='07:00', end='07:20'), dict(TALKS[1], start='07:10', end='07:30')]
        program = self.program(owner, talks=utc, timezone='UTC', source='https://example.org/f.ics', source_kind='feed')
        conference.set_zone(program, 'Europe/Paris')
        with mock.patch('summarizer.conference_import.read_link', return_value={'talks': utc, 'timezone': 'UTC', 'kind': 'feed'}):
            self.assertEqual(conference.refresh(program, timezone.now()), [])
        program.refresh_from_db()
        self.assertEqual((program.timezone, program.talk_list()[0]['start']), ('Europe/Paris', '09:00'))

    def test_a_talk_whose_title_is_corrected_keeps_its_marks(self):
        old = conference.settle([conference.clean_talk(t) for t in TALKS])
        new = conference.settle([conference.clean_talk(t) for t in [dict(TALKS[0], title='Clouds on hot Jupiters seen by JWST')] + TALKS[1:]])
        talks, changes = conference.merge_talks(old, new)
        self.assertEqual((talks[0]['id'], talks[0]['title'], changes), ('t0', 'Clouds on hot Jupiters seen by JWST', []))
        other = conference.settle([conference.clean_talk(t) for t in [dict(TALKS[0], title='Dark matter halos', speakers='Max Power')] + TALKS[1:]])
        _, changes = conference.merge_talks(old, other)  # another talk in its slot
        self.assertEqual(sorted((c['title'], 'added' if c.get('added') else 'cancelled') for c in changes),
                         [('Clouds on hot Jupiters', 'cancelled'), ('Dark matter halos', 'added')])

    def test_a_ranking_leaves_out_the_cancelled_talks(self):
        user = self.user('pro', paid=True)
        program = self.program(user)
        talks = program.talk_list()
        talks[0]['cancelled'] = True
        program.talks = json.dumps(talks)
        program.save()
        result = conference.rank(program, 'Clouds of exoplanets', 'en', user)
        self.assertEqual(([t['title'] for t in result['talks']], result['listed']), (['Exomoons with Gaia'], 3))

    def test_a_reading_by_hand_emails_the_others_and_the_talks_added_close_to_their_research(self):
        owner = self.user('pro', paid=True)
        owner.email = 'pro@example.org'
        owner.save()
        program = self.program(owner, source='https://indico.example.org/event/9/', source_kind='indico')
        row = self.ranked(owner, program)
        mate = self.user('bob')
        mate.email = 'bob@example.org'
        mate.save()
        ConferenceMember.objects.create(program=program, user=mate)
        ConferenceMark.objects.create(program=program, user=mate, talk='t2', state='going')
        new = TALKS[:2] + TALKS[3:] + [{'title': 'Clouds of brown dwarfs', 'speakers': 'Eve Moss', 'day': '2026-07-15',
                                        'start': '15:00', 'abstract': 'Clouds again.'},
                                       {'title': 'Protein design', 'speakers': 'Max Power', 'day': '2026-07-15', 'start': '16:00'}]
        with mock.patch('summarizer.conference_import.read_link', return_value={'talks': new, 'kind': 'indico'}):
            r = self.client_for(mate).post('/en/conference/%s/' % row.key, {'action': 'refresh'})
        self.assertIn('refreshed=3', r['Location'])
        self.assertEqual([m.to[0] for m in mail.outbox], ['pro@example.org'])  # not the one who read it again
        body = mail.outbox[0].body
        self.assertIn('Clouds of brown dwarfs', body)
        self.assertIn('New in the program', body)
        self.assertNotIn('Protein', body)  # far from the research, and the cancelled talk is not in the owner's program

    def test_a_calendar_feed_follows_the_program_without_the_notes(self):
        owner = self.user('pro', paid=True)
        program = self.program(owner, timezone='Europe/Paris')
        row = self.ranked(owner, program)
        client = self.client_for(owner)
        url = '/en/conference/%s/' % row.key
        client.post(url, {'action': 'note', 'talk': 't0', 'note': 'Ask about silicates'})
        feed = client.get(url + '?view=after').context['feed_url']
        self.assertTrue(feed.startswith('webcal://testserver/en/conference/feed/'))
        path = feed[len('webcal://testserver'):]
        calendar = Client().get(path).content.decode().replace('\r\n ', '')
        self.assertIn('DTSTART:20260714T070000Z', calendar)
        self.assertNotIn('silicates', calendar)
        talks = program.talk_list()
        talks[0]['start'], talks[0]['end'] = '10:00', '10:20'  # moved, as a reading again does
        program.talks = json.dumps(talks)
        program.save()
        self.assertIn('DTSTART:20260714T080000Z', Client().get(path).content.decode())
        self.assertIn('DTSTART:20260714T080000Z', client.get(url + 'calendar.ics').content.decode())
        self.assertEqual(Client().get(path.replace('.ics', '0.ics')).status_code, 404)
        self.assertEqual(Client().get('/en/conference/feed/%s-%s-x.ics' % (row.key, owner.pk)).status_code, 404)

    def test_a_ranking_loses_the_batch_or_the_preprints_that_fail_only(self):
        user = self.user('pro', paid=True)
        program = self.program(user)

        def failing(prompt, **kwargs):
            if 'A researcher goes to the conference' in prompt and 'Exomoons' in kwargs['system']:
                raise ValueError('malformed answer')
            return chat(prompt, **kwargs)
        with mock.patch.object(conference, 'JUDGE_BATCH', 1), mock.patch('summarizer.generated.chat_json', side_effect=failing), \
                mock.patch('summarizer.conference.find_paper', side_effect=RuntimeError('OpenAlex')), \
                self.assertLogs('summarizer.conference', 'ERROR'):
            result = conference.rank(program, 'Clouds of exoplanets', 'en', user)
        self.assertEqual(([t['title'] for t in result['talks']], result['unread']), (['Clouds on hot Jupiters'], 1))
        with mock.patch('summarizer.generated.chat_json', side_effect=ValueError('down')), self.assertLogs('summarizer.conference', 'ERROR'), \
                self.assertRaises(conference.generated.GenerationError):
            conference.rank(program, 'Clouds of exoplanets', 'en', user)
