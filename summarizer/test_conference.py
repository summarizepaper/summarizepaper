"""Tests of the conference companion: reading a program (text, Indico, calendar feed), ranking it, the program
day by day with its clashes, marks, notes, the team, the calendar and notes exports, and the follow-up."""
import json
from unittest import mock

import numpy as np
from django.core.cache import cache
from django.test import Client
from django.utils import timezone

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


class ConferenceTests(ToolsTestCase):
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
        self.assertIn('added=1', r['Location'])
        client.post(url, {'action': 'note', 'talk': 't0', 'note': 'Silicate clouds at 1500 K'})
        client.post(url, {'action': 'mail', 'person': 'Ann Lee'})
        draft = Generated.objects.get(kind='conference_mail')
        self.assertEqual(draft.data()['subject'], 'Your clouds talk')
        self.assertContains(client.get(url + '?view=people'), 'Dear Ann, [your name]')
        self.assertEqual(Generated.objects.filter(kind='conference_mail').count(), 1)
        client.post(url, {'action': 'mail', 'person': 'Somebody Else'})  # not one of the people to meet
        self.assertEqual(Generated.objects.filter(kind='conference_mail').count(), 1)
