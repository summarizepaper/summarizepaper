import json
from unittest import mock

from django.test import Client
from django.utils import timezone

from summarizer import students
from summarizer.models import Generated, PremiumGrant, StudentShortlist
from summarizer.test_tools import ToolsTestCase


class StudentTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(students, 'latest_affiliations', side_effect=self.latest)
        patch.start()
        self.addCleanup(patch.stop)

    def latest(self, aid):
        return {'institutions': [{'id': aid.replace('/A', '/I'), 'display_name': 'Example University',
                                  'country_code': 'FR', 'type': 'education'}],
                'date': timezone.now().date().isoformat(), 'checked': timezone.now().isoformat(),
                'paper': {'title': 'Latest independent publication', 'url': 'https://doi.org/10.1/latest'},
                'conflict': False}

    profile = {'level': 'M1', 'kind': 'internship', 'topic': 'Detecting exomoons by astrometry',
               'country': 'France', 'city': 'Paris', 'mobility': 'strict', 'start': '', 'duration': 6, 'skills': 'Python'}

    def paper_data(self, aid='A1', iid='I1', country='FR', title='Exomoon astrometry'):
        return {'id': 'W1', 'display_name': title, 'publication_date': '2026-01-01',
                'doi': 'https://doi.org/10.1/example', 'authorships': [
                    {'author': {'id': 'https://openalex.org/'+aid, 'display_name': 'Alex Example'}, 'author_position': 'first',
                     'institutions': [{'id': 'https://openalex.org/'+iid, 'display_name': 'Example University',
                                       'country_code': country, 'type': 'education'}]}]}

    def details(self, city='Paris', country='France', code='FR'):
        return {'geo': {'city': city, 'country': country, 'country_code': code},
                'country_code': code, 'homepage_url': 'https://example.edu'}

    def candidate(self):
        with mock.patch.object(students, 'institution_details', return_value=self.details()):
            found, _ = students.candidates([self.paper_data()], self.profile)
        return found[0]

    def row(self, user, key='student-demo', found=None):
        return Generated.objects.create(user=user, kind='students', key=key, lang='en', status='done', content=json.dumps({
            'version': students.RESULT_VERSION, 'title': 'Exomoons', 'profile': self.profile, 'candidates': [self.candidate()] if found is None else found}))

    def test_strict_location_never_relaxes_and_unknown_city_is_not_a_match(self):
        for details in (self.details(city='Lyon'), None):
            with mock.patch.object(students, 'institution_details', return_value=details):
                found, _ = students.candidates([self.paper_data()], self.profile)
            self.assertEqual(found, [])
        with mock.patch.object(students, 'institution_details', return_value=self.details('Paris', 'United States', 'US')):
            found, _ = students.candidates([self.paper_data(country='US')], self.profile)
        self.assertEqual(found, [])

    def test_national_headquarters_are_not_the_researchers_city(self):
        paper = self.paper_data()
        paper['authorships'][0]['institutions'][0].update(display_name='Centre National de la Recherche Scientifique', type='government')
        current = self.latest('https://openalex.org/A1')
        current['institutions'] = paper['authorships'][0]['institutions']
        with mock.patch.object(students, 'latest_affiliations', return_value=current), mock.patch.object(students, 'institution_details', return_value=self.details()):
            self.assertEqual(students.candidates([paper], self.profile)[0], [])
            found, _ = students.candidates([paper], dict(self.profile, mobility='open'))
        self.assertEqual(found[0]['city'], '')
        self.assertFalse(found[0]['location_match'])
        self.assertFalse(found[0]['location_checked'])
        # Previously saved snapshots must not display a headquarters as a laboratory either.
        found[0]['city'] = 'Paris'
        students.context_cards(found, 'en', set())
        self.assertEqual(found[0]['city'], '')

    def test_alternatives_are_explicit_and_homonyms_stay_separate(self):
        papers = [self.paper_data(), self.paper_data('A2', 'I2', 'US')]
        papers[1]['id'] = 'W2'
        def details(iid):
            return self.details() if iid.endswith('I1') else self.details('Austin', 'United States', 'US')
        with mock.patch.object(students, 'institution_details', side_effect=details):
            found, _ = students.candidates(papers, dict(self.profile, mobility='open'))
        self.assertEqual(len(found), 2)
        self.assertTrue(found[0]['location_match'])
        self.assertFalse(found[1]['location_match'])
        self.assertNotEqual(found[0]['key'], found[1]['key'])
        self.assertEqual(found[0]['count'], 1)

    def test_joint_affiliation_counts_once_prefers_lab_and_requires_author_identity(self):
        paper = self.paper_data()
        affiliations = paper['authorships'][0]['institutions']
        affiliations.insert(0, {'id': 'https://openalex.org/I2', 'display_name': 'CNRS', 'country_code': 'FR', 'type': 'government'})
        affiliations.append(affiliations[-1])
        with mock.patch.object(students, 'institution_details', return_value=self.details()):
            found, _ = students.candidates([paper], self.profile)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['institution'], 'Example University')
        self.assertEqual(found[0]['count'], 1)
        del paper['authorships'][0]['author']['id']
        self.assertEqual(students.candidates([paper], self.profile)[0], [])

    def test_institution_lookup_is_bounded_cached_and_cannot_fetch_arbitrary_urls(self):
        with mock.patch.object(students.sources, '_json', return_value=self.details()) as fetch:
            self.assertIsNone(students.institution_details('https://evil.example/I1'))
            first = students.institution_details('https://openalex.org/I1')
            self.assertEqual(students.institution_details('https://openalex.org/I1'), first)
        self.assertEqual(fetch.call_count, 1)
        papers = [self.paper_data('A%d'%i, 'I%d'%i) for i in range(40)]
        with mock.patch.object(students, 'institution_details', return_value=self.details()) as fetch:
            found, limited = students.candidates(papers, self.profile)
        self.assertEqual(fetch.call_count, students.MAX_AUTHORS)
        self.assertEqual(len(found), 10)
        self.assertTrue(limited)
        self.assertEqual(students.safe_url('javascript:alert(1)'), '')
        self.assertEqual(students.safe_url('https://[invalid'), '')

    def test_reading_suggestions_cannot_attach_another_persons_paper(self):
        found = [self.candidate()]
        with mock.patch.object(students.generated, 'chat_json', return_value={'candidates': [{
                'key': found[0]['key'], 'name': 'Invented supervisor', 'email': 'invented@example.com',
                'readings': [{'id': 'invented', 'why': 'Invented result'}, {'id': 'W1', 'why': 'Compare the astrometric method.'}]}]}) as chat:
            students.reading_notes(found, self.profile, 'fr')
        self.assertEqual(found[0]['name'], 'Alex Example')
        self.assertEqual(len(found[0]['readings']), 1)
        self.assertEqual(found[0]['readings'][0]['why'], 'Compare the astrometric method.')
        self.assertNotIn('email', found[0])
        self.assertIn('JSON', chat.call_args.kwargs['system'])
        self.assertIn('availability', chat.call_args.kwargs['system'])

    def test_search_does_not_put_skills_or_dates_into_topical_relevance(self):
        with mock.patch.object(students.teams, 'search_papers', return_value={'title': 'Moons', 'works': [], 'limited': True}) as search, \
             mock.patch.object(students, 'reading_notes') as notes:
            data = students.find(self.profile, 'fr')
        self.assertEqual(search.call_args.args[:2], (self.profile['topic'], 'fr'))
        self.assertTrue(data['limited'])
        self.assertEqual(data['profile']['skills'], 'Python')

    def test_access_scope_private_status_csrf_and_no_get_mutations(self):
        pro = self.user('pro', paid=True)
        row = self.row(pro)
        other = self.client_for(self.user('other', paid=True))
        self.assertEqual(other.get('/en/students/student-demo/').status_code, 404)
        self.assertEqual(other.get('/en/ai/%d/'%row.pk).status_code, 404)
        client = self.client_for(pro)
        self.assertEqual(client.get('/en/students/student-demo/save/').status_code, 405)
        self.assertEqual(client.post('/en/students/student-demo/save/', {'candidate': 'invented'}).status_code, 404)
        self.assertEqual(Client().get('/en/students/student-demo/').status_code, 302)
        free = self.user('free')
        free_row = self.row(free, key='free-map')
        free_client = self.client_for(free)
        self.assertEqual(free_client.get('/en/ai/%d/'%free_row.pk).status_code, 404)
        grant = PremiumGrant.objects.create(user=free, tool='students')
        self.assertEqual(free_client.get('/en/students/free-map/').status_code, 200)
        grant.active = False
        grant.save()
        self.assertEqual(free_client.get('/en/students/free-map/').status_code, 404)
        csrf = Client(enforce_csrf_checks=True)
        csrf.force_login(pro)
        self.assertEqual(csrf.post('/en/students/student-demo/save/', {'candidate': self.candidate()['key']}).status_code, 403)

    def test_favourite_is_idempotent_snapshot_private_editable_and_survives_search_removal(self):
        pro = self.user('pro', paid=True)
        client = self.client_for(pro)
        row = self.row(pro)
        candidate = row.data()['candidates'][0]
        for _ in range(2):
            self.assertEqual(client.post('/en/students/student-demo/save/', {'candidate': candidate['key']}).status_code, 302)
        item = StudentShortlist.objects.get(user=pro)
        row.delete()
        self.assertEqual(item.snapshot['name'], 'Alex Example')
        url = '/en/students/shortlist/%d/'%item.pk
        other = self.client_for(self.user('other', paid=True))
        self.assertEqual(other.post(url, {'action': 'remove'}).status_code, 404)
        self.assertEqual(client.post(url, {'status': 'contacted', 'notes': '<script>private</script>'}).status_code, 302)
        item.refresh_from_db()
        self.assertEqual(item.status, 'contacted')
        page = client.get('/en/students/')
        self.assertContains(page, '&lt;script&gt;private&lt;/script&gt;')
        self.assertNotContains(page, '<script>private</script>')
        self.assertEqual(client.post(url, {'status': 'fake', 'notes': 'x'}).status_code, 400)
        self.assertEqual(client.post(url, {'status': 'explore', 'notes': 'x'*1501}).status_code, 400)
        self.assertEqual(client.post(url, {'action': 'remove'}).status_code, 302)
        self.assertFalse(StudentShortlist.objects.filter(pk=item.pk).exists())

    def test_validation_quota_and_generation(self):
        client = self.client_for(self.user('pro', paid=True))
        self.assertEqual(client.post('/en/students/', dict(self.profile, topic='x')).status_code, 200)
        self.assertFalse(Generated.objects.filter(kind='students').exists())
        with mock.patch.object(students, 'find', return_value={'title': 'Moons', 'profile': self.profile, 'candidates': []}) as find:
            for _ in range(students.PER_DAY):
                response = client.post('/en/students/', self.profile)
                self.assertEqual(response.status_code, 302)
            self.assertContains(client.post('/en/students/', self.profile), 'Daily search limit')
        self.assertEqual(find.call_count, students.PER_DAY)
        self.assertContains(client.get(response['Location']), 'No sufficiently supported lead')

    def test_draft_never_claims_reading_or_availability_and_french_page(self):
        person = self.candidate()
        draft = students.draft(person, 'en')
        self.assertIn('first-author publication', draft)
        self.assertNotIn('I have read', draft)
        self.assertIn('opportunities', draft)
        self.assertIn('6 months', draft)
        self.assertIn('Python', draft)
        self.assertContains(Client().get('/fr/students/'), 'Trouver un stage ou une thèse')
        pro = self.user('pro', paid=True)
        row = self.row(pro)
        self.assertContains(self.client_for(pro).get('/fr/students/student-demo/'), 'Lire avant')

    def test_only_explicit_first_author_papers_are_shown(self):
        first = self.paper_data()
        middle = self.paper_data(title='Unrelated long collaboration')
        middle['id'] = 'W2'
        middle['authorships'][0]['author_position'] = 'middle'
        with mock.patch.object(students, 'institution_details', return_value=self.details()):
            found, _ = students.candidates([middle, first], self.profile)
        self.assertEqual(found[0]['count'], 1)
        self.assertEqual([p['id'] for p in found[0]['readings']], ['W1'])
        self.assertTrue(found[0]['readings'][0]['first_author'])

    def test_new_affiliation_outside_topic_replaces_old_paris_institution(self):
        current = self.latest('https://openalex.org/A1')
        current['institutions'][0].update(id='https://openalex.org/I9', display_name='Toronto University', country_code='CA')
        with mock.patch.object(students, 'latest_affiliations', return_value=current), \
             mock.patch.object(students, 'institution_details', return_value=self.details('Toronto', 'Canada', 'CA')):
            self.assertEqual(students.candidates([self.paper_data()], self.profile)[0], [])
            found, _ = students.candidates([self.paper_data()], dict(self.profile, mobility='open'))
        self.assertEqual(found[0]['institution'], 'Toronto University')
        self.assertEqual(found[0]['city'], 'Toronto')
        self.assertFalse(found[0]['location_match'])
        with mock.patch.object(students, 'latest_affiliations', return_value=None):
            found, limited = students.candidates([self.paper_data()], self.profile)
        self.assertEqual(found, [])
        self.assertTrue(limited)

    def test_refresh_search_is_private_and_updates_saved_evidence_without_losing_notes(self):
        user = self.user('pro', paid=True)
        client = self.client_for(user)
        row = self.row(user)
        person = row.data()['candidates'][0]
        saved = StudentShortlist.objects.create(user=user, candidate_key='old-key', snapshot=dict(person, key='old-key'),
                                                status='contacted', notes='Keep these notes')
        client.post('/en/students/student-demo/save/', {'candidate': person['key']})
        saved.refresh_from_db()
        self.assertEqual(saved.notes, 'Keep these notes')
        self.assertEqual(saved.status, 'contacted')
        self.assertEqual(saved.candidate_key, person['key'])
        with mock.patch.object(students, 'find', return_value={'version': students.RESULT_VERSION, 'title': 'Fresh',
                                                             'profile': self.profile, 'candidates': []}) as find:
            response = client.post('/en/students/student-demo/refresh/')
        self.assertEqual(response.status_code, 302)
        self.assertEqual(find.call_args.args[0]['topic'], self.profile['topic'])
        self.assertTrue(Generated.objects.filter(pk=row.pk).exists())
        other = self.client_for(self.user('other', paid=True))
        self.assertEqual(other.post('/en/students/student-demo/refresh/').status_code, 404)
        self.assertEqual(client.get('/en/students/student-demo/refresh/').status_code, 405)
        self.assertContains(client.get('/en/students/'), 'Save progress and notes')
        response = client.post('/en/students/shortlist/%d/' % saved.pk, {'status': 'read', 'notes': 'Updated'})
        self.assertTrue(response['Location'].endswith('#lead-%d' % saved.pk))
        self.assertContains(client.get(response['Location']), 'Progress and notes saved.')

    def test_old_snapshots_do_not_claim_first_authorship_or_current_city(self):
        person = self.candidate()
        person.pop('version')
        person['readings'][0].pop('first_author')
        students.context_cards([person], 'en', set())
        self.assertTrue(person['outdated'])
        self.assertEqual(person['city'], '')
        self.assertNotIn('first-author publication', person['draft'])


class StudentAffiliationSourceTests(ToolsTestCase):
    aid = 'https://openalex.org/A1'

    def work(self, date, iid=None):
        return {'id': 'https://openalex.org/W1', 'display_name': 'Recent publication outside the topic',
                'publication_date': date, 'authorships': [{'author': {'id': self.aid},
                    'institutions': [{'id': iid, 'display_name': 'Lab', 'type': 'education'}] if iid else []}]}

    def test_newest_publication_is_checked_independently_of_topic_and_cached(self):
        newest = self.work('2026-10-01', 'https://openalex.org/I9')
        previous = self.work('2025-01-01', 'https://openalex.org/I1')
        with mock.patch.object(students.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(students.sources, '_json', side_effect=[{'last_known_institutions': [{'id': 'https://openalex.org/I9'}]},
                                                                       {'results': [previous, newest]}]) as call:
            record = students.latest_affiliations(self.aid)
            again = students.latest_affiliations(self.aid)
        self.assertEqual(record, again)
        self.assertEqual(call.call_count, 2)
        self.assertEqual([i['id'] for i in record['institutions']], ['https://openalex.org/I9'])
        self.assertEqual(record['date'], '2026-10-01')
        self.assertFalse(record['conflict'])
        params = call.call_args.args[1]
        self.assertEqual(params['sort'], 'publication_date:desc')
        self.assertIn('author.id:A1', params['filter'])
        self.assertNotIn('search', params)

    def test_missing_recent_affiliation_never_falls_back_to_old_paris(self):
        with mock.patch.object(students.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(students.sources, '_json', side_effect=[{'last_known_institutions': [{'id': 'https://openalex.org/I1'}]},
                 {'results': [self.work('2026-10-01'), self.work('2025-01-01', 'https://openalex.org/I1')]}]):
            record = students.latest_affiliations(self.aid)
        self.assertEqual(record['institutions'], [])

    def test_disagreement_and_provider_failure_are_explicit(self):
        with mock.patch.object(students.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(students.sources, '_json', side_effect=[{'last_known_institutions': [{'id': 'https://openalex.org/I9'}]},
                 {'results': [self.work('2026-10-01', 'https://openalex.org/I1')]}]):
            self.assertTrue(students.latest_affiliations(self.aid)['conflict'])
        with mock.patch.object(students.sources, '_json') as call:
            self.assertIsNone(students.latest_affiliations('https://evil.example/A1'))
            call.assert_not_called()
        with mock.patch.object(students.sources, 'tool_search_allowed', return_value=False):
            self.assertIsNone(students.latest_affiliations('https://openalex.org/A2'))

    def test_recent_dated_author_history_can_resolve_an_unaffiliated_preprint(self):
        author = {'last_known_institutions': [{'id': 'https://openalex.org/I9'}], 'affiliations': [
            {'institution': {'id': 'https://openalex.org/I1', 'display_name': 'Old Paris lab'}, 'years': [2026]},
            {'institution': {'id': 'https://openalex.org/I9', 'display_name': 'New Toronto lab'}, 'years': [2026]}]}
        with mock.patch.object(students.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(students.sources, '_json', side_effect=[author, {'results': [self.work('2026-10-01')]}]):
            record = students.latest_affiliations(self.aid)
        self.assertEqual(record['source'], 'author_record')
        self.assertEqual([i['id'] for i in record['institutions']], ['https://openalex.org/I9'])
        self.assertEqual(record['date'], '2026')
