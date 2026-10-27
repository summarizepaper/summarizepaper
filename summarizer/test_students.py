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
        papers = [dict(self.paper_data('A%d'%i, 'I%d'%i), id='W%d' % i, doi='https://doi.org/10.1234/unique%d' % i) for i in range(40)]
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
        page = client.get('/en/students/leads/')
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
                response = client.post('/en/students/', dict(self.profile, country='FR'))
                self.assertEqual(response.status_code, 302)
            self.assertContains(client.post('/en/students/', dict(self.profile, country='FR')), 'Daily search limit')
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
        self.assertNotContains(Client().get('/fr/students/'), 'student-compare')
        pro = self.user('pro', paid=True)
        row = self.row(pro)
        self.assertContains(self.client_for(pro).get('/fr/students/student-demo/'), 'Lire avant')

    def test_first_author_readings_are_prioritised_and_coauthors_labelled(self):
        first = self.paper_data()
        middle = self.paper_data(title='Unrelated long collaboration')
        middle['id'] = 'W2'
        middle['authorships'][0]['author_position'] = 'middle'
        with mock.patch.object(students, 'institution_details', return_value=self.details()):
            found, _ = students.candidates([middle, first], self.profile)
        self.assertEqual(found[0]['count'], 2)
        self.assertEqual(found[0]['first_count'], 1)
        self.assertEqual([p['id'] for p in found[0]['readings']], ['W1', 'W2'])
        self.assertFalse(found[0]['readings'][1]['first_author'])
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
        self.assertContains(client.get('/en/students/leads/'), 'Save progress and notes')
        response = client.post('/en/students/shortlist/%d/' % saved.pk, {'status': 'read', 'notes': 'Updated'})
        self.assertTrue(response['Location'].endswith('#lead-%d' % saved.pk))
        self.assertContains(client.get(response['Location']), 'Progress and notes saved.')

    def test_old_snapshots_do_not_claim_first_authorship_or_current_city(self):
        person = self.candidate()
        person.pop('version')
        person['readings'][0].pop('first_author')
        person['readings'][0].pop('roles')  # older snapshots knew neither
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


class StudentWorkflowTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(students, "latest_affiliations", side_effect=self.latest)
        patch.start()
        self.addCleanup(patch.stop)
    latest = StudentTests.latest
    profile = StudentTests.profile
    paper_data = StudentTests.paper_data
    details = StudentTests.details
    candidate = StudentTests.candidate
    def test_autosave_revision_prevents_lost_updates_and_reminders_are_private(self):
        user = self.user('workflow', paid=True)
        item = StudentShortlist.objects.create(user=user, candidate_key='one', snapshot=self.candidate())
        client = self.client_for(user)
        url = '/en/students/shortlist/%d/' % item.pk
        fields = {'status': 'contacted', 'notes': 'Private draft', 'remind_on': '2027-01-10', 'revision': 0}
        response = client.post(url, fields, HTTP_ACCEPT='application/json')
        self.assertEqual(response.json(), {'revision': 1})
        self.assertEqual(client.post(url, dict(fields, notes='Stale'), HTTP_ACCEPT='application/json').status_code, 409)
        item.refresh_from_db()
        self.assertEqual(item.notes, 'Private draft')
        self.assertEqual(item.remind_on.isoformat(), '2027-01-10')
        self.assertEqual(client.post(url, dict(fields, revision='bad'), HTTP_ACCEPT='application/json').status_code, 400)
        other = self.client_for(self.user('outsider', paid=True))
        self.assertEqual(other.post(url, fields, HTTP_ACCEPT='application/json').status_code, 404)
        self.assertEqual(client.post(url, dict(fields, revision=1, remind_on='invalid'), HTTP_ACCEPT='application/json').status_code, 400)
        self.assertContains(client.get('/en/students/leads/'), 'Changes are saved automatically.')

    def test_comparison_filters_due_and_ownership(self):
        user = self.user('compare', paid=True)
        client = self.client_for(user)
        first = StudentShortlist.objects.create(user=user, candidate_key='one', snapshot=dict(self.candidate(), name='Private One'),
                                                remind_on=timezone.localdate(), status='contacted')
        second = StudentShortlist.objects.create(user=user, candidate_key='two', snapshot=dict(self.candidate(), name='Private Two'))
        own = '/en/students/compare/?lead=%d&lead=%d' % (first.pk, second.pk)
        self.assertContains(client.get(own), 'Private One')
        other = self.client_for(self.user('othercompare', paid=True))
        self.assertEqual(other.get(own).status_code, 404)
        self.assertEqual(client.get('/en/students/compare/?lead=%d&lead=%d' % (first.pk, first.pk)).status_code, 400)
        self.assertEqual(client.get('/en/students/compare/?lead=not-a-number&lead=1').status_code, 400)
        self.assertNotContains(client.get('/en/students/leads/?status=contacted&due=1'), 'Private Two')
        self.assertContains(client.get('/en/students/leads/?status=contacted&due=1'), 'Reminder due')

    def test_source_import_is_owned_scoped_and_preserves_notes(self):
        user = self.user('sources', paid=True)
        client = self.client_for(user)
        item = StudentShortlist.objects.create(user=user, candidate_key='one', snapshot=self.candidate(), notes='Keep')
        url = '/en/students/shortlist/%d/source/' % item.pk
        self.assertEqual(client.post(url, {'url': 'https://evil.test/profile'}).status_code, 400)
        evidence = {'profiles': [], 'offers': [], 'checked': '2026-10-23', 'partial': False}
        with mock.patch.object(students.student_sources, 'inspect', return_value=evidence) as inspect:
            self.assertEqual(client.post(url, {'url': 'https://example.edu/profile'}).status_code, 200)
            self.assertEqual(client.post(url, {'url': 'https://example.edu/profile'}).status_code, 429)
        item.refresh_from_db()
        self.assertEqual(item.notes, 'Keep')
        self.assertEqual(item.revision, 1)
        self.assertEqual(item.snapshot['official_sources']['checked'], '2026-10-23')
        self.assertEqual(inspect.call_count, 1)
        other = self.client_for(self.user('source-other', paid=True))
        self.assertEqual(other.post(url, {'url': 'https://example.edu/'}).status_code, 404)

    def test_source_validation_preserves_input_and_allows_correction(self):
        user = self.user('source-retry', paid=True)
        client = self.client_for(user)
        item = StudentShortlist.objects.create(user=user, candidate_key='retry', snapshot=self.candidate(), notes='Keep')
        url = '/fr/students/shortlist/%d/source/' % item.pk
        with mock.patch.object(students.student_sources, 'inspect') as inspect:
            response = client.post(url, {'url': 'https://wrong.example/profile?a=1&b=2'})
            self.assertContains(response, 'example.edu', status_code=400)
            self.assertContains(response, 'value="https://wrong.example/profile?a=1&amp;b=2"', status_code=400)
            self.assertContains(response, 'data-student-source', status_code=400)
            self.assertContains(response, 'Cette adresse', status_code=400)
            inspect.assert_not_called()
            inspect.return_value = {'profiles': [], 'offers': [], 'checked': '2026-10-23', 'partial': False}
            self.assertEqual(client.post(url, {'url': '  example.edu/profile  '}).status_code, 200)
            inspect.assert_called_once_with(item.snapshot, 'https://example.edu/profile')
        item.refresh_from_db()
        self.assertEqual(item.notes, 'Keep')
        self.assertEqual(item.revision, 1)

    def test_source_missing_institution_website_explains_refresh(self):
        user = self.user('source-no-site', paid=True)
        item = StudentShortlist.objects.create(user=user, candidate_key='missing', snapshot=dict(self.candidate(), website=''))
        response = self.client_for(user).post('/en/students/shortlist/%d/source/' % item.pk, {'url': 'https://example.edu'})
        self.assertContains(response, 'no usable institution website', status_code=400)
        self.assertNotContains(response, 'data-student-source', status_code=400)

    def test_reading_methods_require_literal_evidence_and_keep_first_authors(self):
        person = self.candidate()
        person['readings'][0]['abstract'] = 'We use optical interferometry to detect astrometric signals.'
        answer = {'candidates': [{'key': person['key'], 'readings': [{'id': 'W1', 'why': 'Relevant',
            'concepts': ['Astrometry'], 'questions': ['How is the signal measured?'],
            'methods': [{'name': 'Interferometry', 'quote': 'optical interferometry to detect'},
                        {'name': 'Invented', 'quote': 'a completely invented method'}]}]}]}
        with mock.patch.object(students.generated, 'chat_json', return_value=answer):
            students.reading_notes([person], self.profile, 'en')
        self.assertEqual(len(person['readings'][0]['methods']), 1)
        self.assertEqual(person['readings'][0]['questions'], ['How is the signal measured?'])


class StudentPublicSourceTests(ToolsTestCase):
    def test_pasted_addresses_are_normalized_without_relaxing_domain_checks(self):
        from . import student_sources as public
        for raw in (' example.edu/profile ', 'http://example.edu/profile', '//example.edu/profile', 'https://example.edu/profile'):
            self.assertEqual(public.normalize_url(raw), 'https://example.edu/profile')
            self.assertTrue(public.official(public.normalize_url(raw), 'https://example.edu'))
        for raw in ('javascript:alert(1)', 'https://example.edu.evil.test', 'https://user:pw@example.edu',
                    'https://example.edu:444/a', 'https://example.edu/\npage', 'https://example.edu\\@evil.test'):
            self.assertFalse(public.official(public.normalize_url(raw), 'https://example.edu'))

    def test_institution_domain_boundary_and_private_ips(self):
        from . import student_sources as public
        for url in ('https://example.edu.evil.test/a', 'http://example.edu/a', 'https://user:pw@example.edu/a',
                    'https://example.edu:444/a', 'https://[invalid'):
            self.assertFalse(public.official(url, 'https://example.edu'))
        self.assertTrue(public.official('https://staff.example.edu/a', 'https://www.example.edu'))
        self.assertTrue(public.official('https://example.edu/a', 'http://example.edu/'))
        for address in ('127.0.0.1', '10.0.0.1', '169.254.169.254', '::1', '::ffff:127.0.0.1'):
            with mock.patch.object(public.socket, 'getaddrinfo', return_value=[(0, 0, 0, '', (address, 443))]), \
                 mock.patch.object(public, 'PinnedHTTPS') as connect:
                with self.assertRaises(public.Unavailable):
                    public.fetch('https://example.edu/', 'https://example.edu')
                connect.assert_not_called()

    def test_redirect_is_revalidated_and_connection_is_pinned(self):
        from . import student_sources as public
        response = mock.Mock(status=302)
        response.getheader.return_value = 'https://127.0.0.1/private'
        with mock.patch.object(public.socket, 'getaddrinfo', return_value=[(0, 0, 0, '', ('93.184.216.34', 443))]), \
             mock.patch.object(public, 'PinnedHTTPS') as connect:
            connect.return_value.getresponse.return_value = response
            with self.assertRaises(public.Unavailable):
                public.fetch('https://example.edu/', 'https://example.edu')
            connect.assert_called_once_with('example.edu', '93.184.216.34')
            connect.return_value.close.assert_called_once()

    def test_structured_profiles_and_offers_are_not_invented_or_expired(self):
        from . import student_sources as public
        data = {'@graph': [
            {'@type': 'Person', 'name': 'Alex Example', 'worksFor': {'name': 'Other University'}, 'jobTitle': 'Researcher'},
            {'@type': 'Person', 'name': 'Namesake Different', 'worksFor': {'name': 'Example University'}},
            {'@type': 'JobPosting', 'title': 'PhD on exomoons', 'validThrough': '2099-01-01', 'educationRequirements': 'Master'},
            {'@type': 'JobPosting', 'title': 'PhD expired', 'validThrough': '2000-01-01'},
            {'@type': 'JobPosting', 'title': 'Accountant'},
            {'@type': 'JobPosting', 'title': 'PhD invalid date', 'validThrough': '2026-99-99'}]}
        page = public.Page('<script type="application/ld+json">'+json.dumps(data)+'</script>')
        profiles, offers = public.records(page, 'https://example.edu/jobs', {'name': 'Alex Example', 'institution': 'Example University'})
        self.assertEqual(len(profiles), 1)
        self.assertFalse(profiles[0]['agrees'])
        self.assertEqual(len(offers), 1)
        self.assertEqual(offers[0]['level'], 'Master')
        self.assertEqual(offers[0]['funding'], '')
        self.assertEqual(public.records(public.Page('<p>PhD vacancies</p>'), 'https://example.edu/', {'name': 'Alex', 'institution': 'Example'})[1], [])

    def test_crawl_has_a_four_page_limit_and_never_follows_external_links(self):
        from . import student_sources as public
        page = public.Page(''.join('<a href="/people/%d">Profile</a>' % i for i in range(20))+'<a href="https://evil.test/jobs">Jobs</a>')
        with mock.patch.object(public, 'fetch', return_value=(page, 'https://example.edu/')) as fetch:
            evidence = public.inspect({'name': 'Alex Example', 'institution': 'Example', 'website': 'https://example.edu/'})
        self.assertEqual(fetch.call_count, 4)
        self.assertTrue(evidence['partial'])
        self.assertTrue(all(call.args[0].startswith('https://example.edu') for call in fetch.call_args_list))


class StudentReadingDiscoveryTests(ToolsTestCase):
    def work(self, key, title='Observing planetary gaps in the gas of debris disks', aid='A1', position='first', doi=None):
        return {'id': 'https://openalex.org/'+key, 'display_name': title, 'doi': doi or 'https://doi.org/10.1234/'+key,
                'publication_date': '2020-01-01', 'authorships': [
                    {'author': {'id': 'https://openalex.org/'+aid, 'display_name': 'Example'}, 'author_position': position}]}

    def test_preprint_and_publication_are_one_reading_with_both_links(self):
        from . import student_papers as papers
        preprint = self.work('W1', doi='https://doi.org/10.48550/arXiv.2411.14241')
        published = self.work('W2', doi='https://doi.org/10.1051/0004-6361/202452097')
        preprint['publication_date'] = published['publication_date'] = '2024-11-21'
        for inputs in ([preprint, published], [published, preprint]):
            result = papers.deduplicate(inputs)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0]['id'], published['id'])
            self.assertEqual(result[0]['alternate_links'], [{'url': 'https://arxiv.org/abs/2411.14241', 'kind': 'arxiv'}])
        old = [papers.reading(w, 'https://openalex.org/A1') for w in [preprint, published]]
        person = {'key': 'lead', 'author_url': 'https://openalex.org/A1', 'version': 3, 'readings': old}
        students.prepare_card(person)
        self.assertEqual(len(person['readings']), 1)  # Cached searches are corrected at display time too.

    def test_generic_titles_different_authors_and_distinct_published_dois_are_kept(self):
        from . import student_papers as papers
        first = self.work('W1')
        second = self.work('W2')
        self.assertEqual(len(papers.deduplicate([first, second])), 2)
        second.update(doi='https://doi.org/10.48550/arXiv.2411.14241')
        second['authorships'][0]['author']['id'] = 'https://openalex.org/A2'
        self.assertEqual(len(papers.deduplicate([first, second])), 2)
        second['authorships'] = first['authorships']
        second['publication_date'] = '2010-01-01'
        self.assertEqual(len(papers.deduplicate([first, second])), 2)

    def test_author_lookup_is_not_restricted_to_recent_topic_results_and_checks_exact_order(self):
        from . import student_papers as papers
        works = [self.work('W1'), self.work('W2', position='middle'), self.work('W3', aid='A2'),
                 self.work('W4', title='VizieR Online Data Catalog: cold-gas-bearing debris-disc stars'),
                 self.work('W5', title='The co-existence of gas (Corrigendum)')]
        with mock.patch.object(papers.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(papers.sources, '_json', return_value={'meta': {'count': 5}, 'results': works}) as fetch:
            found = papers.author_publications('https://openalex.org/A1')
            self.assertEqual(found, papers.author_publications('https://openalex.org/A1'))
        self.assertEqual([w['id'] for w in found['works']], ['https://openalex.org/W1'])
        self.assertEqual(fetch.call_count, 1)
        self.assertNotIn('from_publication_date', fetch.call_args.args[1]['filter'])
        self.assertIn('author.id:A1', fetch.call_args.args[1]['filter'])
        self.assertFalse(found['limited'])

    def test_large_bibliographies_and_failures_are_explicitly_partial_and_bounded(self):
        from . import student_papers as papers
        with mock.patch.object(papers.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(papers.sources, '_json', return_value={'meta': {'count': 1000}, 'results': [self.work('W1')]}) as fetch:
            found = papers.author_publications('https://openalex.org/A1')
        self.assertEqual(fetch.call_count, 3)
        self.assertTrue(found['limited'])
        self.assertEqual(len(found['works']), 1)
        with mock.patch.object(papers.sources, 'tool_search_allowed', return_value=False), mock.patch.object(papers.sources, '_json') as fetch:
            found = papers.author_publications('https://openalex.org/A9')
        self.assertTrue(found['limited'])
        fetch.assert_not_called()
        self.assertEqual(papers.author_publications('https://evil.test/A1')['works'], [])

    def test_older_relevant_first_author_papers_replace_recent_coauthored_readings(self):
        import numpy as np
        from . import student_papers as papers
        works = [self.work('W%d' % i, title='Relevant exomoon method %d' % i) for i in range(1, 4)]
        works.append(self.work('W4', title='Unrelated subject'))
        record = {'works': works, 'limited': False}
        found = [{'name': 'Example', 'author_url': 'https://openalex.org/A1', 'readings': [
            {'id': 'https://openalex.org/W9', 'title': 'Recent coauthored result', 'year': '2026', 'first_author': False}]}]
        with mock.patch.object(papers, 'author_publications', return_value=record), \
             mock.patch.object(papers.similar, 'embed', return_value=np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[0.,1.]])), \
             mock.patch.object(papers, 'topical', side_effect=lambda inputs, topic: inputs) as screen:
            papers.complete(found, {'topic': 'Exomoon detection'})
        self.assertEqual(len(found[0]['readings']), 3)
        self.assertTrue(all(p['first_author'] for p in found[0]['readings']))
        self.assertEqual(found[0]['first_count'], 3)
        self.assertTrue(all(p['year']=='2020' for p in found[0]['readings']))
        self.assertEqual(len(screen.call_args.args[0]), 3)

    def test_author_relevance_uses_abstracts_and_rejects_unknown_ids(self):
        from . import student_papers as papers
        work = self.work('W1', title='Observations of Beta Pictoris')
        work['abstract_inverted_index'] = {'Debris': [0], 'disk': [1], 'observations': [2]}
        with mock.patch.object(papers.generated, 'chat_json', return_value={'relevant_ids': [work['id'], 'invented']}) as chat:
            self.assertEqual(papers.topical([work], 'Debris disks'), [work])
        payload = json.loads(chat.call_args.args[0])
        self.assertEqual(payload['papers'][0]['abstract'], 'Debris disk observations')

    def test_discovery_failure_keeps_existing_readings_and_marks_them_partial(self):
        import requests
        from . import student_papers as papers
        old = {'id': 'W9', 'title': 'Existing reading', 'first_author': False}
        found = [{'author_url': 'https://openalex.org/A1', 'readings': [old]}]
        with mock.patch.object(papers, 'author_publications', return_value={'works': [self.work('W1')], 'limited': False}), \
             mock.patch.object(papers.similar, 'embed', side_effect=requests.ConnectionError):
            papers.complete(found, {'topic': 'Debris disks'})
        self.assertEqual(found[0]['readings'][0]['title'], 'Existing reading')
        self.assertTrue(found[0]['first_author_limited'])

    def test_empty_questions_and_unsupported_personal_descriptions_are_discarded(self):
        person = {'key': 'lead', 'name': 'Example', 'readings': [{'id': 'W1', 'title': 'A study', 'first_author': True}]}
        answer = {'candidates': [{'key': 'lead',
            'portrait': {'text': 'Unsupported biography', 'paper_ids': ['invented']},
            'fit': {'text': 'Methods relevant to astrometry.', 'paper_ids': ['W1']},
            'readings': [{'id': 'W1', 'questions': [' ', '-', '***', 'How is the signal measured?']}]}]}
        with mock.patch.object(students.generated, 'chat_json', return_value=answer):
            students.reading_notes([person], {'topic': 'Astrometry'}, 'en')
        self.assertNotIn('portrait', person)
        self.assertEqual(person['fit']['paper_ids'], ['W1'])
        self.assertEqual(person['readings'][0]['questions'], ['How is the signal measured?'])

    def test_international_headquarters_are_not_a_laboratory_location(self):
        self.assertTrue(students.umbrella({'display_name': 'European Space Agency', 'type': 'facility'}))
        self.assertFalse(students.location_matches({'country': 'France','city':'Paris'},
            {'display_name': 'European Space Agency'}, {'geo': {'city':'Paris'},'country_code':'FR'}))


class StudentRoleAndRankingTests(ToolsTestCase):
    """Who signs what, who is shown first, and what is said about it."""
    def setUp(self):
        super().setUp()
        patch = mock.patch.object(students, 'latest_affiliations', side_effect=self.latest)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(students, 'institution_details', return_value=StudentTests.details(None))
        patch.start()
        self.addCleanup(patch.stop)

    profile = StudentTests.profile

    def latest(self, aid):
        return StudentTests.latest(self, aid)

    def aid(self, name):
        import hashlib
        return 'https://openalex.org/A%d' % int(hashlib.md5(name.encode()).hexdigest()[:8], 16)

    def work(self, key, names, positions=None, corresponding=(), date='2026-03-01', title='Exomoon astrometry'):
        positions = positions or ['first'] + ['middle'] * (len(names) - 2) + ['last'] * (len(names) > 1)
        return {'id': 'https://openalex.org/' + key, 'display_name': title, 'publication_date': date,
                'doi': 'https://doi.org/10.1/' + key, 'authorships': [
                    {'author': {'id': self.aid(name), 'display_name': name}, 'author_position': position,
                     'is_corresponding': i in corresponding,
                     'institutions': [{'id': 'https://openalex.org/I1', 'display_name': 'Example University', 'country_code': 'FR', 'type': 'education'}]}
                    for i, (name, position) in enumerate(zip(names, positions))]}

    def test_roles_follow_the_signature_but_not_alphabetical_lists_or_large_collaborations(self):
        from . import student_papers as papers
        ordered = self.work('W1', ['Zoe Zed', 'Ann Abel', 'Max Moe'], corresponding=(1,))
        roles = papers.author_roles(ordered)
        self.assertEqual(roles[self.aid('Zoe Zed')], ['first'])
        self.assertEqual(roles[self.aid('Ann Abel')], ['corresponding'])
        self.assertEqual(roles[self.aid('Max Moe')], ['last'])
        alphabetical = self.work('W2', ['Ann Abel', 'Bob Bell', 'Cy Cole', 'Dan Dorn'])
        self.assertTrue(papers.alphabetical(alphabetical))
        self.assertEqual(set(map(tuple, papers.author_roles(alphabetical).values())), {()})
        names = ['Person %s Name%d' % (chr(90 - i % 26), i) for i in range(40)]
        roles = papers.author_roles(self.work('W3', names, corresponding=(30,)))
        self.assertIn(self.aid(names[0]), roles)
        self.assertNotIn(self.aid(names[20]), roles)  # a middle author of a collaboration is not a contact
        self.assertEqual(roles[self.aid(names[30])], ['corresponding'])
        self.assertEqual(roles[self.aid(names[39])], ['last'])

    def test_the_last_author_of_relevant_papers_comes_before_a_middle_author(self):
        papers = [self.work('W1', ['Pat Student', 'Kim Middle', 'Lee Leader']),
                  self.work('W2', ['Sam Other', 'Kim Middle', 'Lee Leader'], date='2025-01-01')]
        found, _ = students.candidates(papers, dict(self.profile, city=''), scores={'https://openalex.org/W1': 0.9, 'https://openalex.org/W2': 0.5})
        names = [p['name'] for p in found]
        self.assertLess(names.index('Lee Leader'), names.index('Kim Middle'))
        leader = found[names.index('Lee Leader')]
        self.assertEqual(leader['role_counts'], {'first': 0, 'last': 2, 'corresponding': 0})
        self.assertEqual(leader['years'], ['2025', '2026'])
        self.assertEqual(leader['readings'][0]['roles'], ['last'])
        self.assertEqual(found[names.index('Kim Middle')]['readings'][0]['roles'], [])

    def test_people_signing_together_are_one_team_but_a_collaboration_is_not(self):
        papers = [self.work('W1', ['Pat Student', 'Lee Leader']), self.work('W2', ['Lee Leader', 'Pat Student'], positions=['first', 'last']),
                  self.work('W3', ['Solo Person'])]
        found, _ = students.candidates(papers, dict(self.profile, city=''))
        by_name = {p['name']: p for p in found}
        self.assertEqual(by_name['Pat Student']['team_id'], by_name['Lee Leader']['team_id'])
        self.assertEqual(by_name['Lee Leader']['team'], [{'key': by_name['Pat Student']['key'], 'name': 'Pat Student', 'shared': 2}])
        self.assertIsNone(by_name['Solo Person']['team_id'])
        rows = students.overview(found)
        self.assertEqual(sum(r['team_start'] for r in rows), 1)
        self.assertEqual(len(rows), 3)

    def test_unchecked_authors_are_counted_and_open_mode_keeps_unknown_locations(self):
        papers = [dict(StudentTests.paper_data(self, 'A%d' % i, 'I%d' % i), id='W%d' % i, doi='https://doi.org/10.1/u%d' % i) for i in range(30)]
        report = {}
        with mock.patch.object(students, 'institution_details', return_value=None):
            found, limited = students.candidates(papers, self.profile, report=report)
            self.assertEqual(found, [])
            self.assertEqual(report['unchecked'], 30)
            found, _ = students.candidates(papers[:2], dict(self.profile, mobility='open'))
        self.assertEqual(len(found), 2)
        self.assertFalse(found[0]['location_checked'])
        self.assertFalse(found[0]['location_match'])

    def test_funding_comes_only_from_recent_small_papers_the_person_leads(self):
        from . import student_papers as papers
        award = {'funder_display_name': 'Agence Nationale de la Recherche', 'funder_award_id': 'ANR-22-CE31-0020', 'display_name': 'Moons'}
        led = dict(self.work('W1', ['Lee Leader', 'Pat Student'], positions=['last', 'first']), awards=[award])
        middle = dict(self.work('W2', ['Lee Leader', 'Pat Student', 'Sam Other'], positions=['middle', 'first', 'last']),
                      awards=[dict(award, funder_award_id='OTHER')])
        old = dict(self.work('W3', ['Lee Leader'], date='2015-01-01'), awards=[dict(award, funder_award_id='OLD')])
        big = dict(self.work('W4', ['Lee Leader'] + ['Person %d' % i for i in range(20)]), awards=[dict(award, funder_award_id='BIG')])
        found = papers.funding([led, middle, old, big], self.aid('Lee Leader'))
        self.assertEqual([f['award'] for f in found], ['ANR-22-CE31-0020'])
        self.assertEqual(found[0]['paper']['title'], 'Exomoon astrometry')

    def test_skills_written_in_the_readings_move_a_person_up(self):
        person = {'readings': [{'title': 'Exomoons', 'abstract': 'We fit transit timing with MCMC sampling.'}]}
        self.assertEqual(students.found_skills(person, {'skills': 'Python, MCMC; transit timing'}), ['MCMC', 'transit timing'])
        first = {'name': 'B', 'score': 1.0, 'location_match': True, 'skills_found': ['MCMC', 'transit timing']}
        second = {'name': 'A', 'score': 1.5, 'location_match': True, 'skills_found': []}
        self.assertEqual(sorted([second, first], key=students.rank)[0]['name'], 'B')

    def test_career_ignores_impossible_years(self):
        author = {'works_count': 12, 'counts_by_year': [{'year': 10000, 'works_count': 3}, {'year': 2009, 'works_count': 1},
                                                         {'year': 1850, 'works_count': 1}, {'year': 2004, 'works_count': 0}]}
        self.assertEqual(students.career(author), (2009, 12))
        self.assertEqual(students.career({}), (None, None))

    def test_draft_names_the_role_and_offers_follow_level_and_country(self):
        person = {'name': 'Lee Leader', 'institution': 'Example University', 'profile': self.profile,
                  'readings': [{'title': 'Exomoon astrometry', 'roles': ['last']}]}
        self.assertIn('which you signed as last author', students.draft(person, 'en'))
        self.assertNotIn('first-author', students.draft(person, 'en'))
        names = [p['name'] for p in students.offer_searches({'profile': dict(self.profile, kind='phd', country='FR'), 'queries': ['exomoon astrometry']})]
        self.assertEqual(names, ['Euraxess', 'FindAPhD', 'ADUM', 'CNRS'])
        internship = students.offer_searches({'profile': dict(self.profile, kind='internship', country='GB'), 'title': 'Lunes'})
        self.assertEqual([p['name'] for p in internship], ['Euraxess', 'jobs.ac.uk'])
        self.assertTrue(internship[0]['url'].endswith('keywords=Lunes'))

    def test_reading_keys_open_the_summary_and_chat_pages(self):
        from . import student_papers as papers
        self.assertEqual(papers.site_key({'url': 'https://doi.org/10.1051/0004-6361/202452097',
                                          'alternate_links': [{'url': 'https://arxiv.org/abs/2411.14241', 'kind': 'arxiv'}]}), '2411.14241')
        self.assertEqual(papers.site_key({'url': 'https://doi.org/10.1093/mnras/stad1234'}), 'doi_10.1093--mnras--stad1234')
        self.assertEqual(papers.site_key({'url': 'https://arxiv.org/abs/hep-th/9901001'}), 'hep-th--9901001')
        self.assertEqual(papers.site_key({'url': ''}), '')


class StudentLeadsPageTests(ToolsTestCase):
    """The page of saved leads: drafts, reminders, next steps and links to the other tools."""
    def setUp(self):
        super().setUp()
        self.owner = self.user('owner', paid=True)
        self.client = self.client_for(self.owner)
        with mock.patch.object(students, 'latest_affiliations', side_effect=lambda aid: StudentTests.latest(self, aid)):
            self.person = StudentTests.candidate(self)
        self.item = StudentShortlist.objects.create(user=self.owner, candidate_key=self.person['key'], snapshot=self.person)
        self.url = '/en/students/shortlist/%d/' % self.item.pk

    profile = StudentTests.profile
    paper_data = StudentTests.paper_data
    details = StudentTests.details

    def test_saved_draft_is_kept_and_a_form_without_it_does_not_erase_it(self):
        fields = {'status': 'read', 'notes': 'n', 'remind_on': '', 'revision': 0, 'draft': 'My own email'}
        self.assertEqual(self.client.post(self.url, fields, HTTP_ACCEPT='application/json').json(), {'revision': 1})
        fields.pop('draft')
        self.client.post(self.url, dict(fields, revision=1), HTTP_ACCEPT='application/json')
        self.item.refresh_from_db()
        self.assertEqual(self.item.draft, 'My own email')
        self.assertContains(self.client.get('/en/students/leads/'), 'My own email')
        self.assertContains(self.client.get('/en/students/leads/'), 'form="progress-%d"' % self.item.pk)

    def test_saving_a_lead_keeps_an_edited_draft_only(self):
        row = Generated.objects.create(user=self.owner, kind='students', key='search', lang='en', status='done', content=json.dumps({
            'version': students.RESULT_VERSION, 'title': 'Exomoons', 'profile': self.profile, 'candidates': [self.person]}))
        self.item.delete()
        unchanged = students.draft(dict(self.person), 'en')
        self.client.post('/en/students/search/save/', {'candidate': self.person['key'], 'draft': unchanged})
        self.assertEqual(StudentShortlist.objects.get(user=self.owner).draft, '')
        self.client.post('/en/students/search/save/', {'candidate': self.person['key'], 'draft': 'Edited'})
        self.assertEqual(StudentShortlist.objects.get(user=self.owner).draft, 'Edited')
        page = self.client.get('/en/students/search/')
        self.assertContains(page, 'Edited')
        self.assertContains(page, 'At a glance')
        self.assertContains(page, 'euraxess.ec.europa.eu')
        self.assertContains(page, '/en/teams/?topic=')

    def test_email_reminders_are_sent_once_per_date_and_only_when_asked(self):
        from django.core import mail
        today = timezone.now()
        StudentShortlist.objects.filter(pk=self.item.pk).update(remind_on=today.date(), email_reminder=True, notes='Follow up on Monday')
        StudentShortlist.objects.create(user=self.owner, candidate_key='silent', snapshot=dict(self.person, name='Silent Lead'),
                                        remind_on=today.date())
        now = today.replace(hour=8)
        self.assertEqual(students.send_reminders(now, 10), 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('Alex Example', mail.outbox[0].body)
        self.assertIn('Follow up on Monday', mail.outbox[0].body)
        self.assertNotIn('Silent Lead', mail.outbox[0].body)
        self.assertEqual(students.send_reminders(now, 10), 0)
        # A new date is a new reminder.
        fields = {'status': 'contacted', 'notes': '', 'remind_on': (today.date() + timezone.timedelta(days=1)).isoformat(),
                  'email_reminder': 'on', 'revision': 0}
        self.client.post(self.url, fields, HTTP_ACCEPT='application/json')
        self.item.refresh_from_db()
        self.assertIsNone(self.item.reminded)
        self.assertTrue(self.item.email_reminder)
        self.assertEqual(students.send_reminders(now + timezone.timedelta(days=1), 10), 1)
        self.assertEqual(students.send_reminders(now.replace(hour=3), 10), 0)

    def test_follow_in_the_digest_and_without_one(self):
        from summarizer.models import DigestSubscription
        follow = '/en/students/shortlist/%d/follow/' % self.item.pk
        self.assertTrue(self.client.post(follow)['Location'].endswith('/en/digest/'))
        subscription = DigestSubscription.objects.create(user=self.owner, categories='astro-ph.EP', authors='Ann Lee')
        self.client.post(follow)
        subscription.refresh_from_db()
        self.assertEqual(subscription.authors, 'Ann Lee, Alex Example')
        self.client.post(follow)
        subscription.refresh_from_db()
        self.assertEqual(subscription.authors, 'Ann Lee, Alex Example')
        page = self.client.get('/en/students/leads/')
        self.assertContains(page, 'Their new arXiv papers come first in your digest.')
        other = self.client_for(self.user('intruder', paid=True))
        self.assertEqual(other.post(follow).status_code, 404)

    def test_readings_go_to_a_collection(self):
        from summarizer.models import Collection
        snapshot = dict(self.person, readings=[dict(self.person['readings'][0], url='https://arxiv.org/abs/2101.00001')])
        StudentShortlist.objects.filter(pk=self.item.pk).update(snapshot=snapshot)
        with mock.patch('summarizer.library.add_many') as add:
            response = self.client.post('/en/students/shortlist/%d/collection/' % self.item.pk, {'collection': 'new'})
        collection = Collection.objects.get(user=self.owner)
        self.assertEqual(response['Location'], '/en/collections/%d/' % collection.pk)
        self.assertEqual(collection.name, 'Placement: Alex Example')
        self.assertEqual(add.call_args.args[1], ['2101.00001'])

    def test_checking_a_lead_again_updates_its_sources_but_keeps_notes(self):
        StudentShortlist.objects.filter(pk=self.item.pk).update(notes='Keep me')
        latest = dict(StudentTests.latest(self, self.person['author_url']), since=2012, works_count=40)
        work = {'id': 'https://openalex.org/W8', 'display_name': 'A newer paper', 'publication_date': '2099-01-01',
                'doi': 'https://doi.org/10.1/new', 'authorships': [{'author': {'id': self.person['author_url'], 'display_name': 'Alex Example'},
                                                                    'author_position': 'first'}]}
        with mock.patch.object(students, 'latest_affiliations', return_value=latest), \
             mock.patch.object(students, 'institution_details', return_value=self.details()), \
             mock.patch.object(students.student_papers, 'author_publications', return_value={'works': [work], 'limited': False}):
            response = self.client.post('/en/students/shortlist/%d/update/' % self.item.pk)
        self.assertTrue(response['Location'].endswith('#lead-%d' % self.item.pk))
        self.item.refresh_from_db()
        self.assertEqual(self.item.notes, 'Keep me')
        self.assertEqual(self.item.snapshot['since'], 2012)
        self.assertEqual([p['title'] for p in self.item.snapshot['new_papers']], ['A newer paper'])
        page = self.client.get('/en/students/leads/')
        self.assertContains(page, 'This lead was checked again')
        self.assertContains(page, 'A newer paper')
        with mock.patch.object(students, 'latest_affiliations', return_value=None):
            self.client.post('/en/students/shortlist/%d/update/' % self.item.pk)
        self.assertContains(self.client.get('/en/students/leads/'), 'could not be checked again')

    def test_conference_talks_of_a_lead_are_shown(self):
        Generated.objects.create(user=self.owner, kind='conference', key='conf', lang='en', status='done', content=json.dumps({
            'name': 'EPSC 2026', 'talks': [{'title': 'Moons around giants', 'speakers': 'A. Example and B. Other', 'session': 'S1', 'time': ''}]}))
        self.assertContains(self.client.get('/en/students/leads/'), 'Moons around giants')

    def test_a_shared_search_hides_drafts_and_who_searched(self):
        row = Generated.objects.create(user=self.owner, kind='students', key='search', lang='en', status='done', content=json.dumps({
            'version': students.RESULT_VERSION, 'title': 'Exomoons', 'profile': self.profile, 'candidates': [self.person]}))
        self.client.post('/en/students/search/', {'action': 'share'})
        row.refresh_from_db()
        page = Client().get('/en/students/shared/%s/' % row.share_token)
        self.assertContains(page, 'Alex Example')
        self.assertNotContains(page, 'data-student-draft')
        self.assertNotContains(page, 'owner')
        self.assertNotContains(page, 'Save this lead')
        self.client.post('/en/students/search/', {'action': 'unshare'})
        self.assertEqual(Client().get('/en/students/shared/%s/' % row.share_token).status_code, 404)

    def test_other_tools_fill_the_form_and_old_country_names_still_refresh(self):
        self.assertContains(self.client.get('/en/students/?topic=Exomoon+timing'), 'Exomoon timing</textarea>')
        self.assertContains(self.client.get('/en/teams/?topic=Exomoon+timing'), 'Exomoon timing</textarea>')
        self.assertContains(self.client.get('/en/literature-review/?question=Exomoon+timing'), 'Exomoon timing</textarea>')
        row = Generated.objects.create(user=self.owner, kind='students', key='old', lang='en', status='done', content=json.dumps({
            'version': 4, 'title': 'Exomoons', 'profile': dict(self.profile, country='France'), 'candidates': []}))
        with mock.patch.object(students, 'find', return_value={'title': 'Fresh', 'profile': self.profile, 'candidates': []}) as find:
            self.assertEqual(self.client.post('/en/students/old/refresh/').status_code, 302)
        self.assertEqual(find.call_args.args[0]['country'], 'FR')
