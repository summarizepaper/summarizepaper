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
        self.assertContains(client.get('/en/students/'), 'Changes are saved automatically.')

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
        self.assertNotContains(client.get('/en/students/?status=contacted&due=1'), 'Private Two')
        self.assertContains(client.get('/en/students/?status=contacted&due=1'), 'Reminder due')

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
