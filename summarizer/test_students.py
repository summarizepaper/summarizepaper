import json
from unittest import mock

from django.test import Client
from django.utils import timezone

from summarizer import students
from summarizer.models import Generated, PremiumGrant, StudentShortlist
from summarizer.test_tools import ToolsTestCase


class StudentTests(ToolsTestCase):
    profile = {'level': 'M1', 'kind': 'internship', 'topic': 'Detecting exomoons by astrometry',
               'country': 'France', 'city': 'Paris', 'mobility': 'strict', 'start': '', 'duration': 6, 'skills': 'Python'}

    def paper_data(self, aid='A1', iid='I1', country='FR', title='Exomoon astrometry'):
        return {'id': 'W1', 'display_name': title, 'publication_date': '2026-01-01',
                'doi': 'https://doi.org/10.1/example', 'authorships': [
                    {'author': {'id': 'https://openalex.org/'+aid, 'display_name': 'Alex Example'},
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
            'title': 'Exomoons', 'profile': self.profile, 'candidates': [self.candidate()] if found is None else found}))

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
        with mock.patch.object(students, 'institution_details', return_value=self.details()):
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
        self.assertEqual(fetch.call_count, students.MAX_INSTITUTIONS)
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
        self.assertIn('came across your publication', draft)
        self.assertNotIn('I have read', draft)
        self.assertIn('possible supervision', draft)
        self.assertContains(Client().get('/fr/students/'), 'Trouver un stage ou une thèse')
        pro = self.user('pro', paid=True)
        row = self.row(pro)
        self.assertContains(self.client_for(pro).get('/fr/students/student-demo/'), 'Lire avant')
