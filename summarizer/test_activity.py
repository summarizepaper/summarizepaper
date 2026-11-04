"""Tests of the staff activity page /admin/activite/ (activity.py)."""
import datetime

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from summarizer import activity
from summarizer import visits
from summarizer.models import AIassistant, Generated, PageView, PaperHistory, Search, ToolUse


class ActivityTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('staff', email='staff@example.com', password='x', is_staff=True, is_superuser=True)
        self.reader = User.objects.create_user('reader', email='reader@example.com', password='x')
        PaperHistory.objects.create(arxiv_id='2410.12345', user=self.reader, lang='fr')
        AIassistant.objects.create(arxiv_id='2410.12345', query='What is the main limitation?')
        Search.objects.create(query='dark matter halo')
        Generated.objects.create(kind='refcheck', key='abc123', lang='fr', user=self.reader, status='failed', error='timeout')

    def test_the_day_counts_and_feed(self):
        numbers = {n['key']: n['value'] for n in activity.numbers(activity.today())}
        self.assertEqual((numbers['signup'], numbers['summary'], numbers['question'], numbers['search'], numbers['tool']),
                         (1, 1, 1, 1, 1))  # the staff account is the team's own
        self.assertEqual({n['key']: n['value'] for n in activity.numbers(activity.today(), staff=True)}['signup'], 2)
        self.assertEqual([p['value'] for p in next(n for n in activity.numbers(activity.today()) if n['key'] == 'summary')['series']][-2:], [0, 1])
        kinds = [e['kind'] for e in activity.feed(activity.today())]
        self.assertEqual(sorted(kinds), ['question', 'search', 'signup', 'summary', 'tool'])
        tool = next(e for e in activity.feed(activity.today(), 'tool'))
        self.assertTrue(tool['alert'])
        self.assertEqual(tool['href'], '/fr/refcheck/abc123/')
        self.assertEqual(activity.watch()['failed'][0]['detail'], 'timeout')
        self.assertEqual(activity._paper('astro-ph--9604131v1'), '/en/paper/astro-ph--9604131v1/')
        self.assertEqual(activity._paper('abc123'), '')

    def test_members_only_hides_anonymous_actions(self):
        kinds = {e['kind'] for e in activity.feed(activity.today(), members_only=True)}
        self.assertNotIn('question', kinds)
        self.assertIn('summary', kinds)

    def test_another_day_is_empty(self):
        self.assertEqual(activity.feed(activity.today() - datetime.timedelta(days=3)), [])

    def test_staff_only_page(self):
        self.client.force_login(self.reader)
        self.assertEqual(self.client.get('/admin/activite/').status_code, 302)
        self.client.force_login(self.staff)
        page = self.client.get('/admin/activite/')
        self.assertContains(page, 'dark matter halo')
        self.assertContains(page, 'Outils en échec')
        self.assertContains(self.client.get('/admin/activite/?jour=pas-une-date&type=search'), 'dark matter halo')
        self.assertContains(self.client.get('/admin/activite/?partie=fiabilite'), '', status_code=200)
        self.assertEqual(PageView.objects.count(), 0)  # the admin is not a visit

    def test_the_admin_button_is_for_staff_only(self):
        self.client.force_login(self.reader)
        self.assertNotContains(self.client.get('/fr/faq', follow=True), 'class="nav-admin"')
        self.client.force_login(self.staff)
        self.assertContains(self.client.get('/fr/faq', follow=True), 'class="nav-admin"')

    def test_a_dated_model_is_priced_as_its_alias(self):
        from summarizer.models import AIUsage
        row = AIUsage(model='gpt-6-luna-2026-08-05', prompt_tokens=1_000_000, completion_tokens=1_000_000)
        self.assertAlmostEqual(row.cost(), 0.60)  # the prices of the migration: $0.10 in, $0.50 out
        self.assertIsNone(AIUsage(model='gpt-6-lunatic').cost())


class PageViewTests(TestCase):
    """The pages people open (visits.py) and what the activity page makes of them."""
    BROWSER = {'HTTP_USER_AGENT': 'Mozilla/5.0 Firefox/140.0', 'HTTP_SEC_FETCH_DEST': 'document'}

    def setUp(self):
        self.reader = User.objects.create_user('reader', email='reader@example.com', password='x')
        self.staff = User.objects.create_user('staff', email='staff@example.com', password='x', is_staff=True)

    def test_a_page_opened_in_a_browser_is_kept(self):
        self.client.get('/fr/faq', follow=True, HTTP_REFERER='https://www.google.com/search?q=x', **self.BROWSER)
        view = PageView.objects.get()
        self.assertEqual((view.view, view.referrer, view.first, view.user), ('faq', 'google.com', True, None))
        self.assertEqual(len(view.visitor), 16)

    def test_crawlers_fetches_and_scrapers_are_not(self):
        self.client.get('/fr/faq', follow=True, HTTP_USER_AGENT='Googlebot/2.1', HTTP_SEC_FETCH_DEST='document')
        self.client.get('/fr/faq', follow=True, HTTP_USER_AGENT='Mozilla/5.0', HTTP_SEC_FETCH_DEST='empty')
        self.client.get('/fr/faq', follow=True, HTTP_USER_AGENT='Mozilla/5.0')  # no cookie, no Sec-Fetch: a script
        self.assertEqual(PageView.objects.count(), 0)
        self.client.cookies['sp'] = '1'
        self.client.get('/fr/faq', follow=True, HTTP_USER_AGENT='Mozilla/5.0')
        self.assertFalse(PageView.objects.get().first)

    def test_search_keeps_the_query_and_tokens_are_left_out(self):
        from django.contrib.auth.models import AnonymousUser
        from django.test import RequestFactory
        from django.urls import resolve
        request = RequestFactory().get('/en/search-results/?q=dark+matter&utm_source=digest', **self.BROWSER)
        request.user, request.resolver_match = AnonymousUser(), resolve('/search-results/')
        visits.record(request)
        request = RequestFactory().get('/en/login/link/secret-token/', **self.BROWSER)
        request.user, request.resolver_match = self.reader, resolve('/login/link/secret-token/')
        visits.record(request)
        self.assertEqual(PageView.objects.get(view='login_link').path, '')
        PageView.objects.filter(view='login_link').delete()
        self.assertEqual(PageView.objects.values_list('view', 'query', 'source').get(), ('search_results', 'dark matter', 'digest'))
        self.assertEqual(visits.section('collection_slides_start'), 'Journal club')
        self.assertEqual(visits.section('collection'), 'Collections')

    def test_a_visit_becomes_the_members_once_signed_in_and_the_team_is_left_out(self):
        PageView.objects.create(visitor='a' * 16, view='summarize', path='/fr/', referrer='arxiv.org', first=True)
        PageView.objects.create(visitor='a' * 16, view='refcheck', path='/fr/refcheck/', user=self.reader)
        PageView.objects.create(visitor='b' * 16, view='faq', path='/fr/faq/')
        PageView.objects.create(visitor='b' * 16, view='faq', path='/fr/faq/', user=self.staff)
        Generated.objects.create(kind='refcheck', key='k1', lang='fr', user=self.reader, status='done')
        events = [e for e in activity.feed(activity.today()) if e['kind'] == 'visit']
        self.assertEqual({e['person'] for e in events}, {'u%d' % self.reader.pk})
        self.assertIn('depuis arxiv.org', [e['detail'] for e in events][-1])
        self.assertEqual(len([e for e in activity.feed(activity.today(), staff=True) if e['kind'] == 'visit']), 4)
        row = next(r for r in activity.tools(activity.today())['day'] if r['tool'] == 'Vérification de bibliographie')
        self.assertEqual((row['views'], row['visitors'], row['uses'], row['members']), (1, 1, 1, 1))
        self.assertEqual(activity.sources(activity.today())['entries'], [('Accueil', 1)])
        numbers = {n['key']: n['value'] for n in activity.numbers(activity.today())}
        self.assertEqual((numbers['visitors'], numbers['views']), (1, 2))

    def test_a_result_opened_again_is_shown(self):
        old = Generated.objects.create(kind='presskit', key='2410.1', lang='fr', status='done')
        Generated.objects.filter(pk=old.pk).update(created=timezone.now() - datetime.timedelta(days=3))
        ToolUse.objects.create(user=self.reader, row=old, url='/fr/presskit/2410.1/', title='A paper')
        event = next(e for e in activity.feed(activity.today()) if e['kind'] == 'tool_open')
        self.assertEqual((event['text'], event['href']), ('Kit presse · A paper', '/fr/presskit/2410.1/'))
        self.assertEqual(next(r for r in activity.tools(activity.today())['day'] if r['tool'] == 'Kit presse')['reopened'], 1)

    def test_old_page_views_are_deleted(self):
        PageView.objects.create(visitor='a' * 16, view='faq')
        PageView.objects.update(created=timezone.now() - datetime.timedelta(days=visits.VISIT_DAYS + 1))
        PageView.objects.create(visitor='b' * 16, view='faq')
        self.assertEqual(visits.purge(), 1)
        self.assertEqual(PageView.objects.count(), 1)

    def test_only_pages_a_person_moved_on_count(self):
        self.client.get('/fr/faq', follow=True, **self.BROWSER)
        PageView.objects.create(visitor='r' * 16, view='register', path='/fr/register/', referrer='yandex.com')
        numbers = {n['key']: n['value'] for n in activity.numbers(activity.today())}
        self.assertEqual((numbers['visitors'], numbers['robots']), (0, 2))
        path = PageView.objects.exclude(visitor='r' * 16).get().path
        self.assertEqual(self.client.post('/visit/', {'path': path}, HTTP_USER_AGENT=self.BROWSER['HTTP_USER_AGENT']).status_code, 204)
        self.assertTrue(PageView.objects.exclude(visitor='r' * 16).get().human)
        numbers = {n['key']: n['value'] for n in activity.numbers(activity.today())}
        self.assertEqual((numbers['visitors'], numbers['views'], numbers['robots']), (1, 1, 1))
        self.assertEqual(activity.sources(activity.today())['referrers'], [('(direct ou inconnu)', 1)])
        self.assertEqual(self.client.get('/visit/').status_code, 405)
