import datetime
from unittest import mock
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup
from django.contrib.auth.models import User
from django.core import mail, signing
from django.core.cache import cache
from django.http import QueryDict
from django.test import Client, SimpleTestCase, TestCase
from django.utils import timezone

from . import saved_searches, sources, unified_search
from .models import SavedSearch, SuppressedEmail, UserPlan


def paper(key, **extra):
    return dict(key=key, title='A paper', authors=['An Author'], venue='A journal',
                published='2026-10-21', abstract='An abstract for readers.', **extra)


class ResultWindowTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_progressive_loading_reuses_prior_pages_and_deduplicates(self):
        first = sources.SearchRows([paper('doi_a'), paper('doi_b')], has_more=True)
        second = sources.SearchRows([paper('doi_b'), paper('doi_c')], has_more=False)
        with mock.patch.object(unified_search, 'search', side_effect=[(first, False), (second, False)]) as search:
            one = unified_search.window_search('climate', {})
            two = unified_search.window_search('climate', {}, one['next_token'])
            again = unified_search.window_search('climate', {}, two['token'])
        self.assertEqual([p['key'] for p in two['rows']], ['doi_a', 'doi_b', 'doi_c'])
        self.assertEqual(again['rows'], two['rows'])
        self.assertFalse(two['next_token'])
        self.assertEqual(search.call_count, 2)
        self.assertEqual(search.call_args, mock.call('climate', page=2))

    def test_tokens_are_query_bound_and_require_a_cached_predecessor(self):
        with mock.patch.object(unified_search, 'search', return_value=(sources.SearchRows([paper('a')], True), False)):
            first = unified_search.window_search('climate', {})
            with self.assertRaises(unified_search.ExpiredWindow):
                unified_search.window_search('different query', {}, first['next_token'])
            cache.clear()
            with self.assertRaises(unified_search.ExpiredWindow):
                unified_search.window_search('climate', {}, first['next_token'])
        with mock.patch.object(unified_search, 'search') as search:
            with self.assertRaises(unified_search.ExpiredWindow):
                unified_search.window_search('climate', {}, 'forged')
            search.assert_not_called()

    def test_partial_results_do_not_offer_a_skip_over_a_failed_page(self):
        with mock.patch.object(unified_search, 'search', return_value=(sources.SearchRows([paper('a')], True), True)):
            window = unified_search.window_search('climate', {})
        self.assertTrue(window['failed'])
        self.assertFalse(window['next_token'])

    @mock.patch('summarizer.unified_search.requests.get')
    def test_arxiv_paging_and_abstract_are_preserved(self, get):
        get.return_value.text = '''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:o="http://a9.com/-/spec/opensearch/1.1/">
        <o:totalResults>51</o:totalResults><entry><id>http://arxiv.org/abs/1706.03762v1</id>
        <title>A title</title><summary>Actual source abstract.</summary></entry></feed>'''
        rows = unified_search.arxiv_results('climate', page=2)
        self.assertEqual(get.call_args.kwargs['params']['start'], 25)
        self.assertTrue(rows.has_more)
        self.assertEqual(rows[0]['abstract'], 'Actual source abstract.')

    @mock.patch('summarizer.sources._spend', return_value=True)
    @mock.patch('summarizer.sources._json', return_value={'results': [], 'meta': {'count': 60}})
    def test_openalex_paging_and_exhaustion_follow_provider_metadata(self, get, spend):
        rows = sources.search_papers('climate', page=2, abstracts=True)
        self.assertEqual(get.call_args.args[1]['page'], 2)
        self.assertIn('abstract_inverted_index', get.call_args.args[1]['select'])
        self.assertTrue(rows.has_more)
        rows = sources.search_papers('climate', page=3, abstracts=True)
        self.assertFalse(rows.has_more)

    @mock.patch('summarizer.sources._json', return_value={'message': {'items': [], 'total-results': 60}})
    def test_crossref_fallback_moves_to_the_next_offset(self, get):
        rows = sources.crossref_search('climate', 25, page=2)
        self.assertEqual(get.call_args.args[1]['offset'], 25)
        self.assertTrue(rows.has_more)


class SavedSearchTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user('reader', 'reader@example.com', 'test-pass')
        self.other = User.objects.create_user('other', 'other@example.com', 'test-pass')
        self.client.force_login(self.user)
        self.params = {'q': 'climate', 'field': 'title', 'src': 'all', 'type': '', 'year_from': '2020'}

    def create_saved(self, user=None, **fields):
        params = fields.pop('params', self.params)
        return SavedSearch.objects.create(user=user or self.user, name='My research', params=params,
                                          fingerprint=saved_searches.fingerprint(params), **fields)

    def test_save_is_private_idempotent_and_post_only(self):
        token = signing.dumps(self.params, salt=saved_searches.STATE_SALT)
        for _ in range(2):
            response = self.client.post('/en/saved-searches/save/', {'state': token})
            self.assertEqual(response.status_code, 302)
        self.assertEqual(SavedSearch.objects.count(), 1)
        self.assertContains(self.client.get('/en/saved-searches/'), 'data-funnel-view="saved_search_created"')
        self.assertNotContains(self.client.get('/en/saved-searches/'), 'data-funnel-view="saved_search_created"')
        self.assertEqual(self.user.saved_searches.get().params, self.params)
        self.assertEqual(self.client.get('/en/saved-searches/save/').status_code, 405)
        self.assertEqual(self.client.post('/en/saved-searches/save/', {'state': 'forged'}).status_code, 400)
        self.client.force_login(self.other)
        private_page = self.client.get('/en/saved-searches/')
        self.assertNotContains(private_page, 'climate')
        self.assertIn('private', private_page['Cache-Control'])
        self.assertIn('no-store', private_page['Cache-Control'])

    def test_ownership_and_authentication_are_required(self):
        saved = self.create_saved()
        self.client.force_login(self.other)
        self.assertEqual(self.client.post('/en/saved-searches/', {'search': saved.pk, 'action': 'delete'}).status_code, 404)
        self.assertTrue(SavedSearch.objects.filter(pk=saved.pk).exists())
        self.client.logout()
        self.assertEqual(self.client.get('/en/saved-searches/').status_code, 302)

    def test_enable_queues_initialization_without_waiting_for_sources(self):
        saved = self.create_saved()
        with mock.patch.object(saved_searches, 'latest', side_effect=RuntimeError('source unavailable')) as latest:
            response = self.client.post('/en/saved-searches/', {'search': saved.pk, 'action': 'enable', 'frequency': 'daily'})
        self.assertEqual(response.status_code, 302)
        saved.refresh_from_db()
        self.assertTrue(saved.watch_enabled)
        self.assertEqual(saved.frequency, 'daily')
        latest.assert_not_called()
        self.assertEqual(saved.seen, [])
        self.assertIsNone(saved.last_checked)
        pending_page = self.client.get('/en/saved-searches/')
        self.assertContains(pending_page, 'initialization pending')
        self.assertContains(pending_page, 'data-funnel-view="search_watch_enabled"')
        self.assertNotContains(self.client.get('/en/saved-searches/'), 'data-funnel-view="search_watch_enabled"')
        self.assertEqual(len(mail.outbox), 0)
        self.client.post('/en/saved-searches/', {'search': saved.pk, 'action': 'pause'})
        saved.refresh_from_db()
        self.assertFalse(saved.watch_enabled)

    def test_initialization_retries_outages_and_never_emails_the_baseline(self):
        now = timezone.now()
        saved = self.create_saved(watch_enabled=True, frequency='daily')
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old')], True)):
            self.assertEqual(saved_searches.run(now, 1), 0)
        saved.refresh_from_db()
        self.assertTrue(saved.watch_enabled)
        self.assertIsNone(saved.last_checked)
        self.assertEqual(saved.seen, [])
        self.assertIsNone(saved.watch_claim_until)
        later = now + datetime.timedelta(hours=1)
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old')], False)):
            self.assertEqual(saved_searches.run(later, 1), 0)
        saved.refresh_from_db()
        self.assertEqual(saved.last_checked, later)
        self.assertIn('doi_old', saved.seen)
        self.assertEqual(len(mail.outbox), 0)
        self.assertContains(self.client.get('/en/saved-searches/'), 'Watch active')
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old'), paper('doi_new')], False)):
            self.assertEqual(saved_searches.run(later + datetime.timedelta(days=1), 1), 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('doi_new', mail.outbox[0].body)
        self.assertNotIn('doi_old', mail.outbox[0].body)

    def test_resume_clears_the_old_baseline_and_keeps_post_action_without_javascript(self):
        saved = self.create_saved(last_checked=timezone.now(), last_attempt=timezone.now(), seen=['old'])
        page = self.client.get('/en/saved-searches/')
        soup = BeautifulSoup(page.content, 'html.parser')
        form = soup.select_one('form[data-watch-activation]')
        self.assertEqual(form.select_one('input[name=action]')['value'], 'enable')
        self.assertIsNone(form.select_one('button[type=submit]').get('name'))
        self.assertContains(page, 'summarizer/js/saved-searches.js')
        with mock.patch.object(saved_searches, 'latest') as latest:
            self.client.post('/en/saved-searches/', {'search': saved.pk, 'action': 'enable'})
        latest.assert_not_called()
        saved.refresh_from_db()
        self.assertTrue(saved.watch_enabled)
        self.assertEqual(saved.seen, [])
        self.assertIsNone(saved.last_checked)
        self.assertIsNone(saved.last_attempt)

    def test_tier_watch_limit_is_enforced_before_fetching(self):
        self.create_saved(watch_enabled=True)
        second = self.create_saved(params={**self.params, 'q': 'different'})
        with mock.patch.object(saved_searches, 'latest') as latest:
            self.client.post('/en/saved-searches/', {'search': second.pk, 'action': 'enable'})
        latest.assert_not_called()
        second.refresh_from_db()
        self.assertFalse(second.watch_enabled)
        UserPlan.objects.create(user=self.user, paid=True)
        with mock.patch.object(saved_searches, 'latest', return_value=([], False)):
            self.client.post('/en/saved-searches/', {'search': second.pk, 'action': 'enable'})
        second.refresh_from_db()
        self.assertTrue(second.watch_enabled)

    def test_signed_unsubscribe_requires_post_and_does_not_expose_search_text(self):
        saved = self.create_saved(watch_enabled=True)
        token = signing.dumps(saved.pk, salt=saved_searches.UNSUBSCRIBE_SALT)
        self.client.logout()
        url = '/en/saved-searches/unsubscribe/?' + 'token=' + token
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'climate')
        saved.refresh_from_db()
        self.assertTrue(saved.watch_enabled)
        self.client.post(url)
        saved.refresh_from_db()
        self.assertFalse(saved.watch_enabled)
        self.assertEqual(self.client.get('/en/saved-searches/unsubscribe/?token=bad').status_code, 404)

    def test_worker_sends_only_unseen_records_and_honors_budget_and_schedule(self):
        now = timezone.now()
        saved = self.create_saved(watch_enabled=True, frequency='daily', last_checked=now - datetime.timedelta(days=2), seen=['doi_old'])
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old'), paper('doi_new')], False)) as latest:
            self.assertEqual(saved_searches.run(now, 1), 1)
            self.assertEqual(saved_searches.run(now, 1), 0)
            self.assertEqual(saved_searches.run(now, 0), 0)
        self.assertEqual(latest.call_count, 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, [self.user.email])
        self.assertIn('/en/search-results/?q=climate&field=title', mail.outbox[0].body)
        self.assertNotIn('&amp;', mail.outbox[0].body)
        saved.refresh_from_db()
        self.assertIn('doi_new', saved.seen)
        self.assertEqual(saved.last_sent, now)
        self.assertIsNone(saved.watch_claim_until)

    def test_failed_delivery_and_partial_sources_keep_retry_state(self):
        now = timezone.now()
        old = now - datetime.timedelta(days=2)
        saved = self.create_saved(watch_enabled=True, frequency='daily', last_checked=old, seen=['doi_old'])
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_new')], False)), \
             mock.patch.object(saved_searches.EmailMultiAlternatives, 'send', side_effect=RuntimeError('offline')):
            self.assertEqual(saved_searches.run(now, 1), 0)
        saved.refresh_from_db()
        self.assertEqual(saved.last_checked, old)
        self.assertEqual(saved.seen, ['doi_old'])
        self.assertIsNone(saved.watch_claim_until)
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_new')], True)):
            self.assertEqual(saved_searches.run(now, 1), 0)
        saved.refresh_from_db()
        self.assertEqual(saved.last_checked, old)

    def test_database_lease_and_suppression_prevent_duplicate_or_unwanted_mail(self):
        now = timezone.now()
        saved = self.create_saved(watch_enabled=True, frequency='daily', last_checked=now - datetime.timedelta(days=2),
                                  watch_claim_until=now + datetime.timedelta(minutes=4))
        with mock.patch.object(saved_searches, 'latest') as latest:
            self.assertEqual(saved_searches.run(now, 1), 0)
            latest.assert_not_called()
        saved.watch_claim_until = None
        saved.save(update_fields=['watch_claim_until'])
        SuppressedEmail.objects.create(email=self.user.email)
        with mock.patch.object(saved_searches, 'latest') as latest:
            self.assertEqual(saved_searches.run(now, 1), 0)
            latest.assert_not_called()

    def test_filter_chips_remove_only_the_chosen_filter_and_preview_is_escaped(self):
        cache.set('search-journal:S1', {'id': 'S1', 'name': 'Nature'})
        cache.set('search-journal:S2', {'id': 'S2', 'name': 'Science'})
        rows = [paper('doi_a',)]
        rows[0]['abstract'] = '<script>alert(1)</script>Useful source text.'
        with mock.patch.object(unified_search, 'search', return_value=(rows, False)):
            response = self.client.get('/en/search-results/', {'q': 'climate', 'journals': ['S1', 'S2'], 'year_from': '2020'})
        soup = BeautifulSoup(response.content, 'html.parser')
        chip = next(a for a in soup.select('.active-search-filters a') if 'Nature' in a.get_text())
        args = parse_qs(urlsplit(chip['href']).query)
        self.assertEqual(args['journals'], ['S2'])
        self.assertEqual(args['year_from'], ['2020'])
        self.assertNotIn('window', args)
        self.assertContains(response, 'Useful source text.')
        self.assertNotContains(response, '<script>alert(1)</script>')

    def test_failed_checks_yield_to_other_due_watches(self):
        now = timezone.now()
        old = now - datetime.timedelta(days=2)
        first = self.create_saved(watch_enabled=True, frequency='daily', last_checked=old)
        second = self.create_saved(user=self.other, watch_enabled=True, frequency='daily', last_checked=old)
        with mock.patch.object(saved_searches, 'MAX_CHECKS_PER_RUN', 1), \
             mock.patch.object(saved_searches, 'latest', side_effect=[([], True), ([paper('doi_new')], False)]) as latest:
            self.assertEqual(saved_searches.run(now, 1), 0)
            self.assertEqual(saved_searches.run(now, 1), 1)
        self.assertEqual([call.args[0].pk for call in latest.call_args_list], [first.pk, second.pk])

    def test_dry_run_and_user_command_do_not_send_search_watches(self):
        from django.core.management import call_command
        with mock.patch('summarizer.digest.run', return_value=0), \
             mock.patch.object(saved_searches, 'run') as watches:
            call_command('send_digest', dry_run=True)
            call_command('send_digest', user=self.user.username)
        watches.assert_not_called()

    def test_more_results_are_offered_at_the_end_of_the_loaded_pages(self):
        first = sources.SearchRows([paper('doi_%d' % i) for i in range(25)], has_more=True)
        second = sources.SearchRows([paper('doi_%d' % i) for i in range(25, 50)], has_more=False)
        with mock.patch.object(unified_search, 'search', side_effect=[(first, False), (second, False)]) as search:
            response = self.client.get('/en/search-results/?q=climate')
            self.assertNotContains(response, 'class="load-more-results"')
            response = self.client.get('/en/search-results/?q=climate&page=3')
            soup = BeautifulSoup(response.content, 'html.parser')
            more = soup.select_one('.load-more-results a')['href']
            response = self.client.get('/en/search-results/' + more)
            self.assertEqual(response.context['page_obj'].paginator.count, 50)
        self.assertEqual(search.call_count, 2)
