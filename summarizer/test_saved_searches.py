import datetime
import json
from unittest import mock
from urllib.parse import parse_qs, urlsplit

import numpy as np
from bs4 import BeautifulSoup
from django.contrib.auth.models import User
from django.core import mail, signing
from django.core.cache import cache
from django.http import QueryDict
from django.test import Client, SimpleTestCase, TestCase, override_settings
from django.utils import timezone

from . import digest, saved_searches, sources, unified_search
from .models import (AnnouncedPaper, Collection, DigestSubscription, SavedSearch, SavedSearchFeedback,
                     SuppressedEmail, UserPlan)


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


@override_settings(OPENAI_KEY='')
class SavedSearchTests(TestCase):
    def setUp(self):
        cache.clear()
        batch = mock.patch.object(digest, 'ensure_batch', return_value=[])
        batch.start()
        self.addCleanup(batch.stop)
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
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old')], True, False)):
            self.assertEqual(saved_searches.run(now, 1), 0)
        saved.refresh_from_db()
        self.assertTrue(saved.watch_enabled)
        self.assertIsNone(saved.last_checked)
        self.assertEqual(saved.seen, [])
        self.assertIsNone(saved.watch_claim_until)
        later = now + datetime.timedelta(hours=1)
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old')], False, False)):
            self.assertEqual(saved_searches.run(later, 1), 0)
        saved.refresh_from_db()
        self.assertEqual(saved.last_checked, later)
        self.assertIn('doi_old', saved.seen)
        self.assertEqual(len(mail.outbox), 0)
        self.assertContains(self.client.get('/en/saved-searches/'), 'Watch active')
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old'), paper('doi_new')], False, False)):
            self.assertEqual(saved_searches.run(later + datetime.timedelta(days=1), 1), 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('doi_new', mail.outbox[0].body)
        self.assertNotIn('doi_old', mail.outbox[0].body)

    def test_resume_keeps_the_baseline_and_keeps_post_action_without_javascript(self):
        checked = timezone.now() - datetime.timedelta(days=3)
        saved = self.create_saved(last_checked=checked, last_attempt=checked, seen=['old'])
        page = self.client.get('/en/saved-searches/%d/' % saved.pk)
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
        # the papers announced during the pause are still new: the next check sends them
        self.assertEqual(saved.seen, ['old'])
        self.assertEqual(saved.last_checked, checked)

    def test_tier_watch_limit_is_enforced_before_fetching(self):
        self.create_saved(watch_enabled=True)
        second = self.create_saved(params={**self.params, 'q': 'different'})
        with mock.patch.object(saved_searches, 'latest') as latest:
            self.client.post('/en/saved-searches/', {'search': second.pk, 'action': 'enable'})
        latest.assert_not_called()
        second.refresh_from_db()
        self.assertFalse(second.watch_enabled)
        UserPlan.objects.create(user=self.user, paid=True)
        with mock.patch.object(saved_searches, 'latest', return_value=([], False, False)):
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
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_old'), paper('doi_new')], False, False)) as latest:
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
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_new')], False, False)), \
             mock.patch.object(saved_searches.EmailMultiAlternatives, 'send', side_effect=RuntimeError('offline')):
            self.assertEqual(saved_searches.run(now, 1), 0)
        saved.refresh_from_db()
        self.assertEqual(saved.last_checked, old)
        self.assertEqual(saved.seen, ['doi_old'])
        self.assertIsNone(saved.watch_claim_until)
        with mock.patch.object(saved_searches, 'latest', return_value=([paper('doi_new')], True, False)):
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
             mock.patch.object(saved_searches, 'latest', side_effect=[([], True, False), ([paper('doi_new')], False, False)]) as latest:
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


def announce(key, title, abstract='', authors=('Ada Lovelace',), categories='astro-ph.EP', day=None):
    return AnnouncedPaper.objects.create(arxiv_id=key, title=title, abstract=abstract, authors=json.dumps(list(authors)),
                                         categories=categories, announced=day or timezone.now().date())


@override_settings(OPENAI_KEY='')
class WatchTests(TestCase):
    """The watch reads the daily announcements whole, and the page, digest and feed built on what it finds."""

    def setUp(self):
        cache.clear()
        batch = mock.patch.object(digest, 'ensure_batch', return_value=[])
        batch.start()
        self.addCleanup(batch.stop)
        self.user = User.objects.create_user('reader', 'reader@example.com', 'test-pass')
        self.client.force_login(self.user)
        self.now = timezone.now()

    def create_saved(self, params=None, **fields):
        params = params or {'q': 'exoplanet atmospheres', 'field': 'all', 'src': 'arxiv', 'type': ''}
        return SavedSearch.objects.create(user=self.user, name=params.get('q') or 'search', params=params,
                                          fingerprint=saved_searches.fingerprint(params), **fields)

    def test_query_reads_every_word_plurals_quotes_exclusions_and_alternatives(self):
        q = saved_searches.Query('exoplanet atmospheres -review OR "hot jupiter"')
        self.assertTrue(q.supported)
        self.assertEqual(q.match({'title': 'An exoplanet atmosphere', 'abstract': 'We model it.'}), ['exoplanet', 'atmospheres'])
        self.assertEqual(q.match({'title': 'Exoplanet atmospheres: a review', 'abstract': ''}), [])
        self.assertEqual(q.match({'title': 'Exoplanets', 'abstract': 'no atmosphere word'}), ['exoplanet', 'atmospheres'])
        self.assertEqual(q.match({'title': 'A hot Jupiter', 'abstract': ''}), ['hot jupiter'])
        self.assertEqual(q.match({'title': 'Hot and cold Jupiter', 'abstract': ''}), [])
        self.assertEqual(q.match({'title': 'Planet atmospheres', 'abstract': ''}), [])
        self.assertFalse(saved_searches.Query('au:Smith AND ti:graph').supported)
        authors = saved_searches.Query('Yann LeCun, Hinton', 'author')
        self.assertEqual(authors.match({'authors': ['Y. LeCun', 'G. E. Hinton']}), ['Y. LeCun', 'G. E. Hinton'])
        self.assertEqual(authors.match({'authors': ['Jan LeCun']}), [])
        self.assertEqual(saved_searches.fingerprint({'q': 'a', 'sort': 'newest'}), saved_searches.fingerprint({'q': 'a'}))

    def test_a_broad_search_loses_nothing_and_the_email_ranks_ten_with_the_rest_on_the_page(self):
        old = self.now - datetime.timedelta(days=1)
        saved = self.create_saved(watch_enabled=True, frequency='daily', last_checked=old - datetime.timedelta(days=1),
                                  seen=['2610.00001'])
        announce('2610.00001v1', 'Exoplanet atmospheres seen before')
        for i in range(2, 62):
            announce('2610.%05dv1' % i, 'Exoplanet atmosphere number %d' % i, categories='astro-ph.EP,astro-ph.SR')
        announce('2610.09999v1', 'Galaxy clusters')
        with mock.patch.object(unified_search, 'arxiv_results', side_effect=AssertionError('the announcements answer')):
            self.assertEqual(saved_searches.run(self.now, 1), 1)
        saved.refresh_from_db()
        self.assertEqual(saved.found_total, 60)
        self.assertEqual(len(saved.matches), 60)
        self.assertNotIn('2610.09999v1', [r['key'] for r in saved.matches])
        self.assertIn('2610.00061', saved.seen)
        body = mail.outbox[0].body
        self.assertIn('60 new papers', mail.outbox[0].subject)
        self.assertEqual(body.count('/en/paper/2610.'), 10)
        self.assertIn('50 more new papers on the page of the search', body)
        self.assertIn('/en/saved-searches/%d/' % saved.pk, body)
        self.assertNotIn('/arxiv-id/', body)

    def test_categories_keep_to_the_announcements_and_journal_filters_read_openalex_over_a_window(self):
        saved = self.create_saved(categories='astro-ph.SR')
        announce('2610.00001v1', 'Exoplanet atmospheres', categories='astro-ph.EP')
        announce('2610.00002v1', 'Exoplanet atmospheres of stars', categories='astro-ph.SR')
        rows, failed, truncated = saved_searches.latest(saved, self.now)
        self.assertEqual([r['key'] for r in rows], ['2610.00002v1'])
        journals = self.create_saved(params={'q': 'exoplanet', 'field': 'all', 'src': 'all', 'type': '', 'journals': ['S1'],
                                             'year_to': '2020'}, last_checked=self.now - datetime.timedelta(days=1))
        page = sources.SearchRows([paper('doi_10.1--x%d' % i) for i in range(50)], has_more=True)
        with mock.patch.object(sources, 'search', return_value=page) as search:
            rows, failed, truncated = saved_searches.latest(journals, self.now)
        self.assertEqual(search.call_count, saved_searches.API_PAGES)
        options = search.call_args.kwargs['filters']
        self.assertEqual(options['sort'], 'newest')
        self.assertNotIn('year_to', options)
        self.assertEqual(options['since'], (self.now - datetime.timedelta(days=15)).date().isoformat())
        self.assertFalse(search.call_args.kwargs['keep'])
        self.assertTrue(truncated)
        self.assertFalse(failed)

    def test_bundled_watches_wait_for_the_digest_and_page_only_watches_send_nothing(self):
        UserPlan.objects.create(user=self.user, paid=True)
        DigestSubscription.objects.create(user=self.user, categories='astro-ph.EP', bundle_alerts=True)
        saved = self.create_saved(watch_enabled=True, frequency='daily', last_checked=self.now - datetime.timedelta(days=2))
        quiet = self.create_saved(params={'q': 'hot jupiter', 'field': 'all', 'src': 'arxiv', 'type': ''}, watch_enabled=True,
                                  email=False, frequency='daily', last_checked=self.now - datetime.timedelta(days=2))
        announce('2610.00001v1', 'Exoplanet atmospheres')
        announce('2610.00002v1', 'A hot Jupiter')
        self.assertEqual(saved_searches.run(self.now, 5), 0)
        self.assertEqual(len(mail.outbox), 0)
        saved.refresh_from_db()
        quiet.refresh_from_db()
        self.assertEqual([r['key'] for r in saved.pending], ['2610.00001v1'])
        self.assertEqual([r['key'] for r in quiet.matches], ['2610.00002v1'])
        self.assertEqual(quiet.pending, [])
        section = saved_searches.bundle_section(self.user, 'https://example.com', 'en', self.now)
        self.assertEqual([i['title'] for i in section['items']], ['Exoplanet atmospheres'])
        self.assertEqual(section['items'][0]['url'], 'https://example.com/en/paper/2610.00001v1/')
        section['commit']()
        saved.refresh_from_db()
        self.assertEqual(saved.pending, [])
        self.assertEqual(saved.last_sent, self.now)

    def test_feedback_raises_close_papers_and_lowers_the_others(self):
        saved = self.create_saved()
        vec = lambda *x: np.array(x, dtype=np.float32)
        SavedSearchFeedback.objects.create(saved=saved, key='liked', value=1, vector=vec(0, 1, 0, 0).tobytes())
        SavedSearchFeedback.objects.create(saved=saved, key='disliked', value=-1, vector=vec(0, 0, 1, 0).tobytes())
        vectors = {'near_dislike': vec(0.6, 0, 0.8, 0), 'plain': vec(0.6, 0, 0, 0.8), 'near_like': vec(0.6, 0.8, 0, 0),
                   **{'other%d' % i: vec(0.6, 0, 0, 0.8) for i in range(3)}}
        rows = [paper(k) for k in vectors]
        with self.settings(OPENAI_KEY='sk'), mock.patch.object(digest, 'ensure_vectors', return_value=vectors), \
                mock.patch.object(saved_searches, 'query_vector', return_value=vec(1, 0, 0, 0)):
            ranked = [r['key'] for r in saved_searches.rank(saved, rows)]
        self.assertEqual((ranked[0], ranked[-1]), ('near_like', 'near_dislike'))

    def test_a_vote_does_not_push_back_the_whole_topic(self):
        # every paper is close to the one marked not relevant: only those closer than usual go back
        saved = self.create_saved()
        vec = lambda *x: np.array(x, dtype=np.float32) / np.linalg.norm(x)
        SavedSearchFeedback.objects.create(saved=saved, key='disliked', value=-1, vector=vec(0, 1, 0).tobytes())
        vectors = {'topic%d' % i: vec(0.8, 0.6, 0) for i in range(5)}
        vectors['closest_to_query'] = vec(0.9, 0.6, 0)
        rows = [paper(k) for k in vectors]
        with self.settings(OPENAI_KEY='sk'), mock.patch.object(digest, 'ensure_vectors', return_value=vectors), \
                mock.patch.object(saved_searches, 'query_vector', return_value=vec(1, 0, 0)):
            ranked = [r['key'] for r in saved_searches.rank(saved, rows)]
        self.assertEqual(ranked[0], 'closest_to_query')

    def test_search_engine_papers_need_the_words_in_their_title_or_abstract_and_phrases_rank_first(self):
        params = {'q': 'galaxy cluster', 'field': 'all', 'src': 'open', 'type': 'journals'}
        saved = self.create_saved(params=params, last_checked=self.now - datetime.timedelta(days=1))
        page = sources.SearchRows([dict(paper('doi_a'), title='A radio galaxy', abstract='Its jets.'),
                                   dict(paper('doi_b'), title='Galaxy clusters at z=1', abstract='We observe them.'),
                                   dict(paper('doi_c'), title='Cluster of a galaxy', abstract=''),
                                   dict(paper('doi_d'), title='Notes', abstract='')], has_more=False)
        with mock.patch.object(sources, 'search', return_value=page):
            rows = saved_searches.latest(saved, self.now)[0]
        self.assertEqual([r['key'] for r in rows], ['doi_b', 'doi_c'])
        self.assertEqual(rows[0]['terms'], ['galaxy', 'cluster'])
        self.assertEqual(len(saved_searches.Query('galaxy cluster').phrases()), 1)
        self.assertTrue(saved_searches.Query('galaxy cluster').phrases()[0].search('rich galaxy clusters'))
        self.assertFalse(saved_searches.Query('galaxy cluster').phrases()[0].search('a cluster in the galaxy'))

    def test_page_previews_votes_and_acts_on_the_ticked_papers(self):
        saved = self.create_saved()
        announce('2610.00001v1', 'Exoplanet atmospheres in transit')
        page = self.client.get('/en/saved-searches/%d/' % saved.pk)
        # the preview may ask OpenAlex: the page loads it afterwards, or at once without JavaScript
        self.assertContains(page, 'data-preview-url="/en/saved-searches/%d/preview/"' % saved.pk)
        self.assertNotContains(page, 'Exoplanet atmospheres in transit')
        part = self.client.get('/en/saved-searches/%d/preview/' % saved.pk)
        self.assertContains(part, 'Exoplanet atmospheres in transit')
        self.assertContains(part, '1 new paper')
        self.assertContains(part, 'href="/en/paper/2610.00001v1/"')
        french_part = self.client.get('/fr/saved-searches/%d/preview/' % saved.pk)
        self.assertContains(french_part, 'href="/fr/paper/2610.00001v1/"')
        self.assertContains(self.client.get('/en/saved-searches/%d/' % saved.pk), 'Exoplanet atmospheres in transit')  # cached
        self.assertContains(page, 'form method="post" action="/en/saved-searches/" class="watch-controls" data-watch-activation')
        self.assertContains(page, '/en/teams/?topic=exoplanet%20atmospheres')
        saved.matches = [saved_searches.stored(dict(paper('2610.00001v1'), terms=['exoplanet']), self.now)]
        saved.save()
        page = self.client.get('/en/saved-searches/%d/' % saved.pk)
        self.assertContains(page, 'name="ids" value="2610.00001v1"')
        self.assertContains(page, 'matched: exoplanet')
        with mock.patch.object(digest, 'stored_vectors', return_value={}):
            self.client.post('/en/saved-searches/%d/' % saved.pk, {'action': 'vote', 'vote': '2610.00001v1|1'})
        self.assertEqual(saved.feedback.get().value, 1)
        self.client.post('/en/saved-searches/%d/' % saved.pk, {'action': 'vote', 'vote': '2610.00001v1|0'})
        self.assertFalse(saved.feedback.exists())
        self.assertEqual(self.client.post('/en/saved-searches/%d/' % saved.pk, {'action': 'vote', 'vote': 'other|1'}).status_code, 404)
        chat = self.client.post('/en/saved-searches/%d/' % saved.pk, {'action': 'chat', 'ids': ['2610.00001v1', 'not-found']})
        self.assertEqual(chat['Location'], '/en/chat/?ids=2610.00001v1')
        with mock.patch('summarizer.library.add_many', return_value={'added': 1}) as add:
            response = self.client.post('/en/saved-searches/%d/' % saved.pk, {'action': 'collection', 'collection': 'new', 'ids': ['2610.00001v1']})
        collection = Collection.objects.get(user=self.user)
        self.assertEqual(collection.name, saved.name)
        self.assertEqual(add.call_args.args[1], ['2610.00001v1'])
        self.assertEqual(response['Location'], '/en/collections/%d/' % collection.pk)
        self.assertEqual(saved_searches.matching(self.user, '2610.00001v1'), [{'pk': saved.pk, 'name': saved.name}])
        other = User.objects.create_user('other', 'other@example.com', 'x')
        self.client.force_login(other)
        self.assertEqual(self.client.get('/en/saved-searches/%d/' % saved.pk).status_code, 404)

    def test_feed_is_private_by_its_key(self):
        saved = self.create_saved(matches=[saved_searches.stored(dict(paper('2610.00001v1'), blurb='Two sentences.'), self.now)])
        self.client.get('/en/saved-searches/%d/' % saved.pk)
        saved.refresh_from_db()
        self.client.logout()
        response = self.client.get('/en/saved-searches/feed/%s/' % saved.feed_key)
        self.assertEqual(response['Content-Type'], 'application/atom+xml; charset=utf-8')
        self.assertIn(b'Two sentences.', response.content)
        self.assertIn(b'/en/paper/2610.00001v1/', response.content)
        self.assertEqual(self.client.get('/en/saved-searches/feed/%s/' % ('0' * 32)).status_code, 404)

    def test_editing_a_search_keeps_its_watch_and_history_and_starts_again_from_now(self):
        saved = self.create_saved(watch_enabled=True, last_checked=self.now, seen=['a'],
                                  matches=[{'key': 'a', 'title': 'A', 'found': self.now.isoformat()}])
        params = {'q': 'exoplanet clouds', 'field': 'all', 'src': 'arxiv', 'type': ''}
        token = signing.dumps(params, salt=saved_searches.STATE_SALT)
        response = self.client.post('/en/saved-searches/save/', {'state': token, 'replace': saved.pk})
        self.assertEqual(response['Location'], '/en/saved-searches/%d/' % saved.pk)
        saved.refresh_from_db()
        self.assertEqual(saved.params, params)
        self.assertEqual(saved.name, 'exoplanet clouds')
        self.assertTrue(saved.watch_enabled)
        self.assertIsNone(saved.last_checked)
        self.assertEqual(len(saved.matches), 1)
        self.assertEqual(SavedSearch.objects.count(), 1)
        other = self.create_saved()
        token = signing.dumps(other.params, salt=saved_searches.STATE_SALT)
        response = self.client.post('/en/saved-searches/save/', {'state': token, 'replace': saved.pk})
        self.assertEqual(response['Location'], '/en/saved-searches/%d/' % other.pk)
        with mock.patch.object(unified_search, 'search', return_value=([], False)):
            page = self.client.get('/en/search-results/', dict(params, edit=saved.pk))
        self.assertContains(page, 'Unchanged · back to the saved search')
        with mock.patch.object(unified_search, 'search', return_value=([], False)):
            page = self.client.get('/en/search-results/', dict(params, q='exoplanet hazes', edit=saved.pk))
        self.assertContains(page, 'name="replace" value="%d"' % saved.pk)
        self.assertContains(page, 'name="edit" value="%d"' % saved.pk)

    def test_new_categories_record_what_they_already_hold(self):
        saved = self.create_saved(watch_enabled=True, last_checked=self.now)
        self.client.post('/en/saved-searches/', {'search': saved.pk, 'action': 'settings', 'categories': 'astro-ph.EP, bad code!, cs.LG',
                                                 'delivery': 'page', 'frequency': 'daily', 'next': 'detail'})
        saved.refresh_from_db()
        self.assertEqual(saved.categories, 'astro-ph.EP, cs.LG')
        self.assertFalse(saved.email)
        self.assertIsNone(saved.last_checked)

    def test_singular_plural_latin_plurals_and_prefixes(self):
        self.assertEqual(saved_searches.Query('galaxy').match({'title': 'Two galaxies', 'abstract': ''}), ['galaxy'])
        self.assertEqual(saved_searches.Query('galaxies').match({'title': 'A galaxy', 'abstract': ''}), ['galaxies'])
        self.assertTrue(saved_searches.Query('spectrum').match({'title': 'Spectra of stars', 'abstract': ''}))
        self.assertTrue(saved_searches.Query('nuclei').match({'title': 'The atomic nucleus', 'abstract': ''}))
        self.assertFalse(saved_searches.Query('exoplanet').match({'title': 'Exoplanetary systems', 'abstract': ''}))
        self.assertTrue(saved_searches.Query('exoplanet*').match({'title': 'Exoplanetary systems', 'abstract': ''}))
        self.assertFalse(saved_searches.Query('comet').match({'title': 'Cometary dust', 'abstract': ''}))
        # the database filter lets every form through
        saved = self.create_saved(params={'q': 'galaxy', 'field': 'all', 'src': 'arxiv', 'type': ''})
        announce('2610.00001v1', 'Merging galaxies')
        self.assertEqual([r['key'] for r in saved_searches.latest(saved, self.now)[0]], ['2610.00001v1'])

    def test_page_only_watches_need_no_email_and_announcement_checks_are_not_counted(self):
        SuppressedEmail.objects.create(email=self.user.email)
        quiet = self.create_saved(watch_enabled=True, email=False, frequency='daily', last_checked=self.now - datetime.timedelta(days=2))
        announce('2610.00001v1', 'Exoplanet atmospheres')
        with mock.patch.object(saved_searches, 'MAX_CHECKS_PER_RUN', 0):
            self.assertEqual(saved_searches.run(self.now, 5), 0)
        quiet.refresh_from_db()
        self.assertEqual([r['key'] for r in quiet.matches], ['2610.00001v1'])
        other = self.create_saved(params={'q': 'hot jupiter', 'field': 'all', 'src': 'arxiv', 'type': ''})
        self.client.post('/en/saved-searches/', {'search': other.pk, 'action': 'enable', 'delivery': 'page'})
        other.refresh_from_db()
        self.assertFalse(other.watch_enabled)  # the free plan's one watch is taken
        quiet.delete()
        self.client.post('/en/saved-searches/', {'search': other.pk, 'action': 'enable', 'delivery': 'page'})
        other.refresh_from_db()
        self.assertTrue(other.watch_enabled)
        self.assertFalse(other.email)

    def test_email_vote_links_vote_only_with_the_page_post(self):
        saved = self.create_saved(matches=[saved_searches.stored(paper('2610.00001v1'), self.now)])
        token = saved_searches.vote_token(saved, '2610.00001v1', -1)
        self.client.logout()
        page = self.client.get('/en/saved-searches/vote/%s/' % token)
        self.assertContains(page, 'Not relevant')
        self.assertFalse(saved.feedback.exists())
        with mock.patch.object(digest, 'stored_vectors', return_value={}):
            self.assertContains(Client(enforce_csrf_checks=True).post('/en/saved-searches/vote/%s/' % token), 'Thank you')
        self.assertEqual(saved.feedback.get().value, -1)
        self.assertContains(self.client.get('/en/saved-searches/vote/forged/'), 'This link is invalid.')

    def test_mute_sets_aside_papers_close_to_a_rejected_one_without_emailing_them(self):
        saved = self.create_saved(watch_enabled=True, mute=True, frequency='daily', last_checked=self.now - datetime.timedelta(days=2))
        SavedSearchFeedback.objects.create(saved=saved, key='old', value=-1, vector=np.array([0, 1, 0], dtype=np.float32).tobytes())
        announce('2610.00001v1', 'Exoplanet atmospheres like the rejected one')
        vectors = {'2610.00001v1': np.array([0, 0.9, 0.436], dtype=np.float32)}
        with self.settings(OPENAI_KEY='sk'), mock.patch.object(digest, 'ensure_vectors', return_value=vectors), \
                mock.patch.object(saved_searches, 'query_vector', return_value=np.array([0, 0, 1], dtype=np.float32)), \
                mock.patch.object(saved_searches, 'add_blurbs') as blurbs:
            self.assertEqual(saved_searches.run(self.now, 5), 0)
        self.assertEqual(len(mail.outbox), 0)
        blurbs.assert_called_once_with([], saved.lang)
        saved.refresh_from_db()
        self.assertTrue(saved.matches[0]['muted'])
        self.assertEqual(saved.last_checked, self.now)
        self.assertIsNone(saved.last_sent)
        page = self.client.get('/en/saved-searches/%d/' % saved.pk)
        self.assertContains(page, 'set aside: close to a paper you marked not relevant')
        self.client.get('/en/saved-searches/%d/' % saved.pk)
        saved.refresh_from_db()
        self.assertNotIn(b'2610.00001v1', self.client.get('/en/saved-searches/feed/%s/' % saved.feed_key).content)

    def test_journal_papers_of_the_announcements_show_their_journal(self):
        saved = self.create_saved(params={'q': 'catalysis', 'field': 'all', 'src': 'open', 'type': ''}, categories='journals')
        AnnouncedPaper.objects.create(arxiv_id='doi_10.1--abc', title='Catalysis at scale', abstract='', authors='[]',
                                      categories='journals.1605', venue='Nature Catalysis', announced=self.now.date())
        rows = saved_searches.latest(saved, self.now)[0]
        self.assertEqual(rows[0]['venue'], 'Nature Catalysis')

    def test_a_failed_search_engine_holds_back_only_its_own_papers(self):
        params = {'q': 'exoplanet atmospheres', 'field': 'all', 'src': 'all', 'type': ''}
        saved = self.create_saved(params=params, watch_enabled=True, frequency='daily',
                                  last_checked=self.now - datetime.timedelta(days=2))
        announce('2610.00001v1', 'Exoplanet atmospheres')
        with mock.patch.object(sources, 'search', side_effect=sources.SourceUnavailable('budget')) as search:
            self.assertEqual(saved_searches.run(self.now, 5), 1)
        self.assertTrue(search.call_args.kwargs['watch'])
        saved.refresh_from_db()
        self.assertEqual([r['key'] for r in saved.matches], ['2610.00001v1'])
        # a first check waits for every source: what the engine has would otherwise be sent as new later
        fresh = self.create_saved(params=dict(params, q='hot jupiter'))
        with mock.patch.object(sources, 'search', side_effect=sources.SourceUnavailable('budget')):
            self.assertTrue(saved_searches.latest(fresh, self.now)[1])

    def test_the_mute_floor_follows_how_close_the_watch_papers_usually_are(self):
        saved = self.create_saved(mute=True)
        dislikes = np.array([[1, 0, 0]], dtype=np.float32)
        few = {'a': np.array([1, 0, 0], dtype=np.float32)}
        self.assertEqual(list(saved_searches.mute_floors(saved, dislikes, few, 3)), [saved_searches.MUTE_FALLBACK])
        dense = {str(i): np.array([0.7, 0.714, 0], dtype=np.float32) for i in range(6)}  # a field where papers are close
        self.assertAlmostEqual(float(saved_searches.mute_floors(saved, dislikes, dense, 3)[0]), 0.82, places=2)
        sparse = {str(i): np.array([0.3, 0.954, 0], dtype=np.float32) for i in range(6)}
        self.assertAlmostEqual(float(saved_searches.mute_floors(saved, dislikes, sparse, 3)[0]), saved_searches.MUTE_FLOOR)

    def test_without_the_model_the_expression_and_the_title_rank_first(self):
        saved = self.create_saved(params={'q': 'galaxy cluster', 'field': 'all', 'src': 'arxiv', 'type': ''})
        rows = [dict(paper('star'), title='The oldest star cluster in the Galaxy', abstract='A cluster.'),
                dict(paper('abstract'), title='Radio halos', abstract='We study a galaxy cluster merger.'),
                dict(paper('title'), title='Galaxy clusters in KiDS', abstract='Weak lensing.')]
        self.assertEqual([r['key'] for r in saved_searches.rank(saved, rows)], ['title', 'abstract', 'star'])


class AnonymousSearchTests(TestCase):
    """Visitors without an account share part of the day's OpenAlex budget, and have an hourly limit of their own."""

    def setUp(self):
        cache.clear()
        self.arxiv = mock.patch.object(unified_search, 'arxiv_results', return_value=sources.SearchRows([paper('2610.00001v1', arxiv_id='2610.00001v1')], has_more=False))
        self.arxiv.start()
        self.addCleanup(self.arxiv.stop)

    def openalex(self):
        return mock.patch('summarizer.sources._json', return_value={'results': [], 'meta': {'count': 0}})

    def test_past_the_anonymous_share_visitors_get_arxiv_and_an_account_offer_while_accounts_keep_everything(self):
        with mock.patch.object(sources, 'OPENALEX_ANONYMOUS_PER_DAY', 1), self.openalex() as openalex, \
                mock.patch.object(sources, 'crossref_search') as crossref:
            page = self.client.get('/en/search-results/?q=climate')
            self.assertNotContains(page, 'Without an account, the search is limited')
            self.assertContains(page, 'A free account gives full access')
            page = self.client.get('/en/search-results/?q=oceans')
            self.assertContains(page, 'Without an account, the search is limited')
            self.assertContains(page, '/en/register/?next=')
            self.assertNotContains(page, 'Some sources are temporarily unavailable')
            self.assertEqual(openalex.call_count, 1)
            crossref.assert_not_called()  # scrapers must not move to Crossref
            self.assertEqual(self.client.get('/en/search-journals/?q=nature').status_code, 503)
            user = User.objects.create_user('reader', 'reader@example.com', 'x')
            self.client.force_login(user)
            page = self.client.get('/en/search-results/?q=oceans')  # the visitor's arXiv-only page was not cached
            self.assertNotContains(page, 'Without an account')
            self.assertEqual(openalex.call_count, 2)

    @override_settings(THROTTLE=True)
    def test_an_address_without_an_account_has_an_hourly_limit(self):
        from . import access
        with mock.patch.object(access, 'ANONYMOUS_SEARCHES_PER_HOUR', 1), self.openalex() as openalex:
            self.assertNotContains(self.client.get('/en/search-results/?q=climate'), 'Without an account, the search is limited')
            self.assertContains(self.client.get('/en/search-results/?q=oceans'), 'Without an account, the search is limited')
        self.assertEqual(openalex.call_count, 1)
