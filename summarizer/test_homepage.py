from unittest import mock

import stripe
from bs4 import BeautifulSoup
from django.conf import settings
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings

from . import billing
from .models import ArxivPaper, DigestSubscription, SummaryPaper


class HomepageTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_bilingual_seo_and_free_entry_points(self):
        for lang, title in [('en', 'Your AI research companion'), ('fr', 'Votre compagnon de recherche scientifique')]:
            response = self.client.get('/%s/' % lang)
            soup = BeautifulSoup(response.content, 'html.parser')
            self.assertEqual([h.get_text() for h in soup.select('h1')], [title])
            self.assertEqual(soup.select_one('link[rel=canonical]')['href'], 'https://www.summarizepaper.com/%s/' % lang)
            self.assertEqual({a['hreflang']: a['href'] for a in soup.select('link[hreflang]')},
                             {'en': 'https://www.summarizepaper.com/en/', 'fr': 'https://www.summarizepaper.com/fr/'})
            self.assertEqual(soup.select_one('input[name=arxiv_id]')['id'], 'arxiv-id')
            self.assertIsNotNone(soup.select_one('a[href="/%s/digest/"]' % lang))
            self.assertIsNotNone(soup.select_one('a[href="/%s/missed/"]' % lang))
            self.assertContains(response, 'application/ld+json')
            self.assertNotContains(response, '€7')  # No invented price when Stripe is unconfigured.

    def test_free_tools_lead_to_their_pages_and_signup_is_only_for_visitors(self):
        for lang in ('en', 'fr'):
            soup = BeautifulSoup(self.client.get('/%s/' % lang).content, 'html.parser')
            links = [a['href'] for a in soup.select('#free-tools .home-tool h3 a')]
            self.assertEqual(links, ['#arxiv-id'] + ['/%s/%s/' % (lang, p) for p in ('chat', 'digest', 'letters', 'trends', 'saved-searches')])
            self.assertEqual(soup.select_one('.home-signup a.btn')['href'], '/%s/register/' % lang)
        self.client.force_login(User.objects.create_user('member', 'member@example.com', 'pass'))
        self.assertNotContains(self.client.get('/en/'), 'home-signup')

    def test_example_is_a_ready_summary_in_the_current_language(self):
        paper = ArxivPaper.objects.create(arxiv_id='1706.03762v1', title='Attention Is All You Need')
        SummaryPaper.objects.create(paper=paper, lang='en', summary='A ready summary.')
        # A newer placeholder must not replace the ready summary.
        empty = ArxivPaper.objects.create(arxiv_id='2101.00001v1', title='Empty')
        SummaryPaper.objects.create(paper=empty, lang='en', summary='')
        response = self.client.get('/en/')
        self.assertEqual(response.context['example_paper'].paper_id, paper.pk)
        self.assertContains(response, 'href="/en/paper/1706.03762v1/"')
        response = self.client.get('/fr/')
        self.assertIsNone(response.context['example_paper'])
        self.assertContains(response, 'href="/fr/getting-started/"')

    def test_blank_search_never_calls_external_sources(self):
        with mock.patch('summarizer.views.sources.resolve') as resolve:
            for data in ({}, {'arxiv_id': '   '}):
                self.assertContains(self.client.post('/en/', data), 'Enter a paper link, DOI or research topic.')
        resolve.assert_not_called()

    def test_existing_search_routes_are_preserved(self):
        response = self.client.post('/fr/', {'arxiv_id': '1706.03762v1'})
        self.assertEqual(response['Location'], '/fr/paper/1706.03762v1/')
        with mock.patch('summarizer.views.sources.resolve', return_value=None):
            response = self.client.post('/en/', {'arxiv_id': 'stellar formation', 'src': 'open'})
        self.assertIn('src=open', response['Location'])
        self.assertIn('q=stellar+formation', response['Location'])
        with mock.patch('summarizer.views.sources.resolve', return_value=('doi', 'doi_10.1234_example')):
            response = self.client.post('/en/', {'arxiv_id': '10.1234/example'})
        self.assertEqual(response['Location'], '/en/paper/doi_10.1234_example/')

    def test_signup_links_retain_tool_destination(self):
        for lang in ('en', 'fr'):
            for tool in ('digest', 'refcheck', 'missed', 'scoops'):
                response = self.client.get('/%s/%s/' % (lang, tool))
                soup = BeautifulSoup(response.content, 'html.parser')
                link = soup.select_one('a[href^="/%s/register/?next="]' % lang)
                self.assertIsNotNone(link)
                self.assertEqual(link['href'], '/%s/register/?next=/%s/%s/' % (lang, lang, tool))

    @override_settings(STRIPE_SECRET_KEY='sk_test', STRIPE_PRICE_MONTHLY='price_test')
    def test_shared_price_is_localized_and_cached(self):
        with mock.patch('stripe.Price.retrieve', return_value={'unit_amount': 700, 'currency': 'eur',
                                                              'recurring': {'interval': 'month'}}) as price:
            self.assertContains(self.client.get('/en/'), '€7 / month')
            self.assertContains(self.client.get('/fr/'), '7 € / mois')
            self.assertEqual(billing.price_label(settings.STRIPE_PRICE_MONTHLY), '7.00 EUR / month')
        price.assert_called_once_with('price_test')

    @override_settings(STRIPE_SECRET_KEY='sk_test', STRIPE_PRICE_MONTHLY='price_test')
    def test_price_outage_keeps_homepage_available_and_backs_off(self):
        with mock.patch('stripe.Price.retrieve', side_effect=stripe.error.APIConnectionError('offline')) as price:
            for _ in range(2):
                response = self.client.get('/en/')
                self.assertEqual(response.status_code, 200)
                self.assertIsNone(response.context['home_price'])
                self.assertContains(response, 'See Premium tools')
        price.assert_called_once()

    def test_result_event_requires_a_stored_result(self):
        paper = ArxivPaper.objects.create(arxiv_id='2101.00001v1', title='A paper', abstract='A',
                                          license='http://creativecommons.org/licenses/by/4.0/')
        self.assertNotContains(self.client.get('/en/paper/2101.00001v1/'), 'data-funnel-view="paper_result_view"')
        SummaryPaper.objects.create(paper=paper, summary='Completed summary', lang='en')
        self.assertContains(self.client.get('/en/paper/2101.00001v1/'), 'data-funnel-view="paper_result_view"')

    def test_digest_activation_is_emitted_only_after_a_valid_save(self):
        self.client.force_login(User.objects.create_user('reader', 'reader@example.com', 'pass'))
        marker = 'data-funnel-view="digest_configured"'
        self.assertNotContains(self.client.get('/en/digest/?saved=1'), marker)
        self.assertNotContains(self.client.post('/en/digest/', {'categories': []}), marker)
        self.assertEqual(DigestSubscription.objects.count(), 0)
        response = self.client.post('/en/digest/', {'categories': ['cs.LG']})
        self.assertEqual(DigestSubscription.objects.count(), 1)
        self.assertContains(self.client.get(response['Location']), marker)
        self.assertNotContains(self.client.get(response['Location']), marker)
