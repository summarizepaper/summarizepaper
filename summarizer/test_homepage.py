from unittest import mock

import datetime
import json

import stripe
from bs4 import BeautifulSoup
from django.conf import settings
from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings

from . import billing
from .models import ArxivPaper, DigestItem, DigestSubscription, Letter, SummaryPaper


class HomepageTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_bilingual_seo_and_free_entry_points(self):
        for lang, title in [('en', 'Your AI research companion'), ('fr', 'Votre compagnon de recherche scientifique')]:
            response = self.client.get('/%s/' % lang)
            soup = BeautifulSoup(response.content, 'html.parser')
            self.assertEqual([h.get_text() for h in soup.select('h1')], [title])
            self.assertEqual(soup.select_one('link[rel=canonical]')['href'], 'https://www.summarizepaper.com/%s' % ('' if lang == 'en' else 'fr/'))
            self.assertEqual({a['hreflang']: a['href'] for a in soup.select('link[hreflang]')},
                             {'en': 'https://www.summarizepaper.com/', 'fr': 'https://www.summarizepaper.com/fr/',
                              'x-default': 'https://www.summarizepaper.com/'})
            self.assertEqual(soup.select_one('input[name=arxiv_id]')['id'], 'arxiv-id')
            self.assertIsNotNone(soup.select_one('a[href="/%s/digest/"]' % lang))
            self.assertIsNotNone(soup.select_one('a[href="/%s/missed/"]' % lang))
            self.assertContains(response, 'application/ld+json')
            self.assertNotContains(response, '€7')  # No invented price when Stripe is unconfigured.

    def test_free_tools_lead_to_their_pages_and_signup_is_only_for_visitors(self):
        for lang in ('en', 'fr'):
            soup = BeautifulSoup(self.client.get('/%s/' % lang).content, 'html.parser')
            links = [a['href'] for a in soup.select('#free-tools .home-tool h3 a')]
            self.assertEqual(links, ['#arxiv-id', '/%s/chat/' % lang, '#email-updates'] +
                             ['/%s/%s/' % (lang, p) for p in ('letters', 'trends', 'saved-searches')])
            self.assertEqual(soup.select_one('.home-signup a.btn')['href'], '/%s/register/' % lang)
        self.client.force_login(User.objects.create_user('member', 'member@example.com', 'pass'))
        self.assertNotContains(self.client.get('/en/'), 'home-signup')

    def test_reader_count_is_shown_only_once_meaningful(self):
        from . import homepage
        with mock.patch.object(homepage, 'READERS_SHOWN_FROM', 2):
            for i in range(3):
                user = User.objects.create_user('reader%d' % i, 'r%d@example.com' % i, 'pass')
                DigestSubscription.objects.create(user=user, categories='cs.AI', active=i < 1)
            self.assertNotContains(self.client.get('/en/'), 'home-readers')
            cache.clear()
            DigestSubscription.objects.update(active=True)
            self.assertContains(self.client.get('/en/'), 'Already read by more than 3 researchers.')

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

    def test_hero_previews_real_summaries_by_their_first_point(self):
        paper = ArxivPaper.objects.create(arxiv_id='2610.00001v1', title='Quantum   magnets', category='quant-ph')
        SummaryPaper.objects.create(paper=paper, lang='fr', lay_summary='<ol><li>1. Des aimants minuscules.</li><li>2. Un second point.</li></ol>')
        soup = BeautifulSoup(self.client.get('/fr/').content, 'html.parser')
        item = soup.select_one('.home-latest li')
        self.assertEqual(item.select_one('a')['href'], '/fr/paper/2610.00001v1/')
        self.assertEqual(item.select_one('a').get_text(), 'Quantum magnets')
        self.assertEqual(item.select_one('p').get_text(), 'Des aimants minuscules.')
        self.assertEqual(len(soup.select('#premium-title')), 1)

    def test_email_samples_show_a_real_digest_and_letter_of_each_field(self):
        soup = BeautifulSoup(self.client.get('/en/').content, 'html.parser')
        self.assertIsNone(soup.select_one('.home-samples'))
        self.assertIsNotNone(soup.select_one('#digest-example'))  # the illustrative email stays until there is a real one
        cache.clear()
        when = datetime.datetime(2026, 11, 2, tzinfo=datetime.timezone.utc)
        DigestItem.objects.create(arxiv_id='2611.00001', lang='en', title='Old   AI paper', categories='cs.LG',
                                  blurb='An older blurb.', published=when - datetime.timedelta(days=1))
        DigestItem.objects.create(arxiv_id='2611.00002', lang='en', title='New AI paper', authors='A. Author',
                                  categories='cs.CL,astro-ph.IM', blurb='A short summary.', published=when)
        DigestItem.objects.create(arxiv_id='2611.00003', lang='en', title='Cross-listed', categories='astro-ph.GA,cs.LG',
                                  blurb='Another field.', published=when)
        DigestItem.objects.create(arxiv_id='2611.00004', lang='fr', title='Un article', categories='cs.LG',
                                  blurb='Un résumé.', published=when)
        content = {'sections': [{'heading': 'Models that plan', 'text': 'Text.', 'papers': ['2611.00002', '2611.00001']},
                                {'heading': 'Robots', 'text': 'More text.', 'papers': ['2611.00005']}],
                   'papers': {}, 'closing': ''}
        Letter.objects.create(topic='ai', lang='en', day=datetime.date(2026, 11, 2), title='AI this week',
                              dek='What changed.', content=json.dumps(content))
        soup = BeautifulSoup(self.client.get('/en/').content, 'html.parser')
        panels = {p['data-topic']: p for p in soup.select('.home-sample-field')}
        self.assertEqual(set(panels), {'ai', 'space'})
        self.assertFalse(panels['ai'].has_attr('hidden'))
        self.assertTrue(panels['space'].has_attr('hidden'))
        papers = panels['ai'].select('.home-sample-papers li')
        # only papers whose main category is in the field, newest first
        self.assertEqual([li.select_one('a').get_text() for li in papers], ['New AI paper', 'Old AI paper'])
        self.assertEqual(papers[0].select_one('a')['href'], '/en/paper/2611.00002/')
        self.assertEqual(papers[0].select_one('span').get_text(), 'cs.CL')
        self.assertEqual(papers[0].select_one('p').get_text(), 'A short summary.')
        self.assertIn('/en/digest/example/?field=ai', str(panels['ai']))
        letter = panels['ai'].select('.home-sample')[1]
        self.assertEqual(letter.select_one('strong').get_text(), 'AI this week')
        self.assertIn('3 papers put into perspective', letter.get_text())
        self.assertEqual([li.get_text() for li in letter.select('.home-sample-sections li')], ['Models that plan', 'Robots'])
        self.assertEqual(letter.select('a')[-1]['href'], '/en/letters/ai/2026-11-02/')
        self.assertIsNone(soup.select_one('#digest-example'))
        # Each language shows its own: no French letter yet, only the French digest item.
        soup = BeautifulSoup(self.client.get('/fr/').content, 'html.parser')
        self.assertEqual([a.get_text() for a in soup.select('.home-sample-field[data-topic=ai] .home-sample-papers a')], ['Un article'])
        self.assertEqual(len(soup.select('.home-sample-field')), 1)

    def test_a_field_with_a_whole_sample_is_shown_first(self):
        when = datetime.datetime(2026, 11, 2, tzinfo=datetime.timezone.utc)
        DigestItem.objects.create(arxiv_id='2611.00001', lang='en', title='AI', categories='cs.LG', blurb='B.', published=when)
        for i in range(3):
            DigestItem.objects.create(arxiv_id='2611.0010%d' % i, lang='en', title='Star %d' % i, categories='astro-ph.SR',
                                      blurb='About a star.', published=when)
        content = {'sections': [{'heading': 'Stars', 'text': 'T.', 'papers': ['2611.00100']}], 'papers': {}, 'closing': ''}
        Letter.objects.create(topic='space', lang='en', day=datetime.date(2026, 11, 2), title='Space this week',
                              dek='Stars.', content=json.dumps(content))
        soup = BeautifulSoup(self.client.get('/en/').content, 'html.parser')
        shown = [p for p in soup.select('.home-sample-field') if not p.has_attr('hidden')]
        self.assertEqual([p['data-topic'] for p in shown], ['space'])
        self.assertIn('“Astronomy and space”', shown[0].get_text())

    def test_samples_are_prepared_once_per_announcement_for_every_field(self):
        from . import homepage
        from .models import AnnouncedPaper, ArxivBatch
        day = datetime.date(2026, 11, 3)
        ArxivBatch.objects.create(day=day)
        for aid, categories in [('2611.1', 'cs.LG'), ('2611.2', 'astro-ph.GA,cs.LG'), ('2611.3', 'quant-ph'),
                                ('2611.4', 'cs.CV'), ('2611.5', 'stat.ML'), ('2611.6', 'cs.AI')]:
            AnnouncedPaper.objects.create(arxiv_id=aid, announced=day, title=aid, abstract='An abstract.',
                                          authors='["A"]', categories=categories)
        with mock.patch('summarizer.digest.write_blurbs',
                        side_effect=lambda batch, lang: {p['arxiv_id']: {'summary': 'S.', 'points': {}} for p in batch}) as blurbs:
            self.assertEqual(homepage.prepare_samples(), 6)  # four AI papers, the astronomy one, the quantum one
            self.assertEqual(homepage.prepare_samples(), 0)  # once per announcement
        self.assertEqual(blurbs.call_count, 2)  # one call per language
        self.assertEqual(set(DigestItem.objects.filter(lang='fr').values_list('arxiv_id', flat=True)),
                         {'2611.1', '2611.4', '2611.5', '2611.6', '2611.2', '2611.3'})

    def test_example_digest_page_shows_the_email_of_a_field(self):
        when = datetime.datetime(2026, 11, 2, tzinfo=datetime.timezone.utc)
        DigestItem.objects.create(arxiv_id='2611.00001', lang='en', title='Quantum paper', categories='quant-ph',
                                  blurb='A blurb.', points=json.dumps({'result': 'It works.'}), published=when)
        response = self.client.get('/en/digest/example/?field=quantum')
        soup = BeautifulSoup(response.content, 'html.parser')
        self.assertIn('Quantum physics', soup.select_one('h1').get_text())
        email = BeautifulSoup(soup.select_one('iframe')['srcdoc'], 'html.parser')
        self.assertEqual(email.select_one('.it .t').get_text(), 'Quantum paper')
        self.assertIn('It works.', email.select_one('.it .p').get_text())
        self.assertIn('/en/email-updates/?topic=quantum', [a['href'] for a in email.select('a')])
        self.assertEqual(soup.select_one('.digest-example-cta a')['href'], '/en/email-updates/?topic=quantum')
        empty = BeautifulSoup(self.client.get('/en/digest/example/?field=nope').content, 'html.parser')
        self.assertIsNone(empty.select_one('iframe'))  # an unknown field falls back to AI, which has no paper yet

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
