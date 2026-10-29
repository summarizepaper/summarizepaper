from unittest import mock
from asgiref.sync import async_to_sync
from bs4 import BeautifulSoup
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import translation
from . import synthesis, utils
from .models import ArxivPaper, PaperText, SummaryPaper


@override_settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}})
class SynthesisTests(TestCase):
    def setUp(self):
        cache.clear()
        self.paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Example research',
            license='http://creativecommons.org/licenses/by/4.0/')
        self.group = utils.run_group(self.paper.arxiv_id, 'fr')

    def test_snapshots_are_public_language_specific_and_read_only(self):
        cache.set(self.group, True)
        synthesis.remember(self.group, 'summary', 'Résumé prêt')
        with mock.patch('summarizer.consumers.LoadingConsumer.sendmessages') as start:
            r = self.client.get('/fr/paper-live/2401.04088v1/')
            self.assertEqual(r.json(), {'summary': 'Résumé prêt', 'status': 'running'})
            self.assertEqual(self.client.get('/en/paper-live/2401.04088v1/').json(), {'status': 'idle'})
            self.assertEqual(self.client.get('/fr/paper-live/unknown/').status_code, 404)
            self.assertEqual(self.client.post('/fr/paper-live/2401.04088v1/').status_code, 405)
            start.assert_not_called()
        self.assertEqual(r['Cache-Control'], 'no-store')

    def test_progress_snapshot_is_copied_and_completion_survives_unlock(self):
        from .consumers import HeadlessSummary
        consumer = HeadlessSummary(self.paper.arxiv_id, 'fr')
        consumer.channel_layer = mock.Mock(group_send=mock.AsyncMock())
        message = {'progress': 50, 'loading_message': 'Résumé'}
        async_to_sync(consumer.send_message_now)(message)
        message['progress'] = 100
        self.assertEqual(synthesis.snapshot(self.paper.arxiv_id, 'fr')['progress']['progress'], 50)
        async_to_sync(consumer.send_message_now)(message)
        cache.set(self.group, False)
        self.assertEqual(synthesis.snapshot(self.paper.arxiv_id, 'fr')['status'], 'done')
        async_to_sync(consumer.send_message_blog)('<script>alert(1)</script><p onclick="bad()">Safe</p>')
        blog = synthesis.snapshot(self.paper.arxiv_id, 'fr')['blog']
        self.assertIn('Safe', blog)
        self.assertNotIn('script', blog)
        self.assertNotIn('onclick', blog)

    def test_disconnect_preserves_background_lock(self):
        from .consumers import HeadlessSummary
        consumer = HeadlessSummary(self.paper.arxiv_id, 'fr')
        consumer.channel_name = 'test'
        consumer.channel_layer = mock.Mock(group_discard=mock.AsyncMock())
        consumer.send = mock.AsyncMock()
        cache.set(self.group, True)
        async_to_sync(consumer.disconnect)(1000)
        self.assertTrue(cache.get(self.group))
        consumer.channel_layer.group_discard.assert_awaited_once()

    def test_source_scope_distinguishes_permission_from_available_text(self):
        self.assertEqual(synthesis.license_scope(None), 'checking')
        self.assertEqual(synthesis.license_scope(self.paper), 'full_text_allowed')
        text = PaperText.objects.create(arxiv_id=self.paper.arxiv_id, source='abstract', text='Example')
        self.assertEqual(synthesis.license_scope(self.paper), 'abstract_unavailable')
        text.source = 'html'; text.save()
        self.assertEqual(synthesis.license_scope(self.paper), 'full_text')
        self.paper.license = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
        self.assertEqual(synthesis.license_scope(self.paper), 'abstract_license')

    def test_completed_fragment_bypasses_running_page_and_contains_working_controls(self):
        SummaryPaper.objects.create(paper=self.paper, lang='fr', summary='Résumé prêt',
            notes='["Point"]', lay_summary='Simple', blog='<p onclick="bad()">Blog</p><script>bad()</script>')
        cache.set(self.group, True)
        page = self.client.get('/fr/paper/2401.04088v1/')
        self.assertContains(page, 'id="synthesis-live"')
        result = self.client.get('/fr/paper-live/2401.04088v1/detail/')
        self.assertEqual(result.status_code, 200)
        html = result.content.decode(); soup = BeautifulSoup(html, 'html.parser')
        self.assertIsNotNone(soup.select_one('#paper-detail'))
        for selector in ['#results', '#download-form', '.chatbot-form', '#tools', '.chat-figures', '.vote-actions']:
            self.assertIsNotNone(soup.select_one(selector), selector)
        self.assertLess(html.index('id="results"'), html.index('id="assistant"'))
        self.assertIn('data-base="/fr"', html)
        self.assertIn('/fr/journal-club/2401.04088v1/', html)
        self.assertIn('Sources de la synthèse', html)
        self.assertNotIn('bad()', soup.select_one('#results').get_text())
        self.assertEqual(result['Cache-Control'], 'private, no-store')
        self.assertEqual(self.client.get('/en/paper-live/2401.04088v1/detail/').status_code, 404)

    def test_pre_run_license_warning_precedes_run_button(self):
        self.paper.license = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
        self.paper.save()
        metadata = [1, ['Author'], [''], 'https://arxiv.org/abs/2401.04088', 'Example', '',
                    'Abstract', 'cs.AI', '2024-01-01T00:00:00Z', '2024-01-01T00:00:00Z', '', '', self.paper.license]
        with mock.patch('summarizer.views.arxiv_metadata_or_none', return_value=metadata):
            r = self.client.get('/fr/paper/2401.04088v1/')
        html = r.content.decode()
        self.assertIn('Abstract et métadonnées uniquement', html)
        self.assertLess(html.index('class="source-scope"'), html.index('name="run_button"'))


class SummaryPresentationTests(TestCase):
    def test_legacy_glued_heading_numbered_lists_paragraphs_and_definitions(self):
        from .summary_text import lay_html
        html = lay_html('Résumé 1. Des rochers apportent de l’eau.\n\n2. La vapeur voyage.\n\nDéfinitions :\n- **Vapeur :** De l’eau gazeuse.\n\nUn dernier paragraphe.')
        soup = BeautifulSoup(html, 'html.parser')
        self.assertEqual([li.get_text() for li in soup.select('ol > li')], ['Des rochers apportent de l’eau.', 'La vapeur voyage.'])
        self.assertEqual(len(soup.select('ol')), 1)
        self.assertEqual(soup.h3.get_text(), 'Définitions')
        self.assertEqual(soup.select_one('ul strong').get_text(), 'Vapeur :')
        self.assertEqual(soup.p.get_text(), 'Un dernier paragraphe.')
        self.assertNotIn('Résumé 1.', soup.get_text())

    def test_inline_numbering_does_not_split_decimals_or_execute_markup(self):
        from .summary_text import lay_html
        soup = BeautifulSoup(lay_html('Summary: 1. At 1.3 mm, $M_{\\oplus}$ is measured. 2. <img src=x onerror=bad()>'), 'html.parser')
        self.assertEqual(len(soup.select('ol > li')), 2)
        self.assertIn('1.3 mm', soup.li.get_text())
        self.assertIn('$M_{\\oplus}$', soup.li.get_text())
        self.assertIsNone(soup.img)

    def test_glued_definitions_heading_from_the_reported_paper(self):
        from .summary_text import lay_html
        soup = BeautifulSoup(lay_html('Résumé 1. Une idée.\n\nDéfinitions - Étoile : du gaz chaud.\n- Planète : un astre.'), 'html.parser')
        self.assertEqual(soup.h3.get_text(), 'Définitions')
        self.assertEqual(len(soup.select('ul > li')), 2)
        self.assertEqual(soup.select_one('ul > li strong').get_text(), 'Étoile :')

    def test_existing_results_keep_bullets_and_format_lay_summary_without_arxiv(self):
        paper = ArxivPaper.objects.create(arxiv_id='2607.11677v1', title='Public test')
        SummaryPaper.objects.create(paper=paper, lang='fr', notes='- **Résultat :** Un point.\n- Une limite.',
                                   lay_summary='Résumé 1. Une idée simple.\n2. Une autre idée.')
        with mock.patch('summarizer.utils.get_arxiv_metadata', side_effect=AssertionError('should use stored paper')):
            r = self.client.get('/fr/paper/2607.11677v1/')
        self.assertEqual(r.status_code, 200)
        soup = BeautifulSoup(r.content, 'html.parser')
        self.assertEqual(len(soup.select('.paper-key-points > li')), 2)
        self.assertEqual(len(soup.select('.paper-lay-summary ol > li')), 2)
        self.assertNotIn('list-style: none', str(soup.select_one('.paper-key-points')))
        self.assertNotIn('ASSUMED 1991-2003', r.content.decode())

    def test_source_error_is_translated_in_french(self):
        import asyncio
        with mock.patch('summarizer.utils.get_arxiv_metadata', side_effect=asyncio.TimeoutError):
            r = self.client.get('/fr/paper/2607.99999v1/')
        self.assertContains(r, 'Le service arXiv est temporairement indisponible.', status_code=503)
        self.assertNotContains(r, 'arXiv is not responding', status_code=503)

    def test_live_snapshot_and_socket_share_the_safe_lay_format(self):
        from .consumers import HeadlessSummary
        cache.clear()
        consumer = HeadlessSummary('2607.11677v1', 'fr')
        consumer.channel_layer = mock.Mock(group_send=mock.AsyncMock())
        async_to_sync(consumer.send_message_laysum)('Résumé 1. Un résultat.\n2. Une limite.')
        data = synthesis.snapshot('2607.11677v1', 'fr')
        event = consumer.channel_layer.group_send.call_args.args[1]
        self.assertEqual(event['html'], data['lay_summary_html'])
        self.assertIn('<ol>', event['html'])


class ArxivMetadataResilienceTests(TestCase):
    ATOM = '''<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom"><title>Query</title><entry>
    <id>http://arxiv.org/abs/2607.11677v1</id><title>Paper title</title><summary>Public abstract</summary>
    <updated>2026-07-01T00:00:00Z</updated><published>2026-07-01T00:00:00Z</published>
    <author><name>Example</name></author><arxiv:primary_category term="astro-ph.EP"/>
    </entry></feed>'''

    def setUp(self):
        cache.clear()

    def response(self, body, status=200, error=None):
        response = mock.Mock(status=status, text=mock.AsyncMock(return_value=body))
        manager = mock.MagicMock()
        manager.__aenter__ = mock.AsyncMock(return_value=response, side_effect=error)
        manager.__aexit__ = mock.AsyncMock(return_value=False)
        return manager

    def session(self, responses):
        session = mock.Mock(get=mock.Mock(side_effect=responses))
        manager = mock.MagicMock()
        manager.__aenter__ = mock.AsyncMock(return_value=session)
        manager.__aexit__ = mock.AsyncMock(return_value=False)
        return manager, session

    def test_license_timeout_or_invalid_xml_does_not_hide_valid_metadata(self):
        import asyncio
        for oai in [self.response('', error=asyncio.TimeoutError()), self.response('<html>down'), self.response('', status=503)]:
            cache.clear()
            manager, session = self.session([self.response(self.ATOM), oai])
            with mock.patch.object(utils.aiohttp, 'ClientSession', return_value=manager):
                data = async_to_sync(utils.get_arxiv_metadata)('2607.11677v1')
            self.assertEqual(data[0], 1)
            self.assertEqual(data[4], 'Paper title')
            self.assertEqual(data[12], '')
            self.assertFalse(utils.sources.is_open_license(data[12]))

    def test_positive_records_are_cached_and_no_license_is_fabricated(self):
        manager, session = self.session([self.response(self.ATOM), self.response('<oai/>')])
        with mock.patch.object(utils.aiohttp, 'ClientSession', return_value=manager) as connect:
            first = async_to_sync(utils.get_arxiv_metadata)('2607.11677v1')
            second = async_to_sync(utils.get_arxiv_metadata)('2607.11677v1')
        self.assertEqual(first, second)
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(first[12], '')

    def test_primary_http_errors_stop_before_license_lookup_and_are_not_cached(self):
        manager, session = self.session([self.response('<html>Rate limited</html>', status=429)])
        with mock.patch.object(utils.aiohttp, 'ClientSession', return_value=manager):
            with self.assertRaises(utils.sources.SourceUnavailable):
                async_to_sync(utils.get_arxiv_metadata)('2607.11677v1')
        self.assertEqual(session.get.call_count, 1)
        self.assertIsNone(cache.get('arxiv-metadata-v2:2607.11677v1'))

    def test_known_license_can_be_reused_without_contacting_the_license_service(self):
        license = 'https://creativecommons.org/licenses/by/4.0/'
        cache.set('arxiv-license:2607.11677', license)
        manager, session = self.session([self.response(self.ATOM)])
        with mock.patch.object(utils.aiohttp, 'ClientSession', return_value=manager):
            result = async_to_sync(utils.get_arxiv_metadata)('2607.11677v1')
        self.assertEqual(result[12], license)
        self.assertTrue(utils.sources.is_open_license(result[12]))
        self.assertEqual(session.get.call_count, 1)


class PaperReadingTests(TestCase):
    TEXT = ('## Results\n\nWe measure a cooling rate of 3.2 K per hour in the inner disk. '
            'This suggests that the disk is losing mass to the star.')

    def setUp(self):
        from .models import Author, PaperAuthor
        self.paper = ArxivPaper.objects.create(arxiv_id='2401.05000v1', title='Cooling of a disk', abstract='A disk cools.',
                                               license='http://creativecommons.org/licenses/by/4.0/')
        PaperAuthor.objects.create(paper=self.paper, author=Author.objects.create(name='Ann Vries'), author_order=0)
        SummaryPaper.objects.create(paper=self.paper, lang='en', summary='Long summary.', notes='["Point A", "Point B"]',
                                    lay_summary='Simple.', keywords='disk cooling, accretion',
                                    blog='<p>A recent paper by M. Smith et al., published in Nature Physics, studies disks. '
                                         'It finds that disks cool.</p><p>Ann Vries et al. measure the rate.</p>')

    def answer(self):
        return {'tldr': 'The inner disk cools fast.', 'question': 'How fast do disks cool?', 'method': 'Simulations.',
                'findings': [{'text': 'Cooling at 3.2 K per hour.', 'kind': 'result',
                              'quote': 'We measure a cooling rate of 3.2 K per hour in the inner disk.'},
                             {'text': 'Mass is lost.', 'kind': 'speculation', 'quote': 'An invented passage that is not in the paper at all.'}],
                'limitations': [{'text': 'One disk only.', 'quote': ''}], 'significance': 'Disk models change.'}

    def test_passages_are_kept_only_when_they_are_in_the_paper(self):
        from . import insight
        data = insight.structured(self.answer(), self.TEXT, False)
        self.assertEqual(data['findings'][0]['quote'], 'We measure a cooling rate of 3.2 K per hour in the inner disk.')
        self.assertEqual(data['findings'][1]['quote'], '')
        self.assertEqual(data['findings'][1]['kind'], 'interpretation')

    def test_new_synthesis_shows_the_reading_once_and_old_ones_keep_key_points(self):
        from . import insight
        from .models import Generated
        page = BeautifulSoup(self.client.get('/en/paper/2401.05000v1/').content, 'html.parser')
        self.assertEqual(page.h1.get_text(strip=True), 'Cooling of a disk')
        self.assertEqual([li.get_text(strip=True) for li in page.select('.paper-key-points > li')], ['Point A', 'Point B'])
        self.assertIn('predates the recording of its source', page.select_one('.source-scope').get_text())
        with mock.patch('summarizer.generated.chat_json', return_value=self.answer()), \
             mock.patch('summarizer.paper_source.paper_text', return_value=self.TEXT):
            insight.create('2401.05000v1', 'en', False)
        self.assertEqual(Generated.objects.get(kind='insight').status, 'done')
        page = BeautifulSoup(self.client.get('/en/paper/2401.05000v1/').content, 'html.parser')
        self.assertEqual(page.select_one('.tldr').get_text(), 'The inner disk cools fast.')
        self.assertEqual(len(page.select('.evidence blockquote')), 1)
        self.assertEqual(page.select_one('.claim-result').get_text(), 'Result')
        self.assertNotIn('Point A', page.get_text())  # the reading replaces the key points, no repetition
        self.assertIn('/en/search-results/?q=disk%20cooling', str(page.select_one('.paper-actions')))

    def test_blogs_lose_invented_authors_and_journals_but_keep_the_rest(self):
        page = BeautifulSoup(self.client.get('/en/paper/2401.05000v1/').content, 'html.parser')
        blog = page.select('.message-tab')[2].get_text(' ')
        self.assertNotIn('Smith', blog)
        self.assertNotIn('Nature Physics', blog)
        self.assertIn('It finds that disks cool.', blog)
        self.assertIn('Ann Vries et al. measure the rate.', blog)

    def test_blog_prompt_names_the_real_title_and_authors(self):
        import json as _json
        response = mock.Mock(status_code=200)
        response.json.return_value = {'choices': [{'message': {'content': '<p>Blog</p>'}, 'finish_reason': 'stop'}],
                                      'usage': {'prompt_tokens': 1, 'completion_tokens': 1}, 'model': 'm'}
        with mock.patch('summarizer.utils.requests.post', return_value=response) as post:
            async_to_sync(utils.extract_blog_article)('2401.05000v1', 'en', 'Summary.', False, 'key',
                                                      title='Cooling of a disk', authors=['Ann Vries'])
        prompt = _json.dumps(post.call_args.kwargs['json']['messages'])
        self.assertIn('Cooling of a disk', prompt)
        self.assertIn('Ann Vries', prompt)
        self.assertIn('Name no other author', prompt)

    def test_venue_check_keeps_real_journals_initials_and_figures(self):
        from .summary_text import without_invented_references
        html = ('<p>Gavin A. L. Coleman et Haworth ont publié leur étude dans MNRAS. Elle a été publiée dans MNRAS en 2020. '
                'The results were published in a paper titled “Discs”. The results presented in Figure 2 hold.</p>')
        out = without_invented_references(html, ['Gavin A. L. Coleman', 'Thomas J. Haworth'], '', 'accepted for publication in MNRAS')
        self.assertEqual(BeautifulSoup(out, 'html.parser').get_text(), BeautifulSoup(html, 'html.parser').get_text())


class SummarySourceRefreshTests(TestCase):
    def test_rerun_replaces_old_source_before_the_summary_reads_it(self):
        from .consumers import HeadlessSummary
        paper = ArxivPaper.objects.create(arxiv_id='doi_10.1038--s41550-026-03004-3',
            title='Two debris discs', abstract='Only the abstract.',
            license='https://creativecommons.org/licenses/by/4.0/')
        consumer = HeadlessSummary(paper.arxiv_id, 'fr')
        consumer.send_source = mock.AsyncMock()
        consumer.send_message_now = mock.AsyncMock()

        class StopBeforePaidCall(Exception):
            pass

        for previous, fresh, origin, scope in [
            ('abstract', 'FULL METHODS AND RESULTS ' * 200, 'pdf', 'full_text'),
            ('pdf', 'Only the abstract.', 'abstract', 'abstract_unavailable'),
        ]:
            with self.subTest(previous=previous, origin=origin):
                PaperText.objects.update_or_create(arxiv_id=paper.arxiv_id,
                    defaults={'source': previous, 'text': 'OLD SOURCE'})
                async def inspect_source(*args):
                    # The real detailed_summary reloads context from this database row.
                    from asgiref.sync import sync_to_async
                    stored = await sync_to_async(PaperText.objects.get)(arxiv_id=paper.arxiv_id)
                    self.assertEqual(stored.source, origin)
                    self.assertIn(fresh.strip(), stored.text)
                    self.assertNotIn('OLD SOURCE', stored.text)
                    raise StopBeforePaidCall
                with mock.patch('summarizer.utils.read_paper', mock.AsyncMock(return_value=(fresh, origin))), \
                     mock.patch('summarizer.utils.summarize_book', side_effect=inspect_source), \
                     self.assertRaises(StopBeforePaidCall):
                    async_to_sync(consumer.computesummary)(paper.arxiv_id, 'fr',
                        [paper.license, paper.title, paper.abstract, []], {}, 'HEADER', True)
                consumer.send_source.assert_awaited_with(scope)
                self.assertEqual(synthesis.license_scope(paper), scope)
