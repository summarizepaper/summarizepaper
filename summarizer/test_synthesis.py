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
        page = self.client.get('/fr/arxiv-id/2401.04088v1/')
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
            r = self.client.get('/fr/arxiv-id/2401.04088v1/')
        html = r.content.decode()
        self.assertIn('Abstract et métadonnées uniquement', html)
        self.assertLess(html.index('class="source-scope"'), html.index('name="run_button"'))
