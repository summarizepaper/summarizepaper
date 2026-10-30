from io import StringIO
from unittest import mock

from django.core.management import call_command
from django.test import Client
from asgiref.sync import async_to_sync

from summarizer import activity, generated, insight, letters
from summarizer.models import ArxivPaper, Generated, Letter, PaperText
from summarizer.test_tools import ToolsTestCase


class MissingSourceTests(ToolsTestCase):
    def test_missing_abstract_is_unavailable_without_an_ai_call(self):
        paper = self.paper('doi_10.1007--s11760-026-05575-0', title='Metadata only', abstract='')
        paper.license = ''; paper.save()
        PaperText.objects.create(arxiv_id=paper.arxiv_id, source='abstract', text='Title: Metadata only. Abstract: .')
        with mock.patch('summarizer.generated.chat_json') as chat:
            row = generated.run_now('insight', paper.arxiv_id, 'en', insight.write(paper.arxiv_id, 'en', False))
        self.assertEqual(row.status, 'no_source')
        chat.assert_not_called()
        self.assertIsNone(insight.for_page(paper.arxiv_id, 'en'))
        self.assertNotIn(row.key, [r['key'] for r in activity.watch()['failed']])
        self.assertIn(row.key, [r['key'] for r in activity.watch()['unavailable']])
        response = Client().get('/en/paper/' + paper.arxiv_id + '/')
        self.assertContains(response, 'The structured reading is unavailable')
        self.assertContains(response, 'Previously generated content cannot be checked')

    def test_open_metadata_wrapper_is_not_full_text(self):
        paper = self.paper(abstract='')
        PaperText.objects.create(arxiv_id=paper.arxiv_id, source='abstract', text='Title and authors only.')
        self.assertEqual(insight.source_text(paper), ('', True))

    def test_real_abstract_is_still_used_when_full_text_is_not_permitted(self):
        paper = self.paper(abstract='A real abstract with findings.')
        paper.license = ''; paper.save()
        self.assertEqual(insight.source_text(paper), (paper.abstract, True))

    def test_new_summary_stops_before_indexing_or_ai_when_source_is_missing(self):
        from summarizer.consumers import HeadlessSummary
        paper = self.paper(abstract='')
        consumer = HeadlessSummary(paper.arxiv_id, 'fr')
        consumer.send_source = mock.AsyncMock()
        consumer.send_message_now = mock.AsyncMock()
        with mock.patch('summarizer.utils.createindex', mock.AsyncMock()) as index, \
             mock.patch('summarizer.utils.summarize_book', mock.AsyncMock()) as summarize:
            with self.assertRaises(generated.SourceUnavailable):
                async_to_sync(consumer.computesummary)(paper.arxiv_id, 'fr', ['', paper.title, '', []], {}, '', False)
        index.assert_not_called()
        summarize.assert_not_called()

    def test_real_errors_remain_failures(self):
        def work(pk): raise TimeoutError('network')
        row = generated.run_now('insight', 'example', 'en', work)
        self.assertEqual(row.status, 'failed')
        self.assertTrue(activity.watch()['failed'])


class StoredLetterTitleTests(ToolsTestCase):
    def test_stored_titles_are_fixed_without_regeneration_or_email(self):
        import datetime
        row = Letter.objects.create(topic='ai', lang='fr', day=datetime.date(2026, 10, 26),
                                    title='Intelligence artificielle this week : les résultats', content='{}')
        english = Letter.objects.create(topic='ai', lang='en', day=row.day, title='Artificial intelligence this week', content='{}')
        with mock.patch('summarizer.letters.write') as write, mock.patch('summarizer.letters.send') as send:
            call_command('letters_normalize_titles', dry_run=True, stdout=StringIO())
            row.refresh_from_db(); self.assertIn('this week', row.title)
            call_command('letters_normalize_titles', stdout=StringIO())
            write.assert_not_called(); send.assert_not_called()
        row.refresh_from_db(); english.refresh_from_db()
        self.assertEqual(row.title, 'Intelligence artificielle cette semaine\u00a0: les résultats')
        self.assertEqual(english.title, 'Artificial intelligence this week')
        self.assertContains(Client().get('/fr/letters/'), row.title)
