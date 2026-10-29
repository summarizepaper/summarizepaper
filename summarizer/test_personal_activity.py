
from summarizer import personal_activity
from summarizer.models import AIassistant, ChatConversation, PaperHistory, PaperVisit
from summarizer.test_tools import ToolsTestCase


class ActivityTests(ToolsTestCase):
    def setUp(self):
        self.reader = self.user()
        self.other = self.user('other')
        self.client = self.client_for(self.reader)
        self.article = self.paper()

    def test_visits_deduplicate_and_never_consume_summary_quota(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        self.assertEqual(PaperVisit.objects.filter(user=self.reader).count(), 1)
        self.assertFalse(PaperHistory.objects.exists())
        response = self.client.get('/en/history/')
        self.assertEqual(len(response.context['items']), 1)
        self.assertTrue(response.context['items'][0]['visited'])
        self.assertTrue(response.context['items'][0]['chat'].endswith('#assistant'))

    def test_old_requests_remain_once_and_private(self):
        for i in range(3):
            PaperHistory.objects.create(user=self.reader, arxiv_id=self.article.arxiv_id)
        other_paper = self.paper('2101.00002v1', title='Private paper')
        personal_activity.record_visit(self.other, other_paper.arxiv_id)
        response = self.client.get('/en/history/')
        self.assertEqual(len(response.context['items']), 1)
        self.assertFalse(response.context['items'][0]['visited'])
        self.assertNotContains(response, 'Private paper')
        self.assertContains(response, 'Summary requested')

    def test_search_pagination_and_recent_order(self):
        for i in range(22):
            paper = self.paper('2102.%05dv1' % i, title='Nebula %d' % i)
            personal_activity.record_visit(self.reader, paper.arxiv_id)
        response = self.client.get('/en/history/', {'q': 'Nebula', 'page': 2})
        self.assertEqual(len(response.context['items']), 2)
        self.assertIn('q=Nebula', response.context['previous_url'])
        response = self.client.get('/en/history/', {'q': 'not found'})
        self.assertContains(response, 'No matching activity')

    def test_conversations_are_grouped_owned_active_and_resume_both_types(self):
        room = 'myconversation1234'
        for query in ('First question', 'Latest question'):
            AIassistant.objects.create(user=self.reader, arxiv_id=room, papers=self.article.arxiv_id,
                                       query=query, response='Answer')
        ChatConversation.objects.create(user=self.reader, room=room, title='My named exchange')
        AIassistant.objects.create(user=self.reader, arxiv_id=self.article.arxiv_id, query='Single paper', response='Answer')
        AIassistant.objects.create(user=self.other, arxiv_id='otherroom1234', query='Private question', response='Answer')
        AIassistant.objects.create(user=self.reader, arxiv_id='deletedroom1234', query='Deleted question', response='Answer', active=False)
        response = self.client.get('/en/history/', {'tab': 'conversations'})
        items = response.context['items']
        self.assertEqual(len(items), 2)
        self.assertTrue(items[0]['url'].endswith('#assistant'))
        self.assertIn('c=myconversation1234', items[1]['url'])
        self.assertContains(response, 'My named exchange')
        self.assertNotContains(response, 'Private question')
        self.assertNotContains(response, 'Deleted question')
        response = self.client.get('/en/history/', {'tab': 'conversations', 'q': 'First question'})
        self.assertEqual(len(response.context['items']), 1)

    def test_login_and_french_labels(self):
        self.client.logout()
        self.assertEqual(self.client.get('/fr/history/').status_code, 302)
        self.client.force_login(self.reader)
        response = self.client.get('/fr/history/')
        self.assertContains(response, 'Mon activité')
        self.assertContains(response, 'Aucune lecture')
        self.assertContains(response, 'noindex, nofollow')

    def test_reading_requests_stay_bounded(self):
        for i in range(8):
            paper = self.paper('2103.%05dv1' % i)
            personal_activity.record_visit(self.reader, paper.arxiv_id)
        from django.test.utils import CaptureQueriesContext
        from django.db import connection
        with CaptureQueriesContext(connection) as captured:
            self.client.get('/en/history/')
        paper_queries = [q for q in captured.captured_queries if 'FROM "summarizer_arxivpaper"' in q['sql']]
        self.assertEqual(len(paper_queries), 2)  # count and one page; never one per article

    def test_opening_existing_article_records_an_authenticated_read(self):
        response = self.client.get('/en/paper/' + self.article.arxiv_id + '/')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(PaperVisit.objects.filter(user=self.reader, paper=self.article).count(), 1)
        self.assertFalse(PaperHistory.objects.exists())
        self.client.logout()
        self.client.get('/en/paper/' + self.article.arxiv_id + '/')
        self.assertEqual(PaperVisit.objects.count(), 1)

    def test_identifiers_links_versions_and_word_order(self):
        doi = self.paper('doi_10.1038--s41550-026-03004-3', title='A double debris disc around a white dwarf', authors=('Chris Fox',))
        personal_activity.record_visit(self.reader, doi.arxiv_id)
        for text in ('10.1038/s41550-026-03004-3', 'https://doi.org/10.1038/s41550-026-03004-3',
                     'https://www.summarizepaper.com/fr/paper/' + doi.arxiv_id + '/', 'dwarf debris double'):
            with self.subTest(text=text):
                items = self.client.get('/en/history/', {'q': text}).context['items']
                self.assertEqual([r['id'] for r in items], [doi.arxiv_id])
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        for text in ('2101.00001', '2101.00001v2', 'https://arxiv.org/abs/2101.00001v3',
                     'https://arxiv.org/pdf/2101.00001.pdf?download=1',
                     'https://www.summarizepaper.com/en/paper/2101.00001v1/', 'Ann Lee'):
            with self.subTest(text=text):
                items = self.client.get('/en/history/', {'q': text}).context['items']
                self.assertEqual([r['id'] for r in items], [self.article.arxiv_id])

    def test_multi_paper_search_includes_past_selection_and_respects_boundaries(self):
        room = 'multipaperroom123'
        AIassistant.objects.create(user=self.reader, arxiv_id=room, papers=self.article.arxiv_id,
                                   query='Old selection', response='Answer')
        AIassistant.objects.create(user=self.reader, arxiv_id=room, papers='2101.99999v1',
                                   query='New selection', response='Answer')
        AIassistant.objects.create(user=self.other, arxiv_id='otherroom1234', papers=self.article.arxiv_id,
                                   query='Private', response='Answer')
        for text in ('hot clouds', 'Ann Lee', 'https://arxiv.org/abs/2101.00001'):
            with self.subTest(text=text):
                items = self.client.get('/en/history/', {'tab': 'conversations', 'q': text}).context['items']
                self.assertEqual(len(items), 1)
                self.assertIn(room, items[0]['url'])
        AIassistant.objects.filter(user=self.reader, papers=self.article.arxiv_id).update(papers=self.article.arxiv_id + '9')
        items = self.client.get('/en/history/', {'tab': 'conversations', 'q': 'hot clouds'}).context['items']
        self.assertEqual(items, [])

    def test_clear_search_empty_result_and_time_groups(self):
        from datetime import timedelta
        from django.utils import timezone
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        response = self.client.get('/en/history/', {'q': 'clouds'})
        self.assertContains(response, 'Clear search')
        self.assertContains(response, '1 paper found')
        self.assertEqual(response.context['items'][0]['period'], 'Today')
        PaperVisit.objects.update(visited=timezone.now() - timedelta(days=50))
        response = self.client.get('/en/history/')
        self.assertEqual(response.context['items'][0]['period'], 'Earlier')
        response = self.client.get('/en/history/', {'tab': 'conversations', 'q': 'missing'})
        self.assertContains(response, 'Clear search')
        self.assertIn('conversations', response.context['clear_url'])

    def test_french_search_feedback_and_identifiers_are_escaped(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        response = self.client.get('/fr/history/', {'q': 'clouds'})
        self.assertContains(response, '1 article trouvé')
        self.assertContains(response, 'pour « clouds »')
        self.assertContains(response, 'Effacer la recherche')
        self.assertContains(response, 'Aujourd’hui')
        response = self.client.get('/fr/history/', {'q': '<script>alert(1)</script>'})
        self.assertNotContains(response, '<script>alert(1)</script>')
