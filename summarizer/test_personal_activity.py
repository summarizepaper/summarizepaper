
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
