
from summarizer import personal_activity
import json
from datetime import timedelta

from django.utils import timezone

from summarizer.models import (AIassistant, ChatConversation, ChatShare, Collection, Generated, PaperHistory, PaperScore,
                               PaperVisit)
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
        response = self.client.get('/en/history/', {'tab': 'readings'})
        self.assertEqual(len(response.context['items']), 1)
        self.assertTrue(response.context['items'][0]['visited'])
        self.assertTrue(response.context['items'][0]['chat'].endswith('#assistant'))

    def test_old_requests_remain_once_and_private(self):
        for i in range(3):
            PaperHistory.objects.create(user=self.reader, arxiv_id=self.article.arxiv_id)
            personal_activity.record_request(self.reader, self.article.arxiv_id)
        other_paper = self.paper('2101.00002v1', title='Private paper')
        personal_activity.record_visit(self.other, other_paper.arxiv_id)
        response = self.client.get('/en/history/', {'tab': 'readings'})
        self.assertEqual(len(response.context['items']), 1)
        self.assertFalse(response.context['items'][0]['visited'])
        self.assertNotContains(response, 'Private paper')
        self.assertContains(response, 'Summary requested')

    def test_search_pagination_and_recent_order(self):
        for i in range(22):
            paper = self.paper('2102.%05dv1' % i, title='Nebula %d' % i)
            personal_activity.record_visit(self.reader, paper.arxiv_id)
        response = self.client.get('/en/history/', {'tab': 'readings', 'q': 'Nebula', 'page': 2})
        self.assertEqual(len(response.context['items']), 2)
        self.assertIn('q=Nebula', response.context['previous_url'])
        response = self.client.get('/en/history/', {'tab': 'readings', 'q': 'not found'})
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
        response = self.client.get('/fr/history/', {'tab': 'readings'})
        self.assertContains(response, 'Mon activité')
        self.assertContains(response, 'Aucune lecture')
        self.assertContains(response, 'noindex, nofollow')

    def test_reading_requests_stay_bounded(self):
        from django.test.utils import CaptureQueriesContext
        from django.db import connection

        def queries():
            with CaptureQueriesContext(connection) as captured:
                self.client.get('/en/history/', {'tab': 'readings'})
            return len(captured.captured_queries)
        for i in range(2):
            personal_activity.record_visit(self.reader, self.paper('2103.%05dv1' % i).arxiv_id)
        few = queries()
        for i in range(2, 12):
            personal_activity.record_visit(self.reader, self.paper('2103.%05dv1' % i).arxiv_id)
        self.assertEqual(queries(), few)  # never one query per article

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
                items = self.client.get('/en/history/', {'tab': 'readings', 'q': text}).context['items']
                self.assertEqual([r['id'] for r in items], [doi.arxiv_id])
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        for text in ('2101.00001', '2101.00001v2', 'https://arxiv.org/abs/2101.00001v3',
                     'https://arxiv.org/pdf/2101.00001.pdf?download=1',
                     'https://www.summarizepaper.com/en/paper/2101.00001v1/', 'Ann Lee'):
            with self.subTest(text=text):
                items = self.client.get('/en/history/', {'tab': 'readings', 'q': text}).context['items']
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
        response = self.client.get('/en/history/', {'tab': 'readings', 'q': 'clouds'})
        self.assertContains(response, 'Clear search')
        self.assertContains(response, '1 paper found')
        self.assertEqual(response.context['items'][0]['period'], 'Today')
        PaperVisit.objects.update(visited=timezone.now() - timedelta(days=50))
        response = self.client.get('/en/history/', {'tab': 'readings'})
        self.assertEqual(response.context['items'][0]['period'], 'Earlier')
        response = self.client.get('/en/history/', {'tab': 'conversations', 'q': 'missing'})
        self.assertContains(response, 'Clear search')
        self.assertIn('conversations', response.context['clear_url'])

    def test_french_search_feedback_and_identifiers_are_escaped(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        response = self.client.get('/fr/history/', {'tab': 'readings', 'q': 'clouds'})
        self.assertContains(response, '1 article trouvé')
        self.assertContains(response, 'pour « clouds »')
        self.assertContains(response, 'Effacer la recherche')
        self.assertContains(response, 'Aujourd’hui')
        response = self.client.get('/fr/history/', {'tab': 'readings', 'q': '<script>alert(1)</script>'})
        self.assertNotContains(response, '<script>alert(1)</script>')

    def test_malformed_pasted_link_is_a_normal_empty_search(self):
        for tab in ('readings', 'conversations'):
            response = self.client.get('/en/history/', {'tab': tab, 'q': 'https://[arxiv.org'})
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, 'No matching activity')


class ResearcherSpaceTests(ToolsTestCase):
    """The overview, the reading status, the actions on several papers, the results and the data export."""

    def setUp(self):
        super().setUp()
        self.reader = self.user()
        self.other = self.user('other')
        self.client = self.client_for(self.reader)
        self.article = self.paper()

    def readings(self, **params):
        return self.client.get('/en/history/', dict(tab='readings', **params)).context['items']

    def test_overview_is_the_default_and_shows_every_block(self):
        response = self.client.get('/en/history/')
        self.assertEqual(response.context['tab'], 'overview')
        for text in ('News about your papers', 'Pick up where you left off', 'To read', 'Your latest results',
                     'What you follow', 'Download my data'):
            self.assertContains(response, text)

    def test_unsummarized_paper_is_kept_by_id_and_title_then_linked_once_summarized(self):
        personal_activity.record_visit(self.reader, '2105.00042', title='A  paper\nnot summarized')
        personal_activity.record_visit(self.reader, '2105.00043')  # nothing to show: not kept
        items = self.readings()
        self.assertEqual([i['title'] for i in items], ['A paper not summarized'])
        self.assertFalse(items[0]['summarized'])
        self.assertContains(self.client.get('/en/history/', {'tab': 'readings'}), 'Summarize it')
        self.assertEqual([i['id'] for i in self.readings(q='not summarized')], ['2105.00042'])
        self.assertEqual([i['id'] for i in self.readings(q='https://arxiv.org/abs/2105.00042v1')], ['2105.00042'])
        self.assertEqual([i['id'] for i in self.readings(status='unsummarized')], ['2105.00042'])
        self.paper('2105.00042', title='Now summarized')
        self.assertEqual(self.readings()[0]['title'], 'Now summarized')
        self.assertIsNotNone(PaperVisit.objects.get(arxiv_id='2105.00042').paper_id)

    def test_opening_an_unsummarized_page_records_its_title(self):
        from unittest import mock
        details = [1, [], [], 'https://arxiv.org/abs/2105.00050', 'Title shown by arXiv', '', 'Abstract', 'astro-ph',
                   '2021-05-01T00:00:00Z', '2021-05-01T00:00:00Z', '', '', 'http://creativecommons.org/licenses/by/4.0/']
        with mock.patch('summarizer.views.arxiv_metadata_or_none', return_value=details):
            self.client.get('/en/paper/2105.00050v1/')
        self.assertEqual(PaperVisit.objects.get(user=self.reader).title, 'Title shown by arXiv')

    def test_reading_status_filter_hide_and_reopen(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        visit = PaperVisit.objects.get(user=self.reader)
        response = self.client.post('/en/history/selection/', {'one': 'to_read:%d' % visit.pk, 'next': '/en/history/?tab=readings'})
        self.assertRedirects(response, '/en/history/?tab=readings&done=status', fetch_redirect_response=False)
        self.assertEqual([i['id'] for i in self.readings(status='to_read')], [self.article.arxiv_id])
        self.assertEqual(self.readings(status='read'), [])
        overview = self.client.get('/en/history/').context
        self.assertEqual(overview['to_read_count'], 1)
        self.client.post('/en/history/selection/', {'action': 'hide', 'v': [visit.pk]})
        self.assertEqual(self.readings(), [])
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        self.assertEqual(len(self.readings()), 1)

    def test_selection_never_touches_another_readers_history(self):
        personal_activity.record_visit(self.other, self.article.arxiv_id)
        theirs = PaperVisit.objects.get(user=self.other)
        self.client.post('/en/history/selection/', {'action': 'hide', 'v': [theirs.pk]})
        theirs.refresh_from_db()
        self.assertFalse(theirs.hidden)
        response = self.client.post('/en/history/selection/', {'action': 'hide', 'next': 'https://evil.example/'})
        self.assertTrue(response['Location'].startswith('/en/history/'))

    def test_ticked_papers_go_to_a_new_collection_the_chat_and_bibtex(self):
        second = self.paper('2101.00002v1', title='Second paper')
        for paper in (self.article, second):
            personal_activity.record_visit(self.reader, paper.arxiv_id)
        pks = list(PaperVisit.objects.filter(user=self.reader).values_list('pk', flat=True))
        response = self.client.post('/en/history/selection/', {'action': 'collection', 'collection': 'new',
                                                               'name': 'Hot planets', 'v': pks})
        collection = Collection.objects.get(user=self.reader, name='Hot planets')
        self.assertEqual(set(collection.papers.values_list('arxiv_id', flat=True)), {self.article.arxiv_id, second.arxiv_id})
        self.assertIn('/collections/%d/' % collection.pk, response['Location'])
        self.assertEqual(self.readings()[0]['collections'][0]['name'], 'Hot planets')
        response = self.client.post('/en/history/selection/', {'action': 'chat', 'v': pks})
        self.assertIn('/chat/?ids=', response['Location'])
        response = self.client.post('/en/history/selection/', {'action': 'bib', 'v': pks})
        self.assertIn(b'Clouds on a hot planet', response.content)
        self.assertIn('.bib', response['Content-Disposition'])

    def test_results_tab_lists_own_results_with_their_tool(self):
        Generated.objects.create(kind='literature', key='k1', user=self.reader, status='done',
                                 content=json.dumps({'question': 'How do clouds form on hot planets?'}))
        Generated.objects.create(kind='refcheck', key='k2', user=self.reader, status='pending')
        Generated.objects.create(kind='literature', key='k3', user=self.other, status='done',
                                 content=json.dumps({'question': 'Private question'}))
        response = self.client.get('/en/history/', {'tab': 'results'})
        self.assertContains(response, 'How do clouds form on hot planets?')
        self.assertContains(response, '/en/literature-review/k1/')
        self.assertContains(response, 'In progress')
        self.assertNotContains(response, 'Private question')
        items = self.client.get('/en/history/', {'tab': 'results', 'kind': 'refcheck'}).context['items']
        self.assertEqual([i['url'] for i in items], ['/en/refcheck/k2/'])
        self.assertEqual(len(self.client.get('/en/history/').context['recent_results']), 2)

    def test_news_retraction_new_version_and_publication_since_reading(self):
        from summarizer import publication_status
        retracted = self.paper('2101.00003v1', title='Retracted paper')
        published = self.paper('2101.00004v1', title='Published paper')
        updated = self.paper('2101.00005v1', title='Updated paper')
        self.paper('2101.00005v2', title='Updated paper')
        for paper in (retracted, published, updated):
            personal_activity.record_visit(self.reader, paper.arxiv_id)
        PaperVisit.objects.filter(arxiv_id=published.arxiv_id).update(visited=timezone.now() - timedelta(days=400))
        day = timezone.localdate().isoformat()
        Generated.objects.create(kind=publication_status.KIND, key=retracted.arxiv_id, lang='en', status='done', content=json.dumps(
            {'status': 'published', 'journal': 'J', 'doi': '10.1/x', 'date': '2019-01-01', 'updates': [{'type': 'retraction', 'doi': '10.1/r'}]}))
        Generated.objects.create(kind=publication_status.KIND, key=published.arxiv_id, lang='en', status='done', content=json.dumps(
            {'status': 'published', 'journal': 'MNRAS', 'doi': '10.1/p', 'date': day}))
        news = self.client.get('/en/history/').context['news']
        labels = [(n['id'], x['label']) for n in news for x in n['news']]
        self.assertEqual(labels[0], (retracted.arxiv_id, 'Retracted'))
        self.assertIn((updated.arxiv_id, 'Version 2 is on arXiv'), labels)
        self.assertIn((published.arxiv_id, 'Published in MNRAS since you read it'), labels)
        self.assertNotIn((retracted.arxiv_id, 'Published in J since you read it'), labels)  # published before the reading
        items = {i['id']: i for i in self.readings()}
        self.assertEqual(items[retracted.arxiv_id]['notices'][0]['label'], 'Retracted')
        self.assertEqual(items[updated.arxiv_id]['newer'], 2)

    def test_close_to_reading_and_watches(self):
        neighbour = self.paper('2101.00009v1', title='A neighbour paper')
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        PaperScore.objects.create(from_paper=self.article, to_paper=neighbour, score=0.9)
        Collection.objects.create(user=self.reader, name='Mine')
        context = self.client.get('/en/history/').context
        self.assertEqual([s['title'] for s in context['suggestions']], ['A neighbour paper'])
        self.assertIn('Collections', [str(w['name']) for w in context['watches']])
        personal_activity.record_visit(self.reader, neighbour.arxiv_id)
        self.assertEqual(self.client.get('/en/history/').context['suggestions'], [])  # already read

    def test_conversations_show_papers_and_can_be_deleted(self):
        room = 'myconversation1234'
        second = self.paper('2101.00002v1', title='Second paper')
        AIassistant.objects.create(user=self.reader, arxiv_id=room, papers='%s,%s' % (self.article.arxiv_id, second.arxiv_id),
                                   query='First question', response='Answer')
        AIassistant.objects.create(user=self.reader, arxiv_id=room, papers=self.article.arxiv_id, query='Latest question', response='A')
        ChatShare.objects.create(user=self.reader, room=room, token='t' * 32, upto=timezone.now())
        AIassistant.objects.create(user=self.reader, arxiv_id=self.article.arxiv_id, query='On the paper', response='A')
        items = self.client.get('/en/history/', {'tab': 'conversations'}).context['items']
        chat = next(i for i in items if i['room'] == room)
        self.assertEqual(chat['title'], 'First question')
        self.assertEqual(chat['last'], 'Latest question')
        self.assertTrue(chat['shared'])
        self.assertEqual([p['title'] for p in chat['papers']], ['Clouds on a hot planet'])
        self.client.post('/en/history/conversation/%s/remove/' % self.article.arxiv_id, {'next': '/en/history/?tab=conversations'})
        self.client.post('/en/history/conversation/%s/remove/' % room)
        self.assertEqual(self.client.get('/en/history/', {'tab': 'conversations'}).context['items'], [])
        self.assertFalse(ChatShare.objects.filter(active=True).exists())
        self.assertEqual(self.client.post('/en/history/conversation/%s/remove/' % room).status_code, 404)

    def test_clear_history_and_export_data(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        personal_activity.record_visit(self.other, self.article.arxiv_id)
        AIassistant.objects.create(user=self.reader, arxiv_id=self.article.arxiv_id, query='My question', response='A')
        data = json.loads(self.client.get('/en/history/export/').content)
        self.assertEqual([r['paper'] for r in data['readings']], [self.article.arxiv_id])
        self.assertEqual(data['conversations'][0]['question'], 'My question')
        self.assertEqual(data['account']['username'], self.reader.username)
        self.client.post('/en/history/clear/')
        self.assertEqual(self.readings(), [])
        self.assertFalse(PaperVisit.objects.get(user=self.other).hidden)

    def test_french_overview(self):
        response = self.client.get('/fr/history/')
        for text in ('Vue d’ensemble', 'Du nouveau sur vos articles', 'Télécharger mes données'):
            self.assertContains(response, text)
