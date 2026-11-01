
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
        privacy = self.client.get('/fr/privacy/')
        self.assertContains(privacy, 'Personne d’autre ne les voit')
        self.assertContains(privacy, '/fr/history/export/')


class ReadingListTests(ToolsTestCase):
    """One reading status everywhere, notes, news set aside, the daily check of the papers read."""

    def setUp(self):
        super().setUp()
        self.reader = self.user()
        self.client = self.client_for(self.reader)
        self.article = self.paper()

    def readings(self, **params):
        return self.client.get('/en/history/', dict(tab='readings', **params)).context['items']

    def post(self, url, data):
        return self.client.post(url, data, HTTP_X_REQUESTED_WITH='fetch')

    def test_versions_of_a_paper_are_one_reading(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        self.paper('2101.00001v2', title='Clouds on a hot planet')
        personal_activity.record_visit(self.reader, '2101.00001v2')
        self.assertEqual(list(PaperVisit.objects.filter(user=self.reader).values_list('arxiv_id', flat=True)), ['2101.00001v2'])

    def test_marking_from_anywhere_fills_the_reading_list(self):
        answer = self.post('/en/history/mark/', {'arxiv_id': 'https://arxiv.org/abs/2106.00007', 'status': 'to_read',
                                                 'title': 'A paper of the digest'}).json()
        self.assertEqual(answer['reading'], 'to_read')
        items = self.readings(status='to_read')
        self.assertEqual([(i['title'], i['date_kind']) for i in items], [('A paper of the digest', 'marked')])
        self.post('/en/history/mark/', {'arxiv_id': '2106.00007', 'status': ''})
        self.assertEqual(self.readings(status='to_read'), [])
        response = self.client.post('/en/history/mark/', {'arxiv_id': '2106.00007', 'status': 'read', 'next': '/en/letters/'})
        self.assertEqual(response['Location'], '/en/letters/')
        self.assertEqual(self.client.post('/en/history/mark/', {'arxiv_id': 'nonsense', 'status': 'read'}).status_code, 404)

    def test_paper_page_shows_the_reading_status(self):
        personal_activity.mark(self.reader, self.article.arxiv_id, 'key')
        response = self.client.get('/en/paper/%s/' % self.article.arxiv_id)
        self.assertContains(response, 'id="paper-reading"')
        self.assertContains(response, 'data-status="key" aria-pressed="true"')

    def test_one_status_in_history_and_own_collections(self):
        from summarizer.models import CollectionPaper
        collection = Collection.objects.create(user=self.reader, name='Mine')
        CollectionPaper.objects.create(collection=collection, arxiv_id='2101.00001v2', title='Clouds on a hot planet')
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        visit = PaperVisit.objects.get(user=self.reader)
        self.post('/en/history/selection/', {'one': 'read:%d' % visit.pk})
        self.assertEqual(CollectionPaper.objects.get().reading, 'read')
        self.client.post('/en/collections/%d/' % collection.pk, {'action': 'paper', 'arxiv_id': '2101.00001v2', 'reading': 'key'},
                         HTTP_X_REQUESTED_WITH='fetch')
        self.assertEqual(PaperVisit.objects.get(user=self.reader).reading, 'key')

    def test_notes_are_saved_searched_filtered_and_exported(self):
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        visit = PaperVisit.objects.get(user=self.reader)
        answer = self.post('/en/history/selection/', {'one': 'note:%d' % visit.pk, 'note_%d' % visit.pk: 'Method for chapter three'}).json()
        self.assertEqual(answer['note'], 'Method for chapter three')
        self.assertEqual(len(self.readings(q='chapter three')), 1)
        self.assertEqual(len(self.readings(status='noted')), 1)
        bib = self.client.post('/en/history/selection/', {'action': 'bib', 'v': [visit.pk]}).content.decode()
        self.assertIn('Method for chapter three', bib)

    def test_every_matching_paper_and_sorting(self):
        import datetime
        for i in range(25):
            paper = self.paper('2104.%05dv1' % i, title='Nebula %02d' % (24 - i))
            paper.published_arxiv = datetime.date(2020, 1, 1) + datetime.timedelta(days=i)
            paper.save()
            personal_activity.record_visit(self.reader, paper.arxiv_id)
        self.client.post('/en/history/selection/', {'action': 'to_read', 'all': '1', 'q': 'Nebula'})
        self.assertEqual(PaperVisit.objects.filter(user=self.reader, reading='to_read').count(), 25)
        self.assertEqual(self.readings(sort='title')[0]['title'], 'Nebula 00')
        self.assertEqual(self.readings(sort='published')[0]['id'], '2104.00024v1')

    def test_news_new_since_last_visit_and_set_aside(self):
        from summarizer import publication_status
        from summarizer.models import ActivityState
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        notice = {'status': 'published', 'journal': 'J', 'doi': '10.1/x', 'date': '2019-01-01',
                  'updates': [{'type': 'retraction', 'doi': '10.1/r'}]}
        Generated.objects.create(kind=publication_status.KIND, key=self.article.arxiv_id, lang='en', status='done',
                                 content=json.dumps(notice))
        news = self.client.get('/en/history/').context['news']
        self.assertFalse(news[0]['new'])  # a first visit has nothing to compare with
        ActivityState.objects.update(seen=timezone.now() - timedelta(hours=2), known={})
        context = self.client.get('/en/history/').context
        self.assertTrue(context['news'][0]['new'])
        self.assertEqual(context['news_new'], 1)
        self.post('/en/history/news/dismiss/', {'keys': context['news'][0]['keys']})
        self.assertEqual(self.client.get('/en/history/').context['news'], [])

    def test_results_grouped_and_lost_runs_failed(self):
        for key in ('a', 'b', 'c'):
            Generated.objects.create(kind='literature', key=key, user=self.reader, status='done',
                                     content=json.dumps({'question': 'Same question'}))
        lost = Generated.objects.create(kind='refcheck', key='d', user=self.reader, status='pending')
        Generated.objects.filter(pk=lost.pk).update(updated=timezone.now() - timedelta(hours=1))
        results = self.client.get('/en/history/').context['recent_results']
        self.assertEqual([(r['title'], r['times'], r['status']) for r in results],
                         [('Bibliography check', 1, 'failed'), ('Same question', 3, 'done')])

    def test_key_papers_become_digest_references(self):
        from summarizer.models import DigestSubscription
        paid = self.user('paid', paid=True)
        client = self.client_for(paid)
        digest = DigestSubscription.objects.create(user=paid, categories='astro-ph.EP')
        for i in range(7):
            personal_activity.mark(paid, '2105.%05dv2' % i, 'key', 'Key %d' % i)
        self.assertContains(client.get('/en/history/', {'tab': 'readings', 'status': 'key'}), 'Use them as the reference papers')
        client.post('/en/history/digest-references/', {'digest': digest.pk})
        digest.refresh_from_db()
        self.assertEqual(len(digest.reference_list()), 5)
        self.assertNotIn('v2', digest.reference_papers)
        self.assertEqual(self.client.post('/en/history/digest-references/', {'digest': digest.pk}).status_code, 404)

    def test_daily_check_reads_versions_and_queues_status_checks(self):
        from unittest import mock
        from django.core.cache import cache
        from summarizer import publication_status
        personal_activity.record_visit(self.reader, self.article.arxiv_id)
        with mock.patch('summarizer.digest.fetch_ids', return_value=[{'arxiv_id': '2101.00001v3'}]) as fetch:
            self.assertEqual(personal_activity.check_readings(timezone.now()), 1)
            self.assertEqual(personal_activity.check_readings(timezone.now()), 0)  # once a day
        self.assertEqual(fetch.call_count, 1)
        self.assertIn(self.article.arxiv_id, cache.get(publication_status.QUEUE_KEY))
        labels = [x['label'] for n in self.client.get('/en/history/').context['news'] for x in n['news']]
        self.assertIn('Version 3 is on arXiv', labels)

    def test_daily_check_takes_summary_requests_and_goes_first_in_a_full_queue(self):
        from unittest import mock
        from django.core.cache import cache
        from summarizer import publication_status
        cache.set(publication_status.QUEUE_KEY, ['crawled%d' % i for i in range(publication_status.MAX_QUEUED)])
        personal_activity.record_request(self.reader, self.article.arxiv_id)
        PaperVisit.objects.update(visited=None)
        with mock.patch('summarizer.digest.fetch_ids', return_value=[{'arxiv_id': '2101.00001v2'}]):
            self.assertEqual(personal_activity.check_readings(timezone.now()), 1)
        queue = cache.get(publication_status.QUEUE_KEY)
        self.assertEqual((queue[0], len(queue)), (self.article.arxiv_id, publication_status.MAX_QUEUED))

    def test_digest_save_link_adds_to_the_reading_list(self):
        from django.core import signing
        from summarizer import digest
        from summarizer.models import DigestSubscription
        subscription = DigestSubscription.objects.create(user=self.reader, categories='astro-ph.EP')
        token = signing.dumps([subscription.pk, '2107.00001'], salt=digest.SAVE_SALT, compress=True)
        anonymous = self.client_class()
        response = anonymous.post('/en/digest/save/%s/' % token, {'reading': 'to_read'})
        self.assertContains(response, 'Added to your reading list')
        self.assertEqual(PaperVisit.objects.get(user=self.reader).reading, 'to_read')


class SharedResultsAndCrawlersTests(ToolsTestCase):
    """Press kits and journal club decks in the Results of those who made them; the publication queue for readers."""

    def setUp(self):
        super().setUp()
        self.article = self.paper()

    def test_press_kit_and_decks_started_or_edited_are_listed(self):
        from unittest import mock
        pro = self.user('pro', paid=True)
        client = self.client_for(pro)
        kit = Generated.objects.create(kind='presskit', key=self.article.arxiv_id, lang='en', status='done', content='{}')
        deck = Generated.objects.create(kind='slides_v3', key=self.article.arxiv_id + '|students|10', lang='fr', status='pending')
        with mock.patch('summarizer.generated.start', side_effect=[kit, deck]):
            client.post('/en/press-kit/%s/start/' % self.article.arxiv_id)
            client.post('/en/journal-club/%s/start/' % self.article.arxiv_id,
                        {'deck_lang': 'fr', 'audience': 'students', 'duration': '10'})
        items = client.get('/en/history/', {'tab': 'results'}).context['items']
        self.assertEqual([(str(i['tool']), i['title']) for i in items],
                         [('Journal club', 'Clouds on a hot planet'), ('Press kit', 'Clouds on a hot planet')])
        self.assertIn('/journal-club/%s/?' % self.article.arxiv_id, items[0]['url'])
        self.assertIn('audience=students', items[0]['url'])
        self.assertEqual(items[1]['url'], '/en/press-kit/%s/' % self.article.arxiv_id)
        only = client.get('/en/history/', {'tab': 'results', 'kind': 'journal-club'}).context['items']
        self.assertEqual([str(i['tool']) for i in only], ['Journal club'])
        other = self.client_for(self.user('other'))
        self.assertEqual(other.get('/en/history/', {'tab': 'results'}).context['items'], [])
        export = json.loads(client.get('/en/history/export/').content)
        self.assertEqual(len(export['press_kits_and_journal_clubs']), 2)

    def test_publication_checks_are_queued_for_readers_not_crawlers(self):
        from django.core.cache import cache
        from summarizer import publication_status
        url = '/en/paper/%s/' % self.article.arxiv_id
        browser = 'Mozilla/5.0 (Macintosh) Firefox/130.0'
        for agent in ('Googlebot/2.1', 'python-requests/2.32'):
            self.client.get(url, HTTP_USER_AGENT=agent)
        self.client.get(url, HTTP_USER_AGENT=browser, HTTP_X_FORWARDED_FOR='114.119.130.1')  # a crawler network
        self.assertEqual(cache.get(publication_status.QUEUE_KEY) or [], [])
        self.client.get(url, HTTP_USER_AGENT=browser)
        self.assertEqual(cache.get(publication_status.QUEUE_KEY), [self.article.arxiv_id])
        crowded = ['other%d' % i for i in range(publication_status.ANONYMOUS_QUEUED)]
        cache.set(publication_status.QUEUE_KEY, crowded)
        second = self.paper('2101.00002v1')
        self.client.get('/en/paper/%s/' % second.arxiv_id, HTTP_USER_AGENT=browser)
        self.assertNotIn(second.arxiv_id, cache.get(publication_status.QUEUE_KEY))  # visitors fill half at most
        self.client_for(self.user()).get('/en/paper/%s/' % second.arxiv_id, HTTP_USER_AGENT=browser)
        self.assertEqual(cache.get(publication_status.QUEUE_KEY)[0], second.arxiv_id)  # a reader goes first
