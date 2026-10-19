"""Tests of the paper tools and Premium tools: reading levels, press kit, collections, exports, Zotero,
'What did I miss?', citation alerts, audio digest, rising topics and weekly letters."""
import datetime
import json
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import Client, TestCase
from django.utils import timezone

from summarizer import audio, citations, digest, generated, letters, library, missed, presskit, trends
from summarizer.models import (AnnouncedPaper, ArxivBatch, ArxivPaper, Author, CitationCache, Collection,
                               CollectionPaper, DigestItem, DigestSubscription, FollowedPaper, Generated, Letter,
                               LetterSubscription, PaperAudio, PaperAuthor, PaperScore, RisingTopics, SummaryPaper,
                               TermDay, UserPlan, ZoteroAccount)


class ToolsTestCase(TestCase):
    """Background work runs inline, and keeps the test transaction's connection."""

    def setUp(self):
        cache.clear()
        patches = [mock.patch('summarizer.generated.background', side_effect=lambda work, *args: work(*args)),
                   mock.patch('summarizer.generated.release_connection')]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def user(self, name='ann', paid=False):
        user = User.objects.create_user(name, email='%s@example.com' % name, password='x')
        if paid:
            UserPlan.objects.create(user=user, paid=True)
        return user

    def client_for(self, user):
        client = Client()
        client.force_login(user)
        return client

    def paper(self, arxiv_id='2101.00001v1', title='Clouds on a hot planet', authors=('Ann Lee', 'Bob Ray'),
              abstract='We observe clouds on a hot planet with a large telescope.', summary='Long summary of the clouds.'):
        paper = ArxivPaper.objects.create(arxiv_id=arxiv_id, title=title, abstract=abstract, category='astro-ph.EP',
                                          published_arxiv=datetime.date(2021, 1, 4),
                                          license='http://creativecommons.org/licenses/by/4.0/')
        for i, name in enumerate(authors):
            PaperAuthor.objects.create(paper=paper, author=Author.objects.create(name=name, affiliation=''), author_order=i)
        if summary:
            SummaryPaper.objects.create(paper=paper, lang='en', summary=summary, notes=repr(['- Key: clouds are thick', 'Hot']))
        return paper


class GeneratedTests(ToolsTestCase):
    def test_done_results_are_shared_and_pending_ones_not_restarted(self):
        calls = []

        def work(pk):
            calls.append(pk)
            return {'x': 1}
        row = generated.start('reading', 'k', 'en', work)
        self.assertEqual((row.status, Generated.objects.get(pk=row.pk).data()), ('pending', {'x': 1}))
        generated.start('reading', 'k', 'en', work)
        self.assertEqual(len(calls), 1)
        Generated.objects.create(kind='reading', key='busy', lang='en')
        generated.start('reading', 'busy', 'en', work)
        self.assertEqual(len(calls), 1)

    def test_failure_is_stored_and_premium_results_hidden(self):
        def boom(pk):
            raise ValueError('no')
        row = generated.start('presskit', 'p', 'en', boom)
        self.assertEqual(Generated.objects.get(pk=row.pk).status, 'failed')
        self.assertEqual(Client().get('/en/ai/%d/' % row.pk).status_code, 404)
        self.assertEqual(self.client_for(self.user(paid=True)).get('/en/ai/%d/' % row.pk).json()['status'], 'failed')

    def test_private_results_only_for_their_owner(self):
        owner = self.user('owner', paid=True)
        row = Generated.objects.create(kind='missed', key='abc', lang='en', user=owner, status='done', content='{}')
        self.assertEqual(self.client_for(self.user('other', paid=True)).get('/en/ai/%d/' % row.pk).status_code, 404)
        self.assertEqual(self.client_for(owner).get('/en/ai/%d/' % row.pk).status_code, 200)


class ReadingTests(ToolsTestCase):
    def test_rewrite_is_shared_and_free_visitors_have_a_quota(self):
        self.paper()
        answer = {'title': 'Clouds', 'text': 'First part.\n\nSecond part.'}
        with mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            r = Client().post('/en/reading/2101.00001v1/', {'level': 'child', 'lang': 'es'})
            self.assertEqual(r.status_code, 200)
            row = Generated.objects.get(kind='reading', key='2101.00001v1:child', lang='es')
            self.assertEqual(row.data(), {'title': 'Clouds', 'paragraphs': ['First part.', 'Second part.']})
            self.assertIn('Spanish', chat.call_args[0][0])
            self.assertIn('Long summary of the clouds.', chat.call_args[0][0])
            # already written: free for everyone, no new model call
            for _ in range(5):
                self.assertEqual(Client().post('/en/reading/2101.00001v1/', {'level': 'child', 'lang': 'es'}).json()['status'], 'done')
            self.assertEqual(chat.call_count, 1)
            client = Client(REMOTE_ADDR='10.0.0.2')
            for level in ('highschool', 'undergrad', 'expert'):
                self.assertEqual(client.post('/en/reading/2101.00001v1/', {'level': level, 'lang': 'en'}).status_code, 200)
            self.assertEqual(client.post('/en/reading/2101.00001v1/', {'level': 'child', 'lang': 'de'}).status_code, 429)
            paid = self.client_for(self.user(paid=True))
            self.assertEqual(paid.post('/en/reading/2101.00001v1/', {'level': 'child', 'lang': 'de'}).status_code, 200)

    def test_bad_input(self):
        self.paper()
        self.assertEqual(Client().post('/en/reading/2101.00001v1/', {'level': 'x', 'lang': 'en'}).status_code, 400)
        self.assertEqual(Client().post('/en/reading/2109.99999v1/', {'level': 'child', 'lang': 'en'}).status_code, 404)


class PressKitTests(ToolsTestCase):
    def test_kit_keeps_only_real_sentences_and_independent_experts(self):
        paper = self.paper()
        close = self.paper('2101.00002v1', 'Other clouds', authors=('Bob Ray', 'Cid Moe', 'Dee Fox'))
        PaperScore.objects.create(from_paper=paper, to_paper=close, score=0.1)
        answer = {
            'headline': 'Clouds seen', 'plain_explanation': 'Simple.', 'whats_new': 'New.', 'verdict': 'Solid.',
            'hype_check': [{'signal': 'Preprint', 'detail': 'Not reviewed.', 'level': 'caution'},
                           {'signal': 'Weird', 'detail': 'x', 'level': 'unknown'}],
            'key_sentences': ['We observe clouds on a hot planet with a large telescope.',
                              'We prove that the planet is made of cheese and is habitable.'],
            'questions_for_authors': ['Why?'], 'glossary': [{'term': 'Transit', 'definition': 'A passage.'}],
        }
        with mock.patch('summarizer.paper_source.paper_text', return_value='We observe clouds on a hot planet with a large telescope. More.'), \
             mock.patch('summarizer.generated.chat_json', return_value=answer):
            self.assertEqual(Client().post('/en/press-kit/2101.00001v1/start/').status_code, 403)
            client = self.client_for(self.user(paid=True))
            self.assertEqual(client.post('/en/press-kit/2101.00001v1/start/').status_code, 200)
        kit = Generated.objects.get(kind='presskit', key='2101.00001v1', lang='en').data()
        self.assertEqual(kit['key_sentences'], ['We observe clouds on a hot planet with a large telescope.'])
        self.assertEqual([e['name'] for e in kit['experts']], ['Cid Moe', 'Dee Fox'])  # Bob Ray signed the paper
        self.assertEqual(kit['hype_check'][1]['level'], 'caution')
        page = client.get('/en/press-kit/2101.00001v1/')
        self.assertContains(page, 'Clouds seen')
        self.assertContains(page, 'Cid Moe')
        self.assertNotContains(Client().get('/en/press-kit/2101.00001v1/'), 'Clouds seen')


class CollectionTests(ToolsTestCase):
    def test_add_from_paper_page_limits_and_exports(self):
        self.paper()
        user = self.user()
        client = self.client_for(user)
        r = client.post('/en/collections/add/', {'arxiv_id': '2101.00001v1', 'collection': 'new', 'name': 'Thesis'}).json()
        self.assertEqual(r['status'], 'added')
        pk = r['collection']
        self.assertEqual(client.post('/en/collections/add/', {'arxiv_id': '2101.00001v1', 'collection': pk}).json()['status'], 'already')
        self.assertEqual(Client().post('/en/collections/add/', {'arxiv_id': '2101.00001v1', 'collection': pk}).status_code, 403)
        other = self.client_for(self.user('eve'))
        self.assertEqual(other.post('/en/collections/add/', {'arxiv_id': '2101.00001v1', 'collection': pk}).status_code, 404)
        self.assertEqual(other.get('/en/collections/%s/' % pk).status_code, 404)
        for name in ('B', 'C'):
            client.post('/en/collections/', {'name': name})
        self.assertEqual(Collection.objects.filter(user=user).count(), library.FREE['collections'])
        self.assertEqual(client.post('/en/collections/add/', {'arxiv_id': '2101.00001v1', 'collection': 'new'}).json()['status'], 'too_many')

        page = client.get('/en/collections/%s/' % pk)
        self.assertContains(page, 'Clouds on a hot planet')
        self.assertContains(page, '/en/chat/?ids=2101.00001v1')
        bib = client.get('/en/collections/%s/export/bib/' % pk).content.decode()
        self.assertIn('eprint={2101.00001}', bib)
        ris = client.get('/en/collections/%s/export/ris/' % pk).content.decode()
        self.assertIn('TY  - PREP', ris)
        self.assertIn('AN  - arXiv:2101.00001', ris)
        md = client.get('/en/collections/%s/export/md/' % pk).content.decode()
        self.assertIn('- Key: clouds are thick', md)
        self.assertEqual(client.get('/en/collections/%s/export/csv/' % pk).status_code, 404)

    def test_add_by_id_uses_arxiv_when_unknown(self):
        user = self.user()
        collection = Collection.objects.create(user=user, name='X')
        self.paper()
        feed = mock.Mock(status_code=200, text='feed')
        with mock.patch('summarizer.library.requests.get', return_value=feed) as get, \
             mock.patch('summarizer.digest.parse_feed', return_value=[
                 {'arxiv_id': '2105.01234v2', 'title': 'Far away', 'authors': ['Zed']},
                 {'arxiv_id': '2105.04321v3', 'title': 'Latest version', 'authors': ['Yul']}]):
            self.client_for(user).post('/en/collections/%s/' % collection.pk, {
                'action': 'add', 'ids': 'https://arxiv.org/abs/2105.01234v2 nonsense 2105.04321, 2101.00001v1 2109.00000v1'})
        get.assert_called_once()
        self.assertEqual(get.call_args[1]['params']['id_list'], '2105.01234v2,2105.04321,2109.00000v1')
        rows = {r.arxiv_id: (r.title, r.authors) for r in CollectionPaper.objects.filter(collection=collection)}
        self.assertEqual(rows, {'2105.01234v2': ('Far away', 'Zed'), '2105.04321': ('Latest version', 'Yul'),
                                '2101.00001v1': ('Clouds on a hot planet', 'Ann Lee, Bob Ray')})

    def test_comparison_is_premium_and_keeps_known_papers(self):
        self.paper()
        self.paper('2101.00002v1', 'Second paper')
        user = self.user(paid=True)
        collection = Collection.objects.create(user=user, name='Fusion')
        for aid in ('2101.00001v1', '2101.00002v1'):
            library.add_paper(collection, aid, True)
        answer = {'rows': [{'id': '2101.00001v1', 'question': 'Q1', 'method': 'M1', 'data': 'D1', 'results': 'R1',
                            'limitations': 'L1'},
                           {'id': 'made-up', 'question': 'Q'}],
                  'agreements': ['Both see clouds'], 'disagreements': [], 'gaps': ['No one looks at night']}
        client = self.client_for(user)
        with mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            self.assertEqual(client.post('/en/collections/%s/compare/' % collection.pk).status_code, 200)
        self.assertIn('Long summary of the clouds.', chat.call_args[1]['system'])
        page = client.get('/en/collections/%s/' % collection.pk)
        self.assertContains(page, 'No one looks at night')
        self.assertContains(page, 'R1')
        csv = client.get('/en/collections/%s/export/csv/' % collection.pk).content.decode()
        self.assertIn('2101.00001,Clouds on a hot planet,Q1,M1,D1,R1,L1', csv)
        self.assertNotIn('made-up', csv)
        free = self.user('free')
        mine = Collection.objects.create(user=free, name='F')
        self.assertEqual(self.client_for(free).post('/en/collections/%s/compare/' % mine.pk).status_code, 403)


class ExportTests(ToolsTestCase):
    def test_paper_exports_and_page_tools(self):
        self.paper()
        ris = Client().get('/en/export/2101.00001v1/ris/')
        self.assertIn('AU  - Ann Lee', ris.content.decode())
        self.assertIn('attachment', ris['Content-Disposition'])
        md = Client().get('/en/export/2101.00001v1/md/').content.decode()
        self.assertIn('# Clouds on a hot planet', md)
        self.assertIn('## Summary', md)
        self.assertEqual(Client().get('/en/export/2101.00001v1/pdf/').status_code, 404)
        page = Client().get('/en/arxiv-id/2101.00001v1/')
        self.assertContains(page, '<meta name="citation_title" content="Clouds on a hot planet" />', html=False)
        self.assertContains(page, '<meta name="citation_arxiv_id" content="2101.00001" />', html=False)
        self.assertContains(page, 'id="tools"')
        self.assertContains(page, 'Log in to save papers in collections')

    def test_zotero_connect_and_send(self):
        self.paper()
        user = self.user(paid=True)
        client = self.client_for(user)
        self.assertEqual(client.post('/en/zotero/send/', {'arxiv_id': '2101.00001v1'}).json()['status'], 'setup')
        with mock.patch('summarizer.exports.check_zotero', return_value=False):
            self.assertContains(client.post('/en/integrations/', {'library_id': '123', 'api_key': 'abc'}), 'did not accept')
        with mock.patch('summarizer.exports.check_zotero', return_value=True):
            client.post('/en/integrations/', {'library_id': '123', 'api_key': 'abc'})
        self.assertTrue(ZoteroAccount.objects.filter(user=user, library_id='123').exists())
        posts = []

        def fake_post(url, json=None, **kwargs):
            posts.append((url, json))
            return mock.Mock(json=lambda: {'successful': {'0': {'key': 'ITEMKEY'}}}, raise_for_status=lambda: None)
        with mock.patch('summarizer.exports.requests.post', side_effect=fake_post):
            client.post('/en/zotero/send/', {'arxiv_id': '2101.00001v1'})
        self.assertEqual(posts[0][0], 'https://api.zotero.org/users/123/items')
        item = posts[0][1][0]
        self.assertEqual((item['itemType'], item['archiveID'], item['creators'][0]['lastName']), ('preprint', 'arXiv:2101.00001', 'Lee'))
        note = posts[1][1][0]
        self.assertEqual(note['parentItem'], 'ITEMKEY')
        self.assertIn('Long summary of the clouds.', note['note'])
        self.assertEqual(Generated.objects.get(kind='zotero').data(), {'saved': 1})
        self.assertEqual(self.client_for(self.user('free')).post('/en/zotero/send/', {'arxiv_id': '2101.00001v1'}).status_code, 403)


class MissedTests(ToolsTestCase):
    def candidates(self):
        def paper(aid, title):
            return {'arxiv_id': aid, 'title': title, 'abstract': title + ' abstract', 'authors': ['A B'],
                    'published': timezone.now(), 'categories': ['cs.LG']}
        return [paper('2301.00001v1', 'Already cited by id'), paper('2301.00002v2', 'A paper cited by its long title'),
                paper('2301.00003v1', 'Relevant missing paper'), paper('2301.00004v1', 'Other missing paper')]

    def test_cited_papers_are_dropped_and_the_rest_ranked(self):
        draft = ('My draft on learning. ' * 20 + '\nReferences\n[1] X, arXiv:2301.00001\n'
                 '[2] Y, "A paper cited by its long title", 2023.')
        plan = {'title': 'My draft', 'summary': 'It learns.', 'queries': ['learning', 'deep learning']}
        notes = {'items': [{'id': '2301.00003v1', 'relation': 'Same problem.', 'importance': 'high'},
                           {'id': '2301.00004v1', 'relation': 'Side.', 'importance': 'low'}]}

        def embed(texts):
            return np.array([[1.0, 0.0]] + [[0.6, 0.8] if 'Other' in t else [0.8, 0.6] for t in texts[1:]], dtype=np.float32)
        with mock.patch('summarizer.generated.chat_json', side_effect=[plan, notes]), \
             mock.patch('summarizer.digest.arxiv_get', return_value=self.candidates()) as search, \
             mock.patch('summarizer.similar.embed', side_effect=embed):
            result = missed.find(draft, 'en')
        self.assertEqual(search.call_args_list[1][0][0]['search_query'], 'all:deep AND all:learning')
        self.assertEqual(result['cited'], 2)
        self.assertEqual([i['arxiv_id'] for i in result['items']], ['2301.00003v1', '2301.00004v1'])
        self.assertEqual(result['items'][0]['importance'], 'high')

    def test_page_is_premium_and_draft_text_is_searched(self):
        self.assertContains(self.client_for(self.user()).get('/en/missed/'), 'Discover Premium')
        client = self.client_for(self.user('pro', paid=True))
        self.assertContains(client.post('/en/missed/', {'text': 'too short'}), 'at least the title')
        with mock.patch('summarizer.missed.find', return_value={'title': 'Draft T', 'items': [], 'searched': 3, 'cited': 3}) as find:
            r = client.post('/en/missed/', {'text': 'word ' * 100})
        self.assertEqual(find.call_args[0][0].strip(), ('word ' * 100).strip())
        self.assertContains(client.get(r['Location']), 'already cites every close paper')
        self.assertContains(client.get('/en/missed/'), 'Draft T')


class CitationTests(ToolsTestCase):
    def citing(self, *ids):
        return {pid: {'title': 'Paper %s' % pid, 'arxiv_id': '2201.0000%s' % pid[-1], 'doi': '', 'date': '2026-10-01',
                      'authors': ['Zoe Q'], 'contexts': ['As shown by [3], clouds are thick.'] if pid == 'c3' else [],
                      'intents': []} for pid in ids}

    def test_follow_report_new_citations_once_a_week(self):
        user = self.user(paid=True)
        self.paper()
        client = self.client_for(user)
        now = timezone.now()
        with mock.patch('summarizer.citations.fetch', return_value=self.citing('c1', 'c2')), \
             mock.patch('summarizer.citations.time.sleep'):
            r = client.post('/en/citations/follow/', {'arxiv_id': '2101.00001v1'})
        self.assertEqual(r.json()['status'], 'following')
        followed = FollowedPaper.objects.get(user=user)
        self.assertEqual((followed.arxiv_id, followed.title), ('2101.00001', 'Clouds on a hot planet'))
        self.assertEqual(set(json.loads(followed.known)), {'c1', 'c2'})  # what it had is not news
        self.assertEqual(citations.send_alerts(now, 10), 0)

        with mock.patch('summarizer.citations.fetch', return_value=self.citing('c1', 'c2', 'c3')):
            citations.refresh('2101.00001', now)
        stance = {'items': [{'id': 'c3', 'stance': 'supports', 'note': 'Confirms the thick clouds.'}]}
        with mock.patch('summarizer.generated.chat_json', return_value=stance):
            self.assertEqual(citations.send_alerts(now, 10), 1)
        self.assertIn('1 new citation', mail.outbox[0].subject)
        html = mail.outbox[0].alternatives[0][0]
        self.assertIn('Confirms the thick clouds.', html)
        self.assertIn('/en/arxiv-id/2201.00003/', html)
        followed.refresh_from_db()
        self.assertEqual(followed.pending, '')
        with mock.patch('summarizer.citations.fetch', return_value=self.citing('c1', 'c2', 'c3', 'c4')):
            citations.refresh('2101.00001', now)
        self.assertEqual(citations.send_alerts(now + datetime.timedelta(days=2), 10), 0)  # once a week
        with mock.patch('summarizer.generated.chat_json', return_value={'items': []}):
            self.assertEqual(citations.send_alerts(now + datetime.timedelta(days=8), 10), 1)

    def test_free_users_cannot_follow_and_page_lists(self):
        self.assertEqual(self.client_for(self.user()).post('/en/citations/follow/', {'arxiv_id': '2101.00001'}).status_code, 403)
        user = self.user('pro', paid=True)
        with mock.patch('summarizer.citations.fetch', return_value=self.citing('c1')), \
             mock.patch('summarizer.citations.time.sleep'), \
             mock.patch('summarizer.library.lookup', return_value=('Followed title', 'A')):
            self.client_for(user).post('/en/citations/', {'ids': '2101.00001 https://arxiv.org/abs/hep-th/9901001v2'})
        self.assertEqual(sorted(FollowedPaper.objects.filter(user=user).values_list('arxiv_id', flat=True)),
                         ['2101.00001', 'hep-th/9901001'])
        page = self.client_for(user).get('/en/citations/')
        self.assertContains(page, 'Followed title')
        self.assertContains(page, '1 citation')

    def test_refresh_due_skips_fresh_papers(self):
        user = self.user(paid=True)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', known='[]')
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00002', known='[]')
        CitationCache.objects.create(arxiv_id='2101.00002', checked=timezone.now())
        with mock.patch('summarizer.citations.fetch', return_value={}) as fetch, \
             mock.patch('summarizer.citations.time.sleep'), \
             mock.patch('summarizer.library.lookup', return_value=None):
            self.assertEqual(citations.refresh_due(timezone.now()), 1)
        fetch.assert_called_once_with('2101.00001')


class AudioTests(ToolsTestCase):
    def test_clips_are_shared_and_joined(self):
        items = [DigestItem.objects.create(arxiv_id='2610.0000%dv1' % i, lang='en', title='Paper %d' % i, blurb='Blurb %d.' % i)
                 for i in (1, 2)]
        with mock.patch('summarizer.audio.speak', side_effect=lambda text: ('<%s>' % text[:7]).encode()) as speak:
            audio.ensure(items, 'en', [10])
            audio.ensure(items, 'en', [10])
        self.assertEqual(speak.call_count, 3)  # intro and two papers, once
        user = self.user(paid=True)
        subscription = DigestSubscription.objects.create(user=user, categories='astro-ph.EP')
        token = audio.token(subscription, [i.arxiv_id for i in items])
        self.assertContains(Client().get('/en/digest/listen/%s/' % token), 'Paper 2')
        mp3 = Client().get('/en/digest/listen/%s/episode.mp3' % token).content
        self.assertEqual(mp3, b'<Here ar><Paper 1><Paper 2>')
        self.assertEqual(Client().get('/en/digest/listen/forged/').status_code, 404)

    def test_budget_and_failures(self):
        items = [DigestItem.objects.create(arxiv_id='2610.0000%dv1' % i, lang='en', title='P', blurb='B') for i in range(5)]
        with mock.patch('summarizer.audio.speak', return_value=b'x'):
            audio.ensure(items, 'en', [2])
        self.assertEqual(PaperAudio.objects.count(), 2)
        import requests
        with mock.patch('summarizer.audio.speak', side_effect=requests.ConnectionError) as speak:
            audio.ensure(items, 'en', [10])
        self.assertEqual(speak.call_count, 1)


class TrendTests(ToolsTestCase):
    def announce(self, day, titles, category='cs.LG'):
        for i, title in enumerate(titles):
            AnnouncedPaper.objects.create(arxiv_id='%s.%05d' % (day.strftime('%y%m%d'), i), announced=day, title=title,
                                          abstract='Plain words here.', authors='[]', categories=category + ',stat.ML')
        ArxivBatch.objects.get_or_create(day=day)

    def test_terms(self):
        found = trends.terms('Sparse autoencoders for the interpretation of $x^2$ models. Plain words')
        self.assertIn('sparse autoencoders', found)
        self.assertNotIn('models plain', found)
        self.assertNotIn('for the', found)
        self.assertNotIn('the interpretation', found)

    def test_a_term_rises_against_its_history(self):
        start = datetime.date(2026, 9, 7)  # a Monday
        days = [start + datetime.timedelta(days=d) for d in range(35) if (start + datetime.timedelta(days=d)).weekday() < 5]
        for day in days:
            recent = day > days[-1] - datetime.timedelta(days=7)
            titles = ['Graph networks study %d' % i for i in range(15)]
            titles += ['Sparse autoencoders for task %d' % i for i in range(10 if recent else 1)]
            self.announce(day, titles)
        for day in days:
            trends.count_day(day)
        self.assertEqual(TermDay.objects.filter(scope='cs').count(), len(days))
        rising = trends.rising_terms('*', days[-1])
        self.assertEqual(rising[0][0], 'sparse autoencoders')
        self.assertNotIn('graph networks', [t for t, _, _ in rising])
        self.assertNotIn('sparse', [t for t, _, _ in rising])  # kept as part of the longer term
        answer = {'topics': [{'title': 'Looking inside AI models', 'text': 'Why.', 'terms': ['sparse autoencoders', 'invented'],
                              'papers': [trends.examples('*', days[-1], 'sparse autoencoders')[0]['arxiv_id'], 'x']},
                             {'title': 'Noise', 'text': '', 'terms': ['not a rising term']}]}
        with mock.patch('summarizer.generated.chat_json', return_value=answer):
            content = trends.write('*', days[-1], 'en')
        self.assertEqual(len(content['topics']), 1)
        self.assertEqual(content['topics'][0]['terms'], ['sparse autoencoders'])
        self.assertEqual(len(content['topics'][0]['papers']), 1)
        page = Client().get('/en/trends/')
        self.assertContains(page, 'Looking inside AI models')
        self.assertEqual(trends.for_digest(['cs.LG'], 'en', days[-1]), [])  # nothing written for the cs archive yet

    def test_page_before_history(self):
        self.assertContains(Client().get('/en/trends/'), 'first rising topics')
        self.assertEqual(Client().get('/en/trends/?scope=bogus').status_code, 404)

    def test_free_visitors_see_three_topics(self):
        topics = [{'title': 'Topic %d' % i, 'text': 't', 'terms': ['a'], 'count': 9, 'growth': 3.0, 'papers': []} for i in range(5)]
        RisingTopics.objects.create(scope='*', lang='en', day=datetime.date(2026, 10, 19), content=json.dumps({'topics': topics}))
        page = Client().get('/en/trends/')
        self.assertContains(page, 'Topic 2')
        self.assertNotContains(page, 'Topic 3')
        self.assertContains(page, '2 more rising topics')
        self.assertContains(self.client_for(self.user(paid=True)).get('/en/trends/'), 'Topic 4')


class LetterTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        day = datetime.date(2026, 10, 19)
        for i in range(30):
            AnnouncedPaper.objects.create(arxiv_id='2610.%05d' % i, announced=day, title='Space paper %d' % i,
                                          abstract='About stars.', authors='[]', categories='astro-ph.SR')
        AnnouncedPaper.objects.create(arxiv_id='2610.99999', announced=day, title='A cs paper', abstract='',
                                      authors='[]', categories='cs.LG')
        ArxivBatch.objects.create(day=day)
        self.day = day

    def answers(self):
        return [{'ids': ['2610.00001', '2610.00002', '2610.00003', 'bogus']},
                {'title': 'Stars this week', 'dek': 'Three findings.', 'closing': 'All about stars.',
                 'sections': [{'heading': 'A star', 'text': 'It shines.', 'papers': ['2610.00001', 'bogus']}]}]

    def test_letter_written_once_per_topic_and_shared_choice_across_languages(self):
        with mock.patch('summarizer.generated.chat_json', side_effect=self.answers()):
            letter = letters.write('space', self.day, 'en')
        self.assertEqual(letter.title, 'Stars this week')
        content = json.loads(letter.content)
        self.assertEqual(content['sections'][0]['papers'], ['2610.00001'])
        self.assertEqual(content['count'], 30)
        with mock.patch('summarizer.generated.chat_json', side_effect=self.answers()[1:]) as chat:
            letters.write('space', self.day, 'fr')
        self.assertEqual(chat.call_count, 1)  # the papers chosen in English are reused
        self.assertIsNone(letters.write('quantum', self.day, 'en'))  # too few papers
        page = Client().get('/en/letters/space/2026-10-19/')
        self.assertContains(page, 'It shines.')
        self.assertContains(page, 'hreflang="fr"')
        self.assertContains(Client().get('/en/letters/'), 'Stars this week')
        self.assertContains(Client().get('/en/letters/space/'), 'Stars this week')
        self.assertEqual(Client().get('/en/letters/space/2026-13-40/').status_code, 404)
        self.assertContains(Client().get('/sitemap-letters.xml'), '/fr/letters/space/2026-10-19/')

    def test_subscribers_get_each_letter_once(self):
        with mock.patch('summarizer.generated.chat_json', side_effect=self.answers()):
            letters.write('space', self.day, 'en')
        paid, free = self.user('pro', paid=True), self.user('free')
        self.client_for(paid).post('/en/letters/', {'topics': ['space', 'nonsense']})
        LetterSubscription.objects.create(user=free, topic='space', lang='en')
        self.assertEqual(list(LetterSubscription.objects.filter(user=paid, active=True).values_list('topic', flat=True)), ['space'])
        self.assertEqual(letters.send(timezone.now(), 10), 1)
        self.assertEqual(letters.send(timezone.now(), 10), 0)
        self.assertEqual(mail.outbox[0].subject, 'Stars this week')
        self.assertEqual(mail.outbox[0].to, ['pro@example.com'])
        unsubscribe = mail.outbox[0].extra_headers['List-Unsubscribe'].strip('<>').replace('https://www.summarizepaper.com', '')
        Client().post(unsubscribe)
        self.assertFalse(LetterSubscription.objects.get(user=paid, topic='space').active)

    def test_weekly_update_writes_a_few_per_run(self):
        with mock.patch('summarizer.letters.write', return_value=None) as write:
            letters.update()
        self.assertEqual(write.call_count, len(letters.TOPICS) * 2)
        with mock.patch('summarizer.letters.write', return_value=None) as write:
            letters.update()
        self.assertEqual(write.call_count, 0)  # too few papers is not tried again this week


class PremiumDigestExtrasTests(ToolsTestCase):
    def test_paid_digest_links_to_its_audio(self):
        user = self.user(paid=True)
        subscription = DigestSubscription.objects.create(user=user, categories='astro-ph.EP')
        items = [DigestItem.objects.create(arxiv_id='2610.00001v1', lang='en', title='P', blurb='B')]
        with mock.patch('summarizer.audio.speak', return_value=b'x'):
            url = digest.listen_link(subscription, items, 'https://www.summarizepaper.com', [10])
        self.assertIn('/en/digest/listen/', url)
        digest.send_one(subscription, items, 'https://www.summarizepaper.com', paid=True, listen_url=url,
                        rising=[{'title': 'Rising thing', 'growth': 3.1, 'archive': 'astro-ph', 'text': 'More.'}])
        html = mail.outbox[0].alternatives[0][0]
        self.assertIn('Listen to this digest', html)
        self.assertIn('Rising thing', html)
        with mock.patch('summarizer.audio.speak', side_effect=__import__('requests').ConnectionError):
            self.assertIsNone(digest.listen_link(subscription, [DigestItem(arxiv_id='x', lang='en', title='t', blurb='b')],
                                                 'https://x', [10]))

    def test_hourly_command_runs_every_step(self):
        from django.core.management import call_command
        with mock.patch('summarizer.digest.run', return_value=3), \
             mock.patch('summarizer.trends.update', side_effect=RuntimeError) as trends_update, \
             mock.patch('summarizer.letters.update', return_value=0), \
             mock.patch('summarizer.citations.run', return_value=2) as citation_run, \
             mock.patch('summarizer.letters.send', return_value=0) as letter_send:
            call_command('send_digest')
        trends_update.assert_called_once()
        self.assertEqual(citation_run.call_args[0][1], digest.MAX_EMAILS_PER_RUN - 3)
        self.assertEqual(letter_send.call_args[0][1], digest.MAX_EMAILS_PER_RUN - 5)


class PageTests(ToolsTestCase):
    def test_new_pages_load(self):
        user = self.user(paid=True)
        client = self.client_for(user)
        for url in ('/en/collections/', '/fr/collections/', '/en/integrations/', '/en/missed/', '/en/citations/',
                    '/en/trends/', '/fr/letters/', '/en/letters/ai/', '/en/premium/'):
            self.assertEqual(client.get(url).status_code, 200, url)
        for url in ('/en/collections/', '/en/missed/', '/en/citations/', '/en/integrations/'):
            self.assertEqual(Client().get(url).status_code, 302, url)
        self.assertContains(Client().get('/en/premium/'), 'What did I miss?')
        self.assertEqual(Client().get('/en/letters/nope/').status_code, 404)
