"""Tests of the paper tools and Premium tools: reading levels, press kit, collections, exports, Zotero,
'What did I miss?', citation alerts, audio digest, rising topics and weekly letters."""
import datetime
import json
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import Client, TestCase, override_settings
from django.utils import timezone, translation

from summarizer import audio, changes, citations, conference, digest, generated, letters, library, missed, presskit, questions, refcheck, scoops, slides, teams, trends, usage
from summarizer.models import (AnnouncedPaper, ArxivBatch, ArxivPaper, Author, CitationCache, Collection,
                               CollectionPaper, DigestItem, DigestSubscription, DigestVector, FollowedPaper, Generated, PaperVersion,
                               Letter, LetterSubscription, PaperAudio, PaperAuthor, PaperScore, RisingTopics, ScoopMatch,
                               SummaryPaper, TermDay, UserPlan, WatchedProject, ZoteroAccount)


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


class StaticVersionTests(TestCase):
    def test_scripts_and_styles_change_url_when_their_content_changes(self):
        import re
        page = Client().get('/en/trends/').content.decode()
        self.assertRegex(page, r'/static/summarizer/js/tools\.js\?v=[0-9a-f]{10}')
        self.assertRegex(page, r'/static/summarizer/css/site\.css\?v=[0-9a-f]{10}')


class ProgressTests(ToolsTestCase):
    def test_steps_are_reported_and_shown_while_waiting(self):
        user = self.user(paid=True)
        row = Generated.objects.create(kind='missed', key='k1', lang='en', user=user)
        generated.progress(row.pk, 'search')
        client = self.client_for(user)
        self.assertEqual(client.get('/en/ai/%d/' % row.pk).json()['step'], 'search')
        page = client.get('/en/missed/k1/')
        self.assertContains(page, 'class="spinner"')
        self.assertContains(page, 'data-step="review"')
        self.assertContains(page, 'Searching arXiv and the journals for close papers')
        self.assertContains(page, 'data-started=')


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
    def letter(self, lang='en'):
        return Letter.objects.create(topic='space', lang=lang, day=datetime.date(2026, 10, 19),
            title='The sky this week', content=json.dumps({'count': 3,
                'sections': [{'heading': 'Clouds', 'text': 'Editorial angle <script>alert(1)</script>',
                              'papers': ['2101.00001', '2101.00002', '2101.00002']}],
                'papers': {'2101.00001': 'Clouds on a hot planet', '2101.00002': 'Another observation'}}))

    def test_letter_links_to_kit_with_language_and_editorial_context(self):
        self.paper()
        for lang in ('en', 'fr'):
            letter = self.letter(lang)
            path = '/%s/press-kit/2101.00001v1/?letter=%s' % (lang, letter.pk)
            page = Client().get('/%s/letters/space/2026-10-19/' % lang)
            self.assertContains(page, path)
            self.assertContains(page, 'rel="canonical"')
            kit = Client().get(path)
            self.assertContains(kit, 'Clouds on a hot planet')
            self.assertContains(kit, 'Editorial angle &lt;script&gt;')
            self.assertNotContains(kit, '<script>alert(1)</script>')
            self.assertContains(kit, 'Another observation', count=1)
            self.assertContains(kit, '/%s/letters/space/2026-10-19/' % lang)
            self.assertEqual(kit.context['editorial']['letter'], letter)
            self.assertFalse(Generated.objects.exists())

    def test_editorial_context_rejects_unrelated_wrong_language_or_invalid_letters(self):
        paper = self.paper()
        letter = self.letter()
        other = self.paper('2101.00003v1')
        self.assertIsNone(presskit.editorial_context(letter.pk, other.arxiv_id, 'en'))
        self.assertIsNone(presskit.editorial_context(letter.pk, paper.arxiv_id, 'fr'))
        for value in (None, 'bogus', '9' * 100, '١'):
            self.assertIsNone(presskit.editorial_context(value, paper.arxiv_id, 'en'))
        letter.content = '{'
        letter.save()
        self.assertIsNone(presskit.editorial_context(letter.pk, paper.arxiv_id, 'en'))

    def test_announced_paper_landing_does_not_fetch_generate_or_store_on_get(self):
        AnnouncedPaper.objects.create(arxiv_id='2101.00001', title='Newly announced',
                                      announced=datetime.date(2026, 10, 19))
        letter = self.letter()
        with mock.patch('summarizer.views.arxiv_metadata_or_none') as fetch, \
             mock.patch('summarizer.generated.chat_json') as chat:
            for client in (Client(), self.client_for(self.user(paid=True))):
                page = client.get('/en/press-kit/2101.00001v1/?letter=%s' % letter.pk)
                self.assertContains(page, 'Newly announced')
                self.assertIsNotNone(page.context['editorial'])
            fetch.assert_not_called()
            chat.assert_not_called()
        self.assertFalse(ArxivPaper.objects.exists())
        self.assertFalse(Generated.objects.exists())
        self.assertEqual(Client().get('/en/press-kit/2101.99999v1/').status_code, 404)
        self.assertEqual(Client().post('/en/press-kit/2101.00001v1/start/').status_code, 403)

    def test_announced_paper_imports_metadata_only_when_premium_starts_kit(self):
        AnnouncedPaper.objects.create(arxiv_id='2101.00001', title='Newly announced',
                                      announced=datetime.date(2026, 10, 19))
        client = self.client_for(self.user(paid=True))
        metadata = [1, ['Ann Lee'], [''], 'https://arxiv.org/abs/2101.00001', 'Source title',
                    '', 'Original scientific abstract.', 'astro-ph.EP', '2021-01-01T00:00:00Z',
                    '2021-01-01T00:00:00Z', '', '', '']
        with mock.patch('summarizer.views.arxiv_metadata_or_none', return_value=metadata) as fetch, \
             mock.patch('summarizer.presskit.experts', return_value=[]), \
             mock.patch('summarizer.generated.chat_json', return_value={'headline': 'Source headline'}) as chat:
            client.post('/en/press-kit/2101.00001v1/start/')
        fetch.assert_called_once_with('2101.00001v1')
        kit = Generated.objects.get(kind='presskit')
        self.assertEqual(kit.status, 'done')
        self.assertIn('Original scientific abstract.', chat.call_args.kwargs['system'])
        self.assertNotIn('Editorial angle', chat.call_args.kwargs['system'])
        self.assertEqual(ArxivPaper.objects.get().authors.get().name, 'Ann Lee')

    def test_announced_source_outage_is_retryable_without_partial_paper(self):
        AnnouncedPaper.objects.create(arxiv_id='2101.00001', title='Newly announced',
                                      announced=datetime.date(2026, 10, 19))
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.views.arxiv_metadata_or_none', return_value=None), \
             mock.patch('summarizer.generated.chat_json') as chat:
            client.post('/en/press-kit/2101.00001v1/start/')
            self.assertEqual(Generated.objects.get().status, 'failed')
            self.assertFalse(ArxivPaper.objects.exists())
            chat.assert_not_called()
        self.assertContains(client.get('/en/press-kit/2101.00001v1/'), 'Please try again')

    def test_cached_kit_is_unchanged_by_letter_context(self):
        self.paper()
        letter = self.letter()
        row = Generated.objects.create(kind='presskit', key='2101.00001v1', lang='en', status='done',
                                       content=json.dumps({'headline': 'Scientific kit'}))
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.generated.chat_json') as chat:
            page = client.get('/en/press-kit/2101.00001v1/?letter=%s' % letter.pk)
            self.assertContains(page, 'Scientific kit')
            self.assertContains(page, 'Editorial angle')
            chat.assert_not_called()
        row.refresh_from_db()
        self.assertEqual(row.data(), {'headline': 'Scientific kit'})

    def test_open_paper_fetches_missing_text_or_labels_abstract_fallback(self):
        paper = self.paper()
        client = self.client_for(self.user(paid=True))
        full_text = 'We observe clouds on a hot planet with a large telescope.'
        for text, expected_abstract in ((full_text, False), ('', True)):
            Generated.objects.all().delete()
            with mock.patch('summarizer.paper_source.paper_text', return_value=paper.abstract), \
                 mock.patch('summarizer.paper_source.read_again', return_value=(text, 'html')) as read, \
                 mock.patch('summarizer.paper_source.store_text') as store, \
                 mock.patch('summarizer.presskit.experts', return_value=[]), \
                 mock.patch('summarizer.generated.chat_json', return_value={}):
                client.post('/en/press-kit/2101.00001v1/start/')
            read.assert_called_once_with(paper.arxiv_id)
            self.assertEqual(Generated.objects.get().data()['abstract_only'], expected_abstract)
            if text:
                store.assert_called_once_with(paper.arxiv_id, text, 'html')
            else:
                store.assert_not_called()
                self.assertContains(client.get('/en/press-kit/2101.00001v1/'), 'Only the abstract was available')

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
             mock.patch('summarizer.sources.search_papers', return_value=[]), \
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
        found = lambda ids: {a: ('Title of %s' % a, 'A') for a in ids}
        with mock.patch('summarizer.citations.fetch', return_value=self.citing('c1')), \
             mock.patch('summarizer.citations.time.sleep'), \
             mock.patch('summarizer.library.lookup_many', side_effect=found):
            self.client_for(user).post('/en/citations/', {'ids': '2101.00001 https://arxiv.org/abs/hep-th/9901001v2'})
        self.assertEqual(sorted(FollowedPaper.objects.filter(user=user).values_list('arxiv_id', 'title')),
                         [('2101.00001', 'Title of 2101.00001'), ('hep-th/9901001', 'Title of hep-th--9901001')])
        page = self.client_for(user).get('/en/citations/')
        self.assertContains(page, 'Title of 2101.00001')
        self.assertContains(page, '1 citation')

    def test_title_is_known_without_semantic_scholar(self):
        user = self.user('pro', paid=True)
        # followed before titles were looked up, and Semantic Scholar does not know it
        FollowedPaper.objects.create(user=user, arxiv_id='2607.11677')
        with mock.patch('summarizer.citations.fetch', return_value=None), \
             mock.patch('summarizer.library.lookup_many', return_value={'2607.11677': ('A new disc', 'A')}):
            citations.refresh('2607.11677', timezone.now())
            self.assertContains(self.client_for(user).get('/en/citations/'), 'A new disc')
        self.assertEqual(FollowedPaper.objects.get(user=user).title, 'A new disc')

    @override_settings(ADS_TOKEN='token')
    def test_ads_counts_with_semantic_scholar_sentences(self):
        def ads(params):
            if params['q'] == 'identifier:"arXiv:2101.00001"':
                return {'numFound': 1, 'docs': [{'bibcode': '2021A&A...1K'}]}
            if params['q'].startswith('identifier:'):
                return {'numFound': 0, 'docs': []}
            return {'numFound': 2, 'docs': [
                {'bibcode': '2026MNRAS.1A', 'title': ['Thick  clouds'], 'author': ['Zoe Q'], 'pubdate': '2026-10-00',
                 'identifier': ['2026MNRAS.1A', 'arXiv:2201.00003'], 'doi': ['10.1/a']},
                {'bibcode': '2026ApJ...2B', 'title': ['Not on arXiv'], 'pubdate': '2026-09-00'}]}
        s2 = {'s3': dict(self.citing('c3')['c3'], arxiv_id='2201.00003v2')}
        with mock.patch('summarizer.citations.ads_get', side_effect=ads), \
             mock.patch('summarizer.citations.fetch_s2', return_value=s2) as fetch_s2:
            citing = citations.fetch('2101.00001')
            self.assertEqual(list(citing), ['ads:2026MNRAS.1A', 'ads:2026ApJ...2B'])
            self.assertEqual(citing['ads:2026MNRAS.1A']['contexts'], ['As shown by [3], clouds are thick.'])
            self.assertEqual((citing['ads:2026MNRAS.1A']['title'], citing['ads:2026MNRAS.1A']['date']), ('Thick clouds', '2026-10'))
            # papers ADS does not know keep Semantic Scholar
            self.assertEqual(citations.fetch('cs/0101001'), s2)
            self.assertEqual(fetch_s2.call_count, 2)

    def test_switching_source_keeps_known_and_queued_citations(self):
        user = self.user(paid=True)
        now = timezone.now()
        old = self.citing('c1', 'c2')
        CitationCache.objects.create(arxiv_id='2101.00001', citing=json.dumps(old), total=2, checked=now)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='T', known='["c1"]', pending='["c2"]')
        new = {'ads:' + pid: dict(c) for pid, c in self.citing('c1', 'c2', 'c4').items()}
        with mock.patch('summarizer.citations.fetch', return_value=new):
            citations.refresh('2101.00001', now)
        followed = FollowedPaper.objects.get(user=user)
        self.assertEqual(json.loads(followed.pending), ['ads:c2'])
        self.assertEqual(json.loads(followed.known), ['ads:c1', 'ads:c4'])
        self.assertEqual(CitationCache.objects.get(arxiv_id='2101.00001').total, 3)

    def test_switching_from_a_paper_without_citations_reports_nothing_old(self):
        user = self.user(paid=True)
        now = timezone.now()
        CitationCache.objects.create(arxiv_id='2101.00001', citing='{}', total=0, source='s2', checked=now)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='T', known='[]')
        new = citations.Citing({'ads:' + pid: dict(c) for pid, c in self.citing('c1', 'c2').items()}, 'ads')
        with mock.patch('summarizer.citations.fetch', return_value=new):
            citations.refresh('2101.00001', now)
        followed = FollowedPaper.objects.get(user=user)
        self.assertEqual(followed.pending, '')
        self.assertEqual(sorted(json.loads(followed.known)), ['ads:c1', 'ads:c2'])
        # ADS knows a paper with no citation yet: its first citation is news, not a switch of source
        CitationCache.objects.create(arxiv_id='2101.00002', citing='{}', total=0, source='s2', checked=now)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00002', title='U', known='[]')
        with mock.patch('summarizer.citations.fetch', return_value=citations.Citing({}, 'ads')):
            citations.refresh('2101.00002', now)
        with mock.patch('summarizer.citations.fetch', return_value=new):
            citations.refresh('2101.00002', now)
        self.assertEqual(sorted(json.loads(FollowedPaper.objects.get(arxiv_id='2101.00002').pending)), ['ads:c1', 'ads:c2'])

    def test_refresh_due_skips_fresh_papers(self):
        user = self.user(paid=True)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', known='[]')
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00002', known='[]')
        CitationCache.objects.create(arxiv_id='2101.00002', checked=timezone.now())
        with mock.patch('summarizer.citations.fetch', return_value={}) as fetch, \
             mock.patch('summarizer.citations.time.sleep'), \
             mock.patch('summarizer.library.lookup_many', return_value={}):
            self.assertEqual(citations.refresh_due(timezone.now()), 1)
        fetch.assert_called_once_with('2101.00001')


class ScoopTests(ToolsTestCase):
    DESCRIPTION = 'We measure the clouds of hot Jupiters with JWST transmission spectra and a new retrieval code. ' * 4

    def announce(self, day, prefix):
        """A close paper, the user's own close paper, and a far one."""
        ArxivBatch.objects.create(day=day, papers=3)
        for n, title, author, vector in ((1, 'Clouds of hot Jupiters with JWST', 'Zoe Q', [1.0, 0.0]),
                                         (2, 'Our own clouds paper', 'A. Lee', [1.0, 0.0]),
                                         (3, 'Protein folding', 'Max P', [0.0, 1.0])):
            aid = '%s%d' % (prefix, n)
            AnnouncedPaper.objects.create(arxiv_id=aid, announced=day, title=title, abstract=title + ' abstract.',
                                          authors=json.dumps([author]), categories='astro-ph.EP')
            DigestVector.objects.create(arxiv_id=aid, vector=np.array(vector, dtype=np.float32).tobytes())

    def verdict(self, aid, overlap):
        return {'items': [{'id': aid, 'overlap': overlap, 'shared': 'Same clouds.', 'original': 'Your retrieval code.'}]}

    def test_first_check_is_shown_then_new_papers_are_emailed_once(self):
        today = timezone.now().date()
        self.announce(today - datetime.timedelta(days=1), '2610.0000')
        client = self.client_for(self.user(paid=True))
        abstract = {'abstract': 'Transmission spectra of hot Jupiters observed with JWST are analysed.'}
        with mock.patch('summarizer.similar.embed', return_value=np.array([[1.0, 0.0]], dtype=np.float32)) as embed, \
             mock.patch('summarizer.generated.chat_json', side_effect=[abstract, self.verdict('2610.00001', 'high')]) as chat:
            r = client.post('/en/scoops/', {'title': 'Hot clouds', 'description': self.DESCRIPTION, 'author_name': 'Ann Lee'})
        self.assertEqual(embed.call_args[0][0], ['Hot clouds\n\n' + abstract['abstract']])  # the abstract, not the description
        self.assertEqual(r['Location'], '/en/scoops/#project-%d' % WatchedProject.objects.get().pk)
        listing = chat.call_args[1]['system']
        self.assertIn('2610.00001', listing)
        self.assertNotIn('2610.00002', listing)  # the user's own paper
        self.assertNotIn('2610.00003', listing)  # too far
        match = ScoopMatch.objects.get()
        self.assertTrue(match.emailed)  # the first check is not emailed
        page = client.get('/en/scoops/')
        self.assertContains(page, 'Clouds of hot Jupiters with JWST')
        self.assertContains(page, 'Strong overlap')

        self.announce(today, '2610.0001')
        with mock.patch('summarizer.generated.chat_json', return_value=self.verdict('2610.00011', 'partial')) as chat:
            self.assertEqual(scoops.run(timezone.now(), 5), 1)
            self.assertEqual(scoops.run(timezone.now(), 5), 0)
        chat.assert_called_once()
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('close to your project', mail.outbox[0].subject)
        self.assertIn('Your retrieval code.', mail.outbox[0].body)
        self.assertEqual(WatchedProject.objects.get().last_day, today)

    def test_premium_only_and_short_descriptions_refused(self):
        self.assertContains(self.client_for(self.user()).get('/en/scoops/'), 'Discover Premium')
        self.assertContains(Client().get('/en/scoops/'), 'Create a free account')
        client = self.client_for(self.user('pro', paid=True))
        self.assertContains(client.post('/en/scoops/', {'title': 'T', 'description': 'too short'}), 'at least 300')
        self.assertContains(client.get('/en/scoops/'), 'data-min-chars="300"')
        with translation.override('fr'):
            self.assertContains(client.get('/fr/scoops/'), 'Encore {n} caractères (au moins 300)')
        self.assertFalse(WatchedProject.objects.exists())

    def test_projects_of_lapsed_subscribers_are_not_checked(self):
        user = self.user(paid=True)
        WatchedProject.objects.create(user=user, title='T', description=self.DESCRIPTION,
                                      last_day=timezone.now().date() - datetime.timedelta(days=2))
        self.announce(timezone.now().date(), '2610.0000')
        UserPlan.objects.filter(user=user).update(paid=False)
        with mock.patch('summarizer.generated.chat_json') as chat:
            scoops.run(timezone.now(), 5)
        chat.assert_not_called()


class RefcheckTests(ToolsTestCase):
    BIB = """@article{retracted, title = {A retracted result}, journal = {J}, doi = {10.1000/RETRACTED}}
@misc{pre, author = {Lee, Ann}, title = {{Clouds} on a "hot" planet}, eprint = {2101.00001v1}, archivePrefix = {arXiv}}
@misc{again, title = "Clouds again", howpublished = {arXiv:2101.00001}}
@book{nothing, title = {A book with no identifier}}
@string{foo = "bar"}
"""
    TEX = r"""\documentclass{article}\begin{document}
Clouds are thick on hot planets \citep{pre}. % \cite{retracted} in a comment is not read
Earlier work found no clouds at all \citet[see][]{pre,again}.
\end{document}"""

    def test_parse_and_identify(self):
        entries = refcheck.parse_bibtex(self.BIB)
        self.assertEqual([e['key'] for e in entries], ['retracted', 'pre', 'again', 'nothing'])
        self.assertEqual(entries[1]['fields']['title'], '{Clouds} on a "hot" planet')
        self.assertEqual([refcheck.identify(e) for e in entries],
                         [('', '', '10.1000/retracted'), ('2101.00001', 'v1', ''), ('2101.00001', '', ''), ('', '', '')])
        self.assertEqual(refcheck.citing_sentences(self.TEX),
                         [('Clouds are thick on hot planets \\citep{pre}.', ['pre']),
                          ('Earlier work found no clouds at all \\citet[see][]{pre,again}.', ['pre', 'again'])])

    def test_check_reports_problems_and_corrects_the_bib(self):
        record = {'arxiv_id': '2101.00001v3', 'title': 'Clouds on a hot planet', 'abstract': 'We see thick clouds.',
                  'doi': '10.1093/mnras/x1', 'journal_ref': 'MNRAS 500', 'comment': '', 'version': '3'}
        verdicts = {'items': [{'id': '0', 'verdict': 'supported'}, {'id': '1', 'verdict': 'contradicted', 'reason': 'It finds clouds.'}]}
        with mock.patch('summarizer.refcheck.arxiv_records', return_value={'2101.00001': record}), \
             mock.patch('summarizer.refcheck.openalex_works', return_value={'10.1000/retracted': {'retracted': True, 'abstract': ''}}), \
             mock.patch('summarizer.refcheck.s2_published', return_value={}), \
             mock.patch('summarizer.refcheck.by_title', return_value=('', '')) as by_title, \
             mock.patch('summarizer.refcheck.crossref_fields', return_value={'journal': 'MNRAS & Co', 'year': '2021', 'doi': '10.1093/mnras/x1'}), \
             mock.patch('summarizer.generated.chat_json', return_value=verdicts) as chat:
            result = refcheck.check(self.BIB, self.TEX, 'en')
        by_title.assert_called_once_with('A book with no identifier')
        kinds = {p['key']: [i['kind'] for i in p['issues']] for p in result['problems']}
        self.assertEqual(kinds, {'retracted': ['retracted'], 'pre': ['published', 'old_version'],
                                 'again': ['duplicate', 'published'], 'nothing': ['unknown']})
        self.assertEqual(result['problems'][0]['key'], 'retracted')  # serious problems first
        self.assertIn('journal = {MNRAS \\& Co}', result['fixed_bib'])
        self.assertIn('@article{pre,', result['fixed_bib'])
        self.assertNotIn('archiveprefix', result['fixed_bib'])
        self.assertEqual(result['claims_checked'], 3)
        self.assertEqual([(c['key'], c['verdict']) for c in result['claims']], [('pre', 'contradicted')])
        self.assertIn('We see thick clouds.', chat.call_args[1]['system'])

    def test_page_is_premium_and_report_is_private(self):
        self.assertContains(self.client_for(self.user()).get('/en/refcheck/'), 'Discover Premium')
        client = self.client_for(self.user('pro', paid=True))
        self.assertContains(client.post('/en/refcheck/', {'bibtext': 'no entries here'}), 'We found no BibTeX entry')
        result = {'entries': 1, 'problems': [], 'claims': [], 'claims_checked': 0, 'with_tex': False,
                  'counts': {'error': 0, 'warning': 0, 'info': 0}, 'fixed_bib': '@article{a,\n}', 'corrected': 1}
        with mock.patch('summarizer.refcheck.check', return_value=result) as check:
            r = client.post('/en/refcheck/', {'bibtext': '@misc{a, title={T}}'})
        self.assertEqual(check.call_args[0][1], '')
        self.assertContains(client.get(r['Location']), 'No problem found')
        self.assertEqual(client.get(r['Location'] + 'bib/').content, b'@article{a,\n}')
        self.assertEqual(self.client_for(self.user('other', paid=True)).get(r['Location']).status_code, 404)


class ChangesTests(ToolsTestCase):
    def record(self, version, abstract, comment=''):
        return {'arxiv_id': '2101.00001v%d' % version, 'title': 'Clouds on a hot planet', 'abstract': abstract,
                'comment': comment, 'published': timezone.now(), 'updated': timezone.now(), 'authors': [],
                'categories': [], 'doi': '', 'journal_ref': ''}

    def test_base_id_and_changed_paragraphs(self):
        self.assertEqual(changes.base_id('2101.00001v2'), ('2101.00001', 2))
        self.assertEqual(changes.base_id('hep-th--9901001'), ('hep-th/9901001', 0))
        old = ['Intro paragraph that stays the same in both versions.', 'We find a mass of 3 Jupiter masses for the planet.',
               'Same conclusion paragraph, with a typo fixed here.']
        new = ['Intro paragraph that stays the same in both versions.', 'We find a mass of 5 Jupiter masses for the planet.',
               'Same conclusion paragraph, with a typo fixed hère.', 'A new appendix on the data.']
        diff = changes.changed_paragraphs(old, new)
        self.assertIn('- We find a mass of 3', diff)
        self.assertIn('+ We find a mass of 5', diff)
        self.assertIn('+ A new appendix', diff)
        self.assertNotIn('Intro', diff)

    def test_compare_reads_full_text_only_under_an_open_license(self):
        self.paper()  # CC BY
        answer = {'verdict': 'The mass changed.', 'results_changed': True,
                  'changes': [{'kind': 'writing', 'importance': 'minor', 'detail': 'Typos.'},
                              {'kind': 'results', 'importance': 'major', 'detail': 'Mass from 3 to 5.'}]}
        records = {1: self.record(1, 'Old abstract.'), 2: self.record(2, 'New abstract.', 'v2: corrected mass')}
        with mock.patch('summarizer.changes.version_record', side_effect=lambda base, v: records[v]), \
             mock.patch('summarizer.changes.version_html', side_effect=lambda base, v: ['Mass is %d.' % (v + 2) * 3]) as html, \
             mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            result = changes.compare('2101.00001', 1, 2, 'en')
        self.assertTrue(result['full_text'])
        self.assertEqual(html.call_count, 2)
        self.assertIn('<changed_paragraphs>', chat.call_args[1]['system'])
        self.assertIn('v2: corrected mass', chat.call_args[1]['system'])
        self.assertEqual([c['kind'] for c in result['changes']], ['results', 'writing'])  # major first
        ArxivPaper.objects.update(license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/')
        with mock.patch('summarizer.changes.version_record', side_effect=lambda base, v: records[v]), \
             mock.patch('summarizer.changes.version_html') as html, \
             mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            result = changes.compare('2101.00001', 1, 2, 'en')
        html.assert_not_called()
        self.assertFalse(result['full_text'])
        self.assertNotIn('<changed_paragraphs>', chat.call_args[1]['system'])

    def test_license_from_arxiv_when_the_paper_is_not_stored(self):
        record = mock.Mock(status_code=200, text='<arXiv><license>http://creativecommons.org/licenses/by/4.0/</license></arXiv>')
        with mock.patch('summarizer.changes.requests.get', return_value=record) as get:
            self.assertTrue(changes.can_read('2101.00009'))
            self.assertTrue(changes.can_read('2101.00009'))
        get.assert_called_once()  # cached
        self.paper('2101.00008v1')
        ArxivPaper.objects.filter(arxiv_id='2101.00008v1').update(license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/')
        with mock.patch('summarizer.changes.requests.get') as get:
            self.assertFalse(changes.can_read('2101.00008'))
        get.assert_not_called()  # the stored license comes first

    def test_new_version_of_a_followed_paper_is_emailed_once(self):
        user = self.user(paid=True)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', known='[]')
        now = timezone.now()
        latest = [[self.record(1, 'A')]]
        with mock.patch('summarizer.digest.fetch_ids', side_effect=lambda ids: latest[0]):
            self.assertEqual(changes.check_versions(now), 0)  # first sight: nothing to report
            latest[0] = [self.record(3, 'C')]
            self.assertEqual(changes.check_versions(now), 1)
        self.assertEqual(PaperVersion.objects.get().version, 3)
        result = {'title': 'Clouds on a hot planet', 'old': 1, 'new': 3, 'verdict': 'The mass changed.',
                  'results_changed': True, 'changes': [{'kind': 'results', 'importance': 'major', 'detail': 'Mass from 3 to 5.'}]}
        with mock.patch('summarizer.changes.compare', return_value=result) as compare:
            self.assertEqual(changes.send_alerts(now, 5), 1)
            self.assertEqual(changes.send_alerts(now, 5), 0)
        compare.assert_called_once()
        self.assertIn('New version: Clouds on a hot planet', mail.outbox[0].subject)
        self.assertIn('Mass from 3 to 5.', mail.outbox[0].body)
        self.assertIn('/en/changes/2101.00001v3/?from=1', mail.outbox[0].body)

    def test_crawlers_and_free_accounts_never_make_the_page_query_arxiv(self):
        with mock.patch('summarizer.changes.latest_version') as latest:
            self.assertContains(Client().get('/en/changes/2101.00001v1/'), 'Discover Premium')
            self.assertContains(Client().get('/en/changes/2101.00001v1/?to=abc'), 'Discover Premium')
        latest.assert_not_called()

    def test_page_is_premium(self):
        with mock.patch('summarizer.changes.latest_version', return_value=2):
            self.assertContains(self.client_for(self.user()).get('/en/changes/2101.00001v1/'), 'Discover Premium')
            client = self.client_for(self.user('pro', paid=True))
            page = client.get('/en/changes/2101.00001v1/')
            self.assertContains(page, 'Compare v1 with v2')
            with mock.patch('summarizer.changes.compare', return_value={'title': 'T', 'old': 1, 'new': 2, 'verdict': 'Same.',
                                                                         'results_changed': False, 'changes': [], 'full_text': True}):
                self.assertIn('id', client.post('/en/changes/2101.00001v1/start/', {'from': 1, 'to': 2}).json())
            self.assertContains(client.get('/en/changes/2101.00001v1/'), 'Same main results')
        with mock.patch('summarizer.changes.latest_version', return_value=1):
            self.assertContains(client.get('/en/changes/2101.00001v1/'), 'only one version')


class TeamsTests(ToolsTestCase):
    def work(self, n, title, authors, cited=0):
        return {'id': 'W%d' % n, 'doi': 'https://doi.org/10.1/w%d' % n, 'display_name': title, 'cited_by_count': cited,
                'publication_date': '2026-01-01', 'authorships': [
                    {'author': {'display_name': name}, 'author_position': position,
                     'institutions': [{'id': 'I-' + inst, 'display_name': inst, 'country_code': 'FR'}]}
                    for name, inst, position in authors]}

    def test_groups_by_institution_and_describes_them(self):
        found = [self.work(1, 'Exomoons by astrometry', [('Ann Lee', 'Paris', 'first'), ('Bob Ray', 'Paris', 'last')], 10),
                 self.work(2, 'Exomoon transits', [('Lee, Ann', 'Paris', 'first'), ('Zoe Q', 'Leiden', 'last')], 3),
                 self.work(3, 'Protein folding', [('Max P', 'Basel', 'first')], 99),
                 self.work(4, 'Huge review of exomoons', [('Big %d' % i, 'Inst%d' % i, 'middle') for i in range(30)] +
                           [('Zoe Q', 'Leiden', 'last')])]
        plan = {'title': 'Exomoons', 'queries': ['exomoon astrometry', 'exomoon transits']}
        notes = {'groups': [{'id': 0, 'focus': 'Exomoon searches.', 'approach': 'Astrometry.'}], 'landscape': 'Two groups.'}

        def embed(texts):
            return np.array([[1.0, 0.0]] + [[0.0, 1.0] if 'Protein' in t else [1.0, 0.0] for t in texts[1:]], dtype=np.float32)
        with mock.patch('summarizer.teams.works', return_value=found) as works, \
             mock.patch('summarizer.teams.MIN_CLOSE', 1), \
             mock.patch('summarizer.similar.embed', side_effect=embed), \
             mock.patch('summarizer.generated.chat_json', side_effect=[plan, notes]):
            result = teams.find('exomoons', 'en')
        self.assertEqual(works.call_count, 2)
        self.assertEqual(result['papers'], 3)  # protein folding is about something else
        self.assertEqual([g['institution'] for g in result['groups']], ['Paris', 'Leiden'])
        paris = result['groups'][0]
        self.assertEqual(paris['papers'], 2)
        self.assertEqual(paris['citations'], 13)
        self.assertEqual(paris['people'][0], {'name': 'Ann Lee', 'papers': 2, 'last': 0})  # 'Lee, Ann' is the same person
        self.assertEqual(paris['focus'], 'Exomoon searches.')
        self.assertEqual(result['landscape'], 'Two groups.')
        self.assertNotIn('Inst3', [g['institution'] for g in result['groups']])  # middle authors of a huge paper

    def test_page_is_premium_and_starts_from_a_project(self):
        self.assertContains(self.client_for(self.user()).get('/en/teams/'), 'Discover Premium')
        user = self.user('pro', paid=True)
        project = WatchedProject.objects.create(user=user, title='Moons', description='x' * 300, search_text='Exomoons by astrometry.')
        client = self.client_for(user)
        self.assertContains(client.post('/en/teams/', {'topic': 'short'}), 'a few words at least')
        with mock.patch('summarizer.teams.find', return_value={'title': 'Moons', 'groups': [], 'papers': 0, 'landscape': ''}) as find:
            r = client.post('/en/teams/', {'topic': '', 'project': project.pk})
        self.assertEqual(find.call_args[0][0], 'Moons\n\nExomoons by astrometry.')
        self.assertContains(client.get(r['Location']), 'no recent paper')
        self.assertEqual(self.client_for(self.user('other', paid=True)).get(r['Location']).status_code, 404)


class SlidesTests(ToolsTestCase):
    ANSWER = {'title': 'Clouds on a hot planet', 'slides': [{'title': 'Context', 'bullets': ['Hot planets', ''], 'figure': 2, 'notes': 'Say it.'},
                         {'title': 'Again', 'bullets': ['Same figure'], 'figure': 2},
                         {'title': 'Made up', 'bullets': ['x'], 'figure': 9}, {'bullets': ['no title']}],
              'strengths': ['Big telescope'], 'limitations': ['One planet'], 'questions': ['Why?']}

    def test_figures_only_under_an_open_license_and_once_each(self):
        self.paper()  # CC BY
        shown = [{'number': 1, 'url': 'https://arxiv.org/html/2101.00001v1/x1.png', 'caption': 'Figure 1: the spectrum.'},
                 {'number': 2, 'url': 'https://arxiv.org/html/2101.00001v1/x2.png', 'caption': 'Figure 2: the clouds.'}]
        with mock.patch('summarizer.slides.figures', return_value=shown), \
             mock.patch('summarizer.paper_source.paper_text', return_value='Full text of the clouds paper.'), \
             mock.patch('summarizer.generated.chat_json', return_value=self.ANSWER) as chat:
            deck = slides.write('2101.00001v1', 'en')(0)
        self.assertIn('Figure 2: the clouds.', chat.call_args[0][0])
        self.assertEqual([s['title'] for s in deck['slides']], ['Context', 'Again', 'Made up'])
        self.assertEqual([s['figure'] and s['figure']['number'] for s in deck['slides']], [2, None, None])
        self.assertEqual(deck['slides'][0]['bullets'], ['Hot planets'])
        self.assertEqual(deck['license'], 'CC BY 4.0')
        ArxivPaper.objects.update(license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/')
        with mock.patch('summarizer.slides.figures') as figures, \
             mock.patch('summarizer.generated.chat_json', return_value=self.ANSWER) as chat:
            deck = slides.write('2101.00001v1', 'en')(0)
        figures.assert_not_called()
        self.assertTrue(deck['abstract_only'])
        self.assertNotIn('Figure', chat.call_args[0][0])

    def test_page_is_premium_and_shows_the_deck(self):
        self.paper()
        self.assertContains(self.client_for(self.user()).get('/en/journal-club/2101.00001v1/'), 'Discover Premium')
        client = self.client_for(self.user('pro', paid=True))
        with mock.patch('summarizer.slides.figures', return_value=[]), \
             mock.patch('summarizer.paper_source.paper_text', return_value='Full text.'), \
             mock.patch('summarizer.generated.chat_json', return_value=self.ANSWER):
            self.assertIn('id', client.post('/en/journal-club/2101.00001v1/start/').json())
        page = client.get('/en/journal-club/2101.00001v1/')
        self.assertContains(page, 'Present full screen')
        self.assertContains(page, 'Big telescope')
        self.assertEqual(client.get('/en/journal-club/9999.99999v1/').status_code, 404)


class ConferenceTests(ToolsTestCase):
    def test_lists_ranks_and_explains_the_talks(self):
        program = ('Monday 9:00 Session A\n\nClouds on hot Jupiters, by Ann Lee\n\n' + 'x ' * 7000 +
                   '\n\nTuesday 10:00\n\nProtein folding, by Max P\n\nExomoons with Gaia, by Zoe Q')
        self.assertEqual(len(conference.chunks(program)), 2)

        def chat(prompt, **kwargs):
            if 'piece of a conference program' in prompt:
                found = [('Clouds on hot Jupiters', 'Ann Lee', '')] if 'Clouds' in kwargs['system'] else \
                        [('Protein folding', 'Max P', ''), ('Exomoons with Gaia', 'Zoe Q', 'Tuesday 10:00'),
                         ('clouds on hot jupiters', 'Ann Lee', '')]
                return {'talks': [{'title': t, 'speakers': sp, 'time': tm} for t, sp, tm in found]}
            return {'talks': [{'id': '1', 'priority': 'should', 'why': 'Close.'},
                              {'id': '0', 'priority': 'must', 'why': 'Same planets.'},
                              {'id': '2', 'priority': 'skip'}]}

        def embed(texts):
            return np.array([[1.0, 0.0]] + [[0.0, 1.0] if 'Protein' in t else [0.9, 0.1] if 'Exomoons' in t else [1.0, 0.0]
                                            for t in texts[1:]], dtype=np.float32)
        with mock.patch('summarizer.generated.chat_json', side_effect=chat), \
             mock.patch('summarizer.similar.embed', side_effect=embed):
            result = conference.find(program, 'Clouds of exoplanets', 'en')
        self.assertEqual(result['listed'], 3)  # the talk listed in both pieces counts once
        self.assertEqual([(t['title'], t['priority']) for t in result['talks']],
                         [('Clouds on hot Jupiters', 'must'), ('Exomoons with Gaia', 'should')])
        self.assertEqual(result['people'], ['Ann Lee', 'Zoe Q'])

    def test_page_is_premium_and_uses_the_digest_profile(self):
        self.assertContains(self.client_for(self.user()).get('/en/conference/'), 'Discover Premium')
        user = self.user('pro', paid=True)
        DigestSubscription.objects.create(user=user, categories='astro-ph.EP', profile='I study clouds on hot planets.')
        client = self.client_for(user)
        self.assertContains(client.get('/en/conference/'), 'I study clouds on hot planets.')
        self.assertContains(client.post('/en/conference/', {'program': 'short', 'interests': 'clouds of planets'}), 'Paste the program')
        with mock.patch('summarizer.conference.find', return_value={'talks': [], 'listed': 4, 'people': []}) as find:
            r = client.post('/en/conference/', {'program': 'talk ' * 100, 'interests': 'clouds of hot planets', 'name': 'EAS 2026'})
        self.assertEqual(find.call_args[0][1], 'clouds of hot planets')
        self.assertContains(client.get(r['Location']), 'EAS 2026')
        self.assertContains(client.get('/en/conference/'), 'EAS 2026')


class QuestionsTests(ToolsTestCase):
    def test_reads_closing_sections_and_groups_questions(self):
        user = self.user(paid=True)
        self.paper()  # CC BY
        self.paper('2101.00002v1', title='Closed paper', summary='Summary of the closed paper.')
        ArxivPaper.objects.filter(arxiv_id='2101.00002v1').update(license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/')
        collection = Collection.objects.create(user=user, name='Clouds')
        for aid in ('2101.00001v1', '2101.00002v1'):
            CollectionPaper.objects.create(collection=collection, arxiv_id=aid, title='T ' + aid, authors='Ann Lee')
        text = 'Intro. ' * 600 + '\n\n## 5 Limitations\n\nWe only observed one planet.\n\n## References\n[1] X'
        answer = {'questions': [{'question': 'Are clouds common?', 'kind': 'untested', 'papers': ['2101.00001v1']},
                                {'question': 'Do the papers agree on cloud height?', 'kind': 'disagreement',
                                 'papers': ['2101.00001v1', '2101.00002v1', 'made-up']}],
                  'projects': [{'title': 'A survey', 'idea': 'Observe ten planets.', 'papers': ['2101.00002v1']}]}
        client = self.client_for(user)
        with mock.patch('summarizer.paper_source.paper_text', return_value=text), \
             mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            self.assertIn('id', client.post('/en/collections/%d/questions/' % collection.pk).json())
        blocks = chat.call_args[1]['system']
        self.assertIn('We only observed one planet.', blocks)  # the open paper: its limitations section
        self.assertNotIn('Intro. Intro.', blocks)
        self.assertIn('Summary of the closed paper.', blocks)  # the closed one: abstract and summary
        row = Generated.objects.get(kind='questions')
        self.assertEqual([q['question'] for q in row.data()['questions']],
                         ['Do the papers agree on cloud height?', 'Are clouds common?'])  # shared questions first
        self.assertEqual(len(row.data()['questions'][0]['papers']), 2)
        page = client.get('/en/collections/%d/' % collection.pk)
        self.assertContains(page, 'Papers disagree')
        self.assertContains(page, 'Observe ten planets.')
        self.assertContains(self.client_for(self.user('free')).post('/en/collections/%d/questions/' % collection.pk), '', status_code=404)


class UsageTests(ToolsTestCase):
    def openai(self, content='{"ok": 1}', prompt=1000, completion=200):
        return mock.Mock(status_code=200, raise_for_status=lambda: None, json=lambda: {
            'model': 'gpt-test', 'usage': {'prompt_tokens': prompt, 'completion_tokens': completion},
            'choices': [{'message': {'content': content}}]})

    def test_background_tools_are_counted_by_kind_and_user(self):
        from summarizer.models import AIPrice, AIUsage
        user = self.user(paid=True)
        with mock.patch('summarizer.generated.requests.post', return_value=self.openai()):
            generated.start('refcheck', 'k1', 'en', lambda pk: generated.chat_json('x'), user=user)
            generated.start('refcheck', 'k2', 'en', lambda pk: generated.chat_json('x'), user=user)
            generated.chat_json('outside any task')
        row = AIUsage.objects.get(task='refcheck')
        self.assertEqual((row.user, row.calls, row.prompt_tokens, row.completion_tokens), (user, 2, 2000, 400))
        self.assertEqual(AIUsage.objects.get(task='other').calls, 1)
        AIPrice.objects.create(model='gpt-test', input_per_million=2, output_per_million=10)
        summary = usage.summary(7)
        self.assertEqual(summary['tasks'][0]['name'], 'refcheck')
        self.assertAlmostEqual(summary['tasks'][0]['cost'], (2000 * 2 + 400 * 10) / 1e6)
        self.assertEqual(summary['users'][0]['name'], user.username)
        self.assertEqual(summary['unpriced'], [])

    def test_counting_never_breaks_the_call(self):
        with mock.patch('summarizer.usage._save', side_effect=RuntimeError):
            self.assertEqual(usage.record({'model': 'm', 'usage': {'prompt_tokens': 1}}), {'model': 'm', 'usage': {'prompt_tokens': 1}})
        self.assertEqual(usage.record(None), None)

    def test_admin_shows_the_costs(self):
        from summarizer.models import AIUsage
        AIUsage.objects.create(day=timezone.now().date(), task='summary', model='gpt-unpriced', calls=3, prompt_tokens=10)
        admin = User.objects.create_superuser('boss', 'boss@example.com', 'x')
        client = self.client_for(admin)
        page = client.get('/admin/summarizer/aiusage/')
        self.assertContains(page, 'Par tâche')
        self.assertContains(page, 'gpt-unpriced')
        self.assertContains(client.get('/admin/'), 'Coût IA (7 j)')


class ToolsPageTests(ToolsTestCase):
    def test_tools_page_home_and_faq_present_the_tools(self):
        page = Client().get('/en/tools/')
        for name in ('Scoop watch', 'Bibliography check', 'Who works on it?', 'Conference companion', 'Journal club'):
            self.assertContains(page, name)
        self.assertContains(page, '/en/faq/#journal-club')
        self.assertContains(Client().get('/fr/tools/'), 'Veille anti-scoop')
        self.assertContains(Client().get('/en/faq/'), 'id="scoops"')

    def test_announcement_goes_once_to_premium_subscribers(self):
        from django.core.management import call_command
        from io import StringIO
        self.user('free')
        self.user('pro', paid=True)
        out = StringIO()
        call_command('announce_tools', stdout=out)
        self.assertIn('1 Premium subscriber', out.getvalue())
        self.assertEqual(len(mail.outbox), 0)  # nothing without --send
        call_command('announce_tools', '--send', stdout=StringIO())
        call_command('announce_tools', '--send', stdout=StringIO())
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].to, ['pro@example.com'])
        self.assertIn('/en/scoops/', mail.outbox[0].body)


class OpenAlexBudgetTests(ToolsTestCase):
    def test_tools_leave_a_reserve_to_the_paper_search_and_say_when_limited(self):
        from summarizer import sources
        with mock.patch('summarizer.sources.OPENALEX_SEARCHES_PER_DAY', 3), mock.patch('summarizer.sources.TOOL_RESERVE', 1):
            self.assertEqual([sources.tool_search_allowed() for _ in range(3)], [True, True, False])
            with mock.patch('summarizer.sources._json') as call:
                self.assertIsNone(teams.works('exomoons', timezone.now().date()))
                self.assertEqual(refcheck.by_title('A title'), ('', ''))
                self.assertTrue(refcheck.LIMITED.get())
                call.assert_not_called()
        with mock.patch('summarizer.teams.works', return_value=None), \
             mock.patch('summarizer.generated.chat_json', return_value={'title': 'T', 'queries': ['q']}):
            self.assertTrue(teams.find('exomoons', 'en')['limited'])


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
             mock.patch('summarizer.letters.send', return_value=0) as letter_send, \
             mock.patch('summarizer.saved_searches.run', return_value=0) as search_watches:
            call_command('send_digest')
        trends_update.assert_called_once()
        self.assertEqual(citation_run.call_args[0][1], digest.MAX_EMAILS_PER_RUN - 3)
        self.assertEqual(letter_send.call_args[0][1], digest.MAX_EMAILS_PER_RUN - 5)
        self.assertEqual(search_watches.call_args[0][1], digest.MAX_EMAILS_PER_RUN - 5)


class PageTests(ToolsTestCase):
    def test_new_pages_load(self):
        user = self.user(paid=True)
        client = self.client_for(user)
        for url in ('/en/collections/', '/fr/collections/', '/en/integrations/', '/en/missed/', '/en/citations/',
                    '/en/trends/', '/fr/letters/', '/en/letters/ai/', '/en/premium/'):
            self.assertEqual(client.get(url).status_code, 200, url)
        for url, text in (('/en/collections/', 'Chat with all the papers of a collection'),
                          ('/en/missed/', 'We remove the papers your draft already cites'),
                          ('/fr/citations/', 'Créer un compte gratuit'),
                          ('/en/integrations/', 'See what Premium includes')):
            self.assertContains(Client().get(url), text, msg_prefix=url)
        # fields that take arXiv ids come with a paper search
        self.assertContains(client.get('/en/citations/'), 'data-target="follow-ids"')
        self.assertContains(client.get('/fr/citations/'), 'Titre, mots-clés, auteur ou DOI')
        self.assertContains(client.get('/en/digest/'), 'data-target="reference_papers"')
        self.assertContains(self.client_for(self.user('free')).get('/en/citations/'), 'Discover Premium')
        self.assertEqual(Client().get('/en/missed/abc/').status_code, 302)
        self.assertContains(Client().get('/en/premium/'), 'What did I miss?')
        self.assertEqual(Client().get('/en/letters/nope/').status_code, 404)
