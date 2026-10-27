"""Tests of the paper tools and Premium tools: reading levels, press kit, collections, exports, Zotero,
'What did I miss?', literature review, citation alerts, audio digest, rising topics and weekly letters."""
import datetime
import json
import re
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import Client, TestCase, override_settings
from django.utils import timezone, translation

from summarizer import audio, changes, citations, conference, digest, generated, letters, library, literature, missed, presskit, questions, refcheck, scoops, slides, sources, teams, trends, usage
from summarizer.models import (AnnouncedPaper, ArxivBatch, ArxivPaper, Author, CitationCache, Collection,
                               CollectionPaper, DigestItem, DigestSubscription, DigestVector, FollowedPaper, Generated, PaperVersion,
                               Letter, LetterSubscription, PaperAudio, PaperAuthor, PaperScore, RisingTopics, ScoopMatch,
                               SummaryPaper, TermWeek, UserPlan, WatchedProject, ZoteroAccount)


class ToolsTestCase(TestCase):
    """Background work runs inline, and keeps the test transaction's connection."""

    def setUp(self):
        cache.clear()
        patches = [mock.patch('summarizer.generated.background', side_effect=lambda work, *args: work(*args)),
                   mock.patch('summarizer.generated.release_connection'),
                   mock.patch('summarizer.citations.fetch_meta', return_value=None),
                   mock.patch('summarizer.citations.fetch_html', return_value=''),
                   mock.patch('summarizer.publication_status.lookup', return_value={
                       'status': 'unknown', 'checked_at': timezone.now().isoformat(), 'sources': [],
                       'checked_sources': [], 'unavailable': [], 'peer_review': 'not_independently_verified'})]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def user(self, name='ann', paid=False, tier='plus'):
        """paid: a running subscription, Premium+ unless tier='premium' (which does not open PLUS_TOOLS)."""
        user = User.objects.create_user(name, email='%s@example.com' % name, password='x')
        if paid:
            UserPlan.objects.create(user=user, paid=True, tier=tier)
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
        self.assertIn('Original scientific abstract.', chat.call_args.args[0])
        self.assertNotIn('Original scientific abstract.', chat.call_args.kwargs['system'])
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
            self.assertEqual(Generated.objects.get(kind='presskit').data()['abstract_only'], expected_abstract)
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
        rows = {'rows': [{'id': '2101.00001v1', 'question': 'Q1', 'method': 'M1', 'data': 'D1', 'results': 'R1',
                          'limitations': 'L1'},
                         {'id': '2101.00002v1', 'question': 'Q2'}, {'id': 'made-up', 'question': 'Q'}]}
        lists = {'agreements': ['Both see clouds'], 'disagreements': [], 'gaps': ['No one looks at night']}
        client = self.client_for(user)
        with mock.patch('summarizer.paper_source.paper_text', return_value=''), \
             mock.patch('summarizer.generated.chat_json', side_effect=[rows, lists]) as chat:
            self.assertEqual(client.post('/en/collections/%s/compare/' % collection.pk).status_code, 200)
        self.assertIn('Long summary of the clouds.', chat.call_args_list[0][1]['system'])
        page = client.get('/en/collections/%s/' % collection.pk)
        self.assertContains(page, 'No one looks at night')
        self.assertContains(page, 'R1')
        csv = client.get('/en/collections/%s/export/csv/' % collection.pk).content.decode()
        self.assertIn('2101.00001,Clouds on a hot planet,Q1,,M1,,D1,,R1,,L1,', csv)
        self.assertNotIn('made-up', csv)
        free = self.user('free')
        mine = Collection.objects.create(user=free, name='F')
        self.assertEqual(self.client_for(free).post('/en/collections/%s/compare/' % mine.pk).status_code, 403)


class ExportTests(ToolsTestCase):
    def test_zotero_username_is_not_silently_converted_to_a_user_id(self):
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.exports.check_zotero') as check:
            for username in ('ExampleUser', 'Example123'):
                response = client.post('/en/integrations/', {'library_id': username, 'api_key': 'dummySecretForTest'})
                self.assertContains(response, 'numeric Zotero user ID, not your username')
                self.assertNotContains(response, 'dummySecretForTest')
            check.assert_not_called()
        self.assertFalse(ZoteroAccount.objects.exists())

    def test_zotero_key_is_masked_and_invalid_characters_are_not_silently_removed(self):
        client = self.client_for(self.user(paid=True))
        self.assertContains(client.get('/en/integrations/'), 'type="password" name="api_key"')
        with mock.patch('summarizer.exports.check_zotero') as check:
            response = client.post('/en/integrations/', {'library_id': '123', 'api_key': 'dummy&SecretForTest'})
            self.assertContains(response, 'Copy the API key exactly')
            self.assertNotContains(response, 'dummy&amp;SecretForTest')
            check.assert_not_called()

    def test_ris_import_needs_no_connection_and_contains_original_abstract(self):
        paper = self.paper()
        with mock.patch('summarizer.exports.requests.get') as get:
            response = Client().get('/en/export/2101.00001v1/ris/')
        self.assertContains(response, 'AB  - '+paper.abstract)
        self.assertNotContains(response, 'Long summary of the clouds.')
        self.assertFalse(ZoteroAccount.objects.exists())
        get.assert_not_called()
        self.assertContains(Client().get('/en/integrations/'), 'No API key is needed to import a RIS file')

    def test_paper_exports_and_page_tools(self):
        self.paper()
        ris = Client().get('/en/export/2101.00001v1/ris/')
        self.assertIn('AU  - Ann Lee', ris.content.decode())
        self.assertIn('attachment', ris['Content-Disposition'])
        md = Client().get('/en/export/2101.00001v1/md/').content.decode()
        self.assertIn('# Clouds on a hot planet', md)
        self.assertIn('## Summary', md)
        self.assertEqual(Client().get('/en/export/2101.00001v1/pdf/').status_code, 404)
        page = Client().get('/en/paper/2101.00001v1/')
        self.assertContains(page, '<meta name="citation_title" content="Clouds on a hot planet" />', html=False)
        self.assertContains(page, '<meta name="citation_arxiv_id" content="2101.00001" />', html=False)
        self.assertContains(page, 'id="tools"')
        self.assertContains(page, 'Group papers in collections')
        self.assertContains(page, '/en/register/?next=/en/paper/2101.00001v1/')

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


class LiteratureTests(ToolsTestCase):
    FOUNDING = 'Burial below the oxic layer slows the degradation of plastic particles.'

    def work(self, wid, title, year, cited=10, refs=(), kind='article', doi=None):
        index = {}
        for i, w in enumerate(('%s. %s' % (title, self.FOUNDING if wid == 'W100' else 'We study it in coastal cores.')).split()):
            index.setdefault(w, []).append(i)
        return {'id': 'https://openalex.org/' + wid, 'doi': 'https://doi.org/' + (doi or '10.1000/%s' % wid.lower()),
                'display_name': title, 'authorships': [{'author': {'display_name': 'Ann Lee'}}],
                'abstract_inverted_index': index, 'publication_date': '%d-05-01' % year,
                'type': kind, 'cited_by_count': cited, 'locations': [],
                'referenced_works': ['https://openalex.org/' + r for r in refs]}

    def openalex(self, url, params=None, **kwargs):
        filters = params.get('filter', '')
        if 'openalex:' in filters:
            self.assertEqual(sorted(filters[len('openalex:'):].split('|')), ['W100', 'W101'])
            unread = dict(self.work('W101', 'Lost at sea', 2004, cited=7000), abstract_inverted_index=None)
            return {'results': [self.work('W100', 'Plastic burial in sediments', 2005, cited=900), unread]}
        if 'type:review' in filters:
            return {'results': [self.work('W50', 'Microplastics in sediments: a review', 2022, kind='review')]}
        self.assertNotIn('?', params.get('search', ''))
        if params.get('sort'):
            return {'results': [self.work('W1', 'Sediment microplastic sinks', 2012, cited=400)]}
        if 'from_publication_date' in filters:
            return {'results': [self.work('W7', 'Biofilms protect sediment microplastics', 2025)]}
        return {'results': [self.work('W%d' % i, 'Microplastic persistence study %d' % i, 2015 + i, refs=['W100', 'W101'])
                            for i in range(2, 6)] +
                           [self.work('W9', 'Sediment burial preserves microplastics', 2023, cited=33,
                                      doi='10.48550/arxiv.2301.00009')]}

    def chat(self, prompt, paid=True, max_tokens=0, system=None, timeout=0):
        if 'research question above' in prompt:
            return {'question': 'What mechanisms explain microplastic persistence in marine sediments?',
                    'title': 'Microplastics in sediments', 'queries': ['microplastic sediment?', 'plastic burial'],
                    'include': 'Papers on why plastics last in sediments.', 'concepts': [{'concept': 'burial'}]}
        if 'Rate each paper' in prompt:
            ids = re.findall(r'<paper id="(p\d+)">\nTitle: ([^\n]*)', system)
            return {'papers': [{'id': i, 'relevance': 0 if 'Galaxy' in t else 3, 'approach': 'cores'} for i, t in ids]}
        self.assertIn('<paper n="1">\nTitle: Plastic burial in sediments', system)  # oldest first
        return {'overview': [{'text': 'Burial matters.', 'papers': [1, 99]}],
                'themes': [{'name': 'Burial', 'summary': 'S.', 'papers': [1, 2], 'strengths': 'a', 'limits': 'b',
                            'findings': [{'text': 'Burial slows degradation.', 'papers': [1], 'quote': self.FOUNDING},
                                         {'text': 'Made up.', 'papers': [2], 'quote': 'A sentence no abstract holds at all.'}]}],
                'consensus': [], 'disagreements': [], 'gaps': [{'text': 'Deep sea.', 'papers': []}]}

    def test_question_is_searched_snowballed_screened_and_synthesized_with_checked_quotes(self):
        arxiv = [{'arxiv_id': '2301.00009v1', 'title': 'Sediment burial preserves\n microplastics', 'abstract': 'Cores show it.',
                  'authors': ['Bo Wu'], 'published': timezone.now().replace(year=2023)},
                 {'arxiv_id': '2301.00010v1', 'title': 'Galaxy clusters', 'abstract': 'Far away.', 'authors': ['X'],
                  'published': timezone.now()}]

        def embed(texts):
            return np.array([[0.0, 1.0] if 'Galaxy' in t else [1.0, 0.0] for t in texts], dtype=np.float32)
        with mock.patch('summarizer.sources._json', side_effect=self.openalex), \
             mock.patch('summarizer.digest.arxiv_get', return_value=arxiv), \
             mock.patch('summarizer.similar.embed', side_effect=embed), \
             mock.patch('summarizer.generated.chat_json', side_effect=self.chat):
            result = literature.review('Pourquoi les microplastiques persistent-ils dans les sédiments ?', 'fr')
        papers = {p['key']: p for p in result['papers']}
        merged = papers['2301.00009v1']  # the arXiv preprint and its OpenAlex record are one paper
        self.assertEqual((merged['cited_by'], sorted(merged['found_by'])), (33, ['arxiv', 'relevance']))
        self.assertNotIn('2301.00010v1', papers)  # screened out
        self.assertEqual(result['papers'][0]['title'], 'Plastic burial in sediments')
        self.assertIn('founding', result['papers'][0]['labels'])
        self.assertIn('review', papers['doi_10.1000--w50']['labels'])
        self.assertIn('recent', papers['doi_10.1000--w7']['labels'])
        self.assertEqual(result['overview'][0]['papers'], [1])  # paper 99 does not exist
        found, made_up = result['themes'][0]['findings']
        self.assertEqual((found['quote'], found['quoted']), (self.FOUNDING, 1))
        self.assertEqual(made_up['quote'], '')
        coverage = result['coverage']
        self.assertEqual((coverage['snowballed'], coverage['quoted'], coverage['findings']), (2, 1, 2))
        self.assertEqual([p['title'] for p in coverage['unread']], ['Lost at sea'])  # no abstract: listed apart
        self.assertNotIn('doi_10.1000--w101', papers)
        self.assertTrue(coverage['thin'])
        self.assertEqual(coverage['limited'], [])
        self.assertEqual(result['title'], 'Microplastics in sediments')

    def test_page_is_premium_has_a_quota_and_saves_a_collection(self):
        self.assertContains(self.client_for(self.user()).get('/en/literature-review/'), 'Discover Premium')
        client = self.client_for(self.user('pro', paid=True))
        self.assertContains(client.post('/en/literature-review/', {'question': 'short'}), 'characters')
        done = {'question': 'Why do microplastics persist?', 'title': 'Microplastics', 'coverage': {
            'candidates': 40, 'screened': 30, 'relevant': 20, 'included': 1, 'labels': {}, 'limited': [], 'queries': ['q'],
            'by_source': {'relevance': 40}}, 'overview': [], 'themes': [], 'consensus': [], 'disagreements': [], 'gaps': [],
            'papers': [{'n': 1, 'key': 'doi_10.1000--w1', 'title': 'Sinks', 'authors': 'Ann Lee', 'year': 2012, 'venue': '',
                        'doi': '10.1000/w1', 'cited_by': 4, 'labels': ['founding'], 'approach': '', 'peripheral': False}]}
        with mock.patch('summarizer.literature.review', return_value=done) as review:
            r = client.post('/en/literature-review/', {'question': ' Why do  microplastics persist? '})
            self.assertEqual(review.call_args[0][:2], ('Why do microplastics persist?', 'en'))
            page = client.get(r['Location'])
            self.assertContains(page, 'How this review was searched')
            self.assertContains(page, 'Founding')
            saved = client.post(r['Location'])
            self.assertEqual(list(CollectionPaper.objects.values_list('arxiv_id', 'collection__name')),
                             [('doi_10.1000--w1', 'Microplastics')])
            self.assertContains(client.get(saved['Location']), 'The papers are in a new collection')
            for _ in range(literature.PER_DAY - 1):
                client.post('/en/literature-review/', {'question': 'Another question on microplastics'})
            self.assertContains(client.post('/en/literature-review/', {'question': 'One question too many today'}),
                                'reviews a day')
        self.assertEqual(Client().get('/en/literature-review/abc/').status_code, 302)
        self.assertEqual(self.client_for(self.user('free2')).get(r['Location']).status_code, 404)

    def test_a_review_is_shared_read_only_until_stopped(self):
        owner = self.client_for(self.user('pro', paid=True))
        row = Generated.objects.create(kind='literature', key='k1', user=User.objects.get(username='pro'), status='done',
                                       content=json.dumps({'question': 'Why do microplastics persist?', 'title': 'Microplastics',
                                                           'coverage': {'labels': {}, 'queries': [], 'by_source': {}},
                                                           'overview': [], 'themes': [], 'consensus': [], 'disagreements': [],
                                                           'gaps': [], 'papers': [{'n': 1, 'key': '2101.00001', 'title': 'Sinks',
                                                                                    'authors': 'Ann Lee', 'labels': []}]}))
        self.assertContains(owner.get('/en/literature-review/k1/'), 'Create a link')
        self.assertEqual(owner.post('/en/literature-review/k1/', {'action': 'share'}).status_code, 302)
        row.refresh_from_db()
        link = '/en/literature-review/shared/%s/' % row.share_token
        self.assertContains(owner.get('/en/literature-review/k1/'), link)
        page = Client().get(link)  # no account needed, and nothing to save or share again
        self.assertContains(page, 'Sinks')
        self.assertContains(page, 'Shared literature review')
        self.assertNotContains(page, 'Save these papers in a collection')
        self.assertNotContains(page, 'Create a link')
        self.assertFalse(CollectionPaper.objects.exists())
        owner.post('/en/literature-review/k1/', {'action': 'share'})  # again: the same link, 90 more days
        self.assertEqual(Generated.objects.get(pk=row.pk).share_token, row.share_token)
        self.assertEqual(self.client_for(self.user('other', paid=True)).post('/en/literature-review/k1/', {'action': 'unshare'}).status_code, 404)
        owner.post('/en/literature-review/k1/', {'action': 'unshare'})
        self.assertEqual(Client().get(link).status_code, 404)


class CitationTests(ToolsTestCase):
    def queue(self, user, now, *ids):
        citing = self.citing(*ids)
        CitationCache.objects.create(arxiv_id='2101.00001', citing=json.dumps(citing), total=len(citing), checked=now)
        return FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='Followed paper',
                                          known='[]', pending=json.dumps(list(citing)))

    def test_sent_alert_is_archived_and_survives_cache_changes_and_unfollow(self):
        from .models import CitationAlertHistory
        now = timezone.now()
        user = self.user(paid=True)
        followed = self.queue(user, now, 'c3')
        with mock.patch('summarizer.citations.stances', return_value={'c3': {'stance': 'supports', 'note': 'Confirms it.'}}):
            self.assertEqual(citations.send_alerts(now, 1), 1)
        history = CitationAlertHistory.objects.get(user=user)
        self.assertEqual((history.count, history.sent, history.lang), (1, now, 'en'))
        self.assertEqual(history.papers[0]['items'][0]['note'], 'Confirms it.')
        CitationCache.objects.all().delete()
        client = self.client_for(user)
        client.post('/en/citations/', {'action': 'unfollow', 'pk': followed.pk})
        page = client.get('/en/citations/')
        self.assertContains(page, 'Paper c3')
        self.assertContains(page, 'Confirms it.')
        self.assertContains(page, '/en/paper/2201.00003/')
        self.assertContains(page, 'As shown by [3], clouds are thick.')
        self.assertContains(client.get('/fr/citations/'), '1 nouvelle citation')
        self.assertEqual(citations.send_alerts(now, 1), 0)
        self.assertEqual(CitationAlertHistory.objects.count(), 1)

    def test_failed_or_unsent_email_does_not_archive_or_consume_citations(self):
        from .models import CitationAlertHistory
        now = timezone.now()
        user = self.user(paid=True)
        followed = self.queue(user, now, 'c1')
        for outcome in (RuntimeError('mail offline'), 0):
            kwargs = {'side_effect': outcome} if isinstance(outcome, Exception) else {'return_value': outcome}
            with mock.patch('summarizer.citations.EmailMultiAlternatives.send', **kwargs):
                self.assertEqual(citations.send_alerts(now, 1), 0)
            followed.refresh_from_db()
            self.assertEqual(json.loads(followed.pending), ['c1'])
            self.assertFalse(CitationAlertHistory.objects.exists())

    def test_new_citation_arriving_during_delivery_remains_queued(self):
        now = timezone.now()
        user = self.user(paid=True)
        followed = self.queue(user, now, 'c1')

        def send():
            FollowedPaper.objects.filter(pk=followed.pk).update(pending='["c1", "c2"]')
            return 1

        with mock.patch('summarizer.citations.EmailMultiAlternatives.send', side_effect=send):
            self.assertEqual(citations.send_alerts(now, 1), 1)
        followed.refresh_from_db()
        self.assertEqual(json.loads(followed.pending), ['c2'])
        self.assertEqual(json.loads(followed.known), ['c1'])

    @override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
    def test_online_history_includes_citations_beyond_email_limit(self):
        from .models import CitationAlertHistory
        now = timezone.now()
        user = self.user(paid=True)
        self.queue(user, now, 'c1', 'c2', 'c3')
        with mock.patch('summarizer.citations.MAX_PER_EMAIL', 2), mock.patch('summarizer.citations.stances', return_value={}):
            self.assertEqual(citations.send_alerts(now, 1), 1)
        row = CitationAlertHistory.objects.get()
        self.assertEqual(row.count, 3)
        self.assertEqual(len(row.papers[0]['items']), 3)
        html = mail.outbox[-1].alternatives[0][0]
        self.assertLess(html.index('Paper c3'), html.index('Paper c1'))  # the one with a citation sentence first
        self.assertNotIn('Paper c2', html)
        self.assertContains(self.client_for(user).get('/en/citations/'), 'Paper c2')

    def test_history_is_private_paginated_and_does_not_invent_old_alerts(self):
        from .models import CitationAlertHistory, CitationAlertLog
        now = timezone.now()
        user = self.user(paid=True)
        client = self.client_for(user)
        CitationAlertLog.objects.create(user=user, sent=now)
        page = client.get('/en/citations/')
        self.assertContains(page, 'No archived citation alert yet')
        self.assertContains(page, 'Last citation email sent:')
        self.assertEqual(page.context['last_unarchived_alert'], now)
        for day in range(11):
            CitationAlertHistory.objects.create(user=user, sent=now-datetime.timedelta(days=day), lang='en', count=1,
                papers=[{'title': 'Followed', 'arxiv_id': '2101.00001', 'items': [
                    {'title': 'History entry %d' % day, 'url': '/en/paper/2101.00002v1/',
                     'authors': [], 'note': '<script>privateTest()</script>'}]}])
        page = client.get('/en/citations/')
        self.assertContains(page, 'History entry 0')
        self.assertNotContains(page, 'History entry 10')
        self.assertContains(page, 'alerts_page=2')
        self.assertContains(page, '&lt;script&gt;privateTest()&lt;/script&gt;')
        self.assertContains(client.get('/en/citations/?alerts_page=2'), 'History entry 10')
        self.assertNotContains(self.client_for(self.user('other', paid=True)).get('/en/citations/'), 'History entry')
        self.assertNotContains(Client().get('/en/citations/'), 'History entry')
        UserPlan.objects.filter(user=user).update(paid=False)
        self.assertNotContains(client.get('/en/citations/'), 'History entry')

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
        self.assertIn('/en/paper/2201.00003/', html)
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
    def test_ads_names_are_turned_around_to_tell_self_citations(self):
        self.assertEqual(citations.ads_name('Kral, Q.'), 'Q. Kral')
        self.assertEqual(citations.ads_name('Planck Collaboration'), 'Planck Collaboration')
        self.assertTrue(citations.shares_author(citations.name_keys(['Quentin Kral']), [citations.ads_name('Kral, Q.')]))

    @override_settings(ADS_TOKEN='token')
    def test_papers_only_semantic_scholar_has_are_kept_apart(self):
        def ads(params):
            if params['q'].startswith('identifier:'):
                return {'numFound': 1, 'docs': [{'bibcode': '2021A&A...1K'}]}
            return {'numFound': 1, 'docs': [{'bibcode': '2026MNRAS.1A', 'title': ['Thick clouds'], 'author': ['Q, Zoe'],
                                             'identifier': ['arXiv:2201.00003'], 'citation_count': 12}]}
        s2 = self.citing('c3', 'c5')
        s2['c3'].update(influential=True, fields=['Physics'])
        s2['c4'] = dict(s2['c5'], arxiv_id='', title='Thick  Clouds')  # the same paper, told by its title
        with mock.patch('summarizer.citations.ads_get', side_effect=ads), \
             mock.patch('summarizer.citations.fetch_s2', return_value=s2):
            citing = citations.fetch('2101.00001')
        paper = citing['ads:2026MNRAS.1A']
        self.assertEqual((paper['authors'], paper['cited_by'], paper['influential'], paper['fields']),
                         (['Zoe Q'], 12, True, ['Physics']))
        self.assertEqual([u['id'] for u in citing.unmatched], ['c5'])
        user = self.user(paid=True)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', known='')
        with mock.patch('summarizer.citations.fetch', return_value=citing), \
             mock.patch('summarizer.library.lookup_many', return_value={}):
            citations.refresh('2101.00001', timezone.now())
        followed = FollowedPaper.objects.get(user=user)
        page = self.client_for(user).get('/en/citations/%d/' % followed.pk)
        self.assertContains(page, '1 citing paper known to Semantic Scholar only')
        self.assertContains(page, 'cited 12 times')
        self.assertContains(page, 'influential')

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

    def test_refresh_due_takes_the_never_then_the_oldest_checked_first(self):
        user = self.user(paid=True)
        now = timezone.now()
        for aid in ('2101.00001', '2101.00002', '2101.00003'):
            FollowedPaper.objects.create(user=user, arxiv_id=aid, known='[]')
        CitationCache.objects.create(arxiv_id='2101.00001', checked=now - datetime.timedelta(days=1))
        CitationCache.objects.create(arxiv_id='2101.00002', checked=now - datetime.timedelta(days=3))
        with mock.patch('summarizer.citations.fetch', return_value={}) as fetch, \
             mock.patch('summarizer.citations.time.sleep'), \
             mock.patch('summarizer.library.lookup_many', return_value={}):
            citations.refresh_due(now, limit=2)
            self.assertEqual([c.args[0] for c in fetch.call_args_list], ['2101.00003', '2101.00002'])
            fetch.reset_mock()
            # the next day the one left out comes first, instead of the same ones again
            citations.refresh_due(now + datetime.timedelta(days=1), limit=1)
            self.assertEqual(fetch.call_args_list[0].args[0], '2101.00001')

    HTML = '''<html><body><article>
      <div class="ltx_para"><p class="ltx_p">Hot planets have clouds. Thick clouds were first seen by Lee et al.
        <cite class="ltx_cite">(<a href="#bib.bib2" class="ltx_ref">2021</a>)</cite> with a large telescope, see Fig. 2.
        Other work <cite class="ltx_cite">[<a href="#bib.bib1" class="ltx_ref">1</a>]</cite> found none.</p></div>
      <div class="ltx_para"><p class="ltx_p">We find thinner clouds than <cite class="ltx_cite"><a href="#bib.bib2" class="ltx_ref">Lee et al. 2021</a></cite>, with <math alttext="T&gt;1000">x</math> K.</p></div>
      <ul class="ltx_biblist">
        <li id="bib.bib1" class="ltx_bibitem">Max P. 2019, Protein folding, Nature</li>
        <li id="bib.bib2" class="ltx_bibitem">Lee A., Ray B. 2021, Clouds on a hot planet seen with a large telescope, <a href="https://arxiv.org/abs/2101.00001">arXiv:2101.00001</a></li>
      </ul></article></body></html>'''

    def test_citation_sentences_are_read_from_the_arxiv_html(self):
        found = citations.find_contexts(self.HTML, '2101.00001', 'Clouds on a hot planet')
        self.assertEqual(found, ['Thick clouds were first seen by Lee et al. (2021) with a large telescope, see Fig. 2.',
                                 'We find thinner clouds than Lee et al. 2021, with T>1000 K.'])
        # by its title when the entry has no identifier, and nothing for a paper not in the bibliography
        by_title = self.HTML.replace('<a href="https://arxiv.org/abs/2101.00001">arXiv:2101.00001</a>', '')
        self.assertEqual(len(citations.find_contexts(by_title, '2101.00009', 'Clouds on a hot planet seen with a large telescope')), 2)
        self.assertEqual(citations.find_contexts(self.HTML, '2101.00009', 'Another paper entirely, on stars'), [])
        self.assertEqual(citations.find_contexts(self.HTML, '2101.0000', 'Short'), [])
        # by its journal DOI, and a citation in a table gives its row after the caption
        journal = self.HTML.replace('<a href="https://arxiv.org/abs/2101.00001">arXiv:2101.00001</a>', 'doi: 10.1093/MNRAS/X1')
        self.assertEqual(len(citations.find_contexts(journal, '2101.00001', 'Short', doi='10.1093/mnras/x1')), 2)
        table = self.HTML.replace('</article>', '''<figure class="ltx_table"><figcaption>Table 1: Cloud models. Second sentence.
          </figcaption><table><tr><th>Lee <cite class="ltx_cite"><a href="#bib.bib2" class="ltx_ref">2021</a></cite></th>
          <td>thick</td></tr></table></figure></article>''')
        self.assertIn('Table 1: Cloud models. — Lee 2021 | thick', citations.find_contexts(table, '2101.00001', 'Short'))

    def test_pieces_of_sentences_are_read_again_from_arxiv(self):
        from .models import CitationContext
        pieces = ['3 (Jiang et al., 2023); Gemma (Team, 2024).']
        self.assertTrue(citations.fragmentary(pieces))
        self.assertFalse(citations.fragmentary(['We use the model of [3] to fit the spectra of twelve planets.']))
        c = dict(self.citing('c1')['c1'], contexts=pieces)
        self.assertTrue(citations.needs_reading(c))
        self.assertEqual(citations.with_contexts('2101.00001', {'c1': c})['c1']['contexts'], pieces)
        CitationContext.objects.create(cited='2101.00001', citing='arxiv:2201.00001', checked=timezone.now(),
                                       contexts=json.dumps(['We fine-tune Mistral 7B (Jiang et al., 2023) on code.']))
        self.assertEqual(citations.with_contexts('2101.00001', {'c1': c})['c1']['context_source'], 'arxiv')

    def test_self_citations_count_the_first_authors_of_a_large_collaboration(self):
        team = ['Ann Lee%s%s' % (chr(97 + i // 26), chr(97 + i % 26)) for i in range(300)]
        self.assertFalse(citations.shares_author(citations.self_keys(team), ['A. Leekn']))
        self.assertTrue(citations.shares_author(citations.self_keys(team), ['A. Leeab']))

    def test_formulas_are_plain_text_when_they_can_be(self):
        self.assertEqual(citations.plain_math(r'\text{CO}{\vphantom{\text{X}}}_{\smash[t]{\text{2}}}'), 'CO₂')
        self.assertEqual(citations.plain_math(r'\sim'), '~')
        self.assertEqual(citations.plain_math(r'T_{\rm eq}^{2}'), r'$T_{\rm eq}^{2}$')

    def test_a_citing_paper_without_arxiv_html_is_read_again_a_month_later(self):
        from .models import CitationContext
        user = self.user(paid=True)
        now = timezone.now()
        self.queue(user, now, 'c1')
        CitationContext.objects.create(cited='2101.00001', citing='arxiv:2201.00001', contexts='[]', checked=now)
        with mock.patch('summarizer.citations.read_context') as read:
            self.assertEqual(citations.read_contexts(now), 0)
            self.assertEqual(citations.read_contexts(now + citations.RETRY + datetime.timedelta(days=1)), 1)

    @override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
    def test_missing_sentences_are_read_from_arxiv_judged_once_with_the_abstract(self):
        self.paper()
        now = timezone.now()
        user, other = self.user(paid=True), self.user('bob', paid=True)
        citing = self.citing('c1', 'c2')
        CitationCache.objects.create(arxiv_id='2101.00001', citing=json.dumps(citing), total=2, checked=now)
        for u in (user, other):
            FollowedPaper.objects.create(user=u, arxiv_id='2101.00001', title='Clouds on a hot planet', known='[]',
                                         pending='["c1", "c2"]')
        with mock.patch('summarizer.citations.fetch_html', side_effect=[self.HTML, '']) as html, \
             mock.patch('summarizer.citations.time.sleep'):
            self.assertEqual(citations.read_contexts(now), 2)
            self.assertEqual(citations.read_contexts(now), 0)  # tried once, for every follower
        self.assertEqual(html.call_args_list[0].args[0], '2201.00001')
        CitationCache.objects.filter(arxiv_id='2101.00001').update(abstract='We observe clouds on a hot planet.')
        answer = {'items': [{'id': 'c1', 'stance': 'questions', 'confidence': 'high', 'note': 'Finds thinner clouds.'}]}
        with mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            self.assertEqual(citations.send_alerts(now, 5), 2)
        chat.assert_called_once()  # judged once for both followers
        self.assertTrue(chat.call_args.kwargs['paid'])
        self.assertIn('We observe clouds on a hot planet.', chat.call_args.kwargs['system'])
        self.assertIn('We find thinner clouds than Lee et al. 2021', chat.call_args.kwargs['system'])
        html = mail.outbox[0].alternatives[0][0]
        self.assertIn('questions it', html)
        self.assertIn('Finds thinner clouds.', html)
        self.assertIn('No citation sentence available yet', html)  # c2: arXiv has no HTML for it
        self.assertIn('1 questions it', html)
        page = self.client_for(user).get('/en/citations/%d/' % FollowedPaper.objects.get(user=user).pk)
        self.assertContains(page, 'from the arXiv version')
        self.assertContains(page, 'Finds thinner clouds.')

    def test_self_citations_are_told_apart_and_listed_last(self):
        self.paper()  # by Ann Lee and Bob Ray
        user = self.user(paid=True)
        now = timezone.now()
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='Clouds', known='[]')
        many = ['Author %s Number' % chr(65 + n) for n in range(10)]
        citing = {'c1': dict(self.citing('c1')['c1'], authors=many + ['A. Lee']),
                  'c2': dict(self.citing('c2')['c2'], authors=['Zoe Quinn'], date='2023-01-05')}
        with mock.patch('summarizer.citations.fetch', return_value=citing):
            citations.refresh('2101.00001', now)
        cached = CitationCache.objects.get(arxiv_id='2101.00001')
        self.assertEqual(json.loads(cached.authors), ['Ann Lee', 'Bob Ray'])
        stored = json.loads(cached.citing)
        self.assertEqual((stored['c1']['self'], stored['c2']['self']), (True, False))
        self.assertEqual(len(stored['c1']['authors']), citations.SHOWN_AUTHORS)
        FollowedPaper.objects.filter(user=user).update(pending='["c1", "c2"]')
        context, _ = citations.pending(user, '', 'en', now)
        items = context['papers'][0]['items']
        self.assertEqual([i['id'] for i in items], ['c2', 'c1'])
        self.assertTrue(items[0]['late'])  # published in 2023, seen now
        self.assertEqual(context['papers'][0]['summary'], [('self', 1)])

    def test_a_paper_citing_several_followed_papers_is_shown_once(self):
        user = self.user(paid=True)
        now = timezone.now()
        for aid, title in (('2101.00001', 'First'), ('2101.00002', 'Second')):
            CitationCache.objects.create(arxiv_id=aid, citing=json.dumps(self.citing('c1')), total=1, checked=now)
            FollowedPaper.objects.create(user=user, arxiv_id=aid, title=title, known='[]', pending='["c1"]')
        context, _ = citations.pending(user, '', 'en', now)
        self.assertEqual(context['count'], 2)
        self.assertEqual(sum(len(p['items']) for p in context['papers']), 1)
        shown = next(i for p in context['papers'] for i in p['items'])
        self.assertEqual([a['title'] for a in shown['also']], ['First'])

    def test_rhythm_is_chosen_and_urgent_alerts_go_at_once(self):
        from .models import CitationAlertLog, CitationPreference
        now = timezone.now()
        user = self.user(paid=True)
        client = self.client_for(user)
        client.post('/en/citations/', {'action': 'prefs', 'every': '30'})
        self.assertEqual((user.citation_preference.every, user.citation_preference.urgent), (30, False))
        self.queue(user, now, 'c3')
        CitationAlertLog.objects.create(user=user, sent=now - datetime.timedelta(days=8))
        self.assertEqual(citations.due_users(now), [])
        CitationPreference.objects.filter(user=user).update(every=1)
        self.assertEqual(citations.due_users(now), [user.pk])
        CitationAlertLog.objects.filter(user=user).update(sent=now - datetime.timedelta(hours=23, minutes=30))
        self.assertEqual(citations.due_users(now), [user.pk])  # the hourly run does not push it to the next day
        # urgent: only a paper clearly questioning a followed one, and only when the user wants it
        CitationAlertLog.objects.filter(user=user).update(sent=now)
        questions = {'items': [{'id': 'c3', 'stance': 'questions', 'confidence': 'high', 'note': 'Disputes it.'}]}
        with mock.patch('summarizer.generated.chat_json', return_value=questions):
            self.assertEqual(citations.send_urgent(now, 5), 0)
            CitationPreference.objects.filter(user=user).update(urgent=True)
            with override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend'):
                self.assertEqual(citations.send_urgent(now, 5), 1)
        self.assertIn('A new paper questions one you follow', mail.outbox[-1].subject)
        self.assertIn('sent now rather than at its usual time', mail.outbox[-1].body)
        self.assertEqual(FollowedPaper.objects.get(user=user).pending, '')
        # it does not move the usual day
        CitationAlertLog.objects.filter(user=user).update(sent=now - datetime.timedelta(hours=5))
        FollowedPaper.objects.filter(user=user).update(pending='["c3"]')
        with mock.patch('summarizer.generated.chat_json', return_value=questions), \
             override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend'):
            citations.send_urgent(now, 5)
        self.assertEqual(CitationAlertLog.objects.get(user=user).sent, now - datetime.timedelta(hours=5))

    def test_mentions_are_not_urgent(self):
        now = timezone.now()
        user = self.user(paid=True)
        self.queue(user, now, 'c3')
        low = {'items': [{'id': 'c3', 'stance': 'questions', 'confidence': 'low', 'note': 'Maybe.'}]}
        with mock.patch('summarizer.generated.chat_json', return_value=low):
            self.assertEqual(citations.send_urgent(now, 5), 0)
        self.assertEqual(json.loads(FollowedPaper.objects.get(user=user).pending), ['c3'])

    def test_my_papers_are_followed_from_an_orcid_and_new_ones_as_they_appear(self):
        from .models import CitationAuthorWatch
        user = self.user(paid=True)
        client = self.client_for(user)
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', known='[]')
        self.assertContains(client.post('/en/citations/', {'action': 'import', 'author': 'not an id'}), 'neither an ORCID')
        with mock.patch('summarizer.citations.author_works', return_value=['2101.00001', 'doi_10.1--a']) as works, \
             mock.patch('summarizer.citations.fetch', return_value={}), \
             mock.patch('summarizer.citations.time.sleep'), \
             mock.patch('summarizer.library.lookup_many', return_value={}):
            page = client.post('/en/citations/', {'action': 'import', 'author': 'https://orcid.org/0000-0002-1825-009x'})
        works.assert_called_once_with('orcid:0000-0002-1825-009X')
        self.assertContains(page, '2 papers found.')
        self.assertEqual(sorted(FollowedPaper.objects.filter(user=user, mine=True).values_list('arxiv_id', flat=True)),
                         ['2101.00001', 'doi_10.1--a'])
        watch = CitationAuthorWatch.objects.get(user=user)
        CitationAuthorWatch.objects.filter(pk=watch.pk).update(checked=timezone.now() - datetime.timedelta(days=2))
        with mock.patch('summarizer.citations.author_works', return_value=['2610.00001', '2101.00001']):
            self.assertEqual(citations.watch_authors(timezone.now()), 1)
        self.assertTrue(FollowedPaper.objects.get(user=user, arxiv_id='2610.00001').mine)
        self.assertEqual(FollowedPaper.objects.get(user=user, arxiv_id='2610.00001').known, '')  # its first check comes
        UserPlan.objects.filter(user=user).update(paid=False)
        CitationAuthorWatch.objects.filter(pk=watch.pk).update(checked=None)
        with mock.patch('summarizer.citations.author_works') as works:
            citations.watch_authors(timezone.now())
        works.assert_not_called()

    def test_verdicts_are_judged_again_on_new_sentences_and_corrected_by_readers(self):
        from .models import CitationCorrection, CitationVerdict
        user = self.user(paid=True)
        now = timezone.now()
        citing = self.citing('c3', 'c4')
        citing['c4'].update(contexts=['Like [3], we see clouds.'], authors=['Ann Lee'])
        CitationCache.objects.create(arxiv_id='2101.00001', citing=json.dumps(citing), total=2, checked=now)
        followed = FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='Clouds', known='[]')
        answer = {'items': [{'id': 'c3', 'stance': 'questions', 'confidence': 'high', 'note': 'Disputes it.'},
                            {'id': 'c4', 'stance': 'supports', 'confidence': 'high', 'note': 'Agrees.'}]}
        with mock.patch('summarizer.generated.chat_json', return_value=answer) as chat:
            citations.stances('2101.00001', 'Clouds', citing, 'en')
            citations.stances('2101.00001', 'Clouds', citing, 'en')
            self.assertEqual(chat.call_count, 1)
            citing['c3']['contexts'] = ['A better sentence about [3].']
            citations.stances('2101.00001', 'Clouds', citing, 'en')
            self.assertIn('c3', chat.call_args.kwargs['system'])
            self.assertNotIn('c4', chat.call_args.kwargs['system'])
        # a follower's correction is theirs alone
        url = '/en/citations/%d/' % followed.pk
        client = self.client_for(user)
        client.post(url, {'correct': 'arxiv:2201.00003', 'stance_arxiv:2201.00003': 'builds_on'})
        self.assertEqual(CitationCorrection.objects.get().model_stance, 'questions')
        self.assertContains(client.get(url + '?stance=builds_on'), 'verdict corrected by you')
        self.assertEqual(CitationVerdict.objects.get(citing='arxiv:2201.00003').stance, 'questions')
        client.post(url, {'correct': 'arxiv:2201.00003', 'stance_arxiv:2201.00003': 'questions'})
        self.assertFalse(CitationCorrection.objects.exists())  # back to the model's verdict
        # an author's is everybody's, and the model does not judge it again
        FollowedPaper.objects.filter(pk=followed.pk).update(mine=True)
        client.post(url, {'correct': 'arxiv:2201.00003', 'stance_arxiv:2201.00003': 'builds_on'})
        other = self.user('bob', paid=True)
        theirs = FollowedPaper.objects.create(user=other, arxiv_id='2101.00001', title='Clouds', known='[]')
        self.assertContains(self.client_for(other).get('/en/citations/%d/?stance=builds_on' % theirs.pk),
                            'verdict corrected by an author')
        with mock.patch('summarizer.generated.chat_json') as chat:
            self.assertEqual(citations.stances('2101.00001', 'Clouds', citing, 'fr')['c3']['stance'], 'builds_on')
        self.assertNotIn('c3', chat.call_args.kwargs['system'])
        self.assertEqual(citations.who_cites(citations.all_items(followed, 'en')[1])[0], [])  # one paper each

    def test_analyses_are_limited_per_day(self):
        user = self.user(paid=True)
        CitationCache.objects.create(arxiv_id='2101.00001', citing='{}', checked=timezone.now())
        followed = FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='Clouds', known='[]')
        client = self.client_for(user)
        url = '/en/citations/%d/' % followed.pk
        for _ in range(citations.ANALYZE_PER_DAY):
            client.post(url, {'action': 'analyze'})
        r = client.post(url, {'action': 'analyze'})
        self.assertEqual(r['Location'], url + '?limit=analyze')
        self.assertContains(client.get(r['Location']), 'the most there can be in a day')

    def test_page_of_a_followed_paper_filters_exports_and_collects(self):
        from .models import CitationVerdict
        user = self.user(paid=True)
        now = timezone.now()
        citing = self.citing('c1', 'c3')
        citing['c1'].update(self=True, date='2025-03-01', doi='10.1/x')
        CitationCache.objects.create(arxiv_id='2101.00001', citing=json.dumps(citing), total=2, checked=now)
        followed = FollowedPaper.objects.create(user=user, arxiv_id='2101.00001', title='Clouds', known='[]')
        CitationVerdict.objects.create(cited='2101.00001', citing='arxiv:2201.00003', stance='supports',
                                       confidence='high', notes={'en': 'Confirms it.'}, basis=citations.basis(citing['c3']))
        client = self.client_for(user)
        url = '/en/citations/%d/' % followed.pk
        page = client.get(url)
        self.assertContains(page, '2 citations')
        self.assertContains(page, 'Confirms it.')
        self.assertContains(page, 'self-citation')
        self.assertEqual([y['count'] for y in page.context['years']], [1, 1])
        self.assertNotContains(client.get(url + '?self=hide'), 'Paper c1')
        self.assertNotContains(client.get(url + '?stance=supports'), 'Paper c1')
        self.assertContains(client.get(url + '?stance=none'), 'Paper c1')
        self.assertContains(client.get(url + '?year=2025'), 'Paper c1')
        csv_text = client.get(url + '?export=csv').content.decode('utf-8-sig')
        self.assertIn('Paper c3', csv_text)
        self.assertIn('supports', csv_text)
        self.assertIn('As shown by [3], clouds are thick.', csv_text)
        bib = client.get(url + '?export=bib&stance=supports').content.decode()
        self.assertIn('eprint={2201.00003}', bib)
        self.assertNotIn('Paper c1', bib)
        with mock.patch('summarizer.library.lookup_many', side_effect=lambda ids: {a: ('Title ' + a, 'A') for a in ids}):
            r = client.post(url, {'action': 'collect', 'ids': ['c1', 'c3'], 'collection': 'new', 'name': 'Cites me'})
        collection = Collection.objects.get(user=user)
        self.assertEqual(r['Location'], '/en/collections/%d/' % collection.pk)
        self.assertEqual(collection.name, 'Cites me')
        self.assertEqual(collection.papers.count(), 2)
        with mock.patch('summarizer.generated.chat_json', return_value={'items': []}) as chat:
            client.post(url, {'action': 'analyze'})
        chat.assert_not_called()  # c3 already judged, c1 has no sentence (arXiv HTML: none)
        self.assertEqual(self.client_for(self.user('eve', paid=True)).get(url).status_code, 404)
        self.assertContains(client.get('/fr/citations/%d/' % followed.pk), 'le confirme')


class ScoopTests(ToolsTestCase):
    DESCRIPTION = 'We measure the clouds of hot Jupiters with JWST transmission spectra and a new retrieval code. ' * 4

    def setUp(self):
        super().setUp()
        prior = mock.patch('summarizer.scoops.prior_candidates', return_value=({}, False))
        prior.start()
        self.addCleanup(prior.stop)

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


class ScoopEngineTests(ToolsTestCase):
    """The facets, the signals, the reader's verdicts, prior art, full-text comparisons and the weekly emails."""
    DESCRIPTION = ScoopTests.DESCRIPTION
    FACETS = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)  # the whole project, and its method

    def setUp(self):
        super().setUp()
        patch = mock.patch('summarizer.scoops.project_vectors', return_value=self.FACETS)
        patch.start()
        self.addCleanup(patch.stop)
        self.owner = self.user(paid=True)
        self.project = WatchedProject.objects.create(
            user=self.owner, title='Hot clouds', description=self.DESCRIPTION, last_day=datetime.date(2026, 10, 1),
            facets=json.dumps({'entities': ['WASP-107b']}), competitors='Rival Team')

    def paper(self, aid, vector, title='A paper', authors=('Zoe Q',), day=datetime.date(2026, 10, 5), abstract='Abstract.'):
        return aid, {'arxiv_id': aid, 'title': title, 'abstract': abstract, 'authors': list(authors), 'announced': day}, \
            np.array(vector, dtype=np.float32)

    def run_check(self, rows, verdict='partial', **extra):
        papers = {aid: p for aid, p, _ in rows}
        vectors = {aid: v for aid, _, v in rows}
        listings = []

        def judge(prompt, **kwargs):
            listings.append(kwargs['system'])
            ids = re.findall(r'<paper id="([^"]+)">', kwargs['system'])
            return {'items': [dict({'id': i, 'overlap': verdict, 'shared': 'S', 'original': 'O', 'action': 'cite',
                                    'aspects': {'question': 'same', 'method': 'close', 'data': 'bogus'}}, **extra) for i in ids]}
        with mock.patch('summarizer.generated.chat_json', side_effect=judge):
            scoops.check(self.project, sorted({p['announced'] for p in papers.values()}), papers, vectors)
        return listings

    def test_each_day_has_its_own_candidates(self):
        rows = [self.paper('d1-%d' % i, [1, 0, 0], day=datetime.date(2026, 10, 5)) for i in range(10)] + \
               [self.paper('d2-%d' % i, [1, 0, 0], day=datetime.date(2026, 10, 6)) for i in range(10)]
        self.run_check(rows)
        days = list(ScoopMatch.objects.values_list('announced', flat=True))
        self.assertEqual(days.count(datetime.date(2026, 10, 5)), scoops.CANDIDATES)
        self.assertEqual(days.count(datetime.date(2026, 10, 6)), scoops.CANDIDATES)
        self.assertEqual(json.loads(WatchedProject.objects.get().activity), {'2026-10': 20})

    def test_facets_names_and_competitors_bring_papers_in(self):
        rows = [self.paper('method', [0, 1, 0]),  # far from the whole project, close to its method
                self.paper('named', [0.4, 0, 0.92], title='The atmosphere of WASP-107b'),
                self.paper('rival', [0.35, 0, 0.94], authors=('R. Team',)),
                self.paper('far', [0, 0, 1], title='WASP-107b again', authors=('Rival Team',))]  # below the loose floor
        listing = ''.join(self.run_check(rows))
        self.assertNotIn('"far"', listing)
        matches = {m.arxiv_id: m for m in ScoopMatch.objects.all()}
        self.assertEqual(set(matches), {'method', 'named', 'rival'})
        self.assertEqual(matches['method'].signal_list(), ['similarity'])
        self.assertEqual(matches['named'].signal_list(), ['entity'])
        self.assertEqual(matches['rival'].signal_list(), ['competitor'])
        self.assertEqual(matches['method'].aspect_data(), {'question': 'same', 'method': 'close'})  # 'bogus' dropped
        self.assertEqual(matches['method'].action, 'cite')
        self.assertIsNotNone(matches['method'].vector)

    def test_papers_citing_the_key_references_are_read(self):
        self.project.references = '2101.00001\n10.1093/MNRAS/X1'
        self.project.save()
        rows = [self.paper('2610.00001v1', [0.36, 0, 0.93]), self.paper('2610.00002v1', [0.35, 0, 0.94])]
        answer = mock.Mock(status_code=200)
        answer.json.return_value = [{'references': [{'externalIds': {'ArXiv': '2101.00001'}}, {'externalIds': {'DOI': '10.1093/mnras/x1'}}]},
                                    {'references': [{'externalIds': {'ArXiv': '2101.00001'}}]}]
        with mock.patch('summarizer.scoops.requests.post', return_value=answer) as post:
            self.run_check(rows)
        self.assertEqual(post.call_args[1]['json'], {'ids': ['arXiv:2610.00001', 'arXiv:2610.00002']})
        self.assertEqual(list(ScoopMatch.objects.values_list('arxiv_id', 'signals')), [('2610.00001v1', 'references')])

    def test_a_paper_marked_not_relevant_pushes_similar_ones_down(self):
        self.run_check([self.paper('first', [0.6, 0, 0.8])])
        match = ScoopMatch.objects.get()
        client = self.client_for(self.owner)
        r = client.post('/en/scoops/', {'action': 'mark', 'pk': self.project.pk, 'match': match.pk, 'status': 'irrelevant'})
        self.assertEqual(r['Location'], '/en/scoops/#match-%d' % match.pk)
        self.assertEqual(ScoopMatch.objects.get().status, 'irrelevant')
        self.assertAlmostEqual(scoops.Scorer(self.project)(np.array([0.6, 0, 0.8], dtype=np.float32)), 0.2, places=5)
        listing = self.run_check([self.paper('twin', [0.6, 0, 0.8])])
        self.assertEqual(listing, [])
        page = client.get('/en/scoops/')
        self.assertContains(page, '1 paper marked not relevant')
        # someone else's project or paper cannot be touched
        other = self.client_for(self.user('bob', paid=True))
        other.post('/en/scoops/', {'action': 'mark', 'pk': self.project.pk, 'match': match.pk, 'status': 'relevant'})
        self.assertEqual(ScoopMatch.objects.get().status, 'irrelevant')

    def test_editing_keeps_the_papers_and_searches_prior_art_again(self):
        ScoopMatch.objects.create(project=self.project, arxiv_id='x', title='Kept', announced=datetime.date(2026, 10, 2),
                                  similarity=0.8, overlap='high', emailed=True)
        client = self.client_for(self.owner)
        with mock.patch('summarizer.scoops.refresh') as refresh:
            client.post('/en/scoops/', {'action': 'edit', 'pk': self.project.pk, 'title': 'Hot clouds 2',
                                        'description': self.DESCRIPTION + ' More.', 'competitors': 'A B; C D',
                                        'references': '2101.00001, 2101.00002', 'notify': 'strong'})
        project = WatchedProject.objects.get()
        refresh.assert_called_once_with(project.pk)
        self.assertEqual((project.title, project.notify, project.prior_status), ('Hot clouds 2', 'strong', 'pending'))
        self.assertEqual(project.competitor_list(), ['A B', 'C D'])
        self.assertEqual(project.reference_list(), ['2101.00001', '2101.00002'])
        self.assertTrue(ScoopMatch.objects.filter(title='Kept').exists())

    def test_prior_art_is_judged_shown_and_its_journals_followed(self):
        since = datetime.date(2025, 1, 1)
        found = {'2401.00001': {'arxiv_id': '2401.00001v2', 'title': 'Old clouds', 'abstract': 'A.', 'authors': ['Old Author'],
                                'announced': since, 'subfield': ''},
                 'doi_10.1/own': {'arxiv_id': 'doi_10.1/own', 'title': 'Mine', 'abstract': 'B.', 'authors': ['Ann Lee'],
                                  'announced': since, 'subfield': 'journals.3103'}}
        for n in range(3):
            found['doi_10.1/j%d' % n] = {'arxiv_id': 'doi_10.1/j%d' % n, 'title': 'Journal %d' % n, 'abstract': 'C.',
                                         'authors': ['Old Author'], 'announced': since, 'subfield': 'journals.3103'}
        self.project.author_name = 'Ann Lee'
        self.project.save()
        found['doi_10.1/coauthor'] = dict(found['doi_10.1/own'], arxiv_id='doi_10.1/coauthor', authors=['Bo Chen'])
        self.project.author_name = 'Ann Lee\nBo Chen'
        self.project.save()
        with mock.patch('summarizer.scoops.prior_candidates', return_value=(found, True)), \
             mock.patch('summarizer.similar.embed', side_effect=lambda texts: np.array([[1, 0, 0]] * len(texts), dtype=np.float32)), \
             mock.patch('summarizer.generated.chat_json', side_effect=lambda prompt, **kw: {'items': [
                 {'id': i, 'overlap': 'high'} for i in re.findall(r'<paper id="([^"]+)">', kw['system'])]}):
            scoops.prior_art(self.project)
        project = WatchedProject.objects.get()
        self.assertEqual((project.prior_status, project.prior_found, project.subfields), ('done', 4, 'journals.3103'))
        self.assertTrue(project.prior_limited)
        prior = ScoopMatch.objects.filter(kind='prior')
        self.assertEqual(prior.count(), 4)  # the user's own paper is left out
        self.assertTrue(all(m.emailed for m in prior))
        self.assertIn('journals.3103', digest.followed_journals())
        page = self.client_for(self.owner).get('/en/scoops/')
        self.assertContains(page, 'Already published')
        self.assertContains(page, 'Old clouds')
        self.assertContains(page, '+ Old Author')  # signs several of the results
        self.assertContains(page, 'The journals could not be searched this time')
        UserPlan.objects.filter(user=self.owner).update(paid=False)
        self.assertNotIn('journals.3103', digest.followed_journals())

    def test_strong_overlaps_are_compared_with_the_full_text(self):
        match = ScoopMatch.objects.create(project=self.project, arxiv_id='2610.00001v1', title='Rival', overlap='high',
                                          announced=datetime.date(2026, 10, 5), similarity=0.8)
        answer = {'aspects': [{'aspect': 'method', 'paper': 'Same code', 'project': 'New code', 'level': 'close'},
                              {'aspect': 'mood', 'paper': '?'}], 'verdict': 'It does not take your result.', 'advice': 'Stress the code.'}
        with mock.patch('summarizer.scoops.paper_full_text', return_value='x' * 5000) as text, \
             mock.patch('summarizer.generated.chat_json', return_value=answer):
            scoops.check_due(timezone.now())
        text.assert_called_once_with('2610.00001v1')
        match.refresh_from_db()
        self.assertEqual(match.deep_status, 'done')
        self.assertEqual([a['aspect'] for a in match.deep_data()['aspects']], ['method'])
        scoops.send_alerts(timezone.now(), 5)
        self.assertIn('It does not take your result.', mail.outbox[0].body)
        self.assertIn('/en/scoops/#match-%d' % match.pk, mail.outbox[0].body)
        self.assertContains(self.client_for(self.owner).get('/en/scoops/'), 'Stress the code.')
        # a closed license: no full-text comparison
        other = ScoopMatch.objects.create(project=self.project, arxiv_id='2610.00002v1', title='Closed', overlap='partial',
                                          announced=datetime.date(2026, 10, 5), similarity=0.7)
        with mock.patch('summarizer.scoops.paper_full_text', return_value=''):
            self.client_for(self.owner).post('/en/scoops/', {'action': 'deep', 'pk': self.project.pk, 'match': other.pk})
        other.refresh_from_db()
        self.assertEqual(other.deep_status, 'unavailable')

    def test_partial_overlaps_wait_for_the_weekly_email(self):
        self.project.notify = 'strong'
        self.project.weekly_sent = timezone.now() - datetime.timedelta(days=2)
        self.project.save()
        ScoopMatch.objects.create(project=self.project, arxiv_id='p', title='Partial one', overlap='partial',
                                  announced=datetime.date(2026, 10, 5), similarity=0.6, deep_status='done')
        self.assertEqual(scoops.send_alerts(timezone.now(), 5), 0)
        ScoopMatch.objects.create(project=self.project, arxiv_id='h', title='Strong one', overlap='high',
                                  announced=datetime.date(2026, 10, 5), similarity=0.9, deep_status='done')
        self.assertEqual(scoops.send_alerts(timezone.now(), 5), 1)
        self.assertIn('Strong one', mail.outbox[0].body)
        self.assertNotIn('Partial one', mail.outbox[0].body)
        later = timezone.now() + datetime.timedelta(days=6)
        self.assertEqual(scoops.send_alerts(later, 5), 1)
        self.assertIn('Partial one', mail.outbox[1].body)
        self.assertEqual(WatchedProject.objects.get().weekly_sent, later)

    def test_actions_say_what_they_did(self):
        client = self.client_for(self.owner)
        ajax = {'HTTP_X_REQUESTED_WITH': 'fetch'}
        r = client.post('/en/scoops/', {'action': 'competitor', 'pk': self.project.pk, 'name': 'Jo Park'}, **ajax)
        self.assertIn('Jo Park is now followed', r.json()['message'])
        self.assertEqual(WatchedProject.objects.get().competitor_list(), ['Rival Team', 'Jo Park'])
        r = client.post('/en/scoops/', {'action': 'uncompetitor', 'pk': self.project.pk, 'name': 'Rival Team'}, **ajax)
        self.assertEqual(r.json()['message'], 'Rival Team is no longer followed.')
        with mock.patch('summarizer.scoops.search_prior') as search:
            r = client.post('/en/scoops/', {'action': 'prior', 'pk': self.project.pk}, **ajax)
        search.assert_called_once_with(self.project.pk)
        self.assertIn('Search for prior art started', r.json()['message'])
        r = client.post('/en/scoops/', {'action': 'prior', 'pk': self.project.pk}, **ajax)
        self.assertIn('already running', r.json()['message'])
        # without the page script, the message is shown in the project after the redirect
        r = client.post('/en/scoops/', {'action': 'competitor', 'pk': self.project.pk, 'name': 'Al Ng'})
        page = client.get(r['Location'])
        self.assertContains(page, 'Al Ng is now followed')
        self.assertContains(page, 'Al Ng ×')
        self.assertNotContains(client.get('/en/scoops/'), 'Al Ng is now followed')  # shown once

    def test_a_paper_goes_to_a_collection(self):
        match = ScoopMatch.objects.create(project=self.project, arxiv_id='2610.00001v1', title='Rival', overlap='high',
                                          announced=datetime.date(2026, 10, 5), similarity=0.8)
        client = self.client_for(self.owner)
        with mock.patch('summarizer.library.add_many', return_value={'added': 1}) as add:
            r = client.post('/en/scoops/', {'action': 'collect', 'pk': self.project.pk, 'match': match.pk, 'collection': 'new'})
        collection = Collection.objects.get()
        self.assertEqual(collection.name, 'Hot clouds')
        self.assertEqual(add.call_args[0][:2], (collection, ['2610.00001v1']))
        self.assertContains(client.get(r['Location']), 'Added to')


class ScoopSourceTests(ScoopEngineTests):
    """OpenReview, the papers citing the key references, open journal versions and the kept drafts."""

    def note(self, nid, title, days_ago, venue='ICLR.cc/2027/Conference/Submission'):
        created = timezone.now() - datetime.timedelta(days=days_ago)
        return {'id': nid, 'cdate': created.timestamp() * 1000, 'content': {
            'title': {'value': title}, 'abstract': {'value': title + ' abstract.'}, 'venueid': {'value': venue},
            'authors': {'value': ['Zoe Q', 'Ann Lee']}}}

    def test_openreview_notes_keep_new_submissions_only(self):
        answer = mock.Mock(status_code=200)
        answer.json.return_value = {'notes': [self.note('new1', 'Fresh  idea', 1), self.note('old', 'Old idea', 40),
                                              self.note('copy', 'Journal copy', 1, venue='OpenReview.net/Public_Article')]}
        with mock.patch('summarizer.scoops.requests.get', return_value=answer):
            notes = scoops.openreview_notes('clouds', timezone.now().date() - datetime.timedelta(days=10))
        self.assertEqual([(n['arxiv_id'], n['title'], n['authors']) for n in notes], [('openreview_new1', 'Fresh idea', ['Zoe Q', 'Ann Lee'])])

    def test_openreview_is_searched_once_a_day_and_its_first_results_are_not_emailed(self):
        self.project.facets = json.dumps({'queries': ['hot clouds']})
        self.project.save()
        notes = [{'arxiv_id': 'openreview_a', 'title': 'A', 'abstract': 'A.', 'authors': ['Zoe Q'],
                  'announced': timezone.now().date(), 'subfield': ''}]
        with mock.patch('summarizer.scoops.openreview_notes', return_value=notes) as search, \
             mock.patch('summarizer.similar.embed', return_value=np.array([[1, 0, 0]], dtype=np.float32)), \
             mock.patch('summarizer.generated.chat_json', return_value={'items': [{'id': 'openreview_a', 'overlap': 'high'}]}), \
             mock.patch('summarizer.scoops.paper_full_text', return_value=''):
            scoops.check_due(timezone.now())
            scoops.check_due(timezone.now())
        search.assert_called_once()
        match = ScoopMatch.objects.get()
        self.assertTrue(match.emailed)  # the first search, over the last days
        self.assertEqual(match.link(), 'https://openreview.net/forum?id=a')
        self.assertEqual(scoops.base_id('openreview_abcv2'), 'openreview_abcv2')
        page = self.client_for(self.owner).get('/en/scoops/')
        self.assertContains(page, 'https://openreview.net/forum?id=a')

    def test_prior_art_reads_the_papers_citing_the_key_references(self):
        self.project.references = '2101.00001\n10.1093/key'
        self.project.save()
        work = lambda wid, refs: {'id': 'https://openalex.org/' + wid, 'doi': 'https://doi.org/10.1093/' + wid.lower(),
                                  'display_name': 'Work ' + wid, 'publication_date': '2025-05-01',
                                  'abstract_inverted_index': {'Clouds.': [0]}, 'authorships': [],
                                  'referenced_works': ['https://openalex.org/' + r for r in refs]}

        def openalex(url, params=None, missing=(404,)):
            if '/works/doi:' in url:
                return {'id': 'https://openalex.org/' + ('WA' if 'arxiv' in url else 'WB')}
            return {'results': [work('W1', ['WA', 'WB']), work('W2', ['WA'])]}
        found = {}
        with mock.patch('summarizer.sources._json', side_effect=openalex):
            self.assertTrue(scoops.citing_works(self.project, datetime.date(2024, 1, 1), found))
        self.assertEqual({k: p.get('cites') for k, p in found.items()}, {sources.doi_key('10.1093/w1'): True})

    def test_a_closed_preprint_is_read_through_its_open_journal_version(self):
        with mock.patch('summarizer.sources.arxiv_text', return_value=('', '')), \
             mock.patch('summarizer.digest.fetch_ids', return_value=[{'doi': '10.1093/mnras/x1'}]), \
             mock.patch('summarizer.sources.metadata', return_value={'license': 'https://creativecommons.org/licenses/by/4.0/'}), \
             mock.patch('summarizer.sources.full_text', return_value=('Full journal text', 'pdf')) as full:
            self.assertEqual(scoops.paper_full_text('2610.00001v1'), 'Full journal text')
        full.assert_called_once_with(sources.doi_key('10.1093/mnras/x1'))

    def test_a_kept_draft_is_encrypted_and_used_for_the_comparison(self):
        sealed = scoops.seal('My secret draft')
        self.assertNotIn('secret', sealed)
        self.assertEqual(scoops.unseal(sealed), 'My secret draft')
        self.assertEqual(scoops.unseal('keep'), '')
        self.project.draft_text = sealed
        self.project.save()
        match = ScoopMatch.objects.create(project=self.project, arxiv_id='2610.00001v1', title='Rival', overlap='high',
                                          announced=datetime.date(2026, 10, 5), similarity=0.8)
        with mock.patch('summarizer.scoops.paper_full_text', return_value='x' * 5000), \
             mock.patch('summarizer.generated.chat_json', return_value={'aspects': [], 'verdict': 'V', 'advice': 'A'}) as chat:
            scoops.deepen(match)
        self.assertIn('My secret draft', chat.call_args[1]['system'])

    def test_calibration_reports_the_readers_verdicts(self):
        from io import StringIO
        from django.core.management import call_command
        ScoopMatch.objects.create(project=self.project, arxiv_id='a', title='A', overlap='high', status='irrelevant',
                                  announced=datetime.date(2026, 10, 5), similarity=0.47, signals='entity')
        out = StringIO()
        call_command('scoop_calibrate', '--feedback', stdout=out)
        self.assertIn('0.45-0.50: relevant 0, not relevant 1', out.getvalue())
        self.assertIn('entity: relevant 0, not relevant 1', out.getvalue())


class ResultShareTests(ToolsTestCase):
    RESULTS = {  # path: kind, a finished result, a text the shared page shows
        'missed': ('missed', {'title': 'Clouds draft', 'searched': 3, 'cited': 2, 'items': [
            {'arxiv_id': '2101.00001', 'title': 'Thick clouds', 'authors': 'Ann Lee', 'importance': 'high',
             'similarity': 80, 'relation': 'Measures the same clouds.'}]}, 'Thick clouds'),
        'conference': ('conference', {'name': 'AAS 247', 'listed': 40, 'people': ['Bob Ray'], 'talks': [
            {'title': 'Cloudy hot Jupiters', 'speakers': 'Ann Lee', 'priority': 'must', 'why': 'Your topic.'}]},
            'Cloudy hot Jupiters'),
        'teams': ('teams', {'title': 'Exoplanet clouds', 'groups': [{'institution': 'Lab of clouds', 'papers': 2,
                                                                     'citations': 5, 'titles': []}]}, 'Lab of clouds'),
    }

    def test_each_result_is_shared_read_only_until_stopped(self):
        user = self.user('pro', paid=True)
        owner = self.client_for(user)
        for path, (kind, data, shown) in self.RESULTS.items():
            row = Generated.objects.create(kind=kind, key='k-' + kind, user=user, status='done', content=json.dumps(data))
            mine = '/en/%s/k-%s/' % (path, kind)
            self.assertContains(owner.get(mine), 'Create a link')
            self.assertEqual(owner.post(mine, {'action': 'share'})['Location'], mine)
            row.refresh_from_db()
            link = '/en/%s/shared/%s/' % (path, row.share_token)
            self.assertContains(owner.get(mine), link)
            page = Client().get(link)
            self.assertContains(page, shown)
            self.assertNotContains(page, 'Create a link')
            self.assertNotContains(page, mine)
            owner.post(mine, {'action': 'unshare'})
            self.assertEqual(Client().get(link).status_code, 404)
        pending = Generated.objects.create(kind='teams', key='k-pending', user=user, status='pending')
        owner.post('/en/teams/k-pending/', {'action': 'share'})  # nothing to share before it is done
        pending.refresh_from_db()
        self.assertEqual(pending.share_token, '')


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
             mock.patch('summarizer.bibliography.Resolver.resolve', return_value=None), \
             mock.patch('summarizer.refcheck.crossref_fields', return_value={'journal': 'MNRAS & Co', 'year': '2021', 'doi': '10.1093/mnras/x1'}), \
             mock.patch('summarizer.generated.chat_json', return_value=verdicts) as chat:
            result = refcheck.check(self.BIB, self.TEX, 'en')
        kinds = {p['key']: [i['kind'] for i in p['issues']] for p in result['problems']}
        self.assertEqual(kinds, {'retracted': ['unknown', 'retracted'], 'pre': ['published', 'old_version'],
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
        self.assertContains(client.get(r['Location']), 'No discrepancy detected')
        self.assertEqual(client.get(r['Location'] + 'bib/').content, b'@article{a,\n}')
        self.assertEqual(self.client_for(self.user('other', paid=True)).get(r['Location']).status_code, 404)

    def test_a_report_is_shared_with_its_corrected_bib_until_it_expires(self):
        owner = self.client_for(self.user('pro', paid=True))
        row = Generated.objects.create(kind='refcheck', key='k1', user=User.objects.get(username='pro'), status='done',
                                       content=json.dumps({'entries': 1, 'problems': [], 'claims': [], 'claims_checked': 0,
                                                           'with_tex': True, 'counts': {'error': 0, 'warning': 0},
                                                           'fixed_bib': '@article{a,\n}', 'corrected': 1}))
        self.assertContains(owner.get('/en/refcheck/k1/'), 'quotes the sentences of your manuscript')
        owner.post('/en/refcheck/k1/', {'action': 'share'})
        row.refresh_from_db()
        link = '/en/refcheck/shared/%s/' % row.share_token
        page = Client().get(link)
        self.assertContains(page, 'Shared bibliography check')
        self.assertContains(page, link + 'bib/')
        self.assertNotContains(page, '/refcheck/k1/')
        self.assertEqual(Client().get(link + 'bib/').content, b'@article{a,\n}')
        Generated.objects.filter(pk=row.pk).update(share_expires=timezone.now() - datetime.timedelta(days=1))
        self.assertEqual(Client().get(link).status_code, 404)
        self.assertEqual(Client().get(link + 'bib/').status_code, 404)
        self.assertNotContains(owner.get('/en/refcheck/k1/'), link)


class ChangesTests(ToolsTestCase):
    def saved(self, old, new, lang='en', full=True):
        return Generated.objects.create(kind='changes', key=changes.comparison_key('2101.00001', old, new),
            lang=lang, status='done', content=json.dumps({'old': old, 'new': new, 'full_text': full,
            'results_changed': False, 'verdict': 'Saved %s v%d-v%d' % (lang, old, new), 'changes': []}))

    def test_a_slow_arxiv_cannot_hold_the_page_past_heroku_limit(self):
        import requests
        from summarizer import digest
        self.paper()
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.digest.requests.get', side_effect=requests.Timeout) as get, \
             mock.patch('summarizer.digest.time.sleep') as sleep:
            page = client.get('/en/changes/2101.00001v1/')
        self.assertEqual(page.status_code, 200)  # the versions we know, without arXiv's answer
        self.assertEqual(get.call_count, 1)
        self.assertEqual(get.call_args.kwargs['timeout'], 8)
        self.assertFalse([c for c in sleep.call_args_list if c.args[0] > digest.SECONDS_BETWEEN_CALLS])

    def test_history_keeps_each_pair_and_reopens_last_comparison(self):
        self.paper()
        client = self.client_for(self.user(paid=True))
        first = self.saved(1, 2)
        second = self.saved(2, 3)
        self.saved(1, 3, lang='fr')
        with mock.patch('summarizer.changes.latest_version', return_value=3), \
             mock.patch('summarizer.changes.compare') as compare:
            page = client.get('/en/changes/2101.00001v1/')
            self.assertContains(page, 'Saved en v2-v3')
            self.assertContains(page, '?from=1&amp;to=2')
            self.assertNotContains(page, 'Saved fr')
            self.assertEqual(len(page.context['history']), 2)
            self.assertContains(client.get('/en/changes/2101.00001v1/?from=1&to=2'), 'Saved en v1-v2')
            response = client.post('/en/changes/2101.00001v1/start/', {'from': 1, 'to': 2})
            self.assertEqual(response.json()['id'], first.pk)
            self.assertEqual(Generated.objects.filter(pk__in=[first.pk, second.pk], status='done').count(), 2)
            compare.assert_not_called()

    def test_selected_pair_controls_start_text_and_pending_resume(self):
        self.paper()
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.changes.latest_version', return_value=3):
            page = client.get('/en/changes/2101.00001v1/?from=2&to=3')
            self.assertContains(page, 'Compare v2 with v3')
            self.assertContains(page, 'changes.js')
            row = Generated.objects.create(kind='changes', key=changes.comparison_key('2101.00001', 2, 3), lang='en')
            page = client.get('/en/changes/2101.00001v1/')
            self.assertEqual((page.context['old'], page.context['new']), (2, 3))
            self.assertContains(page, 'follow(%d)' % row.pk)

    def test_restricted_license_warning_precedes_generation_and_partial_badge_is_honest(self):
        paper = self.paper()
        paper.license = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
        paper.save()
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.changes.latest_version', return_value=2), \
             mock.patch('summarizer.changes.requests.get') as get:
            page = client.get('/en/changes/2101.00001v1/')
            self.assertContains(page, 'Only the titles, abstracts')
            self.assertLess(page.content.index(b'Only the titles'), page.content.index(b'id="changes-go"'))
            self.saved(1, 2, full=False)
            page = client.get('/en/changes/2101.00001v1/')
            self.assertContains(page, 'Limited comparison: abstracts and comments only')
            self.assertNotContains(page, 'Same main results')
            get.assert_not_called()

    def test_unknown_license_preview_does_not_query_oai(self):
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.changes.latest_version', return_value=2), \
             mock.patch('summarizer.changes.requests.get') as get:
            self.assertContains(client.get('/en/changes/2101.00001v1/'), 'The license has not been verified yet')
            get.assert_not_called()

    def test_invalid_pair_cannot_start_generation(self):
        client = self.client_for(self.user(paid=True))
        with mock.patch('summarizer.changes.latest_version', return_value=3), \
             mock.patch('summarizer.generated.start') as start:
            for old, new in [(2, 2), (3, 2), (0, 3), (1, -1), (1, 9), ('x', 3)]:
                self.assertEqual(client.post('/en/changes/2101.00001v1/start/', {'from': old, 'to': new}).status_code, 400)
            start.assert_not_called()

    def test_missing_html_is_distinguished_from_license_restriction(self):
        self.paper()
        with mock.patch('summarizer.changes.version_record', return_value=self.record(1, 'Abstract')), \
             mock.patch('summarizer.changes.version_html', side_effect=[['Text'], []]), \
             mock.patch('summarizer.generated.chat_json', return_value={'verdict': 'Limited', 'changes': []}):
            result = changes.compare('2101.00001', 1, 2, 'en')
        self.assertFalse(result['full_text'])
        self.assertEqual(result['source_scope'], 'abstract_unavailable')

    def test_partial_comparison_email_does_not_claim_unchanged_results(self):
        from django.template.loader import render_to_string
        item = {'title': 'Example', 'old': 1, 'new': 2, 'full_text': False, 'results_changed': False,
                'verdict': 'Only the abstract was read.', 'changes': [], 'url': '/en/changes/example/'}
        with translation.override('en'):
            html = render_to_string('changes/email.html', {'items': [item], 'count': 1})
            text = render_to_string('changes/email.txt', {'items': [item], 'count': 1})
        for content in (html, text):
            self.assertIn('Limited comparison: abstracts and comments only', content)
            self.assertNotIn('Same main results', content)
        self.assertIn('border-left:3px solid #ddd;', html)

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
                    {'author': {'id': 'https://openalex.org/A%d' % (sum(map(ord, teams.person_name(name))) % 99991),
                                'display_name': name}, 'author_position': position,
                     'institutions': [{'id': 'I-' + inst, 'display_name': inst, 'country_code': 'FR'}]}
                    for name, inst, position in authors]}

    def test_groups_by_institution_and_counts_who_signs_what(self):
        found = [self.work(1, 'Exomoons by astrometry', [('Ann Lee', 'Paris', 'first'), ('Bob Ray', 'Paris', 'last')], 10),
                 self.work(2, 'Exomoon transits', [('Lee, Ann', 'Paris', 'first'), ('Zoe Q', 'Leiden', 'last')], 3),
                 self.work(3, 'Protein folding', [('Max P', 'Basel', 'first')], 99),
                 self.work(4, 'Huge review of exomoons', [('Big %d' % i, 'Inst%d' % i, 'middle') for i in range(30)] +
                           [('Zoe Q', 'Leiden', 'last')])]
        plan = {'title': 'Exomoons', 'queries': ['exomoon astrometry', 'exomoon transits']}
        notes = {'approaches': [{'id': '0', 'name': 'Exomoon searches', 'description': 'Astrometry.'}], 'landscape': 'Two groups.'}

        def embed(texts):
            return np.array([[1.0, 0.0]] + [[0.0, 1.0] if 'Protein' in t else [1.0, 0.0] for t in texts[1:]], dtype=np.float32)
        with mock.patch('summarizer.teams.works', return_value=found) as works, \
             mock.patch('summarizer.similar.embed', side_effect=embed), \
             mock.patch('summarizer.generated.chat_json', side_effect=[plan, {'grades': {'P1': 2, 'P2': 1, 'P3': 1}}, notes]):
            result = teams.find('exomoons', 'en')
        self.assertEqual(works.call_count, 2)
        self.assertEqual(result['papers'], 3)  # protein folding is about something else
        self.assertEqual([g['institution'] for g in result['institutions']][:2], ['Paris', 'Leiden'])
        paris = result['institutions'][0]
        self.assertEqual((paris['papers'], paris['share'], paris['lead']), (2, 1.5, 2))
        self.assertEqual(result['people'][0]['name'], 'Ann Lee')  # 'Lee, Ann' is the same person
        self.assertEqual(result['people'][0]['papers'], 2)
        self.assertEqual(result['approaches'][0]['name'], 'Exomoon searches')
        self.assertEqual(result['landscape'], 'Two groups.')
        self.assertIn('Inst3', [g['institution'] for g in teams.institutions_of([found[3]], {})])  # affiliations kept
        self.assertEqual(result['version'], teams.RESULT_VERSION)

    def test_page_is_premium_and_starts_from_a_project(self):
        self.assertContains(self.client_for(self.user()).get('/en/teams/'), 'Discover Premium')
        user = self.user('pro', paid=True)
        project = WatchedProject.objects.create(user=user, title='Moons', description='x' * 300, search_text='Exomoons by astrometry.')
        client = self.client_for(user)
        self.assertContains(client.post('/en/teams/', {'topic': 'short'}), 'a few words at least')
        with mock.patch('summarizer.teams.find', return_value={'title': 'Moons', 'papers': 0, 'version': teams.RESULT_VERSION}) as find:
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
    def test_detached_pdf_accents_preserve_names_titles_and_scientific_notation(self):
        cases = {
            'Th´eo Vrignaud': 'Théo Vrignaud',
            'La mission spatiale PLATO: d´emographie et ´evolution des syst`emes plan´etaires':
                'La mission spatiale PLATO: démographie et évolution des systèmes planétaires',
            'The\u0301o Vrignaud': 'Théo Vrignaud',
            "O'Neil · 12:30–12:45 · x^2 · 30′ · α · `name`": "O'Neil · 12:30–12:45 · x^2 · 30′ · α · `name`",
        }
        for before, after in cases.items():
            with self.subTest(before=before):
                self.assertEqual(conference.normalize_accents(before), after)
                self.assertEqual(conference.normalize_accents(after), after)

    def test_model_listing_is_normalized_before_ranking(self):
        with mock.patch('summarizer.generated.chat_json', return_value={'talks': [
                {'title': 'Syst`emes plan´etaires', 'speakers': 'Th´eo Vrignaud'}]}):
            talks = conference.list_talks('Program')
        self.assertEqual(talks[0]['title'], 'Systèmes planétaires')
        self.assertEqual(talks[0]['speakers'], 'Théo Vrignaud')

    def test_cached_results_display_correct_accents_without_regeneration(self):
        user = self.user(paid=True)
        raw = {'name': 'Journ´ees scientifiques', 'listed': 1, 'people': ['Th´eo Vrignaud'],
               'talks': [{'title': 'Syst`emes plan´etaires', 'speakers': 'Th´eo Vrignaud',
                         'priority': 'must', 'why': 'M´ethodes proches.', 'time': '12:30–12:45'}]}
        row = Generated.objects.create(kind='conference', key='accent-test', user=user, lang='fr',
                                       status='done', content=json.dumps(raw))
        with mock.patch('summarizer.generated.chat_json') as chat:
            page = self.client_for(user).get('/fr/conference/accent-test/')
        self.assertContains(page, 'Théo Vrignaud', count=2)
        self.assertContains(page, 'Systèmes planétaires')
        self.assertContains(page, 'Journées scientifiques')
        self.assertContains(page, 'Méthodes proches.')
        chat.assert_not_called()
        row.refresh_from_db()
        self.assertEqual(row.data(), raw)

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
    def questions_fixture(self):
        user = self.user(paid=True)
        collection = Collection.objects.create(user=user, name='Questions test')
        for aid in ('2101.00001v1', '2101.00002v1'):
            self.paper(aid)
            CollectionPaper.objects.create(collection=collection, arxiv_id=aid, title='Example', authors='Ann Lee')
        return self.client_for(user), collection

    def test_no_supported_question_is_saved_as_an_honest_empty_result(self):
        client, collection = self.questions_fixture()
        url = '/en/collections/%d/questions/' % collection.pk
        with mock.patch('summarizer.questions.paper_block', return_value='Abstract only'), \
             mock.patch('summarizer.generated.chat_json', return_value={'questions': [], 'projects': []}) as chat:
            client.post(url)
            row = Generated.objects.get(kind='questions')
            self.assertEqual(row.status, 'done')
            self.assertTrue(row.data()['empty'])
            self.assertEqual(row.data()['papers_read'], 2)
            client.post(url)
            self.assertEqual(chat.call_count, 1)
        page = client.get('/en/collections/%d/' % collection.pk)
        self.assertContains(page, 'No sufficiently supported open question')
        self.assertContains(page, 'id="questions-again"')

    def test_malformed_response_is_retried_once_then_failure_can_be_retried(self):
        client, collection = self.questions_fixture()
        url = '/en/collections/%d/questions/' % collection.pk
        with mock.patch('summarizer.questions.paper_block', return_value='Text'), \
             mock.patch('summarizer.generated.chat_json', return_value={'questions': None}) as chat:
            client.post(url)
        row = Generated.objects.get(kind='questions')
        self.assertEqual(row.status, 'failed')
        self.assertEqual(chat.call_count, 2)
        page = client.get('/en/collections/%d/' % collection.pk)
        self.assertContains(page, 'previous analysis could not be completed')
        self.assertContains(page, 'id="questions-go"')
        with mock.patch('summarizer.questions.paper_block', return_value='Text'), \
             mock.patch('summarizer.generated.chat_json', return_value={'questions': [], 'projects': []}):
            client.post(url)
        row.refresh_from_db()
        self.assertEqual(row.status, 'done')

    def test_invalid_json_retries_and_malformed_citations_cannot_crash_or_be_invented(self):
        client, collection = self.questions_fixture()
        answer = {'questions': [None, {'question': None},
                  {'question': 'Why?', 'papers': [{'id': 'bad'}, None, '2101.00001v1', 'made-up', '2101.00001v1']}],
                  'projects': [{'title': 'No source', 'papers': ['made-up']}]}
        with mock.patch('summarizer.questions.paper_block', return_value='Text'), \
             mock.patch('summarizer.generated.chat_json', side_effect=[json.JSONDecodeError('bad', '', 0), answer]) as chat:
            client.post('/en/collections/%d/questions/' % collection.pk)
        row = Generated.objects.get(kind='questions')
        self.assertEqual(row.status, 'done')
        self.assertEqual(chat.call_count, 2)
        self.assertEqual([p['arxiv_id'] for p in row.data()['questions'][0]['papers']], ['2101.00001v1'])
        self.assertEqual(row.data()['projects'], [])

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
        self.assertContains(page, '/en/journal-club/')
        self.assertContains(page, '/en/press-kit/')
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
    def test_tools_count_apart_from_the_paper_search_and_say_when_limited(self):
        from summarizer import sources
        with mock.patch('summarizer.sources.OPENALEX_SEARCHES_PER_DAY', 2):
            self.assertEqual([sources._spend(sources.OPENALEX_SEARCH_KEY, 2) for _ in range(3)], [True, True, False])
            self.assertTrue(sources.tool_search_allowed())  # the site's searches used up theirs, not the tools'
        with mock.patch('summarizer.sources.TOOL_SEARCHES_PER_DAY', 3):
            self.assertEqual([sources.tool_search_allowed() for _ in range(2)], [True, True])
            self.assertFalse(sources.tool_search_allowed())
            with mock.patch('summarizer.sources._json') as call:
                self.assertIsNone(teams.works('exomoons', timezone.now().date()))
                self.assertEqual(refcheck.by_title('A title'), ('', ''))
                self.assertTrue(refcheck.LIMITED.get())
                call.assert_not_called()
        with mock.patch('summarizer.teams.works', return_value=None), \
             mock.patch('summarizer.generated.chat_json', return_value={'title': 'T', 'queries': ['q']}):
            self.assertTrue(teams.find('exomoons', 'en')['limited'])


class ThreadwatchTests(ToolsTestCase):
    def test_worker_threads_give_their_connection_back(self):
        from concurrent.futures import ThreadPoolExecutor
        from summarizer import sources
        with mock.patch('django.db.connections.close_all') as close, ThreadPoolExecutor(2) as pool:
            self.assertEqual(list(pool.map(sources.closing(lambda x: x * 2), [1, 2, 3])), [2, 4, 6])
        self.assertEqual(close.call_count, 3)

    def test_a_stuck_process_restarts_after_three_minutes(self):
        from summarizer import threadwatch
        state = {'checks': 0, 'stuck_for': 0}
        full = (12, 12, 12, 5)  # every connection in use, 5 requests waiting
        with mock.patch('summarizer.threadwatch.pool_stats', return_value=full), \
             mock.patch('summarizer.threadwatch.restart') as restart, \
             self.assertLogs('summarizer.threadwatch', 'WARNING') as logs:
            for _ in range(threadwatch.STUCK_MINUTES - 1):
                threadwatch.tick(state)
            restart.assert_not_called()
            threadwatch.tick(state)
        restart.assert_called_once()
        self.assertIn('12/12 in use (max 12), 5 waiting', logs.output[-1])
        self.assertIn('restarting', logs.output[-1])
        state = {'checks': 0, 'stuck_for': threadwatch.STUCK_MINUTES - 1}
        with mock.patch('summarizer.threadwatch.pool_stats', return_value=(3, 6, 12, 0)), \
             mock.patch('summarizer.threadwatch.restart') as restart:
            threadwatch.tick(state)  # the pool has room again: the count starts over
        restart.assert_not_called()
        self.assertEqual(state['stuck_for'], 0)


class LoadShedTests(TestCase):
    def test_requests_past_the_limit_get_a_503_at_once(self):
        import asyncio
        from summarizer.loadshed import LoadShedder

        async def run():
            release = asyncio.Event()

            async def app(scope, receive, send):
                if scope['path'] == '/boom/':
                    raise ValueError
                await release.wait()
                await send({'type': 'http.response.start', 'status': 200, 'headers': []})

            shedder = LoadShedder(app, limit=2)

            async def call(path):
                sent = []

                async def send(message):
                    sent.append(message)
                await shedder(({'type': 'http', 'path': path}), None, send)
                return sent[0]['status'], dict(sent[0]['headers'])
            running = [asyncio.create_task(call('/en/paper/%d/' % i)) for i in range(2)]
            await asyncio.sleep(0)
            with self.assertLogs('summarizer.loadshed', 'ERROR'):
                status, headers = await call('/en/paper/3/')
            self.assertEqual((status, headers[b'retry-after']), (503, b'5'))
            static = asyncio.create_task(call('/static/site.css'))  # not counted
            await asyncio.sleep(0)
            self.assertFalse(static.done())
            release.set()
            self.assertEqual([await t for t in running + [static]][0][0], 200)
            with self.assertRaises(ValueError):
                await call('/boom/')
            self.assertEqual(shedder.in_flight, 0)  # a failed request frees its place too
        asyncio.run(run())


class SearchCookieTests(TestCase):
    def test_result_pages_reached_without_the_visitor_cookie_ask_for_it_before_django(self):
        import asyncio
        from summarizer.loadshed import LoadShedder

        async def run():
            reached = []

            async def app(scope, receive, send):
                reached.append(scope['path'])
                await send({'type': 'http.response.start', 'status': 200, 'headers': [(b'content-type', b'text/html')]})
                await send({'type': 'http.response.body', 'body': b'results'})
            shedder = LoadShedder(app)

            async def call(path, query, cookie=None):
                sent = []

                async def send(message):
                    sent.append(message)
                headers = [(b'cookie', cookie)] if cookie else []
                await shedder({'type': 'http', 'path': path, 'query_string': query, 'headers': headers}, None, send)
                return sent[0]['status'], dict(sent[0]['headers']), sent[-1]['body']
            status, headers, body = await call('/en/search-results/', b'q=AI+fairness&page=2&window=abc')
            self.assertEqual((status, reached), (200, []))  # Django never saw it
            self.assertIn(b'sp=1', headers[b'set-cookie'])
            self.assertIn(b'href="/en/search-results/?q=AI+fairness&amp;page=2&amp;window=abc"', body)
            status, headers, body = await call('/en/search-results/', b'q=AI+fairness&page=2&window=abc', b'csrftoken=x; sp=1')
            self.assertEqual((body, reached), (b'results', ['/en/search-results/']))
            self.assertNotIn(b'set-cookie', headers)  # already known
            status, headers, body = await call('/en/search-results/', b'q=AI+fairness')  # a new search always passes
            self.assertEqual(body, b'results')
            self.assertIn(b'sp=1', headers[b'set-cookie'])
        asyncio.run(run())


class CanonicalHostTests(TestCase):
    def test_other_hosts_are_moved_to_www_before_django(self):
        import asyncio
        from summarizer.canonical_host import CanonicalHost

        async def run():
            reached = []

            async def app(scope, receive, send):
                reached.append(scope['path'])
                await send({'type': 'http.response.start', 'status': 200, 'headers': []})
                await send({'type': 'http.response.body', 'body': b'page'})
            app = CanonicalHost(app)

            async def call(host, path, query=b'', method='GET'):
                sent = []

                async def send(message):
                    sent.append(message)
                await app({'type': 'http', 'method': method, 'path': path, 'raw_path': path.encode(),
                           'query_string': query, 'headers': [(b'host', host)]}, None, send)
                return sent[0]['status'], dict(sent[0]['headers']).get(b'location')
            self.assertEqual(await call(b'summarizepaper.com', '/en/guides/arxiv-statistics/'),
                             (301, b'https://www.summarizepaper.com/en/guides/arxiv-statistics/'))
            self.assertEqual(await call(b'summarizepaper.herokuapp.com:443', '/en/search/', b'q=dust'),
                             (301, b'https://www.summarizepaper.com/en/search/?q=dust'))
            self.assertEqual(await call(b'summarizepaper.com', '/en/paper/2302.12218v1/', method='POST'),
                             (308, b'https://www.summarizepaper.com/en/paper/2302.12218v1/'))
            self.assertEqual(reached, [])
            self.assertEqual(await call(b'www.summarizepaper.com', '/en/'), (200, None))
            self.assertEqual(await call(b'localhost', '/en/'), (200, None))
        asyncio.run(run())


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
    WEEK = datetime.date(2026, 10, 12)  # a Monday

    def week_papers(self, week, titles, category='cs.LG', authors=None):
        return [{'key': '%s.%05d' % (week.strftime('%y%m%d'), i), 'day': week + datetime.timedelta(days=i % 5),
                 'title': title, 'abstract': 'Plain words here.', 'categories': [category, 'stat.ML'],
                 'authors': authors[i] if authors else ['Author%d Smith%d' % (i, i)]} for i, title in enumerate(titles)]

    def history(self, rising=10, team=None):
        """Nine weeks of cs.LG papers: 'sparse autoencoders' rare, then used by `rising` papers in the last week;
        team: so many papers of the last week on 'mamba state spaces', all by one group."""
        for back in range(9, -1, -1):
            week = self.WEEK - datetime.timedelta(weeks=back)
            titles = ['Graph networks study %d' % i for i in range(60)]
            titles += ['Sparse autoencoders for task %d' % i for i in range(rising if back == 0 else back % 2)]
            authors = None
            if back == 0 and team:
                titles += ['Mamba state spaces, part %d' % i for i in range(team)]
                authors = [['Author%d Smith%d' % (i, i)] for i in range(len(titles) - team)] + [['Ann Lee', 'Bob Ray']] * team
            papers = self.week_papers(week, titles, authors=authors)
            trends.store(week, trends.count(papers))
            if back == 0:
                trends.keep_papers(papers)

    def test_terms(self):
        found = trends.terms('Sparse autoencoders for the interpretation of $x^2$ models. Plain words. RAG with LLMs and LoRA')
        self.assertIn('sparse autoencoder', found)
        self.assertNotIn('models plain', found)
        self.assertNotIn('for the', found)
        self.assertNotIn('the interpretation', found)
        self.assertTrue({'rag', 'llm', 'lora', 'rag with llm'} <= found)
        self.assertNotIn('with', found)
        self.assertEqual(trends.fold('galaxies'), 'galaxy')
        self.assertEqual(trends.fold('physics'), 'physics')
        self.assertEqual(trends.fold('approaches'), 'approach')
        self.assertEqual(trends.normalize('Large Language Models'), 'large language model')
        self.assertEqual(trends.normalize('JWST'), 'jwst')
        self.assertEqual(trends.normalize('the model'), '')
        self.assertEqual(trends.normalize('large models'), '')
        self.assertIn('diffusion model', trends.terms('Diffusion models for weather'))
        self.assertEqual(trends.normalize('one two three four'), '')

    def test_weeks_close_the_saturday_after(self):
        self.assertEqual(trends.closed_week(datetime.date(2026, 10, 24)), datetime.date(2026, 10, 12))  # Saturday
        self.assertEqual(trends.closed_week(datetime.date(2026, 10, 23)), datetime.date(2026, 10, 5))

    def test_a_term_rises_against_its_history_from_several_teams(self):
        self.history(rising=12, team=12)
        rising, left = trends.rising_terms('cs.LG', self.WEEK, verbose=True)
        terms = [t['term'] for t in rising]
        found = rising[terms.index('sparse autoencoder')]
        self.assertGreaterEqual(found['llr'], trends.MIN_LLR)
        self.assertEqual(found['teams'], 12)
        self.assertNotIn('graph network', terms)
        self.assertNotIn('sparse', terms)  # kept as part of the longer term
        self.assertNotIn('mamba state space', terms)  # one group's series
        self.assertIn('mamba state space', [t['term'] for t in left])
        self.assertIsNone(trends.rising_terms('cs.LG', self.WEEK - datetime.timedelta(weeks=7)))  # too little history

    def test_a_small_rise_is_not_significant(self):
        self.history(rising=4)
        self.assertNotIn('sparse autoencoder', [t['term'] for t in trends.rising_terms('cs.LG', self.WEEK) or []])

    def test_history_reads_one_part_per_week(self):
        with mock.patch('summarizer.trends.PART_TERMS', 5):
            self.history()
        self.assertGreater(TermWeek.objects.filter(scope='cs.LG', week=self.WEEK).count(), 1)
        points = trends.history('cs.LG', ['sparse autoencoder', 'unknown term'])
        self.assertEqual([n for _, _, n in points['sparse autoencoder']][-3:], [0, 0, 10])  # single uses are not stored
        self.assertEqual(points['sparse autoencoder'][-1][1], 70)
        self.assertEqual(sum(n for _, _, n in points['unknown term']), 0)

    def test_a_week_is_counted_a_group_of_scopes_at_a_time(self):
        papers = self.week_papers(self.WEEK, ['Graph networks %d' % i for i in range(5)])
        papers += [dict(papers[0], key='bio', categories=['biorxiv.neuroscience'])]
        trends.count_week(self.WEEK, papers)
        stored = set(TermWeek.objects.filter(week=self.WEEK).values_list('scope', flat=True))
        self.assertEqual(stored, set(trends.count(papers)))
        self.assertEqual(TermWeek.objects.get(scope='*', week=self.WEEK).papers, 5)
        trends.count_week(self.WEEK, papers[-1:], 'bio')  # bioRxiv read again apart: arXiv's counts stay
        self.assertEqual(set(TermWeek.objects.filter(week=self.WEEK).values_list('scope', flat=True)), stored)
        self.assertEqual(TermWeek.objects.get(scope='*', week=self.WEEK).papers, 5)

    def test_kinds_of_rise(self):
        week = self.WEEK

        def points(values):
            return [(week - datetime.timedelta(weeks=len(values) - 1 - i), 1000, n) for i, n in enumerate(values)]
        self.assertEqual(trends.kind(points([0] * 20 + [1, 0, 3, 20]))['kind'], 'new')
        growing = trends.kind(points([5] * 20 + [12, 15, 20]))
        self.assertEqual((growing['kind'], growing['weeks']), ('growing', 3))
        self.assertEqual(trends.kind(points([25] + [5] * 51 + [20]))['kind'], 'recurring')
        comeback = trends.kind(points([5] * 10 + [22] + [5] * 20 + [20]))
        self.assertEqual(comeback['kind'], 'comeback')
        self.assertEqual(comeback['since'], (week - datetime.timedelta(weeks=21)).isoformat())
        self.assertEqual(trends.kind(points([5] * 20 + [20]))['kind'], 'spike')

    def written(self):
        self.history(rising=12)
        key = trends.stored_papers(self.WEEK, 'cs.LG')[-1]['key']
        answer = {'topics': [{'title': {'en': 'Looking inside AI models', 'fr': 'Regarder dans les modèles'},
                              'text': {'en': 'Why.', 'fr': 'Pourquoi.'}, 'query': 'sparse autoencoders for interpretability',
                              'terms': ['sparse autoencoder', 'invented'], 'papers': [key, 'x']},
                             {'title': {'en': 'Noise', 'fr': 'Bruit'}, 'text': {}, 'terms': ['not a rising term']}]}
        vector = np.ones((1, 4), dtype=np.float32) / 2
        with mock.patch('summarizer.generated.chat_json', return_value=answer), \
             mock.patch('summarizer.trends.first_use', return_value={'arxiv_id': '1312.0001', 'title': 'The first one', 'year': 2013}), \
             mock.patch('summarizer.similar.embed', return_value=vector):
            content = trends.write('cs.LG', self.WEEK)
        trends._RECENT.clear()
        return content, key

    def test_topics_are_written_in_every_language_with_their_history(self):
        content, key = self.written()
        topic = content['topics'][0]
        self.assertEqual(len(content['topics']), 1)
        self.assertEqual(topic['terms'], ['sparse autoencoder'])
        self.assertEqual((topic['count'], topic['teams'], topic['kind']), (12, 12, 'spike'))
        self.assertEqual(topic['papers'][0]['arxiv_id'], key)
        self.assertEqual(len(topic['papers']), 4)
        self.assertEqual(len(topic['series']), 10)
        self.assertEqual(topic['first']['year'], 2013)
        self.assertEqual(topic['query'], 'sparse autoencoders for interpretability')
        fr = json.loads(RisingTopics.objects.get(scope='cs.LG', lang='fr').content)['topics'][0]
        self.assertEqual(fr['title'], 'Regarder dans les modèles')
        self.assertEqual(trends.for_digest(['cs.LG'], 'en', self.WEEK + datetime.timedelta(days=14))[0]['archive'], 'cs.LG')
        self.assertEqual(trends.topic_of_paper(key + 'v2', 'en')[0], 'cs.LG')
        self.assertEqual(trends.close_topics(np.ones(4, dtype=np.float32) / 2, 'en')[0][1], 'cs.LG')

    def test_pages(self):
        _, key = self.written()
        paid = self.client_for(self.user(paid=True))
        page = paid.get('/en/trends/?scope=cs.LG')
        self.assertContains(page, 'Looking inside AI models')
        self.assertContains(page, 'from 12 teams')
        self.assertContains(page, 'Sudden rise')
        self.assertContains(page, '/en/teams/?topic=sparse%20autoencoders%20for%20interpretability')
        self.assertContains(page, '<svg class="spark"')
        self.assertContains(page, 'The oldest arXiv paper using these words (2013)')
        self.assertContains(Client().get('/en/trends/?scope=cs.LG'), 'part of Premium')
        self.assertEqual(Client().get('/en/trends/?scope=bogus').status_code, 404)
        self.assertEqual(paid.get('/en/trends/?scope=cs.LG&week=2020-01-06').status_code, 404)
        papers = paid.get('/en/trends/papers/?scope=cs.LG&week=2026-10-12&t=sparse+autoencoders')
        self.assertContains(papers, '12 papers')
        self.assertContains(papers, key)
        explore = paid.get('/en/trends/explore/?scope=cs.LG&t=Sparse+Autoencoders&t=graph+networks&t=the')
        self.assertContains(explore, '"Sparse Autoencoders" is counted as "sparse autoencoder"')
        self.assertContains(explore, '"series": [{"term": "sparse autoencoder"')
        self.assertContains(explore, 'cannot be counted')
        self.assertContains(explore, 'class="term-chart"')
        csv_page = paid.get('/en/trends/explore/?scope=cs.LG&t=sparse+autoencoder&format=csv')
        self.assertIn('2026-10-12,72,12,', csv_page.content.decode())
        free = Client().get('/en/trends/explore/?t=sparse+autoencoder&t=graph+network&format=csv')
        self.assertContains(free, 'Only the first term is shown')
        self.assertNotIn('text/csv', free['Content-Type'])
        subscription = DigestSubscription.objects.create(user=User.objects.get(username='ann'), categories='cs.LG')
        DigestSubscription.objects.filter(pk=subscription.pk).update(
            profile_vector=(np.ones((1, 4), dtype=np.float32) / 2).tobytes(), profile_labels='["me"]')
        self.assertContains(paid.get('/en/trends/?view=foryou'), 'Looking inside AI models')
        self.assertContains(Client().get('/en/trends/?view=foryou'), 'With Premium')

    def test_links_from_papers_and_projects(self):
        from django.template import Context, Template
        _, key = self.written()
        with translation.override('en'):
            shown = Template('{% load rising %}{% rising_topic key as r %}{{ r.title }} {{ r.scope }}').render(Context({'key': key + 'v1'}))
        self.assertEqual(shown, 'Looking inside AI models cs.LG')
        user = self.user(paid=True)
        WatchedProject.objects.create(user=user, title='Probing features', description='x' * 300, last_day=self.WEEK,
                                      vector=(np.ones(4, dtype=np.float32) / 2).tobytes())
        self.assertContains(self.client_for(user).get('/en/scoops/'), 'A close topic is rising in Machine Learning (cs.LG)')

    def test_page_before_history(self):
        self.assertContains(Client().get('/en/trends/'), 'first rising topics')
        self.assertEqual(Client().get('/en/trends/?scope=bogus').status_code, 404)

    def test_free_visitors_see_three_topics(self):
        topics = [{'title': 'Topic %d' % i, 'text': 't', 'terms': ['a'], 'main': 'a', 'count': 9, 'growth': 3.0, 'papers': []}
                  for i in range(5)]
        RisingTopics.objects.create(scope='*', lang='en', day=datetime.date(2026, 10, 19), content=json.dumps({'topics': topics}))
        page = Client().get('/en/trends/')
        self.assertContains(page, 'Topic 2')
        self.assertNotContains(page, 'Topic 3')
        self.assertContains(page, '2 more rising topics')
        self.assertContains(self.client_for(self.user(paid=True)).get('/en/trends/'), 'Topic 4')

    def test_update_reads_the_completed_weeks_then_writes(self):
        now = timezone.make_aware(datetime.datetime(2026, 10, 24, 12))
        with mock.patch('summarizer.trends.harvest') as harvest, mock.patch('summarizer.trends.write', return_value=None):
            trends.update(now)
        self.assertEqual(sorted(harvest.call_args[0][0])[-1], self.WEEK)
        self.assertEqual(len(harvest.call_args[0][0]), 4)

    def test_oai_records_are_dated_by_their_first_version(self):
        from summarizer import trend_sources
        xml = b'''<?xml version="1.0"?><OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
          <record><header><identifier>oai:arXiv.org:2610.00001</identifier></header><metadata>
            <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/"><id>2610.00001</id>
              <version version="v1"><date>Tue, 13 Oct 2026 10:00:00 GMT</date></version>
              <title>A new  title</title><authors>A. Lee, B. Ray and C. Day</authors><categories>cs.LG stat.ML</categories>
              <abstract>Text.</abstract></arXivRaw></metadata></record>
          <record><header><identifier>oai:arXiv.org:0810.5176</identifier></header><metadata>
            <arXivRaw xmlns="http://arxiv.org/OAI/arXivRaw/"><id>0810.5176</id>
              <version version="v1"><date>Wed, 29 Oct 2008 01:34:26 GMT</date></version>
              <version version="v2"><date>Tue, 13 Oct 2026 10:00:00 GMT</date></version>
              <title>Old</title><authors>X</authors><categories>math.NT</categories><abstract>Old.</abstract></arXivRaw></metadata></record>
          <resumptionToken></resumptionToken></ListRecords></OAI-PMH>'''
        with mock.patch('summarizer.trend_sources.requests.get', return_value=mock.Mock(status_code=200, content=xml)):
            papers = list(trend_sources.arxiv_papers(self.WEEK))
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0]['day'], datetime.date(2026, 10, 13))
        self.assertEqual(papers[0]['authors'], ['A. Lee', 'B. Ray', 'C. Day'])
        self.assertEqual(papers[0]['title'], 'A new title')


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
