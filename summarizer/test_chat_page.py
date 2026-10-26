"""The chat page: what the assistant reads of each paper, conversations kept in the page address,
the comparison table, the stop button and the endpoints of the page script."""
import asyncio
import datetime
import json
from unittest import mock

from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import AnonymousUser
from django.contrib.sessions.backends.db import SessionStore
from django.test import Client, TransactionTestCase, override_settings

from summarizer import access, activity, assistant, chat_page, chat_table, evidence
from summarizer.models import AIassistant, ArxivPaper, ChatShare, PaperText
from summarizer.routing import websocket_urlpatterns
from summarizer.test_tools import ToolsTestCase

LOCAL_IP = access.hash_ip('127.0.0.1')


def section(title, words):
    return '## %s\n\n%s\n\n' % (title, ' '.join('%s%d.' % (title.split()[-1].lower(), i) for i in range(words)))


class FitTests(ToolsTestCase):
    def test_a_short_paper_is_given_whole(self):
        self.assertEqual(assistant.fit('Short paper.', 1000), ('Short paper.', 1.0, []))

    def test_references_go_first_then_appendices_then_sections_keep_their_start(self):
        body = section('Introduction', 400) + section('3 Results', 400) + section('5 Conclusion', 20)
        references = '## References\n\n' + 'Smith J. 2020, ApJ, 1. ' * 300 + '\n\n'
        appendix = '## Appendix A Derivations\n\n' + 'Long derivation. ' * 300
        text = body + references + appendix
        fitted, kept, left_out = assistant.fit(text, len(body + appendix) + 10)
        self.assertEqual((left_out, kept), (['references'], 1.0))
        self.assertNotIn('Smith J.', fitted)
        self.assertIn('Long derivation.', fitted)

        fitted, kept, left_out = assistant.fit(text, len(body) + 10)
        self.assertEqual(left_out, ['references', 'appendices'])
        self.assertNotIn('derivation', fitted)
        self.assertLess(kept, 1.0)

        fitted, kept, left_out = assistant.fit(text, len(body) // 2)
        self.assertEqual(left_out, ['references', 'appendices', 'shortened'])
        self.assertLessEqual(len(fitted), len(body) // 2)
        for start in ('## Introduction\n\nintroduction0.', '## 3 Results\n\nresults0.'):
            self.assertIn(start, fitted)
        self.assertIn(section('5 Conclusion', 20).strip(), fitted)  # short: whole
        self.assertEqual(fitted.count(assistant.OMITTED), 2)

    def test_a_pdf_text_without_headings_keeps_the_start_of_each_eighth(self):
        text = '\n\n'.join('Paragraph %d. ' % i + 'word ' * 200 for i in range(40))
        fitted, kept, left_out = assistant.fit(text, len(text) // 3)
        self.assertEqual(left_out, ['shortened'])
        self.assertIn('Paragraph 0.', fitted)
        self.assertIn('Paragraph 36.', fitted)  # the end of the paper is not lost
        self.assertAlmostEqual(kept, 1 / 3, delta=0.05)

    def test_with_a_question_the_passages_about_it_are_kept(self):
        filler = ' '.join('Generic sentence number %d about nothing in particular.' % i for i in range(30))
        text = ('## 1 Introduction\n\nWe introduce the survey.\n\n' +
                '\n\n'.join(filler for _ in range(6)) +
                '\n\nThe spectroscopic redshift calibration reaches 0.3 percent precision.\n\n' +
                '\n\n'.join(filler for _ in range(6)) + '\n\n## 5 Conclusion\n\nWe conclude.')
        fitted, kept, left_out = assistant.fit(text, 1500, 'How precise is the redshift calibration?')
        self.assertEqual(left_out, ['selected'])
        self.assertLessEqual(len(fitted), 1500)
        for kept_text in ('## 1 Introduction', 'We introduce the survey.', '0.3 percent precision', '## 5 Conclusion', 'We conclude.'):
            self.assertIn(kept_text, fitted)
        self.assertIn('[…]', fitted)
        # a question made only of common words does not choose: the sections keep their start
        self.assertEqual(assistant.fit(text, 1500, 'What about this paper?')[2], ['shortened'])

    def test_more_papers_have_more_claims_checked(self):
        self.assertEqual([evidence.max_claims(n) for n in (1, 2, 5, 10)], [8, 11, 20, 20])
        self.assertIn('20 at most', evidence.prompt('English', 20))

    def test_share_gives_the_room_short_papers_leave_to_long_ones(self):
        self.assertEqual(assistant.share([100, 5000, 5000], 6100), [100, 3000, 3000])
        self.assertEqual(assistant.share([100, 200], 1000), [100, 200])


class SourcesTests(TransactionTestCase):  # the answers run in a thread of their own
    paper = ToolsTestCase.paper

    def setUp(self):
        self.a = self.paper('2101.00001v2', authors=('Ann Lee', 'Bob Ray', 'Cy Doe'))
        self.b = self.paper('2101.00002v1', title='Dust', authors=('Ann Lee', 'Bob Ray', 'Eve Fox'))
        self.c = self.paper('2101.00003v1', title='Ice', authors=('Zoe Kim', 'Bob Ray'))
        self.c.license = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
        self.c.journal_ref = 'ApJ 900, 1'
        self.c.save()
        PaperText.objects.create(arxiv_id='2101.00001v2', text='FULL TEXT A ' * 50, source='html')
        PaperText.objects.create(arxiv_id='2101.00002v1', text='FULL TEXT B ' * 50, source='pdf')

    def test_several_papers_are_tagged_dated_and_say_what_is_supplied(self):
        context, supplied = assistant.build_sources(['2101.00001v2', '2101.00002v1', '2101.00003v1', 'unknown'])
        self.assertEqual([s['tag'] for s in supplied], ['[Lee et al. 2021a]', '[Lee et al. 2021b]', '[Kim & Ray 2021]'])
        self.assertIn('Cite as: [Kim & Ray 2021]', context)
        self.assertIn('Date: first submitted to arXiv on 2021-01-04; this is version 2', context)
        self.assertIn('Published in: ApJ 900, 1', context)
        self.assertIn('Category: astro-ph.EP', context)
        self.assertIn('Text supplied: the full text', context)
        self.assertIn('Text supplied: the title, authors and abstract only (the license', context)
        summary = assistant.supplied_summary(supplied, ['2101.00001v2', '2101.00002v1', '2101.00003v1', 'unknown'])
        self.assertEqual(summary['missing'], ['unknown'])
        self.assertEqual([(p['kind'], p['coverage']) for p in summary['papers']],
                         [('full', 100), ('full', 100), ('restricted', 100)])

    def test_a_single_paper_has_no_tag_and_no_multi_paper_rules(self):
        async def complete(messages, paid, max_tokens, on_delta=None, **kwargs):
            calls.append((messages, max_tokens))
            return 'Answer.'
        calls, sent = [], []

        async def on_sources(summary):
            sent.append(summary)
        with mock.patch.object(assistant, 'complete', complete):
            asyncio.run(assistant.answer(['2101.00001v2'], 'en', 'Q?', False, hashed_ip='a', on_sources=on_sources))
            asyncio.run(assistant.answer(['2101.00001v2', '2101.00002v1'], 'en', 'Q?', False, hashed_ip='a'))
        (single, single_tokens), (multi, multi_tokens) = calls
        self.assertNotIn('Cite as', single[1]['content'])
        self.assertNotIn('Several papers are supplied', single[0]['content'])
        self.assertIn('Several papers are supplied', multi[0]['content'])
        self.assertEqual((single_tokens, multi_tokens), (assistant.ANSWER_TOKENS, assistant.ANSWER_TOKENS + assistant.PAPER_TOKENS))
        self.assertEqual(sent[0]['papers'][0]['arxiv_id'], '2101.00001v2')
        self.assertEqual(list(AIassistant.objects.order_by('pk').values_list('papers', flat=True)),
                         ['2101.00001v2', '2101.00001v2,2101.00002v1'])

    def test_history_is_only_about_the_same_papers(self):
        AIassistant.objects.create(arxiv_id='room', lang='en', ip_address='a', query='Old', response='About A',
                                   papers='2101.00001v2')
        AIassistant.objects.create(arxiv_id='room', lang='en', ip_address='a', query='Now', response='About A and B',
                                   papers='2101.00001v2,2101.00002v1')
        AIassistant.objects.create(arxiv_id='room', lang='en', ip_address='a', query='Table', response='|',
                                   papers='2101.00001v2,2101.00002v1', kind='table')
        history = assistant.conversation_history('room', 'en', None, 'a', '2101.00001v2,2101.00002v1')
        self.assertEqual([m['content'] for m in history], ['Now', 'About A and B'])
        self.assertEqual(len(assistant.conversation_history('room', 'en', None, 'a')), 4)

    def test_a_table_uses_one_question_per_paper(self):
        AIassistant.objects.create(arxiv_id='room', lang='en', ip_address='a', query='T', kind='table', cost=3)
        AIassistant.objects.create(arxiv_id='room', lang='en', ip_address='a', query='Q')
        self.assertEqual(access.chat_quota_left(None, 'a'), access.chat_messages_per_day()['anonymous'] - 4)


class TableTests(TransactionTestCase):
    paper = ToolsTestCase.paper

    def setUp(self):
        self.paper('2101.00001v1')
        self.paper('2101.00002v1', title='Dust', authors=('Zoe Kim',))
        PaperText.objects.create(arxiv_id='2101.00001v1', text='## 2 Data\n\nWe observed 42 planets with JWST.', source='html')
        PaperText.objects.create(arxiv_id='2101.00002v1', text='## 2 Data\n\nWe simulate dust grains.', source='html')

    def test_cells_are_checked_against_the_paper(self):
        paper = chat_table.prepare(['2101.00001v1'])[0]
        cells = chat_table.row_of(paper, ['Data', 'Method', 'Limits', 'Sample'], {'cells': [
            {'column': 'data', 'value': '42 planets observed with JWST', 'quote': 'We observed 42 planets with JWST.'},
            {'column': 'Method', 'value': 'NOT_STATED', 'quote': ''},
            {'column': 'Limits', 'value': 'NOT_IN_SUPPLIED_TEXT', 'quote': ''},
            {'column': 'Sample', 'value': '57 planets', 'quote': 'We observed 57 planets.'}]})
        self.assertEqual([c['status'] for c in cells], ['verified', 'missing', 'unknown', 'not_found'])
        self.assertEqual((cells[0]['section'], cells[0]['numbers_absent']), ('2 Data', []))
        self.assertTrue(cells[0]['link'].startswith('https://arxiv.org/html/2101.00001v1#:~:text='))
        self.assertEqual(cells[3]['numbers_absent'], ['57'])
        self.assertEqual(chat_table.clean_columns([' Data ', 'data', 'Method']), ['Data', 'Method'])
        for invalid in ([], ['x'] * 9, [1], 'Data'):
            self.assertIsNone(chat_table.clean_columns(invalid))

    def test_the_table_reads_each_paper_alone_and_is_stored(self):
        async def complete(messages, paid, max_tokens, on_delta=None, task='assistant', user_id=None, json_object=False):
            self.assertEqual((task, json_object), ('table', True))
            source = messages[1]['content']
            calls.append(source if '<paper>' in source else '')
            if 'comparison_table' in source:
                return json.dumps({'agreements': [], 'disagreements': ['[Lee & Ray 2021] observes, [Kim 2021] simulates.'], 'gaps': []})
            self.assertEqual(source.count('<paper>'), 1)
            if 'Dust' in source:
                return 'not json'
            return json.dumps({'cells': [{'column': 'Data', 'value': '42 planets', 'quote': 'We observed 42 planets with JWST.'}]})
        rows, calls = [], []

        async def on_row(index, row):
            rows.append(index)
        with mock.patch.object(assistant, 'complete', complete):
            table = asyncio.run(chat_table.build(['2101.00001v1', '2101.00002v1'], ['Data'], 'en', False,
                                                 hashed_ip='a', conversation_id='room', on_row=on_row))
        self.assertEqual(sorted(rows), [0, 1])
        self.assertEqual([r['cells'][0]['status'] for r in table['rows']], ['verified', 'error'])
        self.assertEqual(table['disagreements'], ['[Lee & Ray 2021] observes, [Kim 2021] simulates.'])
        row = AIassistant.objects.get()
        self.assertEqual((row.kind, row.cost, row.arxiv_id, row.papers), ('table', 2, 'room', '2101.00001v1,2101.00002v1'))
        # built again: the row read before is reused, only the one that failed is read (and counted)
        calls.clear()
        with mock.patch.object(assistant, 'complete', complete):
            table = asyncio.run(chat_table.build(['2101.00001v1', '2101.00002v1'], ['Data'], 'en', False,
                                                 hashed_ip='a', conversation_id='room'))
        self.assertEqual(sum('Dust' in c for c in calls), 1)
        self.assertFalse(any('Clouds' in c for c in calls))
        self.assertEqual(table['rows'][0]['cells'][0]['status'], 'verified')
        self.assertEqual(AIassistant.objects.latest('pk').cost, 1)
        # other columns: read again
        self.assertEqual(chat_table.reusable_rows(['2101.00001v1'], ['Method'], 'en'), {})
        self.assertIn('| [Lee & Ray 2021] Clouds on a hot planet | 42 planets |', row.response)
        self.assertIn('Could not be read', row.response)


class PageTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.paper('2101.00001v1')
        self.paper('2101.00002v1', title='Dust')

    def test_a_conversation_reopens_for_its_owner_only(self):
        AIassistant.objects.create(arxiv_id='abcdef0123456789', lang='en', ip_address=LOCAL_IP, query='My question?',
                                   response='**Answer**', papers='2101.00001v1,2101.00002v1')
        AIassistant.objects.create(arxiv_id='abcdef0123456789', lang='en', ip_address='someone else',
                                   query='THEIR_QUESTION', response='x', papers='2101.00001v1')
        page = Client().get('/en/chat/?c=abcdef0123456789')
        self.assertContains(page, 'My question?')
        self.assertNotContains(page, 'THEIR_QUESTION')
        self.assertContains(page, 'data-room="abcdef0123456789"')
        preset = page.context['preset']
        self.assertEqual([p['id'] for p in preset], ['2101.00001v1', '2101.00002v1'])
        self.assertEqual(preset[0]['title'], 'Clouds on a hot planet')
        # listed among the recent conversations of another page, never a paper page's conversation
        AIassistant.objects.create(arxiv_id='2101.00001v1', lang='en', ip_address=LOCAL_IP, query='Paper page',
                                   response='x', papers='2101.00001v1')
        recent = Client().get('/en/chat/').context['conversations']
        self.assertEqual([(c['room'], c['question']) for c in recent], [('abcdef0123456789', 'My question?')])
        user = self.user()
        self.assertNotContains(self.client_for(user).get('/en/chat/?c=abcdef0123456789'), 'My question?')

    def test_ids_from_a_collection_preset_the_selection(self):
        page = Client().get('/en/chat/?ids=2101.00002v1,hep-th/9901001v1,<script>')
        self.assertEqual([p['id'] for p in page.context['preset']], ['2101.00002v1', 'hep-th--9901001v1'])
        self.assertNotIn('title', page.context['preset'][1])  # unknown here: the page script looks it up
        self.assertEqual(Client().get('/en/chat/?c=bad room').context['room'], '')

    def test_sources_and_exports_of_the_selection(self):
        data = Client().get('/en/chat/sources/?ids=2101.00001v1,2101.00002v1,2101.00009v1').json()
        self.assertEqual(data['missing'], ['2101.00009v1'])
        self.assertEqual([p['tag'] for p in data['papers']], ['[Lee & Ray 2021a]', '[Lee & Ray 2021b]'])
        bib = Client().get('/en/chat/export/bib/?ids=2101.00001v1,2101.00002v1')
        self.assertEqual(bib.status_code, 200)
        self.assertEqual(bib.content.decode().count('@misc{'), 2)
        self.assertEqual(Client().get('/en/chat/export/exe/?ids=2101.00001v1').status_code, 404)
        self.assertEqual(Client().get('/en/chat/sources/?ids=' + ','.join('a%d' % i for i in range(11))).status_code, 404)

    @mock.patch.object(chat_page, 'S2_RETRY_SECONDS', 0)
    def test_references_with_an_arxiv_id_or_a_doi_can_be_added(self):
        answer = mock.Mock(status_code=200, json=lambda: {'data': [
            {'citedPaper': {'title': 'Old  paper', 'year': 2019, 'authors': [{'name': 'A B'}], 'externalIds': {'ArXiv': '1901.00001'}}},
            {'citedPaper': {'title': 'Journal paper', 'year': 2018, 'authors': [], 'externalIds': {'DOI': '10.1/x'}}},
            {'citedPaper': {'title': 'Book', 'externalIds': {}}}]})
        with mock.patch('summarizer.chat_page.requests.get', return_value=answer) as get:
            data = Client().get('/en/chat/references/2101.00001v1/').json()
            Client().get('/en/chat/references/2101.00001v1/')
        get.assert_called_once()  # cached
        self.assertIn('arXiv:2101.00001', get.call_args[0][0])
        self.assertEqual([(r['title'], r['ref']) for r in data['references']], [('Old paper', '1901.00001'), ('Journal paper', '10.1/x')])
        # Semantic Scholar refuses (no key, or a wrong one): OpenAlex, without the references younger than the paper
        works = {'results': [
            {'title': 'Preprint', 'publication_year': 2019, 'doi': 'https://doi.org/10.1/p', 'authorships': [],
             'locations': [{'landing_page_url': 'http://arxiv.org/abs/1902.00002'}]},
            {'title': 'Article', 'publication_year': 2020, 'doi': 'https://doi.org/10.1/a',
             'authorships': [{'author': {'display_name': 'C D'}}], 'locations': []},
            {'title': 'Merged record', 'publication_year': 2026, 'doi': 'https://doi.org/10.1/m', 'authorships': [], 'locations': []}]}
        answers = lambda url, params=None: {'referenced_works': ['https://openalex.org/W1', 'https://openalex.org/W2']} if '/works/doi:' in url else works
        with mock.patch('summarizer.chat_page.requests.get', return_value=mock.Mock(status_code=429)) as s2, \
             mock.patch('summarizer.sources._json', side_effect=answers) as openalex:
            data = Client().get('/en/chat/references/2101.00002v1/').json()
        self.assertEqual(s2.call_count, 2)  # asked again once after its limit
        self.assertIn('doi:10.48550/arXiv.2101.00002', openalex.call_args_list[0][0][0])
        self.assertEqual([(r['ref'], r['authors']) for r in data['references']], [('1902.00002', ''), ('10.1/a', 'C D')])
        with mock.patch('summarizer.chat_page.requests.get', return_value=mock.Mock(status_code=403)), \
             mock.patch('summarizer.sources._json', side_effect=chat_page.sources.SourceUnavailable('down')):
            self.assertEqual(Client().get('/en/chat/references/2101.00009v1/').json(), {'status': 'unavailable'})

    def test_the_table_builder_is_for_premium(self):
        page = Client().get('/en/chat/')
        self.assertNotContains(page, 'id="table-build"')
        self.assertContains(page, 'The comparison table is part of Premium.')
        page = self.client_for(self.user(paid=True, tier='premium')).get('/en/chat/')
        self.assertContains(page, 'id="table-build"')

    def test_a_shared_conversation_is_public_read_only_and_frozen(self):
        room = 'abcdef0123456789'
        AIassistant.objects.create(arxiv_id=room, lang='en', ip_address=LOCAL_IP, query='Shared question?',
                                   response='**Shared answer**', papers='2101.00001v1')
        AIassistant.objects.create(arxiv_id=room, lang='en', ip_address='someone else', query='NOT_MINE', response='x')
        self.assertEqual(Client().post('/en/chat/share/', {'room': 'fedcba9876543210'}).status_code, 404)  # not mine
        url = Client().post('/en/chat/share/', {'room': room}).json()['url']
        token = url.rstrip('/').rsplit('/', 1)[1]
        AIassistant.objects.create(arxiv_id=room, lang='en', ip_address=LOCAL_IP, query='LATER_QUESTION', response='y')
        reader = self.client_for(self.user('reader'))
        page = reader.get(url.split('testserver', 1)[1])
        self.assertContains(page, 'Shared question?')
        self.assertContains(page, 'noindex')
        self.assertContains(page, 'Clouds on a hot planet')
        self.assertContains(page, '/en/chat/?ids=2101.00001v1')
        self.assertNotContains(page, 'NOT_MINE')
        self.assertNotContains(page, 'LATER_QUESTION')
        self.assertNotContains(page, '127.0.0.1')
        # the owner sees the link on the conversation, sharing again shows the later exchanges
        self.assertContains(Client().get('/en/chat/?c=' + room), token)
        self.assertEqual(Client().post('/en/chat/share/', {'room': room}).json()['url'], url)
        self.assertContains(reader.get('/en/chat/shared/%s/' % token), 'LATER_QUESTION')
        self.assertEqual(reader.post('/en/chat/share/%s/stop/' % token).status_code, 404)  # only its owner stops it
        self.assertEqual(Client().post('/en/chat/share/%s/stop/' % token).json(), {'status': 'done'})
        self.assertEqual(reader.get('/en/chat/shared/%s/' % token).status_code, 404)
        self.assertFalse(ChatShare.objects.get().active)

    def test_questions_left_today(self):
        self.assertEqual(Client().get('/en/chat/quota/').json(), {'left': access.chat_messages_per_day()['anonymous']})
        AIassistant.objects.create(arxiv_id='room', lang='en', ip_address=LOCAL_IP, query='Q', cost=2)
        self.assertEqual(Client().get('/en/chat/quota/').json()['left'], access.chat_messages_per_day()['anonymous'] - 2)
        self.assertContains(Client().get('/en/chat/'), 'id="chat-quota" data-left="%d"' % (access.chat_messages_per_day()['anonymous'] - 2))
        staff = self.user('boss')
        staff.is_superuser = True
        staff.save()
        self.assertEqual(self.client_for(staff).get('/en/chat/quota/').json(), {'left': None})

    def test_a_paper_page_conversation_can_be_shared_and_previews_well(self):
        AIassistant.objects.create(arxiv_id='2101.00001v1', lang='en', ip_address=LOCAL_IP, query='What is **new**?',
                                   response='The **clouds** are $x$ thick.', papers='2101.00001v1')
        url = Client().post('/en/chat/share/', {'room': '2101.00001v1'}).json()['url']
        page = Client().get(url.split('testserver', 1)[1])
        self.assertContains(page, '<meta property="og:title" content="What is **new**?">')
        self.assertContains(page, '<meta property="og:description" content="The clouds are x thick.">')
        paper = Client().get('/en/paper/2101.00001v1/')
        self.assertContains(paper, 'id="share-conversation"')
        self.assertContains(paper, '<textarea maxlength="6000"')
        self.assertContains(paper, 'id="stop-btn"')

    def test_the_staff_sees_how_far_answers_hold(self):
        claims = [{'status': 'verified'}, {'status': 'verified', 'numbers_absent': ['57']}, {'status': 'not_found'},
                  {'status': 'general'}]
        AIassistant.objects.create(arxiv_id='r', query='Q', response='As [Lee et al. 2021] shows.', model='gpt-big',
                                   papers='a,b', evidence={'status': 'done', 'claims': claims})
        AIassistant.objects.create(arxiv_id='r', query='Q', response='No tag.', model='gpt-big', papers='a,b',
                                   evidence={'status': 'unavailable'})
        AIassistant.objects.create(arxiv_id='r', query='T', kind='table', model='gpt-big')
        quality = activity.answer_quality(activity.today())
        self.assertEqual({k: quality['total'][k] for k in ('answers', 'checked', 'claims', 'verified', 'not_found', 'general', 'numbers', 'multi', 'tagged')},
                         {'answers': 2, 'checked': '50 %', 'claims': 4, 'verified': '50 %', 'not_found': '25 %',
                          'general': '25 %', 'numbers': '25 %', 'multi': 2, 'tagged': '50 %'})
        self.assertEqual([m['label'] for m in quality['models']], ['gpt-big'])
        staff = self.user('boss')
        staff.is_staff = staff.is_superuser = True
        staff.save()
        self.assertContains(self.client_for(staff).get('/admin/activite/'), 'Fiabilité de l’assistant')

    def test_the_arxiv_relay_takes_several_ids(self):
        with mock.patch('summarizer.views.requests.get') as get:
            get.return_value = mock.Mock(content=b'<feed/>', raise_for_status=lambda: None)
            Client().get('/en/arxiv-api/?id_list=2101.00001v1,hep-th--9901001v1')
        self.assertEqual(get.call_args[1]['params'], {'id_list': '2101.00001v1,hep-th/9901001v1'})
        self.assertEqual(Client().get('/en/arxiv-api/?id_list=2101.00001v1,10.1/x').status_code, 400)


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
class ChatSocketTests(TransactionTestCase):
    async def connect(self, path='/ws/arxivid/abcdef0123456789/en/'):
        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), path)
        comm.scope['user'] = AnonymousUser()
        comm.scope['session'] = SessionStore()
        self.assertTrue((await comm.connect())[0])
        return comm

    async def test_stop_cancels_the_answer(self):
        started = asyncio.Event()

        async def answer(*args, on_delta=None, **kwargs):
            await on_delta('Partial')
            started.set()
            await asyncio.sleep(30)
        comm = await self.connect()
        with mock.patch('summarizer.assistant.answer', answer):
            await comm.send_json_to({'message': 'Q?', 'selectedpapers': ['2101.00001v1']})
            self.assertEqual(await comm.receive_json_from(), {'type': 'chat_delta', 'delta': 'Partial'})
            await comm.send_json_to({'message': 'Another?', 'selectedpapers': ['2101.00001v1']})
            self.assertEqual(await comm.receive_json_from(), {'type': 'chat_busy'})
            await comm.send_json_to({'stop': True})
            self.assertEqual(await comm.receive_json_from(), {'type': 'chat_stopped'})
        await comm.disconnect()

    async def test_a_table_sends_its_rows_then_the_whole_table(self):
        async def build(arxiv_ids, columns, lang, paid, user=None, hashed_ip=None, conversation_id=None,
                        on_sources=None, on_row=None):
            self.assertEqual((arxiv_ids, columns, conversation_id), (['2101.00001v1', 'hep-th--9901001v1'], ['Data'], 'abcdef0123456789'))
            await on_sources({'papers': [], 'missing': []})
            await on_row(0, {'cells': []})
            return {'columns': columns, 'rows': []}
        comm = await self.connect()
        with mock.patch('summarizer.chat_table.build', build), mock.patch('summarizer.access.has_tool', return_value=True):
            await comm.send_json_to({'table': {'columns': ['Data']}, 'selectedpapers': ['2101.00001v1', 'hep-th/9901001v1']})
            self.assertEqual((await comm.receive_json_from())['type'], 'chat_sources')
            self.assertEqual(await comm.receive_json_from(), {'type': 'table_row', 'index': 0, 'row': {'cells': []}})
            self.assertEqual((await comm.receive_json_from())['type'], 'table_done')
            for invalid in ({'table': {'columns': []}}, {'table': 'Data'}, {'table': {'columns': ['x'] * 9}}):
                await comm.send_json_to(dict(invalid, selectedpapers=['2101.00001v1']))
                self.assertIn('Invalid message', (await comm.receive_json_from())['message'])
        await comm.disconnect()

    async def test_a_table_needs_a_question_of_quota_per_paper(self):
        comm = await self.connect()
        with mock.patch('summarizer.access.chat_quota_left', return_value=1), \
             mock.patch('summarizer.access.has_tool', return_value=True), \
             mock.patch('summarizer.chat_table.build') as build:
            await comm.send_json_to({'table': {'columns': ['Data']}, 'selectedpapers': ['2101.00001v1', '2101.00002v1']})
            reply = await comm.receive_json_from()
        self.assertEqual(reply['type'], 'table_error')
        self.assertIn('daily question limit', reply['message'])
        build.assert_not_called()
        await comm.disconnect()

    async def test_the_table_is_premium(self):
        comm = await self.connect()
        with mock.patch('summarizer.chat_table.build') as build:
            await comm.send_json_to({'table': {'columns': ['Data']}, 'selectedpapers': ['2101.00001v1', '2101.00002v1']})
            self.assertEqual(await comm.receive_json_from(), {'type': 'table_error', 'premium': True})
        build.assert_not_called()
        await comm.disconnect()

    async def test_a_long_reading_keeps_the_socket_alive(self):
        async def slow(self, paper_id):
            await asyncio.sleep(0.25)
            return 'http://creativecommons.org/licenses/by/4.0/'
        comm = await self.connect('/ws/emb/')
        with mock.patch('summarizer.consumers.KEEPALIVE_SECONDS', 0.1), \
             mock.patch('summarizer.consumers.EmbeddingConsumer.create_embeddings', slow):
            await comm.send_json_to({'paper_id': '2101.00001v1'})
            self.assertEqual(await comm.receive_json_from(), {'message': 'reading', 'paper_id': '2101.00001v1'})
            self.assertEqual(await comm.receive_json_from(), {'message': 'reading', 'paper_id': '2101.00001v1'})
            self.assertEqual((await comm.receive_json_from())['message'], 'completed')
        await comm.disconnect()

    async def test_a_paper_that_cannot_be_read_is_reported(self):
        comm = await self.connect('/ws/emb/')
        with mock.patch('summarizer.utils.has_paper_text', side_effect=RuntimeError('down')):
            await comm.send_json_to({'paper_id': '2101.00001v1'})
            self.assertEqual(await comm.receive_json_from(), {'message': 'failed', 'paper_id': '2101.00001v1'})
            await comm.send_json_to({'paper_id': '<script>'})
            self.assertTrue(await comm.receive_nothing())
        await comm.disconnect()
