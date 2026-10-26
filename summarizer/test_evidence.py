"""The sources of an assistant answer: claims are shown with a passage only when it really is in the paper."""
import asyncio
import json
from unittest import mock

from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import AnonymousUser
from django.contrib.sessions.backends.db import SessionStore
from django.test import TestCase, TransactionTestCase, override_settings

from summarizer import assistant, evidence
from summarizer.models import AIassistant, ArxivPaper, PaperText
from summarizer.routing import websocket_urlpatterns

TEXT = ('## Abstract\n\nWe study lenses.\n\n## 3 Results\n\n### 3.1 Counts\n\n'
        'An earlier paragraph about the survey design and its cadence.\n\n'
        'We find 170000 galaxy-galaxy lenses in the ten-year dataset, with $\\mathcal{O}(10^{5})$ expected.\n\n'
        'Figure 1: \nForecasted number of galaxy–galaxy lenses to be discovered by the Rubin LSST.\n\n'
        '## 4 Conclusion\n\nRubin will transform strong lensing science.')


def paper(text=TEXT, source='html', arxiv_id='2406.08919v3'):
    return {'arxiv_id': arxiv_id, 'label': 'arXiv: ' + arxiv_id, 'text': text, 'source': source, 'block': ''}


class VerifyTests(TestCase):
    def claim(self, supplied=None, **item):
        item.setdefault('claim', 'Rubin should find many lenses.')
        return evidence.verify({'claims': [item]}, supplied or [paper()])[0]

    def test_a_passage_of_the_paper_is_verified_and_located(self):
        claim = self.claim(basis='paper', paper='arXiv: 2406.08919v3',
                           quote='We find 170000 galaxy-galaxy lenses in the ten-year dataset')
        self.assertEqual(claim['status'], 'verified')
        self.assertEqual(claim['section'], '3.1 Counts')
        self.assertEqual(claim['paragraph'], 2)
        self.assertEqual(claim['figure'], '')
        self.assertTrue(claim['link'].startswith('https://arxiv.org/html/2406.08919v3#:~:text=We%20find%20170000'))
        self.assertEqual(claim['numbers_absent'], [])
        self.assertFalse(claim['gapped'])

    def test_an_invented_passage_is_not_verified(self):
        claim = self.claim(basis='paper', quote='We find 250000 galaxy-galaxy lenses in the ten-year dataset')
        self.assertEqual(claim['status'], 'not_found')
        self.assertEqual(claim['quote'], '')
        self.assertEqual(claim['link'], '')

    def test_only_numbers_the_paper_never_states_are_flagged(self):
        quote = 'Rubin will transform strong lensing science.'
        # 170000 comes from another passage than the quote: the paper does state it
        self.assertEqual(self.claim(claim='Rubin finds 170000 lenses.', basis='paper', quote=quote)['numbers_absent'], [])
        self.assertEqual(self.claim(claim='Rubin finds 250000 lenses.', basis='paper', quote=quote)['numbers_absent'], ['250000'])
        self.assertEqual(self.claim(claim='About 250000 lenses.', basis='unsupported')['numbers_absent'], ['250000'])
        self.assertEqual(self.claim(claim='About 250000 lenses.', basis='general')['numbers_absent'], [])
        # a rounded value is stated
        rounded = paper('## Results\n\nBlock 2 explained 40.73% of the total variance in the sample.')
        self.assertEqual(self.claim([rounded], claim='It explains 40.7% (about 41%).', basis='general', figure='')['numbers_absent'], [])
        self.assertEqual(self.claim([rounded], claim='It explains 40.8%.', basis='unsupported')['numbers_absent'], ['40.8'])

    def test_numbers_written_the_french_way_match_the_paper(self):
        claim = self.claim(claim='Ils trouvent 170 000 lentilles (figure 1, section 3.1).', basis='paper',
                           quote='We find 170000 galaxy-galaxy lenses in the ten-year dataset')
        self.assertEqual(claim['numbers_absent'], [])
        self.assertEqual(evidence.numbers('un taux de 0,5 % sur 12,345,678 objets'), {'0.5', '12345678'})
        self.assertEqual(evidence.numbers('We find 170,000 lenses (Fig. 2, Eq. (4)).'), {'170000'})

    def test_a_caption_names_its_figure(self):
        claim = self.claim(basis='paper', quote='Forecasted number of galaxy–galaxy lenses to be discovered by the Rubin LSST.')
        self.assertEqual(claim['figure'], 'Figure 1')
        self.assertIsNone(claim['paragraph'])

    def test_a_figure_is_named_only_when_the_paper_numbers_it(self):
        self.assertEqual(self.claim(basis='general', figure='Fig. 1')['figure'], 'Figure 1')
        self.assertEqual(self.claim(basis='general', figure='Figure 7')['figure'], '')

    def test_quotes_with_doubled_backslashes_from_the_json_still_match(self):
        claim = self.claim(basis='paper', quote='lenses in the ten-year dataset, with $\\\\mathcal{O}(10^{5})$ expected.')
        self.assertEqual(claim['status'], 'verified')
        # the page renders equations, so the link stops before the first one
        self.assertTrue(claim['link'].endswith('#:~:text=lenses%20in%20the%20ten%2Dyear%20dataset%2C%20with'))
        self.assertNotIn('mathcal', claim['link'])

    def test_general_knowledge_and_unsupported_claims_are_kept_and_labelled(self):
        self.assertEqual(self.claim(basis='general', quote='')['status'], 'general')
        self.assertEqual(self.claim(basis='unsupported')['status'], 'unsupported')
        self.assertEqual(self.claim(basis='made up')['status'], 'unsupported')

    def test_a_quote_attributed_to_the_wrong_paper_is_found_in_the_right_one(self):
        other = paper('## Intro\n\nA different study of exoplanet atmospheres and their clouds.', arxiv_id='2101.00001v1')
        claim = self.claim([other, paper()], basis='paper', paper='arXiv: 2101.00001v1',
                           quote='Rubin will transform strong lensing science.')
        self.assertEqual(claim['status'], 'verified')
        self.assertEqual(claim['paper'], 'arXiv: 2406.08919v3')
        self.assertEqual(claim['section'], '4 Conclusion')

    def test_abstract_only_papers_locate_passages_in_the_abstract_without_link(self):
        claim = self.claim([paper('We observe water clouds on a hot Jupiter.', source='abstract')],
                           basis='paper', quote='We observe water clouds on a hot Jupiter.')
        self.assertEqual((claim['status'], claim['section'], claim['link']), ('verified', 'Abstract', ''))

    def test_pdf_text_gives_no_paragraph_number(self):
        claim = self.claim([paper(source='pdf')], basis='paper', quote='Rubin will transform strong lensing science.')
        self.assertIsNone(claim['paragraph'])
        self.assertEqual(claim['link'], '')

    def test_typography_changed_by_a_pdf_still_matches(self):
        text = 'We apply a con-\nvolutional ﬁlter to the “clean” sample—as in earlier work.\n\nOther text follows here.'
        claim = self.claim([paper(text, source='pdf')], basis='paper',
                           quote='We apply a convolutional filter to the "clean" sample-as in earlier work.')
        self.assertEqual(claim['status'], 'verified')
        self.assertEqual(claim['quote'], 'We apply a con- volutional ﬁlter to the “clean” sample—as in earlier work.')
        # a short passage is accepted, a phrase too short to locate anything is not
        self.assertEqual(self.claim([paper(text, source='pdf')], basis='paper', quote='Other text follows here.')['status'], 'verified')
        self.assertEqual(self.claim([paper(text, source='pdf')], basis='paper', quote='Other text')['status'], 'not_found')

    def test_a_hyphen_followed_by_a_space_in_the_text_matches(self):
        text = 'Children were tested with the WIAT- III composite and an open- loop protocol in all schools.'
        claim = self.claim([paper(text, source='pdf')], basis='paper',
                           quote='tested with the WIAT-III composite and an open-loop protocol')
        self.assertEqual(claim['status'], 'verified')

    def test_a_passage_interrupted_by_the_pdf_is_found_and_said_so(self):
        text = ('## Results\n\nFirst, perfect range Fig. 11. Downtown Toronto trajectory and 5G base stations. '
                'and angle measurements are supplied to all ﬁlters. Position- ing errors arising from this test '
                'are attributed to linearization errors. Block 2 explained 40.73% of the total variance, '
                'F(11, 10 139) = 8.68, p < .001.')
        found = self.claim([paper(text, source='pdf')], basis='paper',
                           quote='First, perfect range and angle measurements are supplied to all filters. Positioning '
                                 'errors arising from this test are attributed to linearization errors.')
        self.assertEqual((found['status'], found['gapped'], found['section']), ('verified', True, 'Results'))
        self.assertTrue(found['quote'].startswith('First, perfect range Fig. 11.'))
        page_number = self.claim([paper(text, source='pdf')], basis='paper',
                                 quote='Block 2 explained 40.73% of the total variance, F(11, 139) = 8.68, p < .001.')
        self.assertEqual((page_number['status'], page_number['gapped']), ('verified', True))

    def test_an_interrupted_passage_with_a_changed_number_or_words_is_not_found(self):
        text = ('## Results\n\nBlock 2 explained 40.73% of the total variance, adjusted for age and school, '
                'F(11, 10 139) = 8.68, p < .001.')
        for quote in ('Block 2 explained 45.73% of the total variance, adjusted for age and school, F(11, 139) = 8.68, p < .001.',
                      'Block 2 explained 40.73% of the variance in reading, unlike age or class, F(11, 139) = 8.68, p < .001.'):
            self.assertEqual(self.claim([paper(text, source='pdf')], basis='paper', quote=quote)['status'], 'not_found')

    def test_anchors_are_kept_only_when_the_answer_contains_them(self):
        answer = 'Rubin finds **170 000** lenses. It will *transform* strong lensing.'
        claims = evidence.verify({'claims': [{'claim': 'a', 'basis': 'general', 'anchor': 'finds 170 000 lenses.'},
                                             {'claim': 'b', 'basis': 'general', 'anchor': 'words the answer never wrote'},
                                             {'claim': 'c', 'basis': 'general', 'anchor': 'lensing'},
                                             {'claim': 'd', 'basis': 'general', 'anchor': 'with $x$ inside'}]},
                                 [paper()], answer)
        self.assertEqual([c['anchor'] for c in claims], ['finds 170 000 lenses', '', '', ''])

    def test_malformed_answers_give_no_claims(self):
        for answer in (None, [], {'claims': 'x'}, {'claims': [None, {'claim': ''}, 'text']}):
            self.assertEqual(evidence.verify(answer, [paper()]), [])
        many = {'claims': [{'claim': 'Claim %d.' % i, 'basis': 'general'} for i in range(20)]}
        self.assertEqual(len(evidence.verify(many, [paper()])), evidence.MAX_CLAIMS)


class CheckAnswerTests(TransactionTestCase):
    def setUp(self):
        ArxivPaper.objects.create(arxiv_id='2406.08919v3', title='Lenses', abstract='We study lenses.',
                                  license='https://creativecommons.org/licenses/by/4.0/')
        PaperText.objects.create(arxiv_id='2406.08919v3', text=TEXT, source='html')

    def test_the_check_reuses_the_answer_messages_and_stores_the_claims(self):
        calls = []

        async def complete(messages, paid, max_tokens, on_delta=None, task='assistant', user_id=None, json_object=False):
            calls.append({'messages': messages, 'task': task, 'json_object': json_object})
            if task == 'evidence':
                return json.dumps({'claims': [{'claim': 'They expect many lenses.', 'basis': 'paper', 'anchor': 'transform **lensing**',
                                               'quote': 'Rubin will transform strong lensing science.'}]})
            return 'Rubin will transform lensing.'

        checks = []
        with mock.patch.object(assistant, 'complete', complete):
            text = asyncio.run(assistant.answer(['2406.08919v3'], 'fr', 'What is new?', False, hashed_ip='a',
                                                on_evidence=lambda answer_id, check: checks.append(check)))
            self.assertEqual(text, 'Rubin will transform lensing.')
            self.assertEqual(len(calls), 1)  # the check waits until the answer has been shown
            result = asyncio.run(checks[0])
        answer_call, check_call = calls
        # same prefix as the answer: the paper text is read again at the cached rate
        self.assertEqual(check_call['messages'][:len(answer_call['messages'])], answer_call['messages'])
        self.assertEqual(check_call['messages'][-2], {'role': 'assistant', 'content': text})
        self.assertEqual(check_call['messages'][-1]['role'], 'system')
        self.assertIn('in French', check_call['messages'][-1]['content'])
        self.assertTrue(check_call['json_object'])
        self.assertEqual(result['status'], 'done')
        self.assertEqual(result['claims'][0]['section'], '4 Conclusion')
        self.assertEqual(result['claims'][0]['anchor'], 'transform lensing')
        self.assertEqual(AIassistant.objects.get().claims, result['claims'])

    def test_a_failed_check_is_unavailable_and_keeps_the_answer(self):
        async def complete(messages, paid, max_tokens, on_delta=None, task='assistant', user_id=None, json_object=False):
            return 'not json' if task == 'evidence' else 'An answer.'

        checks = []
        with mock.patch.object(assistant, 'complete', complete):
            asyncio.run(assistant.answer(['2406.08919v3'], 'en', 'Q?', False, hashed_ip='a',
                                         on_evidence=lambda answer_id, check: checks.append(check)))
            self.assertEqual(asyncio.run(checks[0]), {'status': 'unavailable'})
        row = AIassistant.objects.get()
        self.assertEqual((row.response, row.evidence), ('An answer.', None))

    def test_a_long_paper_is_shortened_from_its_own_text_never_replaced_by_our_summary(self):
        context, supplied = assistant.build_sources(['2406.08919v3'])
        self.assertEqual((supplied[0]['text'], supplied[0]['coverage'], supplied[0]['kind']), (TEXT, 100, 'full'))
        self.assertIn('Text supplied: the full text', context)
        with mock.patch.object(assistant, 'MAX_CONTEXT_CHARS', 2000 + 200):
            context, supplied = assistant.build_sources(['2406.08919v3'])
        text = supplied[0]['text']
        self.assertLessEqual(len(text), 200)
        self.assertLess(supplied[0]['coverage'], 100)
        self.assertEqual(supplied[0]['left_out'], ['shortened'])
        # every section keeps its start, the conclusion (short) all of it, and cuts are marked
        self.assertIn('## 4 Conclusion\n\nRubin will transform strong lensing science.', text)
        self.assertIn(assistant.OMITTED, text)
        self.assertIn('Text supplied: about %d%% of the text' % supplied[0]['coverage'], context)
        self.assertNotIn('summary', context.lower())


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
class ChatEvidenceTransportTests(TransactionTestCase):
    async def test_the_answer_comes_first_then_its_sources(self):
        claims = {'status': 'done', 'claims': [{'claim': 'A claim.', 'status': 'general'}]}

        release = asyncio.Event()

        async def answer(*args, on_evidence=None, **kwargs):
            async def check():
                await release.wait()
                return claims
            on_evidence(len(asked), check())
            asked.append(args[2])
            return 'Answer %d.' % len(asked)

        asked = []

        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), '/ws/arxivid/2406.08919v3/en/')
        comm.scope['user'] = AnonymousUser()
        comm.scope['session'] = SessionStore()
        self.assertTrue((await comm.connect())[0])
        with mock.patch('summarizer.assistant.answer', answer):
            await comm.send_json_to({'message': 'What is new?'})
            self.assertEqual(await comm.receive_json_from(), {'message': 'Answer 1.', 'evidence': 'checking', 'answer_id': 0})
            # the next question is answered while the first answer is still being checked
            await comm.send_json_to({'message': 'And the method?'})
            self.assertEqual(await comm.receive_json_from(), {'message': 'Answer 2.', 'evidence': 'checking', 'answer_id': 1})
            release.set()
            results = [await comm.receive_json_from(), await comm.receive_json_from()]
            self.assertEqual(sorted(r['answer_id'] for r in results), [0, 1])
            self.assertEqual(results[0], {'type': 'chat_evidence', 'answer_id': results[0]['answer_id'], **claims})
        await comm.disconnect()
