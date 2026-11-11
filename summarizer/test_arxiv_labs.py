"""The card of the SummarizePaper toggle on arXiv's abstract pages: /api/arxiv-labs/<arXiv id>."""
import json

from django.test import Client, TestCase

from .models import ArxivPaper, Generated, SummaryPaper

READING = {'tldr': 'Objects change by 5 percent.', 'question': 'Do the objects change?', 'method': 'A sample of 120.',
           'findings': [{'text': 'A change of 5 percent.', 'kind': 'result', 'quote': 'We observe a change.'},
                        {'text': 'It may be general.', 'kind': 'interpretation', 'quote': ''},
                        {'text': 'Third.', 'kind': 'result', 'quote': ''}, {'text': 'Fourth.', 'kind': 'result', 'quote': ''},
                        {'text': 'Fifth.', 'kind': 'result', 'quote': ''}],
           'limitations': [{'text': 'One sample.', 'quote': ''}, {'text': 'Short time.', 'quote': ''}],
           'significance': 'Larger samples.', 'abstract_only': False, 'schema_version': 1}


class ArxivLabsCardTests(TestCase):
    def setUp(self):
        self.client = Client()

    def get(self, reference):
        response = self.client.get('/api/arxiv-labs/' + reference)
        return response, json.loads(response.content)

    def test_reading_with_question_findings_and_cors(self):
        ArxivPaper.objects.create(arxiv_id='2401.12345v1', title='A paper')
        Generated.objects.create(kind='insight', key='2401.12345v1', lang='en', status='done', content=json.dumps(READING))
        response, data = self.get('2401.12345')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Access-Control-Allow-Origin'], '*')
        self.assertNotIn('Set-Cookie', response)
        self.assertEqual(data['kind'], 'reading')
        self.assertEqual(data['question'], 'Do the objects change?')
        self.assertEqual(len(data['findings']), 4)
        self.assertEqual(data['findings'][0], {'text': 'A change of 5 percent.', 'kind': 'result'})  # no quotes
        self.assertEqual(data['limitations'], ['One sample.'])
        self.assertEqual(data['url'], 'https://www.summarizepaper.com/en/paper/2401.12345v1/')

    def test_newest_version_with_content_and_old_style_ids(self):
        for arxiv_id in ('hep-th--9901001v1', 'hep-th--9901001v2', 'hep-th--9901001v10'):
            ArxivPaper.objects.create(arxiv_id=arxiv_id)
        Generated.objects.create(kind='insight', key='hep-th--9901001v2', lang='en', status='done', content=json.dumps(READING))
        Generated.objects.create(kind='insight', key='hep-th--9901001v10', lang='fr', status='done', content=json.dumps(READING))
        response, data = self.get('hep-th/9901001')
        self.assertEqual(data['arxiv_id'], 'hep-th/9901001v2')
        self.assertTrue(data['url'].endswith('/en/paper/hep-th--9901001v2/'))

    def test_summary_opening_and_key_points_when_no_reading(self):
        paper = ArxivPaper.objects.create(arxiv_id='2401.00001v2')
        long_summary = 'First sentence of the summary. ' * 40
        SummaryPaper.objects.create(paper=paper, lang='en', summary=long_summary,
                                    notes=repr(['- **Method:** a survey', r"Result: Lov\'asz bound $4(2q+3)^8$", 'c', 'd', 'e', 'The first sentence of the summary.']))
        response, data = self.get('2401.00001v2')
        self.assertEqual(data['kind'], 'summary')
        self.assertLessEqual(len(data['summary']), 500)
        self.assertTrue(data['summary'].endswith('.'))
        self.assertEqual(data['key_points'][:2], ['Method: a survey', 'Result: Lovász bound 4(2q+3)⁸'])
        self.assertEqual(len(data['key_points']), 4)

    def test_unknown_paper_gives_the_link_only(self):
        response, data = self.get('2402.99999v3')
        self.assertEqual(data, {'available': False, 'arxiv_id': '2402.99999v3',
                                'url': 'https://www.summarizepaper.com/en/paper/2402.99999v3/'})
        response, data = self.get('2402.99999')
        self.assertTrue(data['url'].endswith('/en/paper/2402.99999v1/'))

    def test_not_an_arxiv_id(self):
        response = self.client.get('/api/arxiv-labs/not-a-paper', follow=True)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response['Access-Control-Allow-Origin'], '*')
