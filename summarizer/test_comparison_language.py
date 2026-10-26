import json
from unittest import mock

from . import library
from .models import Collection, Generated
from .test_tools import ToolsTestCase


class ComparisonLanguageTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.paper()
        self.paper('2101.00002v1', 'Second paper')
        self.reader = self.user('comparison-language', paid=True)
        self.paid = self.client_for(self.reader)
        self.collection = Collection.objects.create(user=self.reader, name='Comparaison')
        for key in ['2101.00001v1','2101.00002v1']:
            library.add_paper(self.collection,key,True)
        self.data = {'rows':[{'arxiv_id':'2101.00001v1','title':'Clouds on a hot planet',
            'question':'Question scientifique', 'method':'not stated', 'data':' Not Stated. ',
            'results':'The authors have not stated a causal relationship.', 'limitations':'not stated'}],
            'agreements':[], 'disagreements':[], 'gaps':[]}

    def test_cached_french_table_and_csv_are_localized_without_regeneration_or_data_mutation(self):
        row=Generated.objects.create(kind='compare', key='%s:0123456789abcdef' % self.collection.pk,lang='fr',status='done',content=json.dumps(self.data))
        base='/fr/collections/%s/' % self.collection.pk
        with mock.patch('summarizer.generated.chat_json') as llm:
            page=self.paid.get(base)
            self.assertContains(page,'Non précisé',count=3)
            self.assertContains(page,'The authors have not stated a causal relationship.')
            csv=self.paid.get(base+'export/csv/').content.decode()
            self.assertEqual(csv.count('Non précisé'),3)
            llm.assert_not_called()
        row.refresh_from_db()
        self.assertEqual(row.data(),self.data)
        Generated.objects.create(kind='compare', key='%s:0123456789abcdef' % self.collection.pk,lang='en',status='done',content=json.dumps(self.data))
        self.assertNotContains(self.paid.get('/en/collections/%s/' % self.collection.pk),'Non précisé')

    def test_new_french_generation_requests_and_enforces_localized_missing_values(self):
        answer=dict(self.data,rows=[dict(self.data['rows'][0],id='2101.00001v1')])
        with mock.patch('summarizer.paper_source.paper_text',return_value=''), \
             mock.patch('summarizer.generated.chat_json',return_value=answer) as llm:
            self.paid.post('/fr/collections/%s/compare/' % self.collection.pk)
        self.assertIn('"Non précisé"',llm.call_args_list[0].args[0])
        cells=Generated.objects.get(kind='compare').data()['rows'][0]['cells']
        self.assertEqual(cells['method']['text'],'Non précisé')
        self.assertEqual(cells['data']['text'],'Non précisé')
        self.assertEqual(cells['results']['text'],self.data['rows'][0]['results'])
