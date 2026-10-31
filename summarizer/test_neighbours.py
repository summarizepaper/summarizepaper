"""Tests of the papers around a paper: the neighbourhood (our index and Semantic Scholar), the sentences saying
why each paper is there, the paper page and the map, and the selection sent to the chat or to a collection."""
import datetime
import json
from unittest import mock

import numpy as np
from django.test import Client
from django.utils import timezone

from . import neighbours, similar
from .models import Collection, Generated, PaperScore, SummaryPaper
from .test_tools import ToolsTestCase

AXES = ['galaxy', 'planet', 'disk', 'star']


def fake_embed(texts):
    vectors = np.array([[t.lower().count(a) + 0.01 for a in AXES] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def s2_paper(title, arxiv='', doi='', year=2022):
    ids = {k: v for k, v in (('ArXiv', arxiv), ('DOI', doi)) if v}
    return {'title': title, 'abstract': title + ' abstract', 'year': year, 'publicationDate': '%d-05-01' % year,
            'authors': [{'name': 'Eve Ng'}, {'name': 'Tom Li'}], 'externalIds': ids, 'venue': 'ApJ'}


S2 = {
    'recommendations': {'recommendedPapers': [
        s2_paper('Galaxy disk galaxy', arxiv='2201.00009'),
        s2_paper('Galaxy galaxy disk', arxiv='2101.00002'),  # in our index: counted once
        s2_paper('No identifier at all')]},
    'references': {'data': [
        {'citedPaper': s2_paper('Galaxy planet', doi='10.1234/ABC', year=2015)},
        {'citedPaper': s2_paper('Planet planet', arxiv='2101.00003', year=2016)}]},
    'citations': {'data': [
        {'citingPaper': s2_paper('Galaxy star', arxiv='2301.00005', year=2023)},
        {'citingPaper': s2_paper('Galaxy galaxy disk', arxiv='2101.00001')}]},  # the paper itself
}


def fake_s2(url, params):
    return next(body for key, body in S2.items() if '/%s' % key in url)


class NeighbourhoodTestCase(ToolsTestCase):
    def setUp(self):
        super().setUp()
        for target, value in (('summarizer.similar.embed', mock.Mock(side_effect=fake_embed)),
                              ('summarizer.neighbours.s2_get', mock.Mock(side_effect=fake_s2)),
                              ('summarizer.neighbours.time.sleep', mock.Mock())):
            patcher = mock.patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.main = self.paper('2101.00001v1', title='Galaxy galaxy disk', abstract='galaxy disk')
        self.paper('2101.00002v1', title='Galaxy galaxy disk', abstract='galaxy disk', summary='')
        self.paper('2101.00002v2', title='Galaxy galaxy disk', abstract='galaxy disk')
        self.paper('2101.00003v1', title='Planet planet', abstract='planet', summary='')
        similar.store([p for p in self.main.__class__.objects.all()])


class ComputeTests(NeighbourhoodTestCase):
    def test_index_and_semantic_scholar_on_one_scale_in_three_groups(self):
        with mock.patch('summarizer.neighbours.CLOSE_KEPT', 2):
            data = neighbours.store(self.main.arxiv_id)
        items = {i['ref']: i for i in data['items']}
        refs = [i['ref'] for i in data['items']]
        self.assertEqual(len([r for r in refs if r.startswith('2101.00002')]), 1)  # one version, merged with S2's
        self.assertNotIn('2101.00001v1', refs)  # never the paper itself
        self.assertIn('2201.00009v1', refs)  # not in our tables: its first version's page
        self.assertEqual([i['group'] for i in data['items']][:2], ['close', 'close'])
        self.assertEqual(items['doi_10.1234--abc']['group'], 'prior')
        self.assertTrue(items['doi_10.1234--abc']['cited'])
        self.assertEqual(items['2101.00003v1']['group'], 'prior')  # in our index and cited by the paper
        self.assertEqual(items['2101.00003v1']['source'], 'index')
        self.assertEqual(items['2301.00005v1']['group'], 'followup')
        self.assertTrue(items['2301.00005v1']['cites'])
        self.assertEqual(items['2201.00009v1']['authors'], 'Eve Ng, Tom Li')
        sims = [i['similarity'] for i in data['items'] if i['group'] == 'close']
        self.assertEqual(sims, sorted(sims, reverse=True))
        self.assertTrue(all(len(e) == 3 and e[2] >= neighbours.EDGE_MIN for e in data['edges']))
        self.assertEqual(data['counts']['recommended'], 2)
        self.assertTrue(Generated.objects.filter(kind=neighbours.KIND, key=self.main.arxiv_id, lang='', status='done').exists())

    def test_semantic_scholar_down_leaves_the_index_neighbours(self):
        with mock.patch('summarizer.neighbours.s2_get', return_value=None):
            data = neighbours.store(self.main.arxiv_id)
        self.assertEqual([i['source'] for i in data['items']], ['index', 'index'])

    def test_levels(self):
        self.assertEqual(neighbours.level(0.8)[0], 'very_close')
        self.assertEqual(neighbours.level(0.65)[0], 'close')
        self.assertEqual(neighbours.level(0.3)[0], 'related')

    def test_after_a_synthesis_the_index_now_the_rest_in_the_background(self):
        with mock.patch('summarizer.neighbours.start_why') as why:
            neighbours.after_synthesis(self.main.arxiv_id, 'fr')
        self.assertTrue(PaperScore.objects.filter(from_paper=self.main).exists())
        self.assertIsNotNone(neighbours.stored(self.main.arxiv_id))
        why.assert_called_once_with(self.main.arxiv_id, 'fr')

    def test_hourly_run_fills_the_missing_then_the_oldest(self):
        old = Generated.objects.create(kind=neighbours.KIND, key='2101.00003v1', lang='', status='done', content='{"items": []}')
        Generated.objects.filter(pk=old.pk).update(updated=timezone.now() - datetime.timedelta(days=90))
        SummaryPaper.objects.create(paper=self.main.__class__.objects.get(arxiv_id='2101.00003v1'), lang='en', summary='s')
        with mock.patch('summarizer.neighbours.store') as store:
            done = neighbours.refresh_due(timezone.now())
        self.assertEqual(done, 3)
        self.assertEqual([c.args[0] for c in store.call_args_list], ['2101.00002v2', '2101.00001v1', '2101.00003v1'])


class PageTests(NeighbourhoodTestCase):
    def why_answer(self, *args, **kwargs):
        return {'neighbours': [{'n': 1, 'relation': 'same_question', 'why': 'It studies the same galaxy disks.'},
                               {'n': 2, 'relation': 'made_up', 'why': 'Another sentence.'}]}

    def test_paper_page_lists_the_groups_and_links_the_tools(self):
        neighbours.store(self.main.arxiv_id)
        page = Client().get('/en/paper/2101.00001v1/')
        self.assertContains(page, 'Papers around this one')
        self.assertContains(page, 'Closest papers')
        self.assertContains(page, 'Galaxy disk galaxy')
        # not summarized: its page would read arXiv for every crawler that follows the link
        self.assertContains(page, 'href="/en/paper/2201.00009v1/" rel="nofollow"')
        self.assertContains(page, 'href="/en/paper/2101.00002v2/">')
        self.assertContains(page, 'Cited by this paper')
        self.assertContains(page, 'href="/en/literature-review/?question=')
        self.assertContains(page, 'href="/en/scoops/?title=Galaxy%20galaxy%20disk')
        self.assertContains(page, 'Create a free account to keep them in a collection')
        self.assertContains(page, 'neighbours.js')
        self.assertContains(page, 'data-why-ready="0"')

    def test_older_papers_show_their_index_neighbours_until_computed(self):
        PaperScore.objects.create(from_paper=self.main, to_paper=self.main.__class__.objects.get(arxiv_id='2101.00003v1'), score=0.5)
        page = Client().get('/en/paper/2101.00001v1/')
        self.assertContains(page, 'Planet planet')
        self.assertContains(page, 'Related')
        self.assertContains(page, 'For now only among the papers summarized on this site')

    def test_why_is_written_once_and_shown(self):
        neighbours.store(self.main.arxiv_id)
        url = '/en/neighbours/2101.00001v1/why/'
        with mock.patch('summarizer.generated.chat_json', side_effect=self.why_answer) as chat:
            first = Client().post(url).json()
            again = Client().post(url).json()
        self.assertIn(first['status'], ('pending', 'done'))
        self.assertEqual(again['status'], 'done')
        self.assertEqual(chat.call_count, 1)
        by_ref = again['data']['by_ref']
        first_ref = neighbours.shown(neighbours.stored(self.main.arxiv_id))[0]['ref']
        self.assertEqual(by_ref[first_ref]['relation'], 'same_question')
        self.assertTrue(all(r['relation'] in ('', 'same_question') for r in by_ref.values()))  # an unknown code is dropped
        page = Client().get('/en/paper/2101.00001v1/')
        self.assertContains(page, 'It studies the same galaxy disks.')
        self.assertContains(page, 'data-why-ready="1"')

    def test_why_stops_at_the_daily_budget(self):
        neighbours.store(self.main.arxiv_id)
        with mock.patch('summarizer.neighbours.WHY_PER_DAY', 0):
            self.assertEqual(Client().post('/en/neighbours/2101.00001v1/why/').json()['status'], 'busy')

    def test_map_page(self):
        neighbours.store(self.main.arxiv_id)
        page = Client().get('/en/tree/2101.00001v1/')
        self.assertContains(page, 'id="nb-map-data"')
        self.assertContains(page, 'neighbours-map.js')
        self.assertContains(page, 'Papers around “Galaxy galaxy disk”')
        data = page.context['map_data']
        self.assertEqual(data['main']['date'], '2021-01-04')
        self.assertEqual(len(data['nodes']), len(neighbours.stored(self.main.arxiv_id)['items']))
        self.assertNotContains(page, 'See them on a map')

    def test_selection_goes_to_a_new_collection_with_the_paper(self):
        neighbours.store(self.main.arxiv_id)
        user = self.user()
        client = self.client_for(user)
        page = client.get('/en/paper/2101.00001v1/')
        self.assertContains(page, 'Add to the collection, with this paper')
        answer = client.post('/en/collections/add/', {'ids': '2101.00001v1,2101.00003v1,not an id', 'collection': 'new',
                                                      'name': 'Around: Galaxy'}).json()
        self.assertEqual((answer['status'], answer['added']), ('added', 2))
        collection = Collection.objects.get(user=user)
        self.assertEqual(collection.name, 'Around: Galaxy')
        self.assertEqual(sorted(collection.papers.values_list('arxiv_id', flat=True)), ['2101.00001v1', '2101.00003v1'])
        again = client.post('/en/collections/add/', {'ids': '2101.00003v1', 'collection': collection.pk}).json()
        self.assertEqual(again['status'], 'already')

    def test_french_page(self):
        neighbours.store(self.main.arxiv_id)
        page = Client().get('/fr/paper/2101.00001v1/')
        self.assertContains(page, 'Les articles autour de celui-ci')
        import re
        levels = set(re.findall(r'class="nb-level [^"]*"[^>]*>([^<]*)<', page.content.decode()))
        self.assertTrue(levels and levels <= {'Très proche', 'Proche', 'Apparenté'}, levels)
