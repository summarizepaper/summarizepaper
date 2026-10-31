"""Tests of the papers around a paper: the neighbourhood (our index and Semantic Scholar), the sentences saying
why each paper is there, the paper page and the map, and the selection sent to the chat or to a collection."""
import datetime
import json
from unittest import mock

import numpy as np
from django.test import Client, TestCase
from django.utils import timezone, translation

from . import neighbours, similar
from .models import Collection, Generated, NeighbourFeedback, PaperScore, SummaryPaper
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
                              ('summarizer.neighbours.s2_request', mock.Mock(return_value=None)),
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
        self.assertContains(page, 'href="/en/paper/2101.00002v2/" data-funnel')
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
            written = chat.call_count
            again = Client().post(url).json()
        self.assertIn(first['status'], ('pending', 'done'))
        self.assertEqual(again['status'], 'done')
        self.assertEqual(chat.call_count, written)  # written once for every reader
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


class SignalTests(NeighbourhoodTestCase):
    def test_prior_work_used_in_depth_and_much_cited_follow_ups_come_first(self):
        # similarities to the paper: 0.80, then 0.74 but used in depth and for its method (+0.08)
        refs = [{'citedPaper': s2_paper('Galaxy disk planet', arxiv='1501.00001', year=2015), 'isInfluential': False},
                {'citedPaper': s2_paper('Galaxy galaxy disk planet planet', arxiv='1501.00002', year=2015), 'isInfluential': True,
                 'intents': ['methodology']},
                {'citedPaper': dict(s2_paper('Star star', arxiv='1501.00003'), abstract=None)}]
        cites = [{'citingPaper': dict(s2_paper('Galaxy planet', arxiv='2301.00001'), citationCount=3)},
                 {'citingPaper': dict(s2_paper('Galaxy planet', arxiv='2301.00002'), citationCount=None)}]
        most_cited = [{'ref': '2302.00009', 'title': 'Planet galaxy galaxy', 'authors': 'A B', 'date': '2023-01-01', 'venue': 'Nature',
                       'abstract': 'planet galaxy galaxy', 'citations': 5000, 'status': 'published', 'influential': False,
                       'method': False, 'cited': False, 'cites': True, 'source': 'openalex'}]

        def s2(url, params):
            if 'recommendations' in url:
                return {'recommendedPapers': []}
            return {'data': refs if '/references' in url else cites}
        with mock.patch('summarizer.neighbours.s2_get', side_effect=s2), \
             mock.patch('summarizer.neighbours.LINKED', 2), \
             mock.patch('summarizer.neighbours.openalex_most_cited', return_value=most_cited) as openalex, \
             mock.patch('summarizer.neighbours.CLOSE_KEPT', 0):
            data = neighbours.compute(self.main)
        openalex.assert_called_once()  # a full page of citations: there are more, the most cited come from OpenAlex
        prior = [i for i in data['items'] if i['group'] == 'prior']
        self.assertEqual(prior[0]['ref'], '1501.00002v1')  # used in depth and for its method
        self.assertTrue(prior[0]['influential'] and prior[0]['method'])
        title_only = next(i for i in prior if i['ref'] == '1501.00003v1')
        self.assertTrue(title_only['title_only'])
        followups = [i for i in data['items'] if i['group'] == 'followup']
        self.assertEqual(followups[0]['ref'], '2302.00009v1')
        self.assertEqual(followups[0]['citations'], 5000)
        self.assertEqual(data['counts']['most_cited'], 1)

    def test_status_of_a_venue(self):
        self.assertEqual(neighbours.status('arXiv.org', 'ArXiv'), 'preprint')
        self.assertEqual(neighbours.status('bioRxiv'), 'preprint')
        self.assertEqual(neighbours.status('Annual Meeting of the Association for Computational Linguistics'), 'published')
        self.assertEqual(neighbours.status(''), '')

    def test_openalex_most_cited_reads_the_citing_works(self):
        work = {'id': 'https://openalex.org/W1', 'doi': 'https://doi.org/10.1234/x', 'display_name': 'Galaxy disk', 'type': 'article',
                'authorships': [{'author': {'display_name': 'Eve Ng'}}], 'publication_date': '2024-02-03',
                'primary_location': {'source': {'display_name': 'ApJ'}}, 'cited_by_count': 812, 'locations': []}
        answers = [{'id': 'https://openalex.org/W9'}, {'results': [work]}]
        with mock.patch('summarizer.sources._json', side_effect=answers) as get:
            found = neighbours.openalex_most_cited(self.main)
        self.assertIn('cites:W9', str(get.call_args_list[1]))
        self.assertEqual(found[0]['ref'], 'doi_10.1234--x')
        self.assertEqual((found[0]['citations'], found[0]['status'], found[0]['date']), (812, 'published', '2024-02-03'))
        with mock.patch('summarizer.neighbours.OPENALEX_PER_DAY', 0), mock.patch('summarizer.sources._json') as get:
            self.assertEqual(neighbours.openalex_most_cited(self.main), [])
        get.assert_not_called()

    def test_clusters_group_alike_papers(self):
        a, b = np.array([1, 0, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32)
        near = lambda v, e: (v + e) / np.linalg.norm(v + e)
        vectors = [near(a, np.array([0, 0, 0.1 * i], dtype=np.float32)) for i in range(3)] + \
                  [near(b, np.array([0, 0, 0.1 * i], dtype=np.float32)) for i in range(4)] + [None]
        self.assertEqual(neighbours.clusters(vectors), [[3, 4, 5, 6], [0, 1, 2]])

    def test_sub_topics_are_named_with_the_sentences_and_drawn_on_the_map(self):
        data = neighbours.store(self.main.arxiv_id)
        Generated.objects.filter(kind=neighbours.KIND).update(content=json.dumps(dict(data, clusters=[[0, 1, 2]])))

        def answer(*args, **kwargs):
            self.assertIn('"groups"', args[0])
            return {'neighbours': [], 'groups': [{'g': 1, 'name': 'Galaxy disks'}, {'g': 7, 'name': 'Nothing'}]}
        with mock.patch('summarizer.generated.chat_json', side_effect=answer):
            Client().post('/en/neighbours/2101.00001v1/why/')
        page = Client().get('/en/tree/2101.00001v1/')
        self.assertEqual(page.context['map_data']['topics'], [{'members': [0, 1, 2], 'name': 'Galaxy disks'}])


class MarkTests(NeighbourhoodTestCase):
    def first_close(self):
        return neighbours.for_page(self.main, 'en')['groups'][0]['items'][0]['ref']

    def test_a_reader_hides_a_paper_and_takes_it_back(self):
        neighbours.store(self.main.arxiv_id)
        ref = self.first_close()
        client = self.client_for(self.user())
        url = '/en/neighbours/2101.00001v1/mark/'
        self.assertEqual(client.post(url, {'ref': ref}).json()['status'], 'hidden')
        client.post(url, {'ref': ref})  # twice is once
        self.assertEqual(NeighbourFeedback.objects.count(), 1)
        page = client.get('/en/paper/2101.00001v1/')
        self.assertNotContains(page, 'data-ref="%s"' % ref)
        self.assertContains(Client().get('/en/paper/2101.00001v1/'), 'data-ref="%s"' % ref)  # others still see it
        client.post(url, {'ref': ref, 'undo': '1'})
        self.assertContains(client.get('/en/paper/2101.00001v1/'), 'data-ref="%s"' % ref)

    def test_three_readers_hide_it_for_everyone(self):
        neighbours.store(self.main.arxiv_id)
        ref = self.first_close()
        for i in range(3):
            Client(REMOTE_ADDR='10.0.0.%d' % i).post('/en/neighbours/2101.00001v1/mark/', {'ref': ref})
        self.assertNotEqual(self.first_close(), ref)
        self.assertNotContains(Client().get('/en/paper/2101.00001v1/'), 'data-ref="%s"' % ref)

    def test_visitors_without_account_are_limited(self):
        with mock.patch('summarizer.neighbours.MARKS_PER_DAY', 1):
            client = Client()
            self.assertEqual(client.post('/en/neighbours/2101.00001v1/mark/', {'ref': 'a'}).status_code, 200)
            self.assertEqual(client.post('/en/neighbours/2101.00001v1/mark/', {'ref': 'b'}).status_code, 429)


class DigestCollectionTests(NeighbourhoodTestCase):
    def test_a_new_paper_close_to_a_collected_one_says_so(self):
        from . import digest
        from .models import CollectionPaper, DigestSubscription
        user = self.user()
        collection = Collection.objects.create(user=user, name='Disks')
        CollectionPaper.objects.create(collection=collection, arxiv_id='2101.00003v1', title='Planet planet')
        sub = DigestSubscription.objects.create(user=user, categories='astro-ph.EP')
        ranking = digest.ranking_for(sub, paid=False)
        planet, galaxy = fake_embed(['planet', 'galaxy'])
        self.assertGreater(ranking.score(planet)[0], ranking.score(galaxy)[0])
        closest = ranking.score(planet)[1]
        self.assertEqual(closest['kind'], 'collection')
        entry = {'closest': closest, 'paper': {'categories': ['astro-ph.EP']}, 'primary': True}
        with translation.override('en'):
            self.assertEqual(digest.reasons(entry), ['Close to “Planet planet” in your collection “Disks”'])
            self.assertEqual(digest.reasons(entry, private=False), ['Close to a paper of your collections'])
        with translation.override('fr'):
            self.assertEqual(digest.reasons(entry), ['Proche de « Planet planet » de votre collection « Disks »'])


class SemanticScholarPacingTests(TestCase):
    def test_calls_are_paced_and_a_429_is_tried_again(self):
        answers = [mock.Mock(status_code=429), mock.Mock(status_code=200)]
        neighbours._last_s2[0] = 0.0
        with mock.patch('summarizer.neighbours.requests.request', side_effect=answers) as call, \
             mock.patch('summarizer.neighbours.time.monotonic', side_effect=[100.0, 100.0, 103.2, 103.2]), \
             mock.patch('summarizer.neighbours.time.sleep') as sleep:
            response = neighbours.s2_request('GET', 'https://example.org/x')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(sleep.call_args_list[0].args, (1.5,))  # the wait after the 429; the pace was already kept


class WhyRoundsTests(NeighbourhoodTestCase):
    def test_the_neighbours_the_model_skipped_are_asked_again(self):
        data = neighbours.store(self.main.arxiv_id)
        Generated.objects.filter(kind=neighbours.KIND).update(content=json.dumps(dict(data, clusters=[[0, 1, 2]])))
        answers = [{'neighbours': [{'n': 1, 'relation': 'same_question', 'why': 'First.'}], 'groups': []},
                   {'neighbours': [{'n': 2, 'relation': 'review', 'why': 'Second.'}], 'groups': [{'g': 1, 'name': 'Disks'}]},
                   {'neighbours': [{'n': n, 'relation': 'context', 'why': 'Later.'} for n in range(3, 20)], 'groups': []}]
        with mock.patch('summarizer.generated.chat_json', side_effect=answers) as chat:
            result = neighbours.why_work(self.main.arxiv_id, 'en')(0)
        self.assertEqual(chat.call_count, 3)
        second = json.loads(chat.call_args_list[1].args[0])
        self.assertNotIn(1, [n['n'] for n in second['neighbours']])  # only what is missing
        self.assertEqual(len(result['by_ref']), len(result['refs']))
        self.assertEqual(result['clusters'][0]['name'], 'Disks')

    def test_sentences_written_for_too_few_papers_are_written_again_the_next_day(self):
        data = neighbours.store(self.main.arxiv_id)
        items = neighbours.shown(data)
        row = Generated.objects.create(kind=neighbours.WHY_KIND, key=self.main.arxiv_id, lang='en', status='done',
                                       content=json.dumps({'refs': [i['ref'] for i in items],
                                                           'by_ref': {items[0]['ref']: {'relation': '', 'why': 'x'}}}))
        self.assertTrue(neighbours.covers(row, items))  # not again the same day
        Generated.objects.filter(pk=row.pk).update(updated=timezone.now() - datetime.timedelta(days=2))
        row.refresh_from_db()
        self.assertFalse(neighbours.covers(row, items))
