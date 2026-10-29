"""Tests of collections as a workspace: selection, versions, imports, notes and tags, the comparison table with
checked sources and its updates, columns, checks of the papers, suggestions, sharing and watching."""
import datetime
import json
from unittest import mock

import numpy as np
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client
from django.utils import timezone

from . import changes, collection_tools, comparison, library, publication_status
from .models import (ArxivPaper, CitationCache, Collection, CollectionPaper, FollowedPaper, Generated, PaperEmbedding,
                     PaperVersion, ZoteroAccount)
from .test_tools import ToolsTestCase

OPEN_TEXT = ('## Abstract\n\nWe observe clouds.\n\n## 1 Introduction\n\n' + 'Context. ' * 300 +
             '\n\n## 2 Methods\n\nWe used the JWST NIRSpec instrument on 12 nights. The spectra were fitted with a '
             'retrieval code.\n\n## 3 Results\n\nThe clouds lie at a pressure of 10 mbar with an uncertainty of 2 mbar. '
             'They cover most of the dayside.\n\n## 4 Conclusions\n\nClouds are thick.')


def fake_model(answers=None):
    """A chat_json that fills the table from the papers it is given, and compares them."""
    answers = answers or {}
    calls = []

    def chat(prompt, paid=True, max_tokens=0, system=None, **kw):
        calls.append({'prompt': prompt, 'system': system})
        if 'fill these columns' in prompt:
            ids = [line.split('"')[1] for line in system.splitlines() if line.startswith('<paper id=')]
            return {'rows': [{'id': i, 'cells': answers.get(i, {'method': {'text': 'Spectroscopy', 'quote': ''}})}
                             for i in ids]}
        return {'agreements': [{'text': 'Both see clouds', 'papers': ids_of(system), 'columns': ['method', 'results']}],
                'disagreements': [{'text': 'Cloud height differs', 'papers': ['made-up']}], 'gaps': ['Nobody looks at night']}
    chat.calls = calls
    return chat


def ids_of(system):
    return [row['id'] for row in json.loads(system)]


class CollectionsTestCase(ToolsTestCase):
    def make_collection(self, user, ids=('2101.00001v1', '2101.00002v1'), **fields):
        collection = Collection.objects.create(user=user, name='Clouds', **fields)
        for aid in ids:
            if not ArxivPaper.objects.filter(arxiv_id=aid).exists():
                self.paper(aid, title='Paper ' + aid)
            library.add_paper(collection, aid, True)
        return collection


class SelectionAndVersionTests(CollectionsTestCase):
    def test_selection_decides_what_the_analyses_and_the_chat_read(self):
        user = self.user(paid=True)
        collection = self.make_collection(user, ('2101.00001v1', '2101.00002v1', '2101.00003v1'))
        client = self.client_for(user)
        r = client.post('/en/collections/%d/' % collection.pk, {'action': 'paper', 'arxiv_id': '2101.00002v1', 'selected': '0'},
                        HTTP_X_REQUESTED_WITH='fetch').json()
        self.assertEqual(r, {'status': 'saved', 'tags': '', 'selected': False, 'analysed': 2})
        self.assertEqual([p.arxiv_id for p in comparison.analysed(collection)], ['2101.00001v1', '2101.00003v1'])
        page = client.get('/en/collections/%d/' % collection.pk)
        self.assertContains(page, '/en/chat/?ids=2101.00001v1,2101.00003v1')
        self.assertContains(page, '2 of 3 papers selected')
        french = client.get('/fr/collections/%d/' % collection.pk)
        self.assertContains(french, '2 articles sur 3 sélectionnés')
        self.assertContains(french, 'Tout sélectionner')
        client.post('/en/collections/%d/' % collection.pk, {'action': 'select', 'value': 'none'})
        self.assertEqual(client.post('/en/collections/%d/compare/' % collection.pk).json()['status'], 'too_few')
        self.assertEqual(client.post('/en/collections/%d/questions/' % collection.pk).json()['status'], 'too_few')
        CollectionPaper.objects.filter(arxiv_id='2101.00003v1').update(tags='core')
        client.post('/en/collections/%d/' % collection.pk, {'action': 'select', 'value': 'tag:Core'})
        self.assertEqual([p.arxiv_id for p in comparison.analysed(collection)], ['2101.00003v1'])

    def test_a_paper_is_kept_once_whatever_its_version(self):
        user = self.user()
        collection = self.make_collection(user, ('2101.00001v1',))
        self.paper('2101.00001v2', title='Clouds, second version')
        self.assertEqual(library.add_paper(collection, '2101.00001v2', False), 'updated')
        self.assertEqual(collection.papers.get().title, 'Clouds, second version')
        self.assertEqual(library.add_paper(collection, '2101.00001v1', False), 'already')
        self.assertEqual(list(collection.papers.values_list('arxiv_id', flat=True)), ['2101.00001v2'])
        self.assertTrue(library.paper_collections(user, '2101.00001v1')[0]['has'])
        r = self.client_for(user).post('/en/collections/add/', {'arxiv_id': '2101.00001v2', 'collection': collection.pk}).json()
        self.assertEqual(r['status'], 'already')


class ImportTests(CollectionsTestCase):
    BIB = '''@article{lee2021,
  title={Clouds}, author={Lee, Ann},
  eprint={2101.00001v1}, archivePrefix={arXiv}
}
@article{ray2020, title={Galaxies}, doi={10.1093/mnras/stad1234}, journal={MNRAS}}
@book{old1950, title={A book without identifier}}
@comment{jabref-meta: ignored}
'''
    RIS = 'TY  - JOUR\nTI  - Clouds\nAN  - arXiv:2101.00002\nER  - \n\nTY  - JOUR\nTI  - X\nDO  - 10.1000/xyz.1\nER  - \n'

    def test_bibtex_ris_and_plain_lists_give_their_identifiers(self):
        self.assertEqual(library.references_in(self.BIB), (['2101.00001v1', '10.1093/mnras/stad1234'], 1))
        self.assertEqual(library.references_in(self.RIS), (['2101.00002', '10.1000/xyz.1'], 0))
        self.assertEqual(library.references_in('2101.00001, https://arxiv.org/abs/2101.00002v3\nfoo'),
                         (['2101.00001', 'https://arxiv.org/abs/2101.00002v3', 'foo'], 0))
        ids, skipped = library.wanted_ids([self.BIB, '2101.00001v1 nonsense'])
        self.assertEqual(ids, ['2101.00001v1', 'doi_10.1093--mnras--stad1234'])
        self.assertEqual(skipped, 2)

    def test_uploaded_file_is_added_and_a_long_list_goes_to_the_background(self):
        user = self.user()
        collection = Collection.objects.create(user=user, name='Import')
        self.paper()
        client = self.client_for(user)
        known = {'2101.00001v1': ('Clouds on a hot planet', 'Ann Lee'), 'doi_10.1093--mnras--stad1234': ('Galaxies', 'Bob Ray')}
        with mock.patch('summarizer.library.lookup_many', side_effect=lambda ids: {i: known[i] for i in ids if i in known}):
            r = client.post('/en/collections/%d/' % collection.pk, {
                'action': 'add', 'ids': '', 'file': SimpleUploadedFile('refs.bib', self.BIB.encode())})
            self.assertIn('added=2', r['Location'])
            self.assertIn('skipped=1', r['Location'])
            page = client.get(r['Location'])
            self.assertContains(page, '2 papers added.')
            self.assertContains(page, '1 entry has no arXiv identifier or DOI.')
            many = ' '.join('2102.%05d' % i for i in range(library.SYNC_ADD + 1))
            r = client.post('/en/collections/%d/' % collection.pk, {'action': 'add', 'ids': many})
        self.assertIn('job=', r['Location'])
        job = Generated.objects.get(kind='collection_import')
        self.assertEqual(job.status, 'done')
        self.assertEqual(job.data()['unknown'], library.SYNC_ADD + 1)
        self.assertEqual(collection.papers.count(), 2)

    def test_zotero_collection_is_imported(self):
        user = self.user(paid=True)
        ZoteroAccount.objects.create(user=user, library_id='123', api_key='k')
        collection = Collection.objects.create(user=user, name='Z')
        self.paper()
        items = [{'data': {'itemType': 'preprint', 'archiveID': 'arXiv:2101.00001'}},
                 {'data': {'itemType': 'journalArticle', 'DOI': '10.1093/mnras/stad1234'}},
                 {'data': {'itemType': 'book', 'title': 'No id'}}, {'data': {'itemType': 'note'}}]
        self.assertEqual(collection_tools.zotero_ref({'extra': 'arXiv: 2101.00009v2'}), '2101.00009v2')
        with mock.patch('summarizer.collection_tools.zotero_get', return_value=items), \
             mock.patch('summarizer.library.lookup_many', return_value={'2101.00001': ('Clouds', 'Ann')}):
            r = self.client_for(user).post('/en/collections/%d/zotero/' % collection.pk, {'key': 'ABCD1234'}).json()
        data = Generated.objects.get(pk=r['id']).data()
        self.assertEqual((data['added'], data['unknown'], data['skipped']), (1, 1, 1))
        free = self.user('free')
        mine = Collection.objects.create(user=free, name='F')
        self.assertEqual(self.client_for(free).post('/en/collections/%d/zotero/' % mine.pk, {'key': 'ABCD1234'}).status_code, 403)


class NotesAndExportTests(CollectionsTestCase):
    def test_notes_tags_and_status_are_saved_and_exported(self):
        user = self.user()
        collection = self.make_collection(user)
        client = self.client_for(user)
        url = '/en/collections/%d/' % collection.pk
        r = client.post(url, {'action': 'paper', 'arxiv_id': '2101.00001v1', 'tags': ' core, Core ,method ',
                              'reading': 'key', 'note': 'Use in chapter 2'}, HTTP_X_REQUESTED_WITH='fetch').json()
        self.assertEqual(r['tags'], 'core, method')
        paper = collection.papers.get(arxiv_id='2101.00001v1')
        self.assertEqual((paper.reading, paper.note), ('key', 'Use in chapter 2'))
        client.post(url, {'action': 'paper', 'arxiv_id': '2101.00001v1', 'reading': 'bogus'}, HTTP_X_REQUESTED_WITH='fetch')
        paper.refresh_from_db()
        self.assertEqual(paper.reading, 'key')
        bib = client.get(url + 'export/bib/').content.decode()
        self.assertIn('keywords={core, method}', bib)
        self.assertIn('annote={Use in chapter 2}', bib)
        self.assertTrue(bib.count('@misc'), 2)
        ris = client.get(url + 'export/ris/').content.decode()
        self.assertIn('KW  - core\nKW  - method\nN1  - Use in chapter 2\nER  - ', ris)
        md = client.get(url + 'export/md/?tag=method').content.decode()
        self.assertIn('## My note\n\nUse in chapter 2', md)
        self.assertNotIn('Paper 2101.00002v1', md)
        page = client.get(url)
        self.assertContains(page, 'data-tags="core, method"')
        self.assertContains(page, '<option value="key" selected>')
        self.assertEqual(client.post(url, {'action': 'remove', 'arxiv_id': '2101.00002v1'},
                                     HTTP_X_REQUESTED_WITH='fetch').json()['status'], 'removed')
        self.assertEqual(collection.papers.count(), 1)


class ComparisonTableTests(CollectionsTestCase):
    def setUp(self):
        super().setUp()
        self.owner = self.user(paid=True)
        self.client = self.client_for(self.owner)
        self.paper()  # CC BY: read in its key sections
        self.paper('2101.00002v1', title='Closed paper', summary='Our summary says the clouds are high.')
        ArxivPaper.objects.filter(arxiv_id='2101.00002v1').update(license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/')
        self.collection = self.make_collection(self.owner)
        self.url = '/en/collections/%d/' % self.collection.pk

    def answers(self):
        return {
            '2101.00001v1': {
                'method': {'text': 'JWST NIRSpec spectra over 12 nights', 'quote': 'We used the JWST NIRSpec instrument on 12 nights.'},
                'results': {'text': 'Clouds at 10 mbar (± 3)', 'quote': 'The clouds lie at a pressure of 10 mbar with an uncertainty of 2 mbar.'},
                'data': {'text': 'Twelve spectra', 'quote': 'An invented sentence that the paper never says.'},
                'question': {'text': 'not stated', 'quote': 'We observe clouds.'},
            },
            '2101.00002v1': {'results': {'text': 'Clouds are high', 'quote': 'Our summary says the clouds are high.'}},
        }

    def test_cells_keep_only_quotes_found_in_what_the_model_read(self):
        chat = fake_model(self.answers())
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=chat):
            self.client.post(self.url + 'compare/')
        system = chat.calls[0]['system']
        self.assertIn('## 2 Methods', system)  # the open paper: its key sections
        self.assertNotIn('Context. Context. Context. Context. Context. Context. Context. Context. Context.', system)
        self.assertIn('<our_summary>\nOur summary says the clouds are high.', system)  # the closed one: context only
        rows = {r['arxiv_id']: r for r in Generated.objects.get(kind='compare').data()['rows']}
        cells = rows['2101.00001v1']['cells']
        self.assertEqual(cells['method']['quote'], 'We used the JWST NIRSpec instrument on 12 nights.')
        self.assertEqual(cells['method']['section'], '2 Methods')
        self.assertEqual(cells['results']['numbers'], ['3'])  # '± 3' is not what the passage says
        self.assertEqual(cells['data']['quote'], '')  # invented: dropped
        self.assertEqual(cells['question'], dict(comparison.BLANK_CELL, text='Not stated'))
        self.assertEqual(rows['2101.00002v1']['cells']['results']['quote'], '')  # never from our summary
        self.assertEqual(rows['2101.00002v1']['source'], 'abstract')
        page = self.client.get(self.url)
        self.assertContains(page, '2 of 4 filled cells are backed by a passage of the paper.')
        self.assertContains(page, '1 of 2 papers were read in their key sections')
        self.assertContains(page, 'Not in this passage: 3')
        self.assertContains(page, 'Not checked against the paper')
        self.assertContains(page, '<a href="#row-1">1</a>, <a href="#row-2">2</a>')
        self.assertContains(page, 'Nobody looks at night')
        self.assertNotContains(page, 'no longer matches')
        csv = self.client.get(self.url + 'export/csv/').content.decode()
        self.assertIn('"""We used the JWST NIRSpec instrument on 12 nights."" (2 Methods)"', csv)

    def test_new_papers_make_the_table_out_of_date_and_an_update_reads_only_them(self):
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=fake_model(self.answers())):
            self.client.post(self.url + 'compare/')
        self.paper('2101.00003v1', title='Third paper')
        library.add_paper(self.collection, '2101.00003v1', True)
        page = self.client.get(self.url)
        self.assertContains(page, 'This table no longer matches the selected papers (+1, −0).')
        self.assertContains(page, 'Clouds on a hot planet')  # the old table stays meanwhile
        chat = fake_model()
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=chat):
            self.client.post(self.url + 'compare/', {'mode': 'update'})
        extracted = [c for c in chat.calls if 'fill these columns' in c['prompt']]
        self.assertEqual(len(extracted), 1)
        self.assertIn('<paper id="2101.00003v1">', extracted[0]['system'])
        self.assertNotIn('<paper id="2101.00001v1">', extracted[0]['system'])
        data = Generated.objects.get(kind='compare').data()
        self.assertEqual(data['ids'], ['2101.00001v1', '2101.00002v1', '2101.00003v1'])
        self.assertEqual(data['rows'][0]['cells']['method']['quote'], 'We used the JWST NIRSpec instrument on 12 nights.')
        self.assertNotContains(self.client.get(self.url), 'no longer matches')
        chat = fake_model()
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=chat):
            self.client.post(self.url + 'compare/', {'mode': 'again'})
        self.assertEqual(sum(c['system'].count('<paper id=') for c in chat.calls if 'fill these columns' in c['prompt']), 3)

    def test_columns_come_from_a_template_and_the_users_own(self):
        self.client.post(self.url, {'action': 'columns', 'template': 'ml', 'custom': 'Sample size: number of objects\nH0'})
        self.collection.refresh_from_db()
        codes = [c for c, _l, _h in comparison.columns(self.collection)]
        self.assertEqual(codes, ['task', 'datasets', 'model', 'scores', 'baselines', 'limitations', 'c1', 'c2'])
        Generated.objects.create(kind='compare', key='%d:abcdef' % self.collection.pk, lang='en', status='done',
                                 content=json.dumps({'rows': [{'arxiv_id': '2101.00001v1', 'title': 'Old', 'method': 'Old method'}]}))
        page = self.client.get(self.url)
        self.assertContains(page, 'Old method')  # an older table stays, marked out of date
        self.assertContains(page, 'This table no longer matches the selected papers')
        self.assertContains(page, 'write it again to see them')
        chat = fake_model()
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=chat):
            self.client.post(self.url + 'compare/')
        self.assertIn('"c1" (Sample size): number of objects', chat.calls[0]['prompt'])
        self.assertIn('"c2" (H0): H0', chat.calls[0]['prompt'])
        self.assertIn('"scores" (Metrics and scores)', chat.calls[0]['prompt'])
        page = self.client.get(self.url)
        self.assertContains(page, '<th scope="col">Sample size</th>')
        self.assertNotContains(page, 'no longer matches')
        self.client.post(self.url, {'action': 'columns', 'template': 'custom', 'custom': ''})
        self.collection.refresh_from_db()
        self.assertEqual(comparison.columns(self.collection), comparison.TEMPLATES['general'])
        self.assertContains(self.client.get(self.url), 'This table was written with other columns.')

    def test_free_users_cannot_compare(self):
        free = self.user('free')
        mine = self.make_collection(free)
        self.assertEqual(self.client_for(free).post('/en/collections/%d/compare/' % mine.pk).status_code, 403)
        self.assertEqual(self.client_for(free).get('/en/collections/%d/export/csv/' % mine.pk).status_code, 404)


class PaperFactsTests(CollectionsTestCase):
    def test_collection_uses_publication_year_and_keeps_preprint_fallback(self):
        collection = self.make_collection(self.user())
        paper = ArxivPaper.objects.get(arxiv_id='2101.00001v1')
        paper.journal_ref = 'ApJL 930, L12, 2022 May 12'
        paper.published_arxiv = datetime.date(2023, 11, 15)
        paper.save()
        with mock.patch('summarizer.publication_status.latest', side_effect=AssertionError('Reports are already loaded')):
            facts = collection_tools.facts(list(collection.papers.all()))
        self.assertEqual(facts[paper.arxiv_id]['year'], '2022')
        self.assertEqual(facts['2101.00002v1']['year'], '2021')

    def test_status_retraction_citations_and_newer_version_are_shown_from_stored_data(self):
        user = self.user()
        collection = self.make_collection(user, ('2101.00001v2', '2101.00002v1'))
        report = {'status': 'published', 'journal': 'Nature', 'checked_at': timezone.now().isoformat(),
                  'updates': [{'type': 'retraction', 'doi': '10.1038/retracted'}]}
        Generated.objects.create(kind=publication_status.KIND, key='2101.00001v2', lang='en', status='done', content=json.dumps(report))
        CitationCache.objects.create(arxiv_id='2101.00001', total=12, checked=timezone.now())
        PaperVersion.objects.create(arxiv_id='2101.00001', version=3, checked=timezone.now())
        self.paper('2101.00002v2', title='Second, newer')
        page = self.client_for(user).get('/en/collections/%d/' % collection.pk)
        self.assertContains(page, 'Retracted')
        self.assertContains(page, 'href="https://doi.org/10.1038/retracted"')
        self.assertContains(page, ' · Nature')
        self.assertContains(page, '12 citations')
        self.assertContains(page, 'Version 3 is out')
        self.assertContains(page, '/en/changes/2101.00002v1/?to=2')
        self.assertContains(page, 'data-cited="12"')

    def test_crossref_notices_are_read(self):
        work = {'updated-by': [{'type': 'retraction', 'DOI': '10.1038/R1'}, {'type': 'Correction', 'DOI': '10.1038/C2'},
                               {'type': 'new_version'}, None]}
        self.assertEqual(publication_status.notices(work), [{'type': 'retraction', 'doi': '10.1038/r1'},
                                                            {'type': 'correction', 'doi': '10.1038/c2'}])

    def test_refresh_checks_the_papers_without_touching_version_alerts(self):
        user = self.user(paid=True)
        collection = self.make_collection(user)
        PaperVersion.objects.create(arxiv_id='2101.00001', version=1, checked=timezone.now())
        feed = [{'arxiv_id': '2101.00001v4'}, {'arxiv_id': '2101.00002v1'}]
        with mock.patch('summarizer.publication_status.for_generation') as status, \
             mock.patch('summarizer.citations.refresh') as refresh, \
             mock.patch('summarizer.collection_tools.time.sleep'), \
             mock.patch('summarizer.digest.fetch_ids', return_value=feed):
            r = self.client_for(user).post('/en/collections/%d/refresh/' % collection.pk).json()
        self.assertEqual(status.call_count, 2)
        self.assertEqual(sorted(c.args[0] for c in refresh.call_args_list), ['2101.00001', '2101.00002'])
        self.assertEqual(Generated.objects.get(pk=r['id']).data()['versions'], {'2101.00001': 4, '2101.00002': 1})
        self.assertEqual(PaperVersion.objects.get(arxiv_id='2101.00001').version, 1)
        self.assertContains(self.client_for(user).get('/en/collections/%d/' % collection.pk), 'Version 4 is out')
        free = self.user('free')
        self.assertEqual(self.client_for(free).post('/en/collections/%d/refresh/' % self.make_collection(free).pk).status_code, 403)


class SuggestionTests(CollectionsTestCase):
    def test_common_references_then_the_closest_papers(self):
        user = self.user(paid=True)
        collection = self.make_collection(user, ('2101.00001v1', '2101.00002v1'))
        refs = {'2101.00001v1': [{'ref': '1999.00001', 'title': 'Classic', 'year': 1999, 'authors': 'Old'},
                                 {'ref': '2101.00002', 'title': 'Already in', 'year': 2021, 'authors': ''},
                                 {'ref': '10.1/once', 'title': 'Once', 'year': 2000, 'authors': ''},
                                 {'ref': '1202.3665', 'title': 'emcee: The MCMC Hammer', 'year': 2013, 'authors': ''}],
                '2101.00002v1': [{'ref': '1999.00001v2', 'title': 'Classic', 'year': 1999, 'authors': 'Old'},
                                 {'ref': '1202.3665', 'title': 'emcee: The MCMC Hammer', 'year': 2013, 'authors': ''}]}
        for aid, vector in (('2101.00001v1', [1, 0, 0]), ('2101.00002v1', [0.8, 0.6, 0]), ('2201.00009v1', [0.9, 0.4, 0.1]),
                            ('2201.00010v1', [0, 0, 1])):
            if not ArxivPaper.objects.filter(arxiv_id=aid).exists():
                self.paper(aid, title='Paper ' + aid)
            vec = np.array(vector, dtype=np.float32)
            PaperEmbedding.objects.create(arxiv_id=aid, category='astro-ph.EP', model='text-embedding-3-small',
                                          vector=(vec / np.linalg.norm(vec)).tobytes())
        with mock.patch('summarizer.chat_page.fetch_references', side_effect=lambda aid: refs[aid]), \
             mock.patch('summarizer.generated.chat_json', return_value={'tools': [1]}) as chat, \
             mock.patch('summarizer.collection_tools.time.sleep'):
            r = self.client_for(user).post('/en/collections/%d/suggest/' % collection.pk).json()
        self.assertIn('1. emcee: The MCMC Hammer\n2. Classic', chat.call_args.args[0])
        data = Generated.objects.get(pk=r['id']).data()
        self.assertEqual([(c['ref'], c['count']) for c in data['cited']], [('1999.00001', 2)])
        self.assertEqual([c['title'] for c in data['tools']], ['emcee: The MCMC Hammer'])
        self.assertEqual([c['ref'] for c in data['close']], ['2201.00009v1', '2201.00010v1'])
        page = self.client_for(user).get('/en/collections/%d/' % collection.pk)
        self.assertContains(page, 'cited by 2 papers of the collection')
        self.assertContains(page, '<input type="hidden" name="ids" value="1999.00001">')
        self.assertContains(page, '1 software, data set or method they share')


class ShareAndWatchTests(CollectionsTestCase):
    def test_shared_link_shows_the_collection_without_its_owner_and_can_be_copied(self):
        owner = self.user()
        collection = self.make_collection(owner)
        CollectionPaper.objects.filter(arxiv_id='2101.00001v1').update(note='Private thought', tags='core')
        client = self.client_for(owner)
        client.post('/en/collections/%d/' % collection.pk, {'action': 'share'})
        collection.refresh_from_db()
        url = '/en/collections/shared/%s/' % collection.share_token
        self.assertContains(client.get('/en/collections/%d/' % collection.pk), 'http://testserver' + url)
        page = Client().get(url)
        self.assertContains(page, 'Paper 2101.00002v1')
        self.assertNotContains(page, 'Private thought')
        self.assertNotContains(page, owner.username)
        self.assertNotContains(page, owner.email)
        client.post('/en/collections/%d/' % collection.pk, {'action': 'share', 'notes': 'on'})
        self.assertContains(Client().get(url), 'Private thought')
        reader = self.user('reader')
        r = self.client_for(reader).post(url + 'copy/')
        copy = Collection.objects.get(user=reader)
        self.assertEqual(r['Location'], '/en/collections/%d/' % copy.pk)
        self.assertEqual(sorted(copy.papers.values_list('arxiv_id', flat=True)), ['2101.00001v1', '2101.00002v1'])
        self.assertEqual(copy.papers.get(arxiv_id='2101.00001v1').note, '')
        Collection.objects.filter(pk=collection.pk).update(share_expires=timezone.now() - datetime.timedelta(days=1))
        self.assertEqual(Client().get(url).status_code, 404)
        self.assertContains(client.get('/en/collections/%d/' % collection.pk), 'Expired link.')
        client.post('/en/collections/%d/' % collection.pk, {'action': 'unshare'})
        self.assertEqual(Client().get(url).status_code, 404)
        self.assertEqual(Client().get('/en/collections/shared/x/').status_code, 404)

    def test_a_watched_collection_follows_its_papers_and_those_added_later(self):
        user = self.user(paid=True)
        collection = self.make_collection(user, ('2101.00001v1',))
        client = self.client_for(user)
        with mock.patch('summarizer.citations.refresh_new') as refresh:
            client.post('/en/collections/%d/' % collection.pk, {'action': 'watch', 'value': 'on'})
            self.assertEqual(list(FollowedPaper.objects.filter(user=user).values_list('arxiv_id', flat=True)), ['2101.00001'])
            refresh.assert_not_called()  # the hourly run checks them first, at its own pace
            self.paper('2101.00002v1')
            client.post('/en/collections/%d/' % collection.pk, {'action': 'add', 'ids': '2101.00002v1'})
        self.assertIn(user.pk, changes.watched()['2101.00002'])
        self.assertContains(client.get('/en/collections/%d/' % collection.pk), '2 papers are followed.')
        free = self.user('free')
        mine = self.make_collection(free, ('2101.00001v1',))
        self.client_for(free).post('/en/collections/%d/' % mine.pk, {'action': 'watch', 'value': 'on'})
        mine.refresh_from_db()
        self.assertFalse(mine.watch)


class WorkspaceTests(CollectionsTestCase):
    def test_a_preprint_and_its_published_version_are_one_work(self):
        user = self.user()
        collection = self.make_collection(user, ('2101.00001v1',))
        ArxivPaper.objects.filter(arxiv_id='2101.00001v1').update(link_doi='https://doi.org/10.1093/mnras/stad1234')
        key = 'doi_10.1093--mnras--stad1234'
        self.assertEqual(library.identity('https://doi.org/10.48550/arXiv.2101.00001v3'), 'arxiv:2101.00001')
        self.assertEqual(library.identity(key), 'doi:10.1093/mnras/stad1234')
        with mock.patch('summarizer.library.lookup_many', return_value={key: ('Clouds', 'Ann Lee')}):
            r = self.client_for(user).post('/en/collections/%d/' % collection.pk, {'action': 'add', 'ids': '10.1093/mnras/stad1234'})
        self.assertIn('duplicate=1', r['Location'])
        self.assertContains(self.client_for(user).get(r['Location']), 'was already there as its preprint or published version')
        self.assertEqual(collection.papers.count(), 1)
        CollectionPaper.objects.create(collection=collection, arxiv_id=key, title='Clouds (MNRAS)', authors='Ann Lee')
        groups = library.duplicates(list(collection.papers.all()))
        self.assertEqual([sorted(p.arxiv_id for p in g) for g in groups], [['2101.00001v1', key]])
        self.assertContains(self.client_for(user).get('/en/collections/%d/' % collection.pk), 'look like one work')

    def test_the_owner_orders_the_papers_and_the_analyses_follow(self):
        user = self.user()
        collection = self.make_collection(user, ('2101.00001v1', '2101.00002v1', '2101.00003v1'))
        client = self.client_for(user)
        url = '/en/collections/%d/' % collection.pk
        client.post(url, {'action': 'down', 'arxiv_id': '2101.00001v1'})
        self.assertEqual(list(collection.papers.values_list('arxiv_id', flat=True)), ['2101.00002v1', '2101.00001v1', '2101.00003v1'])
        r = client.post(url, {'action': 'order', 'ids': '2101.00003v1,2101.00001v1'}, HTTP_X_REQUESTED_WITH='fetch').json()
        self.assertEqual(r['status'], 'moved')
        self.assertEqual([p.arxiv_id for p in comparison.analysed(collection)], ['2101.00003v1', '2101.00001v1', '2101.00002v1'])
        self.paper('2101.00004v1')
        library.add_paper(collection, '2101.00004v1', False)
        self.assertEqual(list(collection.papers.values_list('arxiv_id', flat=True))[-1], '2101.00004v1')
        self.assertContains(client.get(url), '/en/chat/?ids=2101.00003v1,2101.00001v1,2101.00002v1,2101.00004v1')

    def test_the_paper_list_exports_as_a_spreadsheet(self):
        user = self.user()
        collection = self.make_collection(user)
        CollectionPaper.objects.filter(arxiv_id='2101.00001v1').update(tags='core', reading='key', note='Cite in ch. 2')
        CitationCache.objects.create(arxiv_id='2101.00001', total=7, checked=timezone.now())
        csv = self.client_for(user).get('/en/collections/%d/export/list/' % collection.pk).content.decode()
        self.assertIn('Identifier,Title,Authors,Year,Venue', csv)
        self.assertIn('2101.00001,Paper 2101.00001v1,"Ann Lee, Bob Ray",2021,,,,7,,Key paper,core,Cite in ch. 2,yes,https://arxiv.org/abs/2101.00001', csv)

    def test_literature_review_collections_keep_one_paper_per_work(self):
        user = self.user()
        collection = Collection.objects.create(user=user, name='Review')
        self.paper()
        known = {'2101.00001v1': ('Clouds', 'Ann'), '2101.00001v2': ('Clouds v2', 'Ann')}
        for key in known:
            library.add_paper(collection, key, False, known)
        self.assertEqual(list(collection.papers.values_list('arxiv_id', flat=True)), ['2101.00001v2'])


class CorrectionAndFindingTests(ComparisonTableTests):
    def write(self, mode=''):
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=fake_model(self.answers())):
            self.client.post(self.url + 'compare/', {'mode': mode})
        return Generated.objects.get(kind='compare')

    def test_a_corrected_cell_is_marked_and_survives_a_rewrite(self):
        row = self.write()
        r = self.client.post(self.url + 'cell/', {'row': row.pk, 'arxiv_id': '2101.00001v1', 'code': 'data',
                                                  'text': '12 NIRSpec spectra, 2.9-5.2 micron'})
        self.assertEqual(r['Location'], self.url + '#row-1')
        page = self.client.get(self.url)
        self.assertContains(page, 'Corrected by hand')
        self.assertContains(page, '12 NIRSpec spectra, 2.9-5.2 micron')
        row = self.write('again')
        cell = row.data()['rows'][0]['cells']['data']
        self.assertEqual((cell['text'], cell['edited']), ('12 NIRSpec spectra, 2.9-5.2 micron', True))
        other = self.user('eve', paid=True)
        mine = self.make_collection(other)
        self.assertEqual(self.client_for(other).post('/en/collections/%d/cell/' % mine.pk, {
            'row': row.pk, 'arxiv_id': '2101.00001v1', 'code': 'data', 'text': 'x'}).status_code, 404)
        self.assertNotContains(Client().get(self._shared_url()), 'name="code"')

    def _shared_url(self):
        library.share(self.collection, False)
        return '/en/collections/shared/%s/' % self.collection.share_token

    def test_findings_show_the_passages_they_rest_on(self):
        self.write()
        table = comparison.display(Generated.objects.get(kind='compare').data(), 'en')
        agreement = table['agreements'][0]
        self.assertEqual(agreement['papers'], [1, 2])
        self.assertEqual([(p['n'], p['label']) for p in agreement['passages']], [(1, 'Method'), (1, 'Main results')])
        self.assertFalse(agreement['backed'])  # paper 2 has no quoted cell in these columns
        self.assertIsNone(table['disagreements'][0]['backed'])  # names no column (and its made-up paper is dropped)
        page = self.client.get(self.url)
        self.assertContains(page, '2 passages behind it')
        self.assertContains(page, 'Some of the papers it names have no passage in these columns.')


class BoundedSuggestionTests(SuggestionTests):
    def test_common_references_then_the_closest_papers(self):
        with mock.patch('summarizer.collection_tools.EMBEDDING_CHUNK', 1):
            super().test_common_references_then_the_closest_papers()


class EditTogetherTests(CollectionsTestCase):
    def test_an_invitation_lets_others_edit_but_not_own(self):
        owner = self.user('owner', paid=True)
        collection = self.make_collection(owner)
        mine = self.client_for(owner)
        url = '/en/collections/%d/' % collection.pk
        mine.post(url, {'action': 'invite'})
        collection.refresh_from_db()
        join = '/en/collections/join/%s/' % collection.invite_token
        self.assertContains(mine.get(url), 'http://testserver' + join)
        editor = self.user('editor', paid=True)
        theirs = self.client_for(editor)
        self.assertEqual(theirs.get(url).status_code, 404)
        self.assertContains(theirs.get(join), 'Join this collection')
        self.assertEqual(theirs.post(join)['Location'], url)
        page = theirs.get(url)
        self.assertContains(page, 'You edit this collection with owner.')
        self.assertNotContains(page, 'name="action" value="delete"')
        theirs.post(url, {'action': 'paper', 'arxiv_id': '2101.00001v1', 'note': 'Seen by both'}, HTTP_X_REQUESTED_WITH='fetch')
        self.assertContains(mine.get(url), 'Seen by both')
        for action in ('delete', 'rename', 'share', 'invite', 'watch'):
            self.assertEqual(theirs.post(url, {'action': action, 'name': 'x', 'value': 'on'}).status_code, 404)
        self.assertTrue(Collection.objects.filter(pk=collection.pk, name='Clouds').exists())
        self.assertContains(theirs.get('/en/collections/'), 'Edited with others')
        self.assertTrue(library.paper_collections(editor, '2101.00001v1')[0]['has'])
        with mock.patch('summarizer.paper_source.paper_text', return_value=OPEN_TEXT), \
             mock.patch('summarizer.generated.chat_json', side_effect=fake_model()):
            started = theirs.post(url + 'compare/').json()
        self.assertEqual(mine.get('/en/ai/%d/' % started['id']).json()['status'], 'done')  # started by the editor
        stranger = self.user('stranger', paid=True)
        self.assertEqual(self.client_for(stranger).get('/en/ai/%d/' % started['id']).status_code, 404)
        mine.post(url, {'action': 'uninvite'})
        self.assertEqual(self.client_for(stranger).get(join).status_code, 404)
        mine.post(url, {'action': 'drop', 'user': editor.pk})
        self.assertEqual(theirs.get(url).status_code, 404)


class PublicationAlertTests(CollectionsTestCase):
    def report(self, status, **more):
        return dict({'status': status, 'checked_at': timezone.now().isoformat(), 'journal': '', 'doi': ''}, **more)

    def test_a_followed_paper_published_or_retracted_is_emailed_once(self):
        from django.core import mail
        from . import publication_alerts
        from .models import PublicationAlert
        user = self.user(paid=True)
        free = self.user('free')
        self.paper('2101.00001v1')
        self.paper('2101.00001v2', title='Clouds, v2')
        self.paper('2101.00002v1')
        for u in (user, free):
            FollowedPaper.objects.create(user=u, arxiv_id='2101.00001')
        FollowedPaper.objects.create(user=user, arxiv_id='2101.00002')
        Generated.objects.create(kind=publication_status.KIND, key='2101.00001v2', lang='en', status='done', content=json.dumps(
            dict(self.report('preprint'), checked_at=(timezone.now() - datetime.timedelta(days=2)).isoformat())))
        published = self.report('published', journal='MNRAS', doi='10.1093/mnras/stad1',
                                updates=[{'type': 'correction', 'doi': '10.1093/mnras/corr'}])
        lookups = {'2101.00001v2': published, '2101.00002v1': self.report('published', journal='ApJ')}
        with mock.patch('summarizer.publication_status.lookup', side_effect=lambda p: lookups[p.arxiv_id]), \
             mock.patch('summarizer.publication_alerts.time.sleep'):
            self.assertEqual(publication_alerts.check(timezone.now()), 4)  # published + correction, for two followers
            self.assertEqual(publication_alerts.check(timezone.now()), 0)  # nothing is due any more
        self.assertFalse(PublicationAlert.objects.filter(arxiv_id='2101.00002').exists())  # its first check: no news
        self.assertEqual(publication_alerts.send_alerts(timezone.now(), 10, base_url='https://x'), 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('MNRAS', mail.outbox[0].body)
        self.assertIn('Correction published', mail.outbox[0].body)
        self.assertIn('https://x/en/paper/2101.00001v2/', mail.outbox[0].body)
        self.assertFalse(PublicationAlert.objects.filter(emailed=False).exists())  # the free user's are dropped


class ReferenceWorkTests(CollectionsTestCase):
    def test_a_work_cited_by_its_arxiv_id_and_its_doi_counts_once(self):
        papers = [CollectionPaper(arxiv_id='2101.00001v1', title='A'), CollectionPaper(arxiv_id='2101.00002v1', title='B'),
                  CollectionPaper(arxiv_id='2101.00003v1', title='C'), CollectionPaper(arxiv_id='2101.00004v1', title='D')]
        refs = {'2101.00001v1': [{'ref': '1999.00001', 'title': 'Classic', 'also': ['10.1234/classic']}],
                '2101.00002v1': [{'ref': '10.1234/classic', 'title': 'Classic', 'also': []}],
                '2101.00003v1': [{'ref': '10.1234/CLASSIC', 'title': 'Classic'}, {'ref': '10.1234/own', 'title': 'Own'},
                                 {'ref': '10.1234/err', 'title': 'Erratum: Classic'}],
                '2101.00004v1': [{'ref': '10.1234/err', 'title': 'ERRATUM: “Classic”'}]}
        with mock.patch('summarizer.chat_page.fetch_references', side_effect=lambda aid: refs[aid]), \
             mock.patch('summarizer.collection_tools.time.sleep'):
            found = collection_tools.common_references(papers, {'doi:10.1234/own'})
        self.assertEqual([(c['ref'], c['count'], c['by']) for c in found], [('1999.00001', 3, ['A', 'B', 'C'])])
