from unittest import mock

import requests
from bs4 import BeautifulSoup
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase

from . import sources, unified_search


class UnifiedSearchTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_identifier_recognition_preserves_versions_and_old_ids(self):
        for value, expected in [
            ('1706.03762', '1706.03762v1'), ('arXiv:1706.03762v12', '1706.03762v12'),
            ('https://arxiv.org/abs/1706.03762v2', '1706.03762v2'),
            ('https://arxiv.org/pdf/1706.03762.pdf?download=1', '1706.03762v1'),
            ('https://arxiv.org/html/1706.03762v2', '1706.03762v2'),
            ('hep-th/9901001', 'hep-th--9901001v1'),
            ('https://arxiv.org/abs/hep-th/9901001v3', 'hep-th--9901001v3'),
            ('black holes', None), ('10.1234/example', None),
            ('https://example.org/1706.03762', None),
        ]:
            with self.subTest(value=value):
                self.assertEqual(unified_search.arxiv_identifier(value), expected)

    @mock.patch('summarizer.unified_search.close_old_connections')
    def test_both_sources_are_interleaved_deduplicated_and_cached(self, connections):
        arxiv = [{'key': '1706.03762v2', 'arxiv_id': '1706.03762v2', 'doi_key': 'doi_example'},
                 {'key': '2001.00001v1', 'arxiv_id': '2001.00001v1'}]
        journals = [{'key': 'doi_example', 'arxiv_id': '1706.03762'}, {'key': 'doi_other'}]
        with mock.patch.object(unified_search, 'arxiv_results', return_value=arxiv) as a, \
             mock.patch.object(sources, 'search', return_value=journals) as b:
            results, failed = unified_search.search('attention')
            self.assertEqual([p['key'] for p in results], ['1706.03762v2', '2001.00001v1', 'doi_other'])
            self.assertFalse(failed)
            self.assertEqual(unified_search.search('attention'), (results, False))
            a.assert_called_once_with('attention'); b.assert_called_once_with('attention')

    @mock.patch('summarizer.unified_search.close_old_connections')
    def test_one_failed_source_keeps_the_other_results(self, connections):
        for failing in ('arxiv', 'journals'):
            cache.clear()
            with mock.patch.object(unified_search, 'arxiv_results',
                                   side_effect=requests.Timeout() if failing == 'arxiv' else None,
                                   return_value=[{'key': '1706.03762v1'}]), \
                 mock.patch.object(sources, 'search',
                                   side_effect=sources.SourceUnavailable('offline') if failing == 'journals' else None,
                                   return_value=[{'key': 'doi_other'}]):
                results, failed = unified_search.search('attention')
                self.assertTrue(failed)
                self.assertEqual(len(results), 1)

    @mock.patch('summarizer.unified_search.requests.get')
    def test_arxiv_query_and_feed_mapping(self, get):
        get.return_value.text = '''<feed xmlns="http://www.w3.org/2005/Atom"><entry>
        <id>http://arxiv.org/abs/1706.03762v2</id><title>Attention</title>
        <author><name>A. Author</name></author><published>2017-06-12T00:00:00Z</published>
        </entry></feed>'''
        rows = unified_search.arxiv_results('attention models')
        self.assertEqual(rows[0]['key'], '1706.03762v2')
        self.assertEqual(rows[0]['authors'], ['A. Author'])
        self.assertEqual(get.call_args.kwargs['params']['search_query'], 'all:attention models')
        unified_search.arxiv_results('au:Vaswani')
        self.assertEqual(get.call_args.kwargs['params']['search_query'], 'au:Vaswani')


class UnifiedSearchPageTests(TestCase):
    def test_home_defaults_to_all_sources_and_recognizes_ids_without_network(self):
        with mock.patch('summarizer.views.sources.resolve', return_value=None) as resolve:
            response = self.client.post('/en/', {'arxiv_id': 'black holes'})
            self.assertEqual(response['Location'], '/en/search-results/?q=black+holes&page=1&src=all')
            resolve.reset_mock()
            response = self.client.post('/fr/', {'arxiv_id': 'https://arxiv.org/pdf/1706.03762v2.pdf'})
            self.assertEqual(response['Location'], '/fr/arxiv-id/1706.03762v2/')
            resolve.assert_not_called()
        self.assertNotContains(self.client.get('/en/'), 'name="src"')

    def test_default_results_and_pagination_keep_all_sources(self):
        rows = [{'key': str(i), 'title': 'Paper %d' % i, 'authors': [], 'arxiv_result': True} for i in range(12)]
        with mock.patch.object(unified_search, 'search', return_value=(rows, True)) as search:
            response = self.client.get('/en/search-results/?q=attention')
            self.assertContains(response, 'All sources')
            self.assertContains(response, 'Results may be incomplete')
            self.assertTrue(any('src=all&type=&page=2' in a['href'] for a in
                                BeautifulSoup(response.content, 'html.parser').select('.pagination a')))
            self.assertContains(response, 'Paper 0')
            response = self.client.get('/en/search-results/?q=attention&src=all&page=2')
            self.assertContains(response, 'Paper 11')
            self.assertEqual(search.call_count, 2)

class ScopedSearchTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    @mock.patch('summarizer.unified_search.close_old_connections')
    def test_scopes_have_separate_caches_and_reach_both_providers(self, connections):
        with mock.patch.object(unified_search, 'arxiv_results', return_value=[]) as arxiv, \
             mock.patch.object(sources, 'search', return_value=[]) as journals:
            unified_search.search('Kral', field='author')
            arxiv.assert_called_once_with('Kral', field='author')
            journals.assert_called_once_with('Kral', field='author')
            unified_search.search('Kral', field='title')
            self.assertEqual(arxiv.call_count, 2)
            self.assertEqual(journals.call_count, 2)
            unified_search.search('Kral', field='author')
            self.assertEqual(arxiv.call_count, 2)

    @mock.patch('summarizer.unified_search.requests.get')
    def test_scoped_arxiv_queries_do_not_search_other_fields(self, get):
        get.return_value.text = '<feed xmlns="http://www.w3.org/2005/Atom"/>'
        unified_search.arxiv_results('Quentin Kral', field='author')
        self.assertEqual(get.call_args.kwargs['params']['search_query'], 'au:"Quentin Kral"')
        unified_search.arxiv_results('Attention Is All You Need', field='title')
        self.assertEqual(get.call_args.kwargs['params']['search_query'], 'ti:"Attention Is All You Need"')

    @mock.patch('summarizer.sources._spend', return_value=True)
    @mock.patch('summarizer.sources._json', return_value={'results': []})
    def test_openalex_scopes_name_to_one_byline_and_title_to_title(self, fetch, spend):
        sources.search_papers('Quentin Kral', field='author')
        params = fetch.call_args.args[1]
        self.assertNotIn('search', params)
        self.assertIn('raw_author_name.search:"Quentin Kral"', params['filter'])
        sources.search_papers('Quentin Kral', field='title')
        self.assertIn('title.search:"Quentin Kral"', fetch.call_args.args[1]['filter'])
        self.assertEqual(fetch.call_count, 2)

    @mock.patch('summarizer.sources._spend', return_value=False)
    @mock.patch('summarizer.sources._json')
    def test_crossref_fallback_checks_the_selected_metadata(self, fetch, spend):
        fetch.return_value = {'message': {'items': [
            {'DOI': '10.1234/a', 'title': ['Unrelated title'],
             'author': [{'given': 'Quentin', 'family': 'Kral'}]},
            {'DOI': '10.1234/b', 'title': ['Quentin Kral'],
             'author': [{'given': 'Other', 'family': 'Author'}]},
            {'DOI': '10.1234/c', 'title': ['Another paper'],
             'author': [{'given': 'Quentin', 'family': 'Other'}, {'given': 'Other', 'family': 'Kral'}]},
        ]}}
        rows = sources.search_papers('Quentin Kral', field='author')
        self.assertEqual([p['doi'] for p in rows], ['10.1234/a'])
        self.assertEqual(fetch.call_args.args[1]['query.author'], 'Quentin Kral')
        rows = sources.search_papers('Quentin Kral', field='title')
        self.assertEqual([p['doi'] for p in rows], ['10.1234/b'])
        self.assertEqual(fetch.call_args.args[1]['query.title'], 'Quentin Kral')

    @mock.patch('summarizer.unified_search.close_old_connections')
    def test_optional_source_filter_does_not_drop_deduplicated_journal_records(self, connections):
        with mock.patch.object(unified_search, 'arxiv_results') as arxiv, \
             mock.patch.object(sources, 'search', return_value=[{'key': 'doi_example'}]) as journals:
            rows, failed = unified_search.search('Kral', field='author', source='open', kind='journals')
            self.assertEqual(rows, [{'key': 'doi_example'}])
            arxiv.assert_not_called()
            journals.assert_called_once_with('Kral', kind='journals', field='author')


class ScopedSearchPageTests(TestCase):
    def test_scope_is_preserved_and_identifiers_still_open_directly(self):
        with mock.patch('summarizer.views.sources.resolve', return_value=None):
            response = self.client.post('/en/', {'arxiv_id': 'Quentin Kral', 'field': 'author'})
        self.assertIn('field=author', response['Location'])
        response = self.client.post('/en/', {'arxiv_id': '1706.03762', 'field': 'author'})
        self.assertEqual(response['Location'], '/en/arxiv-id/1706.03762v1/')
        rows = [{'key': str(i), 'title': 'Paper', 'authors': [], 'arxiv_result': True} for i in range(11)]
        with mock.patch.object(unified_search, 'search', return_value=(rows, False)) as search:
            response = self.client.get('/en/search-results/?q=Kral&field=author')
        search.assert_called_once_with('Kral', field='author', source='all', kind='')
        self.assertContains(response, 'value="author" checked')
        self.assertTrue(any('field=author&src=all&type=&page=2' in a['href'] for a in
                            BeautifulSoup(response.content, 'html.parser').select('.pagination a')))
        self.assertContains(response, 'data-loading="Searching…"')
