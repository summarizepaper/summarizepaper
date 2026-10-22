from unittest import mock

import requests
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
            self.assertContains(response, 'src=all&type=&page=2')
            self.assertContains(response, 'Paper 0')
            response = self.client.get('/en/search-results/?q=attention&src=all&page=2')
            self.assertContains(response, 'Paper 11')
            self.assertEqual(search.call_count, 2)
