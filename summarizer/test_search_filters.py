from unittest import mock
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup
from django.core.cache import cache
from django.test import SimpleTestCase, TestCase

from . import search_filters, sources, unified_search


class FilterValidationTests(SimpleTestCase):
    def test_valid_filters_normalize_orcid_and_journal_order(self):
        form = search_filters.SearchFilters({'year_from': '2020', 'year_to': '2024', 'sort': 'newest',
            'orcid': 'https://orcid.org/0000-0001-6527-4684', 'journals': ['S2', 'S1', 'S2']})
        self.assertTrue(form.is_valid(), form.errors)
        self.assertEqual(form.options()['journals'], ['S1', 'S2'])
        self.assertEqual(form.options()['orcid'], '0000-0001-6527-4684')

    def test_invalid_filters_are_rejected(self):
        for data in ({'year_from': 2025, 'year_to': 2020}, {'year_to': 9999},
                     {'orcid': '0000-0001-6527-4685'}, {'orcid': 'not-an-orcid'},
                     {'journals': ['S1|S2']}, {'journals': ['S1'], 'exclude_journals': ['S1']},
                     {'journals': ['S%d' % n for n in range(6)]}, {'sort': 'injected'}):
            with self.subTest(data=data):
                self.assertFalse(search_filters.SearchFilters(data).is_valid())

    def test_default_filters_do_not_change_search(self):
        form = search_filters.SearchFilters({})
        self.assertTrue(form.is_valid())
        self.assertEqual(form.options(), {})


class ProviderFilterTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    @mock.patch('summarizer.sources._spend', return_value=True)
    @mock.patch('summarizer.sources._json', return_value={'results': []})
    def test_openalex_receives_every_filter_before_retrieving_results(self, fetch, spend):
        sources.search_papers('climate', field='abstract', kind='reviews', filters={
            'year_from': 2020, 'year_to': 2024, 'sort': 'newest', 'journals': ['S1', 'S2'],
            'exclude_journals': ['S3', 'S4'], 'orcid': '0000-0001-6527-4684'})
        params = fetch.call_args.args[1]
        for clause in ('abstract.search:"climate"', 'type:review', 'from_publication_date:2020-01-01',
                       'to_publication_date:2024-12-31', 'primary_location.source.id:S1|S2',
                       'primary_location.source.id:!S3', 'primary_location.source.id:!S4',
                       'authorships.author.orcid:https://orcid.org/0000-0001-6527-4684'):
            self.assertIn(clause, params['filter'])
        self.assertNotIn('search', params)
        self.assertEqual(params['sort'], 'publication_date:desc')

    @mock.patch('summarizer.unified_search.requests.get')
    def test_arxiv_uses_abstract_date_range_and_native_date_sort(self, get):
        get.return_value.text = '<feed xmlns="http://www.w3.org/2005/Atom"/>'
        unified_search.arxiv_results('climate', field='abstract', filters={
            'year_from': 2020, 'year_to': 2024, 'sort': 'oldest'})
        params = get.call_args.kwargs['params']
        self.assertEqual(params['search_query'], '(abs:"climate") AND submittedDate:[202001010000 TO 202412312359]')
        self.assertEqual((params['sortBy'], params['sortOrder']), ('submittedDate', 'ascending'))

    @mock.patch('summarizer.sources._spend', return_value=False)
    def test_unsupported_fallback_cannot_silently_broaden_search(self, spend):
        for kwargs in ({'filters': {'journals': ['S1']}}, {'filters': {'orcid': '0000-0001-6527-4684'}},
                       {'field': 'abstract'}, {'kind': 'reviews'}):
            with self.subTest(kwargs=kwargs), mock.patch.object(sources, 'crossref_search') as crossref:
                with self.assertRaises(sources.SourceUnavailable):
                    sources.search_papers('climate', **kwargs)
                crossref.assert_not_called()

    @mock.patch('summarizer.sources._json', return_value={'message': {'items': []}})
    def test_supported_crossref_fallback_keeps_years_and_sort(self, fetch):
        sources.crossref_search('climate', 25, filters={'year_from': 2020, 'year_to': 2024, 'sort': 'oldest'})
        params = fetch.call_args.args[1]
        self.assertIn('from-pub-date:2020-01-01', params['filter'])
        self.assertIn('until-pub-date:2024-12-31', params['filter'])
        self.assertEqual((params['sort'], params['order']), ('published', 'asc'))

    @mock.patch('summarizer.unified_search.close_old_connections')
    def test_filters_partition_cache_and_sort_across_sources(self, connections):
        with mock.patch.object(unified_search, 'arxiv_results', return_value=[
                {'key': 'a', 'published': '2021-01-01'}, {'key': 'b', 'published': '2024-01-01'}]) as arxiv, \
             mock.patch.object(sources, 'search', return_value=[{'key': 'c', 'published': '2022-01-01'}]):
            rows, _ = unified_search.search('climate', filters={'sort': 'newest'})
            self.assertEqual([r['key'] for r in rows], ['b', 'c', 'a'])
            rows, _ = unified_search.search('climate', filters={'sort': 'oldest'})
            self.assertEqual([r['key'] for r in rows], ['a', 'c', 'b'])
            self.assertEqual(arxiv.call_count, 2)

    @mock.patch('summarizer.unified_search.close_old_connections')
    def test_orcid_and_published_types_do_not_return_unfiltered_arxiv_papers(self, connections):
        with mock.patch.object(unified_search, 'arxiv_results') as arxiv, \
             mock.patch.object(sources, 'search', return_value=[]):
            unified_search.search('climate', filters={'orcid': '0000-0001-6527-4684'})
            unified_search.search('climate', kind='reviews')
            unified_search.search('climate', kind='journals')
            arxiv.assert_not_called()


class SearchFilterPageTests(TestCase):
    def setUp(self):
        cache.clear()
        cache.set('search-journal:S1', {'id': 'S1', 'name': 'Journal One'})
        cache.set('search-journal:S2', {'id': 'S2', 'name': 'Journal Two'})

    def test_filters_survive_get_form_and_pagination(self):
        rows = [{'key': str(i), 'title': 'A paper', 'authors': []} for i in range(12)]
        query = {'q': 'climate', 'field': 'abstract', 'year_from': 2020, 'year_to': 2024,
                 'sort': 'newest', 'type': 'reviews', 'journals': ['S1', 'S2'],
                 'orcid': '0000-0001-6527-4684'}
        with mock.patch.object(unified_search, 'search', return_value=(rows, False)) as search:
            response = self.client.get('/en/search-results/', query)
        options = search.call_args.kwargs
        self.assertEqual(options['filters']['journals'], ['S1', 'S2'])
        self.assertEqual(options['kind'], 'reviews')
        soup = BeautifulSoup(response.content, 'html.parser')
        self.assertEqual(soup.select_one('form[data-research-search]')['method'], 'get')
        next_page = parse_qs(urlsplit(soup.select_one('.pagination a')['href']).query)
        self.assertEqual(next_page['journals'], ['S1', 'S2'])
        self.assertEqual(next_page['orcid'], ['0000-0001-6527-4684'])
        self.assertEqual(next_page['year_from'], ['2020'])
        self.assertEqual(next_page['field'], ['abstract'])
        self.assertContains(response, 'Journal One')
        self.assertContains(response, 'direct arXiv feed is not included')

    def test_invalid_filter_does_not_call_any_search_provider(self):
        with mock.patch.object(unified_search, 'search') as search:
            response = self.client.get('/fr/search-results/?q=climate&year_from=2025&year_to=2020')
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context['filter_form'].errors)
        search.assert_not_called()

    def test_orcid_only_search_and_identifier_routing(self):
        with mock.patch.object(unified_search, 'search', return_value=([], False)) as search:
            response = self.client.get('/en/search-results/', {'orcid': '0000-0001-6527-4684'})
        search.assert_called_once()
        self.assertContains(response, 'Research search results')
        self.assertContains(response, 'Search again without filters')
        response = self.client.get('/en/search-results/?q=1706.03762')
        self.assertEqual(response['Location'], '/en/arxiv-id/1706.03762v1/')

    def test_incompatible_arxiv_filter_reports_error(self):
        with mock.patch.object(unified_search, 'search') as search:
            response = self.client.get('/en/search-results/?q=climate&src=arxiv&type=reviews')
        self.assertTrue(response.context['filter_form'].errors)
        search.assert_not_called()

    @mock.patch('summarizer.sources._spend', return_value=True)
    @mock.patch('summarizer.sources._json')
    def test_journal_suggestions_cache_and_expose_only_public_metadata(self, fetch, spend):
        fetch.return_value = {'results': [{'id': 'https://openalex.org/S1', 'display_name': 'Nature',
                                         'issn_l': '0028-0836', 'host_organization_name': 'Publisher'}]}
        for _ in range(2):
            response = self.client.get('/en/search-journals/?q=Nature')
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['results'][0]['id'], 'S1')
        fetch.assert_called_once()
        self.assertEqual(fetch.call_args.args[1]['filter'], 'type:journal')
        self.assertEqual(search_filters.selected_journals(['S1'])[0]['name'], 'Nature')

    @mock.patch('summarizer.sources._json', side_effect=sources.SourceUnavailable('offline'))
    def test_suggestion_outage_is_reported_without_a_500(self, fetch):
        response = self.client.get('/en/search-journals/?q=Nature')
        self.assertEqual(response.status_code, 503)
        self.assertTrue(response.json()['unavailable'])
