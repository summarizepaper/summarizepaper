import datetime
import json
from types import SimpleNamespace
from unittest import mock

from django.test import SimpleTestCase
from django.utils import timezone, translation

from . import publication_status as status
from .models import Generated
from .test_tools import ToolsTestCase, base_call, offline
from .test_presskit_quality import pdf_text


TITLE = 'Water distribution in the disc of a young star'


def article(**kwargs):
    work = {'DOI': '10.1234/published', 'type': 'journal-article', 'title': [TITLE],
            'container-title': ['Example Journal'], 'published': {'date-parts': [[2024, 3, 12]]},
            'author': [{'given': 'Ann', 'family': 'Lee'}]}
    return dict(work, **kwargs)


def report(**kwargs):
    return dict({'status': 'published', 'checked_at': timezone.now().isoformat(), 'journal': 'Example Journal',
                 'doi': '10.1234/published', 'date': '2024-03-12', 'checked_sources': ['Crossref'],
                 'unavailable': [], 'sources': [{'name': 'Crossref', 'url': 'https://doi.org/10.1234/published',
                                               'detail': 'Example Journal · 2024-03-12', 'kind': 'publisher_metadata'}],
                 'peer_review': 'not_independently_verified'}, **kwargs)


class PublicationLookupTests(SimpleTestCase):
    def setUp(self):
        self.paper = SimpleNamespace(arxiv_id='2401.04088v1', link_doi='https://doi.org/10.1234/published',
                                     title=TITLE, authors=mock.Mock())
        self.paper.authors.values_list.return_value = ['Ann Lee']
        patch = mock.patch.object(status.digest, 'parse_feed', return_value=[{
            'arxiv_id': '2401.04088v2', 'doi': '10.1234/published', 'journal_ref': 'Example Journal', 'comment': ''}])
        patch.start(); self.addCleanup(patch.stop)
        patch = mock.patch.object(status.sources, '_get', return_value=mock.Mock(text='<feed/>'))
        self.http = patch.start(); self.addCleanup(patch.stop)

    def test_latest_arxiv_doi_and_publisher_record_are_dated_sources_not_peer_review_proof(self):
        with mock.patch.object(status.sources, '_json', return_value={'message': article()}):
            data = status.lookup(self.paper)
        self.assertEqual(data['status'], 'published')
        self.assertEqual(data['peer_review'], 'not_independently_verified')
        self.assertEqual(data['date'], '2024-03-12')
        self.assertEqual([s['name'] for s in data['sources']], ['arXiv', 'Crossref'])
        self.assertEqual(self.http.call_args.args[1]['id_list'], '2401.04088')
        self.assertIsInstance(datetime.datetime.fromisoformat(data['checked_at']), datetime.datetime)

    def test_preprint_doi_and_journal_reference_are_not_publication_proof(self):
        with mock.patch.object(status.digest, 'parse_feed', return_value=[{
            'arxiv_id':'2401.04088v2', 'doi':'10.48550/arxiv.2401.04088', 'journal_ref':'Example Journal', 'comment':''}]), \
             mock.patch.object(status.sources, '_json', return_value={'message': {'items': []}}) as search:
            data = status.lookup(self.paper)
        self.assertEqual(data['status'], 'journal_reference')
        self.assertEqual(search.call_count, 1)
        self.assertEqual(search.call_args.args[0], 'https://api.crossref.org/works')

    def test_explicit_preprint_relation_can_find_a_differently_titled_published_version(self):
        self.paper.arxiv_id = 'doi_10.1101--12345'; self.paper.link_doi = '10.1101/12345'
        preprint = {'type':'posted-content', 'relation': {'is-preprint-of':[{'id-type':'doi', 'id':'10.1234/published'}]}}
        with mock.patch.object(status.sources, '_json', side_effect=[{'message':preprint}, {'message':article()}]):
            data = status.lookup(self.paper)
        self.assertEqual(data['status'], 'published')
        self.assertTrue(any(s['kind'] == 'relation' for s in data['sources']))

    def test_title_search_requires_exact_title_and_matching_author(self):
        self.paper.arxiv_id = 'doi_10.48550--arxiv.2401.04088'; self.paper.link_doi = ''
        for work, expected in [(article(title=['A different water distribution']), 'unknown'),
                               (article(author=[{'family':'Other'}]), 'unknown'), (article(), 'published')]:
            with mock.patch.object(status.sources, '_json', return_value={'message':{'items':[work]}}):
                self.assertEqual(status.lookup(self.paper)['status'], expected)

    def test_accepted_comment_is_an_author_declaration_and_outages_are_explicit(self):
        with mock.patch.object(status.digest, 'parse_feed', return_value=[{
            'arxiv_id':'2401.04088v2', 'doi':'', 'journal_ref':'', 'comment':'Accepted for publication in Example Journal'}]), \
             mock.patch.object(status.sources, '_json', side_effect=status.sources.SourceUnavailable('offline')):
            data = status.lookup(self.paper)
        self.assertEqual(data['status'], 'accepted_declared')
        self.assertIn('Crossref', data['unavailable'])
        self.http.side_effect = status.sources.SourceUnavailable('offline')
        with mock.patch.object(status.sources, '_json', side_effect=status.sources.SourceUnavailable('offline')):
            data = status.lookup(self.paper)
        self.assertEqual(data['status'], 'unknown')
        self.assertCountEqual(data['unavailable'], ['arXiv','Crossref'])

    def test_future_dates_and_repository_records_do_not_become_published(self):
        data = report(status='unknown')
        self.assertTrue(status.apply_work(data, article(published={'date-parts':[[2099,1,1]]})))
        self.assertEqual(data['status'], 'forthcoming')
        self.assertFalse(status.apply_work(data, article(type='posted-content')))
        self.assertFalse(status.apply_work(data, article(published={})))


class PublicationKitTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.paper_obj = self.paper()
        self.reader = self.user('publication-reader', paid=True)
        self.paid = self.client_for(self.reader)
        self.base = '/fr/press-kit/2101.00001v1/'
        self.kit = {'questions_for_authors':['Quel est le statut éditorial actuel du manuscrit ?', 'Quelles sont les limites ?']}
        self.row = Generated.objects.create(kind='presskit', key=self.paper_obj.arxiv_id, lang='fr', status='done', content=json.dumps(self.kit))

    def test_existing_kits_check_without_model_or_regeneration_and_adapt_questions_and_pdf(self):
        with mock.patch.object(status, 'lookup', return_value=report()) as lookup, \
             mock.patch('summarizer.generated.chat_json') as llm:
            self.assertContains(self.paid.get(self.base), 'data-auto="1"')
            lookup.assert_not_called()
            started = self.paid.post(self.base+'publication/start/').json()
            job = Generated.objects.get(pk=started['id'])
            self.assertEqual(job.status, 'done')
            result = self.paid.get(self.base+'publication/').json()
            self.assertIn('Publication retrouvée', result['html'])
            self.assertIn('Crossref', result['html'])
            self.assertNotIn('statut éditorial actuel', result['questions'][0])
            self.assertIn('version du texte', result['questions'][0])
            self.assertEqual(result['questions'][1], 'Quelles sont les limites ?')
            pdf = pdf_text(self.paid.get(self.base+'pdf/').content)
            self.assertIn('Publication retrouvée', pdf)
            self.assertNotIn('statut éditorial actuel', pdf)
            self.paid.post(self.base+'publication/start/')
            lookup.assert_called_once()
            llm.assert_not_called()
        self.row.refresh_from_db()
        self.assertEqual(self.row.data(), self.kit)

    def test_access_and_failed_rechecks_do_not_expose_or_change_private_kits(self):
        self.assertEqual(self.client.post(self.base+'publication/start/').status_code, 404)
        self.row.user = self.reader; self.row.save()
        other = self.client_for(self.user('other-status-reader', paid=True))
        self.assertEqual(other.get(self.base+'publication/').status_code, 404)
        self.assertEqual(other.post(self.base+'publication/start/').status_code, 404)
        with mock.patch.object(status, 'lookup', side_effect=RuntimeError('offline')):
            response = self.paid.post(self.base+'publication/start/')
        job = Generated.objects.get(pk=response.json()['id'])
        self.assertEqual(job.status,'failed')
        self.assertEqual(self.client.get('/fr/ai/%s/' % job.pk).status_code, 404)
        self.row.refresh_from_db(); self.assertEqual(self.row.data(),self.kit)

    def test_new_kit_gives_the_publication_check_to_the_model_as_source_data(self):
        self.row.delete()
        with mock.patch.object(status, 'lookup', return_value=report()), \
             mock.patch('summarizer.paper_source.paper_text', return_value='## Results\nThis is the available scientific full text.'), \
             offline(), \
             mock.patch('summarizer.generated.chat_json', return_value={}) as llm:
            self.paid.post(self.base+'start/')
        call = base_call(llm)
        context = json.loads(call.args[0])
        self.assertEqual(context['source_documents']['publication_check']['status'],'published')
        self.assertNotIn('Example Journal',call.kwargs['system'])
        self.assertIn('Do not ask a generic question',call.kwargs['system'])
        self.assertEqual(Generated.objects.get(kind='presskit').data()['publication_status']['doi'],'10.1234/published')

    def test_post_verification_requires_csrf(self):
        from django.test import Client
        client = Client(enforce_csrf_checks=True)
        client.force_login(self.reader)
        page = client.get(self.base)
        self.assertContains(page, 'csrfmiddlewaretoken')
        self.assertEqual(client.post(self.base+'publication/start/').status_code,403)
        with mock.patch.object(status,'lookup',return_value=report()):
            self.assertEqual(client.post(self.base+'publication/start/',HTTP_X_CSRFTOKEN=client.cookies['csrftoken'].value).status_code,200)
