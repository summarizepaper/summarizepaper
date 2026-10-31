"""Whether the model may read a paper's full text (tdm.py): licenses, TDM reservations and sources. No network."""
import datetime
import json
from unittest import mock

import requests
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import timezone

from summarizer import assistant, changes, paper_source, sources, synthesis, tdm
from summarizer.models import ArxivPaper, ExternalPaper, PaperText
from summarizer.test_sources import JATS, MNRAS, MNRAS_KEY, journal_paper

CC_BY = 'http://creativecommons.org/licenses/by/4.0/'
CC_BY_ND = 'http://creativecommons.org/licenses/by-nd/4.0/'
ARXIV = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
ARXIV_OLD = 'http://arxiv.org/licenses/assumed-1991-2003/'
ROBOTS_ARXIV = 'User-agent: *\nAllow: /pdf\nAllow: /html\nDisallow: /e-print\n'
LOCMEM = {'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}}


def response(status=200, text='', headers=None):
    answer = mock.Mock(status_code=status, text=text, headers=headers or {})
    answer.json.side_effect = lambda: json.loads(text)
    return answer


def site(tdmrep=None, robots=ROBOTS_ARXIV, error=None):
    """A fake sources._session.get answering /.well-known/tdmrep.json and /robots.txt."""
    def get(url, **kwargs):
        if error:
            raise error
        if url.endswith('/.well-known/tdmrep.json'):
            return response(404) if tdmrep is None else response(200, tdmrep if isinstance(tdmrep, str) else json.dumps(tdmrep))
        if url.endswith('/robots.txt'):
            return response(404) if robots is None else response(200, robots)
        raise AssertionError('unexpected request to %s' % url)
    return mock.patch('summarizer.sources._session.get', side_effect=get)


@override_settings(CACHES=LOCMEM, TDM_CHECK_NETWORK=True)
class StatusTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_the_license_alone(self):
        for url in (CC_BY, 'https://creativecommons.org/licenses/by-nc-sa/4.0/', 'https://creativecommons.org/publicdomain/zero/1.0/'):
            self.assertEqual(tdm.license_status(url), tdm.ALLOWED, url)
        for url in (ARXIV, ARXIV_OLD, CC_BY_ND, 'https://creativecommons.org/licenses/by-nc-nd/4.0/'):
            self.assertEqual(tdm.license_status(url), tdm.EXCEPTION, url)
        for url in ('', None, 'https://www.elsevier.com/tdm/userlicense/1.0/', 'http://example.org/terms'):
            self.assertEqual(tdm.license_status(url), tdm.REVIEW, url)

    def test_open_licenses_never_ask_the_site(self):
        with site(error=AssertionError('asked')):
            self.assertEqual(tdm.arxiv_status(CC_BY), tdm.ALLOWED)
            self.assertEqual(tdm.status('', 'https://example.org/x.pdf'), tdm.REVIEW)

    def test_arxiv_standard_license_is_eligible_when_arxiv_reserves_nothing(self):
        with site() as get:
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.EXCEPTION)
            self.assertEqual(tdm.arxiv_status(ARXIV_OLD), tdm.EXCEPTION)
            self.assertTrue(tdm.may_read(ARXIV, '2101.00001v1'))
        self.assertEqual(get.call_count, 2)  # tdmrep.json and robots.txt, once: cached for the site

    def test_a_tdmrep_reservation_blocks(self):
        with site(tdmrep=[{'location': '/pdf/*', 'tdm-reservation': 1, 'tdm-policy': 'https://example.org/policy'},
                          {'location': '/*', 'tdm-reservation': 0}]):
            self.assertEqual(tdm.status(ARXIV, 'https://arxiv.org/pdf/2101.00001'), tdm.BLOCKED)
            self.assertEqual(tdm.status(ARXIV, 'https://arxiv.org/html/2101.00001'), tdm.EXCEPTION)
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.BLOCKED)  # the strictest of the HTML and the PDF
            self.assertFalse(tdm.may_read(ARXIV, '2101.00001v1'))

    def test_robots_txt_keeping_us_out_blocks(self):
        with site(robots='User-agent: SummarizePaper\nDisallow: /\n'):
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.BLOCKED)
        cache.clear()
        with site(robots=None):  # no robots.txt: no limit
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.EXCEPTION)

    def test_a_site_that_cannot_be_asked_needs_a_review_and_is_asked_again(self):
        with site(error=requests.ConnectionError()):
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.REVIEW)
        with site(tdmrep='not json'):
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.REVIEW)
        with mock.patch('summarizer.sources._session.get', return_value=response(503)):
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.REVIEW)
        with site():
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.EXCEPTION)

    def test_arxiv_license_on_another_site_needs_a_review(self):
        with site():
            self.assertEqual(tdm.status(ARXIV, 'https://publisher.example/x.pdf'), tdm.REVIEW)
            self.assertEqual(tdm.status(CC_BY_ND, 'https://repository.example/x.pdf'), tdm.EXCEPTION)

    def test_page_loads_never_ask(self):
        with site(error=AssertionError('asked')):
            self.assertFalse(tdm.may_read(ARXIV, '2101.00001v1', ask=False))
        with site():
            tdm.run(timezone.now())  # the hourly run asks arXiv
        with site(error=AssertionError('asked')):
            self.assertTrue(tdm.may_read(ARXIV, '2101.00001v1', ask=False))

    def test_the_hourly_run_keeps_the_last_answer_when_arxiv_does_not_answer(self):
        with site():
            tdm.run(timezone.now())
        with site(error=requests.ConnectionError()):
            tdm.run(timezone.now())
        self.assertEqual(tdm.arxiv_status(ARXIV, ask=False), tdm.EXCEPTION)

    @override_settings(TDM_CHECK_NETWORK=False)
    def test_the_test_suite_never_asks(self):
        with site(error=AssertionError('asked')):
            self.assertEqual(tdm.arxiv_status(ARXIV), tdm.REVIEW)

    def test_documents_reserving_mining(self):
        self.assertTrue(tdm.document_reserves({'tdm-reservation': '1'}))
        self.assertFalse(tdm.document_reserves({'tdm-reservation': '0'}))
        self.assertTrue(tdm.document_reserves({}, '<head><meta name="tdm-reservation" content="1"></head>'))
        self.assertTrue(tdm.document_reserves({}, "<meta content='1' name='tdm-reservation'/>"))
        self.assertFalse(tdm.document_reserves({}, '<meta name="tdm-reservation" content="0">'))
        self.assertFalse(tdm.document_reserves(mock.Mock()))


@override_settings(CACHES=LOCMEM, TDM_CHECK_NETWORK=True)
class ReadingTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_a_reserved_download_is_not_read(self):
        reserved = mock.MagicMock(status_code=200, headers={'tdm-reservation': '1'})
        reserved.__enter__.return_value = reserved
        with mock.patch('summarizer.sources._session.get', return_value=reserved):
            self.assertIsNone(sources._download('https://example.org/x.pdf'))
        reserved.iter_content.assert_not_called()

    def test_a_reserved_arxiv_html_is_not_read(self):
        page = mock.Mock(status_code=200, url='https://arxiv.org/html/2101.00001v1', headers={},
                         text='<html><head><meta name="tdm-reservation" content="1"></head><article>%s</article></html>' % ('x ' * 2000))
        with mock.patch('summarizer.paper_source.requests.get', return_value=page):
            self.assertIsNone(paper_source.fetch_arxiv_html('2101.00001v1'))

    def test_the_arxiv_version_under_arxiv_license_is_read_after_the_check(self):
        row = [1] + [''] * 11 + [ARXIV, '']
        with site(), mock.patch('summarizer.utils.get_arxiv_metadata', mock.AsyncMock(return_value=row)), \
             mock.patch('summarizer.paper_source.fetch_arxiv_html', return_value='arXiv text') as html:
            self.assertEqual(sources.read_candidate('arxiv', '2301.03702'), ('arXiv text', 'html'))
        html.assert_called_once_with('2301.03702')

    def test_openalex_reads_the_copies_whose_rights_are_established_first(self):
        work = {'doi': 'https://doi.org/' + MNRAS, 'display_name': 'T', 'id': 'https://openalex.org/W1',
                'best_oa_location': {'license': 'cc-by'},
                'locations': [
                    {'pdf_url': 'https://publisher.example/x.pdf', 'license': None, 'source': {'type': 'journal'}},
                    {'pdf_url': 'https://repo.example/nd.pdf', 'license': 'cc-by-nd', 'source': {'type': 'repository'}},
                    {'pdf_url': 'https://journal.example/open.pdf', 'license': 'cc-by', 'source': {'type': 'journal'}}]}
        paper = sources.openalex_paper(work)
        self.assertEqual([w for k, w in paper['text'] if k == 'pdf'],
                         ['https://journal.example/open.pdf', 'https://repo.example/nd.pdf', 'https://publisher.example/x.pdf'])
        self.assertEqual(paper['licenses']['https://publisher.example/x.pdf'], '')

    def test_full_text_skips_a_copy_without_established_rights(self):
        paper = journal_paper(text=[('pdf', 'https://publisher.example/x.pdf'), ('jats', 'https://repo.example/x.xml')],
                              licenses={'https://publisher.example/x.pdf': ''})
        ExternalPaper.objects.create(key=MNRAS_KEY, source='openalex', doi=MNRAS, fetched=timezone.now(), metadata=json.dumps(paper))
        long_jats = JATS.replace(b'Stars form', b'Stars form ' + b'very ' * 500)
        with mock.patch('summarizer.sources._download', return_value=long_jats) as download:
            self.assertEqual(sources.full_text(MNRAS_KEY)[1], 'xml')
        self.assertEqual([c.args[0] for c in download.call_args_list], ['https://repo.example/x.xml'])

    def test_a_reserved_nd_copy_is_skipped(self):
        paper = journal_paper(license=CC_BY_ND, text=[('pdf', 'https://repo.example/x.pdf')])
        ExternalPaper.objects.create(key=MNRAS_KEY, source='openalex', doi=MNRAS, fetched=timezone.now(), metadata=json.dumps(paper))
        with site(tdmrep=[{'location': '/*', 'tdm-reservation': 1}]), mock.patch('summarizer.sources._download') as download, \
             mock.patch('summarizer.sources.later_places', return_value=[]):
            self.assertEqual(sources.full_text(MNRAS_KEY)[1], 'abstract')
        download.assert_not_called()


@override_settings(CACHES=LOCMEM, TDM_CHECK_NETWORK=True)
class UsesTests(TestCase):
    def setUp(self):
        cache.clear()
        self.paper = ArxivPaper.objects.create(arxiv_id='2101.00002v1', title='Closed', abstract='Abs closed.', license=ARXIV)
        PaperText.objects.create(arxiv_id='2101.00002v1', text='FULL TEXT OF THE ARXIV LICENSED PAPER', source='pdf')

    def test_the_assistant_reads_it_without_republishing_it(self):
        with site():
            context = assistant.build_context(['2101.00002v1'])
        self.assertIn('FULL TEXT OF THE ARXIV LICENSED PAPER', context)
        self.assertIn(assistant.QUOTE_NOTE, context)

    def test_the_assistant_keeps_to_the_abstract_when_arxiv_reserves_mining(self):
        with site(tdmrep=[{'location': '/*', 'tdm-reservation': 1}]):
            context = assistant.build_context(['2101.00002v1'])
        self.assertNotIn('FULL TEXT OF THE ARXIV LICENSED PAPER', context)
        self.assertIn('Abs closed.', context)

    def test_an_open_paper_gets_no_quote_note(self):
        ArxivPaper.objects.filter(pk=self.paper.pk).update(license=CC_BY)
        with site(error=AssertionError('asked')):
            context = assistant.build_context(['2101.00002v1'])
        self.assertIn('FULL TEXT OF THE ARXIV LICENSED PAPER', context)
        self.assertNotIn(assistant.QUOTE_NOTE, context)

    def test_pages_say_what_was_read_without_asking(self):
        self.assertEqual(synthesis.license_scope(self.paper), 'abstract_license')
        self.assertEqual(changes.license_preview('2101.00002'), 'restricted')
        with site():
            tdm.run(timezone.now())
        self.assertEqual(synthesis.license_scope(self.paper), 'full_text')
        self.assertEqual(changes.license_preview('2101.00002'), 'allowed')


@override_settings(CACHES=LOCMEM, TDM_CHECK_NETWORK=True)
class RetentionTests(TestCase):
    def setUp(self):
        cache.clear()
        old = timezone.now() - datetime.timedelta(days=tdm.KEEP_DAYS + 1)
        for arxiv_id, license, source in (('2101.00001v1', ARXIV, 'html'), ('2101.00002v1', CC_BY, 'html'),
                                          ('2101.00003v1', CC_BY_ND, 'pdf'), ('2101.00004v1', ARXIV, 'abstract')):
            ArxivPaper.objects.create(arxiv_id=arxiv_id, title='T', abstract='Abstract %s.' % arxiv_id, license=license)
            PaperText.objects.create(arxiv_id=arxiv_id, text='Text of %s' % arxiv_id, source=source)
        PaperText.objects.update(updated=old)
        ArxivPaper.objects.create(arxiv_id='2101.00005v1', title='T', license=ARXIV)
        PaperText.objects.create(arxiv_id='2101.00005v1', text='Read yesterday', source='html')

    def test_texts_read_under_the_exception_are_forgotten(self):
        with site():
            self.assertEqual(tdm.run(timezone.now()), 2)
        self.assertEqual(dict(PaperText.objects.values_list('arxiv_id', 'source')), {
            '2101.00001v1': 'forgotten', '2101.00002v1': 'html', '2101.00003v1': 'forgotten',
            '2101.00004v1': 'abstract', '2101.00005v1': 'html'})
        self.assertEqual(PaperText.objects.get(arxiv_id='2101.00001v1').text, '')

    def test_a_forgotten_text_is_read_again_when_asked_about(self):
        tdm.forget_texts()
        with site(), mock.patch('summarizer.paper_source.fetch_arxiv_html', return_value='Whole paper. ' * 500):
            self.assertTrue(paper_source.paper_text('2101.00001v1').startswith('Whole paper.'))
        self.assertEqual(PaperText.objects.get(arxiv_id='2101.00001v1').source, 'html')
        with site(tdmrep=[{'location': '/*', 'tdm-reservation': 1}]):
            cache.clear()
            tdm.forget_texts(timezone.now() + datetime.timedelta(days=tdm.KEEP_DAYS + 1))
            self.assertEqual(paper_source.paper_text('2101.00001v1'), 'Abstract 2101.00001v1.')
