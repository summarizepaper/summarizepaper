"""Whether the model may read a paper's full text (tdm.py): licenses, TDM reservations and sources. No network."""
import datetime
import json
from unittest import mock

import requests
from django.core.cache import cache
from django.test import TestCase, override_settings
from django.utils import timezone

from summarizer import assistant, changes, generated, paper_source, sources, synthesis, tdm
from summarizer.models import ArxivPaper, ExternalPaper, PaperText, PickledData, TdmCount, TdmDecision
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

    def test_only_the_robots_txt_groups_that_apply_to_us_are_kept(self):
        robots = ('Sitemap: https://example.org/s.xml\nUser-agent: GPTBot\nDisallow: /\n\n'
                  'User-agent: Googlebot\nUser-agent: SummarizePaper\nDisallow: /private\n\n'
                  'User-agent: CCBot\nDisallow: /\n\nUser-agent: *\nDisallow: /search\n')
        kept = tdm._our_rules(robots)
        self.assertNotIn('GPTBot', kept)
        self.assertNotIn('CCBot', kept)
        with site(robots=robots):
            self.assertTrue(tdm.site_reserves('https://example.org/private/x.pdf'))
            self.assertFalse(tdm.site_reserves('https://example.org/search'))  # our group applies, not *'s
            self.assertFalse(tdm.site_reserves('https://example.org/paper.pdf'))
        with site(robots='User-agent: *\nDisallow: /search\n'):
            cache.clear()
            self.assertTrue(tdm.site_reserves('https://example.org/search'))

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

    def test_old_indexes_of_papers_without_an_open_license_are_deleted(self):
        ArxivPaper.objects.create(arxiv_id='2001.00001v1', title='T', abstract='Abstract.', license=ARXIV)
        ArxivPaper.objects.create(arxiv_id='2001.00002v1', title='T', abstract='Abstract.', license=CC_BY)
        for arxiv_id in ('2001.00001v1', '2001.00002v1', '2101.00005v1'):
            PickledData.objects.create(arxiv_id=arxiv_id, docstore_pickle=b'x', buffer=b'', index_to_docstore_id_pickle=b'x')
        self.assertEqual(tdm.forget_texts(), 3)
        self.assertEqual(sorted(PickledData.objects.values_list('arxiv_id', flat=True)), ['2001.00002v1'])
        self.assertEqual(PaperText.objects.get(arxiv_id='2001.00001v1').source, 'forgotten')
        self.assertEqual(PaperText.objects.get(arxiv_id='2101.00005v1').source, 'html')  # read yesterday: kept
        self.assertFalse(PaperText.objects.filter(arxiv_id='2001.00002v1').exists())
        with mock.patch('summarizer.paper_source.read_again', return_value=('', '')):
            self.assertEqual(paper_source.paper_text('2001.00001v1'), 'Abstract.')  # never rebuilt from the index

    def test_a_paper_read_thanks_to_a_site_decision_is_forgotten_and_read_again_from_its_source(self):
        ArxivPaper.objects.create(arxiv_id=MNRAS_KEY, title='T', abstract='Abstract.', license=CC_BY_ND)
        ArxivPaper.objects.create(arxiv_id='key:closed', title='T', abstract='Abstract.', license='')
        for key in (MNRAS_KEY, 'key:closed'):
            PaperText.objects.create(arxiv_id=key, text='Text', source='xml')
        PaperText.objects.filter(arxiv_id__in=(MNRAS_KEY, 'key:closed')).update(
            updated=timezone.now() - datetime.timedelta(days=tdm.KEEP_DAYS + 1))
        tdm.forget_texts()
        self.assertEqual(PaperText.objects.get(arxiv_id='key:closed').source, 'forgotten')
        with mock.patch('summarizer.sources.full_text', return_value=('Whole paper. ' * 500, 'xml')) as read:
            self.assertTrue(paper_source.paper_text(MNRAS_KEY).startswith('Whole paper.'))
        read.assert_called_once_with(MNRAS_KEY)
        self.assertEqual(PaperText.objects.get(arxiv_id=MNRAS_KEY).source, 'xml')


def store(paper):
    ExternalPaper.objects.create(key=MNRAS_KEY, source='openalex', doi=MNRAS, fetched=timezone.now(), metadata=json.dumps(paper))


@override_settings(CACHES=LOCMEM, TDM_CHECK_NETWORK=True)
class DecisionTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_a_site_someone_cleared_is_read_under_the_exception_after_the_check(self):
        url = 'https://www.repo.example/x.pdf'
        self.assertEqual(tdm.status('', url), tdm.REVIEW)
        TdmDecision.objects.create(domain='WWW.Repo.example', status=tdm.EXCEPTION, note='CGU lues le 31/10/2026')
        self.assertEqual(TdmDecision.objects.get().domain, 'repo.example')
        with site():
            self.assertEqual(tdm.status('', url), tdm.EXCEPTION)
            self.assertEqual(tdm.status(ARXIV, url), tdm.EXCEPTION)  # its copy of an arXiv-licensed paper too
        cache.delete(tdm.POLICY_KEY % 'https://www.repo.example')
        with site(tdmrep=[{'location': '/*', 'tdm-reservation': 1}]):
            self.assertEqual(tdm.status('', url), tdm.BLOCKED)  # a reservation still wins

    def test_a_blocked_site_is_never_read_and_the_most_specific_decision_wins(self):
        TdmDecision.objects.create(domain='publisher.example', status=tdm.BLOCKED, note='Interdit par les CGU')
        TdmDecision.objects.create(domain='oa.publisher.example', status=tdm.EXCEPTION, note='Portail ouvert')
        self.assertEqual(tdm.status(CC_BY, 'https://publisher.example/x.pdf'), tdm.BLOCKED)
        with site():
            self.assertEqual(tdm.status('', 'https://oa.publisher.example/x.pdf'), tdm.EXCEPTION)
        TdmDecision.objects.get(domain='oa.publisher.example').delete()
        self.assertEqual(tdm.status('', 'https://oa.publisher.example/x.pdf'), tdm.BLOCKED)


@override_settings(CACHES=LOCMEM, TDM_CHECK_NETWORK=True)
class ExternalPaperTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_a_doi_without_license_is_read_from_its_arxiv_version(self):
        store(journal_paper(license='', text=[('pdf', 'https://publisher.example/x.pdf'), ('arxiv', '2301.03702')]))
        self.assertEqual(tdm.paper_status('', MNRAS_KEY), tdm.EXCEPTION)
        row = [1] + [''] * 11 + [ARXIV, '']
        with site(), mock.patch('summarizer.utils.get_arxiv_metadata', mock.AsyncMock(return_value=row)), \
             mock.patch('summarizer.sources._download') as download, \
             mock.patch('summarizer.paper_source.fetch_arxiv_html', return_value='arXiv text ' * 500):
            self.assertEqual(sources.full_text(MNRAS_KEY)[1], 'html')
        download.assert_not_called()  # the publisher's copy has no established rights

    def test_a_doi_without_any_readable_copy_says_so(self):
        store(journal_paper(license='', text=[('pdf', 'https://publisher.example/x.pdf')]))
        self.assertEqual(tdm.paper_status('', MNRAS_KEY), tdm.REVIEW)
        self.assertTrue(sources.rights_kept_out(MNRAS_KEY))
        paper = ArxivPaper.objects.create(arxiv_id=MNRAS_KEY, title='T', license=CC_BY)
        PaperText.objects.create(arxiv_id=MNRAS_KEY, text='Abstract.', source='abstract')
        self.assertEqual(synthesis.license_scope(paper), 'abstract_license')
        ExternalPaper.objects.all().delete()  # copies unknown: nothing says the rights kept us out
        self.assertFalse(sources.rights_kept_out(MNRAS_KEY))
        self.assertEqual(synthesis.license_scope(paper), 'abstract_unavailable')


class CountTests(TestCase):
    def test_summaries_are_counted_by_answer(self):
        day = datetime.date(2026, 10, 31)
        for answer in (tdm.EXCEPTION, tdm.EXCEPTION, tdm.ALLOWED):
            tdm.record(answer, day)
        self.assertEqual(dict(TdmCount.objects.values_list('status', 'count')), {tdm.EXCEPTION: 2, tdm.ALLOWED: 1})


class ReuseRuleTests(TestCase):
    def test_every_generation_is_told_not_to_reproduce_the_papers(self):
        self.assertIn(generated.REUSE_RULE, assistant.SOURCE_POLICY)
        answer = mock.Mock(status_code=200)
        answer.json.return_value = {'choices': [{'message': {'content': '{}'}}], 'usage': {}}
        for system in (None, 'Task.', assistant.SOURCE_POLICY):
            with mock.patch('summarizer.generated.requests.post', return_value=answer) as post:
                generated.chat_json('prompt', system=system)
            sent = post.call_args.kwargs['json']['messages'][0]['content']
            self.assertEqual(sent.count(generated.REUSE_RULE), 1)


class PageTests(TestCase):
    def test_the_public_page_says_what_the_robot_does(self):
        page = self.client.get('/en/tdm/')
        self.assertContains(page, 'SummarizePaper/1.0')
        self.assertContains(page, '%d days after it was read' % tdm.KEEP_DAYS)
        self.assertIn('/en/tdm/', sources.USER_AGENT)

