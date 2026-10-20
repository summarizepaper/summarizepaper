"""Papers from other sources than arXiv (sources.py) and bioRxiv / medRxiv in the digests. No network."""
import datetime
import json
from unittest import mock

from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import Client, TestCase

from summarizer import citations, digest, exports, library, sources, trends, utils
from summarizer.models import AnnouncedPaper, ArxivBatch, ArxivPaper, DigestSubscription, ExternalPaper

MONDAY = datetime.datetime(2026, 10, 19, 7, 0, tzinfo=datetime.timezone.utc)
MNRAS = '10.1093/mnras/stad1234'
MNRAS_KEY = 'doi_10.1093--mnras--stad1234'


def journal_paper(**fields):
    values = dict(key=MNRAS_KEY, source='openalex', source_name='Monthly Notices of the Royal Astronomical Society',
                  title='Galaxy quenching time-scales', abstract='We reconstruct colours.', authors=['Ann Lee', 'Bo Chen'],
                  published='2023-04-27', doi=MNRAS, url='https://doi.org/' + MNRAS,
                  journal='Monthly Notices of the Royal Astronomical Society', category='Physics and Astronomy',
                  license='cc-by', text=[('pdf', 'https://example.org/paper.pdf')])
    values.update(fields)
    return sources._paper(**values)


class KeyTests(TestCase):
    def test_doi_keys_round_trip(self):
        self.assertEqual(sources.doi_key('10.1093/MNRAS/stad1234'), MNRAS_KEY)
        self.assertEqual(sources.key_doi(MNRAS_KEY), MNRAS)
        self.assertTrue(sources.is_external(MNRAS_KEY))
        self.assertFalse(sources.is_external('2101.00001v1'))
        self.assertFalse(sources.is_external('hep-th--9901001v1'))
        self.assertTrue(sources.valid_key(MNRAS_KEY))
        self.assertEqual(sources.label(MNRAS_KEY), 'DOI: ' + MNRAS)
        self.assertEqual(sources.label('2101.00001v1'), 'arXiv: 2101.00001v1')

    def test_doi_with_unsafe_characters_is_hashed_and_found_again(self):
        doi = '10.1016/s0006-3495(98)77654-3'
        key = sources.doi_key(doi)
        self.assertTrue(key.startswith('doi_h'))
        self.assertTrue(sources.valid_key(key))
        ExternalPaper.objects.create(key=key, source='openalex', doi=doi)
        self.assertEqual(sources.key_doi(key), doi)
        # '--' in a DOI could not be read back from the key
        self.assertTrue(sources.doi_key('10.1234/a--b').startswith('doi_h'))

    def test_clean_doi_from_links(self):
        self.assertEqual(sources.clean_doi('https://doi.org/10.1093/mnras/stad1234.'), MNRAS)
        self.assertEqual(sources.clean_doi('https://www.biorxiv.org/content/10.1101/2024.09.26.615260v2.full.pdf'),
                         '10.1101/2024.09.26.615260')
        self.assertEqual(sources.clean_doi('doi:10.31234/osf.io/dz6ut_v1'), '10.31234/osf.io/dz6ut_v1')
        self.assertIsNone(sources.clean_doi('2101.00001'))

    def test_resolve(self):
        self.assertEqual(sources.resolve('https://doi.org/10.48550/arXiv.2301.03702'), ('arxiv', '2301.03702'))
        self.assertEqual(sources.resolve('10.1093/mnras/stad1234'), ('key', MNRAS_KEY))
        self.assertIsNone(sources.resolve('dark matter halos'))
        with mock.patch('summarizer.sources.epmc_search', return_value=[{
                'pmcid': 'PMC12967033', 'doi': '10.1093/tas/txaf131', 'title': 'Beef cows', 'isOpenAccess': 'Y',
                'inEPMC': 'Y', 'license': 'cc by'}]):
            self.assertEqual(sources.resolve('PMC12967033'), ('key', 'doi_10.1093--tas--txaf131'))
        self.assertTrue(ExternalPaper.objects.filter(key='doi_10.1093--tas--txaf131').exists())


class LicenseTests(TestCase):
    def test_normalize(self):
        cases = {
            'cc-by': 'https://creativecommons.org/licenses/by/',
            'cc-by-nc-nd': 'https://creativecommons.org/licenses/by-nc-nd/',
            'cc by-nc': 'https://creativecommons.org/licenses/by-nc/',
            'cc0': 'https://creativecommons.org/publicdomain/zero/1.0/',
            'public-domain': 'https://creativecommons.org/publicdomain/mark/1.0/',
            'http://creativecommons.org/licenses/by-sa/4.0/legalcode': 'https://creativecommons.org/licenses/by-sa/4.0/',
            'other-oa': '',
            'publisher-specific-oa': '',
            'https://www.elsevier.com/tdm/userlicense/1.0/': '',
            'http://arxiv.org/licenses/nonexclusive-distrib/1.0/': 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/',
        }
        for value, expected in cases.items():
            self.assertEqual(sources.normalize_license(value), expected, value)
        self.assertEqual(sources.normalize_license(sources._biorxiv_license('cc_by_nc_nd')),
                         'https://creativecommons.org/licenses/by-nc-nd/4.0/')
        self.assertEqual(sources._biorxiv_license('cc_no'), '')

    def test_open_and_label(self):
        self.assertTrue(sources.is_open_license('http://creativecommons.org/licenses/by/4.0/'))
        self.assertTrue(sources.is_open_license('https://creativecommons.org/licenses/by-nc/'))
        self.assertTrue(sources.is_open_license('http://creativecommons.org/publicdomain/zero/1.0/'))
        self.assertFalse(sources.is_open_license('https://creativecommons.org/licenses/by-nc-nd/4.0/'))
        self.assertFalse(sources.is_open_license('http://arxiv.org/licenses/nonexclusive-distrib/1.0/'))
        self.assertFalse(sources.is_open_license(''))
        self.assertEqual(sources.license_label('http://creativecommons.org/licenses/by-nc-sa/4.0/'), 'CC BY-NC-SA 4.0')
        self.assertEqual(sources.license_label('https://creativecommons.org/licenses/by/'), 'CC BY')
        self.assertEqual(sources.license_label('http://arxiv.org/licenses/nonexclusive-distrib/1.0/'), 'NONEXCLUSIVE-DISTRIB 1.0')


JATS = b"""<?xml version="1.0"?>
<article xmlns:mml="http://www.w3.org/1998/Math/MathML" xmlns:xlink="http://www.w3.org/1999/xlink">
<front><article-meta><abstract><p>Not in the body.</p></abstract></article-meta></front>
<body>
<sec><title>Introduction</title><p>Stars form in <italic>clouds</italic> of mass <inline-formula><tex-math>M_\\odot</tex-math></inline-formula> [<xref>1</xref>].</p>
<fig><caption><p>A cloud.</p></caption></fig>
<sec><title>Model</title><disp-formula><mml:math alttext="E = mc^2"><mml:mi>E</mml:mi></mml:math></disp-formula></sec></sec>
</body>
<back><ref-list><ref>Reference one</ref></ref-list></back>
</article>"""

TEI = b"""<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0"><text><body>
<div><head>Methods</head><p>We used a telescope.</p><formula>x = 1</formula>
<figure><figDesc>The telescope.</figDesc></figure></div>
</body><back><div type="references"><listBibl><biblStruct>Ref</biblStruct></listBibl></div></back></text></TEI>"""


class FullTextTests(TestCase):
    def test_jats(self):
        text = sources.xml_to_text(JATS)
        self.assertIn('## Introduction', text)
        self.assertIn('Stars form in clouds of mass $M_\\odot$ [1].', text)
        self.assertIn('Figure: A cloud.', text)
        self.assertIn('## Model', text)
        self.assertIn('$$E = mc^2$$', text)
        self.assertNotIn('Reference one', text)
        self.assertNotIn('Not in the body', text)

    def test_tei(self):
        text = sources.xml_to_text(TEI)
        self.assertEqual(text, '## Methods\n\nWe used a telescope.\n\n$$x = 1$$\n\nFigure: The telescope.')

    def test_full_text_tries_each_copy_then_falls_back_to_the_abstract(self):
        ExternalPaper.objects.create(key=MNRAS_KEY, source='openalex', doi=MNRAS, fetched=datetime.datetime.now(datetime.timezone.utc),
                                     metadata=json.dumps(journal_paper(text=[('pdf', 'https://a/x.pdf'), ('jats', 'https://b/x.xml')])))
        with mock.patch('summarizer.sources._download', side_effect=[None, JATS.replace(b'Stars form', b'Stars form ' + b'very ' * 500)]):
            text, source = sources.full_text(MNRAS_KEY)
        self.assertEqual(source, 'xml')
        self.assertIn('## Introduction', text)
        with mock.patch('summarizer.sources._download', return_value=None):
            text, source = sources.full_text(MNRAS_KEY)
        self.assertEqual(source, 'abstract')
        self.assertIn('We reconstruct colours.', text)

    def test_openalex_copy_needs_a_key(self):
        with self.settings(OPENALEX_API_KEY=''), mock.patch('summarizer.sources._download') as download:
            self.assertEqual(sources.read_candidate('openalex_tei', 'W1'), ('', ''))
        download.assert_not_called()

    def test_the_arxiv_version_is_read_when_its_license_allows(self):
        arxiv_row = [1] + [''] * 11 + ['http://creativecommons.org/licenses/by/4.0/', '']
        with mock.patch('summarizer.utils.get_arxiv_metadata', mock.AsyncMock(return_value=arxiv_row)), \
             mock.patch('summarizer.paper_source.fetch_arxiv_html', return_value='arXiv text') as html:
            self.assertEqual(sources.read_candidate('arxiv', '2301.03702'), ('arXiv text', 'html'))
        html.assert_called_once_with('2301.03702')
        arxiv_row[12] = 'http://arxiv.org/licenses/nonexclusive-distrib/1.0/'
        with mock.patch('summarizer.utils.get_arxiv_metadata', mock.AsyncMock(return_value=arxiv_row)), \
             mock.patch('summarizer.paper_source.fetch_arxiv_html') as html:
            self.assertEqual(sources.read_candidate('arxiv', '2301.03702'), ('', ''))
        html.assert_not_called()

    def test_openalex_texts_are_capped_and_the_pdf_skipped_after_an_empty_tei(self):
        cache.clear()
        with self.settings(OPENALEX_API_KEY='k'), mock.patch('summarizer.sources._download', return_value=None) as download, \
             mock.patch('summarizer.sources.OPENALEX_CONTENT_PER_DAY', 2):
            for _ in range(4):
                sources.read_candidate('openalex_tei', 'W1')
        self.assertEqual(download.call_count, 2)
        cache.clear()
        ExternalPaper.objects.create(key=MNRAS_KEY, source='openalex', doi=MNRAS, fetched=datetime.datetime.now(datetime.timezone.utc),
                                     metadata=json.dumps(journal_paper(text=[('openalex_tei', 'W1'), ('openalex_pdf', 'W1')])))
        with self.settings(OPENALEX_API_KEY='k'), \
             mock.patch('summarizer.sources._download', return_value=TEI) as download:
            self.assertEqual(sources.full_text(MNRAS_KEY)[1], 'abstract')
        self.assertEqual(download.call_count, 1)  # the TEI only: figures, no text

    def test_read_paper_goes_to_the_source(self):
        import asyncio
        with mock.patch('summarizer.sources.full_text', return_value=('text', 'xml')) as full, \
             mock.patch('summarizer.paper_source.fetch_arxiv_html') as html:
            self.assertEqual(asyncio.run(utils.read_paper(MNRAS_KEY, 'unused')), ('text', 'xml'))
        full.assert_called_once_with(MNRAS_KEY)
        html.assert_not_called()


class MetadataTests(TestCase):
    def test_metadata_is_stored_and_reused_for_a_day(self):
        with mock.patch('summarizer.sources.fetch', return_value=journal_paper()) as fetch:
            self.assertEqual(sources.metadata(MNRAS_KEY)['title'], 'Galaxy quenching time-scales')
            sources.metadata(MNRAS_KEY)
        fetch.assert_called_once()
        with mock.patch('summarizer.sources.fetch', side_effect=sources.SourceUnavailable('down')):
            self.assertEqual(sources.metadata(MNRAS_KEY, refresh=True)['title'], 'Galaxy quenching time-scales')

    def test_list_form_for_the_summary_pipeline(self):
        with mock.patch('summarizer.sources.fetch', return_value=journal_paper()):
            row = sources.metadata_list(MNRAS_KEY)
        self.assertEqual(row[0], 1)
        self.assertEqual(row[1], ['Ann Lee', 'Bo Chen'])
        self.assertEqual(row[5], 'https://doi.org/' + MNRAS)
        self.assertEqual(row[9], '2023-04-27T00:00:00Z')
        self.assertEqual(row[12], 'https://creativecommons.org/licenses/by/')
        self.assertIn('Published in: Monthly Notices', row[13])
        with mock.patch('summarizer.sources.fetch', return_value=None):
            self.assertEqual(sources.metadata_list('doi_10.1/unknown')[0], 0)

    def test_openalex_work(self):
        work = {
            'id': 'https://openalex.org/W1', 'doi': 'https://doi.org/' + MNRAS, 'display_name': 'Quenching',
            'authorships': [{'author': {'display_name': 'Ann Lee'}, 'institutions': [{'display_name': 'UWA'}]}],
            'abstract_inverted_index': {'colours': [1], 'We': [0]}, 'publication_date': '2023-04-27',
            'primary_location': {'landing_page_url': 'https://doi.org/x', 'source': {'display_name': 'MNRAS'}},
            'best_oa_location': {'license': 'cc-by'}, 'type': 'article', 'ids': {},
            'locations': [{'pdf_url': 'https://academic.oup.com/x.pdf', 'source': {'type': 'journal'}},
                          {'landing_page_url': 'http://arxiv.org/abs/2301.03702', 'pdf_url': 'https://arxiv.org/pdf/2301.03702',
                           'source': {'type': 'repository'}}],
            'has_content': {'grobid_xml': True, 'pdf': True},
        }
        paper = sources.openalex_paper(work)
        self.assertEqual(paper['abstract'], 'We colours')
        self.assertEqual(paper['affiliations'], ['UWA'])
        self.assertEqual(paper['arxiv_id'], '2301.03702')
        self.assertEqual([k for k, _ in paper['text']], ['pdf', 'epmc_doi', 'arxiv', 'openalex_tei', 'openalex_pdf'])
        self.assertNotIn('arxiv.org', paper['text'][0][1])

    def test_biorxiv_row(self):
        row = {'title': 'Spinal serotonin', 'authors': 'Fenstermacher, S.; Dasen, J.', 'doi': '10.1101/2024.09.26.615260',
               'date': '2024-10-01', 'version': '2', 'license': 'cc_by', 'category': 'cell biology',
               'jatsxml': 'https://www.biorxiv.org/x.source.xml', 'abstract': 'A.', 'published': 'NA'}
        paper = sources.biorxiv_paper(row, 'biorxiv')
        self.assertEqual(paper['key'], 'doi_10.1101--2024.09.26.615260')
        self.assertEqual(paper['authors'], ['S. Fenstermacher', 'J. Dasen'])
        self.assertEqual(paper['category'], 'biorxiv.cell-biology')
        self.assertTrue(sources.is_open_license(paper['license']))
        self.assertEqual(paper['text'][1], ('pdf', 'https://www.biorxiv.org/content/10.1101/2024.09.26.615260v2.full.pdf'))


class PaperPageTests(TestCase):
    def test_home_form_opens_a_doi(self):
        r = Client().post('/en/', {'arxiv_id': 'https://doi.org/10.1093/mnras/stad1234'})
        self.assertRedirects(r, '/en/arxiv-id/%s/' % MNRAS_KEY, fetch_redirect_response=False)
        r = Client().post('/en/', {'arxiv_id': '10.48550/arXiv.2301.03702'})
        self.assertRedirects(r, '/en/arxiv-id/2301.03702/', fetch_redirect_response=False)
        r = Client().post('/en/', {'arxiv_id': 'black holes', 'src': 'open'})
        self.assertRedirects(r, '/en/search-results/?q=black+holes&page=1&src=open', fetch_redirect_response=False)

    def test_page_of_a_journal_paper(self):
        with mock.patch('summarizer.sources.fetch', return_value=journal_paper()):
            r = Client().get('/en/arxiv-id/%s/' % MNRAS_KEY)
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'Galaxy quenching time-scales')
        self.assertContains(r, 'DOI: ' + MNRAS)
        self.assertContains(r, 'CC BY')
        self.assertContains(r, 'Published on')
        self.assertNotContains(r, 'The license of this specific paper does not allow')

    def test_a_closed_doi_with_an_arxiv_version_goes_to_the_arxiv_page(self):
        with mock.patch('summarizer.sources.fetch', return_value=journal_paper(license='', arxiv_id='2301.03702')):
            r = Client().get('/en/arxiv-id/%s/' % MNRAS_KEY)
        self.assertRedirects(r, '/en/arxiv-id/2301.03702/', fetch_redirect_response=False)

    def test_upper_case_and_bad_keys(self):
        r = Client().get('/en/arxiv-id/DOI_10.1093--MNRAS--stad1234/')
        self.assertRedirects(r, '/en/arxiv-id/%s/' % MNRAS_KEY, fetch_redirect_response=False)
        self.assertEqual(Client().get('/en/arxiv-id/doi_%3Cscript%3E/').status_code, 404)

    def test_unknown_and_unavailable(self):
        with mock.patch('summarizer.sources.fetch', return_value=None):
            self.assertContains(Client().get('/en/arxiv-id/doi_10.1--nothing/'), 'No paper was found with this identifier')
        with mock.patch('summarizer.sources.fetch', side_effect=sources.SourceUnavailable('down')):
            self.assertEqual(Client().get('/en/arxiv-id/doi_10.1--other/').status_code, 503)

    def test_open_search(self):
        results = [{'key': MNRAS_KEY, 'title': 'Galaxy quenching', 'authors': ['Ann Lee'], 'more_authors': False,
                    'venue': 'MNRAS', 'published': '2023-04-27', 'license': 'CC BY', 'open': True, 'arxiv_id': '2301.03702'}]
        with mock.patch('summarizer.sources.search', return_value=results) as search:
            r = Client().get('/en/search-results/', {'q': 'quenching', 'src': 'open'})
            Client().get('/en/search-results/', {'q': 'quenching', 'src': 'open', 'page': 2})
        search.assert_called_once()  # the second page comes from the cache
        self.assertContains(r, '/en/arxiv-id/%s/' % MNRAS_KEY)
        self.assertContains(r, 'Full text summarized')
        self.assertContains(r, 'href="/en/arxiv-id/2301.03702/">Also on arXiv')
        cache.clear()
        with mock.patch('summarizer.sources.search', side_effect=sources.SourceUnavailable('down')):
            r = Client().get('/en/search-results/', {'q': 'quenching', 'src': 'open'})
        self.assertContains(r, 'is not responding right now')


class ToolsTests(TestCase):
    def setUp(self):
        self.paper = ArxivPaper.objects.create(arxiv_id=MNRAS_KEY, title='Galaxy quenching', link_doi='https://doi.org/' + MNRAS,
                                               journal_ref='Monthly Notices of the Royal Astronomical Society',
                                               published_arxiv=datetime.date(2023, 4, 27))

    def test_bibtex_and_ris(self):
        meta = exports.paper_meta(MNRAS_KEY)
        bib = exports.bibtex_entry(meta)
        self.assertTrue(bib.startswith('@article{'))
        self.assertIn('journal={Monthly Notices of the Royal Astronomical Society}', bib)
        self.assertIn('doi={%s}' % MNRAS, bib)
        self.assertNotIn('arXiv', bib)
        ris = exports.ris_entry(meta)
        self.assertIn('TY  - JOUR', ris)
        self.assertIn('UR  - https://doi.org/' + MNRAS, ris)
        self.assertEqual(exports.file_name(MNRAS_KEY), '10.1093_mnras_stad1234')

    def test_collections_and_citations_take_dois(self):
        self.assertEqual(library.canonical_id('https://doi.org/10.1093/mnras/stad1234'), MNRAS_KEY)
        self.assertEqual(library.canonical_id(MNRAS_KEY), MNRAS_KEY)
        self.assertEqual(library.canonical_id('arXiv:2101.00001v2'), '2101.00001v2')
        self.assertEqual(library.lookup(MNRAS_KEY)[0], 'Galaxy quenching')
        self.assertEqual(citations.base_id(MNRAS_KEY), MNRAS_KEY)
        self.assertEqual(citations.base_id('10.1093/mnras/stad1234'), MNRAS_KEY)
        self.assertEqual(citations.base_id('hal_hal-01234567'), None)  # no DOI: Semantic Scholar would not know it
        self.assertEqual(citations.s2_ref(MNRAS_KEY), 'DOI:' + MNRAS)
        self.assertEqual(citations.s2_ref('2101.00001'), 'arXiv:2101.00001')


def bio_row(doi, version='1', category='neuroscience', date='2026-10-16'):
    return {'doi': doi, 'version': version, 'title': 'Preprint %s' % doi, 'authors': 'Lee, A.', 'date': date,
            'license': 'cc_by', 'category': category, 'abstract': 'Abstract.', 'jatsxml': ''}


class BioDigestTests(TestCase):
    def feeds(self, failing=(), day=datetime.date(2026, 10, 19)):
        def read(name):
            if name in failing:
                raise digest.ArxivUnavailable(name)
            return day, [{'arxiv_id': '%s-p' % name, 'title': 'T', 'abstract': 'A', 'authors': ['A B'],
                          'categories': [name + '.XX'], 'announced': day,
                          'published': datetime.datetime.combine(day, datetime.time(), tzinfo=digest.ANNOUNCE_TZ)}], 10
        return read

    def test_window(self):
        monday = datetime.date(2026, 10, 19)
        self.assertEqual(digest.bio_window(monday), (datetime.date(2026, 10, 15), datetime.date(2026, 10, 17)))
        ArxivBatch.objects.create(day=monday)
        self.assertEqual(digest.bio_window(monday + datetime.timedelta(days=1)), (datetime.date(2026, 10, 18),) * 2)
        # after an arXiv holiday the days it did not take go with the next announcement
        self.assertEqual(digest.bio_window(monday + datetime.timedelta(days=2)),
                         (datetime.date(2026, 10, 18), datetime.date(2026, 10, 19)))

    def test_only_first_versions_are_read(self):
        pages = [([bio_row('10.64898/a'), bio_row('10.64898/b', version='2')], 3), ([bio_row('10.64898/c')], 3)]
        with mock.patch('summarizer.digest.bio_page', side_effect=pages) as page:
            papers = digest.read_bio('biorxiv', datetime.date(2026, 10, 19))
        self.assertEqual([p['arxiv_id'] for p in papers], ['doi_10.64898--a', 'doi_10.64898--c'])
        self.assertEqual(papers[0]['categories'], ['biorxiv.neuroscience'])
        self.assertEqual(page.call_args_list[1].args[3], 2)  # the cursor moves by the rows read

    def test_bio_papers_go_into_the_batch(self):
        with mock.patch('summarizer.digest.read_feed', side_effect=self.feeds()), \
             mock.patch('summarizer.digest.bio_page', side_effect=lambda s, *a: ([bio_row('10.64898/%s1' % s)], 1)):
            self.assertEqual(digest.ensure_batch(MONDAY), [MONDAY.date()])
        paper = AnnouncedPaper.objects.get(arxiv_id='doi_10.64898--medrxiv1')
        self.assertEqual(paper.categories, 'medrxiv.neuroscience')
        self.assertEqual(paper.announced, MONDAY.date())
        self.assertEqual(ArxivBatch.objects.get().missing, '')

    def test_a_failing_bio_server_holds_back_only_its_digests(self):
        with mock.patch('summarizer.digest.read_feed', side_effect=self.feeds()), \
             mock.patch('summarizer.digest.read_bio', side_effect=digest.ArxivUnavailable('biorxiv')):
            self.assertEqual(digest.ensure_batch(MONDAY), [MONDAY.date()])  # arXiv digests are not delayed
        self.assertEqual(ArxivBatch.objects.get().missing_list(), ['biorxiv', 'medrxiv'])
        astro = DigestSubscription.objects.create(user=User.objects.create_user('a', email='a@example.com'),
                                                  categories='astro-ph.XX', timezone='UTC', send_hour=7)
        DigestSubscription.objects.create(user=User.objects.create_user('b', email='b@example.com'),
                                          categories='biorxiv.neuroscience', timezone='UTC', send_hour=7)
        blurbs = lambda batch, lang: {p['arxiv_id']: 'Blurb.' for p in batch}  # noqa: E731
        with mock.patch('summarizer.digest.read_bio', side_effect=digest.ArxivUnavailable('biorxiv')), \
             mock.patch('summarizer.digest.write_blurbs', side_effect=blurbs), \
             mock.patch('summarizer.similar.embed', side_effect=lambda texts: __import__('numpy').ones((len(texts), 4), dtype='float32') / 2):
            self.assertEqual(digest.run(now=MONDAY), 1)
        self.assertEqual([m.to for m in mail.outbox], [['a@example.com']])
        astro.refresh_from_db()
        self.assertIsNotNone(astro.last_sent)
        row = bio_row('10.64898/n1', date='2026-10-16')
        with mock.patch('summarizer.digest.bio_page', return_value=([row], 1)), \
             mock.patch('summarizer.digest.write_blurbs', side_effect=blurbs), \
             mock.patch('summarizer.similar.embed', side_effect=lambda texts: __import__('numpy').ones((len(texts), 4), dtype='float32') / 2):
            self.assertEqual(digest.run(now=MONDAY + datetime.timedelta(hours=1)), 1)
        self.assertEqual(mail.outbox[1].to, ['b@example.com'])
        self.assertIn('Preprint 10.64898/n1', mail.outbox[1].body)

    def test_no_bio_papers_kept_on_an_arxiv_holiday(self):
        friday = datetime.date(2026, 10, 16)
        with mock.patch('summarizer.digest.read_feed', side_effect=self.feeds(day=friday)), \
             mock.patch('summarizer.digest.bio_page', return_value=([bio_row('10.64898/h')], 1)):
            digest.ensure_batch(MONDAY)
        self.assertFalse(AnnouncedPaper.objects.filter(arxiv_id='doi_10.64898--h').exists())

    def test_form_offers_bio_categories(self):
        user = User.objects.create_user('f', email='f@example.com', password='x')
        c = Client()
        c.force_login(user)
        r = c.post('/en/digest/', {'digest': 'new', 'categories': ['biorxiv.cell-biology', 'medrxiv.hiv-aids']})
        self.assertEqual(r.status_code, 302)
        self.assertEqual(DigestSubscription.objects.get().categories, 'biorxiv.cell-biology,medrxiv.hiv-aids')
        page = c.get('/en/digest/')
        self.assertContains(page, 'value="biorxiv.cell-biology" checked')
        self.assertEqual(digest.category_name('medrxiv.hiv-aids'), 'HIV AIDS')
        self.assertEqual(digest.category_name('astro-ph.EP'), 'astro-ph.EP')

    def test_health_alert_when_a_bio_server_stays_missing(self):
        tuesday_9am = datetime.datetime(2026, 10, 20, 13, 0, tzinfo=datetime.timezone.utc)  # 09:00 in New York
        ArxivBatch.objects.create(day=datetime.date(2026, 10, 20), missing='medrxiv')
        DigestSubscription.objects.create(user=User.objects.create_user('h', email='h@example.com'), categories='astro-ph.EP')
        self.assertEqual(digest.check_health(tuesday_9am), [])  # nobody follows medRxiv
        DigestSubscription.objects.create(user=User.objects.create_user('m', email='m@example.com'), categories='medrxiv.neurology')
        with self.assertLogs('summarizer.digest', 'ERROR'):
            problems = digest.check_health(tuesday_9am)
        self.assertEqual(problems, ['medRxiv still missing from the announcement of 2026-10-20 at 09:00 in New York'])
        cache.clear()

    def test_rising_topics_of_all_arxiv_leave_bio_papers_out(self):
        day = MONDAY.date()
        AnnouncedPaper.objects.create(arxiv_id='2610.1v1', announced=day, title='Exoplanet atmospheres', categories='astro-ph.EP')
        AnnouncedPaper.objects.create(arxiv_id='doi_10.64898--x', announced=day, title='Neuron spikes', categories='biorxiv.neuroscience')
        trends.count_day(day)
        scopes = {t.scope: t.papers for t in trends.TermDay.objects.filter(day=day)}
        self.assertEqual(scopes, {'*': 1, 'astro-ph': 1, 'biorxiv': 1})


class PipelineTests(TestCase):
    """The paper page pipeline (consumers.LoadingConsumer) on a DOI: metadata, text and summary are stored."""

    def test_summary_of_a_journal_paper(self):
        from django.test import override_settings
        from summarizer.consumers import summarize_now
        from summarizer.models import PaperText, SummaryPaper
        text = '## Introduction\n\n' + 'Galaxies stop forming stars. ' * 200
        layers = {'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}}
        with override_settings(CHANNEL_LAYERS=layers), \
             mock.patch('summarizer.sources.metadata', return_value=journal_paper()), \
             mock.patch('summarizer.sources.read_candidate', return_value=(text, 'xml')) as read, \
             mock.patch('summarizer.utils.summarize_book', mock.AsyncMock(return_value='Long summary.')), \
             mock.patch('summarizer.utils.finalise_and_keywords', mock.AsyncMock(return_value=('Summary.', 'galaxies, quenching'))), \
             mock.patch('summarizer.utils.extract_key_points', mock.AsyncMock(return_value=['Point one.'])), \
             mock.patch('summarizer.utils.extract_simple_summary', mock.AsyncMock(return_value='Simple.')), \
             mock.patch('summarizer.utils.extract_blog_article', mock.AsyncMock(return_value='<p>Blog</p>')), \
             mock.patch('summarizer.utils.refine_blog_article', mock.AsyncMock(return_value='<p>Blog</p>')), \
             mock.patch('summarizer.similar.find_and_store', return_value=[]):
            summarize_now(MNRAS_KEY, 'en')
        read.assert_called_once_with('pdf', 'https://example.org/paper.pdf')
        paper = ArxivPaper.objects.get(arxiv_id=MNRAS_KEY)
        self.assertEqual(paper.title, 'Galaxy quenching time-scales')
        self.assertEqual(paper.link_doi, 'https://doi.org/' + MNRAS)
        self.assertEqual(paper.journal_ref, 'Monthly Notices of the Royal Astronomical Society')
        self.assertEqual(paper.published_arxiv, datetime.date(2023, 4, 27))
        self.assertEqual([a.name for a in paper.authors.all()], ['Ann Lee', 'Bo Chen'])
        stored = PaperText.objects.get(arxiv_id=MNRAS_KEY)
        self.assertEqual(stored.source, 'xml')
        self.assertIn('Published in: Monthly Notices', stored.text)
        self.assertIn('Galaxies stop forming stars.', stored.text)
        self.assertEqual(SummaryPaper.objects.get(paper=paper, lang='en').summary, 'Summary.')
        page = Client().get('/en/arxiv-id/%s/' % MNRAS_KEY)
        self.assertContains(page, 'Summary.')
        self.assertContains(page, 'DOI: ' + MNRAS)
        self.assertContains(page, 'Cite this paper')
        bib = Client().get('/en/bibtex/%s/' % MNRAS_KEY)
        self.assertContains(bib, '@article{')

    def test_closed_license_is_summarized_from_the_abstract(self):
        from django.test import override_settings
        from summarizer.consumers import summarize_now
        from summarizer.models import PaperText
        layers = {'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}}
        book = mock.AsyncMock(return_value='Long summary.')
        with override_settings(CHANNEL_LAYERS=layers), \
             mock.patch('summarizer.sources.metadata', return_value=journal_paper(license='cc-by-nc-nd')), \
             mock.patch('summarizer.sources.read_candidate') as read, \
             mock.patch('summarizer.utils.summarize_book', book), \
             mock.patch('summarizer.utils.finalise_and_keywords', mock.AsyncMock(return_value=('Summary.', 'x'))), \
             mock.patch('summarizer.utils.extract_key_points', mock.AsyncMock(return_value=['Point.'])), \
             mock.patch('summarizer.utils.extract_simple_summary', mock.AsyncMock(return_value='Simple.')), \
             mock.patch('summarizer.utils.extract_blog_article', mock.AsyncMock(return_value='<p>B</p>')), \
             mock.patch('summarizer.utils.refine_blog_article', mock.AsyncMock(return_value='<p>B</p>')), \
             mock.patch('summarizer.similar.find_and_store', return_value=[]):
            summarize_now(MNRAS_KEY, 'en')
        read.assert_not_called()
        self.assertEqual(PaperText.objects.get(arxiv_id=MNRAS_KEY).source, 'abstract')
        self.assertIn('We reconstruct colours.', book.call_args.args[2])
        self.assertContains(Client().get('/en/arxiv-id/%s/' % MNRAS_KEY), 'CC BY-NC-ND')
