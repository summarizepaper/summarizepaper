"""Source-neutral public URLs and compatibility with previously shared links."""
from unittest import mock

from bs4 import BeautifulSoup
from django.core.cache import cache
from django.http import HttpResponse
from django.test import TestCase
from django.urls import reverse
from django.utils import translation

from .models import ArxivPaper, SummaryPaper
from .sitemaps import SummarizedPaperSitemap


class PaperURLTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_reverse_and_model_url_use_neutral_path_for_every_source(self):
        for lang in ('en', 'fr'):
            with translation.override(lang):
                for key in ('2401.04088v1', 'hep-th--9901001v2', 'doi_10.1093--mnras--stad1234', 'pmc_123456'):
                    with self.subTest(lang=lang, key=key):
                        expected = f'/{lang}/paper/{key}/'
                        self.assertEqual(reverse('arxividpage', kwargs={'arxiv_id': key}), expected)
                        self.assertEqual(ArxivPaper(arxiv_id=key).get_absolute_url(), expected)

    def test_legacy_links_permanently_redirect_without_touching_paper_or_query(self):
        cases = (
            ('fr/arxiv-id/doi_10.1093--mnras--stad1234/', 'fr/paper/doi_10.1093--mnras--stad1234/'),
            ('en/arxiv-id/2401.04088v1/', 'en/paper/2401.04088v1/'),
            ('fr/arxiv-id/hep-th/9901001v2/', 'fr/paper/hep-th--9901001v2/'),
            ('en/arxiv-id/2401.04088v1/vote/', 'en/paper/2401.04088v1/vote/'),
        )
        query = '?autorun=1&utm_source=email&topic=a%2Bb&topic=c'
        with mock.patch('summarizer.views.arxividpage') as page:
            for old, new in cases:
                for method in ('get', 'head'):
                    with self.subTest(old=old, method=method):
                        response = getattr(self.client, method)('/' + old + query)
                        self.assertEqual(response.status_code, 301)
                        self.assertEqual(response['Location'], '/' + new + query)
            page.assert_not_called()

    def test_legacy_post_preserves_method_and_form_fields(self):
        # Following a 308 must send the original POST, unlike a 301/302.
        with mock.patch('summarizer.views.utils.generate_pdf', return_value=HttpResponse('pdf')) as export:
            response = self.client.post('/fr/arxiv-id/2401.04088v1/?autorun=1',
                                        {'download_pdf': '1', 'local_date': '2026-10-23'}, follow=True)
            self.assertEqual(response.redirect_chain, [('/fr/paper/2401.04088v1/?autorun=1', 308)])
            request = export.call_args.args[0]
            self.assertEqual(request.method, 'POST')
            self.assertEqual(request.POST['local_date'], '2026-10-23')
            self.assertEqual(request.GET['autorun'], '1')

    def test_canonical_metadata_and_sitemap_only_publish_neutral_paths(self):
        for key in ('2401.04088v1', 'doi_10.1093--mnras--stad1234'):
            paper = ArxivPaper.objects.create(arxiv_id=key, title='Example', abstract='Abstract')
            summary = SummaryPaper.objects.create(paper=paper, lang='fr', summary='Résumé', notes='[]')
            response = self.client.get(f'/fr/paper/{key}/')
            self.assertEqual(response.status_code, 200)
            soup = BeautifulSoup(response.content, 'html.parser')
            expected = f'https://www.summarizepaper.com/fr/paper/{key}/'
            self.assertEqual(soup.select_one('link[rel=canonical]')['href'], expected)
            self.assertEqual(soup.select_one('meta[property="og:url"]')['content'], expected)
            for alternate in soup.select('link[hreflang]'):
                self.assertIn(f'/paper/{key}/', alternate['href'])
            self.assertNotIn('/arxiv-id/', response.content.decode())
            self.assertEqual(SummarizedPaperSitemap().location(summary), f'/fr/paper/{key}/')
