import json
from unittest import mock

import requests
from bs4 import BeautifulSoup
from django.core.cache import cache
from django.template.loader import render_to_string
from django.test import RequestFactory, TestCase, override_settings
from django.utils import translation

from . import chat_figures as figures
from .models import ArxivPaper, ExternalPaper


HTML = '''<figure class="ltx_figure" id="S2.F3"><img src="2401.04088v1/x3.png">
<figure><img src="/html/2401.04088v1/x4.png"></figure>
<img src="https://evil.example/track.png"><object data="javascript:alert(1)"></object>
<figcaption><span class="ltx_tag_figure">Figure 3:</span> A <b>spectrum</b>.</figcaption></figure>'''


@override_settings(CACHES={'default': {'BACKEND': 'django.core.cache.backends.locmem.LocMemCache'}})
class ChatFiguresTests(TestCase):
    def setUp(self):
        cache.clear()
        self.paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Example paper',
                                               license='http://creativecommons.org/licenses/by/4.0/')

    def fetch(self, key=None):
        return self.client.get('/en/paper-figures/%s/' % (key or self.paper.arxiv_id))

    def response(self, status=200, html=HTML):
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status_code = status
        response.iter_content.return_value = [html.encode()]
        return response

    def test_gallery_groups_panels_preserves_caption_and_uses_official_source(self):
        with mock.patch.object(figures.requests, 'get', return_value=self.response()) as get:
            data = self.fetch().json()
        self.assertEqual(data['status'], 'ready')
        self.assertEqual(data['paper_id'], self.paper.arxiv_id)
        self.assertEqual(len(data['figures']), 1)
        figure = data['figures'][0]
        self.assertEqual(figure['images'], ['https://arxiv.org/html/2401.04088v1/x3.png',
                                           'https://arxiv.org/html/2401.04088v1/x4.png'])
        self.assertEqual(figure['caption'], 'Figure 3: A spectrum.')
        self.assertEqual(figure['label'], 'Figure 3:')
        self.assertTrue(figure['source_url'].endswith('#S2.F3'))
        self.assertFalse(get.call_args.kwargs['allow_redirects'])

    def test_cache_is_per_paper_and_license_is_rechecked(self):
        with mock.patch.object(figures.requests, 'get', return_value=self.response()) as get:
            self.fetch()
            self.fetch()
            self.assertEqual(get.call_count, 1)
            other = ArxivPaper.objects.create(arxiv_id='2401.04089v1', license=self.paper.license)
            self.fetch(other.arxiv_id)
            self.assertEqual(get.call_count, 2)
            self.paper.license = ''
            self.paper.save()
            self.assertEqual(self.fetch().json()['status'], 'license')
            self.assertEqual(get.call_count, 2)

    def test_missing_and_unsupported_papers_do_not_fetch_or_generate(self):
        with mock.patch.object(figures.requests, 'get') as get:
            self.assertEqual(self.fetch('2401.99999v1').status_code, 404)
            self.assertEqual(self.fetch('2401.99999v1').json()['status'], 'not_ready')
            self.paper.arxiv_id = 'doi_10.1234--example'
            self.paper.save()
            self.assertEqual(self.fetch().json()['status'], 'unsupported')
            get.assert_not_called()

    def test_redirect_is_not_followed(self):
        response = self.response(status=302)
        response.headers = {'Location': 'http://169.254.169.254/'}
        with mock.patch.object(figures.requests, 'get', return_value=response) as get:
            self.assertEqual(self.fetch().json()['status'], 'empty')
            get.assert_called_once()
            response.iter_content.assert_not_called()

    def test_timeout_and_size_limit_can_be_retried(self):
        with mock.patch.object(figures.requests, 'get', side_effect=requests.Timeout):
            self.assertEqual(self.fetch().status_code, 503)
        with mock.patch.object(figures, 'MAX_HTML', 10), mock.patch.object(figures.requests, 'get', return_value=self.response()):
            self.assertEqual(self.fetch().json()['status'], 'unavailable')
        with mock.patch.object(figures.requests, 'get', return_value=self.response()):
            self.assertEqual(self.fetch().json()['status'], 'ready')

    def test_external_mapping_is_validated_and_pmc_captions_are_supported(self):
        self.paper.arxiv_id = 'doi_10.1234--example'
        self.paper.save()
        row = ExternalPaper.objects.create(key=self.paper.arxiv_id, source='europepmc',
                                          metadata=json.dumps({'text': [['epmc', 'PMC12345']]}))
        html = '<figure id="F1"><img src="https://cdn.ncbi.nlm.nih.gov/pmc/figure.jpg"><figcaption>Fig. 1. Result</figcaption></figure>'
        with mock.patch.object(figures.requests, 'get', return_value=self.response(html=html)) as get:
            data = self.fetch().json()
            self.assertEqual(data['status'], 'ready')
            self.assertEqual(data['figures'][0]['label'], 'Fig. 1')
            self.assertEqual(get.call_args.args[0], 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12345/')
        for metadata in ['[]', '{"text":[["epmc","https://evil.example"]]}', '{"text":null}', 'invalid']:
            row.metadata = metadata
            row.save()
            self.assertIsNone(figures.html_source(self.paper))

    def test_only_trusted_https_images_and_bounded_figures(self):
        for url in ['http://arxiv.org/a.png', 'https://arxiv.org.evil.test/a.png',
                    'https://a:b@arxiv.org/a.png', 'https://arxiv.org:444/a.png',
                    'data:image/svg+xml,abc', 'javascript:alert(1)', 'https://arxiv.org:bad/a']:
            self.assertFalse(figures.safe_image(url), url)
        self.assertEqual(len(figures.extract(HTML * 80, 'https://arxiv.org/html/2401.04088v1', 'arxiv')), 40)
        self.assertEqual(figures.extract('<figure><img src="a.png"></figure>', 'https://arxiv.org/html/id', 'arxiv'), [])

    def test_get_only_and_shared_localized_component(self):
        self.assertEqual(self.client.post('/en/paper-figures/2401.04088v1/').status_code, 405)
        request = RequestFactory().get('/fr/')
        with translation.override('fr'):
            html = render_to_string('summarizer/chat_figures.html', {'figure_paper': self.paper.arxiv_id}, request)
        self.assertIn('Figures et graphiques', html)
        self.assertIn('data-paper="2401.04088v1"', html)
        self.assertTrue(BeautifulSoup(html, 'html.parser').select_one('.chat-figures-picker').has_attr('hidden'))
        self.assertIn('/fr/paper-figures/PAPER_ID/', html)
        empty = render_to_string('summarizer/chat_figures.html', {}, request)
        self.assertIn('data-paper="" data-title=""', empty)
        page = self.client.get('/en/arxiv-id/2401.04088v1/')
        self.assertEqual(page.status_code, 200)
        self.assertContains(page, 'data-paper="2401.04088v1" data-title="Example paper"')
        from pathlib import Path
        for template in ['chat.html', 'arxividpage.html']:
            self.assertIn("include 'summarizer/chat_figures.html'", (Path('templates/summarizer') / template).read_text())

    def test_mathml_is_replaced_once_by_its_tex_source(self):
        html = r'''<figcaption>Figure 1: Peak (<math alttext="\alpha_{\mathrm{ICRS}}"><semantics>
        <msub><mi>α</mi><mi>ICRS</mi></msub><annotation encoding="application/x-tex">\alpha_{\mathrm{ICRS}}</annotation>
        </semantics></math>, <math alttext="\delta_{\mathrm{ICRS}}"><mi>δ</mi><annotation>duplicate</annotation></math>).
        The noise (<math alttext="\sigma"><mi>σ</mi><annotation>\sigma</annotation></math>) is
        20 <math alttext="\mu"><mi>μ</mi><annotation>\mu</annotation></math>Jy beam<sup>-1</sup>.</figcaption>'''
        text = figures.caption_text(BeautifulSoup(html, 'html.parser').figcaption)
        self.assertEqual(text.count(r'\alpha_{\mathrm{ICRS}}'), 1)
        self.assertEqual(text.count(r'\sigma'), 1)
        self.assertIn(r'\(\delta_{\mathrm{ICRS}}\)', text)
        self.assertIn(r'\(\mu\)Jy', text)
        self.assertIn(r'beam\({}^{\text{-1}}\)', text)
        self.assertNotIn('duplicate', text)
        self.assertNotIn('α', text)
        self.assertNotIn('<math', text)

    def test_pmc_alternative_math_is_not_duplicated_and_annotation_is_a_fallback(self):
        html = r'''<figcaption>Ratio <inline-formula><alternatives><tex-math><![CDATA[\frac{a_1}{b_2}]]></tex-math>
        <math alttext="\frac{a_1}{b_2}"><mi>a</mi><mi>b</mi></math></alternatives></inline-formula>;
        <math><semantics><mi>x</mi><annotation encoding="application/x-tex">x_i^2</annotation></semantics></math>;
        <math><mi>y</mi><annotation encoding="application/unknown">hidden alternative</annotation></math>.
        <script>alert(1)</script></figcaption>'''
        text = figures.caption_text(BeautifulSoup(html, 'html.parser').figcaption)
        self.assertEqual(text.count(r'\frac{a_1}{b_2}'), 1)
        self.assertIn(r'\(x_i^2\)', text)
        self.assertIn('y', text)
        self.assertNotIn('hidden alternative', text)
        self.assertNotIn('alert', text)

    def test_formula_images_in_captions_are_not_treated_as_figure_panels(self):
        html = '''<figure class="ltx_figure"><img src="paper/plot.png"><figcaption>Figure 1:
        <img src="paper/formula.png" alt="x"> flux<sub>dust</sub></figcaption></figure>'''
        data = figures.extract(html, 'https://arxiv.org/html/paper', 'arxiv')
        self.assertEqual(data[0]['images'], ['https://arxiv.org/html/paper/plot.png'])
        self.assertIn(r'flux\({}_{\text{dust}}\)', data[0]['caption'])
