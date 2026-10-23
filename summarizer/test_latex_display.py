import json
from pathlib import Path
from unittest import mock

from django.test import SimpleTestCase

from .latex_pdf import prose_html
from .models import Generated, PaperText
from .test_presskit_quality import SOURCE, content, pdf_text
from .test_tools import ToolsTestCase


QUOTE = r'Only the biggest belts with a total mass of 0.1 $\text{M}_{\oplus}$ can be shielded from photodissociation.'
TITLE = r'H$_2$O distribution in the disc of HD 100546 and HD 163296: the role of dust dynamics and planet--disc interaction'


class LatexDisplayTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.paper_obj = self.paper()
        PaperText.objects.create(arxiv_id=self.paper_obj.arxiv_id, text=SOURCE, source='html')
        self.paid = self.client_for(self.user('math-reader', paid=True))
        self.base = '/en/press-kit/2101.00001v1/'

    def test_web_keeps_escaped_source_and_loads_the_safe_math_renderer(self):
        data = content()
        data['key_sentences'] = [QUOTE, '<img src=x onerror=alert(1)>']
        data['experts'] = [{'name': 'L. M. Pirovano', 'arxiv_id': '2401.04088v1', 'title': TITLE}]
        row = Generated.objects.create(kind='presskit', key=self.paper_obj.arxiv_id, lang='en', status='done', content=json.dumps(data))
        response = self.paid.get(self.base)
        self.assertContains(response, QUOTE)
        self.assertContains(response, TITLE)
        self.assertContains(response, 'MathJax-script', count=1)
        self.assertContains(response, 'ui/safe')
        self.assertContains(response, '&lt;img src=x onerror=alert(1)&gt;')
        row.refresh_from_db()
        self.assertEqual(row.data()['key_sentences'][0], QUOTE)
        self.assertEqual(row.data()['experts'][0]['title'], TITLE)

    def test_pdf_has_readable_math_without_changing_stored_quotes(self):
        data = content()
        data['key_sentences'] = [QUOTE]
        data['experts'] = [{'name': 'L. M. Pirovano', 'arxiv_id': '2401.04088v1', 'title': TITLE}]
        row = Generated.objects.create(kind='presskit', key=self.paper_obj.arxiv_id, lang='en', status='done', content=json.dumps(data))
        with mock.patch('requests.get', side_effect=AssertionError('PDF must be offline')):
            response = self.paid.get(self.base + 'pdf/')
        self.assertEqual(response.status_code, 200)
        text = pdf_text(response.content)
        for raw in ['$','\\text','\\oplus','$_2$']:
            self.assertNotIn(raw, text)
        self.assertIn('photodissociation', text)
        self.assertIn('⊕', text)
        self.assertIn('HD 100546', text)
        row.refresh_from_db()
        self.assertEqual(row.data()['key_sentences'][0], QUOTE)


class MathProseTests(SimpleTestCase):
    def test_subscripts_superscripts_nested_groups_and_prose_safety(self):
        self.assertEqual(prose_html(r'0.1 $\text{M}_{\oplus}$, H$_2$O'), '0.1 M<sub>⊕</sub>, H<sub>2</sub>O')
        self.assertEqual(prose_html(r'\(x^{2}_{\mathrm{ICRS}}\)'), 'x<sup>2</sup><sub>ICRS</sub>')
        self.assertEqual(prose_html(r'<img src=x> $x_2$ & 7'), '&lt;img src=x&gt; x<sub>2</sub> &amp; 7')
        self.assertEqual(prose_html(r'$\frac{x_2}{y}$'), 'x<sub>2</sub>/y')
        self.assertIsNone(prose_html('Plain prose costs $7'))
        self.assertIsNone(prose_html('IDs_like_this remain unchanged'))

    def test_scientific_pages_share_the_safe_configuration(self):
        for name in ['presskit','letter','changes','missed_result','refcheck_result','collection','history','digest','home','search_results','tree','citations','trends']:
            self.assertIn("include 'summarizer/mathjax.html'", (Path('templates/summarizer') / (name + '.html')).read_text())
