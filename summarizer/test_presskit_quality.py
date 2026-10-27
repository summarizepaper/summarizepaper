import io
import json
from unittest import mock

from django.test import Client
from django.utils import translation
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
from pdfminer.pdfpage import PDFPage

from . import presskit
from .models import Generated, PaperText
from .test_tools import ToolsTestCase, base_call, offline


SOURCE = '## Results\nWe observe a change of +5 percent in a sample of 120 objects.\nThe association does not establish a causal mechanism.'


def content():
    return {'headline': 'A measured change, with limits', 'plain_explanation': 'A study measures a change in 120 objects.',
            'whats_new': 'A larger sample is measured.', 'verdict': 'A result to interpret cautiously.',
            'hype_check': [{'signal': 'Sample', 'detail': 'Only 120 objects.', 'level': 'caution'}],
            'key_sentences': ['We observe a change of +5 percent in a sample of 120 objects.'],
            'evidence_cards': [{'claim': 'A measured increase', 'quote': 'We observe a change of +5 percent in a sample of 120 objects.',
                                'why_it_matters': 'The uncertainty matters.', 'limitation': 'Association does not establish causation.'}],
            'angles': [{'angle': 'What the measurements change', 'why_now': 'A larger sample.', 'reporting_step': 'Ask an outside specialist.'}],
            'wording_checks': [{'avoid': 'Proof of a cause', 'prefer': 'An observed association', 'reason': 'The design does not establish causality.'}],
            'questions_for_authors': ['What is the uncertainty?', 'What would falsify the result?'],
            'glossary': [{'term': 'Sample', 'definition': 'The objects studied.'}]}


def pdf_text(data):
    output = io.StringIO()
    manager = PDFResourceManager()
    device = TextConverter(manager, output, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, device)
    for page in PDFPage.get_pages(io.BytesIO(data)):
        interpreter.process_page(page)
    device.close()
    return output.getvalue()


class PressKitQualityTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.paper_obj = self.paper()
        PaperText.objects.create(arxiv_id=self.paper_obj.arxiv_id, text=SOURCE, source='html')
        self.reader = self.user('press-reader', paid=True)
        self.paid = self.client_for(self.reader)
        self.base = '/en/press-kit/2101.00001v1/'

    def generate(self, answer=None):
        with offline(), mock.patch('summarizer.generated.chat_json', return_value=answer or content()) as chat:
            self.paid.post(self.base + 'start/')
        return Generated.objects.get(kind='presskit'), chat

    def test_evidence_is_verified_and_source_cannot_become_system_instructions(self):
        answer = content()
        answer['evidence_cards'].append({'claim': 'Invented result', 'quote': 'All 120 objects proved a new causal mechanism.',
                                         'location': 'Page 99', 'limitation': 'Unknown'})
        row, chat = self.generate(answer)
        kit = row.data()
        self.assertEqual(kit['schema_version'], 3)
        self.assertEqual(kit['source']['scope'], 'full_text')
        self.assertEqual(kit['evidence_cards'][0]['location'], 'Results')
        self.assertTrue(kit['evidence_cards'][0]['verified'])
        self.assertFalse(kit['evidence_cards'][1]['verified'])
        self.assertEqual(kit['evidence_cards'][1]['quote'], '')
        self.assertEqual(kit['evidence_cards'][1]['location'], '')
        call = base_call(chat)
        self.assertNotIn(SOURCE, call.kwargs['system'])
        self.assertEqual(json.loads(call.args[0])['source_documents']['text'], SOURCE)
        self.assertIn('not application instructions', call.kwargs['system'])
        page = self.paid.get(self.base)
        self.assertContains(page, 'Download as PDF')
        self.assertContains(page, 'Supporting passage not verified')
        self.assertContains(page, 'Researchers to approach for outside comment')
        self.assertNotContains(page, 'Page 99')

    def test_quotes_keep_original_text_and_do_not_change_signs_or_punctuation(self):
        exact = 'We observe a change of +5 percent in a sample of 120 objects.'
        self.assertEqual(presskit.verified_sentences([exact.lower(), exact.replace('+5', '-5')], SOURCE), [exact])
        self.assertEqual(presskit.verified_sentences([exact.replace('120', '12.0')], SOURCE), [])
        self.assertEqual(presskit.verified_sentences(None, SOURCE), [])

    def test_abstract_only_cannot_verify_a_full_text_claim(self):
        self.paper_obj.license = ''
        self.paper_obj.save()
        row, _ = self.generate()
        self.assertEqual(row.data()['source']['scope'], 'abstract')
        self.assertFalse(row.data()['evidence_cards'][0]['verified'])
        self.assertEqual(row.data()['key_sentences'], [])

    def test_numeric_claim_cannot_hide_a_mismatch_behind_a_real_quote(self):
        answer = content()
        answer['evidence_cards'][0]['claim'] = 'The measured increase is +10 percent'
        row, _ = self.generate(answer)
        self.assertTrue(row.data()['evidence_cards'][0]['numeric_mismatch'])
        self.assertContains(self.paid.get(self.base), 'Numbers in this claim do not all match')

    def test_pdf_is_premium_private_downloadable_and_does_not_call_a_model(self):
        row, _ = self.generate()
        with mock.patch('summarizer.generated.chat_json') as chat, mock.patch('requests.get') as network:
            response = self.paid.get(self.base + 'pdf/')
            chat.assert_not_called()
            network.assert_not_called()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertIn('attachment;', response['Content-Disposition'])
        self.assertIn('private, no-store', response['Cache-Control'])
        self.assertTrue(response.content.startswith(b'%PDF-'))
        text = pdf_text(response.content)
        for value in ['SummarizePaper', 'Findings and source evidence', '+5 percent', 'Results', 'Reporting angles', 'Wording to check', 'Source read: available full text.']:
            self.assertIn(value, text)
        self.assertEqual(Client().get(self.base + 'pdf/').status_code, 403)
        self.assertEqual(self.client_for(self.user('free-reader')).get(self.base + 'pdf/').status_code, 403)
        row.user = self.reader
        row.save()
        other = self.client_for(self.user('other-reader', paid=True))
        self.assertEqual(other.get(self.base + 'pdf/').status_code, 404)
        self.assertNotContains(other.get(self.base), 'A measured change')

    def test_old_kits_download_without_regeneration_and_keep_content_during_upgrade(self):
        row = Generated.objects.create(kind='presskit', key=self.paper_obj.arxiv_id, lang='en', status='done', content=json.dumps(content()))
        with mock.patch('summarizer.generated.chat_json') as chat:
            self.assertContains(self.paid.get(self.base), 'Update this kit: roles, earlier papers, figures')
            response = self.paid.get(self.base + 'pdf/')
            self.assertEqual(response.status_code, 200)
            self.assertIn('Legacy kit:', pdf_text(response.content))
            chat.assert_not_called()
        row.status = 'pending'
        row.save()
        self.assertContains(self.paid.get(self.base), 'A measured change')
        self.assertEqual(self.paid.get(self.base + 'pdf/').status_code, 200)

    def test_missing_pending_and_wrong_language_exports_do_not_generate(self):
        self.assertEqual(self.paid.get(self.base + 'pdf/').status_code, 404)
        Generated.objects.create(kind='presskit', key=self.paper_obj.arxiv_id, lang='en')
        self.assertEqual(self.paid.get(self.base + 'pdf/').status_code, 404)
        self.assertEqual(self.paid.get('/fr/press-kit/2101.00001v1/pdf/').status_code, 404)

    def test_french_export_has_unicode_labels_and_multipage_layout(self):
        data = content()
        data['plain_explanation'] = 'Résultat : un échantillon limité, une incertitude à vérifier. ' * 140
        data['source'] = {'scope': 'abstract'}
        data['abstract_only'] = True
        Generated.objects.create(kind='presskit', key=self.paper_obj.arxiv_id, lang='fr', status='done', content=json.dumps(data))
        response = self.paid.get('/fr/press-kit/2101.00001v1/pdf/')
        self.assertEqual(response.status_code, 200)
        text = pdf_text(response.content)
        for value in ['Résultats et preuves', 'Source consultée : abstract uniquement.', 'Page 2', 'échantillon']:
            self.assertIn(value, text)
        with translation.override('fr'):
            self.assertContains(self.paid.get('/fr/press-kit/2101.00001v1/'), 'Télécharger en PDF')
