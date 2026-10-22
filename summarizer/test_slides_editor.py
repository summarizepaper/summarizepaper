import copy
import json

from django.test import Client

from . import slides
from .models import Generated, UserPlan
from .test_tools import ToolsTestCase


class SlidesEditorTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.article = self.paper()
        self.reader = self.user('editor', paid=True)
        self.client = self.client_for(self.reader)
        self.url = '/en/journal-club/2101.00001v1/'
        self.data = {'slides': [{'title': 'Results', 'bullets': ['Original result'], 'notes': 'Private speaker note',
                                'figure': {'number': 1, 'url': 'https://arxiv.org/html/2101.00001v1/figure.png', 'caption': 'Spectrum'}}],
                     'strengths': ['Evidence'], 'limitations': ['Small sample'], 'questions': ['Why?'], 'license': 'CC BY 4.0'}
        self.original = Generated.objects.create(kind='slides', key=self.article.arxiv_id, lang='en', status='done', content=json.dumps(self.data))

    def payload(self, client=None):
        response = (client or self.client).get(self.url)
        return {'revision': response.context['deck_revision'], 'pages': copy.deepcopy(response.context['pages'])}

    def save(self, payload, client=None):
        return (client or self.client).post(self.url + 'save/', json.dumps(payload), content_type='application/json')

    def test_text_layout_and_notes_are_saved_privately_without_changing_original(self):
        payload = self.payload()
        payload['pages'][1].update(title='My interpretation', bullets=['A new result'], notes='My notes', layout='left', figure_size=65)
        result = self.save(payload)
        self.assertEqual(result.status_code, 200)
        page = self.client.get(self.url)
        self.assertContains(page, 'My interpretation')
        self.assertContains(page, 'data-layout="left"')
        self.assertContains(page, '--figure-share:65%')
        self.assertContains(page, 'My notes')
        self.assertEqual(page.context['deck_revision'], result.json()['revision'])
        self.assertIn('no-store', page['Cache-Control'])
        other = self.client_for(self.user('other-editor', paid=True))
        self.assertNotContains(other.get(self.url), 'My interpretation')
        self.original.refresh_from_db()
        self.assertEqual(self.original.data(), self.data)
        private = Generated.objects.get(kind='slides_edit')
        self.assertEqual(private.user_id, self.reader.pk)
        self.assertEqual(other.get('/en/ai/%s/' % private.pk).status_code, 404)

    def test_stale_tab_cannot_silently_overwrite_saved_changes(self):
        old = self.payload()
        changed = copy.deepcopy(old)
        changed['pages'][0]['title'] = 'A changed title'
        self.assertEqual(self.save(changed).status_code, 200)
        self.assertEqual(self.save(old).status_code, 409)
        self.assertContains(self.client.get(self.url), 'A changed title')

    def test_client_cannot_replace_source_images_or_credits_and_html_is_text(self):
        payload = self.payload()
        payload['pages'][1]['figure'] = {'url': 'http://127.0.0.1/private', 'number': 999}
        payload['pages'][1]['title'] = '<script>alert(1)</script>'
        self.assertEqual(self.save(payload).status_code, 200)
        page = self.client.get(self.url)
        self.assertContains(page, self.data['slides'][0]['figure']['url'])
        self.assertNotContains(page, 'http://127.0.0.1/private')
        self.assertContains(page, 'Figure 1 of the paper (arXiv, CC BY 4.0)')
        self.assertNotContains(page, '<script>alert(1)</script>')
        self.assertContains(page, '&lt;script&gt;alert(1)&lt;/script&gt;')

    def test_invalid_editor_input_is_rejected_without_creating_a_copy(self):
        for field, value in [('layout', 'absolute'), ('figure_size', '50;position:fixed'), ('text_size', 200),
                             ('title', ''), ('bullets', ['x'] * 9), ('notes', 'x' * 3001)]:
            payload = self.payload()
            payload['pages'][1][field] = value
            self.assertEqual(self.save(payload).status_code, 400)
        payload = self.payload()
        payload['pages'].pop()
        self.assertEqual(self.save(payload).status_code, 400)
        self.assertFalse(Generated.objects.filter(kind='slides_edit').exists())

    def test_save_requires_paid_account_csrf_and_post(self):
        payload = self.payload()
        self.assertEqual(self.client.get(self.url + 'save/').status_code, 405)
        self.assertEqual(self.save(payload, Client()).status_code, 403)
        strict = Client(enforce_csrf_checks=True)
        strict.force_login(self.reader)
        self.assertEqual(self.save(payload, strict).status_code, 403)
        UserPlan.objects.filter(user=self.reader).update(paid=False)
        self.assertEqual(self.save(payload).status_code, 403)

    def test_print_and_editor_controls_preserve_generation_and_source_warning(self):
        page = self.client.get(self.url)
        self.assertContains(page, 'slides.css')
        self.assertContains(page, 'slides.js')
        self.assertContains(page, 'Save as PDF</button>', count=0)
        self.assertContains(page, '16:9')
        self.assertContains(page, 'Speaker notes (excluded from PDF)')
        self.assertContains(page, 'data-edit="layout"')
        self.assertContains(page, 'data-edit="figure_size"')
        self.original.status = 'pending'
        self.original.save(update_fields=['status'])
        self.assertNotContains(self.client.get(self.url), 'id="deck-edit"')
