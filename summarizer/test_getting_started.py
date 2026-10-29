from unittest import mock

from bs4 import BeautifulSoup
from django.contrib.auth.models import User
from django.test import TestCase

from . import getting_started
from .models import ArxivPaper, PaperText, SummaryPaper


class GettingStartedTests(TestCase):
    def setUp(self):
        self.paper = ArxivPaper.objects.create(arxiv_id=getting_started.EXAMPLE, title='A real paper',
                    license='https://creativecommons.org/licenses/by/4.0/')
        self.text = PaperText.objects.create(arxiv_id=self.paper.arxiv_id, source='pdf', text='Full article text')
        for lang in ('fr', 'en'):
            SummaryPaper.objects.create(paper=self.paper, lang=lang, lay_summary='Résumé réel' if lang == 'fr' else 'Real summary')

    def test_bilingual_ready_example_with_safe_summary_and_explicit_chat_submission(self):
        for lang, summary in [('fr', 'Résumé réel'), ('en', 'Real summary')]:
            with mock.patch('summarizer.chat_figures.paper_gallery') as gallery:
                response = self.client.get('/%s/getting-started/' % lang)
            gallery.assert_not_called()  # No remote calls or generation on a visit.
            self.assertContains(response, summary)
            soup = BeautifulSoup(response.content, 'html.parser')
            self.assertEqual(len(soup.select('[data-panel]')), 3)
            form = soup.select_one('form')
            self.assertEqual(form['action'], '/%s/chat/' % lang)
            self.assertEqual(form.select_one('[name=ids]')['value'], self.paper.arxiv_id)
            self.assertEqual(form.select_one('[name=tour]')['value'], '1')
            self.assertTrue(form.select_one('textarea[name=ask]').get_text().strip())
            self.assertEqual(SummaryPaper.objects.count(), 2)

    def test_scientific_findings_keep_verified_quotes_and_label_interpretations(self):
        data = {'tldr': 'Scientific overview', 'findings': [
            {'text': 'An observed result', 'kind': 'result', 'quote': '<script>untrusted</script>'},
            {'text': 'A proposed explanation', 'kind': 'interpretation', 'quote': ''}],
            'limitations': [{'text': 'The objects were not directly detected'}]}
        with mock.patch('summarizer.getting_started.insight.for_page', return_value=data):
            response = self.client.get('/fr/getting-started/')
        self.assertContains(response, 'Scientific overview')
        self.assertContains(response, 'Interprétation des auteurs')
        self.assertContains(response, 'The objects were not directly detected')
        self.assertContains(response, '&lt;script&gt;untrusted&lt;/script&gt;')
        self.assertNotContains(response, '<script>untrusted</script>')

    def test_untrusted_summary_is_escaped(self):
        SummaryPaper.objects.filter(lang='fr').update(lay_summary='<script>alert(1)</script>')
        response = self.client.get('/fr/getting-started/')
        self.assertContains(response, '&lt;script&gt;')
        self.assertNotContains(response, '<script>alert(1)</script>')

    def test_unavailable_or_restricted_example_does_not_claim_full_text(self):
        for source in ('abstract', 'index_kept'):
            self.text.source = source
            self.text.save()
            response = self.client.get('/fr/getting-started/')
            self.assertContains(response, 'momentanément indisponible')
            self.assertNotContains(response, 'Texte intégral disponible')
        self.text.source = 'pdf'; self.text.save()
        self.paper.license = ''; self.paper.save()
        self.assertContains(self.client.get('/en/getting-started/'), 'temporarily unavailable')

    def test_missing_translation_does_not_show_the_other_language(self):
        SummaryPaper.objects.filter(lang='fr').delete()
        response = self.client.get('/fr/getting-started/')
        self.assertContains(response, 'momentanément indisponible')
        self.assertNotContains(response, 'Real summary')

    def test_account_activation_enters_walkthrough_but_normal_visits_are_unchanged(self):
        user = User.objects.create_user('beginner')
        self.client.force_login(user)
        self.assertRedirects(self.client.get('/fr/?activated=True'), '/fr/getting-started/')
        self.assertEqual(self.client.get('/fr/').status_code, 200)
        self.client.logout()
        self.assertEqual(self.client.get('/fr/?activated=True').status_code, 200)

    def test_chat_opens_with_paper_selected_and_never_sends_the_draft(self):
        response = self.client.get('/fr/chat/', {'ids': self.paper.arxiv_id, 'tour': '1', 'ask': 'Question privée'})
        self.assertContains(response, 'data-guided="1"')
        self.assertEqual(response.context['preset'][0]['id'], self.paper.arxiv_id)
        self.assertContains(response, 'Revenir au parcours guidé')
        from .models import AIassistant
        self.assertFalse(AIassistant.objects.exists())
