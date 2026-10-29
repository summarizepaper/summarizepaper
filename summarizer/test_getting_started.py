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
            {'text': 'An observed result', 'kind': 'result', 'quote': '<script>untrusted source quotation with enough characters</script>'},
            {'text': 'A proposed explanation', 'kind': 'interpretation', 'quote': ''}],
            'limitations': [{'text': 'The objects were not directly detected'}]}
        self.text.text = data['findings'][0]['quote']
        self.text.save()
        with mock.patch('summarizer.getting_started.insight.for_page', return_value=data):
            response = self.client.get('/fr/getting-started/')
        self.assertContains(response, 'Scientific overview')
        self.assertContains(response, 'Interprétation des auteurs')
        self.assertContains(response, 'The objects were not directly detected')
        self.assertContains(response, '&lt;script&gt;untrusted source quotation with enough characters&lt;/script&gt;')
        self.assertNotContains(response, '<script>untrusted source quotation with enough characters</script>')

    def test_full_paper_sections_and_reading_formats_use_stored_content_safely(self):
        self.paper.abstract = 'The original abstract written by the authors.'
        self.paper.save()
        stored_blog = '<h2>A research story</h2><p>A narrative about the findings.</p><script>alert("blog")</script><img src="https://evil.test/pixel" onerror="alert(1)">'
        SummaryPaper.objects.filter(lang='en').update(
            summary='The full scientific summary, not a shortened preview.',
            lay_summary='An accessible explanation.', notes="['First key point', 'Second key point']", blog=stored_blog)
        data = {'tldr':'The essentials in one sentence', 'question':'The research question',
                'method':'The actual experimental method', 'significance':'Why this research matters',
                'findings':[{'text':'Finding %d' % i, 'kind':'result', 'quote':''} for i in range(5)],
                'limitations':[{'text':'A real study limitation', 'quote':''}]}
        with mock.patch('summarizer.getting_started.insight.for_page', return_value=data):
            response = self.client.get('/en/getting-started/')
        for content in ('The research question', 'The actual experimental method', 'Why this research matters',
                        'Finding 4', 'A real study limitation', 'The full scientific summary, not a shortened preview.',
                        'An accessible explanation.', 'First key point', 'Second key point', 'A research story',
                        'The original abstract written by the authors.'):
            self.assertContains(response, content)
        soup = BeautifulSoup(response.content, 'html.parser')
        self.assertEqual([b['data-reading'] for b in soup.select('[data-reading]')], ['summary','plain','points','blog','abstract'])
        self.assertEqual(len(soup.select('[data-reading-panel]')), 5)
        self.assertNotContains(response, '<script>alert("blog")</script>')
        self.assertNotContains(response, 'https://evil.test/pixel')
        self.assertEqual(SummaryPaper.objects.get(lang='en').blog, stored_blog)
        self.assertContains(response, '/en/paper/%s/#results' % self.paper.arxiv_id)

    def test_all_domains_use_their_own_bilingual_paper_questions_and_chat_return(self):
        for slug, example in getting_started.EXAMPLES.items():
            paper, _ = ArxivPaper.objects.get_or_create(arxiv_id=example['id'], defaults={
                'title': 'Example ' + slug, 'license': self.paper.license})
            PaperText.objects.get_or_create(arxiv_id=paper.arxiv_id, defaults={'source': 'html', 'text': 'Full paper'})
            for lang in ('fr', 'en'):
                SummaryPaper.objects.get_or_create(paper=paper, lang=lang, defaults={'summary': slug + ' ' + lang})
                response = self.client.get('/%s/getting-started/' % lang, {'domain': slug})
                soup = BeautifulSoup(response.content, 'html.parser')
                self.assertEqual(response.context['paper'].pk, paper.pk)
                self.assertEqual(soup.select_one('[name=ids]')['value'], paper.arxiv_id)
                self.assertEqual(soup.select_one('[name=tour_domain]')['value'], slug)
                self.assertContains(response, example['question_' + lang])
                self.assertContains(response, 'data-domain="%s"' % slug)
                self.assertContains(response, 'data-paper="%s"' % paper.arxiv_id)
            chat = self.client.get('/en/chat/', {'ids': paper.arxiv_id, 'tour': '1', 'tour_domain': slug})
            self.assertContains(chat, '/en/getting-started/?domain=%s#question' % slug)

    def test_unknown_domain_falls_back_without_selecting_arbitrary_papers(self):
        response = self.client.get('/en/getting-started/', {'domain': '//evil.test/paper'})
        self.assertEqual(response.context['paper'].pk, self.paper.pk)
        self.assertEqual(response.context['domain'], 'astronomy')
        chat = self.client.get('/en/chat/', {'tour': '1', 'tour_domain': 'evil'})
        self.assertContains(chat, '/en/getting-started/?domain=astronomy#question')

    def test_exercise_only_uses_a_quote_in_the_current_source(self):
        quote = 'An original sentence supported by the full article text.'
        data = {'tldr': 'Overview', 'abstract_only': False, 'findings': [
            {'text': 'A supported finding', 'kind': 'result', 'quote': quote}]}
        with mock.patch('summarizer.getting_started.insight.for_page', return_value=data):
            response = self.client.get('/en/getting-started/')
            self.assertNotContains(response, 'id="walkthrough-evidence"')
            self.text.text = 'Before. ' + quote + ' After.'
            self.text.save()
            response = self.client.get('/en/getting-started/')
            self.assertContains(response, 'id="walkthrough-evidence"')
            self.assertContains(response, 'Read the original passage')
            self.assertContains(response, 'Our AI has summarized the full text')
            self.assertNotContains(response, 'data-verify')
            self.assertContains(response, quote)
            self.assertNotIn('exercise', data['findings'][0])
            data['abstract_only'] = True
            self.assertNotContains(self.client.get('/en/getting-started/'), 'id="walkthrough-evidence"')

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
