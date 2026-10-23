from unittest import mock

from bs4 import BeautifulSoup
from django.test import Client

from summarizer.models import Generated
from summarizer.test_tools import ToolsTestCase


class ToolNavigationTests(ToolsTestCase):
    def test_menu_and_catalog_link_to_both_tools_in_each_language(self):
        for lang in ('en', 'fr'):
            page = Client().get('/%s/tools/' % lang)
            soup = BeautifulSoup(page.content, 'html.parser')
            for path in ('press-kit', 'journal-club'):
                link = '/%s/%s/' % (lang, path)
                self.assertIsNotNone(soup.select_one('#tools-menu a[href="%s"]' % link))
                self.assertIsNotNone(soup.select_one('.tool-card[href="%s"]' % link))
                self.assertEqual(Client().get(link).status_code, 200)

    def test_tool_landing_is_public_and_opens_known_papers_without_generation(self):
        paper = self.paper()
        with mock.patch('summarizer.generated.chat_json') as chat:
            for lang in ('en', 'fr'):
                for tool in ('press-kit', 'journal-club'):
                    base = '/%s/%s/' % (lang, tool)
                    landing = Client().get(base)
                    self.assertContains(landing, base + paper.arxiv_id + '/')
                    self.assertContains(landing, 'name="q"')
                    response = Client().get(base, {'q': 'https://arxiv.org/abs/2101.00001'})
                    self.assertRedirects(response, base + paper.arxiv_id + '/', fetch_redirect_response=False)
            chat.assert_not_called()
        self.assertFalse(Generated.objects.exists())

    def test_doi_and_new_arxiv_identifiers_use_existing_source_pages(self):
        paper = self.paper(arxiv_id='doi_10.1234--example')
        for tool in ('press-kit', 'journal-club'):
            base = '/fr/%s/' % tool
            self.assertRedirects(Client().get(base, {'q': '10.1234/example'}),
                                 base + paper.arxiv_id + '/', fetch_redirect_response=False)
            self.assertRedirects(Client().get(base, {'q': '2303.08774v2'}),
                                 '/fr/paper/2303.08774v2/#tools', fetch_redirect_response=False)
            self.assertRedirects(Client().get(base, {'q': '10.1234/new-paper'}),
                                 '/fr/paper/doi_10.1234--new-paper/#tools', fetch_redirect_response=False)

    def test_title_search_is_encoded_and_cannot_redirect_to_an_external_site(self):
        for tool in ('press-kit', 'journal-club'):
            response = Client().get('/en/%s/' % tool, {'q': 'clouds & atmospheres', 'next': 'https://evil.example'})
            self.assertRedirects(response, '/en/search-results/?q=clouds+%26+atmospheres&src=all',
                                 fetch_redirect_response=False)
        response = Client().get('/en/press-kit/', {'q': 'javascript:alert(1)'})
        self.assertTrue(response.url.startswith('/en/search-results/'))
