from django.test import Client

from summarizer.models import AIassistant
from summarizer.test_tools import ToolsTestCase


class ChatMarkdownPageTests(ToolsTestCase):
    def test_both_chat_pages_load_the_local_parser_before_the_renderer(self):
        self.paper()
        for path in ('/en/chat/', '/fr/chat/', '/en/paper/2101.00001v1/'):
            with self.subTest(path=path):
                page = Client().get(path)
                self.assertContains(page, 'summarizer/vendor/markdown-it/markdown-it.min.js?v=')
                self.assertContains(page, 'summarizer/js/chat-rendering.js?v=')
                html = page.content.decode()
                self.assertLess(html.index('markdown-it.min.js'), html.index('chat-rendering.js'))
                self.assertContains(page, "load: ['ui/safe']")

    def test_history_remains_escaped_and_stored_as_original_markdown(self):
        self.paper()
        user = self.user()
        original = '**Clouds**\n\n- Finding\n\n$x_i$\n<script>alert(123)</script>'
        row = AIassistant.objects.create(user=user, arxiv_id='2101.00001v1', lang='en',
                                         query='<img src=x onerror=alert(1)>', response=original)
        page = self.client_for(user).get('/en/paper/2101.00001v1/')
        self.assertContains(page, 'data-chat-markdown')
        self.assertContains(page, 'chat-answer tex2jax_ignore')
        self.assertContains(page, '**Clouds**')
        self.assertContains(page, '&lt;script&gt;alert(123)&lt;/script&gt;')
        self.assertNotContains(page, '<script>alert(123)</script>')
        self.assertContains(page, '&lt;img src=x onerror=alert(1)&gt;')
        row.refresh_from_db()
        self.assertEqual(row.response, original)
