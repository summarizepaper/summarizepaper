from bs4 import BeautifulSoup
from summarizer.test_tools import ToolsTestCase


class FileUploadTests(ToolsTestCase):
    def test_paid_forms_have_accessible_dropzones_with_server_limits_in_both_languages(self):
        client = self.client_for(self.user(paid=True))
        for lang in ('fr', 'en'):
            for tool, fields, limit in [('refcheck', ('bib', 'tex'), 5), ('missed', ('draft',), 20),
                                        ('scoops', ('draft',), 20), ('conference', ('program_pdf',), 20)]:
                with self.subTest(lang=lang, tool=tool):
                    response = client.get('/%s/%s/' % (lang, tool))
                    self.assertEqual(response.status_code, 200)
                    soup = BeautifulSoup(response.content, 'html.parser')
                    boxes = soup.select('[data-file-upload]')
                    self.assertEqual(len(boxes), len(fields))
                    self.assertEqual(len(soup.select('script[src*="file-upload.js"]')), 1)
                    for box, name in zip(boxes, fields):
                        self.assertEqual(int(box['data-max-bytes']), limit * 1024 * 1024)
                        self.assertEqual(int(box['data-max-mb']), limit)
                        file_input = box.select_one('input[type=file]')
                        self.assertEqual(file_input['name'], name)
                        self.assertEqual(file_input.find_parent('form')['enctype'], 'multipart/form-data')
                        self.assertIsNotNone(soup.find('label', {'for': file_input['id']}))
                        self.assertIsNotNone(soup.find(id=file_input['aria-describedby']))
                        self.assertEqual(box.select_one('[data-file-clear]')['type'], 'button')
                        self.assertEqual(box.select_one('[data-file-error]')['role'], 'alert')
                        self.assertEqual(box.select_one('[data-file-status]')['aria-live'], 'polite')
                        self.assertEqual(box.get('data-required-extension'), '' if tool == 'refcheck' else '.pdf')
                    if lang == 'fr':
                        self.assertIn('Déposez votre fichier ici', soup.get_text())
                        self.assertIn('Jusqu’à %s Mo' % limit, soup.get_text())
                    ids = [tag['id'] for tag in soup.select('[id]')]
                    self.assertEqual(len(ids), len(set(ids)))
