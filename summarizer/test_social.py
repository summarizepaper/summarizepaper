import datetime
import io
import json
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from django.test import Client, TestCase
from PIL import Image

from . import social
from .models import ArxivPaper, Generated, Letter, RisingTopics, SummaryPaper

WEEK = datetime.date(2026, 10, 26)


def topic(i):
    return {'title': 'Topic %d' % i, 'terms': ['term %d' % i], 'main': 'term %d' % i, 'kind': 'spike',
            'growth': '4.%d' % i, 'count': 10 + i, 'teams': 5 + i, 'papers': []}


class PictureTests(TestCase):
    def assertPicture(self, response):
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'image/png')
        self.assertIn('max-age', response['Cache-Control'])
        self.assertEqual(Image.open(io.BytesIO(response.content)).size, (1200, 630))

    def test_card_draws_any_script_and_cuts_long_text(self):
        title = 'Détection de CO$_2$ par JWST : α ≈ 0.3, ∇·B = 0, Шкала и 量子 ' * 6
        Image.open(io.BytesIO(social.card('EYEBROW', title, 'word ' * 400)))
        Image.open(io.BytesIO(social.card('EYEBROW', 'Rising', items=[('×12', 'x' * 300)] * 6, note='Note')))
        self.assertEqual(social.first_sentence(', , , , We present Mixtral 8x7B, a model. It wins.'), 'We present Mixtral 8x7B, a model.')
        lines, complete = social.wrap('word ' * 200, 32, 400, 1000, 3)
        self.assertFalse(complete)
        self.assertEqual(len(lines), 3)
        self.assertTrue(lines[-1].endswith('…'))

    def test_paper_picture(self):
        paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='A $\\Lambda$CDM test', abstract='Abstract.',
                                          category='astro-ph.CO')
        SummaryPaper.objects.create(paper=paper, lang='fr', summary='Le résumé. Une suite.', notes='[]')
        self.assertPicture(Client().get('/og/fr/p/2401.04088v1.png'))
        self.assertPicture(Client().get('/og/en/p/2401.04088v1.png'))  # no summary in English: the abstract
        self.assertEqual(Client().get('/og/de/p/2401.04088v1.png').status_code, 404)
        self.assertEqual(Client().get('/og/fr/p/2401.99999v1.png').status_code, 404)

    def test_trends_and_letter_pictures(self):
        RisingTopics.objects.create(scope='biorxiv', lang='en', day=WEEK, content=json.dumps({'topics': [topic(1)]}))
        self.assertPicture(Client().get('/og/en/trends/biorxiv/%s.png' % WEEK))
        self.assertEqual(Client().get('/og/en/trends/arxiv/%s.png' % WEEK).status_code, 404)
        self.assertEqual(Client().get('/og/en/trends/cs/%s.png' % WEEK).status_code, 404)
        self.assertEqual(Client().get('/og/en/trends/biorxiv/someday.png').status_code, 404)
        Letter.objects.create(topic='space', lang='fr', day=WEEK, title='Des étoiles', dek='Ce qui a changé.', content='{}')
        self.assertPicture(Client().get('/og/fr/letter/space/%s.png' % WEEK))
        self.assertEqual(Client().get('/og/en/letter/space/%s.png' % WEEK).status_code, 404)


class PostTests(TestCase):
    def test_paper_page_offers_a_post_and_its_picture(self):
        paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Gas in debris disks', abstract='Abstract.',
                                          category='astro-ph.EP')
        SummaryPaper.objects.create(paper=paper, lang='en', summary='Long summary.', notes='[]')
        reading = {'tldr': 'Primordial gas can explain the CO seen in debris disks. ' * 3,
                   'findings': [{'text': 'Finding %d.' % i, 'kind': 'result', 'quote': ''} for i in range(5)]}
        Generated.objects.create(kind='insight', key='2401.04088v1', lang='en', status='done', content=json.dumps(reading))
        soup = BeautifulSoup(Client().get('/en/paper/2401.04088v1/').content, 'html.parser')
        self.assertTrue(soup.select_one('meta[property="og:image"]')['content'].startswith(
            'https://www.summarizepaper.com/og/en/p/2401.04088v1.png?v='))
        self.assertEqual(soup.select_one('meta[name="twitter:card"]')['content'], 'summary_large_image')
        box = soup.select_one('[data-share]')
        text = box.select_one('textarea').get_text()
        self.assertIn('Gas in debris disks', text)
        self.assertIn('• Finding 2.', text)
        self.assertNotIn('Finding 3.', text)
        self.assertIn('https://www.summarizepaper.com/en/paper/2401.04088v1/', text)
        self.assertIn('#PlanetaryScience', text)
        self.assertIn('One of the authors?', box.get_text())
        # X counts the link as 23 characters: the short post leaves room for it
        x = parse_qs(urlparse(box.select_one('[data-share-network=x]')['href']).query)
        self.assertLessEqual(len(x['text'][0]) + 1 + 23, 280)
        self.assertEqual(x['url'], ['https://www.summarizepaper.com/en/paper/2401.04088v1/?utm_source=x'])
        bluesky = parse_qs(urlparse(box.select_one('[data-share-network=bluesky]')['href']).query)['text'][0]
        self.assertLessEqual(len(bluesky), 300)
        linkedin = parse_qs(urlparse(box.select_one('[data-share-network=linkedin]')['href']).query)['text'][0]
        self.assertIn('/en/paper/2401.04088v1/?utm_source=linkedin', linkedin)

    def test_paper_page_without_summary_has_no_post(self):
        ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Example', abstract='Abstract')
        soup = BeautifulSoup(Client().get('/en/paper/2401.04088v1/').content, 'html.parser')
        self.assertIsNone(soup.select_one('[data-share]'))

    def test_french_post_of_a_paper_from_its_key_points(self):
        paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Example', abstract='Abstract', category='cs.LG')
        SummaryPaper.objects.create(paper=paper, lang='fr', summary='Résumé : Cet article montre une chose importante. Puis une autre.',
                                    notes="['**Point** un', 'Point deux']")
        soup = BeautifulSoup(Client().get('/fr/paper/2401.04088v1/').content, 'html.parser')
        text = soup.select_one('[data-share] textarea').get_text()
        self.assertIn('Cet article montre une chose importante.', text)
        self.assertNotIn('Puis une autre', text)
        self.assertIn('À retenir :', text)
        self.assertIn('• Point un', text)
        self.assertIn('#MachineLearning #AI #arXiv', text)

    def test_trends_post_holds_only_the_free_topics(self):
        RisingTopics.objects.create(scope='*', lang='en', day=WEEK, content=json.dumps({'topics': [topic(i) for i in range(1, 6)]}))
        soup = BeautifulSoup(Client().get('/en/trends/').content, 'html.parser')
        text = soup.select_one('[data-share] textarea').get_text()
        self.assertIn('1. Topic 1 — ×4.1, 11 papers from 6 teams', text)
        self.assertIn('3. Topic 3', text)
        self.assertNotIn('Topic 4', text)
        self.assertIn('https://www.summarizepaper.com/en/trends/?week=2026-10-26', text)
        self.assertEqual(soup.select_one('meta[property="og:image"]')['content'],
                         'https://www.summarizepaper.com/og/en/trends/arxiv/2026-10-26.png')
        RisingTopics.objects.create(scope='medrxiv', lang='en', day=WEEK, content=json.dumps({'topics': [topic(1)]}))
        text = BeautifulSoup(Client().get('/en/trends/', {'scope': 'medrxiv'}).content, 'html.parser') \
            .select_one('[data-share] textarea').get_text()
        self.assertIn('?scope=medrxiv&week=2026-10-26', text)

    def test_letter_post(self):
        content = {'sections': [{'heading': 'Planets that wander', 'text': 'Text.', 'papers': []}], 'papers': {}, 'closing': '', 'count': 6}
        Letter.objects.create(topic='space', lang='en', day=WEEK, title='Space this week', dek='What changed.',
                              content=json.dumps(content))
        soup = BeautifulSoup(Client().get('/en/letters/space/2026-10-26/').content, 'html.parser')
        text = soup.select_one('[data-share] textarea').get_text()
        self.assertIn('Space this week\n\nWhat changed.', text)
        self.assertIn('• Planets that wander', text)
        self.assertIn('https://www.summarizepaper.com/en/letters/space/2026-10-26/', text)
        self.assertEqual(soup.select_one('meta[property="og:image"]')['content'],
                         'https://www.summarizepaper.com/og/en/letter/space/2026-10-26.png')
