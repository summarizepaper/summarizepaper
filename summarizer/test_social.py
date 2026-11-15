import datetime
import io
import json
from unittest import mock
from urllib.parse import parse_qs, urlparse

from bs4 import BeautifulSoup
from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.test import Client, TestCase, override_settings
from PIL import Image

from . import accounts, letters, social
from .models import ArxivPaper, Author, Generated, Letter, LetterSubscription, PaperAuthor, RisingTopics, SummaryPaper

CC_BY = 'http://creativecommons.org/licenses/by/4.0/'

WEEK = datetime.date(2026, 10, 26)


def topic(i):
    return {'title': 'Topic %d' % i, 'terms': ['term %d' % i], 'main': 'term %d' % i, 'kind': 'spike',
            'growth': '4.%d' % i, 'count': 10 + i, 'teams': 5 + i, 'papers': [], 'text': 'What topic %d is about.' % i}


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
        text = soup.select_one('#share textarea').get_text()
        self.assertIn('1. Topic 1 — ×4.1, 11 papers from 6 teams', text)
        self.assertIn('3. Topic 3', text)
        self.assertNotIn('Topic 4', text)
        self.assertIn('https://www.summarizepaper.com/en/trends/?week=2026-10-26', text)
        self.assertEqual(soup.select_one('meta[property="og:image"]')['content'],
                         'https://www.summarizepaper.com/og/en/trends/arxiv/2026-10-26.png')
        RisingTopics.objects.create(scope='medrxiv', lang='en', day=WEEK, content=json.dumps({'topics': [topic(1)]}))
        text = BeautifulSoup(Client().get('/en/trends/', {'scope': 'medrxiv'}).content, 'html.parser') \
            .select_one('#share textarea').get_text()
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


def png_bytes(size=(800, 500)):
    out = io.BytesIO()
    Image.new('RGB', size, (200, 30, 30)).save(out, 'PNG')
    return out.getvalue()


class FigureTests(TestCase):
    def setUp(self):
        cache.clear()
        self.paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Open paper', abstract='Abstract.', license=CC_BY)
        SummaryPaper.objects.create(paper=self.paper, lang='en', summary='The summary of it all.', notes='[]')

    def test_a_figure_is_fetched_in_the_background_then_drawn(self):
        with mock.patch('summarizer.social.threading.Thread') as thread:
            response = Client().get('/og/en/p/2401.04088v1.png')
        self.assertEqual(response['Cache-Control'], social.CACHE_SOON)
        thread.assert_called_once()
        gallery = {'figures': [{'label': 'Figure 1', 'images': ['https://arxiv.org/html/2401.04088v1/x1.png']}]}
        fetched = mock.Mock(content=png_bytes(), raise_for_status=lambda: None)
        with mock.patch('summarizer.chat_figures.paper_gallery', return_value=gallery), \
                mock.patch('requests.get', return_value=fetched):
            social.fetch_figure('2401.04088v1')
        picture, credit = social.figure_of(self.paper)
        self.assertEqual(credit, 'Figure 1 · CC BY 4.0')
        response = Client().get('/og/en/p/2401.04088v1.png')
        self.assertEqual(response['Cache-Control'], social.CACHE)
        drawn = Image.open(io.BytesIO(response.content)).convert('RGB')
        self.assertEqual(drawn.getpixel((1000, 280)), (200, 30, 30))  # the figure, on the right

    def test_no_figure_for_non_commercial_licenses_nor_papers_without_one(self):
        self.paper.license = 'http://creativecommons.org/licenses/by-nc/4.0/'
        self.assertIsNone(social.figure_of(self.paper))
        self.paper.license = CC_BY
        with mock.patch('summarizer.chat_figures.paper_gallery', return_value={'figures': []}):
            social.fetch_figure('2401.04088v1')
        self.assertIsNone(social.figure_of(self.paper))
        self.assertEqual(Client().get('/og/en/p/2401.04088v1.png')['Cache-Control'], social.CACHE)

    def test_papers_never_summarized_have_no_picture(self):
        ArxivPaper.objects.create(arxiv_id='2401.00001v1', title='Bare', abstract='Abstract')
        self.assertEqual(Client().get('/og/en/p/2401.00001v1.png').status_code, 404)

    def test_too_many_pictures_at_once_get_a_short_503(self):
        with mock.patch.object(social.RENDER_SLOTS, 'acquire', return_value=False), mock.patch('summarizer.social.threading.Thread'):
            response = Client().get('/og/en/p/2401.04088v1.png')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response['Retry-After'], '10')


class AuthorTests(TestCase):
    def setUp(self):
        self.paper = ArxivPaper.objects.create(arxiv_id='2401.04088v1', title='Gas in disks', abstract='Abstract.',
                                               category='astro-ph.EP', published_arxiv=datetime.date.today())
        SummaryPaper.objects.create(paper=self.paper, lang='en', summary='We show a thing. And more.', notes="['Point one']")
        author = Author.objects.create(name='Quentin Kral')
        PaperAuthor.objects.create(author=author, paper=self.paper, author_order=1)
        self.user = User.objects.create_user('qk', 'qk@example.org', 'x')

    def box(self):
        client = Client()
        client.force_login(self.user)
        return BeautifulSoup(client.get('/en/paper/2401.04088v1/').content, 'html.parser').select_one('[data-share]')

    def test_an_author_gets_a_first_person_post_unfolded(self):
        box = self.box()
        self.assertFalse(box.has_attr('open'))
        self.assertIn('Gas in disks\n', box.select_one('textarea').get_text())
        self.user.first_name, self.user.last_name = 'Q.', 'Kral'
        self.user.save()
        box = self.box()
        self.assertTrue(box.has_attr('open'))
        text = box.select_one('textarea').get_text()
        self.assertTrue(text.startswith('Our new paper: Gas in disks'))
        self.assertIn('What we found:', text)
        self.assertIn('You sign this paper', box.get_text())
        self.paper.published_arxiv = datetime.date(2024, 1, 1)
        self.paper.save()
        self.assertTrue(self.box().select_one('textarea').get_text().startswith('Our paper: Gas in disks'))

    def test_google_sign_in_keeps_the_name_when_the_account_has_none(self):
        accounts.remember_name(self.user, {'given_name': 'Quentin', 'family_name': 'Kral'})
        self.user.refresh_from_db()
        self.assertEqual((self.user.first_name, self.user.last_name), ('Quentin', 'Kral'))
        accounts.remember_name(self.user, {'given_name': 'Other', 'family_name': 'Name'})
        self.user.refresh_from_db()
        self.assertEqual(self.user.last_name, 'Kral')


class TopicTests(TestCase):
    def setUp(self):
        RisingTopics.objects.create(scope='*', lang='en', day=WEEK, content=json.dumps({'topics': [topic(i) for i in range(1, 6)]}))

    def test_each_free_topic_has_its_post_and_picture(self):
        soup = BeautifulSoup(Client().get('/en/trends/').content, 'html.parser')
        boxes = soup.select('.topic-card [data-share]')
        self.assertEqual([b['id'] for b in boxes], ['share-topic-1', 'share-topic-2', 'share-topic-3'])
        text = boxes[1].select_one('textarea').get_text()
        self.assertTrue(text.startswith('Rising this week on arXiv: Topic 2'))
        self.assertIn('×4.2, 12 papers from 7 teams', text)
        self.assertIn('What topic 2 is about.', text)
        self.assertIn('https://www.summarizepaper.com/en/trends/?week=2026-10-26&topic=2', text)
        self.assertEqual(len(soup.select('[data-share-open]')), 3)
        self.assertEqual(Client().get('/og/en/topic/arxiv/2026-10-26/2.png').status_code, 200)
        self.assertEqual(Client().get('/og/en/topic/arxiv/2026-10-26/4.png').status_code, 404)

    def test_a_shared_topic_link_previews_that_topic(self):
        soup = BeautifulSoup(Client().get('/en/trends/', {'week': '2026-10-26', 'topic': '2'}).content, 'html.parser')
        self.assertEqual(soup.select_one('meta[property="og:title"]')['content'], 'Topic 2')
        self.assertEqual(soup.select_one('meta[property="og:image"]')['content'],
                         'https://www.summarizepaper.com/og/en/topic/arxiv/2026-10-26/2.png')
        self.assertIn('is-shared', soup.select_one('#topic-2')['class'])
        soup = BeautifulSoup(Client().get('/en/trends/', {'topic': '9'}).content, 'html.parser')
        self.assertEqual(soup.select_one('meta[property="og:title"]')['content'], 'Rising this week on arXiv')


class EmailTests(TestCase):
    @override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
    def test_the_letter_email_links_to_its_post(self):
        content = {'sections': [{'heading': 'Stars', 'text': 'Text.', 'papers': []}], 'papers': {}, 'closing': '', 'count': 6}
        Letter.objects.create(topic='space', lang='en', day=WEEK, title='Space this week', dek='News.', content=json.dumps(content))
        user = User.objects.create_user('reader', 'reader@example.org', 'x')
        LetterSubscription.objects.create(user=user, topic='space', lang='en')
        self.assertEqual(letters.send(None, 5, base_url='https://www.summarizepaper.com'), 1)
        share = 'https://www.summarizepaper.com/en/letters/space/2026-10-26/?utm_source=letter&utm_medium=email&utm_content=share#share'
        self.assertIn(share, mail.outbox[0].body)
        self.assertIn(share.replace('&', '&amp;'), mail.outbox[0].alternatives[0][0])
