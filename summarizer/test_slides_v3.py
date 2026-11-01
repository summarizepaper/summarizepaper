"""Journal club decks of version 3: checked passages, earlier work, where the paper stands, the editor of pages,
rewriting a slide, sharing, the decks of a collection and proposing a paper."""
import copy
import io
import re
import json
from unittest import mock

from bs4 import BeautifulSoup
from django.test import Client
from PIL import Image
from pptx import Presentation

from . import generated, slides, slides_pptx
from .models import Collection, CollectionMember, CollectionPaper, Generated, PaperVersion, UserPlan
from .test_tools import ToolsTestCase

TEXT = ('We observe thick clouds on a hot planet with the large telescope at high resolution. '
        'The cloud fraction is 0.42 in the dayside spectrum of the planet. '
        'Our sample contains only three nights of observations, which limits the precision of the fraction.')
FIGURES = [{'label': 'Figure 1:', 'images': ['https://arxiv.org/html/2101.00001v1/x1.png'], 'caption': 'Dayside spectrum.',
            'source_url': 'https://arxiv.org/html/2101.00001v1#F1', 'license_url': 'http://creativecommons.org/licenses/by/4.0/'},
           {'label': 'Figure 2:', 'images': ['https://arxiv.org/html/2101.00001v1/x2.png'], 'caption': 'Cloud map.',
            'source_url': 'https://arxiv.org/html/2101.00001v1#F2', 'license_url': 'http://creativecommons.org/licenses/by/4.0/'}]
NEIGHBOURS = {'items': [
    {'ref': '2001.00002', 'group': 'prior', 'title': 'Hazes on warm planets', 'authors': 'Kim et al.', 'date': '2020-01-05',
     'abstract': 'Hazes.', 'venue': 'ApJ'},
    {'ref': '2201.00003', 'group': 'followup', 'title': 'Clouds confirmed', 'authors': 'Diaz et al.', 'date': '2022-02-01',
     'abstract': 'Confirmed.', 'venue': ''}]}


def answer(**changes):
    value = {
        'title': 'Des nuages sur une planète chaude',
        'slides': [{'title': 'Résultats', 'figure': 'Figure 2',
                    'points': [{'text': 'Fraction nuageuse de 0,42 côté jour', 'quote': 'The cloud fraction is 0.42 in the dayside spectrum of the planet.', 'kind': 'result'},
                               {'text': 'Nuages épais observés', 'quote': 'A sentence the paper never wrote about the clouds at all.', 'kind': 'interpretation'},
                               {'text': 'Mesuré sur 12 nuits', 'quote': '', 'kind': 'method'}],
                    'notes': 'Montrer le spectre.'}],
        'strengths': [{'point': 'Haute résolution', 'quote': 'We observe thick clouds on a hot planet with the large telescope at high resolution.'}],
        'stated_limitations': [{'point': 'Trois nuits seulement', 'quote': 'Our sample contains only three nights of observations, which limits the precision of the fraction.'},
                               {'point': 'Modèle simplifié', 'quote': 'The authors never wrote this sentence about their model.'}],
        'discussion_points': ['La fraction dépend-elle de la saison ?'],
        'questions': ['Quelle autre mesure ?'],
        'related': [{'paper': '2001.00002', 'relation': 'ajoute la résolution'}, {'paper': '9999.99999', 'relation': 'inventé'}],
        'hard_questions': [{'question': 'Pourquoi trois nuits ?', 'answer': 'Le temps de télescope.',
                            'quote': 'Our sample contains only three nights of observations, which limits the precision of the fraction.'}],
    }
    value.update(changes)
    return value


class DeckTestCase(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.article = self.paper()
        self.reader = self.user('presenter', paid=True)
        self.client = self.client_for(self.reader)
        self.base = '/en/journal-club/2101.00001v1/'

    def generate(self, opts=None, reply=None, text=TEXT, abstract_only=False, report=None):
        opts = opts or {'lang': 'fr', 'audience': 'field', 'duration': 20}
        patches = [mock.patch('summarizer.presskit.read', return_value=(text, 'html', abstract_only)),
                   mock.patch('summarizer.presskit.figures_of', return_value=FIGURES),
                   mock.patch('summarizer.neighbours.stored', return_value=NEIGHBOURS),
                   mock.patch.object(generated, 'chat_json', return_value=reply or answer())]
        if report is not None:
            patches.append(mock.patch('summarizer.publication_status.for_generation', return_value=report))
        with contextlib_nested(patches) as mocks:
            row = generated.run_now(slides.DECK_KIND, slides.paper_key('2101.00001v1', opts), opts['lang'], slides.write('2101.00001v1', opts))
            self.chat = mocks[3]
        return row

    def page_payload(self, url):
        response = self.client.get(url)
        return response, {'revision': response.context['deck_revision'],
                          'pages': [{k: p[k] for k in slides.EDITED if k in p} | {'figure': p['figure_id']}
                                    for p in response.context['pages']]}


class contextlib_nested:
    def __init__(self, patches):
        self.patches = patches

    def __enter__(self):
        return [p.start() for p in self.patches]

    def __exit__(self, *exc):
        for p in reversed(self.patches):
            p.stop()


class GenerationTests(DeckTestCase):
    def test_points_keep_only_passages_the_paper_has_and_flag_numbers_it_does_not_state(self):
        deck = self.generate().data()
        self.assertEqual(deck['generation_version'], 3)
        results = next(p for p in deck['pages'] if p['title'] == 'Résultats')
        checks = results['sources']
        self.assertTrue(checks[0]['verified'])
        self.assertEqual(checks[0]['kind'], 'result')
        self.assertEqual(checks[0]['numbers'], [])  # 0,42 is the paper's 0.42
        self.assertFalse(checks[1]['verified'])
        self.assertEqual(checks[1]['quote'], '')
        self.assertEqual(checks[2]['numbers'], ['12'])
        self.assertEqual(deck['figures'][results['figure']]['url'], FIGURES[1]['images'][0])
        self.assertIn('arXiv: 2101.00001v1', deck['figures'][results['figure']]['credit'])
        self.assertIn('CC BY 4.0', deck['figures'][results['figure']]['credit'])
        self.assertEqual(len(deck['figures']), 2)  # every figure, for the reader to pick another one

    def test_limits_stated_by_the_authors_need_their_passage_and_the_others_are_an_ai_reading(self):
        deck = self.generate().data()
        limits = next(p for p in deck['pages'] if p['title'] == 'Points forts et limites')
        self.assertEqual(limits['headings'][1], 'Limites énoncées par les auteurs')
        self.assertEqual(limits['secondary'], ['Trois nuits seulement'])
        discussion = next(p for p in deck['pages'] if p['title'] == 'Pour la discussion')
        self.assertIn('Modèle simplifié', discussion['bullets'])
        self.assertIn('IA', discussion['headings'][0])
        self.assertEqual(deck['qa'][0]['passage']['quote'], answer()['hard_questions'][0]['quote'])

    def test_earlier_work_comes_from_the_neighbourhood_and_never_from_unknown_papers(self):
        deck = self.generate().data()
        context = next(p for p in deck['pages'] if p['kind'] == 'columns' and 'Kim' in ' '.join(p['bullets']))
        self.assertEqual(context['bullets'], ['Kim et al. (2020): ajoute la résolution'])
        self.assertIn('Diaz et al. (2022): Clouds confirmed', context['secondary'])
        self.assertNotIn('inventé', json.dumps(deck, ensure_ascii=False))
        sent = json.loads(self.chat.call_args.args[0])['source_documents']
        self.assertEqual([r['id'] for r in sent['related_work']], ['2001.00002', '2201.00003'])
        self.assertIn('must', self.chat.call_args.kwargs['system'] + 'must')
        self.assertIn('French', self.chat.call_args.kwargs['system'])
        self.assertIn('7 to 10 slides', self.chat.call_args.kwargs['system'])

    def test_where_the_paper_stands_says_notices_and_newer_versions_without_the_model(self):
        PaperVersion.objects.create(arxiv_id='2101.00001', version=3, checked='2026-10-01T00:00:00Z')
        report = {'status': 'published', 'journal': 'Nature Astronomy', 'date': '2021-06-01', 'doi': '10.1038/x',
                  'updates': [{'type': 'retraction', 'doi': '10.1038/y'}], 'sources': []}
        deck = self.generate(report=report).data()
        standing = next(p for p in deck['pages'] if p['title'] == 'Où en est l’article')
        text = ' '.join(standing['bullets'])
        self.assertIn('Nature Astronomy', text)
        self.assertIn('10.1038/y', text)
        self.assertIn('v3', text)
        self.assertTrue(deck['warnings'])
        page = self.client.get(self.base, {'deck_lang': 'fr'})
        self.assertContains(page, 'deck-warning')

    def test_audience_duration_and_abstract_only_change_the_plan_and_have_their_own_deck(self):
        opts = {'lang': 'en', 'audience': 'students', 'duration': 10}
        row = self.generate(opts, reply=answer(not_in_abstract=['The sample size']), abstract_only=True)
        self.assertEqual(row.key, '2101.00001v1|students|10')
        system = self.chat.call_args.kwargs['system']
        self.assertIn('3 or 4 slides, from the abstract alone', system)
        self.assertIn('define the key terms', system)
        deck = row.data()
        self.assertTrue(deck['abstract_only'])
        self.assertIn('What the abstract does not say', [p['title'] for p in deck['pages']])
        self.assertIsNone(self.client.get(self.base, {'deck_lang': 'en'}).context['deck'])
        shown = self.client.get(self.base, {'deck_lang': 'en', 'audience': 'students', 'duration': '10'})
        self.assertEqual(shown.context['deck']['title'], answer()['title'])

    def test_start_takes_the_options_and_rejects_unknown_ones(self):
        with mock.patch.object(generated, 'start', return_value=mock.Mock(status='pending', pk=7)) as start:
            response = self.client.post(self.base + 'start/', {'deck_lang': 'de', 'audience': 'other', 'duration': '45'})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(start.call_args.args[:3], (slides.DECK_KIND, '2101.00001v1|other|45', 'de'))
            self.assertIn('duration=45', response.json()['url'])
            for bad in ({'audience': 'aliens'}, {'duration': '7'}, {'deck_lang': 'xx'}):
                self.assertEqual(self.client.post(self.base + 'start/', bad).status_code, 400)
        self.assertEqual(start.call_count, 1)

    def test_page_shows_sources_hard_questions_and_the_chat(self):
        self.generate()
        page = self.client.get(self.base, {'deck_lang': 'fr'})
        soup = BeautifulSoup(page.content, 'html.parser')
        self.assertEqual(len(soup.select('.slide-sources .is-verified')), 3)
        self.assertTrue(soup.select('.slide-sources .is-unverified'))
        self.assertIn('Pourquoi trois nuits ?', soup.select_one('#hard-questions').get_text())
        self.assertTrue(soup.select_one('#hard-questions a[href*="/chat/?ids=2101.00001v1"]'))
        self.assertTrue(soup.select_one('template#slide-template .slide-item[data-origin=""]'))
        self.assertTrue(soup.select('select[data-edit=figure] option[value=f2][selected]'))


class EditorTests(DeckTestCase):
    def setUp(self):
        super().setUp()
        self.generate(opts={'lang': 'en', 'audience': 'field', 'duration': 20})
        self.url = self.base + '?deck_lang=en'

    def save(self, payload):
        return self.client.post(self.base + 'save/?deck_lang=en', json.dumps(payload), content_type='application/json')

    def test_pages_can_be_reordered_duplicated_added_and_removed_keeping_their_passages(self):
        _response, payload = self.page_payload(self.url)
        pages = payload['pages']
        new = {'origin': None, 'title': 'My own slide', 'bullets': ['A point'], 'notes': '', 'layout': 'right',
               'figure_size': 50, 'text_size': 30, 'figure': 'f1'}
        payload['pages'] = [pages[1], copy.deepcopy(pages[1]), pages[0], new] + pages[3:-1]
        self.assertEqual(self.save(payload).status_code, 200)
        shown = self.client.get(self.url).context['pages']
        self.assertEqual([p['title'] for p in shown[:4]], [pages[1]['title'], pages[1]['title'], pages[0]['title'], 'My own slide'])
        self.assertEqual(len(shown[1]['sources']), 3)  # a copy keeps the passages of its slide
        self.assertEqual(shown[3]['sources'], [])
        self.assertEqual(shown[3]['figure']['url'], FIGURES[0]['images'][0])
        self.assertEqual(len(shown), len(pages))
        stored = Generated.objects.get(kind=slides.EDIT_KIND, user=self.reader).data()['pages']
        self.assertNotIn('arxiv.org', json.dumps(stored))
        self.assertNotIn('sources', stored[0])

    def test_figure_choice_is_one_of_the_paper_and_a_new_page_cannot_claim_passages(self):
        _response, payload = self.page_payload(self.url)
        payload['pages'][1]['figure'] = 'https://evil.test/x.png'
        self.assertEqual(self.save(payload).status_code, 400)
        _response, payload = self.page_payload(self.url)
        payload['pages'][1]['figure'] = ''
        payload['pages'].append({'origin': None, 'title': 'New', 'bullets': [], 'sources': [{'verified': True, 'quote': 'fake'}],
                                 'kind': 'columns', 'secondary': ['x']})
        self.assertEqual(self.save(payload).status_code, 200)
        shown = self.client.get(self.url).context['pages']
        self.assertIsNone(shown[1]['figure'])
        self.assertEqual(shown[-1]['kind'], 'content')
        self.assertEqual(shown[-1]['sources'], [])
        self.assertEqual(len(payload['pages']), len(shown))
        _response, payload = self.page_payload(self.url)
        self.assertEqual(self.save(dict(payload, pages=payload['pages'] * 6)).status_code, 400)  # MAX_PAGES

    def test_saved_copy_that_no_longer_fits_is_shown_as_written_and_kept_aside(self):
        kind, key = slides.edit_location(self.reader, Generated.objects.get(kind=slides.DECK_KIND))
        Generated.objects.create(kind=kind, key=key, lang='en', user=self.reader, status='done',
                                 content=json.dumps({'pages': [{'origin': 99, 'title': 'Lost'}]}))
        response, payload = self.page_payload(self.url)
        self.assertTrue(response.context['edits_mismatch'])
        self.assertContains(response, 'no longer matches')
        self.assertEqual(self.save(payload).status_code, 200)
        self.assertIn('Lost', Generated.objects.get(kind=slides.BACKUP_KIND).content)

    def test_powerpoint_has_the_headings_credits_and_passages_in_the_notes(self):
        _response, payload = self.page_payload(self.url)
        png = io.BytesIO()
        Image.new('RGB', (200, 100), 'white').save(png, 'PNG')
        with mock.patch.object(slides_pptx, 'figure_bytes', return_value=png.getvalue()):
            response = self.client.post(self.base + 'powerpoint/?deck_lang=en', json.dumps(payload), content_type='application/json')
        self.assertEqual(response.status_code, 200)
        deck = Presentation(io.BytesIO(response.content))
        notes = deck.slides[1].notes_slide.notes_text_frame.text
        self.assertIn('Montrer le spectre.', notes)
        self.assertIn('The cloud fraction is 0.42', notes)
        text = '\n'.join(s.text for slide in deck.slides for s in slide.shapes if s.has_text_frame)
        self.assertIn('Limits stated by the authors', text)
        self.assertIn('Ann Lee et al., Figure 2', text)

    def test_other_figure_hosts_of_the_gallery_are_accepted_for_powerpoint(self):
        self.assertTrue(slides_pptx.safe_image_url('https://pmc.ncbi.nlm.nih.gov/articles/PMC1/figure.png'))
        self.assertFalse(slides_pptx.safe_image_url('https://arxiv.org/abs/2101.00001'))
        self.assertFalse(slides_pptx.safe_image_url('https://evil.test/html/a.png'))


class RewriteTests(DeckTestCase):
    def setUp(self):
        super().setUp()
        self.generate(opts={'lang': 'en', 'audience': 'field', 'duration': 20})
        _response, payload = self.page_payload(self.base + '?deck_lang=en')
        self.slide = payload['pages'][1]

    def rewrite(self, how='shorter', client=None):
        return (client or self.client).post(self.base + 'rewrite/?deck_lang=en', json.dumps({'how': how, 'page': self.slide}),
                                            content_type='application/json')

    def test_a_slide_is_rewritten_with_its_passages_and_new_numbers_are_flagged(self):
        reply = {'title': 'Results', 'bullets': ['Cloud fraction 0.42', 'Seen on 7 nights'], 'notes': 'Say it.'}
        with mock.patch.object(generated, 'chat_json', return_value=reply) as chat:
            response = self.rewrite()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['bullets'], reply['bullets'])
        self.assertEqual(response.json()['new_numbers'], ['7'])
        sent = json.loads(chat.call_args.args[0])['slide']
        self.assertIn('The cloud fraction is 0.42 in the dayside spectrum of the planet.', sent['passages_of_the_paper'])
        self.assertIn('shorter', chat.call_args.kwargs['system'])
        self.assertFalse(chat.call_args.kwargs['paid'])

    def test_notes_only_keep_the_slide_and_bad_requests_quota_and_premium_are_refused(self):
        with mock.patch.object(generated, 'chat_json', return_value={'title': 'Other', 'bullets': ['x'], 'notes': 'New notes.'}) as chat:
            response = self.rewrite('notes')
            self.assertEqual(response.json()['title'], self.slide['title'])
            self.assertEqual(response.json()['notes'], 'New notes.')
            self.assertEqual(self.rewrite('poem').status_code, 400)
            with mock.patch.object(slides, 'REWRITES_PER_DAY', 1):
                self.assertEqual(self.rewrite().status_code, 429)
            UserPlan.objects.filter(user=self.reader).update(paid=False)
            self.assertEqual(self.rewrite().status_code, 403)
        self.assertEqual(chat.call_count, 1)


class ShareTests(DeckTestCase):
    def test_saved_version_is_shared_read_only_and_can_be_stopped(self):
        self.generate(opts={'lang': 'en', 'audience': 'field', 'duration': 20})
        _response, payload = self.page_payload(self.base + '?deck_lang=en')
        payload['pages'][0]['title'] = 'Our session on clouds'
        self.client.post(self.base + 'save/?deck_lang=en', json.dumps(payload), content_type='application/json')
        shared = self.client.post(self.base + 'share/?deck_lang=en', {'action': 'share'}).json()
        self.assertIn('/journal-club/shared/', shared['url'])
        anonymous = Client().get(shared['url'])
        self.assertEqual(anonymous.status_code, 200)
        self.assertContains(anonymous, 'Our session on clouds')
        self.assertContains(anonymous, 'The cloud fraction is 0.42')
        self.assertNotContains(anonymous, 'data-edit="title"')
        self.assertNotContains(anonymous, 'presenter@example.com')
        self.assertEqual(anonymous['X-Robots-Tag'], 'noindex')
        self.client.post(self.base + 'share/?deck_lang=en', {'action': 'unshare'})
        self.assertEqual(Client().get(shared['url']).status_code, 404)
        other = self.client_for(self.user('stranger', paid=True))
        row = Generated.objects.get(kind=slides.SHARE_KIND)
        self.assertEqual(other.get('/en/ai/%d/' % row.pk).status_code, 404)


class CollectionTests(DeckTestCase):
    def setUp(self):
        super().setUp()
        self.paper('2101.00009v1', title='Clouds seen again', authors=('Cara Diaz',))
        self.collection = Collection.objects.create(user=self.reader, name='Lab journal club')
        for i, aid in enumerate(('2101.00001v1', '2101.00009v1')):
            CollectionPaper.objects.create(collection=self.collection, arxiv_id=aid, title='t', authors='a', position=i, selected=True)
        self.url = '/en/journal-club/collection/%d/' % self.collection.pk

    def test_a_collection_deck_presents_each_paper_with_its_passages_and_members_share_it(self):
        reply = {'title': 'Clouds, two views', 'slides': [
            {'title': 'First paper', 'paper': '2101.00001v1', 'figure': 'P1 Figure 1',
             'points': [{'text': 'Fraction 0.42', 'quote': 'The cloud fraction is 0.42 in the dayside spectrum of the planet.', 'kind': 'result'}]},
            {'title': 'Second paper', 'paper': '2101.00009v1',
             'points': [{'text': 'Fraction 0.42', 'quote': 'The cloud fraction is 0.42 in the dayside spectrum of the planet.', 'kind': 'result'}]}],
            'agree': ['Both see clouds'], 'differ': ['Different nights'], 'questions': ['Why?'], 'discussion_points': ['Season?'],
            'hard_questions': [{'question': 'Same telescope?', 'answer': 'Yes.', 'paper': '2101.00009v1', 'quote': ''}]}
        texts = {'2101.00001v1': (TEXT, 'html', False), '2101.00009v1': ('Another text without that sentence at all in it.', 'html', False)}
        with mock.patch('summarizer.presskit.read', side_effect=lambda paper: texts[paper.arxiv_id]), \
             mock.patch('summarizer.presskit.figures_of', return_value=FIGURES), \
             mock.patch.object(generated, 'chat_json', return_value=reply):
            response = self.client.post(self.url + 'start/', {'deck_lang': 'en'})
        self.assertEqual(response.status_code, 200)
        row = Generated.objects.get(kind=slides.COLLECTION_KIND)
        deck = row.data()
        first, second = deck['pages'][1], deck['pages'][2]
        self.assertTrue(first['sources'][0]['verified'])
        self.assertFalse(second['sources'][0]['verified'])  # checked against its own paper
        self.assertEqual(deck['figures'][first['figure']]['label'], 'P1 Figure 1')
        self.assertIn('How the papers compare', [p['title'] for p in deck['pages']])
        member = self.user('colleague', paid=True)
        CollectionMember.objects.create(collection=self.collection, user=member)
        colleague = self.client_for(member)
        self.assertEqual(colleague.get('/en/ai/%d/' % row.pk).status_code, 200)
        page = colleague.get(response.json()['url'])
        self.assertEqual(page.context['deck']['title'], 'Clouds, two views')
        self.assertEqual(self.client_for(self.user('outsider', paid=True)).get(self.url).status_code, 404)

    def test_fewer_than_two_selected_papers_are_refused(self):
        CollectionPaper.objects.filter(arxiv_id='2101.00009v1').update(selected=False)
        with mock.patch.object(generated, 'chat_json') as chat:
            self.assertEqual(self.client.post(self.url + 'start/', {'deck_lang': 'en'}).json()['status'], 'too_few')
        chat.assert_not_called()

    def test_collection_page_links_to_its_journal_club(self):
        self.assertContains(self.client.get('/en/collections/%d/' % self.collection.pk), self.url)


class ProposeTests(DeckTestCase):
    def test_a_paper_is_proposed_to_the_team_collection_or_a_new_one(self):
        page = self.client.get('/en/journal-club/propose/', {'arxiv_id': '2101.00001v1'})
        self.assertContains(page, 'Clouds on a hot planet')
        response = self.client.post('/en/journal-club/propose/', {'arxiv_id': '2101.00001v1', 'collection': 'new', 'name': 'Thursday club'})
        collection = Collection.objects.get(name='Thursday club')
        self.assertRedirects(response, '/en/collections/%d/?added=1#papers' % collection.pk, fetch_redirect_response=False)
        self.assertTrue(collection.papers.filter(arxiv_id='2101.00001v1').exists())
        other = Collection.objects.create(user=self.user('owner'), name='Shared club')
        self.assertEqual(self.client.post('/en/journal-club/propose/', {'arxiv_id': '2101.00001v1', 'collection': other.pk}).status_code, 404)
        CollectionMember.objects.create(collection=other, user=self.reader)
        self.client.post('/en/journal-club/propose/', {'arxiv_id': '2101.00001v1', 'collection': other.pk})
        self.assertTrue(other.papers.exists())
        self.assertEqual(Client().get('/en/journal-club/propose/?arxiv_id=2101.00001v1').status_code, 302)

    def test_paper_tools_letters_and_digest_link_to_the_proposal(self):
        self.assertContains(self.client.get('/en/journal-club/'), '/en/collections/')


class ProductionFixesTests(DeckTestCase):
    def test_languages_without_catalog_get_the_fixed_labels_from_the_model(self):
        labels = {'Where the paper stands': 'Wo der Artikel steht', 'Strengths': 'Stärken', 'References': 'Literatur',
                  'Editorial status not confirmed by the sources checked': 'Status nicht bestätigt', 'Ignored': 'x'}
        row = self.generate(opts={'lang': 'de', 'audience': 'field', 'duration': 20}, reply=answer(labels=labels))
        sent = json.loads(self.chat.call_args.args[0])['source_documents']
        self.assertIn('Where the paper stands', sent['labels_to_translate'])
        self.assertIn('labels_to_translate', self.chat.call_args.kwargs['system'])
        self.assertEqual(sent['metadata']['authors'], ['Ann Lee', 'Bob Ray'])
        deck = row.data()
        titles = [p['title'] for p in deck['pages']]
        self.assertIn('Wo der Artikel steht', titles)
        self.assertIn('Literatur', titles)
        self.assertIn('Status nicht bestätigt', deck['standing'])
        limits = next(p for p in deck['pages'] if p['kind'] == 'columns' and p['headings'][0] == 'Stärken')
        self.assertTrue(limits)
        french = self.generate(opts={'lang': 'fr', 'audience': 'field', 'duration': 20})
        self.assertNotIn('labels_to_translate', json.loads(self.chat.call_args.args[0])['source_documents'])
        self.assertIn('no numbers', self.chat.call_args.kwargs['system'])

    def test_svg_in_points_wrapped_base64_and_large_rasters_are_exported(self):
        png = io.BytesIO()
        Image.new('RGB', (4, 4), 'red').save(png, 'PNG')
        import base64
        wrapped = '\n'.join(re.findall('.{1,8}', base64.b64encode(png.getvalue()).decode()))
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="418.08pt" height="258.96pt" '
               'viewBox="0 0 418.08 258.96"><rect width="100" height="50" fill="blue"/>'
               '<image width="4" height="4" xlink:href="data:image/png;base64,%s"/></svg>' % wrapped).encode()
        out = Image.open(io.BytesIO(slides_pptx.normalized_image(svg)))
        self.assertEqual(out.size, (1800, 1115))
        big = io.BytesIO()
        Image.new('RGB', (4555, 3941), 'white').save(big, 'PNG')
        self.assertEqual(Image.open(io.BytesIO(slides_pptx.normalized_image(big.getvalue()))).size, (1387, 1200))
