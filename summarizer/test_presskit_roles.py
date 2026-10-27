"""Press kit, version 3: the parts of each role, passages found as a PDF extraction writes them, numbers to check,
the novelty against earlier papers, outside commentators without the authors' co-authors, figures with their
credit, and the share link, exports, reports and example kit around it."""
import datetime
import json
from unittest import mock

from django.test import Client

from . import presskit
from .models import Generated, PaperScore, PaperText, PaperVersion, PressKitReport
from .test_tools import ToolsTestCase, base_call, offline

SOURCE = ('## Results\nWe observe a change of +5 percent in a sample of 120 objects.\n\n'
          'The signiﬁcance of the change is 4 sigma for the whole sample of objects.\n\n'
          '## Discussion\nThe association does not establish a causal mechanism for the change.')


def kit_answer(**changes):
    answer = {'headline': 'A measured change', 'plain_explanation': 'A change of 5 percent in 120 objects, 42 percent faster.',
              'whats_new': 'A larger sample.', 'verdict': 'Cautious.',
              'hype_check': [{'signal': 'Preprint', 'detail': 'Not reviewed.', 'level': 'warning'}],
              'evidence_cards': [{'claim': 'A 4 sigma change', 'kind': 'result',
                                  'quote': 'The significance of the change is 4 sigma for the whole sample of objects.',
                                  'why_it_matters': 'Strong.', 'limitation': 'One sample.'}],
              'questions_for_authors': [{'question': 'How large is the sample?', 'in_paper': 'It has 120 objects.',
                                         'quote': 'We observe a change of +5 percent in a sample of 120 objects.'},
                                        {'question': 'Who funded it?', 'in_paper': '', 'quote': ''}],
              'glossary': []}
    answer.update(changes)
    return answer


ROLES = {'author': {'press_release': {'title': 'Objects change', 'lede': 'The team measures a change.',
                                      'body': ['First paragraph.', 'Second paragraph.'], 'quote_slot': 'why it matters'},
                    'likely_questions': [{'question': 'Is it causal?', 'answer': 'No.',
                                          'quote': 'The association does not establish a causal mechanism for the change.'},
                                         {'question': 'Is it new?', 'answer': 'Yes.', 'quote': 'An invented passage that the paper never contains at all.'}],
                    'misreadings': [{'misreading': 'It is a cause', 'correction': 'An association.'}],
                    'lay_summary': 'Objects change a little.', 'social_posts': ['A change in 120 objects.', 'Seen in 260 objects.']},
         'commentator': {'in_brief': ['It shows a change.', 'It does not show a cause.'],
                         'strengths': [{'point': 'A large sample', 'quote': 'We observe a change of +5 percent in a sample of 120 objects.'}],
                         'weaknesses': [{'point': 'No mechanism', 'quote': ''}],
                         'what_would_convince': 'A second sample.', 'comment_draft': 'Solid measure, open cause.'}}


def answers(base=None, roles=None):
    def chat(prompt, **kwargs):
        return base or kit_answer() if 'evidence_cards' in kwargs['system'] else roles or ROLES
    return chat


class PressKitRoleTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.paper_obj = self.paper()
        PaperText.objects.create(arxiv_id=self.paper_obj.arxiv_id, text=SOURCE, source='html')
        self.reader = self.user('press-reader', paid=True)
        self.paid = self.client_for(self.reader)
        self.base = '/en/press-kit/2101.00001v1/'

    def generate(self, chat=None, **patches):
        with offline(), mock.patch('summarizer.generated.chat_json', side_effect=chat or answers()) as llm:
            started = [mock.patch('summarizer.presskit.%s' % name, return_value=value) for name, value in patches.items()]
            for patch in started:
                patch.start()
            try:
                self.paid.post(self.base + 'start/')
            finally:
                for patch in started:
                    patch.stop()
        return Generated.objects.get(kind='presskit').data(), llm

    def test_role_parts_keep_only_real_passages_and_the_page_opens_on_the_role(self):
        kit, llm = self.generate()
        self.assertEqual(kit['schema_version'], 3)
        self.assertEqual(llm.call_count, 2)
        author = kit['roles']['author']
        self.assertEqual(author['numbers_absent'], ['260'])  # the thread says 260 objects, the paper 120
        self.assertEqual(kit['roles']['commentator']['numbers_absent'], [])
        role_call = next(c for c in llm.call_args_list if 'evidence_cards' not in c.kwargs['system'])
        documents = json.loads(role_call.args[0])['source_documents']
        self.assertEqual(sorted(documents), ['metadata', 'publication_check', 'related_work', 'text'])
        self.assertIn('never write that the paper is published or peer reviewed', role_call.kwargs['system'])
        self.assertEqual(author['likely_questions'][0]['passage']['location'], 'Discussion')
        self.assertIsNone(author['likely_questions'][1]['passage'])
        self.assertEqual(kit['roles']['commentator']['strengths'][0]['passage']['location'], 'Results')
        page = self.paid.get(self.base + '?role=author')
        self.assertEqual(page.context['role'], 'author')
        self.assertContains(page, 'Draft press release')
        self.assertContains(page, 'Numbers of these texts that the paper text does not state: 260.')
        self.assertContains(page, 'data-role-panel="commentator" hidden')
        self.assertNotContains(page, 'An invented passage')
        self.assertEqual(self.paid.get(self.base).context['role'], 'commentator')
        self.reader.first_name, self.reader.last_name = 'Ann', 'Lee'  # signs the paper
        self.reader.save()
        self.assertEqual(self.paid.get(self.base).context['role'], 'author')

    def test_role_call_failure_keeps_the_kit(self):
        def chat(prompt, **kwargs):
            if 'evidence_cards' in kwargs['system']:
                return kit_answer()
            raise RuntimeError('timeout')
        kit, _ = self.generate(chat)
        self.assertEqual(kit['headline'], 'A measured change')
        self.assertEqual(kit['roles']['author']['likely_questions'], [])
        self.assertContains(self.paid.get(self.base + '?role=author'), 'This kit was written before this part existed')

    def test_pdf_ligature_passage_is_verified_with_its_place_kind_and_numbers_to_check(self):
        kit, _ = self.generate()
        card = kit['evidence_cards'][0]
        self.assertTrue(card['verified'])
        self.assertIn('signiﬁcance', card['quote'])
        self.assertEqual((card['location'], card['kind'], card['numeric_mismatch']), ('Results', 'result', False))
        self.assertEqual(card['paragraph'], 2)
        self.assertIn('arxiv.org/html/2101.00001v1#:~:text=', card['link'])
        self.assertEqual(kit['numbers_absent'], ['42'])
        page = self.paid.get(self.base)
        self.assertContains(page, 'paper text does not state: 42')
        self.assertContains(page, 'Copy with reference')
        self.assertContains(page, 'data-citation="Ann Lee et al., arXiv: 2101.00001v1"')

    def test_questions_say_what_the_paper_already_answers(self):
        kit, _ = self.generate()
        self.assertEqual(kit['questions_for_authors'], ['How large is the sample?', 'Who funded it?'])
        self.assertEqual(list(kit['question_answers']), ['How large is the sample?'])
        page = self.paid.get(self.base + '?role=writer')
        self.assertContains(page, 'The paper already says:')
        self.assertContains(page, '/en/chat/?ids=2101.00001v1&amp;ask=Who%20funded%20it%3F')
        result = self.paid.get(self.base + 'publication/').json()
        self.assertIn('The paper already says:', result['questions_html'])
        self.assertEqual(result['questions'], ['How large is the sample?', 'Who funded it?'])

    def test_novelty_compares_only_with_the_earlier_papers_supplied(self):
        earlier = self.paper('2001.00002v1', 'Earlier clouds', abstract='We first measured a change in 30 objects.')
        earlier.published_arxiv = datetime.date(2020, 1, 2)
        earlier.save()
        PaperScore.objects.create(from_paper=self.paper_obj, to_paper=earlier, score=0.1)
        base = kit_answer(novelty={'assessment': 'A larger sample of the same change.', 'comparisons': [
            {'paper': '2001.00002v1', 'already': 'Measured it in 30 objects.', 'adds': 'Four times more objects.'},
            {'paper': '1999.99999v1', 'already': 'Invented.', 'adds': 'Invented.'}]})
        kit, llm = self.generate(answers(base))
        related = json.loads(base_call(llm).args[0])['source_documents']['related_work']
        self.assertEqual([r['id'] for r in related], ['2001.00002v1'])
        self.assertIn('30 objects', related[0]['abstract'])
        self.assertEqual([c['arxiv_id'] for c in kit['novelty']['comparisons']], ['2001.00002v1'])
        self.assertContains(self.paid.get(self.base), 'Four times more objects.')

    def test_outside_commentators_leave_out_coauthors_colleagues_and_middle_authors(self):
        close = self.paper('2101.00002v1', 'Other clouds', authors=('Cid Moe', 'Dee Fox'))
        PaperScore.objects.create(from_paper=self.paper_obj, to_paper=close, score=0.1)

        def authorship(aid, name, position, institution, corresponding=False):
            return {'author': {'id': 'https://openalex.org/%s' % aid, 'display_name': name, 'orcid': ''},
                    'author_position': position, 'is_corresponding': corresponding,
                    'institutions': [{'id': institution, 'display_name': institution.upper()}]}
        # merged with its publication: found by its arXiv DOI, it comes back with the journal's
        own = {'id': 'W1', 'doi': 'https://doi.org/10.1093/mnras/stad1', 'publication_date': '2021-01-04',
               'locations': [{'landing_page_url': 'https://arxiv.org/abs/2101.00001v2'}],
               'authorships': [authorship('A1', 'Ann Lee', 'first', 'i1'), authorship('A9', 'Bob Ray', 'last', 'i9')]}
        other = {'id': 'W2', 'doi': 'https://doi.org/10.48550/arxiv.2101.00002', 'publication_date': '2021-03-01',
                 'authorships': [authorship('A2', 'Cid Moe', 'first', 'i2'), authorship('A5', 'Eve Middle', 'middle', 'i5'),
                                 authorship('A4', 'Fay Same', 'middle', 'i1', corresponding=True),
                                 authorship('A3', 'Dee Fox', 'last', 'i3')]}
        coauthored = [{'id': 'W3', 'authorships': [authorship('A1', 'Ann Lee', 'first', 'i1'), authorship('A2', 'Cid Moe', 'last', 'i2')]}]
        calls = []

        def openalex(filters, select, per_page, sort='', search='', page=1):
            calls.append(filters)
            return [own, other] if filters.startswith('doi:') else coauthored
        with mock.patch('summarizer.presskit.openalex', side_effect=openalex):
            found = presskit.experts(self.paper_obj)
        self.assertTrue(found['checked'])
        self.assertEqual([(e['name'], e['affiliation'], e['roles']) for e in found['experts']], [('Dee Fox', 'I3', ['last'])])
        self.assertEqual(found['left_out'], 2)  # Cid Moe signs with Ann Lee, Fay Same works at her institution
        self.assertIn('10.48550/arxiv.2101.00002', calls[0])
        self.assertTrue(calls[1].startswith('author.id:https://openalex.org/A1|https://openalex.org/A9,from_publication_date:'))

    def test_press_figures_keep_supplied_labels_with_credit_and_license(self):
        gallery = [{'label': 'Figure 2:', 'images': ['https://arxiv.org/html/2101.00001v1/x2.png'], 'caption': 'Figure 2: the change.',
                    'source_url': 'https://arxiv.org/html/2101.00001v1#S2.F2'}]
        base = kit_answer(press_figures=[{'label': 'figure 2', 'why': 'Shows the change.'}, {'label': 'Figure 9', 'why': 'x'}])
        kit, _ = self.generate(answers(base), figures_of=gallery)
        self.assertEqual([f['label'] for f in kit['figures']], ['Figure 2'])
        self.assertEqual(kit['figures'][0]['credit'], 'Ann Lee et al., Figure 2, arXiv: 2101.00001v1, CC BY 4.0')
        page = self.paid.get(self.base)
        self.assertContains(page, 'https://arxiv.org/html/2101.00001v1/x2.png')
        self.assertContains(page, 'Reuse allowed with this credit.')

    def test_newer_version_and_full_text_offers(self):
        Generated.objects.create(kind='presskit', key='2101.00001v1', lang='en', status='done',
                                 content=json.dumps(dict(kit_answer(), schema_version=3, roles={},
                                                         source={'scope': 'abstract'}, abstract_only=True)))
        PaperVersion.objects.create(arxiv_id='2101.00001', version=3, checked=datetime.datetime(2026, 10, 1, tzinfo=datetime.timezone.utc))
        page = self.paid.get(self.base)
        self.assertEqual(page.context['newer']['arxiv_id'], '2101.00001v3')
        self.assertContains(page, 'version 3 exists')
        self.assertContains(page, '/en/changes/2101.00001v1/?from=1&amp;to=3')
        self.assertContains(page, 'Read the full text again')

    def test_share_link_is_each_readers_own_frozen_copy(self):
        self.generate()
        url = self.paid.post(self.base + 'share/', {'action': 'share'}).json()['url']
        self.assertIn('/en/press-kit/shared/', url)
        path = url.split('testserver')[1]
        page = Client().get(path)
        self.assertContains(page, 'A measured change')
        self.assertContains(page, 'read-only')
        self.assertNotContains(page, 'data-report=')
        self.assertNotContains(page, 'Write the press kit')
        self.assertEqual(self.client_for(self.user('free-sharer')).post(self.base + 'share/', {'action': 'share'}).status_code, 404)
        # another reader updates the kit and shares and stops their own link: this one is unchanged
        other = self.client_for(self.user('other-sharer', paid=True))
        kit = Generated.objects.get(kind='presskit')
        kit.content = json.dumps(dict(kit.data(), headline='A rewritten headline'))
        kit.save()
        other.post(self.base + 'share/', {'action': 'share'})
        other.post(self.base + 'share/', {'action': 'unshare'})
        self.assertContains(Client().get(path), 'A measured change')
        mine = self.paid.get(self.base)
        self.assertTrue(mine.context['share_outdated'])
        self.assertEqual(self.paid.post(self.base + 'share/', {'action': 'share'}).json()['url'], url)
        self.assertContains(Client().get(path), 'A measured change')  # sharing again keeps the copy
        self.assertEqual(self.paid.post(self.base + 'share/', {'action': 'update'}).json()['url'], url)
        self.assertContains(Client().get(path), 'A rewritten headline')
        self.assertFalse(self.paid.get(self.base).context['share_outdated'])
        self.assertEqual(self.paid.post(self.base + 'share/', {'action': 'delete'}).status_code, 400)
        self.paid.post(self.base + 'share/', {'action': 'unshare'})
        self.assertEqual(Client().get(path).status_code, 404)

    def test_an_author_is_recognised_by_an_orcid_they_follow(self):
        from .models import CitationAuthorWatch
        Generated.objects.create(kind='presskit', key='2101.00001v1', lang='en', status='done',
                                 content=json.dumps(dict(kit_answer(questions_for_authors=['Why?']), schema_version=3,
                                                         author_orcids=['0000-0002-1825-009X'])))
        self.assertEqual(self.paid.get(self.base).context['role'], 'commentator')
        CitationAuthorWatch.objects.create(user=self.reader, ident='orcid:0000-0002-1825-009x')
        self.assertEqual(self.paid.get(self.base).context['role'], 'author')
        self.assertEqual(self.paid.get(self.base + '?role=writer').context['role'], 'writer')

    def test_a_topic_search_completes_old_commentators_with_recent_ones(self):
        year = str(datetime.date.today().year)

        def authorship(aid, name, position):
            return {'author': {'id': 'https://openalex.org/%s' % aid, 'display_name': name}, 'author_position': position,
                    'institutions': []}
        own = {'id': 'W1', 'doi': 'https://doi.org/10.48550/arxiv.2101.00001', 'authorships': [authorship('A1', 'Ann Lee', 'first')]}
        close = self.paper('2101.00002v1', 'Old clouds', authors=('Old Lead', 'Old Last'))
        PaperScore.objects.create(from_paper=self.paper_obj, to_paper=close, score=0.1)
        old = {'id': 'W2', 'doi': 'https://doi.org/10.48550/arxiv.2101.00002', 'publication_date': '2015-01-01',
               'authorships': [authorship('A2', 'Old Lead', 'first'), authorship('A3', 'Old Last', 'last')]}
        recent = {'id': 'W3', 'doi': 'https://doi.org/10.1/new', 'display_name': 'New clouds on hot planets',
                  'publication_date': year + '-01-01', 'authorships': [authorship('A4', 'New Lead', 'first')]}
        off = {'id': 'W4', 'doi': 'https://doi.org/10.1/off', 'display_name': 'Bread recipes', 'publication_date': year + '-01-01',
               'authorships': [authorship('A5', 'Off Topic', 'first')]}
        searches = []

        def openalex(filters, select, per_page, sort='', search='', page=1):
            if search:
                searches.append(search)
                return [recent, off]
            return [own, old] if filters.startswith('doi:') else []
        vectors = __import__('numpy').array([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0]], dtype='float32')
        with mock.patch('summarizer.presskit.openalex', side_effect=openalex), \
             mock.patch('summarizer.similar.embed', return_value=vectors):
            found = presskit.experts(self.paper_obj)
        self.assertEqual(searches, ['Clouds on a hot planet'])
        self.assertEqual([e['name'] for e in found['experts']], ['New Lead', 'Old Last', 'Old Lead'])
        self.assertEqual((found['experts'][0]['arxiv_id'], found['experts'][0]['url']), ('', 'https://doi.org/10.1/new'))

    def test_markdown_export_follows_the_role(self):
        self.generate()
        text = self.paid.get(self.base + 'markdown/?role=author').content.decode()
        self.assertIn('## For the authors: draft press release', text)
        self.assertIn('[Your quote: why it matters]', text)
        self.assertNotIn('## Reporting angles', text)
        self.assertIn('> The association does not establish a causal mechanism for the change.\n> — Discussion · ¶1', text)
        everything = self.paid.get(self.base + 'markdown/').content.decode()
        self.assertIn('## Asked to comment: what this paper shows', everything)
        self.assertIn('## Questions to ask the authors', everything)
        self.assertEqual(Client().get(self.base + 'markdown/').status_code, 403)
        pdf = self.paid.get(self.base + 'pdf/?role=commentator')
        self.assertEqual(pdf.status_code, 200)

    def test_mistakes_are_reported_by_readers_only(self):
        self.generate()
        response = self.paid.post(self.base + 'report/', {'section': 'evidence', 'item': 'A 4 sigma change', 'text': 'Wrong sigma'})
        self.assertEqual(response.json()['status'], 'ok')
        report = PressKitReport.objects.get()
        self.assertEqual((report.section, report.item, report.user), ('evidence', 'A 4 sigma change', self.reader))
        self.assertEqual(self.paid.post(self.base + 'report/', {'text': ''}).status_code, 400)
        self.assertEqual(self.client_for(self.user('free-reporter')).post(self.base + 'report/', {'text': 'x'}).status_code, 403)
        self.assertEqual(Client().post(self.base + 'report/', {'text': 'x'}).status_code, 403)

    def test_a_reported_mistake_is_emailed_to_the_administrators(self):
        from django.contrib.auth.models import User
        from django.core import mail
        User.objects.create_superuser('boss', 'boss@example.com', 'x')
        self.generate()
        self.paid.post(self.base + 'report/', {'section': 'experts', 'item': 'Dee Fox', 'text': 'She is a co-author'})
        self.assertEqual(mail.outbox[-1].to, ['boss@example.com'])
        self.assertIn('She is a co-author', mail.outbox[-1].body)
        self.assertIn('/admin/summarizer/presskitreport/', mail.outbox[-1].body)

    def test_example_kit_is_open_to_visitors_but_never_generated_by_them(self):
        demo = self.paper('2307.01574v1', 'Gas in debris discs')
        Generated.objects.create(kind='presskit', key=demo.arxiv_id, lang='en', status='done',
                                 content=json.dumps(dict(kit_answer(questions_for_authors=['Why?']), schema_version=3)))
        page = Client().get('/en/press-kit/2307.01574v1/')
        self.assertContains(page, 'An example kit, open to everyone.')
        self.assertContains(page, 'A measured change')
        self.assertEqual(Client().get('/en/press-kit/2307.01574v1/markdown/').status_code, 200)
        self.assertEqual(Client().post('/en/press-kit/2307.01574v1/start/').status_code, 403)
        visitor = Client().get(self.base)
        self.assertContains(visitor, 'See an example kit')
        self.assertContains(Client().get('/en/press-kit/'), '/en/press-kit/2307.01574v1/')

    def test_older_kits_still_show_with_an_update_offer(self):
        Generated.objects.create(kind='presskit', key='2101.00001v1', lang='en', status='done',
                                 content=json.dumps({'headline': 'Old kit', 'questions_for_authors': ['Why?'], 'schema_version': 2,
                                                     'experts': [{'name': 'Old Expert', 'affiliation': '', 'arxiv_id': '2101.00002v1', 'title': 'T'}]}))
        page = self.paid.get(self.base + '?role=commentator')
        self.assertContains(page, 'Old kit')
        self.assertContains(page, 'Old Expert')
        self.assertContains(page, 'Update this kit: roles, earlier papers, figures')
        self.assertEqual(self.paid.get(self.base + 'markdown/').status_code, 200)
