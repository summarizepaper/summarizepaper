"""The map of who works on a topic: what it counts, whom it leaves out, the page, its exports and its monthly check."""
import datetime
import json
from collections import Counter
from unittest import mock

import numpy as np
from django.core import mail
from django.utils import timezone, translation

from summarizer import generated, teams
from summarizer.models import CitationAuthorWatch, Collection, Generated, TopicMapWatch, WatchedProject
from summarizer.test_tools import ToolsTestCase

TODAY = timezone.now().date().isoformat()


def inst(key, country='FR'):
    return {'id': 'https://openalex.org/I' + key, 'display_name': 'Institute ' + key, 'country_code': country}


def author(aid, name, position='middle', institutions=None, orcid=None, corresponding=False):
    return {'author': {'id': 'https://openalex.org/A' + aid, 'display_name': name, 'orcid': orcid},
            'author_position': position, 'is_corresponding': corresponding,
            'institutions': institutions if institutions is not None else [inst('1')]}


def work(key, title='Exomoon detection by astrometry', authors=None, date=TODAY, doi=None, kind='article', abstract=''):
    return {'id': 'https://openalex.org/W' + key, 'display_name': title, 'publication_date': date, 'type': kind,
            'doi': doi if doi is not None else 'https://doi.org/10.1/w' + key, 'cited_by_count': 1,
            'authorships': authors or [], 'abstract_inverted_index': {w: [i] for i, w in enumerate(abstract.split())}}


def flat(texts):
    vectors = np.ones((len(texts), 2), dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def two_ways(texts):
    """The topic, then papers whose title ends with an even number on one side and an odd one on the other."""
    vectors = np.array([[1, 1]] + [[1, 0.2] if t.split('\n')[0][-1] in '02468' else [0.2, 1] for t in texts[1:]], dtype=np.float32) \
        if texts and not texts[0].startswith('Exomoon paper') else np.array(
            [[1, 0.2] if t.split('\n')[0][-1] in '02468' else [0.2, 1] for t in texts], dtype=np.float32)
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def model(prompt, paid=True, max_tokens=0, system='', **kwargs):
    """The model calls of a map: the searches, a relevance screen keeping every paper, the approaches named."""
    data = json.loads(prompt)
    if 'literature search queries' in system:
        return {'title': 'Exomoons', 'queries': ['exomoon astrometry', 'exomoon transits']}
    if 'grades' in system:
        return {'grades': {p['id']: 1 for p in data['papers']}}
    if 'relevant_ids' in system:
        return {'relevant_ids': [p['id'] for p in data['papers']]}
    return {'approaches': [{'id': a['id'], 'name': 'Approach named %s' % a['id'], 'description': 'What it does.'}
                           for a in data['approaches']], 'landscape': 'Two ways to find moons.'}


class TeamsCountTests(ToolsTestCase):
    def find(self, found, **options):
        with mock.patch.object(teams, 'works', side_effect=lambda q, since, abstracts=False, counts=None: (
                counts.__setitem__(q, 1234) if counts is not None else None) or found), \
             mock.patch.object(teams.similar, 'embed', side_effect=flat), \
             mock.patch.object(generated, 'chat_json', side_effect=model) as chat:
            return teams.find('Exomoon astrometry', 'en', options=options), chat

    def test_no_minimum_quota_for_weak_embedding_matches(self):
        paper = work('1', 'Exocomet overview')
        with mock.patch.object(teams, 'works', return_value=[paper]), \
             mock.patch.object(teams.similar, 'embed', return_value=np.array([[1., 0.], [0., 1.]])), \
             mock.patch.object(generated, 'chat_json', return_value={'queries': ['moons']}) as chat:
            result = teams.find('Exomoon astrometry', 'en')
        self.assertEqual(result['papers'], 0)
        self.assertEqual(result['people'], [])
        self.assertEqual(chat.call_count, 1)  # no padded selection or invented summary
        self.assertEqual(result['version'], teams.RESULT_VERSION)

    def test_topical_screen_excludes_nearby_field_and_unknown_ids(self):
        papers = [work('1'), work('2', 'Exocomet overview')]
        with mock.patch.object(generated, 'chat_json', return_value={
                'relevant_ids': ['P1', 'P1', 'invented', ['P2'], papers[1]['id'], None]}) as chat:
            kept = teams.relevant_papers('Detecting exomoons', papers)
        self.assertEqual([w['id'] for w in kept], [papers[0]['id']])
        self.assertEqual([p['id'] for p in json.loads(chat.call_args.args[0])['papers']], ['P1', 'P2'])
        self.assertIn('json', chat.call_args.kwargs['system'].lower())
        with mock.patch.object(generated, 'chat_json', return_value={'relevant_ids': 'W1'}):
            with self.assertRaises(generated.GenerationError):
                teams.relevant_papers('Detecting exomoons', papers)

    def test_a_preprint_and_its_publication_count_once_and_abstracts_are_read(self):
        ann = author('1', 'Ann Lee', 'first')
        found = [work('1', 'Exomoons found by astrometry with Gaia', [ann, author('2', 'Bob Ray', 'last')],
                      abstract='We search moons'),
                 work('2', 'Exomoons found by astrometry with Gaia', [ann, author('2', 'Bob Ray', 'last')],
                      doi='https://doi.org/10.48550/arXiv.2501.00001', kind='preprint')]
        result, chat = self.find(found)
        self.assertEqual(result['papers'], 1)
        self.assertEqual(result['duplicates'], 1)
        self.assertEqual(result['people'][0]['papers'], 1)
        self.assertEqual(result['queries'], ['exomoon astrometry', 'exomoon transits'])
        self.assertEqual(result['hits'], {'exomoon astrometry': 1234, 'exomoon transits': 1234})
        relevance = [c for c in chat.call_args_list if 'grades' in c.kwargs['system']][0]
        self.assertEqual(json.loads(relevance.args[0])['papers'][0]['abstract'], 'We search moons')

    def test_a_repository_copy_without_author_records_is_the_same_paper(self):
        from summarizer import student_papers
        article = work('1', 'The CHEOPS view of HD 95338b: a search for exomoons', [author('1', 'Szilárd Kálmán', 'first')])
        copy = work('2', 'The CHEOPS view of HD 95338b: a search for exomoons', [author('1', 'Kálmán, Sz.', 'first')],
                    doi='https://doi.org/10.17169/refubium-50092')
        copy['authorships'][0]['author']['id'] = None
        self.assertEqual(len(student_papers.deduplicate([article, copy])), 1)
        other = work('3', 'The CHEOPS view of HD 95338b: a search for exomoons', [author('7', 'Ann Lee', 'first')],
                     doi='https://doi.org/10.17169/refubium-1')
        other['authorships'][0]['author']['id'] = None
        self.assertEqual(len(student_papers.deduplicate([article, other])), 2)

    def test_a_map_grades_papers_and_keeps_those_on_one_aspect_of_the_topic(self):
        papers = [work('1'), work('2', 'Moons by timing'), work('3', 'Comets'), work('4', 'Rings')]
        with mock.patch.object(generated, 'chat_json', return_value={'grades': {'P1': 2, 'P2': 1, 'P3': 0, 'P4': True}}) as chat:
            kept = teams.relevant_papers('Exomoons by astrometry and timing', papers, abstracts=True, parts=True)
        self.assertEqual([w['display_name'] for w in kept], ['Exomoon detection by astrometry', 'Moons by timing'])
        self.assertIn('need not cover every aspect', chat.call_args.kwargs['system'])
        with mock.patch.object(generated, 'chat_json', return_value={'grades': ['P1']}):
            with self.assertRaises(generated.GenerationError):
                teams.relevant_papers('Exomoons', papers, parts=True)
        with mock.patch.object(generated, 'chat_json', return_value={'relevant_ids': []}) as chat:
            teams.relevant_papers('Exomoons by astrometry and timing', papers, abstracts=True)
        self.assertNotIn('need not cover every aspect', chat.call_args.kwargs['system'])  # placements stay strict

    def test_institutions_count_their_share_and_large_collaborations_do_not_crown_their_members(self):
        small = [work(str(i), 'Exomoon search number %d' % i, authors=[author('1', 'Ann Lee', 'first', [inst('P')]), author('2', 'Bob Ray', 'last', [inst('P')])])
                 for i in (1, 2)]
        big = [work(str(i), authors=[author('9', 'Zed Lead', 'first', [inst('L')])] +
                    [author(str(100 + k), 'Member %02d' % k, 'middle', [inst('C%d' % k)]) for k in range(30)] +
                    [author('8', 'Yan Last', 'last', [inst('L')])]) for i in (3, 4, 5)]
        result, _ = self.find(small + big)
        institutions = [g['institution'] for g in result['institutions']]
        self.assertEqual(institutions[:2], ['Institute L', 'Institute P'])  # 3 papers each for C0..C29, a tiny share and no lead
        member = next(g for g in teams.institutions_of(small + big, {}) if g['institution'] == 'Institute C0')
        self.assertEqual((member['papers'], member['lead']), (3, 0))
        self.assertLess(member['share'], 0.2)
        names = [p['name'] for p in result['people']]
        self.assertIn('Zed Lead', names)
        self.assertIn('Member 00', names)  # among the first fifteen authors
        self.assertNotIn('Member 20', names)  # beyond them, a middle author of a collaboration is not counted
        ann = next(p for p in result['people'] if p['name'] == 'Ann Lee')
        self.assertEqual((ann['papers'], ann['lead'], ann['roles']['first']), (2, 2, 2))

    def test_one_person_per_orcid_and_no_roles_in_alphabetical_lists(self):
        orcid = 'https://orcid.org/0000-0002-1825-0097'
        found = [work('1', 'Moons by astrometry', authors=[author('1', 'Ann Lee', 'first', orcid=orcid), author('2', 'Bob Ray', 'last')]),
                 work('2', 'Moons by transit timing', authors=[author('3', 'A. Lee', 'last', orcid=orcid), author('2', 'Bob Ray', 'first')]),
                 work('3', authors=[author(str(k), name, pos) for k, name, pos in
                                    ((4, 'Al Abel', 'first'), (5, 'Bo Baker', 'middle'), (6, 'Cy Cole', 'middle'), (7, 'Di Dunn', 'last'))])]
        people, _ = teams.people_of(found, {})
        lee = [p for p in people if p['orcid'] == orcid]
        self.assertEqual(len(lee), 1)
        self.assertEqual((lee[0]['papers'], lee[0]['roles']['first'], lee[0]['roles']['last']), (2, 1, 1))
        self.assertEqual(len(lee[0]['aliases']), 2)
        abel = next(p for p in people if p['name'] == 'Al Abel')
        self.assertEqual(abel['lead'], 0)  # Abel, Baker, Cole, Dunn: their order says nothing

    def test_modes_leave_out_the_user_and_their_coauthors(self):
        found = [work('1', authors=[author('1', 'Me Myself', 'first', [inst('H')]), author('2', 'Co Author', 'last', [inst('X')])]),
                 work('2', authors=[author('3', 'Colle Ague', 'first', [inst('H')]), author('4', 'Far Away', 'last', [inst('Y')])]),
                 work('3', authors=[author('5', 'Only Middle', 'middle', [inst('Z')]), author('6', 'Lone Wolf', 'first', [inst('Z')]),
                                    author('7', 'Last One', 'last', [inst('Z')])])]
        relations = {'authors': {'https://openalex.org/A1': 'self', 'https://openalex.org/A2': 'coauthor'},
                     'institutions': {'https://openalex.org/IH'}, 'papers': 4}
        people, excluded = teams.people_of(found, {}, relations, 'overview')
        self.assertEqual({p['name']: p['relation'] for p in people if p['relation']},
                         {'Me Myself': 'self', 'Co Author': 'coauthor', 'Colle Ague': 'colleague'})
        self.assertEqual(excluded, [])
        people, excluded = teams.people_of(found, {}, relations, 'referees')
        self.assertEqual(sorted(p['name'] for p in people), ['Far Away', 'Last One', 'Lone Wolf'])  # no middle author
        self.assertEqual(sorted(p['name'] for p in excluded), ['Co Author', 'Colle Ague', 'Me Myself'])
        people, _ = teams.people_of(found, {}, relations, 'collaborators')
        self.assertIn('Colle Ague', [p['name'] for p in people])
        self.assertNotIn('Co Author', [p['name'] for p in people])

    def test_conflicts_read_the_coauthors_of_recent_small_papers(self):
        orcid = '0000-0002-1825-0097'
        mine = author('1', 'Me', 'first', [inst('H')], orcid='https://orcid.org/' + orcid)
        body = {'results': [{'id': 'W1', 'publication_date': TODAY, 'authorships': [mine, author('2', 'Co', 'last')]},
                            {'id': 'W2', 'publication_date': TODAY, 'authorships': [mine] + [author(str(10 + k), 'M%d' % k) for k in range(25)]}]}
        with mock.patch.object(teams.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(teams.sources, '_json', return_value=body) as fetch:
            found = teams.conflicts([orcid])
        self.assertIn('author.orcid:%s' % orcid, fetch.call_args.args[1]['filter'])
        self.assertEqual(found['authors'], {'https://openalex.org/A1': 'self', 'https://openalex.org/A2': 'coauthor'})
        self.assertEqual(found['institutions'], {'https://openalex.org/IH'})
        with mock.patch.object(teams.sources, 'tool_search_allowed', return_value=False):
            self.assertIsNone(teams.conflicts([orcid]))

    def test_a_national_umbrella_does_not_make_everyone_a_colleague(self):
        orcid = '0000-0002-1825-0097'
        cnrs = dict(inst('CNRS'), display_name='Centre National de la Recherche Scientifique', type='government')
        mine = author('1', 'Me', 'first', [cnrs, inst('LAB')], orcid='https://orcid.org/' + orcid)
        with mock.patch.object(teams.sources, 'tool_search_allowed', return_value=True), \
             mock.patch.object(teams.sources, '_json', return_value={'results': [{'id': 'W1', 'publication_date': TODAY, 'authorships': [mine]}]}):
            self.assertEqual(teams.conflicts([orcid])['institutions'], {'https://openalex.org/ILAB'})

    def test_an_approach_of_one_paper_joins_the_closest(self):
        vectors = np.array([[1, 0], [1, 0.1], [1, 0.2], [0, 1], [0.1, 1], [0.2, 1], [0.7, 0.75]], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        centres = np.array([[1, 0.1], [0.1, 1], [0.7, 0.75]], dtype=np.float32)
        centres /= np.linalg.norm(centres, axis=1, keepdims=True)
        labels, kept = teams.merge_small(vectors, [0, 0, 0, 1, 1, 1, 2], centres)
        self.assertEqual(len(kept), 2)
        self.assertEqual(sorted(Counter(labels).values()), [3, 4])

    def test_options_are_checked(self):
        options = teams.clean_options({'mode': 'boss', 'years': '7', 'queries': [' a  b ', '', 'x' * 500, 'c', 'd', 'e'],
                                       'orcids': ['https://orcid.org/0000-0002-1825-009x', 'nope']})
        self.assertEqual(options['mode'], 'overview')
        self.assertEqual(options['years'], 2)
        self.assertEqual(options['queries'], ['a b', 'x' * teams.QUERY_CHARS, 'c', 'd'])
        self.assertEqual(options['orcids'], ['0000-0002-1825-009X'])
        self.assertEqual(teams.clean_options({'years': 5, 'mode': 'referees'})['years'], 5)

    def test_approaches_group_papers_and_are_named_from_them(self):
        vectors = np.array([[1, 0.05 * i] for i in range(6)] + [[0.05 * i, 1] for i in range(6)], dtype=np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        labels, _ = teams.clusters(vectors, 2)
        self.assertEqual(len(set(labels[:6])), 1)
        self.assertEqual(len(set(labels[6:])), 1)
        self.assertNotEqual(labels[0], labels[6])
        self.assertEqual(teams.clusters(vectors, 2)[0], labels)  # the same map gives the same groups
        found = [work(str(i), authors=[author(str(i), 'Person %d' % i, 'first')]) for i in range(12)]
        with mock.patch.object(teams.similar, 'embed', return_value=vectors):
            labels, members = teams.approaches(found)
        self.assertEqual(sorted(len(m) for m in members.values()), [6, 6])
        with mock.patch.object(generated, 'chat_json', side_effect=model) as chat:
            named, landscape = teams.describe('Exomoons', found, members, 'en')
        self.assertEqual([a['name'] for a in named], ['Approach named 0', 'Approach named 1'])
        self.assertEqual(landscape, 'Two ways to find moons.')
        self.assertIn('Never name or rank institutions or people', chat.call_args.kwargs['system'])


class TeamsPageTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.owner = self.user('pro', paid=True, tier='premium')
        self.client = self.client_for(self.owner)
        self.orcid = '0000-0002-1825-0097'

    def map_data(self, mode='overview'):
        found = [work(str(i), 'Exomoon paper %d' % i, [author('1', 'Ann Lee', 'first', [inst('P')]),
                                                       author('2', '<b>Bob</b> Ray', 'last', [inst('L', 'GB')])],
                      doi='https://doi.org/10.48550/arXiv.2501.0000%d' % i if i < 5 else None) for i in range(10)]
        relations = {'authors': {'https://openalex.org/A2': 'coauthor'}, 'institutions': set(), 'papers': 3}
        with mock.patch.object(teams, 'works', return_value=found), \
             mock.patch.object(teams.similar, 'embed', side_effect=two_ways), \
             mock.patch.object(teams, 'conflicts', return_value=relations), \
             mock.patch.object(generated, 'chat_json', side_effect=model):
            return teams.find('Exomoon astrometry', 'en', options={'mode': mode, 'orcids': [self.orcid]})

    def row(self, data, key='map'):
        return Generated.objects.create(kind='teams', key=key, user=self.owner, lang='en', status='done', content=json.dumps(data))

    def test_the_form_starts_a_map_with_its_options(self):
        CitationAuthorWatch.objects.create(user=self.owner, ident='orcid:' + self.orcid)
        self.assertContains(self.client.get('/en/teams/'), 'value="%s"' % self.orcid)
        self.assertContains(self.client.post('/en/teams/', {'topic': 'Exomoons by astrometry', 'orcid': 'nope'}), 'not an ORCID iD')
        with mock.patch.object(teams, 'find', return_value={'title': 'Moons', 'papers': 0, 'version': teams.RESULT_VERSION}) as find:
            r = self.client.post('/en/teams/', {'topic': 'Exomoons by astrometry', 'mode': 'referees', 'years': '5', 'orcid': self.orcid})
        self.assertEqual(find.call_args.args[3], {'mode': 'referees', 'years': '5', 'orcids': [self.orcid]})
        self.assertContains(self.client.get(r['Location']), 'no recent paper')

    def test_the_digest_profile_can_be_the_topic(self):
        from summarizer.models import DigestSubscription
        DigestSubscription.objects.create(user=self.owner, categories='astro-ph.EP', profile='Exomoons seen by Gaia astrometry.')
        with mock.patch.object(teams, 'find', return_value={'title': 'Moons', 'papers': 0}) as find:
            self.client.post('/en/teams/', {'topic': '', 'use_profile': '1'})
        self.assertEqual(find.call_args.args[0], 'Exomoons seen by Gaia astrometry.')

    def test_the_map_shows_people_institutions_approaches_and_who_was_marked(self):
        data = self.map_data()
        self.row(data)
        page = self.client.get('/en/teams/map/')
        self.assertContains(page, 'Ann Lee')
        self.assertContains(page, '&lt;b&gt;Bob&lt;/b&gt; Ray')
        self.assertNotContains(page, '<b>Bob</b>')
        self.assertContains(page, 'co-author')
        self.assertContains(page, 'Institute P')
        self.assertContains(page, 'Approach named 0')
        self.assertContains(page, 'exomoon astrometry')
        self.assertContains(page, 'first author 10, last 0, corresponding 0')
        self.assertContains(page, '/en/paper/2501.00001/')  # the summary of a paper of the map
        self.assertContains(page, 'Follow this map')
        with translation.override('fr'):
            self.assertContains(self.client.get('/fr/teams/map/'), 'Ann Lee')

    def test_a_shared_map_never_shows_whom_the_orcid_marked(self):
        row = self.row(self.map_data('collaborators'))
        self.client.post('/en/teams/map/', {'action': 'share'})
        row.refresh_from_db()
        page = self.client_for(self.user('other')).get('/en/teams/shared/%s/' % row.share_token)
        self.assertContains(page, 'Ann Lee')
        self.assertNotContains(page, 'Bob')  # left out as a co-author, and not listed as such
        self.assertNotContains(page, 'left out')
        people = self.client.get('/en/teams/shared/%s/export/people/' % row.share_token).content.decode()
        self.assertIn('Ann Lee', people)
        self.assertNotIn('Bob', people)

    def test_exports(self):
        self.row(self.map_data())
        people = self.client.get('/en/teams/map/export/people/')
        self.assertEqual(people['Content-Type'], 'text/csv; charset=utf-8')
        self.assertIn('Ann Lee', people.content.decode())
        self.assertIn('coauthor', people.content.decode())
        self.assertIn('Institute P', self.client.get('/en/teams/map/export/institutions/').content.decode())
        self.assertIn('Exomoon paper 9', self.client.get('/en/teams/map/export/papers/').content.decode())
        bib = self.client.get('/en/teams/map/export/bib/').content.decode()
        self.assertIn('eprint={2501.00001}', bib)
        self.assertIn('RIS', self.client.get('/en/teams/map/').content.decode())
        self.assertEqual(self.client.get('/en/teams/map/export/other/').status_code, 404)
        self.assertEqual(self.client_for(self.user('x', paid=True)).get('/en/teams/map/export/people/').status_code, 404)

    def test_run_again_with_edited_searches(self):
        self.row(self.map_data())
        with mock.patch.object(teams, 'find', return_value={'title': 'Moons', 'papers': 0}) as find:
            r = self.client.post('/en/teams/map/', {'action': 'rerun', 'queries': 'moons astrometry\n\nmoons timing\n',
                                                    'mode': 'competitors', 'years': '5', 'orcid': ''})
        options = find.call_args.args[3]
        self.assertEqual(options['queries'], ['moons astrometry', '', 'moons timing'])  # cleaned by find
        self.assertEqual((options['mode'], options['title']), ('competitors', 'Exomoons'))
        self.assertNotEqual(r['Location'], '/en/teams/map/')
        with mock.patch.object(teams, 'PER_DAY', 0), mock.patch.object(teams, 'find') as find:
            self.assertContains(self.client.post('/en/teams/map/', {'action': 'rerun', 'queries': 'x'}, follow=True), 'searches a day')
        find.assert_not_called()

    def test_papers_go_to_a_collection(self):
        self.row(self.map_data())
        with mock.patch('summarizer.library.add_many', return_value={'added': 5}) as add:
            r = self.client.post('/en/teams/map/', {'action': 'collection', 'collection': 'new', 'approach': ''})
        collection = Collection.objects.get(user=self.owner)
        self.assertEqual(r['Location'], '/en/collections/%d/' % collection.pk)
        self.assertTrue({'2501.0000%d' % i for i in range(5)} <= set(add.call_args.args[1]))

    def test_competing_teams_go_to_the_scoop_watch(self):
        WatchedProject.objects.create(user=self.owner, title='Moons', description='x' * 300)
        self.row(self.map_data())
        self.assertContains(self.client.get('/en/teams/map/'), 'data-competitor="Ann Lee"')

    def test_legacy_results_hide_inferred_claims(self):
        data = {'title': 'Exomoons', 'papers': 1, 'since': 2024, 'landscape': 'UNSUPPORTED LANDSCAPE', 'version': 2,
                'groups': [{'institution': 'CNRS', 'papers': 1, 'citations': 0, 'focus': 'UNSUPPORTED FOCUS',
                            'people': [{'name': 'UNSUPPORTED LEADER', 'papers': 1}], 'titles': [{'title': 'Exomoon study'}]}]}
        self.row(data, 'old-map')
        page = self.client.get('/en/teams/old-map/')
        self.assertContains(page, 'earlier version')
        self.assertContains(page, 'Exomoon study')
        self.assertContains(page, 'Create a link')
        self.assertNotContains(page, 'UNSUPPORTED')


class TeamsWatchTests(ToolsTestCase):
    def test_a_followed_map_emails_the_new_papers_and_the_new_names_every_month(self):
        owner = self.user('pro', paid=True, tier='premium')
        known = author('1', 'Ann Lee', 'first')
        data = {'version': teams.RESULT_VERSION, 'title': 'Exomoons', 'topic': 'Exomoon astrometry', 'queries': ['moons'],
                'works': [{'id': 'https://openalex.org/W1'}], 'people': [{'key': 'https://openalex.org/A1', 'aliases': ['https://openalex.org/A1']}]}
        row = Generated.objects.create(kind='teams', key='map', user=owner, lang='en', status='done', content=json.dumps(data))
        client = self.client_for(owner)
        client.post('/en/teams/map/', {'action': 'follow'})
        watch = TopicMapWatch.objects.get(user=owner, row=row)
        old, first = work('1'), work('2', 'Exomoons again')
        now = timezone.now()
        with mock.patch.object(teams, 'works', return_value=[old, first]) as search:
            self.assertEqual(teams.watch_maps(now, 10), 0)  # the first check learns what exists
        self.assertEqual(search.call_args.args[0], 'moons')
        watch.refresh_from_db()
        self.assertEqual(watch.checked, now.date())
        self.assertEqual(teams.watch_maps(now, 10), 0)  # not due before a month
        new = work('3', 'A new exomoon', [known, author('9', 'New Name', 'last')])
        later = now + datetime.timedelta(days=teams.WATCH_EVERY)
        with mock.patch.object(teams, 'works', return_value=[old, first, new]), \
             mock.patch.object(teams.similar, 'embed', side_effect=flat), \
             mock.patch.object(generated, 'chat_json', side_effect=model):
            self.assertEqual(teams.watch_maps(later, 10), 1)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('A new exomoon', mail.outbox[0].body)
        self.assertIn('On the map: Ann Lee', mail.outbox[0].body)
        self.assertIn('New names: New Name', mail.outbox[0].body)
        self.assertIn('/en/teams/map/', mail.outbox[0].body)
        client.post('/en/teams/map/', {'action': 'unfollow'})
        self.assertFalse(TopicMapWatch.objects.filter(user=owner).exists())

    def test_at_most_five_maps_are_followed(self):
        owner = self.user('pro', paid=True, tier='premium')
        rows = [Generated.objects.create(kind='teams', key='m%d' % i, user=owner, lang='en', status='done',
                                         content=json.dumps({'version': teams.RESULT_VERSION})) for i in range(6)]
        self.assertEqual([teams.follow(owner, r) for r in rows], ['watching'] * 5 + ['watch_too_many'])
