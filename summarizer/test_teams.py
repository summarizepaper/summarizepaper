"""Publication evidence and conservative author attribution in the topic map."""
import json
from unittest import mock

import numpy as np
from django.utils import translation

from summarizer import generated, teams
from summarizer.models import Generated
from summarizer.test_tools import ToolsTestCase


class TeamsEvidenceTests(ToolsTestCase):
    def work(self, key, title='Exomoon detection', authors=None):
        return {'id': key, 'display_name': title, 'publication_date': '2026-01-01',
                'doi': 'https://doi.org/10.1/example', 'authorships': authors or []}

    def author(self, key, name, position='middle', institutions=None):
        return {'author': {'id': key, 'display_name': name}, 'author_position': position,
                'institutions': institutions or [{'id': 'I1', 'display_name': 'CNRS'}]}

    def test_no_minimum_quota_for_weak_embedding_matches(self):
        paper = self.work('W1', 'Exocomet overview')
        with mock.patch.object(teams, 'works', return_value=[paper]), \
             mock.patch.object(teams.similar, 'embed', return_value=np.array([[1., 0.], [0., 1.]])), \
             mock.patch.object(generated, 'chat_json', return_value={'queries': ['moons']}) as chat:
            result = teams.find('Exomoon astrometry', 'en')
        self.assertEqual(result['papers'], 0)
        self.assertEqual(result['groups'], [])
        self.assertEqual(chat.call_count, 1)  # no padded selection or invented summary

    def test_topical_screen_excludes_nearby_field_and_unknown_ids(self):
        papers = [self.work('W1'), self.work('W2', 'Exocomet overview')]
        with mock.patch.object(generated, 'chat_json', return_value={
                'relevant_ids': ['W1', 'W1', 'invented', ['W2'], None]}) as chat:
            kept = teams.relevant_papers('Detecting exomoons', papers)
        self.assertEqual([w['id'] for w in kept], ['W1'])
        self.assertEqual(json.loads(chat.call_args.args[0])['topic'], 'Detecting exomoons')
        self.assertIn('json', chat.call_args.kwargs['system'].lower())
        with mock.patch.object(generated, 'chat_json', return_value={'relevant_ids': 'W1'}):
            with self.assertRaises(generated.GenerationError):
                teams.relevant_papers('Detecting exomoons', papers)

    def test_secondary_affiliation_identity_and_supporting_papers(self):
        cnrs = {'id': 'CNRS', 'display_name': 'CNRS', 'country_code': 'FR'}
        lab = {'id': 'LAB', 'display_name': 'Local laboratory', 'country_code': 'FR'}
        papers = [self.work('W1', authors=[
            self.author('A1', 'Same Name', 'last', [cnrs, lab, cnrs]),
            self.author('A2', 'Same Name', 'middle', [cnrs])]),
            self.work('W2', authors=[self.author('A1', 'Name, Same', 'first', [lab, cnrs])])]
        groups = {g['id']: g for g in teams.group(papers)}
        self.assertEqual(groups['CNRS']['papers'], 2)
        people = groups['CNRS']['people']
        self.assertEqual(len(people), 2)  # author IDs keep homonyms distinct
        self.assertEqual([p['papers'] for p in people], [2, 1])
        self.assertEqual([t['title'] for t in people[0]['titles']], ['Exomoon detection'] * 2)
        self.assertNotIn('last', people[0])
        self.assertEqual(groups['LAB']['people'][0]['papers'], 2)

    def test_last_author_does_not_win_a_tie_and_large_collaborations_keep_affiliations(self):
        authors = [self.author('A%d' % i, 'Zed %02d' % i, 'last' if i == 0 else 'middle') for i in range(30)]
        authors.append(self.author('A99', 'Alice Example', 'middle'))
        group = teams.group([self.work('W1', authors=authors)])[0]
        self.assertEqual(group['people'][0]['name'], 'Alice Example')
        self.assertEqual(group['papers'], 1)

    def test_bounded_candidates_and_all_selected_titles_in_summary(self):
        authors = [self.author('A1', 'Alice')]
        papers = [self.work('W%d' % i, title='Exomoon %d' % i, authors=authors) for i in range(125)]
        with mock.patch.object(teams, 'works', return_value=papers), \
             mock.patch.object(teams.similar, 'embed', return_value=np.ones((126, 1))), \
             mock.patch.object(teams, 'relevant_papers', side_effect=lambda topic, candidates: candidates) as relevant, \
             mock.patch.object(generated, 'chat_json', side_effect=[{'queries': ['moons']}, {'groups': []}]) as chat:
            result = teams.find('Exomoons', 'en')
        self.assertTrue(result['truncated'])
        self.assertEqual(len(relevant.call_args.args[1]), 120)
        self.assertEqual(len(result['groups'][0]['evidence']), 120)
        self.assertEqual(len(result['groups'][0]['titles']), teams.TITLES)
        # Summary evidence includes more than just the six most cited papers.
        self.assertEqual(chat.call_args.args[0].count('Exomoon '), 120)

    def test_legacy_results_hide_inferred_claims_and_current_results_expose_evidence(self):
        client = self.client_for(self.user('pro', paid=True))
        data = {'title': 'Exomoons', 'papers': 1, 'since': 2024, 'landscape': 'UNSUPPORTED LANDSCAPE',
                'groups': [{'institution': 'CNRS', 'papers': 1, 'citations': 0,
                            'focus': 'UNSUPPORTED FOCUS', 'people': [{'name': 'UNSUPPORTED LEADER', 'papers': 1}],
                            'titles': [{'title': 'Exomoon study'}]}]}
        row = Generated.objects.create(kind='teams', key='old-map', user_id=client.session.get('_auth_user_id'),
                                       lang='en', status='done', content=json.dumps(data))
        page = client.get('/en/teams/old-map/')
        self.assertContains(page, 'previous grouping method')
        self.assertContains(page, 'Exomoon study')
        self.assertNotContains(page, 'UNSUPPORTED')
        data['version'] = teams.RESULT_VERSION
        data['landscape'] = 'Selected studies use astrometry.'
        data['groups'] = teams.group([self.work('W1', authors=[self.author('A1', '<b>Alice</b>')])])
        row.content = json.dumps(data)
        row.save()
        with translation.override('fr'):
            page = client.get('/fr/teams/old-map/')
        self.assertContains(page, 'Auteur')
        self.assertContains(page, '&lt;b&gt;Alice&lt;/b&gt;')
        self.assertNotContains(page, '<b>Alice</b>')
        self.assertContains(page, 'Exomoon detection', count=2)  # person evidence and institution papers
