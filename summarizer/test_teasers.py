import json
import uuid

from django.contrib.auth.models import AnonymousUser
from django.test import Client
from django.utils import timezone

from summarizer import teasers
from summarizer.models import Generated
from summarizer.test_tools import ToolsTestCase


def missed_result(n=5):
    return {'version': 2, 'title': 'Clouds on hot planets', 'searched_on': '2026-11-10',
            'items': [{'key': '2101.0000%dv1' % i, 'title': 'Paper number %d' % i, 'authors': 'Lee et al.', 'year': 2025,
                       'importance': 'high' if i < 2 else 'low', 'relation': 'Same question, other telescope %d.' % i}
                      for i in range(n)]}


class TeaserTests(ToolsTestCase):
    def row(self, user, kind='missed', status='done', content=None, lang='en', trial=True, **kwargs):
        return Generated.objects.create(kind=kind, key=uuid.uuid4().hex, lang=lang, user=user, status=status, trial=trial,
                                        content=json.dumps(content if content is not None else missed_result()), **kwargs)

    def test_a_free_account_tries_each_tool_once_and_a_failed_try_is_not_spent(self):
        ann = self.user()
        self.assertTrue(teasers.trial_open(ann, 'missed'))
        self.assertEqual(teasers.audience(ann, 'missed'), 'trial')
        self.row(ann, status='failed')
        self.assertTrue(teasers.trial_open(ann, 'missed'))
        self.row(ann)
        self.assertFalse(teasers.trial_open(ann, 'missed'))
        self.assertEqual(teasers.audience(ann, 'missed'), 'premium')
        self.assertTrue(teasers.trial_open(ann, 'refcheck'))  # each tool has its own
        self.assertFalse(teasers.trial_open(ann, 'literature'))  # Premium+, too costly to give away
        self.assertEqual(teasers.audience(ann, 'literature'), 'plus')
        self.assertFalse(teasers.trial_open(AnonymousUser(), 'missed'))
        self.assertEqual(teasers.audience(AnonymousUser(), 'missed'), 'signup')
        bob = self.user('bob', paid=True)
        self.assertFalse(teasers.trial_open(bob, 'missed'))
        self.assertEqual(teasers.audience(bob, 'missed'), 'open')

    def test_a_try_shows_its_first_results_and_only_counts_the_others(self):
        ann = self.user()
        row = self.row(ann)
        page = self.client_for(ann).get('/en/missed/%s/' % row.key)
        self.assertEqual(page.status_code, 200)
        html = page.content.decode()
        self.assertIn('Paper number 0', html)
        self.assertIn('Paper number 1', html)
        self.assertNotIn('Paper number 2', html)  # greyed rows carry nothing, not even hidden
        self.assertNotIn('other telescope 4', html)
        self.assertIn('3 more results hidden', html)
        self.assertIn('/en/premium/', html)
        # nor does the status API give the whole result away
        self.assertEqual(self.client_for(ann).get('/en/ai/%d/' % row.pk).status_code, 404)
        # its actions (share, mark...) stay closed: the page is the teaser again
        self.client_for(ann).post('/en/missed/%s/' % row.key, {'action': 'share'})
        row.refresh_from_db()
        self.assertEqual(row.share_token, '')
        # a result of a lapsed subscription stays closed, teaser or not
        lapsed = self.row(ann, trial=False)
        self.assertEqual(self.client_for(ann).get('/en/missed/%s/' % lapsed.key).status_code, 404)
        # someone else's result stays out of reach
        self.assertEqual(self.client_for(self.user('eve')).get('/en/missed/%s/' % row.key).status_code, 404)

    def test_a_subscriber_still_sees_the_whole_result(self):
        bob = self.user('bob', paid=True)
        row = self.row(bob)
        html = self.client_for(bob).get('/en/missed/%s/' % row.key).content.decode()
        self.assertIn('Paper number 4', html)
        self.assertIsNotNone(self.client_for(bob).get('/en/ai/%d/' % row.pk).json()['data'])

    def test_the_tool_page_shows_a_staff_example_never_someone_elses_search(self):
        ann = self.user()
        self.row(ann, showcase=True, content=missed_result())  # not staff: never published
        anonymous = Client().get('/en/missed/').content.decode()
        self.assertNotIn('Paper number 0', anonymous)
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        self.row(staff, showcase=True, content=missed_result(4))
        anonymous = Client().get('/en/missed/').content.decode()
        self.assertIn('A real example', anonymous)
        self.assertIn('Paper number 1', anonymous)
        self.assertNotIn('Paper number 2', anonymous)
        self.assertIn('2 more results hidden', anonymous)
        self.assertIn('/en/register/', anonymous)
        # a free account that has not tried it gets the form, under the example
        page = self.client_for(self.user('cid')).get('/en/missed/').content.decode()
        self.assertIn('id="tool-form"', page)
        self.assertIn('Your first search with this tool is free', page)

    def test_the_example_of_the_page_language_comes_first(self):
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        self.row(staff, showcase=True, lang='en', content=missed_result())
        french = self.row(staff, showcase=True, lang='fr', content=missed_result())
        self.assertEqual(teasers.showcase('missed', 'fr'), french)
        self.assertEqual(teasers.showcase('missed', 'de').kind, 'missed')

    def test_an_example_from_an_older_version_never_breaks_the_page(self):
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        self.row(staff, kind='teams', showcase=True, content={'version': 1})
        self.assertIsNone(teasers.teaser(teasers.showcase('teams', 'en')))
        self.assertEqual(Client().get('/en/teams/').status_code, 200)

    def test_a_free_try_runs_the_tool_once(self):
        from unittest import mock
        ann = self.user()
        def start_like(request, topic, options):  # creates its row, as start() does
            Generated.objects.create(kind='teams', key='first', lang='en', user=ann, status='pending')
            from django.http import HttpResponseRedirect
            return HttpResponseRedirect('/en/teams/first/')
        with mock.patch('summarizer.teams.start', side_effect=start_like) as start:
            self.client_for(ann).post('/en/teams/', {'topic': 'Detection of exomoons by astrometry', 'mode': 'overview'})
            self.assertEqual(start.call_count, 1)
        self.assertTrue(Generated.objects.get(key='first').trial)
        with mock.patch('summarizer.teams.start') as start:
            self.client_for(ann).post('/en/teams/', {'topic': 'Detection of exomoons by astrometry', 'mode': 'overview'})
            start.assert_not_called()

    def test_people_around_a_paper(self):
        data = {'items': [{'authors': 'Ann Lee, Bob Ray, Cid Roe et al.'}, {'authors': 'Ann Lee, Dan Poe'},
                          {'authors': 'Bob Ray, Ann Lee'}, {'authors': 'Eve Moe'}]}
        found = teasers.people_around(data)
        self.assertEqual(found['items'], [{'name': 'Ann Lee', 'papers': 3}])
        self.assertEqual(found['hidden'], 1)  # Bob Ray, twice; the others once are not counted
        self.assertEqual(found['papers'], 4)
        self.assertIsNone(teasers.people_around({'items': [{'authors': 'Eve Moe'}]}))
        self.assertIsNone(teasers.people_around(None))

    def test_the_paper_page_names_who_signs_the_papers_around_it(self):
        from summarizer import neighbours
        self.paper()
        items = [dict(ref='2102.0000%dv1' % i, title='Neighbour %d' % i,
                      authors=authors, similarity=0.7, group='close', date='2021-02-01', venue='', abstract='', source='index')
                 for i, authors in enumerate(['Zoe Kim, Bob Ray', 'Zoe Kim, Dan Poe', 'Zoe Kim, Bob Ray', 'Eve Moe'])]
        Generated.objects.create(kind=neighbours.KIND, key='2101.00001v1', lang='', status='done',
                                 content=json.dumps({'items': items, 'counts': {'cites': 12}, 'computed': timezone.now().isoformat()}))
        html = Client().get('/en/paper/2101.00001v1/').content.decode()
        self.assertIn('<strong>Zoe Kim</strong> (3 papers)', html)
        self.assertIn('1 more person signs several of them', html)
        self.assertIn('12 papers already cite it', html)
        self.assertIn('Create a free account: your first map is free', html)
        html = self.client_for(self.user()).get('/en/paper/2101.00001v1/').content.decode()
        self.assertIn('your first map is free', html)
        self.assertIn('#tool-form', html)
        html = self.client_for(self.user('bob', paid=True)).get('/en/paper/2101.00001v1/').content.decode()
        self.assertNotIn('already cite it', html)  # its alerts are theirs already

    def test_every_tool_keeps_lapsed_results_closed(self):
        ann = self.user()
        for kind, path in (('teams', 'teams'), ('missed', 'missed'), ('refcheck', 'refcheck'), ('literature', 'literature-review')):
            row = self.row(ann, kind=kind, trial=False, content={})
            self.assertEqual(self.client_for(ann).get('/en/%s/%s/' % (path, row.key)).status_code, 404, kind)
