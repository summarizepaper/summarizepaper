import json
import re
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
                       'importance': 'high' if i < 2 else 'low', 'relation': 'Same question, other telescope %d.' % i,
                       'position': 'Cite it as the first detection %d.' % i, 'section': 'Introduction',
                       'draft_quote': 'A secret sentence of the draft %d.' % i}
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
        self.assertIn('How to position your work against each one', html)
        self.assertNotIn('secret sentence', html)  # never the draft's own sentences
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

    def test_an_email_sign_up_from_a_teaser_comes_back_to_the_tool(self):
        from unittest import mock
        from django.core import mail
        with mock.patch('summarizer.accounts.looks_like_bot', return_value=False), \
             mock.patch('summarizer.mailcheck.undeliverable_reason', return_value=''):
            Client().post('/en/register/?next=/en/teams/%3Ftopic%3DExomoons', {
                'username': 'newcomer', 'email': 'newcomer@example.org', 'password1': 'a-Long-pass-123',
                'password2': 'a-Long-pass-123', 'next': '/en/teams/?topic=Exomoons'})
        link = re.search(r'https?://\S+/login/link/\S+', mail.outbox[-1].body).group(0)
        path = link.split('://', 1)[1].split('/', 1)[1]
        response = Client().post('/' + path)
        self.assertEqual(response['Location'], '/en/teams/?topic=Exomoons')

    def test_a_first_subscription_starts_with_seven_free_days_and_only_the_first(self):
        from types import SimpleNamespace
        from unittest import mock
        from django.test import override_settings
        from summarizer.models import UserPlan
        ann = self.user()
        with override_settings(STRIPE_SECRET_KEY='sk_test', STRIPE_PRICE_MONTHLY='price_test'), \
             mock.patch('stripe.checkout.Session.create', return_value=SimpleNamespace(url='https://checkout.stripe.com/t')) as create:
            self.client_for(ann).post('/en/premium/checkout/', {'period': 'monthly'})
            params = create.call_args.kwargs
            self.assertEqual(params['subscription_data']['trial_period_days'], 7)
            self.assertEqual(params['payment_method_collection'], 'always')
            with mock.patch('summarizer.billing.price_label', return_value='7 EUR'):
                self.assertContains(self.client_for(ann).get('/en/premium/'), '7 days free')
            UserPlan.objects.create(user=ann, paid=False, stripe_subscription_id='sub_old')  # subscribed before
            self.client_for(ann).post('/en/premium/checkout/', {'period': 'monthly'})
            self.assertNotIn('trial_period_days', create.call_args.kwargs['subscription_data'])
            with mock.patch('summarizer.billing.price_label', return_value='7 EUR'):
                self.assertNotContains(self.client_for(ann).get('/en/premium/'), '7 days free')

    def test_the_premium_page_shows_the_first_result_of_each_example(self):
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        self.row(staff, showcase=True, content=missed_result(4))
        html = Client().get('/en/premium/').content.decode()
        self.assertIn('Real results, before you subscribe', html)
        self.assertIn('Cite it as the first detection 0.', html)  # what the tool writes, not only its list
        self.assertIn('Where to cite each one, section by section', html)
        self.assertNotIn('Paper number 1', html)
        self.assertNotIn('secret sentence', html)

    def test_the_paper_page_shows_the_first_change_of_its_latest_version(self):
        from unittest import mock
        self.paper(arxiv_id='2101.00001v3')
        anonymous = Client().get('/en/paper/2101.00001v3/').content.decode()
        self.assertNotIn('From v2 to v3', anonymous)  # nothing computed, and a visitor never starts it
        self.assertFalse(Generated.objects.filter(kind='changes').exists())
        comparison = {'changes': [{'kind': 'results', 'importance': 'major', 'detail': 'The mass is now 3 Jupiter masses.'},
                                  {'kind': 'figures', 'importance': 'minor', 'detail': 'Figure 2 redrawn.'},
                                  {'kind': 'writing', 'importance': 'minor', 'detail': 'Typos.'}], 'results_changed': True}
        with mock.patch('summarizer.changes.compare', return_value=comparison):
            self.client_for(self.user()).get('/en/paper/2101.00001v3/')
        row = Generated.objects.get(kind='changes', key='2101.00001:2:3', lang='en')
        self.assertEqual(row.status, 'done')
        anonymous = Client().get('/en/paper/2101.00001v3/').content.decode()
        self.assertIn('From v2 to v3', anonymous)
        self.assertIn('The mass is now 3 Jupiter masses.', anonymous)
        self.assertNotIn('Figure 2 redrawn', anonymous)
        self.assertIn('2 more changes, and whether the main results changed, with Premium.', anonymous)
        paid = self.client_for(self.user('bob', paid=True)).get('/en/paper/2101.00001v3/').content.decode()
        self.assertNotIn('more changes, and whether', paid)
        self.assertIsNone(teasers.version_changes(None, '2101.00001v1'))

    def test_a_correction_says_which_fields_change(self):
        before = '@article{k,\n  title={A},\n  year={2016}\n}'
        after = '@article{k,\n  title = {A},\n  year = {2017},\n  doi = {10.1/x}\n}'
        self.assertEqual(teasers.bib_changes(before, after), ['year: 2016 → 2017', 'doi: + 10.1/x'])

    def test_the_example_follows_the_visitors_field(self):
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        astro = missed_result(3)
        astro['items'][0]['position'] = 'Cite it for the exomoons.'
        bio = missed_result(3)
        bio['items'][0]['position'] = 'Cite it for the protein folding.'
        self.row(staff, showcase=True, showcase_field='astro', content=astro)
        self.row(staff, showcase=True, showcase_field='bio', content=bio)
        self.assertEqual(teasers.field_of('astro-ph.EP'), 'astro')
        self.assertEqual(teasers.field_of('cs.CL'), 'cs')
        self.assertEqual(teasers.field_of('q-bio.NC'), 'bio')
        self.assertEqual(teasers.field_of('biorxiv'), 'bio')
        self.assertEqual(teasers.field_of('math.AP'), '')
        client = Client()
        html = client.get('/en/missed/?ex=bio').content.decode()
        self.assertIn('protein folding', html)
        self.assertIn('?ex=astro#example', html)  # the other fields, one click away
        html = client.get('/en/missed/?ex=astro').content.decode()
        self.assertIn('exomoons', html)
        # without a choice: the field of the last paper read, else the default (cs: none here, so another one)
        self.assertIn('Cite it for the', client.get('/en/missed/').content.decode())
        session = client.session
        session['example_field'] = 'bio'
        session.save()
        self.assertIn('protein folding', client.get('/en/missed/').content.decode())
        self.assertIn('protein folding', client.get('/en/premium/').content.decode())

    def test_an_example_shows_its_source_its_measured_quality_and_a_complete_example(self):
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        result = missed_result(3)
        result['items'][0]['abstract_quote'] = 'We detect the gap in CO emission.'
        self.row(staff, showcase=True, content=result)
        html = Client().get('/en/missed/').content.decode()
        self.assertIn('We detect the gap in CO emission.', html)
        self.assertIn('/en/paper/2101.00000v1/', html)
        self.assertIn('/en/guides/what-did-i-miss-uncited-papers/', html)
        self.assertIn('Measured in November 2026', Client().get('/en/conference/').content.decode())

    def test_the_four_examples_of_slides_tables_and_alert_emails(self):
        staff = self.user('staff')
        staff.is_staff = True
        staff.save()
        deck = {'generation_version': 3, 'title': 'Clouds', 'duration': 20, 'papers': [{'id': '2101.00001v1', 'title': 'Clouds on a hot planet'}],
                'qa': [{'q': 'Why?'}], 'pages': [
                    {'kind': 'title', 'title': 'Clouds on a hot planet', 'bullets': ['Ann Lee'], 'sources': []},
                    {'kind': 'content', 'title': 'What they found', 'bullets': ['Thick clouds at the terminator'],
                     'sources': [{'verified': True, 'quote': 'We detect thick clouds at the terminator.'}]},
                    {'kind': 'content', 'title': 'Method', 'bullets': ['Transit spectra'], 'sources': []}]}
        Generated.objects.create(kind='slides_v3', key='2101.00001v1', lang='en', status='done', showcase=True, content=json.dumps(deck))
        html = Client().get('/en/journal-club/').content.decode()
        self.assertIn('3 slides for a 20-minute talk', html)
        self.assertIn('We detect thick clouds at the terminator.', html)
        self.assertIn('Create a free account, then Premium+ runs it on your own work.', html)
        self.assertNotIn('your first search with this tool is free', html)
        table = {'version': 2, 'template': 'general', 'columns': [['question', ''], ['method', '']], 'ids': ['a', 'b'],
                 'rows': [{'arxiv_id': 'a', 'title': 'Paper A', 'source': 'full_text', 'cells': {
                              'question': {'text': 'Clouds?', 'quote': 'We ask about clouds.'}, 'method': {'text': 'Spectra', 'quote': ''}}},
                          {'arxiv_id': 'b', 'title': 'Paper B', 'source': 'abstract', 'cells': {
                              'question': {'text': 'Hazes?', 'quote': 'We ask about hazes.'}, 'method': {'text': 'Imaging', 'quote': ''}}}],
                 'agreements': [], 'gaps': [],
                 'disagreements': [{'text': 'A finds clouds, B hazes.', 'papers': ['a', 'b'], 'columns': ['question']}]}
        self.row(staff, kind='compare', showcase=True, content=table, trial=False)
        html = self.client_for(self.user()).get('/en/collections/').content.decode()
        self.assertIn('2 papers compared on 2 columns', html)
        self.assertIn('A finds clouds, B hazes.', html)
        self.assertIn('We ask about clouds.', html)
        email = '<p>A paper close to your project</p>'
        self.row(staff, kind='scoop_example', showcase=True, trial=False, content={
            'project': 'Exomoons', 'subject': '1 close paper', 'html': email,
            'matches': [{'title': 'Moons by timing', 'overlap': 'high', 'shared': 'Same method.', 'original': 'Your data.'},
                        {'title': 'Other', 'overlap': 'partial', 'shared': 'Same object.', 'original': ''}]})
        html = Client().get('/en/scoops/').content.decode()
        self.assertIn('2 new papers close to the project “Exomoons”', html)
        self.assertIn('The email you would have received', html)
        self.assertIn('srcdoc="&lt;p&gt;A paper close to your project&lt;/p&gt;"', html)  # escaped, in a sandbox
        self.assertIn('sandbox=""', html)
        self.row(staff, kind='citation_example', showcase=True, trial=False, content={
            'paper': 'Planck 2018', 'subject': '3 new citations', 'html': email,
            'items': [{'title': 'A tension', 'label': 'questions it', 'note': 'Finds a lower H0.'}]})
        html = Client().get('/en/citations/').content.decode()
        self.assertIn('1 new paper citing “Planck 2018”', html)
        self.assertIn('Finds a lower H0.', html)
        premium = Client().get('/en/premium/').content.decode()
        for name in ('Journal club', 'Collections and comparison tables', 'Scoop watch', 'Citation alerts'):
            self.assertIn(name, premium)
