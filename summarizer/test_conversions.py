import datetime
import json
import uuid
from unittest import mock

from django.contrib.auth.models import User
from django.core import mail
from django.test import override_settings
from django.utils import timezone

from summarizer import billing, conversion_emails, conversions, example_pages
from summarizer.models import ConversionEmail, Generated, PageView, UserPlan
from summarizer.test_teasers import missed_result
from summarizer.test_tools import ToolsTestCase


class ConversionEmailTests(ToolsTestCase):
    def trial(self, user, hours_ago=24, **kwargs):
        row = Generated.objects.create(kind='missed', key=uuid.uuid4().hex, lang='en', user=user, status='done', trial=True,
                                       content=json.dumps(missed_result()), **kwargs)
        Generated.objects.filter(pk=row.pk).update(updated=timezone.now() - datetime.timedelta(hours=hours_ago))
        return row

    def test_the_day_after_a_free_try_one_email_tells_what_is_hidden(self):
        ann = self.user()
        row = self.trial(ann)
        self.trial(self.user('fresh'), hours_ago=2)  # read a moment ago: not yet
        self.trial(self.user('old'), hours_ago=24 * 6)  # too old to write about
        self.assertEqual(conversion_emails.run(timezone.now(), 10), 1)
        self.assertEqual(len(mail.outbox), 1)
        message = mail.outbox[0]
        self.assertEqual(message.to, ['ann@example.com'])
        self.assertIn('3 more results', message.subject)
        html = message.alternatives[0][0]
        self.assertIn('Paper number 0', html)
        self.assertNotIn('Paper number 3', html)  # greyed in the email too
        self.assertNotIn('secret sentence', html)
        self.assertIn('/en/missed/%s/' % row.key, html)
        self.assertIn('/en/premium/', html)
        self.assertIn('7 days are free', html)
        self.assertIn('List-Unsubscribe', message.extra_headers)
        # once only
        self.assertEqual(conversion_emails.run(timezone.now(), 10), 0)

    def test_no_email_to_a_subscriber_nor_after_the_opt_out(self):
        bob = self.user('bob', paid=True)
        self.trial(bob)
        ann = self.user()
        self.trial(ann)
        page = self.client.get('/en/emails/stop/%s/' % conversion_emails.optout_token(ann))
        self.assertEqual(page.status_code, 200)
        self.assertEqual(self.client.get('/en/emails/stop/forged/').status_code, 404)
        self.assertEqual(conversion_emails.run(timezone.now(), 10), 0)
        self.assertEqual(mail.outbox, [])

    @override_settings(STRIPE_SECRET_KEY='sk_test', STRIPE_PRICE_MONTHLY='price_m')
    def test_two_days_before_a_trial_ends_one_email_with_the_payment_and_what_was_made(self):
        ann = self.user()
        now = timezone.now()
        UserPlan.objects.create(user=ann, paid=True, stripe_subscription_id='sub_1', trial_end=now + datetime.timedelta(hours=40))
        UserPlan.objects.create(user=self.user('later'), paid=True, stripe_subscription_id='sub_2', trial_end=now + datetime.timedelta(days=5))
        made = Generated.objects.create(kind='missed', key='k1', lang='en', user=ann, status='done', content=json.dumps(missed_result()))
        subscription = {'status': 'trialing', 'items': {'data': [{'price': {'unit_amount': 900, 'currency': 'eur', 'recurring': {'interval': 'month'}}}]}}
        stripe = mock.Mock()
        stripe.Subscription.retrieve.return_value = subscription
        with mock.patch.object(billing, '_stripe', return_value=stripe):
            self.assertEqual(conversion_emails.run(now, 10), 1)
            self.assertEqual(conversion_emails.run(now, 10), 0)
        stripe.Subscription.retrieve.assert_called_once_with('sub_1')
        html = mail.outbox[0].alternatives[0][0]
        self.assertIn('9.00 EUR', html)
        self.assertIn('Clouds on hot planets', html)
        self.assertIn('/en/missed/%s/' % made.key, html)
        self.assertIn('Manage my subscription', html)

    def test_stripe_trial_end_is_kept_on_the_plan(self):
        ann = self.user()
        end = int((timezone.now() + datetime.timedelta(days=7)).timestamp())
        plan = billing.sync_subscription({'id': 'sub_9', 'customer': 'cus', 'status': 'trialing', 'trial_end': end,
                                          'metadata': {'user_id': ann.pk}, 'items': {'data': []}, 'current_period_end': end})
        self.assertEqual(int(plan.trial_end.timestamp()), end)
        self.assertTrue(plan.paid)


class ExamplePageTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.staff = User.objects.create_user('demo', is_staff=True)
        self.row = Generated.objects.create(kind='missed', key='ex1', lang='en', user=self.staff, status='done', showcase=True,
                                            showcase_field='astro', content=json.dumps(missed_result()))

    def test_an_example_has_its_own_public_page_listed_in_the_index_and_sitemap(self):
        address = example_pages.slug(self.row)
        self.assertEqual(address, 'missed-papers-clouds-on-hot-planets')
        index = self.client.get('/en/examples/').content.decode()
        self.assertIn('/en/examples/%s/' % address, index)
        page = self.client.get('/en/examples/%s/' % address)
        self.assertEqual(page.status_code, 200)
        html = page.content.decode()
        self.assertIn('Clouds on hot planets', html)
        self.assertIn('Paper number 4', html)  # more than on the tool's page
        self.assertNotIn('secret sentence', html)
        self.assertIn('rel="canonical"', html)
        self.assertEqual(self.client.get('/en/examples/nothing-here/').status_code, 404)
        self.assertIn('/en/examples/%s/' % address, self.client.get('/sitemap-examples.xml').content.decode())

    def test_a_members_search_never_gets_a_page(self):
        ann = self.user()
        Generated.objects.create(kind='missed', key='private', lang='en', user=ann, status='done', showcase=True,
                                 content=json.dumps(dict(missed_result(), title='My private draft')))
        self.assertNotIn('my-private-draft', self.client.get('/en/examples/').content.decode())
        self.assertEqual(self.client.get('/en/examples/missed-papers-my-private-draft/').status_code, 404)


class ConversionsPageTests(ToolsTestCase):
    def test_the_staff_page_follows_a_sign_up_back_to_the_pages_it_saw(self):
        ann = self.user()
        joined = ann.date_joined
        for minutes, view, user in ((-30, 'missed', None), (-20, 'premium', None), (-10, 'register', None), (1, 'home', ann)):
            pv = PageView.objects.create(visitor='v1', view=view, path='/en/%s/' % view, human=True, user=user, referrer='google.com')
            PageView.objects.filter(pk=pv.pk).update(created=joined + datetime.timedelta(minutes=minutes))
        Generated.objects.create(kind='missed', key='t', lang='en', user=ann, status='done', trial=True, content='{}')
        report = conversions.report(30)
        journey = report['journeys'][0]
        self.assertEqual(journey['user'], ann)
        self.assertEqual(journey['seen'], ['Qu’ai-je manqué ?', 'Premium'])
        self.assertEqual(journey['origin'], 'google.com')
        self.assertEqual(journey['tries'], ['Qu’ai-je manqué ?'])
        admin = User.objects.create_superuser('boss', 'boss@example.com', 'x')
        page = self.client_for(admin).get('/admin/conversions/?jours=7')
        self.assertEqual(page.status_code, 200)
        self.assertIn('Les inscriptions, une par une', page.content.decode())
        self.assertNotEqual(self.client_for(ann).get('/admin/conversions/').status_code, 200)
