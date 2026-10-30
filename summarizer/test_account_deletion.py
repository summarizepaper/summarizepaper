from unittest.mock import patch

from django.contrib.auth.models import User
from django.test import TestCase

from summarizer.models import (Collection, DigestSubscription, EmailEvent, OutreachContact, SuppressedEmail,
                               UserPlan)


class AccountDeletionTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('ada', 'ada@example.com', 'pw')
        self.client.force_login(self.user)

    def test_anonymous_visitor_is_sent_to_sign_in(self):
        self.client.logout()
        self.assertEqual(self.client.get('/en/account/delete/').status_code, 302)

    def test_nothing_is_deleted_without_confirmation(self):
        self.assertEqual(self.client.post('/en/account/delete/').status_code, 400)
        self.assertTrue(User.objects.filter(pk=self.user.pk).exists())

    def test_deletes_account_its_data_and_rows_kept_by_email(self):
        Collection.objects.create(user=self.user, name='Reading')
        DigestSubscription.objects.create(user=self.user)
        OutreachContact.objects.create(user=self.user, email='Ada@example.com', subject='Feedback?')
        EmailEvent.objects.create(recipient='ada@example.com', kind='delivered')
        SuppressedEmail.objects.create(email='ada@example.com', reason='complaint')

        response = self.client.post('/en/account/delete/', {'confirm': 'yes'})

        self.assertContains(response, 'Your account has been deleted')
        self.assertFalse(User.objects.filter(pk=self.user.pk).exists())
        self.assertFalse(Collection.objects.exists())
        self.assertFalse(DigestSubscription.objects.exists())
        self.assertFalse(OutreachContact.objects.exists())
        self.assertFalse(EmailEvent.objects.exists())
        # we must still never write to an address that complained
        self.assertTrue(SuppressedEmail.objects.exists())
        self.assertNotIn('_auth_user_id', self.client.session)

    def test_running_subscription_is_cancelled_first(self):
        UserPlan.objects.create(user=self.user, paid=True, stripe_subscription_id='sub_1', stripe_customer_id='cus_1')
        self.assertContains(self.client.get('/en/account/delete/'), 'without refund')
        with patch('summarizer.billing.stripe.Subscription.cancel') as cancel:
            self.client.post('/en/account/delete/', {'confirm': 'yes'})
        cancel.assert_called_once_with('sub_1')
        self.assertFalse(User.objects.filter(pk=self.user.pk).exists())

    def test_account_is_kept_when_stripe_cannot_cancel(self):
        UserPlan.objects.create(user=self.user, paid=True, stripe_subscription_id='sub_1')
        with patch('summarizer.billing.stripe.Subscription.cancel', side_effect=RuntimeError('down')):
            response = self.client.post('/en/account/delete/', {'confirm': 'yes'})
        self.assertEqual(response.status_code, 503)
        self.assertTrue(User.objects.filter(pk=self.user.pk).exists())

    def test_staff_accounts_are_not_deleted_here(self):
        self.user.is_staff = True
        self.user.save()
        self.client.post('/en/account/delete/', {'confirm': 'yes'})
        self.assertTrue(User.objects.filter(pk=self.user.pk).exists())
