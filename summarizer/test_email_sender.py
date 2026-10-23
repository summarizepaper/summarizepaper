from email.utils import parseaddr
from unittest import mock

from django.contrib.auth.models import User
from django.core import mail
from django.test import Client, RequestFactory, TestCase, override_settings
from django.utils import translation

from summarizer import accounts, digest
from summarizer.models import DigestItem, DigestSubscription


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
class EmailSenderTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('mail-reader', email='reader@example.com', password='test-password')

    def assert_sender(self, message, address='contact@summarizepaper.com'):
        # Check the actual MIME header, not just the setting or constructor argument.
        self.assertEqual(parseaddr(str(message.message()['From'])), ('SummarizePaper', address))

    def test_new_paper_digest_has_product_name_and_keeps_unsubscribe_headers(self):
        subscription = DigestSubscription.objects.create(user=self.user, categories='astro-ph.EP', lang='fr')
        item = DigestItem.objects.create(arxiv_id='2610.00001v1', lang='fr', title='Un article', blurb='Un résultat.')
        digest.send_one(subscription, [item], 'https://www.summarizepaper.com')
        self.assertEqual(len(mail.outbox), 1)
        message = mail.outbox[0]
        self.assert_sender(message)
        self.assertEqual(message.to, [self.user.email])
        self.assertIn('List-Unsubscribe', message.message())
        self.assertTrue(message.alternatives)

    def test_activation_and_login_links_use_the_same_branded_sender(self):
        request = RequestFactory().get('/en/login/')
        with translation.override('en'), mock.patch('summarizer.mailcheck.undeliverable_reason', return_value=None):
            for active in (False, True):
                self.user.is_active = active
                self.user.save(update_fields=['is_active'])
                self.assertTrue(accounts.send_login_link(request, self.user))
                self.assert_sender(mail.outbox[-1])
                self.assertIn('/en/login/link/', mail.outbox[-1].body)
        self.assertEqual(len(mail.outbox), 2)

    def test_django_password_reset_also_uses_the_branded_default_sender(self):
        response = Client().post('/en/password_reset/', {'email': self.user.email})
        self.assertEqual(response.status_code, 302)
        self.assertEqual(len(mail.outbox), 1)
        self.assert_sender(mail.outbox[0])
        self.assertIn('/en/reset/', mail.outbox[0].body)

    def test_contact_form_keeps_its_existing_branded_communication_address(self):
        response = Client().post('/en/contact/', {'name': 'Reader', 'email': self.user.email, 'message': 'A question.'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assert_sender(mail.outbox[0], 'communication@summarizepaper.com')
