from unittest.mock import patch

from django.core import mail
from django.core.cache import cache
from django.test import TestCase, override_settings


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
class ContactProtectionTests(TestCase):
    def setUp(self):
        cache.clear()
        self.fields = {'name': 'Reader', 'email': 'reader@example.com', 'message': 'A real question.'}

    def test_invalid_or_missing_input_never_sends(self):
        for fields in ({}, {**self.fields, 'email': 'invalid'}, {**self.fields, 'name': 'Header\r\nInjection'},
                       {**self.fields, 'message': 'x' * 10001}):
            self.assertEqual(self.client.post('/en/contact/', fields).status_code, 400)
        self.assertEqual(len(mail.outbox), 0)

    def test_honeypot_does_not_send(self):
        self.assertEqual(self.client.post('/en/contact/', {**self.fields, 'website': 'spam'}).status_code, 302)
        self.assertEqual(len(mail.outbox), 0)

    def test_duplicate_and_refresh_send_once_and_reply_goes_to_reader(self):
        for _ in range(2):
            response = self.client.post('/en/contact/', self.fields)
            self.assertEqual(response.status_code, 302)
            self.assertEqual(self.client.get(response.url).status_code, 200)
        self.assertEqual(len(mail.outbox), 1)
        self.assertEqual(mail.outbox[0].reply_to, ['reader@example.com'])

    def test_ip_rate_limit_applies_even_when_email_changes(self):
        for number in range(4):
            response = self.client.post('/en/contact/', {**self.fields, 'email': 'reader%s@example.com' % number})
            self.assertEqual(response.status_code, 302 if number < 3 else 429)
        self.assertEqual(len(mail.outbox), 3)

    def test_email_rate_limit_applies_even_when_ip_changes(self):
        for number in range(4):
            response = self.client.post('/en/contact/', {**self.fields, 'message': str(number)},
                                        REMOTE_ADDR='192.0.2.%s' % number)
            self.assertEqual(response.status_code, 302 if number < 3 else 429)
        self.assertEqual(len(mail.outbox), 3)

    def test_global_limit_applies_to_distributed_submissions(self):
        cache.set('contact:limit:global:all:123', 30, 7200)
        with patch('summarizer.contact_form.time.time', return_value=123 * 3600):
            response = self.client.post('/en/contact/', self.fields)
        self.assertEqual(response.status_code, 429)
        self.assertEqual(len(mail.outbox), 0)

    def test_cache_outage_never_bypasses_protection(self):
        with patch('summarizer.contact_form.cache.get', side_effect=ConnectionError('unavailable')):
            response = self.client.post('/en/contact/', self.fields)
        self.assertEqual(response.status_code, 503)
        self.assertEqual(len(mail.outbox), 0)

    def test_send_failure_allows_retry(self):
        with patch('summarizer.contact_form.EmailMessage.send', side_effect=RuntimeError('SMTP unavailable')):
            self.assertEqual(self.client.post('/en/contact/', self.fields).status_code, 503)
        self.assertEqual(self.client.post('/en/contact/', self.fields).status_code, 302)
        self.assertEqual(len(mail.outbox), 1)
