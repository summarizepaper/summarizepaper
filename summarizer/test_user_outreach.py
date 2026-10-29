import datetime
import io
import smtplib
from unittest.mock import patch

from django.contrib.auth.models import User
from django.core import mail, signing
from django.core.management import call_command
from django.test import TestCase, override_settings
from django.urls import reverse
from django.utils import timezone, translation

from summarizer import user_outreach as outreach
from summarizer.models import OutreachContact, SuppressedEmail, DigestSubscription


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend', MAILCHECK_DNS=False)
class OutreachTests(TestCase):
    def user(self, name='reader', email='reader@example.com', **kwargs):
        return User.objects.create_user(name, email=email, last_login=timezone.now(), **kwargs)

    def test_preview_is_read_only(self):
        self.user()
        output = io.StringIO()
        call_command('contact_users', stdout=output)
        self.assertIn('1 candidats', output.getvalue())
        self.assertFalse(OutreachContact.objects.exists())
        self.assertFalse(getattr(mail, 'outbox', []))

    def test_candidates_exclude_suppressed_inactive_staff_old_and_aliases(self):
        user = self.user(email='a.b@gmail.com')
        self.user('alias', 'ab+other@gmail.com')
        self.user('staff', 'owner@gmail.com', is_staff=True)
        self.user('staff-alias', 'ow.ner+test@gmail.com')
        self.user('inactive', 'inactive@example.com', is_active=False)
        self.user('blocked', 'blocked@example.com')
        SuppressedEmail.objects.create(email='blocked@example.com')
        old = self.user('old', 'old@example.com')
        User.objects.filter(pk=old.pk).update(last_login=timezone.now()-datetime.timedelta(days=400))
        self.assertEqual(len(outreach.candidates()), 1)
        OutreachContact.objects.create(user=user, email='ab@gmail.com')
        self.assertEqual(outreach.candidates(), [])

    def test_suppressed_gmail_alias_is_excluded(self):
        self.user(email='a.b+new@gmail.com')
        SuppressedEmail.objects.create(email='ab@gmail.com')
        self.assertEqual(outreach.candidates(), [])

    def test_batch_cap_headers_snapshot_cooldown_and_no_repeats(self):
        for i in range(8): self.user('reader%s' % i, 'reader%s@example.com' % i)
        with self.assertRaises(outreach.BatchStopped): outreach.send_batch(7)
        self.assertEqual(len(outreach.send_batch()), 6)
        self.assertEqual(len(mail.outbox), 6)
        self.assertEqual(OutreachContact.objects.filter(status='sent').count(), 6)
        message = mail.outbox[0]
        self.assertIn('List-Unsubscribe', message.extra_headers)
        self.assertEqual(message.reply_to, ['contact@summarizepaper.com'])
        self.assertTrue(OutreachContact.objects.filter(body=message.body).exists())
        with self.assertRaises(outreach.BatchStopped): outreach.send_batch()
        OutreachContact.objects.update(created=timezone.now()-datetime.timedelta(days=3))
        self.assertEqual(len(outreach.send_batch()), 2)
        self.assertEqual(len(mail.outbox), 8)

    def test_next_batch_is_blocked_before_24_hours_and_allowed_after(self):
        self.user()
        row = OutreachContact.objects.create(email='previous@example.com', status='sent')
        OutreachContact.objects.filter(pk=row.pk).update(created=timezone.now()-datetime.timedelta(hours=23))
        with self.assertRaises(outreach.BatchStopped): outreach.send_batch()
        self.assertFalse(getattr(mail, 'outbox', []))
        OutreachContact.objects.filter(pk=row.pk).update(created=timezone.now()-datetime.timedelta(hours=25))
        self.assertEqual(len(outreach.send_batch()), 1)

    def test_older_segment_requires_recent_pilot(self):
        with self.assertRaises(outreach.BatchStopped): outreach.send_batch(segment='older')

    def test_unsubscribe_get_is_confirmation_post_is_effective(self):
        user = self.user()
        digest = DigestSubscription.objects.create(user=user, categories='astro-ph.EP')
        row = OutreachContact.objects.create(user=user, email=user.email, status='sent')
        with translation.override('fr'):
            url = reverse('outreach_unsubscribe', args=[signing.dumps(row.pk, salt=outreach.SALT)])
        response = self.client.get(url)
        self.assertEqual(response.status_code, 200)
        row.refresh_from_db()
        self.assertEqual(row.status, 'sent')
        self.assertEqual(self.client.post(url).status_code, 200)
        row.refresh_from_db()
        self.assertEqual(row.status, 'unsubscribed')
        self.assertTrue(User.objects.filter(pk=user.pk).exists())
        self.assertTrue(DigestSubscription.objects.filter(pk=digest.pk).exists())
        self.assertEqual(outreach.candidates(), [])
        self.assertEqual(self.client.get(url+'bad').status_code, 404)

    def test_unknown_mailbox_stops_batch_and_suppresses(self):
        user = self.user()
        error = smtplib.SMTPRecipientsRefused({user.email: (550, b'5.1.1 unknown user')})
        with patch.object(outreach.EmailMessage, 'send', side_effect=error):
            with self.assertRaises(outreach.BatchStopped): outreach.send_batch()
        self.assertEqual(OutreachContact.objects.get().status, 'bounced')
        self.assertTrue(SuppressedEmail.objects.filter(email=user.email).exists())
        self.assertTrue(User.objects.filter(pk=user.pk).exists())

    def test_policy_rejection_and_uncertain_error_are_not_invalid_addresses(self):
        user = self.user()
        error = smtplib.SMTPRecipientsRefused({user.email: (550, b'5.7.1 blocked by policy')})
        with patch.object(outreach.EmailMessage, 'send', side_effect=error):
            with self.assertRaises(outreach.BatchStopped): outreach.send_batch()
        self.assertEqual(OutreachContact.objects.get().status, 'failed')
        self.assertFalse(SuppressedEmail.objects.exists())
        self.assertEqual(outreach.candidates(), [])

    def test_log_and_optout_survive_account_deletion(self):
        user = self.user()
        row = OutreachContact.objects.create(user=user, email=user.email, status='unsubscribed')
        user.delete()
        row.refresh_from_db()
        self.assertIsNone(row.user_id)
        self.user('replacement')
        self.assertEqual(outreach.candidates(), [])
