import datetime
from unittest import mock

from django.template import Context, Template
from django.test import SimpleTestCase, TestCase
from django.utils import timezone, translation

from . import result_share
from .models import Generated


class ShareTimingTests(SimpleTestCase):
    def setUp(self):
        self.now = datetime.datetime(2026, 10, 29, 10, 0, tzinfo=datetime.timezone.utc)

    def test_existing_links_show_last_activation_and_precise_remaining_time(self):
        expiry = self.now + datetime.timedelta(days=12, hours=4, minutes=20)
        with translation.override('en'), timezone.override('Europe/Paris'):
            text = result_share.timing(expiry, now=self.now)
        self.assertIn('Activated or renewed on', text)
        self.assertIn('Expires on', text)
        self.assertIn('(CET)', text)
        self.assertIn('Time remaining: 12 days, 4 hours.', text)
        self.assertNotIn('20 minutes', text)

    def test_short_and_expired_links(self):
        with translation.override('en'):
            for delta, expected in [(datetime.timedelta(hours=1, minutes=7), '1 hour, 7 minutes'),
                                    (datetime.timedelta(seconds=20), 'Less than a minute'),
                                    (datetime.timedelta(), 'Expired link.'),
                                    (datetime.timedelta(days=-1), 'Expired link.')]:
                self.assertIn(expected, result_share.timing(self.now + delta, now=self.now))
            self.assertEqual(result_share.timing(None), 'No scheduled expiration.')

    def test_french_template_and_daylight_saving_dates(self):
        with translation.override('fr'), timezone.override('Europe/Paris'), \
             mock.patch('summarizer.result_share.timezone.now', return_value=self.now):
            text = Template('{% load share_timing %}{% share_timing expires %}').render(
                Context({'expires': self.now + datetime.timedelta(days=1, hours=2)}))
        self.assertIn('Activé ou renouvelé le', text)
        self.assertIn('Temps restant : 1 jour, 2 heures.', text)
        self.assertIn('(CEST)', text)  # activation before the autumn time change
        self.assertIn('(CET)', text)


class ShareRenewalTimingTests(TestCase):
    def test_renewing_keeps_link_and_reports_new_activation_time(self):
        row = Generated.objects.create(kind='literature', key='timing', status='done')
        start = datetime.datetime(2026, 10, 20, 10, 0, tzinfo=datetime.timezone.utc)
        with mock.patch('summarizer.result_share.timezone.now', return_value=start):
            result_share.act(row, 'share')
        token = row.share_token
        later = start + datetime.timedelta(days=5)
        with mock.patch('summarizer.result_share.timezone.now', return_value=later):
            result_share.act(row, 'share')
        row.refresh_from_db()
        self.assertEqual(row.share_token, token)
        self.assertEqual(row.share_expires - datetime.timedelta(days=90), later)
        result_share.act(row, 'unshare')
        row.refresh_from_db()
        self.assertFalse(row.share_token)
        self.assertIsNone(row.share_expires)
