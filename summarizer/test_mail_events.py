import datetime
import json

from django.test import TestCase, override_settings
from django.utils import timezone

from summarizer.models import EmailEvent, OutreachContact, SuppressedEmail

URL = '/cloudmailin/events/s3cret/'


@override_settings(CLOUDMAILIN_EVENTS_SECRET='s3cret')
class CloudMailinEventTests(TestCase):
    def post(self, *events, url=URL):
        return self.client.post(url, json.dumps({'events': list(events)}), content_type='application/json')

    def contact(self, email='reader@example.com', status='sent'):
        return OutreachContact.objects.create(email=email, status=status, subject='Feedback', sent=timezone.now())

    def test_wrong_or_unset_secret_is_not_found(self):
        self.assertEqual(self.post({'kind': 'delivery', 'recipient': 'a@b.co'}, url='/cloudmailin/events/wrong/').status_code, 404)
        with override_settings(CLOUDMAILIN_EVENTS_SECRET=''):
            self.assertEqual(self.post({'kind': 'delivery', 'recipient': 'a@b.co'}, url='/cloudmailin/events/x/').status_code, 404)
        self.assertFalse(EmailEvent.objects.exists())

    def test_malformed_payload_is_rejected(self):
        self.assertEqual(self.client.post(URL, 'nope', content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post(URL, '{}', content_type='application/json').status_code, 400)

    def test_events_are_kept_and_outreach_follows_delivery(self):
        row = self.contact()
        response = self.post({'kind': 'dispatch', 'recipient': 'Reader@Example.com', 'message_id': 'q1', 'details': {}},
                             {'kind': 'delivery', 'recipient': 'reader@example.com', 'message_id': 'q1'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(EmailEvent.objects.filter(recipient='reader@example.com').count(), 2)
        row.refresh_from_db()
        self.assertEqual(row.status, 'delivered')
        self.post({'kind': 'dispatch', 'recipient': 'reader@example.com'})  # late or out-of-order event
        row.refresh_from_db()
        self.assertEqual(row.status, 'delivered')

    def test_bounce_blocks_the_address_but_never_undoes_a_reply(self):
        bounced, replied = self.contact(), self.contact('other@example.com', 'replied')
        self.post({'kind': 'bounce', 'recipient': 'reader@example.com', 'details': {'reason': '550 5.1.1'}},
                  {'kind': 'bounce', 'recipient': 'other@example.com'})
        bounced.refresh_from_db()
        replied.refresh_from_db()
        self.assertEqual((bounced.status, replied.status), ('bounced', 'replied'))
        self.assertEqual(SuppressedEmail.objects.count(), 2)

    def test_suppressed_marks_outreach_without_blocking_the_address_ourselves(self):
        row = self.contact()
        self.post({'kind': 'suppressed', 'recipient': 'reader@example.com'})
        row.refresh_from_db()
        self.assertEqual(row.status, 'suppressed')
        self.assertFalse(SuppressedEmail.objects.exists())

    def test_old_invitations_are_not_touched(self):
        row = self.contact()
        OutreachContact.objects.filter(pk=row.pk).update(created=timezone.now() - datetime.timedelta(days=60))
        self.post({'kind': 'delivery', 'recipient': 'reader@example.com'})
        row.refresh_from_db()
        self.assertEqual(row.status, 'sent')
