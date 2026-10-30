import datetime
import json
from unittest import mock

from django.test import Client
from django.utils import timezone

from summarizer import trends, trends_web
from summarizer.models import RisingTopics, TermWeek
from summarizer.test_tools import ToolsTestCase


class TrendRetryTests(ToolsTestCase):
    WEEK = datetime.date(2026, 10, 19)
    SIGNAL = [{'term': 'pretraining objective', 'n': 14, 'expected': 3.3, 'growth': 4.2, 'llr': 18.9, 'teams': 14}]

    def setUp(self):
        super().setUp()
        TermWeek.objects.create(scope='*', week=self.WEEK, papers=1000, counts='{}')

    def state(self):
        return trends.result_state('*', 'en', self.WEEK)

    def test_no_signal_and_insufficient_history_are_distinct_and_rechecked(self):
        with mock.patch('summarizer.generated.chat_json') as chat:
            trends.write('*', self.WEEK, rising=[])
            self.assertEqual(self.state()['status'], 'no_signal')
            self.assertNotIn('*', trends.due_scopes(self.WEEK))
            self.assertIn('*', trends.due_scopes(self.WEEK, timezone.now() + datetime.timedelta(days=2)))
            with mock.patch('summarizer.trends.rising_terms', return_value=None):
                trends.write('*', self.WEEK)
            self.assertEqual(self.state()['status'], 'insufficient_data')
            chat.assert_not_called()

    def test_old_empty_result_is_due_but_published_legacy_topics_are_not(self):
        for lang in trends.LANGS:
            RisingTopics.objects.create(scope='*', lang=lang, day=self.WEEK, content='{"topics": []}')
        self.assertIn('*', trends.due_scopes(self.WEEK))
        RisingTopics.objects.update(content='{"topics": [{"title": "Existing"}]}')
        self.assertNotIn('*', trends.due_scopes(self.WEEK))
        RisingTopics.objects.filter(lang='fr').delete()
        self.assertIn('*', trends.due_scopes(self.WEEK))

    def test_model_filtering_is_not_no_signal_and_attempts_are_bounded(self):
        with mock.patch('summarizer.trends._write_topics', return_value=None) as compose:
            for i in range(5):
                trends.write('*', self.WEEK, rising=self.SIGNAL)
            self.assertEqual(compose.call_count, 3)
            self.assertEqual(self.state()['status'], 'filtered')
            self.assertEqual(self.state()['candidate_count'], 1)
            self.assertEqual(self.state()['attempts'], 3)
            changed = [dict(self.SIGNAL[0], n=15)]
            trends.write('*', self.WEEK, rising=changed)
            self.assertEqual(compose.call_count, 4)
            self.assertEqual(self.state()['attempts'], 1)
        self.assertEqual(trends.result_state('*', 'fr', self.WEEK)['status'], 'filtered')

    def test_failure_is_retryable_and_does_not_claim_no_rise(self):
        with mock.patch('summarizer.trends._write_topics', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                trends.write('*', self.WEEK, rising=self.SIGNAL)
        self.assertEqual(self.state()['status'], 'failed')
        self.assertNotIn('*', trends.due_scopes(self.WEEK))
        self.assertIn('*', trends.due_scopes(self.WEEK, timezone.now() + datetime.timedelta(hours=2)))
        page = Client().get('/en/trends/')
        self.assertContains(page, 'awaiting validation or generation')
        self.assertNotContains(page, 'Nothing rose clearly')

    def test_failed_refresh_preserves_published_results(self):
        for lang in trends.LANGS:
            RisingTopics.objects.create(scope='*', lang=lang, day=self.WEEK, content='{"topics": [{"title": "Existing"}]}')
        with mock.patch('summarizer.trends._write_topics', side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                trends.write('*', self.WEEK, rising=self.SIGNAL)
        self.assertEqual(self.state()['topics'][0]['title'], 'Existing')

    def test_empty_result_messages_and_scope_links_do_not_advertise_empty_scopes(self):
        messages = [('no_signal', 'No term passed'), ('filtered', 'no research topic was retained'),
                    ('insufficient_data', 'not yet enough papers'), ('pending', 'awaiting validation')]
        for status, message in messages:
            RisingTopics.objects.update_or_create(scope='*', lang='en', day=self.WEEK,
                defaults={'content': json.dumps({'topics': [], 'status': status})})
            self.assertContains(Client().get('/en/trends/'), message)
        RisingTopics.objects.create(scope='cs', lang='en', day=self.WEEK,
                                    content='{"topics": [], "status": "filtered"}')
        self.assertEqual(trends_web.scope_links('en', self.WEEK, '*')[0], [])

    def test_invalid_model_schema_is_a_failure_not_a_filtered_result(self):
        with mock.patch('summarizer.trends.stored_papers', return_value=[]), \
             mock.patch('summarizer.generated.chat_json', return_value={'wrong': []}):
            with self.assertRaises(Exception):
                trends.write('*', self.WEEK, rising=self.SIGNAL)
        self.assertEqual(self.state()['status'], 'failed')
