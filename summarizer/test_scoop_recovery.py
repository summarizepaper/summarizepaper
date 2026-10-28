"""Interrupted Scoop Watch jobs can be retried; active and completed jobs stay protected."""
import datetime
from unittest import mock
from django.contrib.auth.models import User
from django.test import RequestFactory, TestCase
from django.utils import timezone, translation
from summarizer import generated, scoops
from summarizer.models import ScoopMatch, WatchedProject


class ScoopRecoveryTests(TestCase):
    def setUp(self):
        self.owner = User.objects.create_user('scoop-recovery')
        self.project = WatchedProject.objects.create(user=self.owner, title='Demo', description='Public demo project')
        self.now = timezone.now()

    def act(self, action, **fields):
        request = RequestFactory().post('/en/scoops/', {'action': action, 'pk': self.project.pk, **fields})
        request.user = self.owner
        with translation.override('en'), mock.patch('summarizer.generated.background') as background:
            result = scoops.act(request, WatchedProject.objects.filter(user=self.owner), action)
        return result, background

    def test_prior_search_recovers_from_stale_or_missing_start(self):
        for started in (None, self.now - generated.STALE - datetime.timedelta(seconds=1)):
            with self.subTest(started=started):
                self.project.prior_status, self.project.prior_checked = 'pending', started
                self.project.save()
                result, background = self.act('prior')
                background.assert_called_once_with(scoops.search_prior, self.project.pk)
                self.assertIn('Search for prior art started', result[1])
                self.project.refresh_from_db()
                self.assertEqual(self.project.prior_status, 'pending')
                self.assertGreaterEqual(self.project.prior_checked, self.now)

    def test_recent_prior_search_is_not_duplicated(self):
        self.project.prior_status, self.project.prior_checked = 'pending', self.now
        self.project.save()
        result, background = self.act('prior')
        background.assert_not_called()
        self.assertIn('already running', result[1])

    def match(self, status, started):
        return ScoopMatch.objects.create(project=self.project, arxiv_id='2608.22781v1', title='Example',
            announced=self.now.date(), similarity=0.8, overlap='high', deep_status=status, deep_started=started)

    def test_full_text_comparison_recovers_from_stale_or_missing_start(self):
        for started in (None, self.now - generated.STALE - datetime.timedelta(seconds=1)):
            with self.subTest(started=started):
                match = self.match('pending', started)
                result, background = self.act('deep', match=match.pk)
                background.assert_called_once_with(scoops.compare, match.pk)
                match.refresh_from_db()
                self.assertEqual(match.deep_status, 'pending')
                self.assertGreaterEqual(match.deep_started, self.now)
                match.delete()

    def test_recent_full_text_comparison_is_not_duplicated(self):
        match = self.match('pending', self.now)
        _, background = self.act('deep', match=match.pk)
        background.assert_not_called()

    def test_completed_full_text_comparison_is_not_replaced(self):
        match = self.match('done', self.now - generated.STALE)
        _, background = self.act('deep', match=match.pk)
        background.assert_not_called()
