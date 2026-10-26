"""Tests of the staff activity page /admin/activite/ (activity.py)."""
import datetime

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from summarizer import activity
from summarizer.models import AIassistant, Generated, PaperHistory, Search


class ActivityTests(TestCase):
    def setUp(self):
        self.staff = User.objects.create_user('staff', email='staff@example.com', password='x', is_staff=True, is_superuser=True)
        self.reader = User.objects.create_user('reader', email='reader@example.com', password='x')
        PaperHistory.objects.create(arxiv_id='2410.12345', user=self.reader, lang='fr')
        AIassistant.objects.create(arxiv_id='2410.12345', query='What is the main limitation?')
        Search.objects.create(query='dark matter halo')
        Generated.objects.create(kind='refcheck', key='abc123', lang='fr', user=self.reader, status='failed', error='timeout')

    def test_the_day_counts_and_feed(self):
        numbers = {n['key']: n['value'] for n in activity.numbers(activity.today())}
        self.assertEqual((numbers['signup'], numbers['summary'], numbers['question'], numbers['search'], numbers['tool']),
                         (2, 1, 1, 1, 1))
        kinds = [e['kind'] for e in activity.feed(activity.today())]
        self.assertEqual(sorted(kinds), ['question', 'search', 'signup', 'signup', 'summary', 'tool'])
        tool = next(e for e in activity.feed(activity.today(), 'tool'))
        self.assertTrue(tool['alert'])
        self.assertEqual(tool['href'], '/fr/refcheck/abc123/')
        self.assertEqual(activity.watch()['failed'][0]['detail'], 'timeout')
        self.assertEqual(activity._paper('astro-ph--9604131v1'), '/en/paper/astro-ph--9604131v1/')
        self.assertEqual(activity._paper('abc123'), '')

    def test_members_only_hides_anonymous_actions(self):
        kinds = {e['kind'] for e in activity.feed(activity.today(), members_only=True)}
        self.assertNotIn('question', kinds)
        self.assertIn('summary', kinds)

    def test_another_day_is_empty(self):
        self.assertEqual(activity.feed(activity.today() - datetime.timedelta(days=3)), [])

    def test_staff_only_page(self):
        self.client.force_login(self.reader)
        self.assertEqual(self.client.get('/admin/activite/').status_code, 302)
        self.client.force_login(self.staff)
        page = self.client.get('/admin/activite/')
        self.assertContains(page, 'dark matter halo')
        self.assertContains(page, 'Outils en échec')
        self.assertContains(self.client.get('/admin/activite/?jour=pas-une-date&type=search'), 'dark matter halo')

    def test_the_admin_button_is_for_staff_only(self):
        self.client.force_login(self.reader)
        self.assertNotContains(self.client.get('/fr/faq', follow=True), 'class="nav-admin"')
        self.client.force_login(self.staff)
        self.assertContains(self.client.get('/fr/faq', follow=True), 'class="nav-admin"')
