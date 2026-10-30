from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings

from . import arxiv_taxonomy, digest, email_updates
from .models import DigestSubscription, LetterSubscription


@override_settings(EMAIL_BACKEND='django.core.mail.backends.locmem.EmailBackend')
class QuickEmailSignupTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user('quick-reader', 'reader@example.com', 'pass')
        self.data = {'topic': 'ai', 'products': ['digest', 'letters']}

    def login(self):
        self.client.force_login(self.user)

    def test_home_has_visible_bilingual_shortcut_and_optin(self):
        for lang in ('en', 'fr'):
            response = self.client.get('/%s/' % lang)
            self.assertContains(response, 'href="#email-updates"')
            self.assertContains(response, 'action="/%s/email-updates/"' % lang)
            self.assertContains(response, 'name="products" value="digest" checked')
            self.assertContains(response, 'name="products" value="letters" checked')

    def test_anonymous_choices_survive_signup_without_activating_by_get(self):
        response = self.client.post('/fr/email-updates/', self.data)
        self.assertEqual(response.status_code, 302)
        self.assertEqual(urlparse(response.url).path, '/fr/register/')
        next_url = parse_qs(urlparse(response.url).query)['next'][0]
        self.assertEqual(parse_qs(urlparse(next_url).query)['products'], ['digest', 'letters'])
        self.login()
        response = self.client.get(next_url)
        self.assertContains(response, '<option value="ai" selected>')
        self.assertFalse(DigestSubscription.objects.exists())
        self.assertFalse(LetterSubscription.objects.exists())

    @patch('django.core.mail.EmailMessage.send')
    def test_both_formats_activate_in_page_language_without_immediate_email(self, send):
        self.login()
        for _ in range(2):
            response = self.client.post('/fr/email-updates/', self.data, follow=True)
            self.assertContains(response, 'Vos abonnements sont actifs')
        subscription = DigestSubscription.objects.get()
        self.assertEqual(subscription.user, self.user)
        self.assertEqual(subscription.lang, 'fr')
        self.assertEqual(subscription.frequency, 'daily')
        self.assertEqual(subscription.send_hour, 7)
        self.assertEqual(subscription.timezone, 'Europe/Paris')
        self.assertEqual(subscription.category_list(), email_updates.STARTERS['ai'].split(','))
        letter = LetterSubscription.objects.get()
        self.assertTrue(letter.active)
        self.assertEqual((letter.topic, letter.lang), ('ai', 'fr'))
        send.assert_not_called()

    def test_existing_digest_preferences_and_letter_delivery_marker_are_preserved(self):
        self.login()
        subscription = DigestSubscription.objects.create(user=self.user, categories=email_updates.STARTERS['ai'],
            frequency='weekly', send_hour=13, keywords='attention', lang='en')
        import datetime
        previous = datetime.date(2026, 10, 26)
        letter = LetterSubscription.objects.create(user=self.user, topic='ai', last_letter=previous, lang='en')
        self.assertEqual(self.client.post('/fr/email-updates/', self.data).status_code, 302)
        subscription.refresh_from_db(); letter.refresh_from_db()
        self.assertEqual((subscription.frequency, subscription.send_hour, subscription.keywords, subscription.lang),
                         ('weekly', 13, 'attention', 'en'))
        self.assertEqual((letter.last_letter, letter.lang), (previous, 'en'))
        self.assertEqual(DigestSubscription.objects.count(), 1)

    def test_free_digest_limit_does_not_overwrite_or_partially_subscribe(self):
        self.login()
        old = DigestSubscription.objects.create(user=self.user, categories='astro-ph.EP')
        response = self.client.post('/fr/email-updates/', self.data)
        self.assertEqual(response.status_code, 409)
        old.refresh_from_db()
        self.assertEqual(old.categories, 'astro-ph.EP')
        self.assertFalse(LetterSubscription.objects.exists())

    def test_free_letter_limit_does_not_replace_or_partially_subscribe(self):
        self.login()
        old = LetterSubscription.objects.create(user=self.user, topic='space')
        self.assertEqual(self.client.post('/en/email-updates/', self.data).status_code, 409)
        old.refresh_from_db()
        self.assertTrue(old.active)
        self.assertFalse(DigestSubscription.objects.exists())

    def test_single_format_and_reactivation_work(self):
        self.login()
        old = LetterSubscription.objects.create(user=self.user, topic='ai', active=False)
        self.assertEqual(self.client.post('/fr/email-updates/', {'topic': 'ai', 'products': 'letters'}).status_code, 302)
        old.refresh_from_db()
        self.assertTrue(old.active)
        self.assertEqual(old.lang, 'fr')
        self.assertFalse(DigestSubscription.objects.exists())

    def test_invalid_selection_and_csrf_never_subscribe(self):
        self.login()
        for data in ({}, {'topic': 'bad', 'products': 'digest'}, {'topic': 'ai', 'products': 'unexpected'}):
            self.assertEqual(self.client.post('/en/email-updates/', data).status_code, 400)
        from django.test import Client
        client = Client(enforce_csrf_checks=True); client.force_login(self.user)
        self.assertEqual(client.post('/en/email-updates/', self.data).status_code, 403)
        self.assertFalse(DigestSubscription.objects.exists())
        self.assertFalse(LetterSubscription.objects.exists())

    def test_all_starter_fields_are_real_and_fit_free_limits(self):
        known = set(arxiv_taxonomy.NAMES) | set(digest.BIO_CODES)
        for categories in email_updates.STARTERS.values():
            selected = categories.split(',')
            self.assertLessEqual(len(selected), digest.FREE['categories'])
            self.assertTrue(set(selected) <= known, set(selected) - known)
