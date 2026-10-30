import datetime
from unittest import mock

from django.test import Client
from django.urls import reverse

from summarizer.models import Letter, LetterSubscription
from summarizer.test_tools import ToolsTestCase


class LetterEditionTests(ToolsTestCase):
    LATEST = datetime.date(2026, 10, 26)
    OLDER = datetime.date(2026, 10, 19)

    def setUp(self):
        super().setUp()
        for lang in ('fr', 'en'):
            Letter.objects.create(topic='space', lang=lang, day=self.LATEST,
                                  title='New astronomy', dek='New research', content='{}')
            Letter.objects.create(topic='space', lang=lang, day=self.OLDER,
                                  title='Earlier astronomy', dek='Earlier research', content='{}')
        Letter.objects.create(topic='ai', lang='en', day=self.OLDER,
                              title='English only earlier AI', content='{}')

    def test_latest_page_shows_schedule_and_archive_controls(self):
        response = Client().get('/fr/letters/')
        self.assertContains(response, 'Dernière édition : 26 octobre 2026')
        self.assertContains(response, 'lundi 2 novembre 2026')
        self.assertContains(response, 'Afficher l’édition')
        self.assertContains(response, '?edition=2026-10-19')
        self.assertContains(response, 'New astronomy')
        self.assertNotContains(response, 'Earlier astronomy')
        self.assertEqual(response.context['editions'], [self.LATEST, self.OLDER])

    def test_archive_is_exact_not_a_mix_with_newer_or_other_language_letters(self):
        response = Client().get('/fr/letters/', {'edition': '2026-10-19'})
        self.assertContains(response, 'Earlier astronomy')
        self.assertNotContains(response, 'New astronomy')
        self.assertNotContains(response, 'English only earlier AI')
        self.assertContains(response, '1 lettre disponible')
        self.assertContains(response, 'Aucune lettre n’a été publiée')
        self.assertNotContains(response, 'La première lettre paraîtra')
        self.assertContains(response, '?edition=2026-10-26')
        self.assertContains(response, 'Revenir aux dernières lettres')
        self.assertIsNone(response.context['older_edition'])

    def test_invalid_or_unpublished_edition_is_not_generated(self):
        with mock.patch('summarizer.letters.write') as write, mock.patch('summarizer.letters.send') as send:
            for day in ('invalid', '2026-10-99', '2026-11-02'):
                self.assertEqual(Client().get('/fr/letters/', {'edition': day}).status_code, 404)
            self.assertEqual(Client().get('/fr/letters/', {'edition': '2026-10-19'}).status_code, 200)
            write.assert_not_called()
            send.assert_not_called()

    def test_existing_subscriptions_are_not_affected_by_archive_reads(self):
        reader = self.user('reader')
        LetterSubscription.objects.create(user=reader, topic='space', lang='fr', active=True)
        self.client_for(reader).get('/fr/letters/', {'edition': '2026-10-19'})
        self.assertEqual(LetterSubscription.objects.filter(user=reader, active=True).count(), 1)
        self.assertEqual(LetterSubscription.objects.get(user=reader).lang, 'fr')

    def test_all_old_editions_remain_accessible_and_empty_site_renders(self):
        old = datetime.date(2023, 1, 2)
        Letter.objects.create(topic='space', lang='fr', day=old, title='Old archive', content='{}')
        self.assertContains(Client().get('/fr/letters/', {'edition': old.isoformat()}), 'Old archive')
        Letter.objects.all().delete()
        response = Client().get('/fr/letters/')
        self.assertEqual(response.status_code, 200)
        self.assertNotContains(response, 'id="letter-edition"')
