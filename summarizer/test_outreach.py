import datetime
import json
from unittest import mock

from django.contrib.auth.models import User
from django.core import mail, signing
from django.test import Client, override_settings
from django.utils import timezone

from summarizer import bluesky, conversions, outreach
from summarizer.models import BlueskyPost, Letter, PageView, Prospect, SuppressedEmail
from summarizer.test_tools import ToolsTestCase

AUTHORS_HTML = '''<html><body><article><div class="ltx_authors">
<span class="ltx_creator">Ann Lee <a href="mailto:ann.lee@uni.edu">ann.lee@uni.edu</a></span>
<span class="ltx_creator">Bob Ray, contact: bray@lab.org; info@lab.org</span></div><p>Text</p></article></body></html>'''


class OutreachTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.now = timezone.now()
        self.day = self.now.date() - datetime.timedelta(days=2)

    def letter(self, topic='quantum', ids=('2101.00001',), lang='en'):
        return Letter.objects.create(topic=topic, lang=lang, day=self.day, title='Qubits this week', dek='What changed.',
                                     content=json.dumps({'sections': [], 'order': list(ids),
                                                         'papers': {i: 'T' for i in ids}}))

    def quantum_paper(self, arxiv_id='2101.00001v1', **kwargs):
        paper = self.paper(arxiv_id, **kwargs)
        paper.category = 'quant-ph'
        paper.save()
        return paper

    def test_author_emails_come_from_the_authors_block_without_generic_addresses(self):
        self.assertEqual(outreach.author_emails(AUTHORS_HTML), ['ann.lee@uni.edu', 'bray@lab.org'])
        self.assertEqual(outreach.author_emails('<html><body><p>x@y.org</p></body></html>'), [])

    def test_a_letter_paper_gets_one_pending_email_greeting_its_author(self):
        self.quantum_paper()
        self.letter()
        fetch = mock.Mock(return_value=['ann.lee@uni.edu', 'bray@lab.org'])
        self.assertEqual(outreach.prepare(self.now, fetch=fetch), 1)
        prospect = Prospect.objects.get()
        self.assertEqual((prospect.status, prospect.email, prospect.name), ('pending', 'ann.lee@uni.edu', 'Ann Lee'))
        self.assertIn('Dear Ann Lee,', prospect.body)
        self.assertIn('Quantum physics this week', prospect.subject)
        self.assertIn('/en/paper/2101.00001v1/?utm_source=outreach-author', prospect.body)
        self.assertIn('#own-paper', prospect.body)
        self.assertIn('an open source site', prospect.body)
        self.assertEqual(prospect.body.count('https://'), 1)  # one link: the paper's page and its citation alert
        outreach.prepare(self.now, fetch=fetch)  # considered once
        self.assertEqual(Prospect.objects.count(), 1)
        self.assertEqual(fetch.call_count, 1)

    def test_long_titles_are_cut_on_a_word_in_the_subject(self):
        self.assertEqual(outreach.short_title('Short title'), 'Short title')
        cut = outreach.short_title('A study of ' + 'very ' * 20 + 'long titles')
        self.assertTrue(cut.endswith('very…') and len(cut) <= 61, cut)

    def test_a_letter_paper_without_its_page_yet_is_tried_again_later(self):
        self.letter()
        fetch = mock.Mock(return_value=['ann.lee@uni.edu'])
        outreach.prepare(self.now, fetch=fetch)
        self.assertFalse(Prospect.objects.exists())
        self.quantum_paper()
        outreach.prepare(self.now, fetch=fetch)
        self.assertEqual(Prospect.objects.get().status, 'pending')

    def test_latex_dashes_become_real_dashes(self):
        paper = self.quantum_paper()
        paper.title = 'The Thomas--Fermi limit at 74--159 K --- revisited'
        paper.save()
        subject, body = outreach.write(paper, self.letter(), '')
        self.assertIn('Thomas–Fermi limit at 74–159 K — revisited', body)
        self.assertNotIn('--', subject)

    def test_astronomy_is_left_out_even_cross_listed(self):
        self.paper()  # astro-ph.EP
        self.letter(topic='climate')
        self.quantum_paper('2101.00002v1')
        self.letter(topic='space', ids=('2101.00002',))
        fetch = mock.Mock(return_value=['a@b.org'])
        outreach.prepare(self.now, fetch=fetch)
        fetch.assert_not_called()
        self.assertFalse(Prospect.objects.filter(status='pending').exists())

    def test_members_suppressed_and_already_written_addresses_are_skipped(self):
        User.objects.create_user('ann', email='Ann.Lee@uni.edu')
        SuppressedEmail.objects.create(email='bray@lab.org', reason='bounce')
        self.quantum_paper()
        self.letter()
        outreach.prepare(self.now, fetch=lambda _id: ['ann.lee@uni.edu', 'bray@lab.org'])
        self.assertEqual(Prospect.objects.get().status, 'no_email')

    def test_only_approved_emails_are_sent_at_most_per_day_with_a_stop_link(self):
        for i in range(3):
            Prospect.objects.create(email='p%d@uni.edu' % i, subject='S', body='Hello\n\nhttps://www.summarizepaper.com/x',
                                    status='approved', decided=self.now)
        Prospect.objects.create(email='wait@uni.edu', subject='S', body='B')
        with mock.patch.object(outreach, 'PER_DAY', 2):
            self.assertEqual(outreach.send_approved(self.now, 10), 2)
            self.assertEqual(outreach.send_approved(self.now, 10), 0)
        self.assertEqual(len(mail.outbox), 2)
        message = mail.outbox[0]
        self.assertIn('/en/outreach/stop/', message.body)
        self.assertIn('List-Unsubscribe', message.extra_headers)
        self.assertEqual(message.reply_to, ['contact@summarizepaper.com'])
        self.assertEqual(Prospect.objects.get(email='wait@uni.edu').status, 'pending')

    def test_the_stop_link_suppresses_the_address(self):
        token = signing.dumps('ann.lee@uni.edu', salt=outreach.OPTOUT_SALT)
        self.assertEqual(Client().get('/en/outreach/stop/%s/' % token).status_code, 200)
        self.assertTrue(SuppressedEmail.objects.filter(email='ann.lee@uni.edu').exists())
        self.assertEqual(Client().get('/en/outreach/stop/bad/').status_code, 404)

    def test_staff_approve_or_set_aside_on_the_queue_page(self):
        staff = User.objects.create_user('boss', password='x', is_staff=True, is_superuser=True)
        a = Prospect.objects.create(email='a@uni.edu', subject='Your paper', body='B')
        b = Prospect.objects.create(email='b@uni.edu', subject='Your paper', body='B')
        client = Client()
        client.force_login(staff)
        self.assertContains(client.get('/admin/prospection/'), 'a@uni.edu')
        client.post('/admin/prospection/', {'ids': [a.pk], 'action': 'approve'})
        client.post('/admin/prospection/', {'ids': [b.pk], 'action': 'reject'})
        a.refresh_from_db(), b.refresh_from_db()
        self.assertEqual((a.status, b.status), ('approved', 'rejected'))
        self.assertEqual(Client().get('/admin/prospection/').status_code, 302)  # staff only

    def test_conversions_count_visitors_by_source(self):
        PageView.objects.create(visitor='v1', path='/en/paper/x/', view='arxividpage', source='bluesky-bot', human=True)
        sources = conversions.report(7)['sources']
        self.assertEqual([(s['source'], s['visitors']) for s in sources], [('bluesky-bot', 1)])


@override_settings(BLUESKY_ACCOUNTS={'*': {'handle': 'main.bsky.social', 'password': 'p'},
                                     'ai': {'handle': 'ai.bsky.social', 'password': 'p'}})
class BlueskyTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.now = timezone.now().replace(hour=15)
        self.day = self.now.date() - datetime.timedelta(days=2)

    def letter(self, topic='quantum', ids=('2101.00001',)):
        return Letter.objects.create(topic=topic, lang='en', day=self.day, title='Qubits this week', dek='What changed.',
                                     content=json.dumps({'sections': [], 'order': list(ids), 'papers': {i: 'T' for i in ids}}))

    def test_facets_count_bytes(self):
        text = 'Café résumé #AI #Physics'
        found = bluesky.facets(text)
        self.assertEqual([f['features'][0]['tag'] for f in found], ['AI', 'Physics'])
        start = found[0]['index']['byteStart']
        self.assertEqual(text.encode()[start:found[0]['index']['byteEnd']], b'#AI')

    def test_a_theme_posts_its_letter_then_one_paper_a_day(self):
        self.paper()
        self.paper('2101.00002v1', summary='')  # not summarized: skipped
        self.letter(ids=('2101.00002', '2101.00001'))
        posted = []
        with mock.patch('summarizer.social.figure_of', return_value=None), mock.patch.object(bluesky, 'post', side_effect=lambda account, *a, **k: posted.append((account['handle'], a)) or 'at://x'):
            self.assertEqual(bluesky.run(self.now), 1)
            self.assertEqual(bluesky.run(self.now), 0)  # once a day
            self.assertEqual(posted[0][0], 'main.bsky.social')  # quantum has no account of its own
            self.assertIn('utm_source=bluesky-bot', posted[0][1][1])
            BlueskyPost.objects.update(created=self.now - datetime.timedelta(days=1))
            self.assertEqual(bluesky.run(self.now), 1)
        self.assertEqual(list(BlueskyPost.objects.order_by('created').values_list('kind', 'key')),
                         [('letter', 'quantum/%s' % self.day.isoformat()), ('paper', '2101.00001v1')])
        self.assertIn('/en/paper/2101.00001v1/', posted[1][1][1])

    def test_nothing_before_the_hour_or_without_accounts(self):
        self.paper()
        self.letter()
        with mock.patch.object(bluesky, 'post') as post:
            self.assertEqual(bluesky.run(self.now.replace(hour=5)), 0)
            with override_settings(BLUESKY_ACCOUNTS={}):
                self.assertEqual(bluesky.run(self.now), 0)
            post.assert_not_called()


@override_settings(BLUESKY_ACCOUNTS={'*': {'handle': 'main.bsky.social', 'password': 'p'}},
                   MASTODON_ACCOUNTS={'*': {'instance': 'https://fediscience.org', 'token': 't'}})
class MastodonTests(ToolsTestCase):
    def test_each_network_posts_on_its_own_with_its_utm(self):
        self.paper()
        day = timezone.now().date() - datetime.timedelta(days=2)
        Letter.objects.create(topic='quantum', lang='en', day=day, title='Qubits this week', dek='What changed.',
                              content=json.dumps({'sections': [], 'order': ['2101.00001'], 'papers': {'2101.00001': 'T'}}))
        now = timezone.now().replace(hour=15)
        sent = []
        with mock.patch('summarizer.social.figure_of', return_value=None), \
                mock.patch.object(bluesky, 'post', return_value='at://x'), \
                mock.patch('summarizer.mastodon.post', side_effect=lambda account, *a, **k: sent.append(a) or 'https://m/1'):
            self.assertEqual(bluesky.run(now), 1)
            self.assertEqual(bluesky.run(now, network='mastodon'), 1)
            self.assertEqual(bluesky.run(now, network='mastodon'), 0)
        self.assertIn('utm_source=mastodon-bot', sent[0][1])
        self.assertEqual(sorted(BlueskyPost.objects.values_list('network', flat=True)), ['bluesky', 'mastodon'])

    def test_status_text_keeps_the_link_within_the_limit(self):
        from summarizer import mastodon
        text = mastodon.status_text('word ' * 200, 'https://www.summarizepaper.com/en/x/')
        self.assertTrue(text.endswith('https://www.summarizepaper.com/en/x/'))
        self.assertLessEqual(len(text) - len('https://www.summarizepaper.com/en/x/') + 23, 500)


class OwnPaperAlertTests(ToolsTestCase):
    def test_an_author_from_our_email_is_offered_the_free_alert_once(self):
        self.paper()
        page = Client().get('/en/paper/2101.00001v1/?utm_source=outreach-author')
        self.assertContains(page, 'Is this your paper?')
        self.assertContains(page, '/en/citations/mine/2101.00001v1/')
        self.assertNotContains(Client().get('/en/paper/2101.00001v1/'), 'Is this your paper?')

    def test_following_ones_own_paper_is_free_for_one_paper(self):
        from summarizer import citations
        from summarizer.models import FollowedPaper
        self.paper()
        self.paper('2101.00002v1')
        client = Client()
        self.assertEqual(client.get('/en/citations/mine/2101.00001v1/').status_code, 302)  # to sign in
        ann = self.user()
        client.force_login(ann)
        with mock.patch.object(citations.generated, 'background'):
            response = client.get('/en/citations/mine/2101.00001v1/')
            self.assertIn('own=following', response['Location'])
            followed = FollowedPaper.objects.get(user=ann)
            self.assertTrue(followed.mine and followed.free)
            self.assertTrue(citations.alerted(ann))
            self.assertIn('own=taken', client.get('/en/citations/mine/2101.00002v1/')['Location'])
        self.assertEqual(FollowedPaper.objects.filter(user=ann).count(), 1)
        self.assertContains(client.get('/en/paper/2101.00001v1/?own=following'), 'Done: you will get an email')

    def test_premium_follows_without_using_the_free_slot(self):
        from summarizer import citations
        from summarizer.models import FollowedPaper
        self.paper()
        boss = self.user('boss', paid=True)
        client = Client()
        client.force_login(boss)
        with mock.patch.object(citations.generated, 'background'):
            client.get('/en/citations/mine/2101.00001v1/')
        self.assertFalse(FollowedPaper.objects.get(user=boss).free)

    def test_a_free_account_without_its_own_paper_gets_no_alerts(self):
        from summarizer import citations
        from summarizer.models import FollowedPaper
        ann = self.user()
        FollowedPaper.objects.create(user=ann, arxiv_id='2101.00009', pending='["x"]')
        self.assertFalse(citations.alerted(ann))
        self.assertFalse(citations.can_email(ann, set()))
