import os
from django.contrib.auth.models import User
from django.contrib.sessions.backends.db import SessionStore
from django.test import TestCase, RequestFactory
from django.utils import timezone

from summarizer import access
from summarizer.models import AIassistant, PaperHistory, SiteSettings, UserPlan


class ClientIpTests(TestCase):
    def ip(self, **meta):
        return access.client_ip(RequestFactory().get('/', REMOTE_ADDR='10.0.0.1', **meta))

    def test_no_proxy_uses_remote_addr(self):
        self.assertEqual(self.ip(), '10.0.0.1')

    def test_forged_first_hop_is_ignored(self):
        # the client sent "1.2.3.4", Heroku appended the real address
        self.assertEqual(self.ip(HTTP_X_FORWARDED_FOR='1.2.3.4, 81.0.0.9'), '81.0.0.9')

    def test_cloudflare_edge_trusts_cf_connecting_ip(self):
        self.assertEqual(
            self.ip(HTTP_X_FORWARDED_FOR='81.0.0.9, 162.158.1.1', HTTP_CF_CONNECTING_IP='81.0.0.9'),
            '81.0.0.9')

    def test_cf_header_ignored_when_not_from_cloudflare(self):
        self.assertEqual(
            self.ip(HTTP_X_FORWARDED_FOR='81.0.0.9', HTTP_CF_CONNECTING_IP='1.2.3.4'),
            '81.0.0.9')

    def test_scope_headers(self):
        scope = {'headers': [(b'x-forwarded-for', b'1.2.3.4, 81.0.0.9')], 'client': ('10.0.0.1', 1)}
        self.assertEqual(access.client_ip_from_scope(scope), '81.0.0.9')


class PaidTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('u', password='x')

    def test_anonymous_and_no_plan(self):
        self.assertFalse(access.is_paid(None))
        self.assertFalse(access.is_paid(self.user))

    def test_paid_plan(self):
        UserPlan.objects.create(user=self.user, paid=True)
        self.assertTrue(access.is_paid(self.user))

    def test_expired_plan(self):
        UserPlan.objects.create(user=self.user, paid=True, expires=timezone.now() - timezone.timedelta(days=1))
        self.assertFalse(access.is_paid(self.user))


class QuotaTests(TestCase):
    def test_run_quota(self):
        ip = access.hash_ip('81.0.0.9')
        self.assertEqual(access.run_quota(None, ip), (True, None))
        PaperHistory.objects.create(arxiv_id='2101.00001v1', ip_address=ip)
        self.assertEqual(access.run_quota(None, ip), (False, 'robot'))
        PaperHistory.objects.filter(ip_address=ip).update(created=timezone.now() - timezone.timedelta(minutes=5))
        for i in range(access.runs_per_day() - 1):
            PaperHistory.objects.create(arxiv_id='x', ip_address=ip)
        PaperHistory.objects.filter(ip_address=ip).update(created=timezone.now() - timezone.timedelta(minutes=5))
        self.assertEqual(access.run_quota(None, ip), (False, 'toomany'))

    def test_chat_quota_anonymous(self):
        ip = access.hash_ip('81.0.0.9')
        start = access.chat_quota_left(None, ip)
        AIassistant.objects.create(arxiv_id='a', query='q', ip_address=ip)
        self.assertEqual(access.chat_quota_left(None, ip), start - 1)


class RunTokenTests(TestCase):
    def test_token_is_single_use_and_scoped(self):
        session = SessionStore()
        access.grant_run(session, '2101.00001v1', 'run')
        self.assertFalse(access.consume_run(session, '2101.00001v1', 'close'))
        self.assertFalse(access.consume_run(session, '2101.00002v1', 'run'))
        self.assertTrue(access.consume_run(session, '2101.00001v1', 'run'))
        self.assertFalse(access.consume_run(session, '2101.00001v1', 'run'))


from channels.routing import URLRouter
from channels.testing import WebsocketCommunicator
from django.contrib.auth.models import AnonymousUser
from django.core.cache import cache
from django.test import TransactionTestCase, override_settings
from asgiref.sync import sync_to_async


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
class ConsumerTests(TransactionTestCase):
    async def connect(self, session):
        from summarizer.routing import websocket_urlpatterns
        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), '/ws/arxivid/2101.00001v1/en/',
                                     headers=[(b'x-forwarded-for', b'81.0.0.9')])
        comm.scope['user'] = AnonymousUser()
        comm.scope['session'] = session
        connected, _ = await comm.connect()
        self.assertTrue(connected)
        return comm

    async def test_forged_start_does_nothing(self):
        cache.delete('ar_2101.00001v1')
        comm = await self.connect(SessionStore())
        await comm.send_json_to({'command': 'start_background_task', 'paid': True})
        self.assertTrue(await comm.receive_nothing(timeout=0.3))
        self.assertFalse(cache.get('ar_2101.00001v1'))
        await comm.disconnect()

    async def test_chat_over_quota_is_refused(self):
        ip = access.hash_ip('81.0.0.9')
        for i in range(access.chat_messages_per_day()['anonymous']):
            await sync_to_async(AIassistant.objects.create)(arxiv_id='a', query='q', ip_address=ip)
        comm = await self.connect(SessionStore())
        await comm.send_json_to({'message': 'hello', 'user': 'admin', 'ip': 'x', 'paid': True})
        response = await comm.receive_json_from(timeout=2)
        self.assertIn('daily question limit', response['message'])
        await comm.disconnect()


import hashlib as _hashlib
import hmac
import json as _json
import time as _time
from unittest import mock

from django.test import Client


@override_settings(STRIPE_SECRET_KEY='sk_test_x', STRIPE_WEBHOOK_SECRET='whsec_test',
                   STRIPE_PRICE_MONTHLY='price_m', STRIPE_PRICE_YEARLY='')
class BillingTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('payer', email='p@example.com', password='x')

    def post_event(self, event):
        payload = _json.dumps(event)
        ts = int(_time.time())
        sig = hmac.new(b'whsec_test', ('%d.%s' % (ts, payload)).encode(), _hashlib.sha256).hexdigest()
        return Client().post('/stripe/webhook/', payload, content_type='application/json',
                             HTTP_STRIPE_SIGNATURE='t=%d,v1=%s' % (ts, sig))

    def subscription(self, status, end_offset_days=30):
        end = int(_time.time()) + end_offset_days * 86400
        return {'id': 'sub_1', 'object': 'subscription', 'customer': 'cus_1', 'status': status,
                'metadata': {'user_id': str(self.user.id)},
                'items': {'data': [{'current_period_end': end}]}}

    def event(self, type_, obj):
        return {'id': 'evt_1', 'object': 'event', 'type': type_, 'data': {'object': obj}}

    def test_bad_signature_rejected(self):
        r = Client().post('/stripe/webhook/', '{}', content_type='application/json', HTTP_STRIPE_SIGNATURE='t=1,v1=bad')
        self.assertEqual(r.status_code, 400)

    def test_subscription_lifecycle(self):
        self.assertEqual(self.post_event(self.event('customer.subscription.created', self.subscription('active'))).status_code, 200)
        self.assertTrue(access.is_paid(self.user))
        plan = UserPlan.objects.get(user=self.user)
        self.assertEqual((plan.stripe_customer_id, plan.stripe_subscription_id), ('cus_1', 'sub_1'))

        cancelled = self.subscription('active')
        cancelled['cancel_at_period_end'] = True
        self.post_event(self.event('customer.subscription.updated', cancelled))
        self.assertTrue(access.is_paid(self.user))
        self.assertTrue(UserPlan.objects.get(user=self.user).canceling)
        c = Client(); c.force_login(self.user)
        with mock.patch('summarizer.billing.price_label', return_value=None):
            self.assertContains(c.get('/fr/premium/'), 'Premium reste actif jusqu')

        self.post_event(self.event('customer.subscription.deleted', self.subscription('canceled')))
        self.assertFalse(access.is_paid(self.user))
        self.assertEqual(UserPlan.objects.filter(user=self.user).count(), 1)

    def test_checkout_completed(self):
        session = {'id': 'cs_1', 'object': 'checkout.session', 'mode': 'subscription',
                   'client_reference_id': str(self.user.id), 'subscription': 'sub_1'}
        with mock.patch('stripe.Subscription.retrieve', return_value=self.subscription('active')):
            self.post_event(self.event('checkout.session.completed', session))
        self.assertTrue(access.is_paid(self.user))

    def test_premium_page(self):
        with mock.patch('summarizer.billing.price_label', return_value='9.00 EUR / month'):
            r = Client().get('/en/premium/')
            self.assertContains(r, 'Create a free account to subscribe')
            c = Client(); c.force_login(self.user)
            r = c.get('/en/premium/')
            self.assertContains(r, '9.00 EUR / month')
            self.assertContains(r, 'Subscribe monthly')

    def test_checkout_redirects_to_stripe(self):
        c = Client(); c.force_login(self.user)
        with mock.patch('stripe.checkout.Session.create', return_value=mock.Mock(url='https://checkout.stripe.com/x')) as create:
            r = c.post('/en/premium/checkout/', {'period': 'monthly'})
        self.assertEqual(r['Location'], 'https://checkout.stripe.com/x')
        kwargs = create.call_args.kwargs
        self.assertEqual(kwargs['client_reference_id'], str(self.user.id))
        self.assertEqual(kwargs['line_items'][0]['price'], 'price_m')
        self.assertEqual(kwargs['success_url'], 'http://testserver/en/premium/?success=1')
        self.assertEqual(kwargs['cancel_url'], 'http://testserver/en/premium/')

    def test_portal_return_url(self):
        UserPlan.objects.create(user=self.user, paid=True, stripe_customer_id='cus_1')
        c = Client(); c.force_login(self.user)
        with override_settings(SITE_URL='https://www.summarizepaper.com'), \
             mock.patch('stripe.billing_portal.Session.create', return_value=mock.Mock(url='https://billing.stripe.com/x')) as create:
            c.post('/fr/premium/portal/')
        self.assertEqual(create.call_args.kwargs['return_url'], 'https://www.summarizepaper.com/fr/premium/')


from django.core import mail

from summarizer import digest
from summarizer.models import DigestDelivery, DigestItem, DigestSubscription

FEED = '''<?xml version='1.0' encoding='UTF-8'?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2610.22354v1</id>
    <title>LSST and Keck characterization of
      outer Main Belt comet</title>
    <summary>We characterize a comet with an exoplanet-free abstract.</summary>
    <category term="astro-ph.EP"/><published>2026-10-15T17:58:04Z</published>
    <author><name>Bryce Bolin</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/hep-th/9901001v2</id>
    <title>Old style id</title><summary>Strings.</summary>
    <category term="hep-th"/><published>2026-10-16T10:00:00Z</published>
    <author><name>A</name></author>
  </entry>
</feed>'''


import datetime


class DigestTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user('reader', email='r@example.com', password='x')

    def papers(self):
        return {p['arxiv_id']: p for p in digest.parse_feed(FEED)}

    def test_parse_feed(self):
        papers = self.papers()
        self.assertEqual(set(papers), {'2610.22354v1', 'hep-th--9901001v2'})
        self.assertEqual(papers['2610.22354v1']['title'], 'LSST and Keck characterization of outer Main Belt comet')
        self.assertEqual(papers['2610.22354v1']['categories'], ['astro-ph.EP'])

    def test_clean_categories(self):
        self.assertEqual(digest.clean_categories(['cs.LG', ' cs.LG', 'bad cat!', 'quant-ph', '']), ['cs.LG', 'quant-ph'])

    def test_select_by_category_keywords_and_history(self):
        sub = DigestSubscription.objects.create(user=self.user, categories='astro-ph.EP')
        self.assertEqual([p['arxiv_id'] for p in digest.select_papers(sub, self.papers(), set())], ['2610.22354v1'])
        self.assertEqual(digest.select_papers(sub, self.papers(), {'2610.22354v1'}), [])
        sub.keywords = 'black hole'
        self.assertEqual(digest.select_papers(sub, self.papers(), set()), [])
        sub.keywords = 'black hole, comet'
        self.assertEqual(len(digest.select_papers(sub, self.papers(), set())), 1)

    def test_run_sends_once_and_falls_back_to_abstract(self):
        DigestSubscription.objects.create(user=self.user, categories='astro-ph.EP,hep-th', lang='fr')
        with mock.patch('summarizer.digest.fetch_all', return_value=self.papers()), \
             mock.patch('summarizer.digest.write_blurbs', return_value={'2610.22354v1': 'Une comète active.'}):
            morning = datetime.datetime(2026, 10, 19, 6, 0, tzinfo=datetime.timezone.utc)  # 08:00 in Paris
            self.assertEqual(digest.run(now=morning), 1)
            self.assertEqual(digest.run(now=morning), 0)  # already sent today
        self.assertEqual(len(mail.outbox), 1)
        message = mail.outbox[0]
        self.assertEqual(message.to, ['r@example.com'])
        self.assertIn('Une comète active.', message.body)
        self.assertIn('Strings.', message.body)  # blurb missing from the model answer -> abstract
        self.assertIn('/fr/arxiv-id/hep-th--9901001v2/?autorun=1', message.alternatives[0][0])
        self.assertIn('List-Unsubscribe', message.extra_headers)
        self.assertEqual(DigestDelivery.objects.count(), 2)
        self.assertEqual(DigestItem.objects.filter(lang='fr').count(), 2)

    def test_schedule_follows_local_time(self):
        utc = datetime.timezone.utc
        at = lambda d, h: datetime.datetime(2026, 10, d, h, 0, tzinfo=utc)
        sub = DigestSubscription(user=self.user, categories='cs.LG', send_hour=7, timezone='America/Los_Angeles')
        self.assertFalse(sub.is_due(at(19, 6)))   # 23:00 in LA: too late, wait for tomorrow 07:00
        self.assertTrue(sub.is_due(at(19, 14)))   # 07:00 in LA
        self.assertTrue(sub.is_due(at(19, 16)))   # 09:00: a missed run is caught up
        self.assertFalse(sub.is_due(at(19, 17)))  # 10:00: too late
        sub.last_sent = at(19, 14)
        self.assertFalse(sub.is_due(at(19, 15)))  # once per day
        self.assertTrue(sub.is_due(at(20, 14)))   # next day
        late = DigestSubscription(user=self.user, categories='cs.LG', send_hour=23, timezone='UTC', last_sent=at(19, 23))
        self.assertFalse(late.is_due(at(20, 1)))  # 01:00 belongs to yesterday's 23:00 slot
        self.assertTrue(late.is_due(at(20, 23)))
        sub.timezone = 'Not/AZone'
        self.assertTrue(sub.is_due(at(20, 8)))    # unknown zone falls back to UTC

    def test_no_digest_on_weekends(self):
        utc = datetime.timezone.utc
        at = lambda d, h: datetime.datetime(2026, 10, d, h, 0, tzinfo=utc)
        sub = DigestSubscription(user=self.user, categories='cs.LG', send_hour=7, timezone='Europe/Paris')
        self.assertTrue(sub.is_due(at(23, 5)))    # Friday 07:00 in Paris
        self.assertFalse(sub.is_due(at(24, 5)))   # Saturday
        self.assertFalse(sub.is_due(at(25, 6)))   # Sunday
        self.assertTrue(sub.is_due(at(26, 6)))    # Monday
        tokyo = DigestSubscription(user=self.user, categories='cs.LG', send_hour=7, timezone='Asia/Tokyo')
        self.assertFalse(tokyo.is_due(at(23, 22)))  # Friday 22:00 UTC is already Saturday in Tokyo
        self.assertTrue(tokyo.is_due(at(25, 22)))   # Monday 07:00 in Tokyo

    def test_run_skips_subscribers_not_due(self):
        DigestSubscription.objects.create(user=self.user, categories='astro-ph.EP', send_hour=7, timezone='Asia/Tokyo')
        with mock.patch('summarizer.digest.fetch_all') as fetch:
            # 21:00 UTC is 06:00 in Tokyo: not yet, and no arXiv call when nobody is due
            self.assertEqual(digest.run(now=datetime.datetime(2026, 10, 19, 21, 0, tzinfo=datetime.timezone.utc)), 0)
        fetch.assert_not_called()
    def test_unsubscribe_link(self):
        sub = DigestSubscription.objects.create(user=self.user, categories='cs.LG')
        url = '/en/digest/unsubscribe/%s/' % digest.unsubscribe_token(sub)
        self.assertContains(Client().get(url), 'Unsubscribe from the daily digest?')
        Client(enforce_csrf_checks=True).post(url)
        sub.refresh_from_db()
        self.assertFalse(sub.active)
        self.assertContains(Client().get('/en/digest/unsubscribe/forged/'), 'invalid')

    def test_digest_page_saves_subscription(self):
        c = Client(); c.force_login(self.user)
        self.assertContains(c.get('/en/digest/'), 'Start my digest')
        r = c.post('/en/digest/', {'categories': ['cs.LG', 'quant-ph'], 'other': 'math.AP', 'keywords': 'transformer ,  ', 'lang': 'fr',
                                   'send_hour': '18', 'timezone': 'America/New_York'})
        self.assertEqual(r.status_code, 302)
        sub = DigestSubscription.objects.get(user=self.user)
        self.assertEqual((sub.categories, sub.keywords, sub.lang), ('cs.LG,quant-ph,math.AP', 'transformer', 'fr'))
        self.assertEqual((sub.send_hour, sub.timezone), (18, 'America/New_York'))
        c.post('/en/digest/', {'categories': ['cs.LG'], 'send_hour': '99', 'timezone': '<script>'})
        sub.refresh_from_db()
        self.assertEqual((sub.send_hour, sub.timezone), (23, 'America/New_York'))
        self.assertContains(c.post('/en/digest/', {'other': ''}), 'Choose at least one category.')


from summarizer import assistant, paper_source
from summarizer.models import ArxivPaper, PaperText, SummaryPaper


class FakeStreamResponse:
    status = 200

    def __init__(self, lines):
        self._lines = lines

    @property
    def content(self):
        async def gen():
            for line in self._lines:
                yield line
        return gen()

    async def text(self):
        return ''

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class FakeSession:
    last_payload = None

    def __init__(self, *args, **kwargs):
        pass

    def post(self, url, json=None, headers=None):
        FakeSession.last_payload = json
        chunks = [{'choices': [{'delta': {'content': piece}}]} for piece in ('The ', 'answer.')]
        lines = [b': keep-alive\n'] + [('data: %s\n' % _json.dumps(c)).encode() for c in chunks] + [b'data: [DONE]\n']
        return FakeStreamResponse(lines)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class PaperSourceTests(TestCase):
    def test_merge_chunks_drops_overlap(self):
        a = 'Para one.\n\nPara two is long enough to be a probe for the overlap detection.'
        b = 'Para two is long enough to be a probe for the overlap detection.\n\nPara three.'
        self.assertEqual(paper_source._merge_chunks([a, b]), a + '\n\nPara three.')

    def test_html_to_text_keeps_latex_and_drops_references(self):
        html = ('<article><h2>1 Introduction</h2><p>Energy <math alttext="E=mc^{2}"><mi>E</mi></math> holds.</p>'
                '<section class="ltx_bibliography"><p>[1] Someone 1905</p></section></article>')
        text = paper_source.html_to_text(html)
        self.assertIn('## 1 Introduction', text)
        self.assertIn('$E=mc^{2}$', text)
        self.assertNotIn('Someone', text)

    def test_store_text_strips_api_answer(self):
        paper_source.store_text('1', '<?xml version="1.0"?><feed>x</feed>    Real text', 'pdf')
        self.assertEqual(PaperText.objects.get(arxiv_id='1').text, 'Real text')

    def test_store_text_drops_nul_bytes(self):
        paper_source.store_text('2', 'Real\x00 text', 'index')
        self.assertEqual(PaperText.objects.get(arxiv_id='2').text, 'Real text')


class AssistantTests(TransactionTestCase):
    def setUp(self):
        self.open = ArxivPaper.objects.create(arxiv_id='2101.00001v1', title='Open\n  paper', abstract='Abs open.',
                                              license='http://creativecommons.org/licenses/by/4.0/')
        self.closed = ArxivPaper.objects.create(arxiv_id='2101.00002v1', title='Closed', abstract='Abs closed.',
                                                license='http://arxiv.org/licenses/nonexclusive-distrib/1.0/')
        PaperText.objects.create(arxiv_id='2101.00001v1', text='FULL TEXT OF THE OPEN PAPER', source='html')
        PaperText.objects.create(arxiv_id='2101.00002v1', text='FULL TEXT OF THE CLOSED PAPER', source='pdf')

    def test_context_respects_licenses(self):
        context = assistant.build_context(['2101.00001v1', '2101.00002v1'])
        self.assertIn('FULL TEXT OF THE OPEN PAPER', context)
        self.assertNotIn('FULL TEXT OF THE CLOSED PAPER', context)
        self.assertIn('Abs closed.', context)
        self.assertIn('Title: Open paper', context)

    def test_answer_streams_stores_and_reuses_history(self):
        import asyncio
        deltas = []

        async def on_delta(d):
            deltas.append(d)

        with mock.patch('summarizer.assistant.aiohttp.ClientSession', FakeSession):
            text = asyncio.run(assistant.answer(['2101.00001v1'], 'en', 'What is it?', False, hashed_ip='h', on_delta=on_delta))
            self.assertEqual(text, 'The answer.')
            self.assertEqual(deltas, ['The ', 'answer.'])
            self.assertEqual(AIassistant.objects.get().response, 'The answer.')
            asyncio.run(assistant.answer(['2101.00001v1'], 'en', 'And then?', True, hashed_ip='h'))
        messages = FakeSession.last_payload['messages']
        self.assertEqual([m['role'] for m in messages], ['system', 'user', 'assistant', 'user'])
        self.assertEqual(FakeSession.last_payload['model'], SiteSettings.current().model_paid)
        self.assertTrue(FakeSession.last_payload['stream'])


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
class ConsumerChatTests(TransactionTestCase):
    async def test_chat_streams_deltas_then_message(self):
        from summarizer.routing import websocket_urlpatterns

        async def fake_answer(arxiv_ids, lang, question, paid, user=None, hashed_ip=None, on_delta=None, conversation_id=None):
            self.assertEqual((arxiv_ids, paid, conversation_id), (['2101.00001v1'], False, '2101.00001v1'))
            await on_delta('Hel')
            await on_delta('lo')
            return 'Hello'

        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), '/ws/arxivid/2101.00001v1/en/')
        comm.scope['user'] = AnonymousUser()
        comm.scope['session'] = SessionStore()
        await comm.connect()
        with mock.patch('summarizer.assistant.answer', fake_answer):
            await comm.send_json_to({'message': 'hi', 'paid': True})
            self.assertEqual(await comm.receive_json_from(), {'type': 'chat_delta', 'delta': 'Hel'})
            self.assertEqual(await comm.receive_json_from(), {'type': 'chat_delta', 'delta': 'lo'})
            self.assertEqual(await comm.receive_json_from(), {'message': 'Hello'})
        await comm.disconnect()


class LoginTests(TestCase):
    def test_inactive_account_gets_a_message_not_a_500(self):
        User.objects.create_user('sleepy', password='pw-123456', is_active=False)
        r = Client().post('/en/login/', {'username': 'sleepy', 'password': 'pw-123456'})
        self.assertEqual(r.status_code, 200)

    def test_login_follows_next(self):
        User.objects.create_user('awake', password='pw-123456')
        r = Client().post('/en/login/?next=/en/premium/', {'username': 'awake', 'password': 'pw-123456'})
        self.assertEqual(r['Location'], '/en/premium/')


@override_settings(CHANNEL_LAYERS={'default': {'BACKEND': 'channels.layers.InMemoryChannelLayer'}})
class ConsumerFailureTests(TransactionTestCase):
    async def test_failed_run_releases_lock_and_tells_the_user(self):
        from summarizer.routing import websocket_urlpatterns
        session = SessionStore()
        access.grant_run(session, '2101.00009v1', 'run')
        comm = WebsocketCommunicator(URLRouter(websocket_urlpatterns), '/ws/arxivid/2101.00009v1/en/')
        comm.scope['user'] = AnonymousUser()
        comm.scope['session'] = session
        await comm.connect()
        with mock.patch('summarizer.utils.get_arxiv_metadata', side_effect=RuntimeError('arXiv down')):
            await comm.send_json_to({'command': 'start_background_task'})
            last = ''
            for _ in range(5):
                last = (await comm.receive_json_from(timeout=5))['message']['loading_message']
                if 'Something went wrong' in last:
                    break
        self.assertIn('Something went wrong', last)
        self.assertFalse(cache.get('ar_2101.00009v1'))
        await comm.disconnect()


class BibtexTests(TestCase):
    def test_bibtex_entry(self):
        from summarizer.models import Author, PaperAuthor
        import datetime
        paper = ArxivPaper.objects.create(arxiv_id='2303.11544v1', title='A Constraint on the\n Amount of Hydrogen',
                                          category='astro-ph.EP', published_arxiv=datetime.date(2023, 3, 21))
        for i, name in enumerate(['Kazunari Iwasaki', 'Hiroshi Kobayashi']):
            PaperAuthor.objects.create(paper=paper, author=Author.objects.create(name=name, affiliation=''), author_order=i)
        body = Client().get('/en/bibtex/2303.11544v1/').content.decode()
        self.assertIn('@misc{iwasaki2023constraint,', body)
        self.assertIn('author={Kazunari Iwasaki and Hiroshi Kobayashi},', body)
        self.assertIn('title={A Constraint on the Amount of Hydrogen},', body)
        self.assertIn('eprint={2303.11544},', body)
        self.assertIn('url={https://arxiv.org/abs/2303.11544}', body)


from summarizer import utils as summarizer_utils


class GeneratedTextCleanupTests(TestCase):
    def test_lay_summary_markdown_from_production(self):
        raw = ('**Summary:**\n1. Some giant planets are stretched.\n2. Measuring it helps.\n\n**\n\n\n'
               'Definitions:**\n- **Giant planet:** A very large planet.')
        self.assertEqual(summarizer_utils.clean_lay_summary(raw),
                         'Summary:\n1. Some giant planets are stretched.\n2. Measuring it helps.\n\n'
                         'Definitions:\n- Giant planet: A very large planet.')

    def test_lay_summary_french(self):
        out = summarizer_utils.clean_lay_summary('Résumé :\n1. Une planète.\nDéfinitions :\n- Planète : un astre.')
        self.assertIn('Une planète.\n\nDéfinitions', out)

    def test_blog_wraps_paragraphs_and_strips_unsafe_html(self):
        raw = ('<h2>Title</h2>\n\nFirst paragraph.\n\n<ul>\n<li>One</li>\n</ul>\n\n'
               '<p onclick="x()">Second <script>alert(1)</script><a href="javascript:1">link</a>.</p>')
        html = summarizer_utils.clean_blog_html(raw)
        self.assertIn('<h2>Title</h2>', html)
        self.assertIn('<p>First paragraph.</p>', html)
        self.assertIn('<li>One</li>', html)
        self.assertNotIn('onclick', html)
        self.assertNotIn('<script', html)
        self.assertNotIn('alert', html)
        self.assertNotIn('<a', html)

    def test_truncated_blog_drops_unfinished_block(self):
        raw = '<h2>A</h2>\n\nComplete paragraph.\n\n<h2>B</h2>\n\nThe target ranking offers a practical starting point'
        html = summarizer_utils.clean_blog_html(raw, truncated=True)
        self.assertTrue(html.endswith('<p>Complete paragraph.</p>'), html)


class AutorunTests(TestCase):
    def setUp(self):
        self.paper = ArxivPaper.objects.create(arxiv_id='2101.00005v1', title='T', abstract='A',
                                               license='http://creativecommons.org/licenses/by/4.0/')

    def test_autorun_script_only_without_summary(self):
        marker = 'arriving from the daily digest'
        self.assertContains(Client().get('/en/arxiv-id/2101.00005v1/?autorun=1'), marker)
        self.assertNotContains(Client().get('/en/arxiv-id/2101.00005v1/'), marker)
        SummaryPaper.objects.create(paper=self.paper, lang='en', summary='Done.')
        self.assertNotContains(Client().get('/en/arxiv-id/2101.00005v1/?autorun=1'), marker)
        # a summary in another language does not count
        self.assertContains(Client().get('/fr/arxiv-id/2101.00005v1/?autorun=1'), marker)


class DigestRateLimitTests(TestCase):
    def test_sends_at_most_the_hourly_cap(self):
        papers = {p['arxiv_id']: p for p in digest.parse_feed(FEED)}
        for i in range(digest.MAX_EMAILS_PER_RUN + 3):
            u = User.objects.create_user('r%d' % i, email='r%d@example.com' % i, password='x')
            DigestSubscription.objects.create(user=u, categories='astro-ph.EP', timezone='UTC', send_hour=7)
        at = lambda h: datetime.datetime(2026, 10, 19, h, 0, tzinfo=datetime.timezone.utc)
        with mock.patch('summarizer.digest.fetch_all', return_value=papers), \
             mock.patch('summarizer.digest.write_blurbs', return_value={}):
            self.assertEqual(digest.run(now=at(7)), digest.MAX_EMAILS_PER_RUN)
            self.assertEqual(digest.run(now=at(8)), 3)  # the rest go out the next hour
        self.assertEqual(len(mail.outbox), digest.MAX_EMAILS_PER_RUN + 3)


class KeyPointTests(TestCase):
    RAW = ('Voici les points clés :\n\n'
           '- **Nature de l’étude :** il s’agit d’une étude de faisabilité.\n'
           '- **Performances envisagées :** 200 à 300 antennes, **0,3 µJy par faisceau**.\n'
           '* Point without label <script>x</script>\n'
           '3. Numbered point.')

    def test_markdown_is_removed_when_stored(self):
        self.assertEqual(summarizer_utils.keypoints.clean_points(self.RAW), [
            'Nature de l’étude : il s’agit d’une étude de faisabilité.',
            'Performances envisagées : 200 à 300 antennes, 0,3 µJy par faisceau.',
            'Point without label <script>x</script>',
            'Numbered point.',
        ])

    def test_old_markdown_points_render_with_bold_label(self):
        from summarizer.templatetags.seo import keypoint
        self.assertEqual(keypoint('- **Nature de l’étude :** il s’agit d’une étude.'),
                         '<strong>Nature de l’étude :</strong> il s’agit d’une étude.')
        self.assertEqual(keypoint('Point <script>x</script>'), 'Point &lt;script&gt;x&lt;/script&gt;')
        self.assertEqual(keypoint('The ratio is 3.5: high'), 'The ratio is 3.5: high')


class AccountTests(TestCase):
    def setUp(self):
        from django.core.cache import cache
        cache.clear()

    def stamp(self, age=10):
        import time
        from django.core import signing
        from summarizer import accounts
        return signing.dumps(time.time() - age, salt=accounts.STAMP_SALT)

    def register(self, **extra):
        data = {'username': 'newbie', 'email': 'new@example.org', 'password1': 'Abcdef1!x',
                'password2': 'Abcdef1!x', 'form_stamp': self.stamp()}
        data.update(extra)
        return Client().post('/en/register/', data)

    def link_from_mail(self):
        from django.core import mail
        import re
        return re.search(r'http\S+/login/link/\S+/', mail.outbox[-1].body).group(0)

    def test_signup_sends_a_one_time_activation_link(self):
        from django.core import mail
        r = self.register()
        self.assertEqual(r.status_code, 302)
        user = User.objects.get(username='newbie')
        self.assertFalse(user.is_active)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('Activate', mail.outbox[0].subject)
        path = self.link_from_mail().split('://', 1)[1].split('/', 1)[1]
        client = Client()
        r = client.get('/' + path)  # a mail scanner opening the link changes nothing
        self.assertContains(r, 'Activate and log me in')
        user.refresh_from_db()
        self.assertFalse(user.is_active)
        r = client.post('/' + path)
        self.assertEqual(r['Location'], '/en/?activated=True')
        user.refresh_from_db()
        self.assertTrue(user.is_active)
        self.assertEqual(int(client.session['_auth_user_id']), user.pk)
        self.assertContains(client.get('/en/'), '(newbie)')  # still logged in on the next page
        self.assertEqual(Client().post('/' + path).status_code, 400)

    def test_bots_get_no_account(self):
        from django.core import mail
        self.register(website='http://spam.example')
        self.register(username='fast', form_stamp=self.stamp(age=0))
        self.register(username='nostamp', form_stamp='')
        self.assertFalse(User.objects.exists())
        self.assertEqual(len(mail.outbox), 0)

    def test_email_already_used(self):
        User.objects.create_user('first', 'new@example.org', 'pw', is_active=False)
        r = self.register(email='NEW@example.org')
        self.assertContains(r, 'already uses this email')
        self.assertEqual(User.objects.count(), 1)

    def test_inactive_login_offers_a_new_activation_email(self):
        User.objects.create_user('sleepy', 'sleepy@example.org', 'pw-123456', is_active=False)
        r = Client().post('/en/login/', {'username': 'sleepy', 'password': 'pw-123456'})
        self.assertContains(r, 'Resend the activation email')
        self.assertContains(r, 'sleepy@example.org')
        r = Client().post('/en/login/', {'username': 'sleepy', 'password': 'wrong'})
        self.assertNotContains(r, 'Resend the activation email')
        self.assertNotContains(r, 'sleepy@example.org')

    def test_login_with_email_address(self):
        User.objects.create_user('awake', 'Awake@example.org', 'pw-123456')
        r = Client().post('/en/login/', {'username': 'awake@example.org', 'password': 'pw-123456'})
        self.assertEqual(r.status_code, 302)

    def test_email_link_login(self):
        from django.core import mail
        User.objects.create_user('awake', 'awake@example.org', 'pw-123456')
        r = Client().post('/en/login/email/', {'email': 'awake@example.org', 'form_stamp': self.stamp(),
                                                'next': '/en/premium/'})
        self.assertContains(r, 'Check your email')
        link = self.link_from_mail()
        self.assertIn('?next=/en/premium/', mail.outbox[0].body)
        path = '/' + link.split('://', 1)[1].split('/', 1)[1]
        r = Client().post(path)
        self.assertEqual(r['Location'], '/en/premium/')

    def test_unknown_email_and_rate_limit(self):
        from django.core import mail
        from summarizer import accounts
        r = Client().post('/en/login/email/', {'email': 'nobody@example.org', 'form_stamp': self.stamp()})
        self.assertContains(r, 'Check your email')
        User.objects.create_user('awake', 'awake@example.org', 'pw-123456')
        for _ in range(accounts.LINKS_PER_EMAIL_PER_HOUR + 2):
            Client().post('/en/login/email/', {'email': 'awake@example.org', 'form_stamp': self.stamp()})
        self.assertEqual(len(mail.outbox), accounts.LINKS_PER_EMAIL_PER_HOUR)

    def test_offsite_next_is_ignored(self):
        User.objects.create_user('awake', 'awake@example.org', 'pw-123456')
        Client().post('/en/login/email/', {'email': 'awake@example.org', 'form_stamp': self.stamp(),
                                           'next': 'https://evil.example/'})
        self.assertNotIn('evil', self.link_from_mail())

    def test_form_errors_do_not_use_the_signup_quota(self):
        from summarizer import accounts
        for _ in range(accounts.SIGNUPS_PER_IP_PER_DAY + 2):
            self.register(password2='mismatch')
        self.register()
        self.assertTrue(User.objects.filter(username='newbie').exists())


@override_settings(GOOGLE_CLIENT_ID='cid', GOOGLE_CLIENT_SECRET='secret', SITE_URL='https://www.summarizepaper.com')
class GoogleLoginTests(TestCase):
    def client_on_site(self):
        # sign-in starts on SITE_URL's host, as a visitor of www.summarizepaper.com does
        return Client(HTTP_HOST='www.summarizepaper.com')

    def start(self, client, next_url=''):
        from urllib.parse import parse_qs, urlparse
        r = client.get('/fr/login/google/' + ('?next=' + next_url if next_url else ''))
        self.assertTrue(r['Location'].startswith('https://accounts.google.com/'))
        query = parse_qs(urlparse(r['Location']).query)
        self.assertEqual(query['redirect_uri'], ['https://www.summarizepaper.com/google/callback/'])
        return query['state'][0]

    def callback(self, client, state, profile):
        with mock.patch('summarizer.accounts._google_profile', return_value=profile):
            return client.get('/google/callback/', {'state': state, 'code': 'c'})

    def test_new_google_user_gets_an_active_account(self):
        client = self.client_on_site()
        r = self.callback(client, self.start(client), {'email': 'Jane.Doe@gmail.com', 'email_verified': True})
        self.assertContains(r, 'content="0;url=/fr/?activated=True"')
        user = User.objects.get(email='jane.doe@gmail.com')
        self.assertTrue(user.is_active)
        self.assertFalse(user.has_usable_password())
        self.assertContains(client.get('/fr/'), '(%s)' % user.username)  # still logged in on the next page

    def test_existing_inactive_account_is_activated_and_next_followed(self):
        old = User.objects.create_user('jane', 'jane@gmail.com', 'pw', is_active=False)
        client = self.client_on_site()
        r = self.callback(client, self.start(client, '/fr/premium/'), {'email': 'jane@gmail.com', 'email_verified': True})
        self.assertContains(r, 'window.location.replace("/fr/premium/")')
        old.refresh_from_db()
        self.assertTrue(old.is_active)
        self.assertEqual(User.objects.count(), 1)

    def test_bad_state_or_unverified_email_is_refused(self):
        client = self.client_on_site()
        self.start(client)
        self.assertEqual(self.callback(client, 'forged', {'email': 'a@b.c', 'email_verified': True}).status_code, 400)
        r = self.callback(client, self.start(client), {'email': 'a@b.c', 'email_verified': False})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(User.objects.exists())

    def test_next_cannot_break_out_of_the_script(self):
        client = self.client_on_site()
        r = self.callback(client, self.start(client, '/fr/%3C/script%3E%3Cscript%3Ealert(1)'),
                          {'email': 'jane@gmail.com', 'email_verified': True})
        self.assertNotContains(r, '</script><script>')

    def test_username_collision(self):
        User.objects.create_user('jane', 'other@example.org', 'pw')
        client = self.client_on_site()
        self.callback(client, self.start(client), {'email': 'jane@gmail.com', 'email_verified': True})
        self.assertTrue(User.objects.filter(username='jane2', email='jane@gmail.com').exists())

    def test_other_host_is_sent_to_site_url_first(self):
        r = Client(HTTP_HOST='summarizepaper.com').get('/fr/login/google/?next=/fr/premium/')
        self.assertEqual(r['Location'], 'https://www.summarizepaper.com/fr/login/google/?next=/fr/premium/')

    def test_button_only_when_configured(self):
        self.assertContains(Client().get('/en/login/'), 'Continue with Google')
        with override_settings(GOOGLE_CLIENT_ID=''):
            self.assertNotContains(Client().get('/en/login/'), 'Continue with Google')


class PasswordRuleTests(TestCase):
    stamp = AccountTests.stamp
    register = AccountTests.register

    def test_simple_long_password_is_enough(self):
        self.register(password1='blue sky river', password2='blue sky river')
        self.assertTrue(User.objects.filter(username='newbie').exists())

    def test_common_password_is_refused(self):
        r = self.register(password1='password123', password2='password123')
        self.assertFalse(User.objects.exists())
        self.assertContains(r, 'too common')


class SimilarPapersTests(TestCase):
    def fake_embed(self, texts):
        import numpy as np
        # direction picked by keywords so that similarity is predictable
        axes = ['galaxy', 'planet', 'language']
        vectors = np.array([[t.count(a) + 0.01 for a in axes] for t in texts], dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def paper(self, arxiv_id, title, category='astro-ph.GA'):
        return ArxivPaper.objects.create(arxiv_id=arxiv_id, title=title, abstract=title, category=category)

    def test_closest_same_category_one_version_and_scores(self):
        from summarizer import similar
        from summarizer.models import PaperScore
        main = self.paper('2401.00001v1', 'galaxy galaxy')
        self.paper('2401.00001v2', 'galaxy galaxy')          # another version of the same paper
        self.paper('2401.00002v1', 'galaxy planet')
        self.paper('2401.00002v2', 'galaxy planet')          # only one version may be listed
        self.paper('2401.00003v1', 'planet planet')
        self.paper('2401.00004v1', 'galaxy galaxy', category='cs.CL')  # other category
        with mock.patch('summarizer.similar.embed', side_effect=self.fake_embed) as embed:
            similar.store(list(ArxivPaper.objects.exclude(arxiv_id=main.arxiv_id)))
            found = similar.find_and_store(main.arxiv_id)
        self.assertEqual([aid[:10] for aid, _ in found], ['2401.00002', '2401.00003'])
        self.assertLess(found[0][1], found[1][1])
        self.assertEqual(embed.call_count, 2)  # the main paper was embedded on demand
        scores = list(PaperScore.objects.filter(from_paper=main).order_by('score'))
        self.assertEqual(len(scores), 2)
        self.assertEqual(similar.percent(0.25), '75.0')

    def test_embed_papers_command(self):
        from io import StringIO
        from django.core.management import call_command
        from summarizer.models import PaperEmbedding, PaperScore
        for i, title in enumerate(['galaxy', 'galaxy planet', 'planet']):
            p = self.paper('2402.0000%dv1' % i, title)
            SummaryPaper.objects.create(paper=p, lang='en', summary='s')
        out = StringIO()
        with mock.patch('summarizer.similar.embed', side_effect=self.fake_embed):
            call_command('embed_papers', '--rescore', stdout=out)
        self.assertEqual(PaperEmbedding.objects.count(), 3)
        self.assertEqual(PaperScore.objects.count(), 6)
        self.assertIn('3 papers to embed', out.getvalue())


class ReactivationTests(TestCase):
    def inactive(self, username, email, days_ago=100):
        user = User.objects.create_user(username, email, 'pw', is_active=False)
        user.date_joined = timezone.now() - timezone.timedelta(days=days_ago)
        user.save()
        return user

    def test_random_usernames_are_left_out(self):
        from summarizer.reactivation import looks_random
        for name in ('zBAybMjDxpQcJ', 'eNbfmlwp', 'EGGTozxHH', 'DLRGhhVCAQKAR', 'UYQWSlYkZM'):
            self.assertTrue(looks_random(name, 'victim@yahoo.com'), name)
        for name, email in (('asif_chishti', 'a@gmail.com'), ('heyman.jonathan', 'j@gmail.com'),
                            ('JohnSmith', 'john.smith@gmail.com'), ('kazuki', 'k@x.jp'), ('Ayodamola', 'a@gmail.com')):
            self.assertFalse(looks_random(name, email), name)

    def test_candidates_and_single_send(self):
        from django.core import mail
        from summarizer import reactivation
        real = self.inactive('brieuc.langlois', 'brieuc@example.org', 10)
        self.inactive('brieuc2', 'BRIEUC@example.org', 20)        # same address: one email only
        self.inactive('zBAybMjDxpQcJ', 'victim@yahoo.com')       # bombing victim
        self.inactive('taken', 'active@example.org')
        User.objects.create_user('activeone', 'active@example.org', 'pw')
        fresh = self.inactive('just.signed.up', 'new@example.org', 0)  # still has a working activation link
        fresh.date_joined = reactivation.CUTOFF + timezone.timedelta(hours=1)
        fresh.save()
        self.assertEqual(reactivation.candidates(), [real])
        self.assertEqual(reactivation.run('https://www.summarizepaper.com', 10), 1)
        self.assertEqual(reactivation.run('https://www.summarizepaper.com', 10), 0)
        self.assertEqual(len(mail.outbox), 1)
        self.assertIn('List-Unsubscribe', mail.outbox[0].extra_headers)
        self.assertIn('/en/login/link/', mail.outbox[0].body)

    def test_activation_link_lasts_30_days_and_normal_links_3(self):
        import re
        from django.core import mail
        from summarizer import accounts, reactivation
        user = self.inactive('brieuc.langlois', 'brieuc@example.org')
        reactivation.run('https://www.summarizepaper.com', 10)
        path = re.search(r'https://www\.summarizepaper\.com(/en/login/link/\S+/)', mail.outbox[0].body).group(1)
        short = accounts.make_link_token(user)
        later = _time.time() + 10 * 24 * 3600
        with mock.patch('django.core.signing.time.time', return_value=later):
            self.assertIsNone(accounts.user_from_link_token(short))
            r = Client().post(path)
        self.assertEqual(r.status_code, 302)
        user.refresh_from_db()
        self.assertTrue(user.is_active)

    def test_delete_link_removes_only_unused_accounts(self):
        import re
        from django.core import mail
        from summarizer import reactivation
        user = self.inactive('brieuc.langlois', 'brieuc@example.org')
        reactivation.run('https://www.summarizepaper.com', 10)
        path = re.search(r'https://www\.summarizepaper\.com(/en/account/delete/\S+/)', mail.outbox[0].body).group(1)
        self.assertContains(Client().get(path), 'Delete the account')
        self.assertTrue(User.objects.filter(pk=user.pk).exists())
        Client().post(path)
        self.assertFalse(User.objects.filter(pk=user.pk).exists())
        used = User.objects.create_user('used', 'u@example.org', 'pw')
        r = Client().post('/en/account/delete/%s/' % reactivation.delete_token(used))
        self.assertContains(r, 'not deleted')
        self.assertTrue(User.objects.filter(pk=used.pk).exists())


class ArxivApiRelayTests(TestCase):
    def setUp(self):
        from django.core.cache import cache
        cache.clear()

    def fake(self, body=b'<feed/>'):
        return mock.Mock(content=body, raise_for_status=mock.Mock())

    def test_search_is_relayed_with_capped_results(self):
        with mock.patch('summarizer.views.requests.get', return_value=self.fake(b'<feed>x</feed>')) as get:
            r = Client().get('/en/arxiv-api/', {'search_query': 'dark matter', 'max_results': '500'})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.content, b'<feed>x</feed>')
        self.assertTrue(r['Content-Type'].startswith('application/atom+xml'))
        self.assertEqual(get.call_args.kwargs['params'], {'search_query': 'dark matter', 'max_results': 20})

    def test_answers_are_cached(self):
        with mock.patch('summarizer.views.requests.get', return_value=self.fake()) as get:
            Client().get('/en/arxiv-api/', {'id_list': '2304.01202v1'})
            Client().get('/en/arxiv-api/', {'id_list': '2304.01202v1'})
        self.assertEqual(get.call_count, 1)

    def test_needs_exactly_one_kind_of_query(self):
        with mock.patch('summarizer.views.requests.get') as get:
            self.assertEqual(Client().get('/en/arxiv-api/').status_code, 400)
            self.assertEqual(Client().get('/en/arxiv-api/', {'id_list': 'a', 'search_query': 'b'}).status_code, 400)
        get.assert_not_called()

    def test_arxiv_down_gives_502(self):
        import requests
        with mock.patch('summarizer.views.requests.get', side_effect=requests.ConnectionError()):
            self.assertEqual(Client().get('/en/arxiv-api/', {'id_list': '2304.01202v1'}).status_code, 502)


class TreePageTests(TestCase):
    def test_fifteen_related_papers_in_constant_queries(self):
        from summarizer.models import Author, PaperAuthor, PaperScore
        main = ArxivPaper.objects.create(arxiv_id='2401.00001v1', title='main', category='astro-ph.GA')
        for i in range(15):
            other = ArxivPaper.objects.create(arxiv_id='2401.%05dv1' % (i + 2), title='p%d' % i, category='astro-ph.GA')
            for j in range(2):
                PaperAuthor.objects.create(paper=other, author=Author.objects.create(name='A%d-%d' % (i, j), affiliation=''),
                                           author_order=j)
            # close scores put every paper on the same ring: the old random placement struggled here
            PaperScore.objects.create(from_paper=main, to_paper=other, score=0.30 + i / 1000, active=True)
        with self.assertNumQueries(9):
            r = Client().get('/en/tree/2401.00001v1/')
        self.assertEqual(r.status_code, 200)
        data = _json.loads(r.context['relpapers_json'])
        self.assertEqual(len(data), 16)
        self.assertEqual(data[1]['first_author'], 'A0-0')


class ArxivTimeoutTests(TestCase):
    def test_slow_arxiv_gives_503_instead_of_hanging(self):
        import asyncio
        with mock.patch('summarizer.utils.get_arxiv_metadata', side_effect=asyncio.TimeoutError()):
            self.assertEqual(Client().get('/en/arxiv-id/2401.99999v1/').status_code, 503)
            self.assertEqual(Client().get('/en/tree/2401.99999v1/').status_code, 503)

    def test_non_xml_arxiv_answer_gives_503(self):
        from xml.etree import ElementTree
        with mock.patch('summarizer.utils.get_arxiv_metadata', side_effect=ElementTree.ParseError('syntax error: line 1, column 0')):
            self.assertEqual(Client().get('/en/tree/2401.99999v1/').status_code, 503)


from summarizer.models import Vote


class VoteTests(TestCase):
    def setUp(self):
        self.paper = ArxivPaper.objects.create(arxiv_id='2101.00003v1', title='Voted', abstract='Abs.')

    def test_fetch_vote_returns_score(self):
        c = Client(HTTP_HOST='localhost')
        r = c.post('/fr/vote/2101.00003v1/', {'direction': 'up'}, HTTP_X_REQUESTED_WITH='fetch')
        self.assertEqual(r.json(), {'total': 1, 'message': 'Merci pour votre vote', 'already': False})
        r = c.post('/fr/vote/2101.00003v1/', {'direction': 'down'}, HTTP_X_REQUESTED_WITH='fetch')
        self.assertEqual(r.json(), {'total': 1, 'message': 'Vous avez déjà voté', 'already': True})
        self.assertEqual(Vote.objects.filter(paper=self.paper).count(), 1)

    def test_form_vote_redirects(self):
        c = Client(HTTP_HOST='localhost')
        r = c.post('/en/vote/2101.00003v1/', {'direction': 'down'})
        self.assertRedirects(r, '/en/arxiv-id/2101.00003v1/', fetch_redirect_response=False)
        r = c.post('/en/vote/2101.00003v1/', {'direction': 'down'})
        self.assertRedirects(r, '/en/arxiv-id/2101.00003v1/vote#totvote', fetch_redirect_response=False)


class GuideTests(TestCase):
    GUIDE = """---
title: Keeping up with arXiv
description: How to follow new papers.
date: 2026-10-20
faq:
  - q: Is it free?
    a: Yes.
sources:
  - label: arXiv stats
    url: https://arxiv.org/stats
---
Read the **listings** <b>daily</b>.
"""

    def setUp(self):
        import tempfile
        from unittest import mock
        from summarizer import guides
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, 'en'))
        os.makedirs(os.path.join(self.dir, 'fr'))
        self.write('en', 'keep-up', self.GUIDE)
        patcher = mock.patch.object(guides, 'CONTENT_DIR', self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def write(self, lang, slug, text):
        with open(os.path.join(self.dir, lang, slug + '.md'), 'w') as f:
            f.write(text)

    def test_guide_page_renders_markdown_faq_and_schema(self):
        r = self.client.get('/en/guides/keep-up/')
        self.assertEqual(r.status_code, 200)
        html = r.content.decode()
        self.assertIn('<strong>listings</strong>', html)
        self.assertIn('"@type": "FAQPage"', html)
        self.assertIn('"@type": "Article"', html)
        self.assertIn('https://arxiv.org/stats', html)
        self.assertNotIn('hreflang="fr"', html)
        self.assertNotIn('SoftwareApplication', html)

    def test_dataset_adds_dataset_markup(self):
        self.write('en', 'stats', self.GUIDE.replace('date:', 'dataset:\n  name: Usage stats\n  description: Counts.\n  temporalCoverage: 2023-02-24/2026-10-18\ndate:'))
        html = self.client.get('/en/guides/stats/').content.decode()
        self.assertIn('"@type": "Dataset"', html)
        self.assertIn('"temporalCoverage": "2023-02-24/2026-10-18"', html)

    def test_product_flag_adds_software_application(self):
        self.write('en', 'vs', self.GUIDE.replace('date:', 'product: true\ndate:'))
        self.assertIn('"@type": "SoftwareApplication"', self.client.get('/en/guides/vs/').content.decode())

    def test_missing_language_is_404(self):
        self.assertEqual(self.client.get('/fr/guides/keep-up/').status_code, 404)

    def test_both_languages_link_with_hreflang(self):
        self.write('fr', 'keep-up', self.GUIDE.replace('Keeping up', 'Suivre'))
        html = self.client.get('/en/guides/keep-up/').content.decode()
        self.assertIn('hreflang="fr" href="https://www.summarizepaper.com/fr/guides/keep-up/"', html)

    def test_draft_hidden_from_public_lists_and_sitemap(self):
        self.write('en', 'wip', self.GUIDE.replace('date:', 'draft: true\ndate:'))
        self.assertEqual(self.client.get('/en/guides/wip/').status_code, 404)
        self.assertNotIn('/en/guides/wip/', self.client.get('/en/guides/').content.decode())
        self.assertNotIn('wip', self.client.get('/sitemap-guides.xml').content.decode())
        self.assertNotIn('wip', self.client.get('/llms.txt').content.decode())

    def test_draft_visible_to_staff_with_noindex(self):
        self.write('en', 'wip', self.GUIDE.replace('date:', 'draft: true\ndate:'))
        staff = User.objects.create_user('staff', password='x', is_staff=True)
        self.client.force_login(staff)
        r = self.client.get('/en/guides/wip/')
        self.assertEqual(r.status_code, 200)
        self.assertIn('noindex', r.content.decode())

    def test_staff_sees_drafts_in_the_index(self):
        self.write('en', 'wip', self.GUIDE.replace('date:', 'draft: true\ndate:'))
        self.client.force_login(User.objects.create_user('staff', password='x', is_staff=True))
        html = self.client.get('/en/guides/').content.decode()
        self.assertIn('/en/guides/wip/', html)
        self.assertIn('min read', html)

    def test_published_guide_listed_in_sitemap_and_llms_txt(self):
        self.assertIn('/en/guides/keep-up/', self.client.get('/sitemap-guides.xml').content.decode())
        self.assertIn('- [Keeping up with arXiv](https://www.summarizepaper.com/en/guides/keep-up/)',
                      self.client.get('/llms.txt').content.decode())

    def test_broken_front_matter_is_skipped(self):
        self.write('en', 'broken', '---\ntitle: a: b\n---\nText')
        self.assertEqual(self.client.get('/en/guides/broken/').status_code, 404)
        self.assertEqual(self.client.get('/sitemap-guides.xml').status_code, 200)

    def test_bad_slug_is_404(self):
        self.assertEqual(self.client.get('/en/guides/..%2Fsecret/').status_code, 404)


class GuideFileTests(TestCase):
    def test_every_guide_in_the_repo_parses(self):
        from summarizer import guides
        for lang in os.listdir(guides.CONTENT_DIR):
            for name in os.listdir(os.path.join(guides.CONTENT_DIR, lang)):
                if name.endswith('.md'):
                    guides._parse.cache_clear()
                    path = os.path.join(guides.CONTENT_DIR, lang, name)
                    meta = guides._parse(path, 0)
                    for key in ('title', 'description', 'date'):
                        self.assertTrue(meta.get(key), '%s has no %s' % (path, key))
                    self.assertLessEqual(len(meta['description']), 160, path)


class AuthorPageTests(TestCase):
    def test_author_page_in_both_languages(self):
        en = self.client.get('/en/author/quentin-kral/')
        self.assertEqual(en.status_code, 200)
        self.assertIn('orcid.org/0000-0001-6527-4684', en.content.decode())
        fr = self.client.get('/fr/author/quentin-kral/')
        self.assertIn('Observatoire de Paris depuis 2018', fr.content.decode())


class MailcheckTests(TestCase):
    def test_reasons(self):
        from summarizer import mailcheck
        mailcheck.suppress('Bounced@Example.org', 'bounce')
        self.assertEqual(mailcheck.undeliverable_reason('bounced@example.org'), 'suppressed')
        self.assertEqual(mailcheck.undeliverable_reason('not an address'), 'invalid')
        self.assertEqual(mailcheck.undeliverable_reason('kebagiy835@ixunbo.com'), 'disposable')
        self.assertEqual(mailcheck.undeliverable_reason('o.k.or.ol.a.d.ed.ep21@gmail.com'), 'gmail dots')
        self.assertEqual(mailcheck.undeliverable_reason('jane.m.doe@gmail.com'), '')
        with override_settings(MAILCHECK_DNS=True), \
             mock.patch('summarizer.mailcheck.domain_accepts_mail', return_value=False):
            self.assertEqual(mailcheck.undeliverable_reason('someone@gone.example'), 'no mail server')

    def test_null_mx_and_missing_domain(self):
        import dns.resolver
        from summarizer import mailcheck
        mailcheck.domain_accepts_mail.cache_clear()
        null_mx = [mock.Mock(exchange='.')]
        with mock.patch('dns.resolver.Resolver.resolve', return_value=null_mx):
            self.assertFalse(mailcheck.domain_accepts_mail('nullmx.example'))
        with mock.patch('dns.resolver.Resolver.resolve', side_effect=dns.resolver.NXDOMAIN):
            self.assertFalse(mailcheck.domain_accepts_mail('nxdomain.example'))
        with mock.patch('dns.resolver.Resolver.resolve', return_value=[mock.Mock(exchange='mx.ok.example.')]):
            self.assertTrue(mailcheck.domain_accepts_mail('ok.example'))
        mailcheck.domain_accepts_mail.cache_clear()

    def test_suppressed_addresses_get_nothing(self):
        from summarizer import mailcheck, reactivation
        user = User.objects.create_user('janedoe', 'jane@example.org', 'x')
        user.is_active = False
        user.date_joined = reactivation.CUTOFF - timezone.timedelta(days=30)
        user.save()
        self.assertEqual(reactivation.candidates(), [user])
        mailcheck.suppress('jane@example.org', 'bounce')
        self.assertEqual(reactivation.candidates(), [])
        request = RequestFactory().get('/')
        from summarizer import accounts
        self.assertFalse(accounts.send_login_link(request, user))
        self.assertEqual(len(mail.outbox), 0)

    def test_signup_refuses_throwaway_address(self):
        cache.clear()
        with mock.patch('summarizer.accounts.looks_like_bot', return_value=False):
            response = self.client.post('/en/register/', {'username': 'quirky', 'email': 'quirky@typingsquirrel.com',
                                                          'password1': 'a long enough passphrase', 'password2': 'a long enough passphrase'})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(User.objects.filter(username='quirky').exists())
        self.assertEqual(len(mail.outbox), 0)


class AdminTests(TestCase):
    def setUp(self):
        self.addCleanup(SiteSettings.clear_cache)
        SiteSettings.clear_cache()
        self.admin = User.objects.create_superuser('boss', 'boss@example.org', 'x')

    def test_quotas_follow_site_settings(self):
        self.assertEqual(access.runs_per_day(), 10)
        cfg = SiteSettings.current()
        cfg.free_runs_per_day = 2
        cfg.chat_anonymous_per_day = 1
        cfg.save()
        ip = access.hash_ip('9.9.9.9')
        PaperHistory.objects.create(arxiv_id='1', ip_address=ip, created=timezone.now())
        PaperHistory.objects.filter(arxiv_id='1').update(created=timezone.now() - timezone.timedelta(minutes=5))
        self.assertEqual(access.run_quota(None, ip), (True, None))
        PaperHistory.objects.create(arxiv_id='2', ip_address=ip)
        PaperHistory.objects.filter(arxiv_id='2').update(created=timezone.now() - timezone.timedelta(minutes=5))
        self.assertEqual(access.run_quota(None, ip), (False, 'toomany'))
        self.assertEqual(access.chat_quota_left(None, ip), 1)

    def test_models_follow_site_settings(self):
        from summarizer import utils
        cfg = SiteSettings.current()
        cfg.model_free, cfg.model_paid = 'model-a', 'model-b'
        cfg.save()
        self.assertEqual(utils.ai_models(), ('model-a', 'model-b'))

    def test_admin_pages_load(self):
        self.client.force_login(self.admin)
        PaperHistory.objects.create(arxiv_id='1', user=self.admin, ip_address='h')
        UserPlan.objects.create(user=self.admin, paid=True)
        r = self.client.get('/admin/auth/user/?premium=yes')
        self.assertEqual(r.status_code, 200)
        self.assertContains(r, 'boss@example.org')
        r = self.client.get('/admin/summarizer/sitesettings/')
        self.assertRedirects(r, '/admin/summarizer/sitesettings/1/change/')
        self.assertEqual(self.client.get('/admin/summarizer/sitesettings/1/change/').status_code, 200)
        self.assertEqual(self.client.get('/admin/summarizer/guide/add/').status_code, 200)
        self.assertEqual(self.client.get('/admin/auth/user/%d/change/' % self.admin.pk).status_code, 200)


class DatabaseGuideTests(TestCase):
    def setUp(self):
        import tempfile
        from summarizer import guides
        self.dir = tempfile.mkdtemp()
        os.makedirs(os.path.join(self.dir, 'en'))
        with open(os.path.join(self.dir, 'en', 'from-file.md'), 'w') as f:
            f.write('---\ntitle: File version\ndescription: d\ndate: 2026-10-01\n---\nFile body.\n')
        patcher = mock.patch.object(guides, 'CONTENT_DIR', self.dir)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_database_guide_is_published_and_replaces_the_file(self):
        from summarizer.models import Guide
        import datetime
        Guide.objects.create(lang='en', slug='from-file', title='Admin version', description='d',
                             body='Edited **here**.', date=datetime.date(2026, 10, 2), draft=False,
                             extra='faq:\n  - q: Why?\n    a: Because.\n')
        Guide.objects.create(lang='en', slug='wip', title='Draft', description='d', body='x',
                             date=datetime.date(2026, 10, 3))
        html = self.client.get('/en/guides/from-file/').content.decode()
        self.assertIn('Edited <strong>here</strong>', html)
        self.assertIn('"@type": "FAQPage"', html)
        self.assertNotIn('File body', html)
        self.assertEqual(self.client.get('/en/guides/wip/').status_code, 404)
        self.assertNotIn('/en/guides/wip/', self.client.get('/sitemap-guides.xml').content.decode())

    def test_import_command_copies_files(self):
        from django.core.management import call_command
        from summarizer.models import Guide
        call_command('import_guides', stdout=open(os.devnull, 'w'))
        guide = Guide.objects.get(lang='en', slug='from-file')
        self.assertEqual(guide.title, 'File version')
        self.assertEqual(guide.body, 'File body.\n')
        self.assertFalse(guide.draft)
