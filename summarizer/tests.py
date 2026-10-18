from django.contrib.auth.models import User
from django.contrib.sessions.backends.db import SessionStore
from django.test import TestCase, RequestFactory
from django.utils import timezone

from summarizer import access
from summarizer.models import AIassistant, PaperHistory, UserPlan


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
        for i in range(access.RUNS_PER_DAY - 1):
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
        for i in range(access.CHAT_MESSAGES_PER_DAY['anonymous']):
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
        self.assertEqual(FakeSession.last_payload['model'], assistant.MODEL_PAID)
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
        client = Client()
        r = self.callback(client, self.start(client), {'email': 'Jane.Doe@gmail.com', 'email_verified': True})
        self.assertContains(r, 'content="0;url=/fr/?activated=True"')
        user = User.objects.get(email='jane.doe@gmail.com')
        self.assertTrue(user.is_active)
        self.assertFalse(user.has_usable_password())
        self.assertContains(client.get('/fr/'), '(%s)' % user.username)  # still logged in on the next page

    def test_existing_inactive_account_is_activated_and_next_followed(self):
        old = User.objects.create_user('jane', 'jane@gmail.com', 'pw', is_active=False)
        client = Client()
        r = self.callback(client, self.start(client, '/fr/premium/'), {'email': 'jane@gmail.com', 'email_verified': True})
        self.assertContains(r, 'window.location.replace("/fr/premium/")')
        old.refresh_from_db()
        self.assertTrue(old.is_active)
        self.assertEqual(User.objects.count(), 1)

    def test_bad_state_or_unverified_email_is_refused(self):
        client = Client()
        self.start(client)
        self.assertEqual(self.callback(client, 'forged', {'email': 'a@b.c', 'email_verified': True}).status_code, 400)
        r = self.callback(client, self.start(client), {'email': 'a@b.c', 'email_verified': False})
        self.assertEqual(r.status_code, 400)
        self.assertFalse(User.objects.exists())

    def test_next_cannot_break_out_of_the_script(self):
        client = Client()
        r = self.callback(client, self.start(client, '/fr/%3C/script%3E%3Cscript%3Ealert(1)'),
                          {'email': 'jane@gmail.com', 'email_verified': True})
        self.assertNotContains(r, '</script><script>')

    def test_username_collision(self):
        User.objects.create_user('jane', 'other@example.org', 'pw')
        client = Client()
        self.callback(client, self.start(client), {'email': 'jane@gmail.com', 'email_verified': True})
        self.assertTrue(User.objects.filter(username='jane2', email='jane@gmail.com').exists())

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
