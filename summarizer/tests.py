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
        self.assertTrue(kwargs['success_url'].endswith('/en/premium/?success=1'))
