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
