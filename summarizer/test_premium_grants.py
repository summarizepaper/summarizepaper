import json
from types import SimpleNamespace
from unittest import mock

from django.contrib.auth.models import AnonymousUser, User
from django.test import Client, override_settings
from django.utils import timezone

from summarizer import access, billing, generated
from summarizer.models import Generated, PremiumGrant, UserPlan
from summarizer.premium_tools import GENERATED_TOOLS, PLUS_TOOLS, TOOL_KEYS
from summarizer.test_tools import ToolsTestCase


class PremiumGrantTests(ToolsTestCase):
    def grant(self, user, tool, **kwargs):
        return PremiumGrant.objects.create(user=user, tool=tool, **kwargs)

    def test_every_tool_grant_is_isolated_from_full_premium_and_other_tools(self):
        user=self.user()
        for tool in TOOL_KEYS - {'all', 'plus'}:
            with self.subTest(tool=tool):
                grant=self.grant(user,tool)
                self.assertTrue(access.has_tool(user,tool))
                self.assertFalse(access.is_paid(user))
                for other in TOOL_KEYS - {tool}:
                    self.assertFalse(access.has_tool(user,other))
                grant.delete()
        full=self.grant(user,'all')
        self.assertTrue(access.is_paid(user)); self.assertFalse(access.is_plus(user))
        self.assertTrue(all(access.has_tool(user,tool) for tool in TOOL_KEYS - PLUS_TOOLS))
        self.assertFalse(any(access.has_tool(user,tool) for tool in PLUS_TOOLS))  # full Premium is not Premium+
        full.delete(); self.grant(user,'plus')
        self.assertTrue(access.is_paid(user)); self.assertTrue(access.is_plus(user))
        self.assertTrue(all(access.has_tool(user,tool) for tool in TOOL_KEYS))
        self.assertFalse(access.has_tool(user,'typo'))
        self.assertFalse(access.has_tool(AnonymousUser(),'presskit'))
        self.assertFalse(access.is_paid(None))

    def test_expiry_revocation_inactive_users_and_cumulative_grants(self):
        user=self.user()
        grant=self.grant(user,'presskit',expires=timezone.now()-timezone.timedelta(seconds=1))
        self.assertFalse(access.has_tool(user,'presskit'))
        grant.expires=timezone.now()+timezone.timedelta(days=30);grant.save()
        self.grant(user,'journal_club')
        self.assertTrue(access.has_tool(user,'presskit'));self.assertTrue(access.has_tool(user,'journal_club'))
        PremiumGrant.objects.filter(pk=grant.pk).update(active=False)
        self.assertFalse(access.has_tool(user,'presskit'));self.assertTrue(access.has_tool(user,'journal_club'))
        user.is_active=False;user.save()
        self.assertFalse(access.has_tool(user,'journal_club'))

    def test_stripe_events_do_not_remove_gifts_and_revoking_a_gift_does_not_cancel_a_paid_plan(self):
        user=self.user();grant=self.grant(user,'all')
        subscription={'id':'sub_test','customer':'cus_test','status':'canceled','current_period_end':int(timezone.now().timestamp()),'metadata':{'user_id':str(user.pk)}}
        billing.sync_subscription(subscription)
        self.assertTrue(access.is_paid(user))
        subscription.update(status='active',current_period_end=int((timezone.now()+timezone.timedelta(days=30)).timestamp()))
        billing.sync_subscription(subscription)
        grant.active=False;grant.save()
        self.assertTrue(access.is_paid(user))
        plan=UserPlan.objects.get(user=user)
        self.assertEqual(plan.stripe_subscription_id,'sub_test');self.assertTrue(plan.paid)
        self.assertFalse(plan.canceling)

    def test_only_selected_tool_can_generate_show_status_and_download(self):
        self.paper();user=self.user();self.grant(user,'presskit');client=self.client_for(user)
        kit=Generated.objects.create(kind='presskit',key='2101.00001v1',lang='en',status='done',content=json.dumps({'headline':'A result'}))
        deck=Generated.objects.create(kind='slides_v2',key='2101.00001v1',lang='en',status='done',content='{}')
        self.assertTrue(client.get('/en/press-kit/2101.00001v1/').context['paid'])
        self.assertFalse(client.get('/en/journal-club/2101.00001v1/').context['paid'])
        with mock.patch('summarizer.generated.start',return_value=SimpleNamespace(status='done',pk=kit.pk)) as start:
            self.assertEqual(client.post('/en/press-kit/2101.00001v1/start/').status_code,200)
            self.assertEqual(client.post('/en/journal-club/2101.00001v1/start/').status_code,403)
            self.assertEqual(start.call_count,1)
        self.assertEqual(client.get('/en/ai/%d/'%kit.pk).status_code,200)
        self.assertEqual(client.get('/en/ai/%d/'%deck.pk).status_code,404)
        with mock.patch('summarizer.presskit_pdf.document',return_value=b'%PDF-fixture'):
            self.assertEqual(client.get('/en/press-kit/2101.00001v1/pdf/').status_code,200)
        self.assertEqual(client.post('/en/journal-club/2101.00001v1/save/',data='{}',content_type='application/json').status_code,403)
        self.assertEqual(client.post('/en/journal-club/2101.00001v1/powerpoint/',data='{}',content_type='application/json').status_code,403)
        PremiumGrant.objects.filter(user=user).update(active=False)
        self.assertEqual(client.get('/en/ai/%d/'%kit.pk).status_code,404)
        self.assertEqual(client.get('/en/press-kit/2101.00001v1/pdf/').status_code,403)

    def test_journal_club_grant_includes_all_deck_versions_but_not_presskit_or_private_edits(self):
        self.paper();user=self.user();other=self.user('other');self.grant(user,'journal_club');client=self.client_for(user)
        self.assertTrue(client.get('/en/journal-club/2101.00001v1/').context['paid'])
        self.assertFalse(client.get('/en/press-kit/2101.00001v1/').context['paid'])
        self.assertEqual(client.post('/en/press-kit/2101.00001v1/start/').status_code,403)
        for kind,tool in GENERATED_TOOLS.items():
            row=Generated.objects.create(kind=kind,key='scope-check',lang='en',status='done',content='{}')
            self.assertEqual(client.get('/en/ai/%d/'%row.pk).status_code,200 if tool=='journal_club' else 404,kind)
        private=Generated.objects.create(kind='slides_edit_v2',key='private',lang='en',user=other,status='done',content='{}')
        self.assertEqual(client.get('/en/ai/%d/'%private.pk).status_code,404)

    def test_expired_or_wrong_scope_cannot_read_private_reports_or_bib_downloads(self):
        user=self.user();self.grant(user,'presskit');client=self.client_for(user)
        for tool in ('refcheck','missed','conference','teams'):
            self.grant(user,tool,expires=timezone.now()-timezone.timedelta(days=1))
            row=Generated.objects.create(kind=tool,key='private',lang='en',user=user,status='done',content='{}')
            self.assertEqual(client.get('/en/%s/private/'%tool).status_code,404)
            self.assertEqual(client.get('/en/ai/%d/'%row.pk).status_code,404)
        self.assertEqual(client.get('/en/refcheck/private/bib/').status_code,404)

    @override_settings(STRIPE_SECRET_KEY='sk_test', STRIPE_PRICE_MONTHLY='price_test')  # checkout is off without them
    def test_limited_grants_keep_global_quotas_and_allow_full_subscription_checkout(self):
        user=self.user();self.grant(user,'presskit',note='note interne privée');client=self.client_for(user)
        self.assertEqual(access.chat_quota_left(user,'not-used'),access.chat_messages_per_day()['user'])
        with mock.patch('stripe.checkout.Session.create',return_value=SimpleNamespace(url='https://checkout.stripe.com/test')) as checkout:
            response=client.post('/en/premium/checkout/',{'period':'monthly'})
            self.assertEqual(response.status_code,302);checkout.assert_called_once()
        with mock.patch('summarizer.billing.price_label',return_value='7 EUR'):
            page=client.get('/fr/premium/')
        self.assertFalse(page.context['is_paid']);self.assertEqual(len(page.context['tool_grants']),1)
        self.assertNotContains(page,'note interne privée')

    def test_admin_can_create_update_and_revoke_scopes_with_audit_but_regular_staff_cannot(self):
        owner=self.user();boss=User.objects.create_superuser('boss','boss@example.org','x');client=self.client_for(boss)
        url='/admin/summarizer/premiumgrant/add/'
        response=client.post(url,{'user':owner.pk,'tool':'presskit','active':'on','expires_0':'','expires_1':'','note':'note interne privée','granted_by':owner.pk,'_save':'Save'})
        self.assertEqual(response.status_code,302)
        grant=PremiumGrant.objects.get(user=owner)
        self.assertEqual(grant.granted_by,boss);self.assertTrue(access.has_tool(owner,'presskit'))
        self.assertFalse(access.is_paid(owner))
        page=client.get('/admin/auth/user/%d/change/'%owner.pk)
        self.assertContains(page,'Premium complet ou outil au choix')
        self.assertContains(client.get('/admin/auth/user/?granted_tool=presskit'),'ann@example.com')
        self.assertNotContains(client.get('/admin/auth/user/?premium=yes'),'ann@example.com')
        changed=client.post('/admin/summarizer/premiumgrant/%d/change/'%grant.pk,{'user':owner.pk,'tool':'presskit','expires_0':'','expires_1':'','note':'revoked','_save':'Save'})
        self.assertEqual(changed.status_code,302);self.assertFalse(access.has_tool(owner,'presskit'))
        staff=self.user('staff');staff.is_staff=True;staff.save()
        self.assertEqual(self.client_for(staff).get(url).status_code,403)
        self.assertEqual(self.client_for(staff).post(url,{'user':owner.pk,'tool':'all','active':'on'}).status_code,403)
        self.assertEqual(self.client_for(owner).get(url).status_code,302)
        self.assertFalse(PremiumGrant.objects.filter(user=owner,tool='all').exists())

    def test_admin_requires_an_explicit_scope_and_user_inline_records_the_grantor(self):
        from bs4 import BeautifulSoup
        user=self.user();boss=User.objects.create_superuser('boss','boss@example.org','x');client=self.client_for(boss)
        response=client.post('/admin/summarizer/premiumgrant/add/',{'user':user.pk,'tool':'','active':'on','note':'Missing scope','_save':'Save'})
        self.assertEqual(response.status_code,200);self.assertFalse(PremiumGrant.objects.exists())
        url='/admin/auth/user/%d/change/'%user.pk
        page=client.get(url);soup=BeautifulSoup(page.content,'html.parser')
        data={}
        for field in soup.select('input[name]'):
            if field.get('type') in ('submit','button','file') or field.get('disabled'):
                continue
            if field.get('type') in ('checkbox','radio') and not field.has_attr('checked'):
                continue
            data[field['name']]=field.get('value','')
        data.update({'premium_grants-TOTAL_FORMS':'1','premium_grants-0-tool':'journal_club',
                     'premium_grants-0-active':'on','premium_grants-0-expires_0':'','premium_grants-0-expires_1':'',
                     'premium_grants-0-note':'Inline test','_continue':'Save'})
        response=client.post(url,data)
        self.assertEqual(response.status_code,302)
        grant=PremiumGrant.objects.get(user=user)
        self.assertEqual(grant.granted_by,boss);self.assertEqual(grant.tool,'journal_club')
        self.assertTrue(access.has_tool(user,'journal_club'));self.assertFalse(access.has_tool(user,'presskit'))

    def test_admin_full_premium_filter_includes_gifts_and_respects_the_latest_billing_plan(self):
        user=self.user();self.grant(user,'all');UserPlan.objects.create(user=user,paid=False)
        boss=User.objects.create_superuser('boss','boss@example.org','x');client=self.client_for(boss)
        self.assertContains(client.get('/admin/auth/user/?premium=yes'),'ann@example.com')
        self.assertTrue(access.is_paid(user))
        self.assertContains(client.get('/admin/'),'abonnements ou accès complets offerts')
