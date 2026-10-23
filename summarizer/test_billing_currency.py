from unittest import mock

from django.contrib.auth.models import User
from django.core.cache import cache
from django.test import TestCase, override_settings

from .models import UserPlan


@override_settings(STRIPE_SECRET_KEY='sk_test', STRIPE_PRICE_MONTHLY='price_eur',
                   STRIPE_PRICE_YEARLY='price_year_eur', STRIPE_PRICE_MONTHLY_USD='price_usd',
                   STRIPE_PRICE_YEARLY_USD='')
class BillingCurrencyTests(TestCase):
    def setUp(self):
        cache.clear()
        self.user = User.objects.create_user('currency-preview', email='test@example.com')
        self.price = mock.patch('stripe.Price.retrieve', side_effect=lambda price: {
            'unit_amount': 900 if price == 'price_usd' else 700,
            'currency': 'usd' if price == 'price_usd' else 'eur', 'recurring': {'interval': 'month'}})
        self.price.start()
        self.addCleanup(self.price.stop)

    def test_only_trusted_us_geography_defaults_to_usd_in_either_language(self):
        headers = {'HTTP_CF_IPCOUNTRY': 'US', 'HTTP_X_FORWARDED_FOR': '198.51.100.4, 173.245.48.1'}
        for lang in ('fr', 'en'):
            response = self.client.get('/%s/' % lang, **headers)
            self.assertEqual(response.context['billing_currency'], 'usd')
            self.assertIn('9', response.context['home_price'])
            self.assertIn('no-store', response['Cache-Control'])
        self.assertEqual(self.client.get('/en/', HTTP_CF_IPCOUNTRY='US').context['billing_currency'], 'eur')
        self.assertEqual(self.client.get('/en/', HTTP_CF_IPCOUNTRY='GB',
            HTTP_X_FORWARDED_FOR='173.245.48.1').context['billing_currency'], 'eur')

    def test_choice_persists_across_home_language_login_and_premium(self):
        response = self.client.get('/fr/?currency=usd')
        self.assertContains(response, '9 $ US / mois')
        self.client.force_login(self.user)
        response = self.client.get('/en/premium/')
        self.assertContains(response, '9.00 USD / month')
        self.assertContains(response, 'name="currency" value="usd"')
        self.assertNotContains(response, 'Subscribe yearly')
        self.assertEqual(self.client.get('/en/').context['billing_currency'], 'usd')
        self.assertEqual(self.client.get('/en/?currency=eur').context['billing_currency'], 'eur')
        self.assertEqual(self.client.get('/en/premium/').context['billing_currency'], 'eur')

    def test_usd_checkout_uses_fixed_price_currency_and_disables_conversion(self):
        self.client.force_login(self.user)
        with mock.patch('stripe.checkout.Session.create', return_value=mock.Mock(url='https://checkout.stripe.com/test')) as create:
            self.client.post('/en/premium/checkout/', {'period': 'monthly', 'currency': 'usd'})
        params = create.call_args.kwargs
        self.assertEqual(params['line_items'], [{'price': 'price_usd', 'quantity': 1}])
        self.assertEqual(params['currency'], 'usd')
        self.assertEqual(params['adaptive_pricing'], {'enabled': False})
        self.assertEqual(self.client.session['billing_currency'], 'usd')

    def test_rendered_form_currency_wins_over_session_from_another_tab(self):
        self.client.force_login(self.user)
        self.client.get('/en/?currency=eur')
        with mock.patch('stripe.checkout.Session.create', return_value=mock.Mock(url='https://checkout.stripe.com/test')) as create:
            self.client.post('/en/premium/checkout/', {'period': 'monthly', 'currency': 'usd'})
        self.assertEqual(create.call_args.kwargs['line_items'][0]['price'], 'price_usd')

    @override_settings(STRIPE_PRICE_YEARLY_USD='price_year_usd')
    def test_fixed_usd_yearly_price_is_displayed_and_used_by_checkout(self):
        cache.set('stripe_price_label_price_year_usd', '80.00 USD / year', 3600)
        self.client.force_login(self.user)
        page = self.client.get('/en/premium/?currency=usd')
        self.assertContains(page, '80.00 USD / year')
        self.assertContains(page, 'Subscribe yearly')
        with mock.patch('stripe.checkout.Session.create', return_value=mock.Mock(url='https://checkout.stripe.com/test')) as create:
            self.client.post('/en/premium/checkout/', {'period': 'yearly', 'currency': 'usd'})
        params = create.call_args.kwargs
        self.assertEqual(params['line_items'][0]['price'], 'price_year_usd')
        self.assertEqual(params['currency'], 'usd')
        self.assertEqual(params['adaptive_pricing'], {'enabled': False})

    def test_unconfigured_or_invalid_currency_never_creates_checkout(self):
        self.client.force_login(self.user)
        with mock.patch('stripe.checkout.Session.create') as create:
            for data in ({'currency': 'gbp'}, {'currency': 'usd', 'period': 'yearly'}):
                self.assertEqual(self.client.post('/en/premium/checkout/', data).status_code, 400)
            with override_settings(STRIPE_PRICE_MONTHLY_USD=''):
                self.assertEqual(self.client.post('/en/premium/checkout/', {'currency': 'usd'}).status_code, 400)
                self.assertEqual(self.client.get('/en/?currency=usd').context['billing_currency'], 'eur')
            create.assert_not_called()

    def test_existing_subscriber_cannot_open_new_subscription_in_another_currency(self):
        self.client.force_login(self.user)
        plan = UserPlan.objects.create(user=self.user, paid=True, stripe_subscription_id='sub_existing')
        with mock.patch('stripe.checkout.Session.create') as create:
            response = self.client.post('/en/premium/checkout/', {'currency': 'usd'})
            self.assertEqual(response['Location'], '/en/premium/')
            create.assert_not_called()
        plan.refresh_from_db()
        self.assertEqual(plan.stripe_subscription_id, 'sub_existing')
