import csv
import datetime
import io

from django.contrib.auth.models import User
from django.test import TestCase
from django.utils import timezone

from . import usage
from .models import AIPrice, AIUsage


class UsageReportingTests(TestCase):
    def setUp(self):
        self.today = timezone.localdate()
        self.staff = User.objects.create_superuser('cost-admin', 'cost@example.com', 'test')
        self.reader = User.objects.create_user('reader-costs')
        self.client.force_login(self.staff)
        AIPrice.objects.create(model='cost-small', input_per_million=2, output_per_million=10)
        AIPrice.objects.create(model='cost-large', input_per_million=20, output_per_million=30)
        self.small = self.row('summary', 'cost-small', 4, 1_000_000, 100_000, user=self.reader)
        self.large = self.row('presskit', 'cost-large', 2, 2_000_000, 1_000_000)
        self.base = '/admin/summarizer/aiusage/'

    def row(self, task, model, calls, prompt, completion, **kwargs):
        return AIUsage.objects.create(day=kwargs.pop('day', self.today), task=task, model=model,
                                      calls=calls, prompt_tokens=prompt, completion_tokens=completion, **kwargs)

    def test_breakdowns_reconcile_and_identify_volume_and_unit_cost(self):
        report = usage.summary(7)
        self.assertEqual(report['total'], 73)
        self.assertEqual(report['totals']['input_cost'], 42)
        self.assertEqual(report['totals']['output_cost'], 31)
        self.assertEqual(report['totals']['calls'], 6)
        for name in ('tasks', 'models', 'combinations', 'daily', 'users'):
            self.assertEqual(sum(r['cost'] for r in report[name]), 73)
        self.assertEqual(report['tasks'][0]['name'], 'presskit')
        self.assertEqual(report['tasks'][0]['average'], 35)
        self.assertAlmostEqual(report['tasks'][0]['share'], 70 / 73 * 100)
        self.assertIn('Sans utilisateur', report['users'][0]['name'])

    def test_tool_groups_include_subtasks_and_can_be_drilled_down(self):
        self.row('insight', 'cost-small', 1, 1_000_000, 0)
        self.row('evidence', 'cost-small', 1, 1_000_000, 0)
        page = self.client.get(self.base)
        tools = page.context['usage']['tools']
        summaries = next(r for r in tools if r['name'] == 'summaries')
        self.assertEqual(summaries['cost'], 7)
        drilldown = self.client.get(self.base + summaries['url'])
        self.assertEqual(drilldown.status_code, 200)
        self.assertEqual(drilldown.context['usage']['total'], 7)
        self.assertEqual({r['name'] for r in drilldown.context['usage']['tasks']}, {'summary', 'insight', 'evidence'})

    def test_unknown_prices_are_partial_not_free_or_a_false_average(self):
        self.row('summary', 'cost-unknown', 5, 9_000, 10)
        report = usage.summary()
        self.assertEqual(report['total'], 73)
        self.assertEqual(report['unpriced'], ['cost-unknown'])
        self.assertEqual(report['totals']['unpriced_calls'], 5)
        summary = next(r for r in report['tasks'] if r['name'] == 'summary')
        self.assertEqual(summary['calls'], 9)
        self.assertEqual(summary['cost'], 3)
        self.assertIsNone(summary['average'])
        page = self.client.get(self.base)
        self.assertContains(page, 'Coût incomplet')
        self.assertContains(page, 'cost-unknown')

    def test_snapshot_resolution_is_consistent_with_the_existing_cost(self):
        row = self.row('summary', 'cost-small-2026-10-01', 1, 8_000, 400)
        prices = {p.model: p for p in AIPrice.objects.all()}
        self.assertAlmostEqual(sum(usage.costs(row, prices)), row.cost(prices))
        AIPrice.objects.create(model='cost-small-special', input_per_million=7, output_per_million=8)
        row.model = 'cost-small-special-2026-10-01'
        prices = {p.model: p for p in AIPrice.objects.all()}
        self.assertEqual(usage.price_for(row.model, prices).model, 'cost-small-special')
        self.assertAlmostEqual(sum(usage.costs(row, prices)), row.cost(prices))

    def test_default_dates_presets_and_custom_range_match_the_list(self):
        old = self.row('older', 'cost-small', 1, 1_000_000, 0,
                       day=self.today - datetime.timedelta(days=40))
        recent = self.row('week-two', 'cost-small', 1, 1_000_000, 0,
                          day=self.today - datetime.timedelta(days=8))
        future = self.row('future', 'cost-small', 1, 1_000_000, 0,
                          day=self.today + datetime.timedelta(days=1))
        for params, expected in [({}, 75), ({'periode': '7'}, 73), ({'periode': '90'}, 77),
                                 ({'periode': 'all'}, 79),
                                 ({'day__gte': old.day.isoformat(), 'day__lte': old.day.isoformat()}, 2)]:
            with self.subTest(params=params):
                page = self.client.get(self.base, params)
                self.assertEqual(page.status_code, 200)
                self.assertEqual(page.context['usage']['total'], expected)
                self.assertAlmostEqual(sum(r.cost() for r in page.context['cl'].queryset), expected)

    def test_task_model_search_and_user_drilldown_are_applied_to_all_tables(self):
        for params in ({'task__exact': 'summary'}, {'model__exact': 'cost-small'},
                       {'q': 'reader-costs'}, {'user__id__exact': self.reader.pk}):
            page = self.client.get(self.base, params)
            self.assertEqual(page.status_code, 200)
            self.assertEqual(page.context['usage']['total'], 3)
            self.assertEqual(len(page.context['usage']['models']), 1)
            self.assertIn('task__exact=summary', page.context['usage']['tasks'][0]['url'])
        page = self.client.get(self.base, {'user__isnull': 'True'})
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.context['usage']['total'], 70)

    def test_csv_uses_the_same_filters_and_leaves_missing_costs_empty(self):
        self.row('summary', 'cost-unknown', 5, 1, 2)
        response = self.client.get(self.base + 'export.csv', {'task__exact': 'summary', 'periode': '7'})
        self.assertEqual(response.status_code, 200)
        rows = list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
        self.assertEqual(len(rows), 2)
        priced = next(r for r in rows if r['Tarif connu'] == 'oui')
        missing = next(r for r in rows if r['Tarif connu'] == 'non')
        self.assertEqual(float(priced['Coût total USD']), 3)
        self.assertEqual(missing['Coût total USD'], '')
        self.assertEqual(missing['Coût entrée USD'], '')

    def test_csv_is_permission_checked_and_does_not_export_spreadsheet_formulas(self):
        self.reader.username = '=1+1'
        self.reader.save()
        response = self.client.get(self.base + 'export.csv')
        rows = list(csv.DictReader(io.StringIO(response.content.decode('utf-8-sig'))))
        self.assertEqual(next(r for r in rows if r['Utilisateur'])['Utilisateur'], "'=1+1")
        self.client.logout()
        self.assertEqual(self.client.get(self.base + 'export.csv').status_code, 302)
        staff = User.objects.create_user('staff-no-cost-access', is_staff=True)
        self.client.force_login(staff)
        self.assertEqual(self.client.get(self.base + 'export.csv').status_code, 403)

    def test_empty_selection_and_invalid_dates_do_not_crash(self):
        page = self.client.get(self.base, {'q': 'no-such-usage-record'})
        self.assertEqual(page.status_code, 200)
        self.assertEqual(page.context['usage']['total'], 0)
        self.assertContains(page, 'Aucun usage enregistré')
        self.assertEqual(self.client.get(self.base, {'day__gte': 'invalid'}).status_code, 302)
        self.assertEqual(self.client.get(self.base + 'export.csv', {'day__gte': 'invalid'}).status_code, 400)
