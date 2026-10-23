"""Regression cases from public Crossref/ADS/OpenAlex records retrieved 2026-10-23."""
import json
from pathlib import Path
from unittest import mock

from django.core.cache import cache
from django.test import SimpleTestCase, override_settings

from summarizer import bibliography as bib, refcheck, sources
from summarizer.test_tools import ToolsTestCase
from summarizer.models import Generated

FIXTURES = Path(__file__).parent / 'fixtures' / 'bibliography'


def fixture(name):
    return json.loads((FIXTURES / (name + '.json')).read_text())


BIB = r'''@article{Andrews2018,
 author={Andrews, Sean M.}, title={Observations of Protoplanetary Disk Structures},
 journal={Annual Review of Astronomy and Astrophysics}, year={2019}, volume={58}, pages={483--528}}
@article{KenyonBromley2004,
 author={Kenyon, Scott J. and Bromley, Benjamin C.}, title={Formation and growth of objects in the Kuiper belt},
 journal={AJ}, year={2004}, volume={128}, pages={1916--1947}}
@article{ShakuraSunyaev1973,
 author={Shakura, N. I. and Sunyaev, R. A.}, title={Black holes in binary systems. Observational appearance.},
 journal={\aap}, year={1973}, volume={24}, pages={337}}
@article{Armitage2010,
 author={Armitage, Philip J.}, title={Astrophysics of Planet Formation}, year={2010}, publisher={Cambridge University Press}}
'''


@override_settings(ADS_TOKEN='test-fixture-token')
class BibliographyTests(SimpleTestCase):
    def setUp(self):
        cache.clear()

    def test_public_records_resolve_the_four_regressions_and_field_differences(self):
        entries = refcheck.parse_bibtex(BIB)
        expected = [{'year'}, {'title', 'pages'}, set(), {'type'}]
        for entry, name, changed in zip(entries, ['andrews', 'kenyon', 'shakura', 'armitage'], expected):
            with self.subTest(name=name):
                data = fixture(name)
                resolver = bib.Resolver()
                with mock.patch.object(resolver, 'crossref', return_value=[bib.crossref_record(w) for w in data['crossref']]), \
                     mock.patch.object(resolver, 'ads', return_value=[bib.ads_record(w) for w in data.get('ads', [])]), \
                     mock.patch.object(resolver, 'openalex', return_value=[bib.openalex_record(w) for w in data['openalex']]), \
                     mock.patch.object(resolver, 'doi', side_effect=lambda doi: next((bib.crossref_record(w) for w in data['crossref'] if w.get('DOI','').lower()==doi),None)):
                    record = resolver.resolve(entry)
                self.assertIsNotNone(record)
                self.assertEqual({c['field'] for c in bib.differences(entry, record)}, changed)
                if name == 'shakura':
                    self.assertEqual(record['source'], 'NASA ADS')
                    self.assertEqual(record['doi'], '')
                if name == 'kenyon':
                    self.assertEqual(record['fields']['pages'], '1916-1926')
                if name == 'armitage':
                    self.assertEqual(record['fields']['year'], '2010')
                    self.assertEqual(record['type'], 'book')

    def test_wrong_title_recovers_via_locator_and_enriches_ads_first_page_with_crossref_range(self):
        entry=refcheck.parse_bibtex(BIB)[1]
        data=fixture('kenyon'); resolver=bib.Resolver()
        queries=[]
        def ads(query):
            queries.append(query)
            return [bib.ads_record(w) for w in data['ads']] if 'page:' in query else []
        with mock.patch.object(resolver,'crossref',return_value=[]), mock.patch.object(resolver,'openalex',return_value=[]), \
             mock.patch.object(resolver,'ads',side_effect=ads), mock.patch.object(resolver,'doi',return_value=bib.crossref_record(data['crossref'][0])):
            record=resolver.resolve(entry)
        self.assertTrue(any('page:"1916"' in q and 'volume:"128"' in q for q in queries))
        self.assertEqual(record['doi'],'10.1086/423697')
        self.assertEqual(record['fields']['pages'],'1916-1926')

    def test_similar_conference_version_reprint_and_book_review_cannot_replace_original(self):
        shakura=refcheck.parse_bibtex(BIB)[2]
        data=fixture('shakura')
        for record in [bib.crossref_record(w) for w in data['crossref']]+[bib.openalex_record(w) for w in data['openalex']]:
            self.assertEqual(bib.match_score(shakura,record),0)
        book=refcheck.parse_bibtex(BIB)[3]
        data=fixture('armitage')
        review=bib.ads_record(data['ads'][-1])
        self.assertEqual(bib.match_score(book,review),0)
        self.assertEqual(bib.match_score(book,bib.crossref_record(data['crossref'][0])),0) # 2020 edition

    def test_ambiguous_matching_records_are_left_unresolved(self):
        entry=refcheck.parse_bibtex(BIB)[0]
        a=bib.crossref_record(fixture('andrews')['crossref'][0]); b=dict(a,doi='10.1000/another')
        record,ambiguous=bib.select(entry,[a,b])
        self.assertIsNone(record);self.assertTrue(ambiguous)
        unrelated=dict(a,authors=['someone else'])
        self.assertEqual(bib.match_score(entry,unrelated),0)

    def test_a_later_provider_cannot_silently_resolve_an_existing_ambiguity(self):
        entry=refcheck.parse_bibtex(BIB)[0]
        a=bib.crossref_record(fixture('andrews')['crossref'][0]); b=dict(a,doi='10.1000/another')
        resolver=bib.Resolver()
        with mock.patch.object(resolver,'crossref',return_value=[a,b]), mock.patch.object(resolver,'ads',return_value=[]), \
             mock.patch.object(resolver,'openalex',return_value=[a]):
            self.assertIsNone(resolver.resolve(entry))
        self.assertTrue(resolver.ambiguous)

    def test_locator_lookup_also_works_when_the_title_is_missing(self):
        entry=refcheck.parse_bibtex(BIB)[1];entry['fields'].pop('title')
        record=bib.crossref_record(fixture('kenyon')['crossref'][0]);resolver=bib.Resolver()
        with mock.patch.object(resolver,'ads',return_value=[]),mock.patch.object(resolver,'crossref',return_value=[record]):
            self.assertEqual(resolver.resolve(entry)['doi'],'10.1086/423697')

    def test_scientific_symbols_remain_distinct_and_a_slow_source_is_bounded(self):
        self.assertNotEqual(bib.normalized(r'$\alpha$ particles'),bib.normalized(r'$\beta$ particles'))
        resolver=bib.Resolver();resolver.deadline=0
        with mock.patch('summarizer.sources._json') as get:
            self.assertIsNone(resolver.doi('10.1000/test'));get.assert_not_called()
        self.assertTrue(resolver.limited)

    def test_typography_journal_macros_page_abbreviations_and_print_online_years_are_not_errors(self):
        entry=refcheck.parse_bibtex(BIB)[1]
        entry['fields'].update(title='The {Size Distribution} of Kuiper Belt Objects',journal=r'\aj',pages='1916–26',year='2003')
        record=bib.crossref_record(fixture('kenyon')['crossref'][0]);record['years'].append('2003')
        self.assertEqual(bib.differences(entry,record),[])
        entry['fields']['pages']='1916'
        self.assertEqual(bib.differences(entry,record),[])

    def test_cached_metadata_avoids_repeated_calls_and_failures_are_not_cached_as_missing(self):
        resolver=bib.Resolver();doi='10.1086/423697'
        body={'message':fixture('kenyon')['crossref'][0]}
        with mock.patch('summarizer.sources._json',return_value=body) as get:
            self.assertIsNotNone(resolver.doi(doi));self.assertIsNotNone(resolver.doi(doi));self.assertEqual(get.call_count,1)
        cache.clear()
        with mock.patch('summarizer.sources._json',side_effect=sources.SourceUnavailable('offline')) as get:
            self.assertIsNone(resolver.doi(doi));self.assertIsNone(resolver.doi(doi));self.assertEqual(get.call_count,2)
        self.assertIn('Crossref',resolver.unavailable)

    def test_request_and_openalex_budget_caps_are_explicit(self):
        resolver=bib.Resolver(openalex_allowed=lambda:False)
        with mock.patch('summarizer.sources._json') as get:
            self.assertEqual(resolver.openalex('A title'),[]);get.assert_not_called()
            resolver.remaining=0
            self.assertIsNone(resolver.doi('10.1000/test'));get.assert_not_called()
        self.assertTrue(resolver.limited)

    def test_suggested_correction_preserves_book_type_key_and_unrelated_fields(self):
        entry=refcheck.parse_bibtex(BIB)[3];entry['fields']['note']='Keep my note';entry['fields']['howpublished']='Keep this field too'
        text=refcheck.with_fields(entry,{'isbn':'9780521887458'},'book')
        parsed=refcheck.parse_bibtex(text)[0]
        self.assertEqual(parsed['key'],'Armitage2010');self.assertEqual(parsed['type'],'book')
        self.assertEqual(parsed['fields']['howpublished'],'Keep this field too');
        self.assertEqual(parsed['fields']['note'],'Keep my note');self.assertEqual(parsed['fields']['year'],'2010')


class BibliographyReportTests(ToolsTestCase):
    def test_reports_separate_discrepancies_unresolved_and_identified_without_model_calls(self):
        entries=refcheck.parse_bibtex(BIB)
        data={entries[0]['key']:bib.crossref_record(fixture('andrews')['crossref'][0]),
              entries[1]['key']:bib.crossref_record(fixture('kenyon')['crossref'][0]),
              entries[2]['key']:bib.ads_record(fixture('shakura')['ads'][0]),
              entries[3]['key']:bib.ads_record(fixture('armitage')['ads'][0])}
        source=BIB+'\n@article{unresolved,title={A paper we cannot match},author={Unknown, A.},year={2022}}'
        with mock.patch('summarizer.bibliography.Resolver.resolve',side_effect=lambda entry,*args:data.get(entry['key'])), \
             mock.patch('summarizer.refcheck.openalex_works',return_value={}), \
             mock.patch('summarizer.refcheck.arxiv_records',return_value={}), mock.patch('summarizer.generated.chat_json') as model:
            result=refcheck.check(source,'','fr')
        model.assert_not_called()
        self.assertEqual(result['summary'],{'incorrect':3,'identified':1,'published':0,'unverified':1})
        self.assertEqual(result['corrected'],3)
        self.assertIn('@book{Armitage2010,',result['fixed_bib'])
        self.assertIn('year = {2020}',result['fixed_bib'])
        self.assertIn('pages = {1916-1926}',result['fixed_bib'])
        self.assertEqual(result['identified'][0]['key'],'ShakuraSunyaev1973')
        user=self.user(paid=True)
        row=Generated.objects.create(kind='refcheck',key='regression',lang='fr',user=user,status='done',content=json.dumps(result))
        page=self.client_for(user).get('/fr/refcheck/regression/')
        self.assertContains(page,'1 référence non résolue');self.assertNotContains(page,'1 références');
        self.assertContains(page,'2019');self.assertContains(page,'2020');self.assertContains(page,'NASA ADS')
        self.assertContains(page,'1973A%26A....24..337S')
        self.assertContains(page,'10.1086/423697')
        self.assertEqual(self.client_for(self.user('other',paid=True)).get('/fr/refcheck/regression/').status_code,404)

    def test_unresolved_doi_and_lookup_limit_are_not_treated_as_valid(self):
        with mock.patch('summarizer.bibliography.Resolver.resolve',return_value=None), \
             mock.patch('summarizer.refcheck.openalex_works',return_value={}), mock.patch('summarizer.refcheck.TITLE_SEARCHES',0):
            result=refcheck.check('@article{a,doi={10.1000/missing}}\n@article{b,title={Unknown},author={Unknown, A.}}','','en')
        self.assertEqual(result['summary']['unverified'],2)
        self.assertEqual(result['counts']['warning'],0)
        self.assertTrue(result['lookup_limited'])
        self.assertEqual(result['fixed_bib'],'')
