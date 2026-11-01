"""The bibliography check: identification and its corrections, the citations read against the cited papers,
the inputs it reads, the report's marks, downloads and next steps, and the watch until publication."""
import contextlib
import datetime
import gzip
import io
import json
import zipfile
from unittest import mock

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.utils import timezone

from summarizer import refcheck, refcheck_claims, refcheck_sources, refcheck_watch
from summarizer.models import Collection, Generated, PublicationAlert, ReferenceWatch
from summarizer.test_tools import ToolsTestCase

BIB = """@article{retracted, author = {Roe, Al}, title = {A retracted result}, journal = {J}, year = {2019}, doi = {10.1000/RETRACTED}}
@misc{pre, author = {Lee, Ann}, title = {{Clouds} on a "hot" planet}, eprint = {2101.00001v1}, archivePrefix = {arXiv}}
@misc{again, title = "Clouds again", howpublished = {arXiv:2101.00001}}
@book{nothing, title = {A book with no identifier}}
@string{foo = "bar"}
"""
TEX = r"""\documentclass{article}\begin{document}
Clouds are thick on hot planets \citep{pre}. % \cite{retracted} in a comment is not read
Earlier work, e.g. Lee et al., found no clouds at all \Citet[see][]{pre,again}.
A key missing from the .bib \cite{ghost} is reported too.
\end{document}"""
RECORD = {'arxiv_id': '2101.00001v3', 'title': 'Clouds on a hot planet', 'abstract': 'We see thick clouds.',
          'doi': '10.1093/mnras/x1', 'journal_ref': 'MNRAS 500', 'comment': '', 'version': '3'}
WORKS = {
    '10.1000/retracted': {'title': ['A retracted result'], 'updated-by': [{'type': 'retraction', 'DOI': '10.1000/notice'}]},
    '10.1093/mnras/x1': {'title': ['Clouds'], 'container-title': ['MNRAS & Co'], 'published': {'date-parts': [[2021]]},
                         'volume': '500', 'page': '1-10'},
}


@contextlib.contextmanager
def lookups(records=None, works=None, openalex=None, resolved=None, s2=None, full_text=None):
    """Every source of the check answered from the test, nothing from the network."""
    with mock.patch('summarizer.refcheck.arxiv_records', return_value=records if records is not None else {'2101.00001': RECORD}), \
         mock.patch('summarizer.refcheck.openalex_works', return_value=openalex or {}), \
         mock.patch('summarizer.refcheck.s2_published', return_value=s2 or {}), \
         mock.patch('summarizer.bibliography.Resolver.resolve', side_effect=lambda entry, *a: (resolved or {}).get(entry['key'])), \
         mock.patch('summarizer.bibliography.Resolver.work', side_effect=lambda doi: (works if works is not None else WORKS).get(doi, {})), \
         mock.patch('summarizer.refcheck.biorxiv_published', side_effect=lambda doi: {'10.1101/2020.02.02.2': '10.1038/later'}.get(doi, '')), \
         mock.patch('summarizer.refcheck_claims.full_text', side_effect=lambda paper: (full_text or {}).get(paper['arxiv'] or paper['doi'])):
        yield


def matched(doi='', **fields):
    return {'doi': doi, 'fields': dict({'title': 'T'}, **fields), 'type': 'article', 'source': 'Crossref',
            'url': 'https://doi.org/' + doi, 'years': [fields.get('year', '')], 'arxiv': ''}


class ReadingTests(ToolsTestCase):
    def test_sentences_keep_their_file_and_line_and_every_citation_command(self):
        sentences = refcheck_claims.citing_sentences([('main.tex', TEX)])
        self.assertEqual([(s['line'], s['keys']) for s in sentences], [(2, ['pre']), (3, ['pre', 'again']), (4, ['ghost'])])
        self.assertEqual(sentences[1]['sentence'], 'Earlier work, e.g. Lee et al., found no clouds at all [pre, again].')
        self.assertEqual(sentences[1]['previous'], 'Clouds are thick on hot planets [pre].')
        self.assertEqual(refcheck_claims.citations(r'\cites[see][]{a,b}[p.~2]{c} and \nocite{d} \citeyear{e}'),
                         [(0, 27, ['a', 'b', 'c']), (43, 55, ['e'])])
        self.assertEqual(refcheck_claims.cited_keys([('m', r'\cite{a} \nocite{*}')]), (['a'], True))
        self.assertEqual(refcheck_claims.readable(r'The value $H_0 = 73.0 \pm 1.0$~\citep{b} in \S\ref{s}.'),
                         r'The value $H_0 = 73.0 \pm 1.0$ [b] in ….')

    def test_sentences_that_attribute_a_result_are_read_first(self):
        claims = [{'sentence': 'See e.g. the reviews [a, b, c, d, e].', 'keys': list('abcde'), 'order': 0},
                  {'sentence': 'Smith showed a 12.5% excess [s].', 'keys': ['s'], 'order': 1},
                  {'sentence': 'Stars form in clouds [c].', 'keys': ['c'], 'order': 2}]
        self.assertEqual([c['order'] for c in refcheck_claims.chosen(claims, limit=2)], [1, 2])

    def test_an_overleaf_archive_gives_its_files_in_reading_order_and_its_bib(self):
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('p/main.tex', '\\begin{document}\nIntro \\input{sec/a}\n\\bibliography{refs}\n\\end{document}')
            z.writestr('p/sec/a.tex', 'We use the method of \\citet{m} on our data.')
            z.writestr('p/refs.bib', '@article{m, title={M}}')
            z.writestr('p/other.bib', '@article{z, title={Z}}')
            z.writestr('p/figure.png', b'png')
        files = refcheck_sources.archive_files(archive.getvalue())
        texts = refcheck_sources.manuscript(files)
        self.assertEqual([name for name, _ in texts], ['p/main.tex', 'p/sec/a.tex'])
        self.assertEqual(refcheck_sources.references_of(files, texts), ('@article{m, title={M}}', 'bib'))
        self.assertEqual(sorted(refcheck_sources.archive_files(gzip.compress(b'\\begin{document}x'), name='2401.1')),
                         ['2401.1.tex'])
        with self.assertRaises(refcheck_sources.Unreadable):
            refcheck_sources.archive_files(b'not an archive')

    def test_a_bibliography_written_in_the_latex_is_read_as_a_bbl(self):
        files = {'ms.tex': '\\begin{document}\nText \\cite{a}.\n\\begin{thebibliography}{1}\n\\bibitem{a} Lee, A. 2020, ApJ, 1, 2\n\\end{thebibliography}\n\\end{document}'}
        rows = {'references': {'0': ['Lee', '2020', '', 'ApJ', '1', '2', '', '']}}
        with mock.patch('summarizer.generated.chat_json', return_value=rows):
            bib, origin = refcheck_sources.references_of(files, refcheck_sources.manuscript(files))
        self.assertEqual((origin, [e['key'] for e in refcheck.parse_bibtex(bib)]), ('bbl', ['a']))

    def test_a_bbl_without_bib_becomes_entries_that_are_never_corrected(self):
        bbl = '\\bibitem[Lee(2020)]{lee20} Lee, A. 2020, ApJ, 900, 1, doi:10.3847/abc\n\\bibitem{roe} Roe, arXiv:2101.00001'
        rows = {'references': {'0': ['Lee', '2020', 'Clouds', 'ApJ', '900', '1', '', ''], '1': ['Roe', '', '', '', '', '', '', '']}}
        with mock.patch('summarizer.generated.chat_json', return_value=rows):
            bib = refcheck_sources.bbl_bibtex(bbl)
        entries = refcheck.parse_bibtex(bib)
        self.assertEqual([(e['key'], e['type']) for e in entries], [('lee20', 'article'), ('roe', 'misc')])
        self.assertEqual(entries[0]['fields']['doi'], '10.3847/abc')
        self.assertEqual(entries[1]['fields']['eprint'], '2101.00001')
        with lookups(works={}):
            result = refcheck.check(bib, '', 'en', origin='bbl')
        self.assertEqual((result['fixed_bib'], result['corrections']), ('', {}))


class CheckTests(ToolsTestCase):
    def test_check_reports_problems_and_corrects_the_bib(self):
        verdicts = {'items': [{'id': '1', 'verdict': 'contradicted', 'keys': ['pre'], 'use': 'result', 'reason': 'It finds clouds.'},
                              {'id': '0', 'verdict': 'supported'}]}
        with lookups(), mock.patch('summarizer.generated.chat_json', return_value=verdicts) as chat:
            result = refcheck.check(BIB, TEX, 'en')
        kinds = {p['key']: [i['kind'] for i in p['issues']] for p in result['problems']}
        self.assertEqual(kinds, {'retracted': ['retracted'], 'pre': ['published', 'old_version'],
                                 'again': ['duplicate', 'published'], 'nothing': ['unknown', 'incomplete']})
        self.assertEqual(result['problems'][0]['issues'][0]['notice'], '10.1000/notice')
        again = next(p for p in result['problems'] if p['key'] == 'again')
        self.assertEqual((again['issues'][0]['other'], again['issues'][0]['pair']), ('pre', False))  # both cite the preprint
        self.assertIn('journal = {MNRAS \\& Co}', result['fixed_bib'])
        self.assertIn('pages = {1-10}', result['fixed_bib'])
        self.assertIn('@article{pre,', result['fixed_bib'])
        self.assertNotIn('archiveprefix', result['fixed_bib'])
        self.assertEqual(sorted(result['corrections']), ['again', 'pre'])
        # each sentence is judged once, with all the papers it cites
        self.assertEqual((result['claims_checked'], result['claims_citing']), (2, 3))
        self.assertEqual(chat.call_count, 1)
        self.assertEqual(chat.call_args[1]['system'].count('We see thick clouds.'), 3)
        claim = result['claims'][0]
        self.assertEqual((claim['key'], claim['verdict'], claim['line'], claim['keys']), ('pre', 'contradicted', 3, ['pre', 'again']))
        self.assertEqual(claim['papers'][0]['chat'], '2101.00001')
        self.assertEqual((result['undefined'], result['uncited']), (['ghost'], ['retracted', 'nothing']))
        self.assertEqual({r['key'] for r in result['refs']}, {'retracted', 'pre', 'again'})

    def test_a_journal_article_with_an_eprint_and_no_doi_is_not_a_preprint(self):
        bib = '@article{a, author={Lee, Ann}, title={Clouds}, journal={ApJ}, year={2021}, eprint={2101.00001}}'
        with lookups():
            result = refcheck.check(bib, '', 'en')
        self.assertEqual(result['problems'], [])

    def test_a_preprint_and_its_published_version_cited_twice_are_one_work(self):
        bib = '@misc{pre, title={Clouds}, eprint={2101.00001}}\n@article{pub, title={Clouds}, journal={MNRAS}, doi={10.1093/mnras/x1}}'
        with lookups(resolved={'pub': matched('10.1093/mnras/x1', journal='MNRAS')}):
            result = refcheck.check(bib, '', 'en')
        duplicate = next(i for p in result['problems'] if p['key'] == 'pub' for i in p['issues'] if i['kind'] == 'duplicate')
        self.assertEqual((duplicate['other'], duplicate['pair']), ('pre', True))

    def test_a_preprint_server_doi_published_since_and_notices_of_the_published_paper(self):
        bib = '@article{bio, author={Lee, Ann}, title={Cells}, journal={bioRxiv}, publisher={Cold Spring Harbor}, year={2020}, doi={10.1101/2020.01.01.1}}'
        works = {'10.1101/2020.01.01.1': {'title': ['Cells'], 'type': 'posted-content',
                                          'relation': {'is-preprint-of': [{'id-type': 'doi', 'id': '10.1038/cells'}]}},
                 '10.1038/cells': {'title': ['Cells'], 'container-title': ['Nature'], 'published': {'date-parts': [[2021]]},
                                   'updated-by': [{'type': 'expression_of_concern', 'DOI': '10.1038/eoc'},
                                                  {'type': 'correction', 'DOI': '10.1038/fix'}]}}
        with lookups(records={}, works=works):
            result = refcheck.check(bib, '', 'en')
        kinds = [i['kind'] for i in result['problems'][0]['issues']]
        self.assertEqual(kinds, ['concern', 'corrected', 'published'])
        after = result['corrections']['bio']['after']
        self.assertIn('journal = {Nature}', after)
        self.assertIn('doi = {10.1038/cells}', after)
        self.assertNotIn('publisher', after)

    def test_biorxiv_gives_the_published_version_crossref_does_not_link(self):
        bib = '@misc{bio, author={Lee, Ann}, title={Cells}, year={2020}, doi={10.1101/2020.02.02.2}}'
        works = {'10.1101/2020.02.02.2': {'title': ['Cells'], 'type': 'posted-content'},
                 '10.1038/later': {'title': ['Cells'], 'container-title': ['Nature']}}
        with lookups(records={}, works=works):
            result = refcheck.check(bib, '', 'en')
        self.assertEqual([(i['kind'], i.get('doi')) for i in result['problems'][0]['issues']], [('published', '10.1038/later')])

    def test_a_retraction_label_in_the_title_of_the_record_is_not_a_correction(self):
        from summarizer import bibliography
        entry = refcheck.parse_bibtex('@article{w, title={Ileal-lymphoid-nodular hyperplasia in children}, year={1998}}')[0]
        record = matched('10.1016/x', title='RETRACTED: Ileal-lymphoid-nodular hyperplasia in children', year='1998')
        self.assertEqual(bibliography.differences(entry, record), [])

    def test_a_missing_doi_volume_and_pages_are_filled_from_the_record_that_identifies_the_entry(self):
        bib = '@article{a, author={Lee, Ann}, title={A long enough title for matching}, journal={ApJ}, year={2021}}'
        record = matched('10.3847/a', title='A long enough title for matching', journal='ApJ', year='2021', volume='9', pages='1-5')
        with lookups(records={}, resolved={'a': record}):
            result = refcheck.check(bib, '', 'en')
        issue = result['problems'][0]['issues'][0]
        self.assertEqual((issue['kind'], issue['filled']), ('incomplete', {'doi': '10.3847/a', 'volume': '9', 'pages': '1-5'}))
        self.assertIn('doi = {10.3847/a}', result['fixed_bib'])

    def test_flagged_sentences_are_read_again_in_the_full_text_with_a_passage_found_word_for_word(self):
        text = ('# Results\n\n' + 'Filler paragraph about telescopes and instruments. ' * 40 +
                '\n\nWe detect thick clouds on the dayside of the hot planet at high significance.\n\n' +
                'Another filler paragraph. ' * 60)
        paper = {'text': text, 'source': 'html', 'arxiv_id': '2101.00001'}
        answers = [{'items': [{'id': '0', 'verdict': 'unclear', 'keys': ['pre'], 'reason': 'Not in the abstract.'},
                              {'id': '1', 'verdict': 'unclear', 'keys': ['pre'], 'reason': 'Not said.'}]},
                   {'items': [{'id': '0', 'verdict': 'supported', 'paper': 'pre',
                               'quote': 'We detect thick clouds on the dayside of the hot planet', 'reason': 'Said.'},
                              {'id': '1', 'verdict': 'supported', 'paper': 'pre', 'quote': 'A sentence the paper never wrote at all.'}]}]
        with lookups(full_text={'2101.00001': paper}), mock.patch('summarizer.generated.chat_json', side_effect=answers):
            result = refcheck.check(BIB, TEX, 'en')
        self.assertEqual(len(result['confirmed']), 1)
        passage = result['confirmed'][0]['passage']
        self.assertEqual((passage['key'], passage['location']), ('pre', 'Results'))
        self.assertIn('arxiv.org/html/2101.00001', passage['link'])
        self.assertEqual([(c['verdict'], c.get('not_in_full_text')) for c in result['claims']], [('unclear', True)])
        self.assertNotIn('passage', result['claims'][0])

    def test_a_value_the_cited_paper_never_states_keeps_the_sentence_flagged(self):
        tex = '\\begin{document}\nThe planet is 12.7 times hotter than expected \\citep{pre}.\n\\end{document}'
        text = 'The planet is hot, hotter than expected. ' * 80
        answers = [{'items': [{'id': '0', 'verdict': 'unclear', 'keys': ['pre']}]},
                   {'items': [{'id': '0', 'verdict': 'supported', 'paper': 'pre', 'quote': 'The planet is hot, hotter than expected.'}]}]
        with lookups(full_text={'2101.00001': {'text': text, 'source': 'pdf', 'arxiv_id': '2101.00001'}}), \
             mock.patch('summarizer.generated.chat_json', side_effect=answers):
            result = refcheck.check(BIB, tex, 'en')
        self.assertEqual((result['claims'][0]['verdict'], result['claims'][0]['numbers_absent']), ('unclear', ['12.7']))

    def test_a_check_again_keeps_the_marks_and_says_what_is_solved_and_new(self):
        previous = Generated(key='old', created=timezone.now(), content=json.dumps({
            'problems': [{'key': 'pre', 'issues': [{'kind': 'published', 'mark': 'intended'}]},
                         {'key': 'gone', 'issues': [{'kind': 'retracted'}]}],
            'claims': [{'sentence': 'Clouds are thick on hot planets [pre].', 'mark': 'fixed'}]}))
        result = {'problems': [{'key': 'pre', 'issues': [{'kind': 'published'}, {'kind': 'old_version'}]}],
                  'claims': [{'sentence': 'Clouds are thick on hot planets  [pre].'}, {'sentence': 'Another one.'}]}
        solved = refcheck.carried(result, previous.data())
        self.assertEqual(solved, [{'key': 'gone', 'kind': 'retracted'}])
        self.assertEqual(result['problems'][0]['issues'], [{'kind': 'published', 'mark': 'intended'}, {'kind': 'old_version', 'new': True}])
        self.assertEqual([c.get('mark') or c.get('new') for c in result['claims']], ['fixed', True])


class PageTests(ToolsTestCase):
    def report(self, user, **extra):
        with lookups(), mock.patch('summarizer.generated.chat_json', return_value={'items': [
                {'id': '1', 'verdict': 'contradicted', 'keys': ['pre'], 'reason': 'It finds clouds.'}]}):
            result = refcheck.check(BIB, TEX, 'en')
        result.update(extra)
        return Generated.objects.create(kind='refcheck', key='k1', user=user, status='done', content=json.dumps(result))

    def test_page_is_premium_and_report_is_private(self):
        self.assertContains(self.client_for(self.user()).get('/en/refcheck/'), 'Discover Premium')
        client = self.client_for(self.user('pro', paid=True))
        self.assertContains(client.post('/en/refcheck/', {'bibtext': 'no entries here'}), 'We found no BibTeX entry')
        self.assertContains(client.post('/en/refcheck/', {'bibtext': '', 'arxiv': 'not an id'}), 'is not an arXiv identifier')
        result = {'entries': 1, 'problems': [], 'claims': [], 'claims_checked': 0, 'with_tex': False,
                  'counts': {'error': 0, 'warning': 0, 'info': 0}, 'fixed_bib': '@article{a,\n}', 'corrected': 1}
        with mock.patch('summarizer.refcheck.check', return_value=result) as check:
            r = client.post('/en/refcheck/', {'bibtext': '@misc{a, title={T}}'})
        self.assertEqual((check.call_args[0][1], check.call_args[1]['origin']), ([], 'bib'))
        self.assertContains(client.get(r['Location']), 'No discrepancy detected')
        self.assertEqual(client.get(r['Location'] + 'bib/').content, b'@article{a,\n}')
        self.assertEqual(self.client_for(self.user('other', paid=True)).get(r['Location']).status_code, 404)

    def test_an_overleaf_archive_or_an_arxiv_id_is_read_in_the_background(self):
        client = self.client_for(self.user('pro', paid=True))
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, 'w') as z:
            z.writestr('main.tex', TEX)
            z.writestr('refs.bib', BIB)
        with mock.patch('summarizer.refcheck.check', return_value={'entries': 4, 'problems': []}) as check:
            client.post('/en/refcheck/', {'source': SimpleUploadedFile('paper.zip', archive.getvalue())})
        self.assertEqual(check.call_args[0][0], BIB)
        self.assertEqual(check.call_args[0][1][0][0], 'main.tex')
        with mock.patch('summarizer.refcheck_sources.arxiv_source', side_effect=refcheck_sources.Unreadable('arxiv_pdf_only')):
            r = client.post('/en/refcheck/', {'arxiv': 'arXiv:2401.01234'})
        self.assertContains(client.get(r['Location']), 'only its PDF')

    def test_marks_filters_and_the_choice_of_corrections(self):
        user = self.user('pro', paid=True)
        client = self.client_for(user)
        self.report(user)
        page = client.get('/en/refcheck/k1/')
        self.assertContains(page, 'data-filter="error"')
        self.assertContains(page, 'main.tex, line 3')
        self.assertContains(page, '/en/chat/?ids=2101.00001&amp;ask=')
        self.assertContains(page, '/en/changes/2101.00001/')
        self.assertContains(page, 'Missing from the .bib')
        client.post('/en/refcheck/k1/', {'action': 'mark', 'target': 'pre|old_version', 'mark': 'intended'})
        client.post('/en/refcheck/k1/', {'action': 'mark', 'target': 'claim|0', 'mark': 'fixed'})
        shown = refcheck.shown(Generated.objects.get(key='k1'))
        pre = next(p for p in shown['problems'] if p['key'] == 'pre')
        self.assertEqual([i['kind'] for i in pre['marked']], ['old_version'])
        self.assertEqual((shown['open_claims'], len(shown['marked_claims'])), ([], 1))
        whole = client.get('/en/refcheck/k1/bib/', {'sel': '1', 'k': ['again']}).content.decode()
        self.assertIn('@misc{pre,', whole)
        self.assertIn('@article{again,', whole)
        only = client.get('/en/refcheck/k1/bib/', {'sel': '1', 'k': ['pre'], 'only': '1'}).content.decode()
        self.assertTrue(only.startswith('@article{pre,') and '@book' not in only and 'again' not in only)

    def test_references_are_kept_in_a_collection_and_watched_until_publication(self):
        user = self.user('pro', paid=True)
        client = self.client_for(user)
        self.report(user)
        with mock.patch('summarizer.collection_tools.start_import') as start:
            r = client.post('/en/refcheck/k1/', {'action': 'collection'})
        collection = Collection.objects.get(user=user)
        self.assertEqual(r['Location'], '/en/collections/%s/' % collection.pk)
        self.assertEqual(sorted(start.call_args[0][2]), ['2101.00001', 'doi_10.1000--retracted'])
        client.post('/en/refcheck/k1/', {'action': 'watch'})
        self.assertContains(client.get('/en/refcheck/k1/'), 'Your references are watched')
        client.post('/en/refcheck/k1/', {'action': 'unwatch'})
        self.assertFalse(ReferenceWatch.objects.exists())

    def test_what_did_i_miss_runs_on_the_same_manuscript_when_asked(self):
        client = self.client_for(self.user('pro', paid=True))
        with mock.patch('summarizer.refcheck.check', return_value={'entries': 4, 'problems': []}), \
             mock.patch('summarizer.missed.write', return_value=lambda pk: {'items': []}) as write:
            long_tex = TEX.replace('\\end{document}', 'Our method reads the clouds of many hot planets at once. ' * 5 + '\\end{document}')
            r = client.post('/en/refcheck/', {'bibtext': BIB, 'textext': long_tex, 'missed': '1'})
        row = Generated.objects.get(key=r['Location'].split('/')[-2])
        self.assertTrue(Generated.objects.filter(kind='missed', key=row.data()['missed_key']).exists())
        self.assertIn('Clouds are thick on hot planets', write.call_args[0][1])
        self.assertContains(client.get(r['Location']), '/en/missed/%s/' % row.data()['missed_key'])

    def test_what_did_i_miss_checks_the_same_bib_when_asked(self):
        from summarizer import missed
        client = self.client_for(self.user('pro', paid=True))
        with mock.patch('summarizer.refcheck.check', return_value={'entries': 4, 'problems': []}), \
             mock.patch('summarizer.missed.find', return_value={'items': []}):
            client.post('/en/missed/', {'text': 'x' * 300, 'refcheck': '1',
                                        'bib': SimpleUploadedFile('refs.bib', BIB.encode())})
        self.assertEqual(Generated.objects.filter(kind='refcheck').count(), 1)
        self.assertEqual(missed.Generated.objects.get(kind='missed').data()['refcheck_key'],
                         Generated.objects.get(kind='refcheck').key)


class WatchTests(ToolsTestCase):
    def test_what_changes_after_the_first_check_is_emailed_with_the_publication_alerts(self):
        from django.core import mail
        from summarizer import publication_alerts
        user = self.user('pro', paid=True)
        row = Generated.objects.create(kind='refcheck', key='k1', user=user, status='done', content='{}')
        refcheck_watch.start(user, row, {'refs': [{'key': 'pre', 'title': 'Clouds', 'arxiv': '2101.00001', 'doi': ''},
                                                  {'key': 'art', 'title': 'Art', 'arxiv': '', 'doi': '10.1000/art'}]})
        now = timezone.now()
        crossref = {'10.1000/art': {}, '10.1093/mnras/x1': {'container-title': ['MNRAS']}}
        with mock.patch('summarizer.refcheck.arxiv_records', return_value={'2101.00001': dict(RECORD, doi='')}), \
             mock.patch('summarizer.publication_status.crossref', side_effect=lambda doi: crossref[doi]), \
             mock.patch('summarizer.refcheck_watch.time.sleep'):
            self.assertEqual(refcheck_watch.check(now), 0)  # the first check only records
            self.assertEqual(refcheck_watch.check(now), 0)  # nothing is due again the same day
            crossref['10.1000/art'] = {'updated-by': [{'type': 'retraction', 'DOI': '10.1000/notice'}]}
        with mock.patch('summarizer.refcheck.arxiv_records', return_value={'2101.00001': RECORD}), \
             mock.patch('summarizer.publication_status.crossref', side_effect=lambda doi: crossref[doi]), \
             mock.patch('summarizer.refcheck_watch.time.sleep'):
            self.assertEqual(refcheck_watch.check(now + datetime.timedelta(days=1, minutes=1)), 2)
        self.assertEqual(sorted(PublicationAlert.objects.values_list('kind', 'arxiv_id')),
                         [('published', '2101.00001'), ('retraction', 'doi_10.1000--art')])
        self.assertEqual(publication_alerts.send_alerts(now, 10, base_url='https://x'), 1)
        self.assertIn('https://x/en/paper/doi_10.1000--art/', mail.outbox[0].body)
        ReferenceWatch.objects.update(until=now - datetime.timedelta(days=1))
        refcheck_watch.check(now)
        self.assertFalse(ReferenceWatch.objects.exists())
        self.assertTrue(User.objects.filter(pk=user.pk).exists())
