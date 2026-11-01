"""How well the bibliography check judges citations, on sentences annotated by hand. Writes nothing.

python manage.py refcheck_benchmark            # every case of fixtures/refcheck_benchmark.json
python manage.py refcheck_benchmark --only T   # the cases whose id starts with T

Each case is a sentence citing well-known arXiv papers, true or false, annotated "supported" or "flagged", and
by kind: said in the abstract, a wrong value, a wrong claim, true but only said in the full text, a background
citation, a statement cited to two papers of which one says it. The sentences are read as a manuscript is
(refcheck_claims.run, each case its own file so no sentence gives context to another), with the abstracts arXiv
gives. It prints the flagged sentences that are true (false alarms) and the false sentences let through, by
kind, and the full texts read. It costs a few model calls and reads a few arXiv full texts.
"""
import json
from pathlib import Path

from django.core.management.base import BaseCommand

from summarizer import refcheck, refcheck_claims

CASES = Path(__file__).resolve().parents[2] / 'fixtures' / 'refcheck_benchmark.json'


class Command(BaseCommand):
    help = 'Judge hand-annotated citations of well-known papers and report false alarms and misses.'

    def add_arguments(self, parser):
        parser.add_argument('--only', default='', help='the cases whose id starts with this')
        parser.add_argument('--lang', default='en')

    def handle(self, *args, **options):
        out = self.stdout.write
        data = json.loads(CASES.read_text())
        cases = [c for c in data['cases'] if c[0].startswith(options['only'])]
        records = refcheck.arxiv_records(list(data['papers'].values()))
        papers = {}
        for key, arxiv_id in data['papers'].items():
            record = records.get(arxiv_id)
            if record is None:
                out('arXiv did not give %s (%s): its cases cannot be judged' % (key, arxiv_id))
                continue
            papers[key] = {'title': record['title'], 'abstract': record['abstract'], 'arxiv': arxiv_id, 'doi': ''}
        texts = [(case_id, '\\begin{document}\n%s\n\\end{document}' % sentence) for case_id, _e, _k, sentence in cases]
        result = refcheck_claims.run(texts, papers, options['lang'])
        flagged = {c['file']: c for c in result['claims']}
        confirmed = {c['file']: c for c in result['confirmed']}
        counts, wrong = {}, []
        for case_id, expected, kind, sentence in cases:
            got = 'flagged' if case_id in flagged else 'supported'
            row = counts.setdefault(kind, {'cases': 0, 'right': 0})
            row['cases'] += 1
            row['right'] += got == expected
            if got != expected:
                claim = flagged.get(case_id) or confirmed.get(case_id) or {}
                wrong.append((case_id, expected, kind, sentence, claim))
        true_ones = [c for c in cases if c[1] == 'supported']
        false_ones = [c for c in cases if c[1] == 'flagged']
        alarms = [w for w in wrong if w[1] == 'supported']
        misses = [w for w in wrong if w[1] == 'flagged']
        out('\n%d cases: %d sentences read, full texts read %d of %d' % (
            len(cases), result['checked'], result['full_texts_read'], result['full_texts_tried']))
        out('False alarms: %d of %d true sentences flagged' % (len(alarms), len(true_ones)))
        out('Misses: %d of %d false sentences let through' % (len(misses), len(false_ones)))
        for kind, row in sorted(counts.items()):
            out('  %-10s %d/%d right' % (kind, row['right'], row['cases']))
        for case_id, expected, kind, sentence, claim in wrong:
            out('\n%s expected %s (%s): %s' % (case_id, expected, kind, sentence))
            if claim:
                out('   verdict %s%s: %s' % (claim.get('verdict'), ' (full text read)' if claim.get('full_text') else '',
                                           claim.get('reason', '')))
                if claim.get('passage'):
                    out('   passage: %s' % claim['passage']['quote'][:300])
                if claim.get('numbers_absent'):
                    out('   values not found: %s' % ', '.join(claim['numbers_absent']))
