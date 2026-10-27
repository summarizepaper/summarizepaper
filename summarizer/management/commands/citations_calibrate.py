"""How well the citation alerts read real citations, without writing anything.

python manage.py citations_calibrate                          # two sample papers (astrophysics, machine learning)
python manage.py citations_calibrate 2208.11692 --citing 20   # a paper of yours, its 20 newest citing arXiv papers
python manage.py citations_calibrate --feedback               # only the readers' corrections of the verdicts

For each paper, its newest citing arXiv papers are read as the alerts read them: the citation sentences of
Semantic Scholar, else the ones found in the citing paper's arXiv HTML; then the model judges them. It prints how
often a sentence is found, the verdicts by stance and confidence, how many would have been sent at once (a
confident "questions" that is not a self-citation), and each verdict with its sentence, to read them against it.
It costs a model call per paper and a few seconds per arXiv page, and stores nothing.
"""
import time

import requests
from django.core.management.base import BaseCommand
from django.db.models import Count

from summarizer import citations
from summarizer.models import CitationCorrection

SAMPLES = ['2208.11692', '2310.06825']


def patiently(fetch, arxiv_id):
    """Semantic Scholar's public rate is low without a key: wait and try again."""
    for wait in (10, 30, 60):
        try:
            return fetch(arxiv_id)
        except requests.HTTPError as error:
            if error.response is None or error.response.status_code != 429:
                raise
            time.sleep(wait)
    return fetch(arxiv_id)


class Command(BaseCommand):
    help = 'Read and judge the newest citations of sample papers, and show the readers\' corrections.'

    def add_arguments(self, parser):
        parser.add_argument('papers', nargs='*', help='arXiv ids of the cited papers')
        parser.add_argument('--citing', type=int, default=15, help='citing arXiv papers to read per paper')
        parser.add_argument('--feedback', action='store_true', help='only the readers\' corrections so far')

    def handle(self, *args, **options):
        self.feedback()
        if options['feedback']:
            return
        for arxiv_id in options['papers'] or SAMPLES:
            self.calibrate(citations.base_id(arxiv_id) or arxiv_id, options['citing'])

    def calibrate(self, arxiv_id, size):
        out = self.stdout.write
        title = citations.titles([arxiv_id]).get(arxiv_id, '')
        authors, abstract, doi = patiently(citations.fetch_meta, arxiv_id) or ([], '', '')
        citing = patiently(citations.fetch_s2, arxiv_id) or {}
        out('\n== %s %s\n%d citing papers, %d with sentences from Semantic Scholar' % (
            arxiv_id, title, len(citing), sum(1 for c in citing.values() if c['contexts'])))
        newest = sorted((pid for pid in citing if citing[pid]['arxiv_id']), key=lambda p: citing[p]['date'],
                        reverse=True)[:size]
        found = {'s2': 0, 'arxiv': 0, 'no_html': 0, 'not_found': 0}
        sample = {}
        for pid in newest:
            c = dict(citing[pid])
            if c['contexts'] and not citations.fragmentary(c['contexts']):
                found['s2'] += 1
            else:
                time.sleep(citations.ARXIV_SECONDS)
                html = citations.fetch_html(c['arxiv_id'])
                read = citations.find_contexts(html, arxiv_id, title, doi) if html else []
                found['arxiv' if read else 'no_html' if not html else 'not_found'] += 1
                c['contexts'] = read or c['contexts']  # Semantic Scholar's pieces, when there is nothing better
            c['self'] = citations.shares_author(citations.self_keys(authors), c['authors'])
            sample[pid] = c
        out('sentences: %(s2)d from Semantic Scholar, %(arxiv)d read on arXiv, %(not_found)d not found in the '
            'arXiv HTML, %(no_html)d without arXiv HTML' % found)
        verdicts = citations.judge(title, abstract, {p: c for p, c in sample.items() if c['contexts']}, 'en')
        counts = {}
        for v in verdicts.values():
            counts[(v['stance'], v['confidence'])] = counts.get((v['stance'], v['confidence']), 0) + 1
        out('verdicts (high / low confidence):')
        for stance in citations.STANCES:
            out('  %-10s %d / %d' % (stance, counts.get((stance, 'high'), 0), counts.get((stance, 'low'), 0)))
        urgent = [p for p, v in verdicts.items()
                  if v['stance'] == 'questions' and v['confidence'] == 'high' and not sample[p]['self']]
        out('sent at once: %d of %d judged' % (len(urgent), len(verdicts)))
        for pid, c in sample.items():
            v = verdicts.get(pid)
            out('\n  [%s%s%s] %s' % (v['stance'] if v else '-', ' ' + v['confidence'] if v else '',
                                     ' self' if c['self'] else '', c['title'][:100]))
            for sentence in c['contexts'][:2]:
                out('    > %s' % sentence[:300])
            if v and v['note']:
                out('    = %s' % v['note'])

    def feedback(self):
        """The readers' corrections: what the model said, and what they say instead."""
        out = self.stdout.write
        rows = (CitationCorrection.objects.values('model_stance', 'stance').annotate(n=Count('pk'))
                .order_by('-n'))
        out('Verdicts corrected by their readers: %d' % sum(r['n'] for r in rows))
        for r in rows:
            out('  %-10s -> %-10s %d' % (r['model_stance'] or '(author)', r['stance'], r['n']))
