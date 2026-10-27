"""How well the scoop watch thresholds fit real papers, without writing anything.

python manage.py scoop_calibrate                       # two sample projects (astrophysics, biology)
python manage.py scoop_calibrate --text project.txt    # a project of yours: its title on the first line
python manage.py scoop_calibrate --feedback            # only the readers' verdicts on the papers already found

For each project, the papers of the last stored announcements are scored as a check would score them; the model
reads a sample of each closeness band, so that the share of overlaps it finds below FLOOR says whether FLOOR is
too high (overlaps missed) and the share above it whether it is too low (papers read for nothing). It costs a few
model calls, and stores nothing.
"""
import random

import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Count

from summarizer import digest, scoops, similar
from summarizer.models import ArxivBatch, ScoopMatch, WatchedProject

SAMPLES = [
    ('Clouds of hot Jupiters with JWST',
     'We measure the cloud properties of a sample of twelve hot Jupiters observed with JWST NIRSpec PRISM transmission '
     'spectroscopy. We develop a new atmospheric retrieval code that couples cloud microphysics with chemistry, and use '
     'it to infer cloud-top pressure, particle sizes and condensate composition (silicates, iron). We expect a trend of '
     'cloudiness with equilibrium temperature and compare with previous HST results and with 3D general circulation '
     'models. Targets include WASP-107b, WASP-39b and HD 209458b.'),
    ('Single-cell atlas of zebrafish fin regeneration',
     'We build a single-cell RNA-seq and ATAC-seq atlas of the regenerating zebrafish caudal fin across seven time '
     'points after amputation, to identify the transcription factors that reprogram osteoblasts into progenitors. We '
     'use lineage tracing with Cre-lox reporters and CRISPR knockouts of candidate factors (runx2, sp7, msx). We expect '
     'that a small set of factors, conserved with axolotl limb regeneration, drives dedifferentiation.'),
]
BANDS = [(0.30, 0.35), (0.35, 0.40), (0.40, 0.45), (0.45, 0.50), (0.50, 0.55), (0.55, 1.01)]
PER_BAND = 8


class Command(BaseCommand):
    help = 'Score sample projects against the last announcements and have the model read each closeness band.'

    def add_arguments(self, parser):
        parser.add_argument('--text', help='a file: the title on the first line, the description after it')
        parser.add_argument('--days', type=int, default=3, help='announcements to read')
        parser.add_argument('--feedback', action='store_true', help='only the readers\' verdicts so far')

    def handle(self, *args, **options):
        self.feedback()
        if options['feedback']:
            return
        if options['text']:
            with open(options['text']) as f:
                title, _, description = f.read().strip().partition('\n')
            projects = [(title.strip(), description.strip())]
        else:
            projects = SAMPLES
        days = sorted(ArxivBatch.objects.order_by('-day').values_list('day', flat=True)[:options['days']])
        papers = digest.load_papers(days)
        vectors = digest.stored_vectors(list(papers))
        self.stdout.write('Announcements %s: %d papers, %d embedded' % (', '.join(map(str, days)), len(papers), len(vectors)))
        for title, description in projects:
            self.calibrate(WatchedProject(title=title, description=description, lang='en'), papers, vectors, len(days))

    def calibrate(self, project, papers, vectors, days):
        out = self.stdout.write
        facets = scoops.describe(project)
        texts = ['%s\n\n%s' % (project.title, facets['abstract'])] + [facets[a] for a in scoops.ASPECTS if facets[a]]
        rows = similar.embed([t[:similar.MAX_INPUT_CHARS] for t in texts])
        out('\n== %s\nentities: %s\nqueries: %s' % (project.title, ', '.join(facets['entities']), ' | '.join(facets['queries'])))
        scored = [(float(np.max(rows @ v)), float(rows[0] @ v), aid) for aid, v in vectors.items()]
        out('papers per day above each threshold (facets / whole project only):')
        for floor in (0.30, 0.35, 0.40, 0.45, 0.50, 0.55):
            out('  >= %.2f: %6.1f / %6.1f' % (floor, sum(s[0] >= floor for s in scored) / days, sum(s[1] >= floor for s in scored) / days))
        named = [s for s in scored if scoops.names_entity(facets['entities'], papers[s[2]])]
        out('naming an entity: %.1f per day, %.1f above the loose floor' % (
            len(named) / days, sum(s[0] >= scoops.LOOSE_FLOOR for s in named) / days))
        random.seed(0)
        sample = []
        for low, high in BANDS:
            band = [s for s in scored if low <= s[0] < high]
            sample += [(low, high, s) for s in random.sample(band, min(PER_BAND, len(band)))]
        verdicts = scoops.judge(project, [(papers[s[2]], s[0]) for _, _, s in sample], 'en')
        out('model verdicts by closeness band (strong / partial / read):')
        for low, high in BANDS:
            ids = [s[2] for lo, _, s in sample if lo == low]
            found = [verdicts[i]['overlap'] for i in ids if i in verdicts]
            out('  %.2f-%.2f: %d / %d / %d' % (low, min(high, 1), found.count('high'), found.count('partial'), len(ids)))
        for low, high, s in sample:
            verdict = verdicts.get(s[2], {}).get('overlap', '-')
            if verdict != '-' or s[0] >= scoops.FLOOR:
                out('  %.2f %-7s %s' % (s[0], verdict, papers[s[2]]['title'][:100]))
        out('FLOOR is %.2f: overlaps found below it are missed by the closeness alone; above it, "-" are read for '
            'nothing.' % scoops.FLOOR)

    def feedback(self):
        """The readers' verdicts on the papers found, by closeness band and by why they were read."""
        out = self.stdout.write
        rated = ScoopMatch.objects.exclude(status='')
        out('Papers found: %d, rated by their readers: %d' % (ScoopMatch.objects.count(), rated.count()))
        for low, high in BANDS:
            counts = dict(rated.filter(similarity__gte=low, similarity__lt=high).values('status')
                          .annotate(n=Count('pk')).values_list('status', 'n'))
            if counts:
                out('  %.2f-%.2f: relevant %d, not relevant %d' % (low, min(high, 1), counts.get('relevant', 0),
                                                                     counts.get('irrelevant', 0)))
        for signal in scoops.SIGNALS:
            counts = dict(rated.filter(signals__contains=signal).values('status').annotate(n=Count('pk'))
                          .values_list('status', 'n'))
            if counts:
                out('  %s: relevant %d, not relevant %d' % (signal, counts.get('relevant', 0), counts.get('irrelevant', 0)))
