from collections import defaultdict

import numpy as np
from django.core.management.base import BaseCommand
from django.db.models import Q

from summarizer import similar
from summarizer.models import ArxivPaper, PaperEmbedding, PickledData


class Command(BaseCommand):
    help = 'Embed summarized papers that have no embedding yet; with --rescore, recompute every closest-papers list.'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true')
        parser.add_argument('--rescore', action='store_true')
        parser.add_argument('--limit', type=int, default=0)

    def handle(self, dry_run=False, rescore=False, limit=0, **options):
        summarized = ArxivPaper.objects.filter(
            Q(summarypaper__isnull=False) | Q(arxiv_id__in=PickledData.objects.values('arxiv_id'))).distinct()
        done = set(PaperEmbedding.objects.filter(model=similar.MODEL).values_list('arxiv_id', flat=True))
        missing = [p for p in summarized.only('arxiv_id', 'title', 'abstract', 'category') if p.arxiv_id not in done]
        if limit:
            missing = missing[:limit]
        self.stdout.write('%d papers to embed' % len(missing))
        if not dry_run:
            for start in range(0, len(missing), 500):
                similar.store(missing[start:start + 500])
                self.stdout.write('  embedded %d' % min(start + 500, len(missing)))
        if not rescore:
            return

        by_category = defaultdict(list)
        for aid, category, vector in PaperEmbedding.objects.filter(model=similar.MODEL).values_list('arxiv_id', 'category', 'vector'):
            by_category[category].append((aid, np.frombuffer(vector, dtype=np.float32)))
        papers = list(ArxivPaper.objects.filter(summarypaper__isnull=False).distinct().only('arxiv_id', 'category'))
        self.stdout.write('%d closest-papers lists to recompute' % len(papers))
        if dry_run:
            return
        for n, paper in enumerate(papers, 1):
            candidates = by_category.get(paper.category or '', [])
            if any(aid == paper.arxiv_id for aid, _ in candidates):
                similar.save_closest(paper, similar.closest(paper, candidates=candidates))
            if n % 1000 == 0:
                self.stdout.write('  rescored %d' % n)
        self.stdout.write('done')
