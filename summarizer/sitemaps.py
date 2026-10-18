from django.contrib.sitemaps import Sitemap
from django.db.models import Q
from django.shortcuts import reverse

from .models import SummaryPaper


class StaticViewSitemap(Sitemap):
    """Reverse 'static' views for XML sitemap."""
    changefreq = "weekly"
    priority = 0.6
    protocol = 'https'

    def items(self):
        return ['summarize','about','faq','contact','privacy','legal']

    def location(self, item):
        return reverse(item)


class SummarizedPaperSitemap(Sitemap):
    """One URL per paper and language that actually has a summary.

    Papers that were only looked up show a bare "Run" page, which search
    engines treat as thin content, so they stay out of the sitemap.
    """
    priority = 0.9
    protocol = 'https'
    limit = 10000

    def items(self):
        return (SummaryPaper.objects
                .exclude(Q(summary__isnull=True) | Q(summary=''))
                .select_related('paper')
                .only('lang', 'updated', 'paper__arxiv_id')
                .order_by('id'))

    def location(self, item):
        return '/%s/arxiv-id/%s/' % (item.lang, item.paper.arxiv_id)

    def lastmod(self, item):
        return item.updated
