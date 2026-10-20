from django.contrib.sitemaps import Sitemap
from django.db.models import Q
from django.shortcuts import reverse

from . import guides
from .models import Letter, SummaryPaper


class StaticViewSitemap(Sitemap):
    """Reverse 'static' views for XML sitemap."""
    changefreq = "weekly"
    priority = 0.6
    protocol = 'https'

    def items(self):
        items = ['summarize','about','author','faq','contact','privacy','legal','digest','premium','letters','trends','collections','missed','citations','scoops','integrations']
        if guides.published('en'):
            items.append('guides')
        return items

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


class GuideSitemap(Sitemap):
    """Published Markdown guides, every language."""
    priority = 0.8
    protocol = 'https'

    def items(self):
        return guides.published()

    def location(self, item):
        return '/%s/guides/%s/' % (item['lang'], item['slug'])

    def lastmod(self, item):
        return item.get('updated')


class LetterSitemap(Sitemap):
    """The weekly letters, every language."""
    priority = 0.7
    protocol = 'https'

    def items(self):
        return Letter.objects.only('topic', 'lang', 'day', 'created').order_by('-day')

    def location(self, item):
        return '/%s/letters/%s/%s/' % (item.lang, item.topic, item.day.isoformat())

    def lastmod(self, item):
        return item.created
