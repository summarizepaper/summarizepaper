from django.contrib.sitemaps import Sitemap
from django.shortcuts import reverse

class StaticViewSitemap(Sitemap):
    """Reverse 'static' views for XML sitemap."""
    changefreq = "weekly"
    priority = 0.6
    protocol = 'https'
    
    def items(self):
        return ['summarize','about','faq','contact','history','privacy','legal','register']#'arxividpage'
    def location(self, item):
        return reverse(item)
