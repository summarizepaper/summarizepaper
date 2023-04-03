"""arxivsummarizer URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import include, path, re_path
from django.conf import settings
from django.conf.urls.i18n import i18n_patterns
from django.contrib.sitemaps.views import sitemap
from django.contrib.sitemaps import GenericSitemap
from summarizer.sitemaps import StaticViewSitemap
from summarizer.models import ArxivPaper

app_name = 'summarizer'

sitemaps = {
    'static': StaticViewSitemap
}

def get_absolute_url2(obj):
    return f"/test/{obj.slug}"


sitemaps = {
    'arxividpage': GenericSitemap({
        'queryset': ArxivPaper.objects.filter(arxiv_id__isnull=False),
        'date_field': 'updated',
    }, priority=0.9,protocol = 'https'),
    #'tree': GenericSitemap({
    #    'queryset': ArxivPaper.objects.filter(arxiv_id__isnull=False),
    #    'date_field': 'updated',
        #'location': lambda obj: obj.get_absolute_url2(),
    #    'location': get_absolute_url2,
    #}, priority=0.8,protocol = 'https'),
    # START
    'static': StaticViewSitemap,
    # END
}

urlpatterns = [
    path('admin/', admin.site.urls),
    path('sitemap.xml', sitemap, {'sitemaps': sitemaps}),
    path('', include('summarizer.urls')),
]

if 'rosetta' in settings.INSTALLED_APPS:
    urlpatterns += [
        re_path(r'^rosetta/', include('rosetta.urls')),
        #path('en/', include('arxivsummarizer.urls')),
        #path('fr/', include('arxivsummarizer.urls')),
    ]


urlpatterns += i18n_patterns (
    path('',include('summarizer.urls')),
    #path('',include('blog.urls', namespace="blog2")),
)
