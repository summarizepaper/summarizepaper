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
from django.contrib.sitemaps import views as sitemap_views
from summarizer.sitemaps import GuideSitemap, LetterSitemap, StaticViewSitemap, SummarizedPaperSitemap
from summarizer import accounts, billing
from summarizer.views import llms_txt, robots_txt

app_name = 'summarizer'

sitemaps = {
    'papers': SummarizedPaperSitemap,
    'guides': GuideSitemap,
    'letters': LetterSitemap,
    'static': StaticViewSitemap,
}

urlpatterns = [
    path('admin/', admin.site.urls),
    path('robots.txt', robots_txt),
    path('llms.txt', llms_txt),
    path('stripe/webhook/', billing.webhook, name='stripe_webhook'),
    path('google/callback/', accounts.google_callback, name='google_callback'),
    path('sitemap.xml', sitemap_views.index, {'sitemaps': sitemaps, 'sitemap_url_name': 'sitemap-section'}),
    path('sitemap-<section>.xml', sitemap_views.sitemap, {'sitemaps': sitemaps}, name='sitemap-section'),
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
