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
from summarizer.admin import activity_view
from django.urls import include, path, re_path
from django.conf import settings
from django.conf.urls.i18n import i18n_patterns
from django.contrib.sitemaps import views as sitemap_views
from summarizer.sitemaps import GuideSitemap, LetterSitemap, StaticViewSitemap, SummarizedPaperSitemap
from summarizer import accounts, billing, mail_events, social, visits
from summarizer.views import llms_txt, robots_txt

app_name = 'summarizer'

sitemaps = {
    'papers': SummarizedPaperSitemap,
    'guides': GuideSitemap,
    'letters': LetterSitemap,
    'static': StaticViewSitemap,
}

urlpatterns = [
    path('admin/activite/', admin.site.admin_view(activity_view), name='admin_activity'),
    path('admin/', admin.site.urls),
    path('robots.txt', robots_txt),
    path('llms.txt', llms_txt),
    path('stripe/webhook/', billing.webhook, name='stripe_webhook'),
    path('cloudmailin/events/<str:secret>/', mail_events.webhook, name='cloudmailin_events'),
    path('google/callback/', accounts.google_callback, name='google_callback'),
    path('visit/', visits.seen, name='visit_seen'),
    # the pictures shown when a page is shared (og:image), see social.py
    path('og/<str:lang>/p/<str:arxiv_id>.png', social.paper_image, name='og_paper'),
    path('og/<str:lang>/trends/<str:server>/<str:day>.png', social.trends_image, name='og_trends'),
    path('og/<str:lang>/letter/<slug:topic>/<str:day>.png', social.letter_image, name='og_letter'),
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
