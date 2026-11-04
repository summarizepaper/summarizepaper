"""The pages of the site people open, for the staff activity page /admin/activite/ (activity.py): which pages and
tools, from which site, and the path of a visit through the day.

Only pages opened in a browser are kept: HTML answers to GET, not the fragments a page fetches (Sec-Fetch-Dest),
not the crawlers (access.crawler) nor the scrapers, which never keep the visitor cookie of loadshed.py and send no
Sec-Fetch headers. Some robots pass all that: they drive real browsers, keep the cookie and make up where they
come from (on 4 November, one browser opened sign-in and sign-up pages "from" Google, Bing, Yahoo, Yandex and
Baidu in turn, a few seconds apart). So a page only counts once a person has moved on it: visit.js tells seen()
at the first scroll, touch, key or mouse move, and the activity page only counts the visitors with such a page
or signed in (activity.py). A visitor is a hash of the IP address and the browser with the day, so the address is not kept
and a visit cannot be followed from one day to the next. The rows are deleted after VISIT_DAYS days (purge, from
the hourly send_digest)."""
import datetime
import hashlib
import logging
from urllib.parse import urlsplit

from django.conf import settings
from django.http import HttpResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import access
from .loadshed import COOKIE
from .models import PageView

logger = logging.getLogger(__name__)

VISIT_DAYS = 60
# Robots that fake mouse moves too came "from" Google to the sign-in page (12 of the 17 confirmed visitors of
# 4 November, 14:30): search engines never send anyone to the pages robots.txt keeps from them
ENGINES = ('google.com', 'bing.com', 'yahoo.com', 'yandex.com', 'yandex.ru', 'baidu.com', 'duckduckgo.com', 'ecosia.org', 'qwant.com')
UNLISTED = ('login', 'email_login', 'register', 'logout')  # /*/login/, /*/register/, /*/logout/ in robots.txt
SEEN_WITHIN = datetime.timedelta(minutes=30)  # how late visit.js may confirm a page
SKIPPED_PREFIXES = ('/admin/', '/rosetta/', '/static/')
SKIPPED_SUFFIXES = ('_fragment', '_state', '_status', '_feed', '_quota', '_cell', '_why', '_sw')
# pages whose address holds a sign-in or unsubscribe token: only their name is kept
PRIVATE = ('login_link', 'activate', 'password_reset_confirm', 'reactivation_delete', 'digest_link', 'digest_open')
OWN_HOSTS = ('summarizepaper.com', 'herokuapp.com', 'localhost', 'testserver', '127.0.0.1')

# url name prefixes -> the section of the site, the first that matches; the tools have the names of
# activity.TOOL_NAMES, so their page views and their results add up in the same row
SECTIONS = (
    (('collection_slides', 'slides'), 'Journal club'),
    (('presskit',), 'Kit presse'),
    (('literature',), 'Revue de littérature'),
    (('missed',), 'Qu’ai-je manqué ?'),
    (('refcheck',), 'Vérification de bibliographie'),
    (('student',), 'Stage ou thèse'),
    (('teams',), 'Qui travaille dessus ?'),
    (('conference',), 'Compagnon de conférence'),
    (('reading',), 'Niveau de lecture'),
    (('collection_compare',), 'Comparaison'),
    (('collection_questions',), 'Questions ouvertes'),
    (('changes',), 'Changements de version'),
    (('zotero', 'integrations'), 'Zotero'),
    (('chat',), 'Assistant'),
    (('collection',), 'Collections'),
    (('arxividpage', 'tree', 'bibtex', 'paper_figures', 'export_paper', 'neighbours'), 'Article'),
    (('search_results', 'search_journals'), 'Recherche'),
    (('saved_search', 'save_search'), 'Recherches enregistrées'),
    (('digest',), 'Digest'),
    (('letter',), 'Lettres'),
    (('citation',), 'Alertes de citation'),
    (('scoops',), 'Veille anti-scoop'),
    (('trends',), 'Tendances'),
    (('history', 'trash', 'delete'), 'Mon activité'),
    (('premium',), 'Premium'),
    (('login', 'email_login', 'google_login', 'register', 'password', 'activate', 'account', 'logout'), 'Compte'),
    (('summarize',), 'Accueil'),
    (('tools', 'getting_started', 'guide'), 'Outils et guides'),
)


def section(view):
    for prefixes, label in SECTIONS:
        if view.startswith(prefixes):
            return label
    return 'Autres pages'


def visitor(request, day=None):
    day = day or timezone.now().date()
    seed = '%s|%s|%s|%s' % (settings.SECRET_KEY, day.isoformat(), access.client_ip(request), request.META.get('HTTP_USER_AGENT', ''))
    return hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]


def _referrer(request):
    host = (urlsplit(request.META.get('HTTP_REFERER', '')).hostname or '').lower()
    if not host or any(host == own or host.endswith('.' + own) for own in OWN_HOSTS):
        return ''
    return host.removeprefix('www.')[:100]


def wanted(request, response):
    """Whether this answer is a page opened in a browser by a person."""
    match = getattr(request, 'resolver_match', None)
    if request.method != 'GET' or response.status_code != 200 or getattr(response, 'streaming', False) or not match:
        return False
    if not response.get('Content-Type', '').startswith('text/html') or request.path.startswith(SKIPPED_PREFIXES):
        return False
    name = match.url_name or ''
    if not name or name.endswith(SKIPPED_SUFFIXES):
        return False
    dest = request.META.get('HTTP_SEC_FETCH_DEST')
    if dest and dest != 'document':
        return False
    if access.crawler(request):
        return False
    signed_in = getattr(request, 'user', None) is not None and request.user.is_authenticated
    return signed_in or COOKIE.decode() in request.COOKIES or dest == 'document'


def record(request):
    name = request.resolver_match.url_name
    user = request.user if request.user.is_authenticated else None
    PageView.objects.create(
        user=user, visitor=visitor(request), view=name[:60],
        path='' if name in PRIVATE or 'unsubscribe' in name else request.path[:300],
        query=(request.GET.get('q') or '').strip()[:200] if name.startswith('search') else '',
        referrer=_referrer(request), source=(request.GET.get('utm_source') or '')[:40],
        lang=(getattr(request, 'LANGUAGE_CODE', '') or '')[:5], first=COOKIE.decode() not in request.COOKIES)


class PageViewMiddleware:
    """Keeps the pages people open (record); a failure here never touches the page."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        try:
            if wanted(request, response):
                record(request)
        except Exception:
            logger.exception('Page view not recorded')
        return response


def purge(now=None):
    now = now or timezone.now()
    return PageView.objects.filter(created__lt=now - datetime.timedelta(days=VISIT_DAYS)).delete()[0]


@csrf_exempt
@require_POST
def seen(request):
    """visit.js: a person moved on this page (its path), so the page is theirs, not a robot's."""
    path = (request.POST.get('path') or '')[:300]
    if path.startswith('/') and not access.crawler(request):
        PageView.objects.filter(visitor=visitor(request), path=path, human=False,
                                created__gte=timezone.now() - SEEN_WITHIN).update(human=True)
    return HttpResponse(status=204)
