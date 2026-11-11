"""Refuse HTTP requests at once when too many are already being served, instead of letting them sink the process.

Each request served by Django holds a thread and a Postgres connection, and Postgres (essential-1) allows 20.
Scrapers sometimes send over a thousand page requests in a minute (24 October, 21:29 and 23:50): past about
20 at once, Postgres refused connections ("too many connections"), the process ran out of threads ("can't start
new thread"), and every visitor got an error. Past LIMIT requests in flight, a request now gets a short 503 with
Retry-After, without a thread or a connection, so the requests already running end and the site stays up.

Static files hold no database connection and pages need them: they are not counted. Websockets are not either:
they go through channels, not through here.

On 26 October the scrapers came from hundreds of addresses at once (529 for 546 searches in 19 seconds), so no
limit per address could stop them; but they followed the result pages' links (a `window` token) without ever
keeping a cookie. Every page now sets a small cookie, and a result page with a window token asked without it
gets a short page that sets it and links to the results: a person clicks once, a scraper loops for nothing.

On 1-3 November they came back for paper and topic map pages of old arXiv ids (4 a second from hundreds of
addresses), and the web process grew past its 512 MB quota. Visitors without the cookie (scrapers, crawlers,
a person's very first page) now share ANONYMOUS_LIMIT of the LIMIT places, so people who already have it still
get in; and a paper not in the database, which costs an arXiv call, asks for the cookie first (first_visit).

The homepage, cheap and where new visitors arrive, is not held to ANONYMOUS_LIMIT (only to LIMIT); and the
busy page sets the cookie and reloads itself, so a person turned away gets in on the second try, in the
places of known visitors, while a scraper that drops cookies is turned away again.

Signed-in accounts go first: AccountCookieMiddleware gives them a second cookie (ACCOUNT_COOKIE), and requests
without it share GUEST_LIMIT of the LIMIT places, so the last ones are kept for accounts. The busy page says so
and links to the sign-up page. The cookie only says who to let in first; it opens nothing else.
"""
import html
import logging
import time
from urllib.parse import parse_qs

logger = logging.getLogger(__name__)

LIMIT = 14  # of the 20 connections: the rest for background results, websockets, the hourly job and the admin
ANONYMOUS_LIMIT = 8  # of LIMIT, for requests without the visitor cookie
GUEST_LIMIT = 11  # of LIMIT, for requests without the account cookie: the other places are kept for accounts
RETRY_AFTER = 5  # seconds
ALERT_EVERY = 600  # seconds between two Sentry alerts while requests are refused
EXEMPT = ('/static/', '/favicon.ico', '/robots.txt')
HOMEPAGES = ('/', '/en/', '/fr/')
# answers of a few milliseconds asked from other sites, which never carry our cookies: the card of arXiv's
# abstract pages (arxiv_labs.py) would otherwise be the first turned away whenever scrapers fill the anonymous places
CHEAP = ('/api/arxiv-labs/',)
COOKIE = b'sp'
SET_COOKIE = b'sp=1; Max-Age=31536000; Path=/; SameSite=Lax; Secure'
ACCOUNT_COOKIE = b'spa'
CONTINUE = ('<!doctype html><meta charset="utf-8"><meta name="robots" content="noindex"><title>SummarizePaper</title>'
            '<p><a href="%s">See the results</a> · <a href="%s">Voir les résultats</a></p>')
CONTINUE_PAGE = ('<!doctype html><meta charset="utf-8"><meta name="robots" content="noindex"><title>SummarizePaper</title>'
                 '<p><a href="%s">Open the paper</a> · <a href="%s">Ouvrir l\'article</a></p>')
BODY = (b'<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="%d"><title>Busy</title>' % RETRY_AFTER +
        b'<p>SummarizePaper is very busy right now. Please try again in a few seconds.<br>'
        b'Signed-in users go first: <a href="/en/register/">create a free account</a> to see this page less often.</p>'
        b'<p>Le site est tr\xc3\xa8s sollicit\xc3\xa9. R\xc3\xa9essayez dans quelques secondes.<br>'
        b'Les utilisateurs connect\xc3\xa9s passent en priorit\xc3\xa9\xc2\xa0: <a href="/fr/register/">cr\xc3\xa9ez un compte gratuit</a> '
        b'pour voir cette page moins souvent.</p>')


class LoadShedder:
    def __init__(self, app, limit=LIMIT, anonymous_limit=ANONYMOUS_LIMIT, guest_limit=GUEST_LIMIT):
        self.app, self.limit, self.anonymous_limit, self.guest_limit = app, limit, anonymous_limit, guest_limit
        self.in_flight = 0
        self.anonymous = 0
        self.guests = 0
        self.refused = 0
        self.alerted = 0.0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path', '').startswith(EXEMPT):
            return await self.app(scope, receive, send)
        known = has_cookie(scope)
        if not known and scrolled_search(scope):
            return await self.ask_for_cookie(scope, send)
        homepage = scope.get('path', '') in HOMEPAGES or scope.get('path', '').startswith(CHEAP)
        counted = not known and not homepage
        guest = not homepage and not has_cookie(scope, ACCOUNT_COOKIE)
        if (self.in_flight >= self.limit or (counted and self.anonymous >= self.anonymous_limit)
                or (guest and self.guests >= self.guest_limit)):
            return await self.refuse(scope, send)
        self.in_flight += 1  # one event loop: no lock needed
        self.anonymous += counted
        self.guests += guest
        try:
            return await self.app(scope, receive, send if known else with_cookie(send))
        finally:
            self.in_flight -= 1
            self.anonymous -= counted
            self.guests -= guest

    async def ask_for_cookie(self, scope, send):
        url = html.escape(scope.get('path', '') + '?' + scope.get('query_string', b'').decode('latin1'))
        await send({'type': 'http.response.start', 'status': 200, 'headers': [
            (b'content-type', b'text/html; charset=utf-8'), (b'set-cookie', SET_COOKIE),
            (b'cache-control', b'no-store'), (b'x-robots-tag', b'noindex')]})
        await send({'type': 'http.response.body', 'body': (CONTINUE % (url, url)).encode()})

    async def refuse(self, scope, send):
        self.refused += 1
        now = time.monotonic()
        if now - self.alerted >= ALERT_EVERY:
            self.alerted = now
            logger.error('loadshed: %d requests in flight, refusing new ones with 503 (%d refused so far), e.g. %s',
                         self.in_flight, self.refused, scope.get('path', ''))  # Sentry
        await send({'type': 'http.response.start', 'status': 503, 'headers': [
            (b'content-type', b'text/html; charset=utf-8'), (b'retry-after', str(RETRY_AFTER).encode()),
            (b'cache-control', b'no-store'), (b'set-cookie', SET_COOKIE)]})
        await send({'type': 'http.response.body', 'body': BODY})


def labs_token(path, day=None):
    """The pass of a link given by the arXivLabs card (arxiv_labs.py) to one page, valid the day it is given and
    the next: a reader coming from arXiv, who never has our cookie, opens that paper at once. A scraper could
    only reuse it for that paper, and only after asking our card for it."""
    import datetime
    import hashlib
    import hmac
    from django.conf import settings
    day = day or datetime.datetime.now(datetime.timezone.utc).date()
    message = ('%s|%s' % (path, day.isoformat())).encode()
    return hmac.new(settings.SECRET_KEY.encode(), message, hashlib.sha256).hexdigest()[:20]


def labs_pass(request):
    import datetime
    import hmac
    token = request.GET.get('labs', '')
    if not token:
        return False
    today = datetime.datetime.now(datetime.timezone.utc).date()
    return any(hmac.compare_digest(token, labs_token(request.path, today - datetime.timedelta(days=days)))
               for days in (0, 1))


def first_visit(request):
    """For a page that is costly to make, a short page that sets the visitor cookie and links to it, when the
    request does not carry the cookie yet and comes from no account (nor from a link of the arXivLabs card);
    None otherwise."""
    if request.method != 'GET' or COOKIE.decode() in request.COOKIES or request.user.is_authenticated:
        return None
    if labs_pass(request):
        return None
    from django.http import HttpResponse
    url = html.escape(request.get_full_path())
    response = HttpResponse(CONTINUE_PAGE % (url, url))
    response['Cache-Control'] = 'no-store'
    response['X-Robots-Tag'] = 'noindex'
    return response


def has_cookie(scope, cookie=COOKIE):
    for name, value in scope.get('headers', []):
        if name == b'cookie' and any(part.strip().split(b'=')[0] == cookie for part in value.split(b';')):
            return True
    return False


class AccountCookieMiddleware:
    """Gives signed-in accounts the cookie that lets them in first when the site is busy, and takes it back at
    sign-out."""
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        response = self.get_response(request)
        name = ACCOUNT_COOKIE.decode()
        user = getattr(request, 'user', None)
        if user is not None and user.is_authenticated:
            if name not in request.COOKIES:
                response.set_cookie(name, '1', max_age=31536000, samesite='Lax', secure=True)
        elif name in request.COOKIES:
            response.delete_cookie(name, samesite='Lax')
        return response


def scrolled_search(scope):
    """A result page reached through the links of another one (its window token), not a new search."""
    return '/search-results/' in scope.get('path', '') and 'window' in parse_qs(scope.get('query_string', b'').decode('latin1'))


def with_cookie(send):
    """send, adding the visitor cookie to HTML pages."""
    async def wrapped(message):
        if message['type'] == 'http.response.start':
            headers = list(message.get('headers', []))
            if any(n.lower() == b'content-type' and v.startswith(b'text/html') for n, v in headers):
                message = dict(message, headers=headers + [(b'set-cookie', SET_COOKIE)])
        await send(message)
    return wrapped
