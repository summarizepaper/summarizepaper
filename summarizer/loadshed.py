"""Refuse HTTP requests at once when too many are already being served, instead of letting them sink the process.

Each request served by Django holds a thread and a Postgres connection, and Postgres (essential-1) allows 20.
Scrapers sometimes send over a thousand page requests in a minute (24 October, 21:29 and 23:50): past about
20 at once, Postgres refused connections ("too many connections"), the process ran out of threads ("can't start
new thread"), and every visitor got an error. Past LIMIT requests in flight, a request now gets a short 503 with
Retry-After, without a thread or a connection, so the requests already running end and the site stays up.

Static files hold no database connection and pages need them: they are not counted. Websockets are not either:
they go through channels, not through here.
"""
import logging
import time

logger = logging.getLogger(__name__)

LIMIT = 14  # of the 20 connections: the rest for background results, websockets, the hourly job and the admin
RETRY_AFTER = 5  # seconds
ALERT_EVERY = 600  # seconds between two Sentry alerts while requests are refused
EXEMPT = ('/static/', '/favicon.ico', '/robots.txt')
BODY = (b'<!doctype html><meta charset="utf-8"><title>Busy</title>'
        b'<p>SummarizePaper is very busy right now. Please try again in a few seconds.</p>'
        b'<p>Le site est tr\xc3\xa8s sollicit\xc3\xa9. R\xc3\xa9essayez dans quelques secondes.</p>')


class LoadShedder:
    def __init__(self, app, limit=LIMIT):
        self.app, self.limit = app, limit
        self.in_flight = 0
        self.refused = 0
        self.alerted = 0.0

    async def __call__(self, scope, receive, send):
        if scope['type'] != 'http' or scope.get('path', '').startswith(EXEMPT):
            return await self.app(scope, receive, send)
        if self.in_flight >= self.limit:
            return await self.refuse(scope, send)
        self.in_flight += 1  # one event loop: no lock needed
        try:
            return await self.app(scope, receive, send)
        finally:
            self.in_flight -= 1

    async def refuse(self, scope, send):
        self.refused += 1
        now = time.monotonic()
        if now - self.alerted >= ALERT_EVERY:
            self.alerted = now
            logger.error('loadshed: %d requests in flight, refusing new ones with 503 (%d refused so far), e.g. %s',
                         self.in_flight, self.refused, scope.get('path', ''))  # Sentry
        await send({'type': 'http.response.start', 'status': 503, 'headers': [
            (b'content-type', b'text/html; charset=utf-8'), (b'retry-after', str(RETRY_AFTER).encode()),
            (b'cache-control', b'no-store')]})
        await send({'type': 'http.response.body', 'body': BODY})
