"""One host for the whole site: www.summarizepaper.com.

Until 26 October summarizepaper.com (and the herokuapp address) served the same pages as www. Our canonical
tags point to www, but Google chose summarizepaper.com for the home page and indexed some guides there, while
their www copies stayed unknown to it: the signals were split between two sites. Every request to another of
our hosts now gets a permanent redirect to the same path on www, before Django (no thread, no connection).
"""
CANONICAL = 'www.summarizepaper.com'
OTHERS = (b'summarizepaper.com', b'summarizepaper.herokuapp.com')


class CanonicalHost:
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope['type'] == 'http' and host(scope) in OTHERS:
            return await redirect(scope, send)
        return await self.app(scope, receive, send)


def host(scope):
    for name, value in scope.get('headers', []):
        if name == b'host':
            return value.split(b':')[0].lower()
    return b''


async def redirect(scope, send):
    path = scope['raw_path'].decode('latin1') if scope.get('raw_path') else scope.get('path', '/')
    url = 'https://' + CANONICAL + path
    if scope.get('query_string'):
        url += '?' + scope['query_string'].decode('latin1')
    # 308 keeps the method and body of a form sent to the old host; a page is moved with 301
    status = 301 if scope.get('method') in ('GET', 'HEAD') else 308
    await send({'type': 'http.response.start', 'status': status, 'headers': [
        (b'location', url.encode('latin1')), (b'cache-control', b'max-age=86400')]})
    await send({'type': 'http.response.body', 'body': b''})
