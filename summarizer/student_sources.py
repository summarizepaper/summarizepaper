"""Bounded public institutional evidence, never generated employment or vacancies."""
import hashlib
import http.client
import ipaddress
import json
import re
import socket
import ssl
from html.parser import HTMLParser
from urllib.parse import urlsplit, urljoin

from django.core.cache import cache
from django.utils import timezone
from django.utils.dateparse import parse_date


class Unavailable(Exception):
    pass


def normalize_url(value):
    """Accept pasted institutional addresses while always requesting HTTPS."""
    value = value.strip()
    if value.startswith('//'):
        value = 'https:' + value
    elif not re.match(r'^[a-z][a-z0-9+.-]*:', value, re.I):
        value = 'https://' + value
    elif value.lower().startswith('http://'):
        value = 'https://' + value[7:]
    return value


def host(url, allow_http=False):
    try:
        p = urlsplit(url)
        allowed = ('https', 'http') if allow_http else ('https',)
        if p.scheme not in allowed or p.username or p.password or p.port not in (None, 80 if p.scheme == 'http' else 443):
            return ''
        if any(c.isspace() or ord(c) < 32 for c in url) or '\\' in url:
            return ''
        return (p.hostname or '').encode('idna').decode().lower().removeprefix('www.')
    except (ValueError, UnicodeError):
        return ''


def official(url, website):
    root, candidate = host(website, allow_http=True), host(url)
    return bool(root and candidate and (candidate == root or candidate.endswith('.' + root)))


class PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, hostname, address):
        super().__init__(hostname, timeout=5, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        # Connect to the checked IP, preserving TLS hostname verification; no DNS rebinding.
        sock = socket.create_connection((self.address, 443), timeout=self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class Page(HTMLParser):
    def __init__(self, html):
        super().__init__(convert_charrefs=True)
        self.links, self.nodes, self.text = [], [], []
        self.script = None
        self.hidden = 0
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'a' and attrs.get('href'):
            self.links.append(attrs['href'])
        if tag == 'script':
            self.script = [] if attrs.get('type', '').lower() == 'application/ld+json' else None
        if tag in ('script', 'style'):
            self.hidden += 1

    def handle_data(self, data):
        if self.script is not None:
            self.script.append(data)
        if not self.hidden:
            self.text.append(data)

    def handle_endtag(self, tag):
        if tag == 'script' and self.script is not None:
            try:
                self.walk(json.loads(''.join(self.script)))
            except (ValueError, RecursionError):
                pass
            self.script = None
        if tag in ('script', 'style'):
            self.hidden = max(0, self.hidden - 1)

    def walk(self, obj, depth=0):
        if depth > 12:
            return
        if isinstance(obj, dict):
            self.nodes.append(obj)
            for value in obj.values():
                self.walk(value, depth+1)
        elif isinstance(obj, list):
            for value in obj[:100]:
                self.walk(value, depth+1)


def fetch(url, website):
    """No credentials, proxies, private IPs, external redirects or unlimited responses."""
    key = 'student-public-page-v2:' + hashlib.sha256(url.encode()).hexdigest()
    if not official(url, website):
        raise Unavailable()
    cached = cache.get(key)
    if cached is not None:
        return Page(cached[0]), cached[1]
    for _ in range(3):
        if not official(url, website):
            raise Unavailable()
        parsed = urlsplit(url)
        try:
            addresses = {r[4][0] for r in socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)}
            if not addresses or any(not ipaddress.ip_address(a).is_global for a in addresses):
                raise Unavailable()
            connection = PinnedHTTPS(parsed.hostname, sorted(addresses)[0])
            try:
                path = (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
                connection.request('GET', path,
                                   headers={'User-Agent': 'SummarizePaper/1.0 (public research placement sources)', 'Accept': 'text/html'})
                response = connection.getresponse()
                if response.status in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.getheader('Location', ''))
                    continue
                if response.status != 200 or 'text/html' not in response.getheader('Content-Type', ''):
                    raise Unavailable()
                raw = response.read(1_000_001)
                if len(raw) > 1_000_000:
                    raise Unavailable()
                html = raw.decode('utf-8', errors='replace')
                if len(raw) < 256 * 1024:  # large pages would crowd the shared Redis
                    cache.set(key, (html, url), 6*3600)
                return Page(html), url
            finally:
                connection.close()
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise Unavailable() from exc
    raise Unavailable()


def text(value, limit=1200):
    if not isinstance(value, str):
        return ''
    parser = Page(value)
    return ' '.join(' '.join(parser.text).split())[:limit]


def records(page, url, person):
    from .students import normalized
    today = timezone.now().date()
    profiles, offers = [], []
    for node in page.nodes:
        types = node.get('@type', [])
        types = types if isinstance(types, list) else [types]
        if 'Person' in types and normalized(text(node.get('name'))) == normalized(person['name']):
            employer = node.get('worksFor') or node.get('affiliation') or []
            employer = employer if isinstance(employer, list) else [employer]
            names = [text(e.get('name')) for e in employer if isinstance(e, dict)]
            agrees = normalized(person['institution']) in [normalized(n) for n in names]
            profiles.append({'url': url, 'checked': today.isoformat(), 'role': text(node.get('jobTitle'), 200),
                             'institutions': names, 'agrees': agrees})
        if 'JobPosting' not in types or not text(node.get('title')):
            continue
        try:
            deadline = parse_date(str(node.get('validThrough') or '')[:10])
        except ValueError:
            continue
        if node.get('validThrough') and not deadline:
            continue
        if deadline and deadline < today:
            continue
        description = text(node.get('description'), 2000)
        # Only advertise student/research training opportunities, never arbitrary staff vacancies.
        if not re.search(r'ph\.?d|doctoral|doctorat|internship|\bstage\b|studentship', text(node.get('title'))+' '+description, re.I):
            continue
        salary = node.get('baseSalary')
        if isinstance(salary, dict):
            value = salary.get('value')
            if isinstance(value, dict):
                amount = value.get('value')
                if amount is None and value.get('minValue') is not None and value.get('maxValue') is not None:
                    amount = str(value['minValue']) + '–' + str(value['maxValue'])
                salary = ' '.join(str(v) for v in (amount, salary.get('currency'), value.get('unitText')) if v is not None) if amount is not None else ''
            else:
                salary = ' '.join(str(v) for v in (value, salary.get('currency')) if v is not None) if isinstance(value, (str, int, float)) else ''
        offer_url = urljoin(url, node.get('url')) if isinstance(node.get('url'), str) else url
        if not official(offer_url, person.get('website') or url):
            offer_url = url
        offers.append({'title': text(node['title'], 300), 'url': offer_url,
                       'description': description, 'deadline': deadline.isoformat() if deadline else '',
                       'level': text(node.get('educationRequirements')),
                       'funding': text(salary) if isinstance(salary, str) else '',
                       'checked': today.isoformat(), 'institution': person['institution']})
    return profiles, offers


def inspect(person, supplied_url=None):
    """At most four pages. A directory name alone never proves employment."""
    website = person.get('website', '')
    result = {'profiles': [], 'offers': [], 'checked': timezone.now().date().isoformat(), 'partial': False}
    parsed = urlsplit(website)
    start_url = website
    if parsed.scheme == 'http' and host(website, allow_http=True):
        start_url = 'https://' + parsed.hostname + (parsed.path or '/') + ('?' + parsed.query if parsed.query else '')
    queue = [supplied_url or start_url]
    seen = set()
    parts = person.get('name', '').split()
    surname = parts[-1] if parts else ''
    while queue and len(seen) < 4:
        url = queue.pop(0)
        if url in seen or not official(url, website):
            continue
        seen.add(url)
        try:
            page, final = fetch(url, website)
        except Unavailable:
            result['partial'] = True
            continue
        profiles, offers = records(page, final, person)
        result['profiles'].extend(p for p in profiles if p not in result['profiles'])
        result['offers'].extend(o for o in offers if o not in result['offers'])
        if not supplied_url:
            for href in page.links:
                target = urljoin(final, href).split('#')[0]
                if (re.search(r'annuaire|directory|people|staff|equipe|careers|recrut|vacanc|jobs|'+re.escape(surname or 'NOMATCH'), href, re.I)
                        and official(target, website) and target not in seen and target not in queue):
                    queue.append(target)
    result['partial'] |= bool(queue)
    return result
