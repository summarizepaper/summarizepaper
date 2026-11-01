"""Conference programs read from a link: an Indico event through its timetable API (exact days, times, rooms,
presenters and abstracts, for thousands of contributions), a calendar feed (.ics, as Sched, Pretalx and Indico
give, or the one a page links to), or any other page or PDF, whose text the model then reads
(conference.list_talks).

Only public addresses are fetched: a program link is typed by a researcher, never an address of our network.
"""
import datetime
import ipaddress
import json
import re
import socket
import time
from urllib.parse import urljoin, urlsplit

import requests
from bs4 import BeautifulSoup

from . import sources

MAX_BYTES = 12 * 1024 * 1024
TIMEOUT = (5, 30)
REDIRECTS = 4
RETRY_SECONDS = 3
ABSTRACT_CHARS = 1500
INDICO_RE = re.compile(r'^(?P<base>.*?)/event/(?P<id>\d+)(?:/|$)')
PRETALX_RE = re.compile(r'^(?P<event>https?://[^/]+/[\w-]+)/(?:schedule|talk|speaker)(?:/|$)')
ANSI_RE = re.compile(r'\x1b\[[0-9;]*[A-Za-z]')


class LinkError(Exception):
    """The link cannot be read: shown to the researcher as 'url'."""


def public_host(host):
    """True when every address of host is a public one (no loopback, private, link-local or reserved network)."""
    try:
        addresses = {info[4][0] for info in socket.getaddrinfo(host, None)}
    except (socket.gaierror, UnicodeError):
        return False
    return bool(addresses) and all(ipaddress.ip_address(a.split('%')[0]).is_global for a in addresses)


def fetch(url):
    """(final url, content type, bytes) of a public http(s) link, redirects checked one by one; a server error is
    tried again once, a few seconds later."""
    try:
        return fetch_once(url)
    except LinkError as e:
        if not str(e).startswith('answered 5'):
            raise
        time.sleep(RETRY_SECONDS)
        return fetch_once(url)


def fetch_once(url):
    for _ in range(REDIRECTS + 1):
        parts = urlsplit(url)
        if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password \
                or parts.port not in (None, 80, 443) or not public_host(parts.hostname):
            raise LinkError('not a public link')
        try:
            with requests.get(url, timeout=TIMEOUT, allow_redirects=False, stream=True,
                              headers={'User-Agent': sources.USER_AGENT, 'Accept': '*/*'}) as response:
                if response.status_code in (301, 302, 303, 307, 308):
                    url = urljoin(url, response.headers.get('Location', ''))
                    continue
                if response.status_code != 200:
                    raise LinkError('answered %s' % response.status_code)
                chunks, size = [], 0
                for chunk in response.iter_content(65536):
                    size += len(chunk)
                    if size > MAX_BYTES:
                        raise LinkError('too large')
                    chunks.append(chunk)
                kind = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
                return url, kind, b''.join(chunks)
        except requests.RequestException as e:
            raise LinkError(str(e)[:200])
    raise LinkError('too many redirects')


def html_text(data):
    """The readable text of a page, one block per line."""
    soup = BeautifulSoup(data, 'html.parser')
    for tag in soup(['script', 'style', 'noscript', 'svg', 'nav', 'footer', 'header', 'form']):
        tag.decompose()
    text = soup.get_text('\n')
    return re.sub(r'\n\s*\n+', '\n\n', re.sub(r'[ \t\r\f\v]+', ' ', text)).strip()


def title_of(data):
    soup = BeautifulSoup(data[:200000], 'html.parser')
    return ' '.join((soup.title.string or '').split())[:200] if soup.title and soup.title.string else ''


# --- Indico ---------------------------------------------------------------------

def indico_text(html):
    return re.sub(r'\s+', ' ', BeautifulSoup(html or '', 'html.parser').get_text(' ')).strip()


def indico_talks(timetable, base, event):
    """The contributions of an Indico timetable (export/timetable/<id>.json), in time order, and its time zone."""
    days = next(iter((timetable.get('results') or {}).values()), None) or {}
    talks, zone = [], ''

    def add(entry, session='', room='', poster=False):
        nonlocal zone
        start, end = entry.get('startDate') or {}, entry.get('endDate') or {}
        zone = zone or start.get('tz') or ''
        presenters = [p for p in entry.get('presenters') or [] if isinstance(p, dict) and p.get('name')]
        talk = {'title': ' '.join(str(entry.get('title') or '').split()),
                'speakers': ', '.join(p['name'] for p in presenters),
                'presenter': presenters[0]['name'] if presenters else '',
                'affiliation': (presenters[0].get('affiliation') or '') if presenters else '',
                'session': session, 'kind': 'poster' if poster else 'talk',
                'day': start.get('date') or '', 'start': (start.get('time') or '')[:5], 'end': (end.get('time') or '')[:5],
                'room': entry.get('room') or room, 'abstract': indico_text(entry.get('description'))[:ABSTRACT_CHARS],
                'url': '%s/event/%s/contributions/%s/' % (base, event, entry['contributionId'])
                       if entry.get('contributionId') else ''}
        if talk['title']:
            talks.append(talk)

    for day in sorted(days):
        for entry in sorted((days[day] or {}).values(), key=lambda e: ((e.get('startDate') or {}).get('time') or '')):
            if entry.get('entryType') == 'Session':
                name = ' '.join(str(entry.get('slotTitle') or entry.get('title') or '').split())
                for child in sorted((entry.get('entries') or {}).values(),
                                    key=lambda e: ((e.get('startDate') or {}).get('time') or '')):
                    if child.get('entryType') == 'Contribution':
                        add(child, name, entry.get('room') or '', bool(entry.get('isPoster')))
            elif entry.get('entryType') == 'Contribution':
                add(entry)
    return talks, zone


def read_indico(url):
    """(name, talks, time zone) of an Indico event, or None when the link is not one or its timetable is closed."""
    match = INDICO_RE.match(url)
    if not match:
        return None
    base, event = match['base'], match['id']
    try:
        _, _, body = fetch('%s/export/timetable/%s.json' % (base, event))
        timetable = json.loads(body)
    except (LinkError, ValueError):
        return None
    talks, zone = indico_talks(timetable, base, event)
    if not talks:
        return None
    name = ''
    try:
        _, _, body = fetch('%s/export/event/%s.json' % (base, event))
        name = ' '.join(str((json.loads(body).get('results') or [{}])[0].get('title') or '').split())[:200]
    except (LinkError, ValueError, IndexError, AttributeError):
        pass
    return name, talks, zone


# --- calendar feeds ---------------------------------------------------------------

def ics_value(value):
    return re.sub(r'\\([,;\\])', r'\1', value.replace('\\n', '\n').replace('\\N', '\n')).strip()


def ics_time(name, params, value):
    """(day, 'HH:MM', zone) of a DTSTART or DTEND line."""
    zone = re.search(r'TZID=([^;:]+)', params)
    value = value.strip()
    if len(value) == 8:  # a whole day
        return '%s-%s-%s' % (value[:4], value[4:6], value[6:8]), '', ''
    match = re.match(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})', value)
    if not match:
        return '', '', ''
    return '%s-%s-%s' % match.groups()[:3], '%s:%s' % match.groups()[3:], 'UTC' if value.endswith('Z') else (
        zone[1].strip('"') if zone else '')


def ics_talks(text):
    """The events of a calendar feed as talks, and its time zone."""
    text = re.sub(r'\r?\n[ \t]', '', text)  # unfold the long lines
    pretalx = bool(re.search(r'^PRODID:.*pretalx', text, re.I | re.M))  # its summaries are 'Title - Speaker, Speaker'
    talks, zone = [], ''
    for block in re.findall(r'BEGIN:VEVENT(.*?)END:VEVENT', text, re.S):
        fields = {}
        for line in block.splitlines():
            name, _, value = line.partition(':')
            key, _, params = name.partition(';')
            fields.setdefault(key.upper(), (params, value))
        summary = ics_value(fields.get('SUMMARY', ('', ''))[1])
        if not summary:
            continue
        day, start, start_zone = ics_time('DTSTART', *fields.get('DTSTART', ('', '')))
        _, end, _ = ics_time('DTEND', *fields.get('DTEND', ('', '')))
        zone = zone or start_zone
        description = ics_value(fields.get('DESCRIPTION', ('', ''))[1])
        speakers = re.search(r'^(?:speakers?|presenters?|orateurs?)\s*:\s*(.+)$', description, re.I | re.M)
        speakers = speakers[1].strip() if speakers else ''
        if pretalx and not speakers and ' - ' in summary:
            summary, speakers = summary.rsplit(' - ', 1)
        talks.append({'title': ' '.join(summary.split()), 'speakers': speakers,
                      'presenter': '', 'affiliation': '', 'session': ics_value(fields.get('CATEGORIES', ('', ''))[1]),
                      'kind': 'talk', 'day': day, 'start': start, 'end': end,
                      'room': ics_value(fields.get('LOCATION', ('', ''))[1]),
                      'abstract': ' '.join(description.split())[:ABSTRACT_CHARS],
                      'url': ics_value(fields.get('URL', ('', ''))[1])[:500]})
    talks.sort(key=lambda t: (t['day'], t['start']))
    name = re.search(r'^X-WR-CALNAME:(.+)$', text, re.M)
    return (ics_value(name[1])[:200] if name else ''), talks, zone


def is_ics(kind, data):
    return kind == 'text/calendar' or data.lstrip()[:15].upper() == b'BEGIN:VCALENDAR'


# --- any link -----------------------------------------------------------------------

def feed_links(url, kind, data):
    """The calendar feeds a page leads to, best first: the export of a Pretalx schedule (which answers a script
    with a drawing of its timetable), the feed a page declares or links to."""
    links = []
    match = PRETALX_RE.match(url)
    if match:
        links.append(match['event'] + '/schedule/export/schedule.ics')
    if 'html' in kind:
        soup = BeautifulSoup(data[:2000000], 'html.parser')
        for tag in soup.find_all(['link', 'a'], href=True):
            if tag.get('type', '').lower() == 'text/calendar' or re.search(r'\.ics(?:$|\?)', tag['href']):
                links.append(urljoin(url, tag['href']))
    return list(dict.fromkeys(links))[:3]


def read_feed(url):
    try:
        _, kind, data = fetch(url)
    except LinkError:
        return None
    if not is_ics(kind, data):
        return None
    name, talks, zone = ics_talks(data.decode('utf-8', 'replace'))
    return {'name': name, 'talks': talks, 'timezone': zone, 'source': url} if talks else None


def read_link(url, pdf_text):
    """{'name', 'talks', 'timezone'} when the program comes structured, else {'name', 'text'} for the model to read.
    pdf_text: the reader of a PDF's text (missed.pdf_text). Raises LinkError."""
    url = url.strip()
    if not re.match(r'https?://', url, re.I):
        url = 'https://' + url
    indico = read_indico(url)
    if indico:
        return {'name': indico[0], 'talks': indico[1], 'timezone': indico[2], 'source': url}
    final, kind, data = fetch(url)
    indico = read_indico(final) if final != url else None
    if indico:
        return {'name': indico[0], 'talks': indico[1], 'timezone': indico[2], 'source': url}
    if is_ics(kind, data):
        name, talks, zone = ics_talks(data.decode('utf-8', 'replace'))
        if not talks:
            raise LinkError('an empty calendar')
        return {'name': name, 'talks': talks, 'timezone': zone, 'source': url}
    if kind == 'application/pdf' or data[:5] == b'%PDF-':
        return {'name': '', 'text': pdf_text(data), 'source': url}
    for link in feed_links(final, kind, data):
        feed = read_feed(link)
        if feed:
            return dict(feed, name=feed['name'] or (title_of(data) if 'html' in kind else ''))
    if 'html' in kind or data.lstrip()[:1] == b'<':
        return {'name': title_of(data), 'text': html_text(data), 'source': url}
    return {'name': '', 'text': ANSI_RE.sub('', data.decode('utf-8', 'replace')), 'source': url}


def utc_stamp(day, time, zone):
    """'20240718T133800Z' of a local day and time in zone (IANA), or the floating '20240718T153800' when the zone
    is unknown: calendars then show the time as written."""
    stamp = datetime.datetime.strptime('%s %s' % (day, time), '%Y-%m-%d %H:%M')
    if zone:
        try:
            from zoneinfo import ZoneInfo
            return stamp.replace(tzinfo=ZoneInfo(zone)).astimezone(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        except (KeyError, ValueError, ImportError):
            pass
    return stamp.strftime('%Y%m%dT%H%M%S')
