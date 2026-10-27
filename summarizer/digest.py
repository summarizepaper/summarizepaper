"""Email digests of new arXiv, bioRxiv, medRxiv and journal papers, by category, keywords and, with Premium, a research
profile.

Run every hour (Heroku Scheduler): python manage.py send_digest

arXiv announces its new papers once a day. Once per announcement, the RSS feed of
every archive is read and the whole batch is stored (AnnouncedPaper); a digest then
gets the best papers of the batches announced since its previous email.

bioRxiv and medRxiv post every day, weekends included. Each arXiv announcement also
takes their new preprints (first versions) of the days since the previous one, read
from api.biorxiv.org, up to the day before yesterday, so the days read are complete
(see bio_window). Their categories are written 'biorxiv.neuroscience'. When their
API fails, the batch is stored without them and only the digests following their
categories wait for them, as for an arXiv archive.

The journals, and the preprint servers other than these three (ChemRxiv, OSF, SSRN...),
come from OpenAlex by subfield ('journals.1605', see journal_taxonomy), only for the
subfields some digest follows. OpenAlex cannot tell when it added a work, so each
announcement reads the works published over the last JOURNAL_DAYS days and keeps those
not stored yet (see read_journals).

Whatever costs a model call is done once and shared by every subscriber: the
2-sentence blurb (per paper and language), the embedding (per paper), the full
summary (per paper and language) and the weekly synthesis (per category,
language and day). Only the ranking is computed per subscriber.
"""
import datetime
import hashlib
import json
import logging
import re
import time
import unicodedata
import zoneinfo
from xml.etree import ElementTree

import numpy as np
import requests
from django.conf import settings
from django.core import signing
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.utils.crypto import constant_time_compare, salted_hmac
from django.db import models
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.dateparse import parse_datetime

from . import access, arxiv_taxonomy, audio, journal_taxonomy, mailcheck, similar, sources, usage
from .models import (AnnouncedPaper, ArxivBatch, CategorySynthesis, DigestDelivery, DigestEdition, DigestFeedback,
                     DigestItem, DigestSubscription, DigestVector, PaperAudio, SummaryPaper)
from .utils import ai_models, chat_params

logger = logging.getLogger(__name__)

ARXIV_API = 'https://export.arxiv.org/api/query'
ATOM = '{http://www.w3.org/2005/Atom}'
ARXIV_NS = '{http://arxiv.org/schemas/atom}'
DC = '{http://purl.org/dc/elements/1.1/}'

FREE = {'categories': 5, 'papers': 10, 'digests': 1}
PREMIUM = {'categories': 15, 'papers': 25, 'digests': 5}
MAX_CATEGORIES = FREE['categories']
PAPERS_PER_EMAIL = FREE['papers']
MAX_AUTHORS = 10
MAX_REFERENCES = 5
PRESUMMARIZED = 3  # the top papers of a Premium digest get their full summary written before the email
MAX_PRESUMMARIES_PER_RUN = 9  # each takes about a minute
MAX_SYNTHESES_PER_EMAIL = 5
SYNTHESIS_MIN_PAPERS = 5
SYNTHESIS_MAX_PAPERS = 600
SYNTHESIS_MAX_BLURBS = 150
NOTABLE_SHORTLIST = 15

# CloudMailin sends at most 20 emails/hour on our plan; keep room for account emails.
# Subscribers left over are sent in the next hourly runs (see DigestSubscription.CATCH_UP_HOURS).
MAX_EMAILS_PER_RUN = 15
WEEK = datetime.timedelta(days=7)
KEEP_DAYS = 10  # announcements a run reads, for weekly digests and syntheses
PAGE_DAYS = 14  # announced papers kept, for the web page of a weekly digest
EDITION_DAYS = 90  # what a digest email showed, for the click statistics
RSS = 'https://rss.arxiv.org/atom/'
ARCHIVES = ['astro-ph', 'cond-mat', 'cs', 'econ', 'eess', 'gr-qc', 'hep-ex', 'hep-lat', 'hep-ph', 'hep-th', 'math',
            'math-ph', 'nlin', 'nucl-ex', 'nucl-th', 'physics', 'q-bio', 'q-fin', 'quant-ph', 'stat']
BIO_ARCHIVES = ['biorxiv', 'medrxiv']
JOURNALS = 'journals'
OTHER_ARCHIVES = BIO_ARCHIVES + [JOURNALS]  # read with each arXiv announcement, never a sign of an arXiv holiday
SERVER_NAMES = {'biorxiv': 'bioRxiv', 'medrxiv': 'medRxiv', JOURNALS: 'journals and preprint servers'}
BIORXIV_DETAILS = 'https://api.biorxiv.org/details/%s/%s/%s/%d'
BIO_MAX_PAGES = 200
JOURNAL_DAYS = 9  # publication days read again at each announcement: OpenAlex has most works 3 to 4 days after
JOURNAL_FIRST_DAYS = 2  # a subfield read for the first time sends only these days, and keeps the others as seen
JOURNAL_MAX_PAGES = 25  # of 200 works, per subfield and announcement
# arXiv, bioRxiv and medRxiv come from their own feeds; Zenodo and Figshare are mostly data, slides and posters
JOURNAL_SKIPPED_SOURCES = ('S4306400194', 'S4306402567', 'S3005729997', 'S4306400562', 'S4377196282')
RSS_MAX_ENTRIES = 2000  # a longer feed is cut by arXiv; its followed categories are then read one by one
PARTIAL_AFTER = datetime.timedelta(hours=3)  # then a batch is accepted without the archives that still fail
QUIET_KEY = 'arxiv-rss-quiet:%s'  # set when no archive has the expected day hours after it was due (arXiv holiday)
QUIET_SECONDS = 3 * 3600
SECONDS_BETWEEN_CALLS = 3  # arXiv API terms: at most one request every 3 seconds
RETRY_WAITS = (10, 30, 90)  # arXiv answers 429 "Rate exceeded" when busy
EMBED_BATCH = 256
UNSUBSCRIBE_SALT = 'digest-unsubscribe'
FEEDBACK_SALT = 'digest-feedback'
EDITION_SALT = 'digest-edition'
OPEN_SALT = 'digest-open'
LINK_SALT = 'digest-link'
LINK_SIGNATURE = 16  # hex characters: 64 bits
LINK_RE = re.compile(r'^(([okv])(\d+)\.(\d+)([sapud]?))\.([0-9a-f]+)$')
TARGETS = {'s': 'summary', 'a': 'abs', 'p': 'pdf'}
SAVE_SALT = 'digest-save'
PROFILE_VERSION = 'parts-v1'  # profile_vector holds one row per profile paragraph and reference paper
PROFILE_PARTS = 5
PROFILE_CACHE = 'digest-profile:%s'  # embeddings of a profile, shared by the previews and the saved digest
MAX_FEEDBACK = 60  # latest votes and clicks a ranking reads
LIKE_WEIGHTS = {'vote': 0.9, 'click': 0.75}  # a liked or opened paper counts a bit less than the profile
DISLIKE_WEIGHT = 0.5
KEYWORD_BOOST = 0.04  # per keyword found (2 at most), with a profile
DIVERSITY = 0.15
DIVERSE_POOL = 200
DIVERSE_TOP = 60
# cosine similarity (text-embedding-3-small, 512 dimensions) a paper needs in a 'threshold' digest
THRESHOLDS = {'high': 0.55, 'medium': 0.48, 'low': 0.42}
THRESHOLD_MAX_EMAIL = 50  # the others are on the digest's web page
POINTS = ('contribution', 'method', 'result')
LOCK_KEY = 'digest-run-lock'
SENDING_KEY = 'digest-sending:%s'
HEALTH_KEY = 'digest-health:%s:%s'  # one alert per problem and day
PREVIEW_KEY = 'digest-preview:%s:%s'  # one preview per digest and day
PREVIEWS_PER_HOUR = 5  # previews share the mail provider's hourly limit with the scheduled digests
BASE_URL = 'https://www.summarizepaper.com'
LOCK_SECONDS = 2 * 3600
# arXiv's RSS feeds switch to the next announcement around 00:05 in New York, Monday to Friday
ANNOUNCE_TZ = zoneinfo.ZoneInfo('America/New_York')
RSS_DELAY = datetime.timedelta(minutes=30)

# bioRxiv and medRxiv subject areas offered in the form (their API names, see sources.category_code)
BIO_CATEGORIES = [
    ('biorxiv', ['Animal Behavior and Cognition', 'Biochemistry', 'Bioengineering', 'Bioinformatics', 'Biophysics',
                 'Cancer Biology', 'Cell Biology', 'Developmental Biology', 'Ecology', 'Evolutionary Biology', 'Genetics', 'Genomics', 'Immunology', 'Microbiology', 'Molecular Biology',
                 'Neuroscience', 'Paleontology', 'Pathology', 'Pharmacology and Toxicology', 'Physiology',
                 'Plant Biology', 'Scientific Communication and Education', 'Synthetic Biology',
                 'Systems Biology', 'Zoology']),
    ('medrxiv', ['Addiction Medicine', 'Allergy and Immunology', 'Anesthesia', 'Cardiovascular Medicine',
                 'Dentistry and Oral Medicine', 'Dermatology', 'Emergency Medicine', 'Endocrinology', 'Epidemiology',
                 'Forensic Medicine', 'Gastroenterology', 'Genetic and Genomic Medicine', 'Geriatric Medicine',
                 'Health Economics', 'Health Informatics', 'Health Policy', 'Health Systems and Quality Improvement',
                 'Hematology', 'HIV AIDS',
                 'Infectious Diseases', 'Intensive Care and Critical Care Medicine', 'Medical Education',
                 'Medical Ethics', 'Nephrology', 'Neurology', 'Nursing', 'Nutrition', 'Obstetrics and Gynecology',
                 'Occupational and Environmental Health', 'Oncology', 'Ophthalmology', 'Orthopedics',
                 'Otolaryngology', 'Pain Medicine', 'Palliative Medicine', 'Pathology', 'Pediatrics',
                 'Pharmacology and Therapeutics', 'Primary Care Research', 'Psychiatry and Clinical Psychology',
                 'Public and Global Health', 'Radiology and Imaging', 'Rehabilitation Medicine and Physical Therapy',
                 'Respiratory Medicine', 'Rheumatology', 'Sexual and Reproductive Health', 'Sports Medicine',
                 'Surgery', 'Toxicology', 'Transplantation', 'Urology']),
]

TIMEZONES = {tz for tz in zoneinfo.available_timezones() if '/' in tz and not tz.startswith(('Etc/', 'SystemV/', 'US/'))} | {'UTC'}
CATEGORY_RE = re.compile(r'^[a-z\-]+(\.[A-Za-z0-9\-]+)?$')
BIO_CODES = {sources.category_code(server, name): (server, name) for server, names in BIO_CATEGORIES for name in names}
ARXIV_ID_RE = re.compile(r'(\d{4}\.\d{4,5}(?:v\d+)?|[a-z\-]+(?:\.[A-Z]{2})?/\d{7}(?:v\d+)?)')


class ArxivUnavailable(Exception):
    pass


def limits(paid):
    return PREMIUM if paid else FREE


def clean_categories(values, limit=MAX_CATEGORIES):
    seen = []
    for value in values:
        value = value.strip()
        if value and CATEGORY_RE.match(value) and value not in seen:
            seen.append(value)
    return seen[:limit]


def clean_references(text):
    """Papers found in what the user typed: arXiv ids or links, in arXiv form; DOIs, other links and
    keys of other sources, as keys (see sources.py)."""
    found = []
    for token in re.split(r'[\s,;]+', text):
        key = token.strip().lower()
        if not key:
            continue
        if sources.is_external(key):
            ref = key if sources.valid_key(key) else None
        else:
            doi = sources.clean_doi(token)
            match = None if doi and not sources.ARXIV_DOI_RE.match(doi) else ARXIV_ID_RE.search(token.replace('--', '/'))
            ref = sources.doi_key(doi) if doi and not match else (match.group(1) if match else None)
        if ref and ref not in found:
            found.append(ref)
    return found[:MAX_REFERENCES]


def clean_authors(text):
    names = []
    for name in text.split(','):
        name = re.sub(r'\s+', ' ', name).strip()[:80]
        if name_key(name) and name not in names:
            names.append(name)
    return names[:MAX_AUTHORS]


def name_key(name):
    """'Yann LeCun' -> ('lecun', 'yann'), 'Y. LeCún' -> ('lecun', 'y'): last name and first name or initial."""
    plain = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower()
    parts = re.findall(r'[a-z]+', plain)
    if len(parts) < 2:
        return None
    return parts[-1], parts[0]


def same_author(key, name):
    """Same last name, and same first name, or same initial when one of them is only an initial."""
    other = name_key(name)
    if not other or key[0] != other[0] or key[1][0] != other[1][0]:
        return False
    return len(key[1]) == 1 or len(other[1]) == 1 or key[1] == other[1]


def site_id(arxiv_abs_id):
    """'hep-th/9901001v1' -> 'hep-th--9901001v1', the form used in our URLs."""
    return arxiv_abs_id.replace('/', '--')


def parse_feed(xml_text):
    papers = []
    root = ElementTree.fromstring(xml_text)
    for entry in root.findall(ATOM + 'entry'):
        abs_url = entry.findtext(ATOM + 'id', '')
        if '/abs/' not in abs_url:
            continue
        papers.append({
            'arxiv_id': site_id(abs_url.split('/abs/')[-1]),
            'title': re.sub(r'\s+', ' ', entry.findtext(ATOM + 'title', '')).strip(),
            'abstract': re.sub(r'\s+', ' ', entry.findtext(ATOM + 'summary', '')).strip(),
            'authors': [a.findtext(ATOM + 'name', '') for a in entry.findall(ATOM + 'author')],
            'categories': [c.get('term') for c in entry.findall(ATOM + 'category')],
            'published': parse_datetime(entry.findtext(ATOM + 'published', '')),
            'updated': parse_datetime(entry.findtext(ATOM + 'updated', '')),
            'doi': entry.findtext(ARXIV_NS + 'doi', '').strip(),
            'journal_ref': re.sub(r'\s+', ' ', entry.findtext(ARXIV_NS + 'journal_ref', '')).strip(),
            'comment': re.sub(r'\s+', ' ', entry.findtext(ARXIV_NS + 'comment', '')).strip(),
        })
    return papers


_last_call = [0.0]


def arxiv_get(params, timeout=60, retry=True):
    """One arXiv API query, spaced from the previous one and retried when arXiv asks us to slow down.
    A page request passes a short timeout and no retry: Heroku cuts requests after 30 seconds."""
    error = None
    for wait in (RETRY_WAITS if retry else ()) + (None,):
        pause = _last_call[0] + SECONDS_BETWEEN_CALLS - time.monotonic()
        if pause > 0:
            time.sleep(pause)
        _last_call[0] = time.monotonic()
        try:
            response = requests.get(ARXIV_API, params=params, timeout=timeout)
            if response.status_code == 200:
                return parse_feed(response.text)
            error = 'HTTP %s: %s' % (response.status_code, response.text[:80])
        except (requests.RequestException, ElementTree.ParseError) as e:
            error = repr(e)
        if wait is not None:
            logger.info('arXiv query %s failed (%s), retrying in %ds', params.get('search_query'), error, wait)
            time.sleep(wait)
    raise ArxivUnavailable(error)


def expected_batch(now):
    """Day of the latest arXiv announcement that should be in the RSS feeds at `now`."""
    day = (now - RSS_DELAY).astimezone(ANNOUNCE_TZ).date()
    while day.weekday() >= 5:
        day -= datetime.timedelta(days=1)
    return day


def latex_to_text(text):
    """'Osnabr\\"uck' -> 'Osnabrück'-ish ascii: accents written in LaTeX are dropped, the letter kept."""
    text = re.sub(r'\\[\'"`^~=.uvHckrb]\{?([A-Za-z])\}?', r'\1', text)
    return text.replace('{', '').replace('}', '').replace('\\', '')


def split_authors(creators):
    """'A. One (Univ X, Lab), B. Two and C. Three' -> ['A. One', 'B. Two', 'C. Three']."""
    text = latex_to_text(creators)
    while re.search(r'\([^()]*\)', text):
        text = re.sub(r'\([^()]*\)', '', text)
    names = re.split(r',|\band\b', text)
    return [re.sub(r'\s+', ' ', n).strip() for n in names if n.strip()]


def parse_rss(xml_text):
    """(announcement day, [paper]) of an arXiv RSS feed, new and cross-listed papers only."""
    root = ElementTree.fromstring(xml_text)
    day, papers = None, []
    for entry in root.findall(ATOM + 'entry'):
        if entry.findtext(ARXIV_NS + 'announce_type', '') not in ('new', 'cross'):
            continue  # replacements are new versions of papers already announced
        published = parse_datetime(entry.findtext(ATOM + 'published', ''))
        if published is None:
            continue
        day = published.date()
        summary = entry.findtext(ATOM + 'summary', '')
        papers.append({
            'arxiv_id': site_id(entry.findtext(ATOM + 'id', '').split(':')[-1]),
            'title': re.sub(r'\s+', ' ', entry.findtext(ATOM + 'title', '')).strip(),
            'abstract': re.sub(r'\s+', ' ', summary.split('Abstract:', 1)[-1]).strip(),
            'authors': split_authors(entry.findtext(DC + 'creator', '')),
            'categories': [c.get('term') for c in entry.findall(ATOM + 'category')],
            'published': published,
            'announced': day,
        })
    return day, papers


def read_feed(name):
    """(day, papers, entries in the feed) of one arXiv RSS feed, retried like the API."""
    error = None
    for wait in RETRY_WAITS + (None,):
        try:
            response = requests.get(RSS + name, timeout=120)
            if response.status_code == 200:
                day, papers = parse_rss(response.text)
                return day, papers, response.text.count('<entry>')
            error = 'HTTP %s' % response.status_code
        except (requests.RequestException, ElementTree.ParseError) as e:
            error = repr(e)
        if wait is not None:
            logger.info('arXiv RSS %s failed (%s), retrying in %ds', name, error, wait)
            time.sleep(wait)
    raise ArxivUnavailable('%s: %s' % (name, error))


def ensure_batch(now, categories=()):
    """Store the latest arXiv announcement if it is out and not complete yet; return the stored days."""
    expected = expected_batch(now)
    batch = ArxivBatch.objects.filter(day=expected).first()
    if (batch is None or batch.missing) and not cache.get(QUIET_KEY % expected):
        fetch_batch(expected, now, categories, batch)
    return list(ArxivBatch.objects.filter(day__gte=expected - datetime.timedelta(days=KEEP_DAYS))
                .order_by('day').values_list('day', flat=True))


def fetch_batch(expected, now, categories=(), batch=None):
    """Read the archive feeds (only the missing ones of a partial batch) and store the papers of `expected`.

    The batch is stored once every archive was read for that day, or after PARTIAL_AFTER without the
    archives that still fail; those are read again at each run until they answer.
    """
    archives = batch.missing_list() if batch else ARCHIVES + BIO_ARCHIVES + [JOURNALS] * bool(followed_journals())
    found, bio_found, missing, on_time, stale = {}, {}, [], 0, 0
    for archive in archives:
        if archive in OTHER_ARCHIVES:
            try:
                papers = read_journals(expected) if archive == JOURNALS else read_bio(archive, expected)
            except ArxivUnavailable:
                logger.exception('%s could not be read', SERVER_NAMES[archive])
                missing.append(archive)
                continue
            for paper in papers:
                bio_found.setdefault(paper['arxiv_id'], paper)
            continue  # never on time or stale: arXiv's feeds alone tell a holiday
        try:
            day, papers, entries = read_feed(archive)
            if entries >= RSS_MAX_ENTRIES:
                names = [c for c in categories if archive_of(c) == archive]
                logger.warning('arXiv RSS %s has %d entries, reading %s one by one', archive, entries, names)
                for name in names:
                    papers += read_feed(name)[1]
        except ArxivUnavailable:
            logger.exception('arXiv RSS %s could not be read', archive)
            missing.append(archive)
            continue
        if day is not None and day != expected:
            logger.info('arXiv RSS %s is still on %s, waiting for %s', archive, day, expected)
            missing.append(archive)
            stale += 1
            continue
        on_time += day is not None
        for paper in papers:
            if paper['announced'] == expected:
                found.setdefault(paper['arxiv_id'], paper)
    if batch is not None or on_time or not stale:
        # on an arXiv holiday no batch is stored: the next one reads these days again
        found.update(bio_found)
    AnnouncedPaper.objects.bulk_create([AnnouncedPaper(
        arxiv_id=p['arxiv_id'], announced=p['announced'], title=p['title'], abstract=p['abstract'],
        authors=json.dumps(p['authors']), categories=','.join(p['categories'])) for p in found.values()],
        ignore_conflicts=True, batch_size=500)
    if batch is not None:
        batch.missing = ','.join(missing)
        batch.papers = AnnouncedPaper.objects.filter(announced=expected).count()
        batch.save(update_fields=['missing', 'papers'])
    elif stale and not on_time:
        if now >= announcement_time(expected) + PARTIAL_AFTER:
            cache.set(QUIET_KEY % expected, True, QUIET_SECONDS)  # feeds still on an older day: arXiv holiday
    elif not [m for m in missing if m not in OTHER_ARCHIVES] or now >= announcement_time(expected) + PARTIAL_AFTER:
        # bioRxiv, medRxiv or the journals missing hold back only the digests that follow them (see _run)
        if missing:
            logger.warning('arXiv batch %s accepted without %s', expected, missing)
        ArxivBatch.objects.get_or_create(day=expected, defaults={
            'papers': AnnouncedPaper.objects.filter(announced=expected).count(), 'missing': ','.join(missing)})


def bio_window(expected):
    """(first, last) posting days of the bioRxiv and medRxiv preprints that go with the arXiv announcement of
    `expected`: from the day before the previous announcement to the day before yesterday. Each day is taken
    once (Monday takes Thursday to Saturday), and only once it is over everywhere."""
    previous = (ArxivBatch.objects.filter(day__lt=expected, day__gte=expected - WEEK)
                .order_by('-day').values_list('day', flat=True).first())
    if previous is None:
        previous = expected - datetime.timedelta(days=1)
        while previous.weekday() >= 5:
            previous -= datetime.timedelta(days=1)
    return previous - datetime.timedelta(days=1), expected - datetime.timedelta(days=2)


def bio_page(server, first, last, cursor):
    """(rows, total) of one page of api.biorxiv.org, retried like arXiv."""
    error = None
    for wait in RETRY_WAITS + (None,):
        try:
            response = requests.get(BIORXIV_DETAILS % (server, first, last, cursor), timeout=60,
                                    headers={'User-Agent': sources.USER_AGENT})
            if response.status_code == 200:
                body = response.json()
                message = (body.get('messages') or [{}])[0]
                return body.get('collection') or [], int(message.get('total') or 0)
            error = 'HTTP %s' % response.status_code
        except (requests.RequestException, ValueError) as e:
            error = repr(e)
        if wait is not None:
            logger.info('%s page %d failed (%s), retrying in %ds', server, cursor, error, wait)
            time.sleep(wait)
    raise ArxivUnavailable('%s: %s' % (server, error))


def read_bio(server, expected):
    """New preprints (first versions) of bioRxiv or medRxiv for the announcement of `expected`."""
    first, last = bio_window(expected)
    papers, cursor = [], 0
    for _ in range(BIO_MAX_PAGES):
        rows, total = bio_page(server, first, last, cursor)
        for row in rows:
            if str(row.get('version')) != '1' or not row.get('doi'):
                continue  # later versions are revisions of preprints already sent
            paper = sources.biorxiv_paper(row, server)
            published = datetime.datetime.fromisoformat(paper['published']) if paper['published'] else \
                datetime.datetime.combine(expected, datetime.time())
            papers.append({
                'arxiv_id': paper['key'], 'title': paper['title'], 'abstract': paper['abstract'],
                'authors': paper['authors'], 'categories': [paper['category']],
                'published': published.replace(tzinfo=ANNOUNCE_TZ), 'announced': expected,
            })
        cursor += len(rows)
        if not rows or cursor >= total:
            break
    return papers


def followed_journals():
    """The journal subfields at least one active digest follows, or where a watched project's prior art is published."""
    followed = set()
    for categories in DigestSubscription.objects.filter(active=True, categories__contains=JOURNALS + '.') \
            .values_list('categories', flat=True):
        followed.update(c.strip() for c in categories.split(',') if c.strip() in journal_taxonomy.NAMES)
    from . import scoops
    return sorted(followed | scoops.followed_subfields())


def openalex_page(params):
    """One page of OpenAlex works, retried like arXiv."""
    error = None
    for wait in RETRY_WAITS + (None,):
        try:
            return (sources._json(sources.OPENALEX + '/works', sources._openalex_params(params)) or {}).get('results') or []
        except sources.SourceUnavailable as e:
            error = str(e)
        if wait is not None:
            logger.info('OpenAlex page failed (%s), retrying in %ds', error, wait)
            time.sleep(wait)
    raise ArxivUnavailable('OpenAlex: %s' % error)


def read_journals(expected):
    """New works of the followed journal subfields for the announcement of `expected`: those published over the
    last JOURNAL_DAYS days, as OpenAlex has them now; fetch_batch stores only the ones not stored yet. The first
    time a subfield is read, its works older than JOURNAL_FIRST_DAYS are stored on their publication day, so that
    they count as seen without going out as new."""
    first, last = expected - datetime.timedelta(days=JOURNAL_DAYS), expected - datetime.timedelta(days=1)
    recent = expected - datetime.timedelta(days=JOURNAL_FIRST_DAYS)
    papers = []
    for code in followed_journals():
        new_subfield = not AnnouncedPaper.objects.filter(categories=code).exists()
        params = {'filter': ','.join(['primary_topic.subfield.id:%s' % code.split('.', 1)[1],
                                      'from_publication_date:%s' % first, 'to_publication_date:%s' % last,
                                      'has_abstract:true', 'has_doi:true', 'type:article|preprint|review',
                                      'primary_location.source.id:!' + '|'.join(JOURNAL_SKIPPED_SOURCES)]),
                  'select': sources.OPENALEX_FIELDS, 'per_page': 200}
        for page in range(1, JOURNAL_MAX_PAGES + 1):  # pages, not a cursor: OpenAlex wants a key for cursors
            works = openalex_page(dict(params, page=page))
            for work in works:
                paper = sources.openalex_paper(work)
                if not paper['key'] or paper['doi'].startswith(sources.BIORXIV_PREFIXES) or not paper['abstract']:
                    continue
                published = datetime.date.fromisoformat(paper['published']) if paper['published'] else last
                papers.append({
                    'arxiv_id': paper['key'], 'title': paper['title'], 'abstract': paper['abstract'],
                    'authors': paper['authors'], 'categories': [code],
                    'announced': published if new_subfield and published < recent else expected,
                })
            if len(works) < params['per_page']:
                break
    return papers


def announcement_time(day):
    return datetime.datetime.combine(day, datetime.time(), tzinfo=ANNOUNCE_TZ) + RSS_DELAY


def archive_of(category):
    return category.split('.')[0]


def category_name(category):
    """'biorxiv.cell-biology' -> 'Cell Biology', 'journals.1605' -> 'Organic Chemistry'; arXiv codes stay as they are."""
    if category in journal_taxonomy.NAMES:
        with translation.override('en'):
            return str(journal_taxonomy.NAMES[category])
    return BIO_CODES.get(category, (None, category))[1]


def load_papers(days):
    """{arxiv_id: paper} announced on these days."""
    papers = {}
    for row in AnnouncedPaper.objects.filter(announced__in=days).iterator():
        papers[row.arxiv_id] = {
            'arxiv_id': row.arxiv_id, 'title': row.title, 'abstract': row.abstract,
            'authors': json.loads(row.authors or '[]'), 'categories': row.categories.split(','),
            'announced': row.announced,
            'published': datetime.datetime.combine(row.announced, datetime.time(), tzinfo=ANNOUNCE_TZ),
        }
    return papers


def author_index(papers):
    """{last name: [(paper, author name)]}, to find the followed authors in a batch quickly."""
    index = {}
    for paper in papers.values():
        for name in paper['authors']:
            key = name_key(name)
            if key:
                index.setdefault(key[0], []).append((paper, name))
    return index


def fetch_ids(ids, **options):
    if not ids:
        return []
    return arxiv_get({'id_list': ','.join(ids), 'max_results': len(ids)}, **options)


# --- ranking -----------------------------------------------------------------

def paper_text(paper):
    return re.sub(r'[ \t\r\f\v]+', ' ', '%s\n\n%s' % (paper['title'], paper['abstract'])).strip()[:similar.MAX_INPUT_CHARS]


def ensure_vectors(papers):
    """{arxiv_id: vector} for these papers, embedding only the ones never seen before."""
    ids = [p['arxiv_id'] for p in papers]
    vectors = stored_vectors(ids)
    missing = [p for p in papers if p['arxiv_id'] not in vectors]
    for start in range(0, len(missing), EMBED_BATCH):
        batch = missing[start:start + EMBED_BATCH]
        try:
            embedded = similar.embed([paper_text(p) for p in batch])
        except (requests.RequestException, KeyError, ValueError):
            logger.exception('Embedding of %d digest papers failed', len(batch))
            continue
        DigestVector.objects.bulk_create([DigestVector(arxiv_id=p['arxiv_id'], vector=v.tobytes())
                                          for p, v in zip(batch, embedded)], ignore_conflicts=True)
        vectors.update((p['arxiv_id'], v) for p, v in zip(batch, embedded))
    return vectors


def stored_vectors(ids):
    """{arxiv_id: vector} of the papers already embedded; pages use only these, never the model."""
    vectors = {}
    for start in range(0, len(ids), 1000):
        for aid, vec in DigestVector.objects.filter(arxiv_id__in=ids[start:start + 1000]).values_list('arxiv_id', 'vector'):
            vectors[aid] = np.frombuffer(vec, dtype=np.float32)
    return vectors


class Keywords:
    """The keywords of a digest: 'comet, "debris disk", -simulation'.

    A term matches whole words, with or without a plural 's'; a quoted term matches as written; a term
    after '-' leaves out the papers that use it.
    """

    def __init__(self, text):
        self.include, self.exclude = [], []
        for raw in re.findall(r'-?\s*"[^"]+"|[^,"]+', text or ''):
            term = raw.strip()
            negative = term.startswith('-')
            term = term[1:].strip() if negative else term
            exact = len(term) > 1 and term[0] == term[-1] == '"'
            term = term.strip('"').strip()
            pattern = keyword_pattern(term, exact)
            if pattern is not None:
                (self.exclude if negative else self.include).append((term, pattern))

    def __bool__(self):
        return bool(self.include or self.exclude)

    def excluded(self, paper):
        text = paper['title'] + ' ' + paper['abstract']
        return any(pattern.search(text) for _, pattern in self.exclude)

    def matches(self, paper):
        """(score, [terms found]): mentions in the title and abstract, a title mention counting more."""
        text = paper['title'] + ' ' + paper['abstract']
        score, found = 0, []
        for term, pattern in self.include:
            count = len(pattern.findall(text))
            if count:
                found.append(term)
                score += count + 3 * bool(pattern.search(paper['title']))
        return score, found


def keyword_pattern(term, exact=False):
    words = re.findall(r'[^\s\-]+', term.lower())
    if not words:
        return None
    parts = [re.escape(w) for w in words]
    last = words[-1]
    if not exact and re.fullmatch(r'[a-z]{3,}', last):
        if len(last) > 4 and last.endswith('s') and not last.endswith(('ss', 'is', 'us')):
            last = last[:-1]
        parts[-1] = re.escape(last) + '(?:s|es)?'
    return re.compile(r'(?<![a-z0-9])' + r'[\s\-]+'.join(parts) + r'(?![a-z0-9])', re.I)


def profile_parts(text):
    """The paragraphs of a research profile, each embedded on its own (a researcher may have several subjects)."""
    parts = [p.strip() for p in re.split(r'\n+', text or '') if len(p.split()) >= 4]
    if not parts and (text or '').strip():
        parts = [text.strip()]
    return parts[:PROFILE_PARTS]


def reference_entries(refs):
    """[(reference, title and abstract, title)] of the reference papers of a research profile."""
    entries = [(p['arxiv_id'], paper_text(p), p['title']) for p in fetch_ids([r for r in refs if not sources.is_external(r)])]
    for key in (r for r in refs if sources.is_external(r)):
        try:
            paper = sources.metadata(key)
        except sources.SourceUnavailable:
            logger.warning('Reference %s of a research profile could not be read', key)
            continue
        if paper:
            entries.append((key, paper_text(paper), paper['title']))
    return entries


def reference_texts(refs):
    """Title and abstract of the reference papers of a research profile, from arXiv and the other sources."""
    return [text for _, text, _ in reference_entries(refs)]


def short(text, length=70):
    text = re.sub(r'\s+', ' ', text or '').strip()
    return text if len(text) <= length else text[:length].rsplit(' ', 1)[0] + '…'


def profile_matrix(subscription, save=True, quick=False):
    """(rows, labels): one embedding per paragraph of the research profile and per reference paper,
    recomputed only when they change. A paper is scored by the closest row, so a researcher with
    two subjects is not given the average of both.

    save=False leaves the subscription as it is (a preview of settings not saved yet); quick leaves out
    reference papers not embedded yet, which take arXiv calls a page cannot wait for.
    """
    text = subscription.profile.strip()
    refs = subscription.reference_list()[:MAX_REFERENCES]
    if not text and not refs:
        return None, []
    key = hashlib.sha256(json.dumps([PROFILE_VERSION, similar.MODEL, text, refs]).encode()).hexdigest()
    if subscription.profile_key == key and subscription.profile_vector and subscription.profile_labels:
        labels = json.loads(subscription.profile_labels)
        return np.frombuffer(bytes(subscription.profile_vector), dtype=np.float32).reshape(len(labels), -1), labels
    cached = cache.get(PROFILE_CACHE % key)
    if cached is None and quick:
        refs = []
        cached = cache.get(PROFILE_CACHE % hashlib.sha256(json.dumps([PROFILE_VERSION, similar.MODEL, text, refs]).encode()).hexdigest())
        if not text:
            return None, []
    if cached is not None:
        rows, labels = np.frombuffer(cached[0], dtype=np.float32).reshape(len(cached[1]), -1), cached[1]
        if save and subscription.pk:
            subscription.profile_vector, subscription.profile_key = cached[0], key
            subscription.profile_labels = json.dumps(labels)
            subscription.save(update_fields=['profile_vector', 'profile_key', 'profile_labels'])
        return rows, labels
    try:
        parts = profile_parts(text)
        entries = reference_entries(refs)
        inputs = parts + [t for _, t, _ in entries]
        if not inputs:
            return None, []
        rows = similar.embed(inputs)
    except (ArxivUnavailable, requests.RequestException, KeyError, ValueError):
        logger.exception('Profile embedding failed for digest %s', subscription.pk)
        return None, []
    rows = (rows / np.linalg.norm(rows, axis=1, keepdims=True)).astype(np.float32)
    labels = ([{'kind': 'profile', 'text': short(p, 60)} for p in parts] +
              [{'kind': 'reference', 'id': ref, 'title': short(title)} for ref, _, title in entries])
    key = hashlib.sha256(json.dumps([PROFILE_VERSION, similar.MODEL, text, refs]).encode()).hexdigest()
    cache.set(PROFILE_CACHE % key, (rows.tobytes(), labels), 86400)
    if save and subscription.pk:
        subscription.profile_vector, subscription.profile_key = rows.tobytes(), key
        subscription.profile_labels = json.dumps(labels)
        subscription.save(update_fields=['profile_vector', 'profile_key', 'profile_labels'])
    return rows, labels


class Ranking:
    """How close a paper is to what its reader asked for: the closest part of the research profile, or the
    closest paper they liked or opened (a little less). A paper closer to one they disliked goes down."""

    def __init__(self, rows=None, labels=(), liked=(), liked_labels=(), liked_weights=(), disliked=()):
        self.rows, self.labels = rows, list(labels)
        self.liked = np.array(liked, dtype=np.float32) if len(liked) else None
        self.liked_labels, self.liked_weights = list(liked_labels), np.array(liked_weights, dtype=np.float32)
        self.disliked = np.array(disliked, dtype=np.float32) if len(disliked) else None

    @property
    def has_profile(self):
        return self.rows is not None

    def __bool__(self):
        return self.rows is not None or self.liked is not None

    def score(self, vector):
        """(score, label of what the paper is closest to)."""
        best, label = -1.0, None
        if self.rows is not None and self.rows.shape[1] == vector.shape[0]:
            sims = self.rows @ vector
            i = int(np.argmax(sims))
            best, label = float(sims[i]), self.labels[i]
        if self.liked is not None and self.liked.shape[1] == vector.shape[0]:
            sims = (self.liked @ vector) * self.liked_weights
            i = int(np.argmax(sims))
            if sims[i] > best:
                best, label = float(sims[i]), self.liked_labels[i]
        if self.disliked is not None and self.disliked.shape[1] == vector.shape[0]:
            worst = float(np.max(self.disliked @ vector))
            if worst > best:
                best -= DISLIKE_WEIGHT * (worst - best)
        return best, label


def ranking_for(subscription, paid, save=True, quick=False):
    """The Ranking of a digest: its research profile (Premium), and the papers its reader liked, disliked or opened."""
    rows, labels = profile_matrix(subscription, save, quick) if paid else (None, [])
    liked, liked_ids, weights, disliked = [], [], [], []
    feedback = DigestFeedback.objects.filter(subscription_id=subscription.pk) if subscription.pk else DigestFeedback.objects.none()
    for value, vector, source, arxiv_id in (feedback.exclude(vector=None).order_by('-created')
                                            .values_list('value', 'vector', 'source', 'arxiv_id')[:MAX_FEEDBACK]):
        vector = np.frombuffer(bytes(vector), dtype=np.float32)
        if value > 0:
            liked.append(vector)
            liked_ids.append((arxiv_id, source))
            weights.append(LIKE_WEIGHTS.get(source, LIKE_WEIGHTS['vote']))
        else:
            disliked.append(vector)
    if rows is None and not liked:
        return None
    titles = dict(DigestItem.objects.filter(arxiv_id__in=[a for a, _ in liked_ids]).values_list('arxiv_id', 'title'))
    liked_labels = [{'kind': 'liked' if source == 'vote' else 'opened', 'id': aid, 'title': short(titles.get(aid, aid))}
                    for aid, source in liked_ids]
    return Ranking(rows, labels, liked, liked_labels, weights, disliked)


def candidates(subscription, papers, already_sent, days=None, categories=None, exclude=()):
    wanted = set(categories or subscription.category_list())
    return [p for p in papers.values()
            if p['arxiv_id'] not in already_sent and p['arxiv_id'] not in exclude
            and wanted.intersection(p['categories']) and (days is None or p['announced'] in days)]


def rank_papers(subscription, papers, already_sent, days=None, categories=None, ranking=None, vectors=None, exclude=()):
    """Every paper of this digest, best first, each with why it is there:
    [{'paper', 'score', 'closest', 'keywords', 'primary'}].

    Keywords leave papers out ('-term') and, without a research profile, keep only the papers that use one
    of them; with a profile they raise the papers that use them instead. Papers are then ordered by
    closeness to the profile and to the papers liked or opened; without any, papers whose primary category
    is followed come first. The first ones are picked one by one, each a bit less likely when it is very
    close to one already picked, so that ten papers do not tell the same story.
    """
    keywords = Keywords(subscription.keywords)
    wanted = set(categories or subscription.category_list())
    vectors = vectors or {}
    profiled = ranking is not None and ranking.has_profile
    entries = []
    for paper in candidates(subscription, papers, already_sent, days, categories, exclude):
        if keywords.excluded(paper):
            continue
        matched, found = keywords.matches(paper)
        if keywords.include and not found and not profiled:
            continue
        score, closest = None, None
        vector = vectors.get(paper['arxiv_id'])
        if ranking and vector is not None:
            score, closest = ranking.score(vector)
            if profiled and found:
                score += KEYWORD_BOOST * min(2, len(found))
        entries.append({'paper': paper, 'score': score, 'closest': closest, 'keywords': found, 'matched': matched,
                        'primary': bool(paper['categories']) and paper['categories'][0] in wanted})
    if ranking:
        entries.sort(key=lambda e: (e['score'] is not None, e['score'] or 0, e['primary'], e['matched'],
                                    e['paper']['arxiv_id']), reverse=True)
    else:
        entries.sort(key=lambda e: (e['matched'], e['primary'], e['paper']['arxiv_id']), reverse=True)
    return diversify(entries, vectors, bool(ranking))


def diversify(entries, vectors, ranked):
    """The first DIVERSE_TOP entries picked one by one: by score minus DIVERSITY times the closeness to the
    papers already picked (ranked), or, without scores, among the best by keywords and primary category:
    first the paper closest to the heart of the batch, then each time the one least like those picked."""
    head = [e for e in entries[:DIVERSE_POOL] if e['paper']['arxiv_id'] in vectors]
    if len(head) < 3:
        return entries
    matrix = np.array([vectors[e['paper']['arxiv_id']] for e in head], dtype=np.float32)
    closest = np.zeros(len(head), dtype=np.float32)
    base = np.array([e['score'] or 0 for e in head], dtype=np.float32) if ranked else None
    tiers = [(e['matched'], e['primary']) for e in head]
    centrality = matrix @ (matrix.mean(axis=0))
    left, picked = list(range(len(head))), []
    while left and len(picked) < DIVERSE_TOP:
        if ranked:
            i = max(left, key=lambda j: base[j] - DIVERSITY * closest[j])
        else:
            top = max(tiers[j] for j in left)
            i = min((j for j in left if tiers[j] == top), key=lambda j: (closest[j], -centrality[j]))
        picked.append(i)
        left.remove(i)
        closest = np.maximum(closest, matrix @ matrix[i])
    chosen = [head[i] for i in picked]
    ids = {e['paper']['arxiv_id'] for e in chosen}
    return chosen + [e for e in entries if e['paper']['arxiv_id'] not in ids]


def select_papers(subscription, papers, already_sent, days=None, limit=PAPERS_PER_EMAIL, categories=None,
                  profile=None, vectors=None, exclude=()):
    """The best new papers for this digest (see rank_papers); profile is a Ranking."""
    return [e['paper'] for e in rank_papers(subscription, papers, already_sent, days, categories, profile, vectors,
                                            exclude)[:limit]]


def above_threshold(entries, volume):
    """The entries at least as close as the volume a Premium digest chose ('high', 'medium', 'low')."""
    floor = THRESHOLDS[volume]
    return [e for e in entries if e['score'] is not None and e['score'] >= floor]


def author_papers(subscription, index, already_sent, days):
    """(papers by the followed authors, in any category, newest first; {arxiv_id: [followed names on it]})."""
    found, via = {}, {}
    for name in subscription.author_list()[:MAX_AUTHORS]:
        key = name_key(name)
        for paper, author in index.get(key[0], []) if key else []:
            if paper['arxiv_id'] not in already_sent and paper['announced'] in days and same_author(key, author):
                found.setdefault(paper['arxiv_id'], paper)
                if name not in via.setdefault(paper['arxiv_id'], []):
                    via[paper['arxiv_id']].append(name)
    return sorted(found.values(), key=lambda p: p['published'], reverse=True), via


# --- shared content ------------------------------------------------------------

def write_blurbs(papers, lang):
    """Per paper, a single model call per batch of 10: {'summary': 2 plain-language sentences, 'points': what a
    researcher of the field wants (contribution, method, result)}."""
    language = 'French' if lang == 'fr' else 'English'
    listing = '\n\n'.join('ID: %s\nTitle: %s\nAbstract: %s' % (p['arxiv_id'], p['title'], p['abstract']) for p in papers)
    prompt = (
        'For each paper below, write in %s:\n'
        '- "summary": 2 sentences that a curious non-specialist can understand: what was done and why it matters. '
        'No jargon.\n'
        '- for a researcher of the field, who knows its vocabulary: "contribution": what is new, in one precise '
        'sentence; "method": the approach, data, sample or instrument, in one short sentence; "result": the main '
        'result, with its key number when the abstract gives one, in one short sentence.\n'
        'Use only what the abstract says: write "" for a part it does not give. No LaTeX, no hype.\n'
        'Answer with a JSON object {"summaries": [{"id": "...", "summary": "...", "contribution": "...", '
        '"method": "...", "result": "..."}]}.\n\n%s' % (language, listing))
    content = _chat(prompt, ai_models()[0], False, 4500)
    found = {}
    for item in json.loads(content).get('summaries', []):
        if isinstance(item, dict) and item.get('id') and item.get('summary'):
            found[item['id']] = {'summary': str(item['summary']).strip(),
                                 'points': {k: str(item.get(k) or '').strip() for k in POINTS}}
    return found


def blurb_parts(answer):
    """(summary, points JSON) from an answer of write_blurbs (a bare string is a summary alone)."""
    if isinstance(answer, str):
        return answer.strip(), '{}'
    return answer['summary'], json.dumps({k: v for k, v in answer.get('points', {}).items() if v})


def _chat(prompt, model, paid, max_tokens):
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
        json={'model': model, 'messages': [{'role': 'user', 'content': prompt}],
              'response_format': {'type': 'json_object'}, **chat_params(paid, max_tokens)},
        timeout=300,
    )
    response.raise_for_status()
    return usage.record(response.json())['choices'][0]['message']['content']


def ensure_items(papers, lang):
    """Return {arxiv_id: DigestItem}, generating the missing blurbs (and the points of items written before them)."""
    ids = [p['arxiv_id'] for p in papers]
    items = {i.arxiv_id: i for i in DigestItem.objects.filter(arxiv_id__in=ids, lang=lang)}
    missing = [p for p in papers if p['arxiv_id'] not in items or not items[p['arxiv_id']].points]
    for start in range(0, len(missing), 10):
        batch = missing[start:start + 10]
        try:
            blurbs = write_blurbs(batch, lang)
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
            logger.exception('Blurb generation failed, falling back to abstracts')
            blurbs = {}
        for paper in batch:
            answer = blurbs.get(paper['arxiv_id'])
            summary, points = blurb_parts(answer) if answer else ('', '')
            old = items.get(paper['arxiv_id'])
            if old is not None:
                if answer:  # an item written before the points: they are added once (or '{}' when the model had none)
                    old.points = points
                    old.save(update_fields=['points'])
                continue
            item = DigestItem(
                arxiv_id=paper['arxiv_id'], lang=lang, title=paper['title'],
                authors=', '.join(paper['authors'][:6]) + (' et al.' if len(paper['authors']) > 6 else ''),
                categories=','.join(paper['categories']), published=paper['published'],
                blurb=summary, points=points)
            if item.blurb:
                # stored once and reused by every subscriber; get_or_create in case two runs overlap
                item, _ = DigestItem.objects.get_or_create(arxiv_id=item.arxiv_id, lang=lang, defaults={
                    f: getattr(item, f) for f in ('title', 'authors', 'categories', 'published', 'blurb', 'points')})
            else:
                # not stored, so the next subscriber or run tries the model again
                item.blurb = paper['abstract'][:400].rsplit(' ', 1)[0] + '…'
            items[paper['arxiv_id']] = item
    return items


def summarized(arxiv_ids, lang):
    return set(SummaryPaper.objects.filter(paper__arxiv_id__in=arxiv_ids, lang=lang)
               .exclude(summary__isnull=True).exclude(summary='').exclude(summary__startswith='Error')
               .values_list('paper__arxiv_id', flat=True))


def presummarize(arxiv_ids, lang, budget):
    """Write the full summary of these papers now, so the links in the email open on a finished page.

    budget is a one-element list: how many summaries this run may still write.
    """
    from .consumers import summarize_now
    done = summarized(arxiv_ids, lang)
    for arxiv_id in arxiv_ids:
        if arxiv_id in done or budget[0] <= 0:
            continue
        budget[0] -= 1
        try:
            summarize_now(arxiv_id, lang)
        except Exception:
            logger.exception('Summary of %s for the digest failed', arxiv_id)


def notable_shortlist(category, lang, day, week):
    """The papers of the week the model may call notable, chosen by what can be counted rather than by titles:
    how many research profiles of the digests following the category they come close to, whether a rising
    topic of their archive cites them, and how many categories they were posted in."""
    from . import trends  # trends imports this module
    vectors = stored_vectors([p['arxiv_id'] for p in week])
    profiles = []
    for rows, labels in (DigestSubscription.objects.filter(active=True, categories__contains=category)
                         .exclude(profile_labels='').values_list('profile_vector', 'profile_labels')):
        count = len(json.loads(labels))
        if rows and count:
            profiles.append(np.frombuffer(bytes(rows), dtype=np.float32).reshape(count, -1))
    rising = set()
    for scope in (category, archive_of(category)):
        _, topics = trends.latest(scope, lang, since=day - 3 * WEEK)  # a week of submissions is written the Saturday after it
        rising |= {key for t in topics for key in t.get('ids') or [p.get('arxiv_id') for p in t.get('papers', [])]}
    scored = []
    for paper in week:
        vector = vectors.get(paper['arxiv_id'])
        interest = 0 if vector is None else sum(
            1 for rows in profiles if rows.shape[1] == vector.shape[0] and float(np.max(rows @ vector)) >= THRESHOLDS['medium'])
        scored.append(((interest, re.sub(r'v\d+$', '', paper['arxiv_id']) in rising, len(paper['categories'])), paper))
    scored.sort(key=lambda s: s[0], reverse=True)
    return [paper for _, paper in scored[:NOTABLE_SHORTLIST]]


def category_synthesis(category, lang, day, papers):
    """{'trends': [...], 'notable': [...]} for the last 7 days of a category, written once per language and day.
    Each trend cites the papers it is drawn from, and each notable paper says why, so a reader can check them."""
    existing = CategorySynthesis.objects.filter(category=category, lang=lang, day=day).first()
    if existing:
        return json.loads(existing.content)
    week = sorted((p for p in papers.values() if category in p['categories'] and p['announced'] > day - WEEK),
                  key=lambda p: p['published'], reverse=True)[:SYNTHESIS_MAX_PAPERS]
    if len(week) < SYNTHESIS_MIN_PAPERS:
        return None
    by_id = {p['arxiv_id']: p for p in week}
    shortlist = notable_shortlist(category, lang, day, week)
    # the 2-sentence summaries already written for subscribers say more than titles
    blurbs = dict(DigestItem.objects.filter(arxiv_id__in=list(by_id), lang=lang)
                  .values_list('arxiv_id', 'blurb')[:SYNTHESIS_MAX_BLURBS])
    language = 'French' if lang == 'fr' else 'English'
    prompt = (
        'Here are the %d papers announced on %s in the category %s over the last 7 days (ID | title, '
        'and a short summary for some of them), then a shortlist of papers with their abstracts. '
        'In %s, write for a researcher of this field: the 3 to 5 main trends or topics of the week, each with a '
        'short title, 2 sentences and the IDs of the 2 or 3 papers of the list it is drawn from; then pick from '
        'the shortlist only the 3 papers most likely to matter, each with one sentence saying why, from its abstract. '
        'Be concrete, no hype, no LaTeX, and never write paper IDs in the texts: they are read by people. '
        'Answer with a JSON object {"trends": [{"title": "...", "text": "...", "papers": ["ID", ...]}], '
        '"notable": [{"id": "ID", "why": "..."}]}.\n\nPapers:\n%s\n\nShortlist:\n%s'
        % (len(week), SERVER_NAMES.get(archive_of(category), 'arXiv'), category_name(category), language, '\n'.join(
            '%s | %s%s' % (p['arxiv_id'], p['title'], ' | ' + blurbs[p['arxiv_id']] if p['arxiv_id'] in blurbs else '')
            for p in week), '\n\n'.join('%s | %s\n%s' % (p['arxiv_id'], p['title'], short(p['abstract'], 700))
                                          for p in shortlist)))
    allowed = {p['arxiv_id'] for p in shortlist}
    try:
        answer = json.loads(_chat(prompt, ai_models()[1], True, 3500))
        content = {
            'category': category_name(category),
            'trends': [{'title': str(t['title']).strip(), 'text': str(t['text']).strip(),
                        'papers': [{'arxiv_id': aid, 'title': by_id[aid]['title']}
                                   for aid in dict.fromkeys(t.get('papers') or []) if isinstance(aid, str) and aid in by_id][:3]}
                       for t in answer.get('trends', [])[:5] if isinstance(t, dict) and t.get('title') and t.get('text')],
            'notable': [],
        }
        for pick in answer.get('notable', []):
            aid, why = (pick.get('id'), str(pick.get('why') or '').strip()) if isinstance(pick, dict) else (pick, '')
            if isinstance(aid, str) and aid in allowed and aid not in [n['arxiv_id'] for n in content['notable']]:
                content['notable'].append({'arxiv_id': aid, 'title': by_id[aid]['title'], 'why': why})
        content['notable'] = content['notable'][:3]
    except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
        logger.exception('Synthesis of %s failed', category)
        return None
    if not content['trends']:
        return None
    CategorySynthesis.objects.get_or_create(category=category, lang=lang, day=day, defaults={
        'content': json.dumps(content), 'papers': len(week)})
    return content


# --- sending -----------------------------------------------------------------

def unsubscribe_token(subscription):
    return signing.dumps(subscription.pk, salt=UNSUBSCRIBE_SALT)


def subscription_from_token(token):
    try:
        pk = signing.loads(token, salt=UNSUBSCRIBE_SALT)
    except signing.BadSignature:
        return None
    return DigestSubscription.objects.filter(pk=pk).first()


def feedback_token(subscription, arxiv_id, value):
    return signing.dumps([subscription.pk, arxiv_id, value], salt=FEEDBACK_SALT, compress=True)


def edition_token(edition):
    return signing.dumps([edition.subscription_id, edition.pk], salt=EDITION_SALT)


def open_token(subscription, edition, arxiv_id, rank, target):
    return signing.dumps([subscription.pk, edition.pk if edition else None, arxiv_id, rank, target],
                         salt=OPEN_SALT, compress=True)


def save_token(subscription, arxiv_id):
    return signing.dumps([subscription.pk, arxiv_id], salt=SAVE_SALT, compress=True)


def link_code(kind, edition, rank, extra=''):
    """Short signed code of a link of an email: what to do ('o' open, 'k' keep in a collection, 'v' vote),
    with which paper of which email. Signed tokens of the arXiv id made each email ~1 KB per paper heavier,
    and Gmail cuts emails past 102 KB, unsubscribe link included."""
    payload = '%s%d.%d%s' % (kind, edition.pk, rank, extra)
    return '%s.%s' % (payload, salted_hmac(LINK_SALT, payload).hexdigest()[:LINK_SIGNATURE])


def read_link_code(code):
    """(kind, edition id, rank, extra) of a valid code, or None."""
    match = LINK_RE.match(code or '')
    if not match or not constant_time_compare(salted_hmac(LINK_SALT, match.group(1)).hexdigest()[:LINK_SIGNATURE],
                                              match.group(6)):
        return None
    return match.group(2), int(match.group(3)), int(match.group(4)), match.group(5)


def tracked(url, campaign, content):
    """Link with the UTM tags Umami reports on, to see which parts of the digest get clicked."""
    return '%s%sutm_source=digest&utm_medium=email&utm_campaign=%s&utm_content=%s' % (
        url, '&' if '?' in url else '?', campaign, content)


def source_links(arxiv_id):
    """(identifier shown, abstract page, PDF or None) of a paper at its source."""
    if sources.is_external(arxiv_id):
        doi = sources.key_doi(arxiv_id) if arxiv_id.startswith('doi_') else ''
        return ('doi:%s' % doi if doi else arxiv_id), ('https://doi.org/%s' % doi if doi else None), None
    plain = arxiv_id.replace('--', '/')
    return 'arXiv:%s' % plain, 'https://arxiv.org/abs/%s' % plain, 'https://arxiv.org/pdf/%s' % plain


def category_label(code):
    """'astro-ph.EP' -> 'Earth and Planetary Astrophysics', 'biorxiv.cell-biology' -> 'Cell Biology', in the
    active language."""
    if code in arxiv_taxonomy.NAMES:
        return str(arxiv_taxonomy.NAMES[code])
    if code in journal_taxonomy.NAMES:
        return str(journal_taxonomy.NAMES[code])
    return translation.gettext(category_name(code)) if code in BIO_CODES else code


def reasons(entry, wanted=(), private=True):
    """Why a paper is in a digest, in the active language: ['Close to your reference paper “…”', 'Keywords: …'].
    private=False leaves out what only its reader should read (their profile, the papers they liked or opened):
    for a page anyone with its link can open."""
    from django.utils.translation import gettext as _
    if not entry:
        return []
    found, closest = [], entry.get('closest') or {}
    kind = closest.get('kind')
    if kind == 'reference':
        found.append(_('Close to your reference paper “%(title)s”') % {'title': closest.get('title')})
    elif kind == 'profile':
        found.append(_('Close to your research profile: “%(text)s”') % {'text': closest.get('text')} if private
                     else _('Close to your research profile'))
    elif kind == 'liked':
        found.append(_('Close to a paper you liked: “%(title)s”') % {'title': closest.get('title')} if private
                     else _('Close to a paper you liked'))
    elif kind == 'opened':
        found.append(_('Close to a paper you opened: “%(title)s”') % {'title': closest.get('title')} if private
                     else _('Close to a paper you opened'))
    if entry.get('keywords'):
        found.append(_('Keywords: %(terms)s') % {'terms': ', '.join(entry['keywords'])})
    categories = entry['paper']['categories']
    if categories and not entry.get('primary'):
        followed = [c for c in categories if c in set(wanted)]
        found.append(_('Cross-listed in %(category)s from %(primary)s') % {
            'category': followed[0] if followed else '', 'primary': categories[0]})
    return found


def send_one(subscription, items, base_url, from_authors=(), syntheses=(), ready=(), now=None, paid=False, via=None,
             teaser=None, listen_url=None, rising=(), edition=None, entries=None, sections=(), next_door=()):
    """entries: {arxiv_id: entry of rank_papers}, to say why each paper is there; sections: alerts sent with it;
    next_door: weekly letters of other fields close to the user's research profile."""
    lang = subscription.lang
    campaign = 'weekly' if subscription.frequency == 'weekly' else 'daily'
    unsubscribe_url = '%s/%s/digest/unsubscribe/%s/' % (base_url, lang, unsubscribe_token(subscription))
    ready = set(ready)
    entries = entries or {}
    wanted = subscription.category_list()

    def opened(item, rank, target):
        if edition is not None:
            return '%s/%s/digest/l/%s/' % (base_url, lang, link_code('o', edition, rank, target[0]))
        return '%s/%s/digest/open/%s/' % (base_url, lang, open_token(subscription, edition, item.arxiv_id, rank, target))

    with translation.override(lang):
        for rank, item in enumerate(list(from_authors) + list(items), 1):
            item.rank = rank
            item.url = opened(item, rank, 'summary')
            item.ident, abs_url, pdf_url = source_links(item.arxiv_id)
            item.source = 'DOI' if sources.is_external(item.arxiv_id) else 'arXiv'
            item.abs_url = opened(item, rank, 'abs') if abs_url else None
            item.pdf_url = opened(item, rank, 'pdf') if pdf_url else None
            item.save_url = ('%s/%s/digest/l/%s/' % (base_url, lang, link_code('k', edition, rank)) if edition else
                             '%s/%s/digest/save/%s/' % (base_url, lang, save_token(subscription, item.arxiv_id)))
            item.ready = item.arxiv_id in ready
            item.via = ', '.join((via or {}).get(item.arxiv_id, []))
            categories = [c for c in item.categories.split(',') if c]
            item.category = categories[0] if categories else ''
            item.reasons = reasons(entries.get(item.arxiv_id), wanted)
            points = json.loads(item.points or '{}') if subscription.level == 'expert' else {}
            item.point_list = [(label, points[key]) for key, label in (
                ('contribution', translation.gettext('Contribution')), ('method', translation.gettext('Method')),
                ('result', translation.gettext('Result'))) if points.get(key)]
            if edition is not None:
                item.more_url, item.less_url = ('%s/%s/digest/l/%s/' % (base_url, lang, link_code('v', edition, rank, v))
                                                for v in 'ud')
            else:
                item.more_url, item.less_url = ('%s/%s/digest/feedback/%s/' % (base_url, lang, feedback_token(subscription, item.arxiv_id, v))
                                                for v in (1, -1))
        link = lambda aid: tracked('%s/%s/paper/%s/?autorun=1' % (base_url, lang, aid), campaign, 'synthesis')
        syntheses = [dict(synthesis, notable=[dict(paper, url=link(paper['arxiv_id'])) for paper in synthesis['notable']],
                          trends=[dict(trend, papers=[dict(paper, url=link(paper['arxiv_id'])) for paper in trend.get('papers', [])])
                                  for trend in synthesis['trends']])
                     for synthesis in syntheses]
        first = (list(from_authors) + list(items) or [None])[0]
        context = {
            'subscription': subscription,
            'items': items,
            'from_authors': from_authors,
            'count': len(items) + len(from_authors),
            'top_title': short(first.title, 70) if first else '',
            'top_author': first.via.split(', ')[0] if first is not None and from_authors and first.via else '',
            'categories_label': ', '.join(category_label(c) for c in wanted),
            'total': edition.candidates if edition else 0,
            'edition_url': tracked('%s/%s/digest/e/%s/' % (base_url, lang, edition_token(edition)), campaign, 'edition')
            if edition else None,
            'sections': sections,
            'syntheses': syntheses,
            'paid': paid,
            'teaser': teaser,
            'listen_url': listen_url,
            'rising': rising,
            'trends_url': tracked('%s/%s/trends/?view=foryou' % (base_url, lang), campaign, 'rising'),
            'next_door': [{'title': letter.title, 'dek': letter.dek, 'url': tracked('%s/%s/letters/%s/%s/' % (
                base_url, lang, letter.topic, letter.day.isoformat()), campaign, 'letter')} for letter in next_door],
            'letters_url': tracked('%s/%s/letters/' % (base_url, lang), campaign, 'letters'),
            'premium_url': tracked('%s/%s/premium/' % (base_url, lang), campaign, 'premium-teaser'),
            'weekly': campaign == 'weekly',
            'base_url': base_url,
            'manage_url': '%s/%s/digest/?d=%s' % (base_url, lang, subscription.pk),
            'unsubscribe_url': unsubscribe_url,
            'date': now or timezone.now(),
        }
        context['more'] = max(0, context['total'] - context['count'])
        subject = render_to_string('digest/subject.txt', context).strip()
        text = render_to_string('digest/email.txt', context)
        html = render_to_string('digest/email.html', context)
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [subscription.user.email],
                                     headers={'List-Unsubscribe': '<%s>' % unsubscribe_url,
                                              'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click'})
    message.attach_alternative(html, 'text/html')
    message.send()


def premium_teaser(plan):
    """For a free digest: the first trend of the week in one of its categories, if a Premium digest already paid for it."""
    for synthesis in CategorySynthesis.objects.filter(category__in=plan.categories, lang=plan.subscription.lang, day=plan.today):
        trends = json.loads(synthesis.content).get('trends')
        if trends:
            return {'category': category_name(synthesis.category), 'title': trends[0]['title']}
    return {}


class Plan:
    """What one due digest needs: its limits, its Premium extras, and which arXiv announcements it covers."""

    def __init__(self, subscription, paid, now):
        self.subscription = subscription
        self.paid = paid
        self.limits = limits(paid)
        self.categories = subscription.category_list()[:self.limits['categories']]
        self.weekly = subscription.frequency == 'weekly'
        try:
            local = now.astimezone(zoneinfo.ZoneInfo(subscription.timezone))
        except Exception:
            local = now
        self.today = expected_batch(now)
        # Premium: what happened this week, in the weekly digest or in Monday's daily digest
        self.synthesis = paid and (self.weekly or local.weekday() == 0)
        self.teaser = not paid and (self.weekly or local.weekday() == 0)
        self.authors = subscription.author_list()[:MAX_AUTHORS] if paid else []

    def batches(self, days):
        """The stored announcements not sent yet: the last one for a daily digest, the week's for a weekly one."""
        recent = [d for d in days if d > self.today - WEEK]
        last = self.subscription.last_batch
        new = [d for d in recent if last is None or d > last]
        if last is None and not self.weekly:
            new = new[-1:]
        return set(new)


def due_plans(now, only_user=None):
    subscriptions = DigestSubscription.objects.filter(active=True).select_related('user').exclude(user__email='')
    if only_user:
        subscriptions = list(subscriptions.filter(user__username=only_user))
    else:
        suppressed = mailcheck.suppressed()
        subscriptions = [s for s in subscriptions if s.is_due(now) and s.user.email.strip().lower() not in suppressed]
    paid, first = {}, {}
    for s in subscriptions:
        if s.user_id not in paid:
            paid[s.user_id] = access.has_tool(s.user, 'digest')
            first[s.user_id] = DigestSubscription.objects.filter(user_id=s.user_id).values_list('pk', flat=True).first()
    # without Premium (or after it ended) only the first digest of an account is sent
    plans = [Plan(s, paid[s.user_id], now) for s in subscriptions if paid[s.user_id] or s.pk == first[s.user_id]]
    if not only_user:
        # Premium first, then the ones about to miss their slot, then longest-waiting, and no more than the mail
        # provider allows per hour (a digest left out still has its web page, see DigestEdition)
        plans.sort(key=lambda p: (not p.paid, -p.subscription.hours_late(now), p.subscription.last_sent or p.subscription.created))
        if len(plans) > MAX_EMAILS_PER_RUN:
            logger.warning('%d digests due, sending %d this hour', len(plans), MAX_EMAILS_PER_RUN)
            plans = plans[:MAX_EMAILS_PER_RUN]
    return plans


def upcoming_plans(now, current):
    """Premium digests due next hour, whose top papers can be summarized before their email."""
    later = now + datetime.timedelta(hours=1)
    current = {plan.subscription.pk for plan in current}
    return [plan for plan in due_plans(later)
            if plan.paid and plan.subscription.pk not in current and not plan.subscription.is_due(now)]


def choose(plan, days, papers, index):
    """(papers by followed authors, other papers, {arxiv_id: followed names}, every ranked entry) for one digest."""
    subscription = plan.subscription
    already = set(DigestDelivery.objects.filter(subscription=subscription).values_list('arxiv_id', flat=True))
    from_authors, via = author_papers(subscription, index, already, days) if plan.authors else ([], {})
    from_authors = from_authors[:plan.limits['papers']]
    ranking = ranking_for(subscription, plan.paid)
    vectors = ensure_vectors(candidates(subscription, papers, already, days, plan.categories))
    entries = rank_papers(subscription, papers, already, days, plan.categories, ranking, vectors,
                          exclude={p['arxiv_id'] for p in from_authors})
    if plan.paid and subscription.volume in THRESHOLDS and ranking is not None and ranking.has_profile:
        chosen = above_threshold(entries, subscription.volume)[:THRESHOLD_MAX_EMAIL - len(from_authors)]
    else:
        chosen = entries[:plan.limits['papers'] - len(from_authors)]
    return from_authors, [e['paper'] for e in chosen], via, entries


def run(base_url=BASE_URL, dry_run=False, only_user=None, stdout=None, now=None):
    """Send the digests that are due (meant to run every hour); --user ignores the schedule."""
    now = now or timezone.now()
    if only_user:
        return _run(base_url, dry_run, only_user, stdout, now)
    # a run slowed down by arXiv must not overlap the next one, or the same digests would go out twice
    if not cache.add(LOCK_KEY, now.isoformat(), LOCK_SECONDS):
        logger.warning('The previous digest run is still going, skipping this one')
        return 0
    try:
        return _run(base_url, dry_run, only_user, stdout, now)
    finally:
        cache.delete(LOCK_KEY)
        if not dry_run:
            check_health(now)


def check_health(now):
    """Log an error (an alert in Sentry), once a day, when the digests have silently stopped."""
    local = now.astimezone(ANNOUNCE_TZ)
    if local.weekday() >= 5:
        return []
    problems = []
    expected = expected_batch(now)
    batch = ArxivBatch.objects.filter(day=expected).first()
    stored = batch is not None
    if expected == local.date() and local.hour >= 6 and not stored and not cache.get(QUIET_KEY % expected):
        problems.append(('batch', 'The arXiv announcement of %s is still not stored at %s in New York'
                         % (expected, local.strftime('%H:%M'))))
    old_enough = DigestSubscription.objects.filter(active=True, created__lt=now - datetime.timedelta(days=2))
    if stored and local.weekday() in (1, 2, 3, 4) and now.hour >= 12 and old_enough.exists():
        last = DigestSubscription.objects.exclude(last_sent=None).order_by('-last_sent').values_list('last_sent', flat=True).first()
        if last is None or last < now - datetime.timedelta(hours=30):
            problems.append(('sent', 'No digest was sent since %s' % last))
    bio_missing = [SERVER_NAMES[a] for a in batch.missing_list() if a in OTHER_ARCHIVES] if batch else []
    if bio_missing and expected == local.date() and local.hour >= 9 and DigestSubscription.objects.filter(
            active=True).filter(models.Q(categories__contains='biorxiv.') | models.Q(categories__contains='medrxiv.') |
                                models.Q(categories__contains=JOURNALS + '.')).exists():
        problems.append(('bio', '%s still missing from the announcement of %s at %s in New York'
                         % (' and '.join(bio_missing), expected, local.strftime('%H:%M'))))
    for name, message in problems:
        if cache.add(HEALTH_KEY % (name, now.date()), True, 86400):
            logger.error('Digest health: %s', message)
    return [message for _, message in problems]


def _run(base_url, dry_run, only_user, stdout, now):
    plans = due_plans(now, only_user)
    upcoming = [] if (only_user or dry_run) else upcoming_plans(now, plans)
    if not plans and not upcoming:
        return 0

    days = ensure_batch(now, {c for plan in plans + upcoming for c in plan.categories})
    papers = load_papers(days)
    index = author_index(papers) if any(plan.authors for plan in plans + upcoming) else {}
    incomplete = {b.day: set(b.missing_list()) for b in ArxivBatch.objects.filter(day__in=days).exclude(missing='')}

    budget = [MAX_PRESUMMARIES_PER_RUN]
    audio_budget = [audio.MAX_NEW_PER_RUN]
    sent = 0
    for plan in plans:
        batches = plan.batches(days)
        if not batches:
            continue  # no new announcement yet: tried again next hour
        if incomplete.get(max(batches), set()).intersection(archive_of(c) for c in plan.categories):
            continue  # today's batch still lacks one of its archives: tried again next hour
        sent += deliver(plan, batches, papers, index, base_url, now, budget, stdout=stdout, dry_run=dry_run,
                        audio_budget=audio_budget)

    # with what is left of the budget, get next hour's Premium emails ready
    for plan in upcoming:
        if budget[0] <= 0:
            break
        batches = plan.batches(days)
        if batches:
            from_authors, chosen, _, _ = choose(plan, batches, papers, index)
            presummarize([p['arxiv_id'] for p in from_authors + chosen][:PRESUMMARIZED], plan.subscription.lang, budget)
    return sent


def deliver(plan, batches, papers, index, base_url, now, budget, stdout=None, dry_run=False, new_syntheses=True,
            audio_budget=None):
    """Choose, write and send one digest from these announcements; True if it was sent."""
    from . import digest_bundle
    subscription = plan.subscription
    if not cache.add(SENDING_KEY % subscription.pk, True, 600):
        return False  # the same digest is being sent by a preview or another run
    try:
        from_authors, chosen, via, entries = choose(plan, batches, papers, index)
        bundled = plan.paid and subscription.bundle_alerts and not dry_run
        sections = digest_bundle.collect(subscription.user, subscription.lang, base_url, now) if bundled else []
        if not chosen and not from_authors and not sections:
            return False
        items = ensure_items(from_authors + chosen, subscription.lang)
        author_items = [items[p['arxiv_id']] for p in from_authors]
        ordered = [items[p['arxiv_id']] for p in chosen]
        if stdout:
            stdout.write('%s: %d papers from %s%s' % (subscription.user.username, len(author_items) + len(ordered),
                                                       ', '.join(str(d) for d in sorted(batches)),
                                                       ' (Premium)' if plan.paid else ''))
        if dry_run:
            return False
        syntheses = []
        if plan.synthesis:
            for category in plan.categories[:MAX_SYNTHESES_PER_EMAIL]:
                if new_syntheses:
                    content = category_synthesis(category, subscription.lang, plan.today, papers)
                else:
                    row = CategorySynthesis.objects.filter(category=category, lang=subscription.lang, day=plan.today).first()
                    content = json.loads(row.content) if row else None
                if content:
                    syntheses.append(content)
        all_ids = [i.arxiv_id for i in author_items + ordered]
        listen_url, rising, next_door = None, [], []
        if plan.synthesis or plan.teaser:
            from . import letters  # letters reads the thresholds of this module
            next_door = letters.for_digest(subscription.user, plan.categories, subscription.lang, plan.today)
        if plan.paid:
            presummarize(all_ids[:PRESUMMARIZED], subscription.lang, budget)
            listen_url = listen_link(subscription, author_items + ordered, base_url, audio_budget or [0])
            if plan.synthesis:
                from . import trends  # trends reads the archive list of this module
                rising = trends.for_digest(plan.categories, subscription.lang, plan.today)
        scores = {e['paper']['arxiv_id']: e['score'] for e in entries}
        edition = DigestEdition.objects.create(
            subscription=subscription, sent=now, days=','.join(str(d) for d in sorted(batches)),
            papers=json.dumps(all_ids), candidates=len(entries) + len(author_items),
            scores=json.dumps([None if scores.get(aid) is None else round(scores[aid], 4) for aid in all_ids]))
        try:
            send_one(subscription, ordered, base_url, author_items, syntheses, summarized(all_ids, subscription.lang), now,
                     paid=plan.paid, via=via, teaser=premium_teaser(plan) if plan.teaser else None,
                     listen_url=listen_url, rising=rising, edition=edition, next_door=next_door,
                     entries={e['paper']['arxiv_id']: e for e in entries}, sections=sections)
        except Exception:
            logger.exception('Digest email to %s failed', subscription.user.username)
            edition.delete()
            return False
        for section in sections:
            section['commit']()
        DigestDelivery.objects.bulk_create(
            [DigestDelivery(subscription=subscription, arxiv_id=aid) for aid in all_ids], ignore_conflicts=True)
        subscription.last_sent = now
        subscription.last_batch = max([max(batches)] + ([subscription.last_batch] if subscription.last_batch else []))
        subscription.save(update_fields=['last_sent', 'last_batch'])
        return True
    finally:
        cache.delete(SENDING_KEY % subscription.pk)


def listen_link(subscription, items, base_url, audio_budget):
    """Link to this digest read aloud, once its clips exist (new clips are made within audio_budget)."""
    try:
        audio.ensure(items, subscription.lang, audio_budget)
    except Exception:
        logger.exception('Audio of the digest failed')
    ids = [i.arxiv_id for i in items[:audio.MAX_CLIPS]]
    have = set(PaperAudio.objects.filter(arxiv_id__in=ids, lang=subscription.lang).values_list('arxiv_id', flat=True))
    ids = [a for a in ids if a in have]
    if not ids:
        return None
    return tracked('%s/%s/digest/listen/%s/' % (base_url, subscription.lang, audio.token(subscription, ids)),
                   'weekly' if subscription.frequency == 'weekly' else 'daily', 'listen')


def preview(subscription, base_url=BASE_URL):
    """Send a digest right away with the latest stored announcement, from the digest page.

    Nothing slow happens here: no arXiv call, no full summary, no new synthesis.
    """
    days = list(ArxivBatch.objects.order_by('-day').values_list('day', flat=True)[:1])
    if not days:
        return False
    plan = Plan(subscription, access.has_tool(subscription.user, 'digest'), timezone.now())
    papers = load_papers(days)
    index = author_index(papers) if plan.authors else {}
    return deliver(plan, set(days), papers, index, base_url, timezone.now(), [0], new_syntheses=False)


def start_preview(subscription_pk):
    """Run preview() in a thread, so the page answers at once (blurbs take a few seconds to write)."""
    import threading
    from django.db import connections

    def work():
        try:
            subscription = DigestSubscription.objects.select_related('user').get(pk=subscription_pk)
            preview(subscription)
        except Exception:
            logger.exception('Digest preview %s failed', subscription_pk)
        finally:
            connections.close_all()
    threading.Thread(target=work, daemon=True).start()


def cleanup(days=30):
    """Forget the papers and embeddings too old to appear in a digest or on its web page again."""
    AnnouncedPaper.objects.filter(announced__lt=timezone.now().date() - datetime.timedelta(days=PAGE_DAYS)).delete()
    DigestEdition.objects.filter(sent__lt=timezone.now() - datetime.timedelta(days=EDITION_DAYS)).delete()
    return DigestVector.objects.filter(created__lt=timezone.now() - timezone.timedelta(days=days)).delete()[0]


# --- views -------------------------------------------------------------------

from django.http import Http404  # noqa: E402
from django.shortcuts import redirect, render  # noqa: E402
from django.utils.translation import get_language  # noqa: E402
from django.views.decorators.csrf import csrf_exempt  # noqa: E402


def digest_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/digest.html', _digest_context(request, None, [], False))
    paid = access.has_tool(request.user, 'digest')
    lim = limits(paid)
    digests = list(request.user.digests.all())
    wanted = request.POST.get('digest') if request.method == 'POST' else request.GET.get('d')
    if wanted == 'new' or (not wanted and not digests):
        subscription = None
    elif wanted:
        subscription = next((d for d in digests if str(d.pk) == wanted), None)
        if subscription is None:
            raise Http404
    else:
        subscription = digests[0]
    can_create = len(digests) < lim['digests']
    if subscription is None and not can_create:
        return redirect('/%s/digest/' % get_language())

    if request.method == 'POST':
        if request.POST.get('action') == 'delete' and subscription is not None:
            subscription.delete()
            return redirect('/%s/digest/' % get_language())
        if request.POST.get('action') == 'preview' and subscription is not None and subscription.active:
            hour = 'digest-previews:%s' % timezone.now().strftime('%Y%m%d%H')
            cache.add(hour, 0, 3600)
            if not cache.add(PREVIEW_KEY % (subscription.pk, timezone.now().date()), True, 86400):
                state = 'again'
            elif cache.incr(hour) > PREVIEWS_PER_HOUR:
                cache.delete(PREVIEW_KEY % (subscription.pk, timezone.now().date()))
                state = 'busy'
            else:
                start_preview(subscription.pk)
                state = 'sent'
            return redirect('/%s/digest/?d=%s&preview=%s' % (get_language(), subscription.pk, state))
        if request.POST.get('action') == 'import':
            from . import digest_web
            draft = digest_web.preview_subscription(request, subscription, paid)
            draft.pk = subscription.pk if subscription else None
            draft.name = request.POST.get('name', '').strip()[:80]
            draft.authors = ', '.join(clean_authors(request.POST.get('authors', ''))) if paid else ''
            for field in ('send_hour', 'weekday'):
                setattr(draft, field, _int(request.POST.get(field), 7 if field == 'send_hour' else 0, 0, 23 if field == 'send_hour' else 4))
            draft.timezone = request.POST['timezone'] if request.POST.get('timezone') in TIMEZONES else 'Europe/Paris'
            draft.bundle_alerts = paid and request.POST.get('bundle_alerts') == '1'
            found = digest_web.import_publications(request.POST.get('identity', ''))
            if 'error' not in found:
                digest_web.apply_import(draft, found, paid)
            return render(request, 'summarizer/digest.html', dict(_digest_context(request, draft, digests, paid), imported=found,
                                                                  identity=request.POST.get('identity', '')))
        categories = clean_categories(request.POST.getlist('categories') + request.POST.get('other', '').split(','),
                                      lim['categories'])
        if not categories:
            return render(request, 'summarizer/digest.html', _digest_context(request, subscription, digests, paid, error=True))
        if subscription is None:
            subscription = DigestSubscription(user=request.user)
        subscription.categories = ','.join(categories)
        subscription.keywords = ', '.join(k.strip() for k in request.POST.get('keywords', '').split(',') if k.strip())[:500]
        subscription.lang = request.POST.get('lang') if request.POST.get('lang') in ('en', 'fr') else get_language()
        subscription.frequency = 'weekly' if request.POST.get('frequency') == 'weekly' else 'daily'
        subscription.weekday = _int(request.POST.get('weekday'), 0, 0, 4)
        subscription.send_hour = _int(request.POST.get('send_hour'), 7, 0, 23)
        if request.POST.get('timezone') in TIMEZONES:
            subscription.timezone = request.POST['timezone']
        subscription.level = 'general' if request.POST.get('level') == 'general' else 'expert'
        if paid:  # kept as they are when Premium has ended, and used again if it comes back
            subscription.volume = request.POST.get('volume') if request.POST.get('volume') in dict(DigestSubscription.VOLUMES) else 'top'
            subscription.bundle_alerts = request.POST.get('bundle_alerts') == '1'
            subscription.name = request.POST.get('name', '').strip()[:80]
            subscription.profile = request.POST.get('profile', '').strip()[:2000]
            subscription.reference_papers = ', '.join(clean_references(request.POST.get('reference_papers', '')))
            subscription.authors = ', '.join(clean_authors(request.POST.get('authors', '')))
        subscription.active = True
        subscription.save()
        request.session['digest_configured'] = subscription.pk
        return redirect('/%s/digest/?d=%s&saved=1' % (get_language(), subscription.pk))
    return render(request, 'summarizer/digest.html', _digest_context(request, subscription, digests, paid))


def _int(value, default, low, high):
    try:
        return min(high, max(low, int(value)))
    except (TypeError, ValueError):
        return default


def hour_label(hour):
    """'7:00 AM' in English, '07:00' in French."""
    if get_language() == 'fr':
        return '%02d:00' % hour
    return '%d:00 %s' % (hour % 12 or 12, 'AM' if hour < 12 else 'PM')


def taxonomy_choices(taxonomy=arxiv_taxonomy):
    """[(group, [(code, name in the active language, English name)])]: a category is found by either name."""
    with translation.override('en'):
        english = {code: str(name) for code, name in taxonomy.NAMES.items()}
    return [(str(group), [(code, str(name), english[code]) for code, name in codes]) for group, codes in taxonomy.GROUPS]


def _digest_context(request, subscription, digests, paid, error=False):
    lim = limits(paid)
    chosen = subscription.category_list() if subscription else []
    known = set(arxiv_taxonomy.NAMES) | set(BIO_CODES) | set(journal_taxonomy.NAMES)
    configured = bool(not error and subscription and subscription.pk and
                      request.session.get('digest_configured') == subscription.pk)
    if configured:
        request.session.pop('digest_configured')
    return {
        'digest_configured': configured,
        'subscription': subscription,
        'digests': digests,
        'paid': paid,
        'can_create': len(digests) < lim['digests'],
        'inactive_digests': [] if paid else digests[1:],
        'taxonomy': taxonomy_choices(),
        'journal_taxonomy': taxonomy_choices(journal_taxonomy),
        'digest_tabs': [(d, d.name or ', '.join(category_label(c) for c in d.category_list()[:2]) +
                         ('…' if len(d.category_list()) > 2 else '')) for d in digests],
        'chosen_labels': [(c, category_label(c)) for c in chosen],
        'editions': [(e, edition_token(e)) for e in DigestEdition.objects.filter(subscription=subscription)[:8]]
        if subscription and subscription.pk else [],
        'volumes': DigestSubscription.VOLUMES,
        'thresholds': THRESHOLDS,
        'bio_categories': [(SERVER_NAMES[server], [(sources.category_code(server, n), translation.gettext(n), n) for n in names])
                           for server, names in BIO_CATEGORIES],
        'chosen': chosen,
        'other': ', '.join(c for c in chosen if c not in known),
        'max_categories': lim['categories'],
        'papers_per_email': lim['papers'],
        'premium': PREMIUM,
        'max_authors': MAX_AUTHORS,
        'max_references': MAX_REFERENCES,
        'saved': request.GET.get('saved') == '1',
        'preview': request.GET.get('preview'),
        'hours': [(h, hour_label(h)) for h in range(24)],
        'weekdays': range(5),
        'timezones': sorted(TIMEZONES),
        'send_hour': subscription.send_hour if subscription else 7,
        'timezone': subscription.timezone if subscription else '',
        'error': error,
    }


def paper_vector(arxiv_id):
    row = DigestVector.objects.filter(arxiv_id=arxiv_id).values_list('vector', flat=True).first()
    if row is not None:
        return bytes(row)
    item = DigestItem.objects.filter(arxiv_id=arxiv_id).first()
    if item is None:
        return None
    try:
        return similar.embed(['%s\n\n%s' % (item.title, item.blurb)])[0].tobytes()
    except (requests.RequestException, KeyError, ValueError):
        logger.exception('Embedding for feedback on %s failed', arxiv_id)
        return None


@csrf_exempt  # the link comes from an email; the signed token is the authorization
def feedback(request, token):
    """"More like this" / "less like this" from a Premium email.

    The link opens a page that records the choice with a POST sent by the page itself,
    so the link checkers of mail providers, which only follow GET links, do not vote.
    """
    try:
        pk, arxiv_id, value = signing.loads(token, salt=FEEDBACK_SALT)
    except (signing.BadSignature, ValueError, TypeError):
        pk = arxiv_id = value = None
    subscription = DigestSubscription.objects.filter(pk=pk).first() if pk else None
    return feedback_view(request, subscription, arxiv_id, value)


def feedback_view(request, subscription, arxiv_id, value):
    item = DigestItem.objects.filter(arxiv_id=arxiv_id).first() if subscription else None
    done = False
    if subscription is not None and request.method == 'POST' and value in (1, -1):
        DigestFeedback.objects.update_or_create(subscription=subscription, arxiv_id=arxiv_id,
                                                defaults={'value': value, 'source': 'vote', 'vector': paper_vector(arxiv_id)})
        done = True
    return render(request, 'summarizer/digest_feedback.html', {
        'subscription': subscription, 'item': item, 'value': value, 'done': done})


@csrf_exempt  # mail clients send the one-click unsubscribe POST without a CSRF token
def unsubscribe(request, token):
    subscription = subscription_from_token(token)
    done = False
    if subscription is not None and request.method == 'POST':
        subscription.active = False
        subscription.save(update_fields=['active'])
        done = True
    return render(request, 'summarizer/digest_unsubscribe.html', {'subscription': subscription, 'done': done})
