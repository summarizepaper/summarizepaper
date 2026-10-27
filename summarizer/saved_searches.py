"""Private saved searches and their watches: the new papers matching a search, by email, in the digest, on the
page of the search and in its RSS feed, using the shared hourly budget.

A watch reads every new paper of the daily announcements the digests keep (AnnouncedPaper: all of arXiv, bioRxiv
and medRxiv, and the followed journal subfields), so a broad search loses nothing to a window of latest results.
A paper is kept when its title or abstract has every word of the search (a plural counts, "quoted words" match as
written, -word leaves a paper out, OR separates alternatives). What the announcements cannot answer (journals,
filters on journals, ORCID or publication type, arXiv's own query syntax) is read from OpenAlex or arXiv's newest
results over a window wide enough for late indexing; a check that could not read its whole window says so.

The new papers of a check are ranked by closeness to the search and to the papers its user marked relevant or
not, the first ones get two sentences of summary, and all of them stay on the page of the search.
"""
import datetime
import hashlib
import json
import logging
import os
import re
import secrets
import time
import unicodedata
from urllib.parse import urlencode
from xml.etree import ElementTree

import numpy as np
import requests
from django.conf import settings
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core import signing
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.db import transaction
from django.db.models import Q, F
from django.http import Http404, HttpResponse, HttpResponseBadRequest, QueryDict
from django.shortcuts import get_object_or_404, redirect, render
from django.template.loader import render_to_string
from django.urls import reverse
from django.utils import feedgenerator, timezone, translation
from django.utils.translation import get_language, gettext as _
from django.views.decorators.http import require_POST
from django.views.decorators.cache import never_cache
from django.views.decorators.csrf import csrf_exempt

from . import access, digest, mailcheck, search_filters, sources, unified_search
from .models import AnnouncedPaper, DigestItem, SavedSearch, SavedSearchFeedback

logger = logging.getLogger(__name__)
STATE_SALT = 'save-research-search'
UNSUBSCRIBE_SALT = 'pause-research-watch'
MAX_SAVED = 50
MAX_CHECKS_PER_RUN = 10  # checks that ask arXiv or OpenAlex
RUN_SECONDS = 600  # checks over the announcements alone cost a query each: as many as fit in this time
MAX_EMAIL_ITEMS = 10  # ranked first in the email, with their summary; the others are on the page
MAX_HISTORY = 200  # new papers kept on the page of a search
MAX_SEEN = 10000
POOL_DAYS = digest.PAGE_DAYS  # announcements kept by the digests
LOOKBACK = datetime.timedelta(days=3)  # an announcement completed late is read again
API_LAG = datetime.timedelta(days=14)  # OpenAlex has a journal paper days or weeks after its publication date
API_MAX_DAYS = 30
API_PAGES = 4
API_PER_PAGE = 50
ARXIV_PAGES = 4  # of 25, for a search in arXiv's own syntax
PREVIEW_SHOWN = 15
RANKED = 300
LIKE_WEIGHT, DISLIKE_WEIGHT = 0.5, 0.5
# `mute` sets a new paper aside when it is much closer to a paper marked not relevant than the watch's papers
# usually are to it: how close papers are varies by field (median 0.43 for language models, 0.57 for galaxy
# clusters, measured on a week of arXiv), the same subtopic is about 0.12 above it
MUTE_MARGIN = 0.12
MUTE_FLOOR = 0.6
MUTE_FALLBACK = 0.68  # with fewer than MUTE_SAMPLE papers to measure the usual closeness
MUTE_SAMPLE = 5
VOTE_SALT = 'saved-search-vote'
STOPWORDS = frozenset('a an and the of in on for to with by from at as is are its into via using based '
                      'de des du la le les et en un une pour sur dans par aux au'.split())
ARXIV_SYNTAX = re.compile(r'\b(?:all|au|ti|abs|cat|co|jr|id|rn):|\bANDNOT\b|[()]')
FEED_ITEMS = 50


def watch_limit(user):
    return 5 if access.has_tool(user, 'saved_searches') else 1


def fingerprint(params):
    # the sort order changes how a search is shown, not what it finds
    return hashlib.sha256(json.dumps({k: v for k, v in params.items() if k != 'sort'}, sort_keys=True).encode()).hexdigest()


def state_params(state):
    return {key: state.getlist(key) if key in ('journals', 'exclude_journals') else state[key] for key in state}


def owned_edit(request):
    """The saved search the results page is editing (?edit=<pk>), when it is the user's."""
    pk = request.GET.get('edit', '')
    if not (request.user.is_authenticated and pk.isdigit()):
        return None
    return SavedSearch.objects.filter(user=request.user, pk=pk).first()


def action_context(request, state):
    params = state_params(state)
    saved = SavedSearch.objects.filter(user=request.user, fingerprint=fingerprint(params)).first() if request.user.is_authenticated else None
    return {'save_token': signing.dumps(params, salt=STATE_SALT, compress=True), 'saved_search': saved,
            'editing_search': owned_edit(request),
            'search_return': reverse('search_results') + '?' + state.urlencode()}


def load_params(token):
    try:
        params = signing.loads(token, salt=STATE_SALT, max_age=86400)
        state = QueryDict(urlencode(params, doseq=True))
        form = search_filters.SearchFilters(state)
        if not form.is_valid() or not (state.get('q', '').strip() or state.get('orcid') or state.get('journals')):
            raise ValueError
        if state.get('src', 'all') not in ('all', 'arxiv', 'open') or state.get('field', 'all') not in ('all', 'author', 'title', 'abstract'):
            raise ValueError
        if len(state.get('q', '')) > 2000:
            raise ValueError
        return params
    except (signing.BadSignature, ValueError, TypeError, AttributeError):
        return None


def default_name(params):
    query = ' '.join((params.get('q') or '').split())
    if params.get('field') == 'author' and query:
        name = _('Author: %(name)s') % {'name': query}
    elif params.get('orcid'):
        name = (query + ' · ' if query else '') + 'ORCID ' + params['orcid']
    elif query:
        name = query
    elif params.get('journals'):
        name = ', '.join(j['name'] for j in search_filters.selected_journals(params['journals']))
    else:
        name = _('Journal search')
    return name[:120]


# --- what a watch keeps ---------------------------------------------------------

class Query:
    """The words of a search as a watch reads them in a title or an abstract (see the module docstring)."""

    def __init__(self, text, field='all'):
        text = text or ''
        self.field = field if field in ('author', 'title', 'abstract') else 'all'
        self.supported = not ARXIV_SYNTAX.search(text)
        self.alternatives = []  # [([(term, pattern)] required, [(term, pattern)] excluded)]
        self.names = []  # field 'author': [(last name, first name or initial or '')]
        if self.field == 'author':
            for name in re.split(r'\s+OR\s+|[,;|]', text):
                key = digest.name_key(name)
                last = re.findall(r'[a-z]+', unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode().lower())
                if key:
                    self.names.append(key)
                elif last:
                    self.names.append((last[-1], ''))
            self.supported = self.supported and bool(self.names)
            return
        for part in re.split(r'\s+OR\s+|\s*\|\s*', text):
            include, exclude = [], []
            for raw in re.findall(r'-?"[^"]+"|\S+', part):
                negative = raw.startswith('-') and len(raw) > 1
                term = raw[1:] if negative else raw
                exact = len(term) > 1 and term[0] == term[-1] == '"'
                term = term.strip('"').strip()
                if not exact and (term == 'AND' or term.lower() in STOPWORDS):
                    continue
                pattern = digest.keyword_pattern(term, exact)
                if pattern is not None:
                    (exclude if negative else include).append((term, pattern))
            if include:
                self.alternatives.append((include, exclude))
        self.supported = self.supported and bool(self.alternatives)

    def text(self, paper):
        if self.field == 'title':
            return paper['title']
        if self.field == 'abstract':
            return paper['abstract']
        return paper['title'] + ' ' + paper['abstract']

    def match(self, paper):
        """The terms (or authors) found, [] when the paper does not match."""
        if self.field == 'author':
            found = []
            for last, first in self.names:
                for author in paper['authors']:
                    if digest.same_author((last, first), author) if first else (digest.name_key(author) or ('',))[0] == last:
                        found.append(author)
                        break
            return found
        text = self.text(paper)
        found = []
        for include, exclude in self.alternatives:
            if all(p.search(text) for _, p in include) and not any(p.search(text) for _, p in exclude):
                found.extend(t for t, _ in include if t not in found)
        return found

    def prefilter(self):
        """A database filter every match passes, so that only a few papers are read in Python."""
        if self.field == 'author':
            return Q(*[Q(authors__icontains=last) for last, _ in self.names], _connector=Q.OR)
        fields = {'title': ['title'], 'abstract': ['abstract']}.get(self.field, ['title', 'abstract'])
        alternatives = []
        for include, _ in self.alternatives:
            word = max((w.rstrip('*') for term, _ in include for w in re.findall(r'[^\s\-]+', term.lower())), key=len)
            word = os.path.commonprefix(sorted(digest.word_forms(word))) or word  # galax: galaxy and galaxies
            alternatives.append(Q(*[Q(**{'%s__icontains' % f: word}) for f in fields], _connector=Q.OR))
        return Q(*alternatives, _connector=Q.OR)


def category_list(text):
    return [c for c in re.split(r'[\s,]+', text or '') if c][:10]


def clean_categories(text):
    """The arXiv, bioRxiv or medRxiv codes of a watch ('astro-ph.EP, cs.LG', or a whole archive: 'astro-ph')."""
    kept = []
    for code in category_list(text):
        known = digest.archive_of(code) in digest.ARCHIVES + digest.BIO_ARCHIVES or code in digest.BIO_CODES
        if digest.CATEGORY_RE.match(code) and known and code not in kept:
            kept.append(code)
    return ', '.join(kept)


def in_categories(paper, wanted):
    return any(c == w or c.startswith(w + '.') for c in paper['categories'] for w in wanted)


def scope(params, categories=''):
    """(sources of the announcements the watch reads: 'arxiv' and/or 'other', search engines it asks: 'arxiv'
    and/or 'openalex'). Categories keep a watch to the announcements, when they can answer it."""
    query = Query(params.get('q', ''), params.get('field', 'all'))
    src = params.get('src', 'all')
    if params.get('type') in ('journals', 'reviews') or any(params.get(k) for k in ('journals', 'exclude_journals', 'orcid')):
        return set(), ({'openalex'} if src != 'arxiv' else set())
    pool, api = set(), set()
    if src in ('all', 'arxiv'):
        (pool if query.supported else api).add('arxiv')
    if src in ('all', 'open'):
        if query.supported:
            pool.add('other')
        api.add('openalex')
    if categories and pool:
        api = set()
    return pool, api


def pool_row(paper, terms):
    authors = json.loads(paper.authors or '[]')
    categories = paper.categories.split(',')
    external = sources.is_external(paper.arxiv_id)
    archive = digest.archive_of(categories[0]) if categories else ''
    venue = 'arXiv' if not external else {'biorxiv': 'bioRxiv', 'medrxiv': 'medRxiv'}.get(archive) or paper.venue or \
        digest.category_name(categories[0])
    row = {'key': paper.arxiv_id, 'title': ' '.join(paper.title.split()), 'authors': authors[:6],
           'more_authors': len(authors) > 6, 'venue': venue, 'published': paper.announced.isoformat(),
           'abstract': paper.abstract, 'terms': terms}
    if not external:
        row['arxiv_id'] = paper.arxiv_id
    return row


def pool_rows(params, categories, since, kinds):
    """The announced papers since this day that match (and the categories), oldest announcement first."""
    query = Query(params.get('q', ''), params.get('field', 'all'))
    rows = AnnouncedPaper.objects.filter(announced__gte=since).filter(query.prefilter()).order_by('announced', 'pk')
    wanted = category_list(categories)
    found = []
    for paper in rows.iterator():
        external = sources.is_external(paper.arxiv_id)
        if ('other' if external else 'arxiv') not in kinds:
            continue
        data = {'title': paper.title, 'abstract': paper.abstract, 'authors': json.loads(paper.authors or '[]'),
                'categories': paper.categories.split(',')}
        if wanted and not in_categories(data, wanted):
            continue
        terms = query.match(data)
        if terms:
            found.append(pool_row(paper, terms))
    return found


def in_pool(row, pool):
    """A paper of a search engine that the announcements already give."""
    if 'arxiv' in pool and row.get('arxiv_id'):
        return True
    return 'other' in pool and row['key'].startswith(tuple(sources.doi_key(p + 'x')[:-1] for p in sources.BIORXIV_PREFIXES))


def api_rows(params, since, engines, pool, pages=API_PAGES):
    """(papers published since this date, a source failed, the window was cut) from the search engines."""
    form = search_filters.SearchFilters(params)
    if not form.is_valid():
        raise ValueError('Invalid saved search')
    options = {k: v for k, v in form.options().items() if k not in ('year_from', 'year_to', 'sort')}
    query, field = params.get('q', ''), params.get('field', 'all')
    terms = [query] if query else []
    rows, failed, truncated = [], False, False
    if 'arxiv' in engines:
        for page in range(1, min(pages, ARXIV_PAGES) + 1):
            try:
                batch = unified_search.arxiv_results(query, field=field, filters={'sort': 'newest'}, page=page)
            except (requests.RequestException, ElementTree.ParseError):
                failed = True
                break
            recent = [r for r in batch if str(r['published'])[:10] >= since.isoformat()]
            rows.extend(dict(r, terms=terms) for r in recent)
            if len(recent) < len(batch) or not batch.has_more:
                break
        else:
            truncated = True
    if 'openalex' in engines:
        kind = params.get('type') if params.get('type') in sources.SEARCH_TYPES else ''
        for page in range(1, pages + 1):
            try:
                batch = sources.search(query, kind, per_page=API_PER_PAGE, field=field, page=page, keep=False, watch=True,
                                       filters=dict(options, sort='newest', since=since.isoformat()))
            except (sources.SourceUnavailable, requests.RequestException):
                failed = True
                break
            rows.extend(dict(r, terms=terms) for r in batch if not in_pool(r, pool))
            if not batch.has_more:
                break
        else:
            truncated = True
    return rows, failed, truncated


def distinct(rows):
    kept, seen = [], set()
    for row in rows:
        aliases = unified_search.paper_aliases(row)
        if not aliases & seen:
            kept.append(row)
        seen.update(aliases)
    return kept


def collect(saved, since, api_since, pages=API_PAGES):
    """(matching papers, a source failed, the window was cut): the announcements since `since` (a date) and what
    the search engines have published since `api_since`, over at most `pages` pages."""
    params = saved.params
    pool, engines = scope(params, saved.categories)
    rows = pool_rows(params, saved.categories, since, pool) if pool else []
    failed = truncated = False
    if engines:
        found, failed, truncated = api_rows(params, api_since, engines, pool, pages)
        rows += found
        if failed and pool and saved.last_checked is not None:
            # the announcements are read whole: their papers go out now, and the search engine's are found again
            # by a later check, whose window starts weeks before the last one
            logger.warning('Saved search %s: a search engine failed, its papers wait for the next check', saved.pk)
            failed = False
    return distinct(rows), failed, truncated


def windows(saved, now):
    """(first announcement day, first publication day) a check of this watch reads."""
    oldest = (now - datetime.timedelta(days=POOL_DAYS)).date()
    if saved.last_checked is None:
        return oldest, (now - API_LAG).date()
    return (max((saved.last_checked - LOOKBACK).date(), oldest),
            max((saved.last_checked - API_LAG).date(), (now - datetime.timedelta(days=API_MAX_DAYS)).date()))


def latest(saved, now=None):
    """(the papers a check of this watch reads now, a source failed, the window was cut)."""
    now = now or timezone.now()
    return collect(saved, *windows(saved, now))


# --- ranking and summaries ------------------------------------------------------

def query_vector(saved):
    key = 'saved-search-vector:%s' % saved.fingerprint
    cached = cache.get(key)
    if cached is not None:
        return np.frombuffer(cached, dtype=np.float32)
    from . import similar
    vector = similar.embed([saved.name + '\n' + (saved.params.get('q') or '')])[0]
    cache.set(key, vector.tobytes(), 30 * 86400)
    return vector


def feedback_vectors(saved):
    likes, dislikes = [], []
    for value, vector in saved.feedback.exclude(vector__isnull=True).values_list('value', 'vector'):
        (likes if value > 0 else dislikes).append(np.frombuffer(bytes(vector), dtype=np.float32))
    return np.array(likes), np.array(dislikes)


def rank(saved, rows):
    """The papers by closeness to the search, raised by the ones marked relevant and lowered by the others; with
    `mute`, a paper closer to one marked not relevant than to any marked relevant is set aside (row['muted']).
    As they are when the model cannot be reached."""
    muting = saved.mute and saved.feedback.filter(value__lt=0).exists()
    if not rows or not getattr(settings, 'OPENAI_KEY', '') or (len(rows) < 2 and not muting):
        return rows
    head, tail = rows[:RANKED], rows[RANKED:]
    try:
        vectors = digest.ensure_vectors([{'arxiv_id': r['key'], 'title': r['title'], 'abstract': r.get('abstract') or ''}
                                         for r in head])
        query = query_vector(saved)
    except Exception:
        logger.warning('Ranking of saved search %s failed', saved.pk, exc_info=True)
        return rows
    likes, dislikes = feedback_vectors(saved)
    floors = mute_floors(saved, dislikes, vectors, len(query)) if muting and len(dislikes) else None

    def score(row):
        vector = vectors.get(row['key'])
        if vector is None or len(vector) != len(query):
            return -1.0
        value = float(query @ vector)
        liked = float(np.max(likes @ vector)) if len(likes) else 0.0
        disliked = dislikes @ vector if len(dislikes) else np.zeros(1)
        if floors is not None and np.any(disliked >= floors) and float(np.max(disliked)) > liked:
            row['muted'] = True
        return value + LIKE_WEIGHT * liked - DISLIKE_WEIGHT * float(np.max(disliked))
    ranked = sorted(head, key=score, reverse=True)
    return [r for r in ranked if not r.get('muted')] + tail + [r for r in ranked if r.get('muted')]


def mute_floors(saved, dislikes, vectors, size):
    """Per paper marked not relevant, the closeness past which a new paper is set aside: the median closeness of
    the watch's papers (this check's and the history's) to it, plus MUTE_MARGIN."""
    known = digest.stored_vectors([r['key'] for r in saved.matches if r['key'] not in vectors])
    sample = [v for v in list(vectors.values()) + list(known.values()) if len(v) == size]
    if len(sample) < MUTE_SAMPLE:
        return np.full(len(dislikes), MUTE_FALLBACK)
    return np.maximum(np.median(dislikes @ np.array(sample).T, axis=1) + MUTE_MARGIN, MUTE_FLOOR)


def add_blurbs(rows, lang):
    """Two plain sentences on each paper, from the digests' when they wrote one, else from the model."""
    if not rows:
        return
    known = dict(DigestItem.objects.filter(arxiv_id__in=[r['key'] for r in rows], lang=lang).values_list('arxiv_id', 'blurb'))
    missing = [r for r in rows if not known.get(r['key']) and r.get('abstract')]
    if missing and getattr(settings, 'OPENAI_KEY', ''):
        try:
            written = digest.write_blurbs([{'arxiv_id': r['key'], 'title': r['title'], 'abstract': r['abstract']}
                                           for r in missing], lang)
            known.update((k, v['summary']) for k, v in written.items())
        except Exception:
            logger.warning('Summaries of saved search papers failed', exc_info=True)
    for row in rows:
        if known.get(row['key']):
            row['blurb'] = known[row['key']]


def stored(row, now):
    """What the page, the digest and the feed keep of a new paper."""
    kept = {k: row[k] for k in ('key', 'title', 'authors', 'venue', 'published', 'terms', 'blurb', 'arxiv_id', 'more_authors', 'muted')
            if row.get(k)}
    kept['published'] = str(row.get('published') or '')[:10]
    kept['abstract'] = re.sub(r'<[^>]+>', '', row.get('abstract') or '')[:700]
    kept['found'] = now.isoformat()
    return kept


def found_on(row):
    try:
        return datetime.datetime.fromisoformat(row['found'])
    except (KeyError, TypeError, ValueError):
        return None


def paper_url(base_url, lang, key):
    return '%s/%s/paper/%s/' % (base_url.rstrip('/'), lang, key)


# --- the hourly check -----------------------------------------------------------

def check(saved, now, base_url, mails, queue):
    """Check one claimed watch: 'sent', 'checked' or None (to retry). State changes only after a complete
    read of the sources and, for an email, after it went out."""
    saved.refresh_from_db()
    rows, failed, truncated = latest(saved, now)
    if failed:
        return None
    seen = set(saved.seen)
    if saved.last_checked is None:
        # first complete check after activation or an edit: what exists already is recorded, never sent
        known = list(dict.fromkeys(saved.seen + [a for r in rows for a in sorted(unified_search.paper_aliases(r))]))[-MAX_SEEN:]
        SavedSearch.objects.filter(pk=saved.pk).update(seen=known, last_checked=now, truncated=truncated)
        return 'checked'
    new = [r for r in rows if not (unified_search.paper_aliases(r) & seen)]
    if not new:
        SavedSearch.objects.filter(pk=saved.pk, watch_enabled=True).update(last_checked=now, truncated=truncated)
        return 'checked'
    new = rank(saved, new)
    add_blurbs([r for r in new if not r.get('muted')][:MAX_EMAIL_ITEMS], saved.lang)
    kept = [stored(r, now) for r in new]
    delivered = [r for r in kept if not r.get('muted')]  # the muted ones stay on the page only
    mails = mails and bool(delivered)
    queue = queue and bool(delivered)
    if mails:
        if not SavedSearch.objects.filter(pk=saved.pk, watch_enabled=True).exists():
            return None
        if not send(saved, delivered, truncated, base_url):
            return None
    known = list(dict.fromkeys(saved.seen + [a for r in new for a in sorted(unified_search.paper_aliases(r))]))[-MAX_SEEN:]
    fields = {'seen': known, 'matches': (kept + saved.matches)[:MAX_HISTORY], 'found_total': F('found_total') + len(kept),
              'last_checked': now, 'truncated': truncated}
    if mails:
        fields['last_sent'] = now
    if queue:
        fields['pending'] = (saved.pending + delivered)[-MAX_HISTORY:]
    SavedSearch.objects.filter(pk=saved.pk).update(**fields)
    return 'sent' if mails else 'checked'


def send(saved, papers, truncated, base_url):
    base = base_url.rstrip('/')
    token = signing.dumps(saved.pk, salt=UNSUBSCRIBE_SALT)
    papers = [dict(p, vote_yes=base + '/%s/saved-searches/vote/%s/' % (saved.lang, vote_token(saved, p['key'], 1)),
                   vote_no=base + '/%s/saved-searches/vote/%s/' % (saved.lang, vote_token(saved, p['key'], -1)))
              for p in papers]
    context = {'saved': saved, 'papers': papers[:MAX_EMAIL_ITEMS], 'more': max(len(papers) - MAX_EMAIL_ITEMS, 0),
               'total': len(papers), 'truncated': truncated, 'base_url': base,
               'search_url': base + '/%s/search-results/?' % saved.lang + urlencode(saved.params, doseq=True),
               'page_url': base + '/%s/saved-searches/%d/' % (saved.lang, saved.pk),
               'manage_url': base + '/%s/saved-searches/' % saved.lang,
               'unsubscribe_url': base + '/%s/saved-searches/unsubscribe/?' % saved.lang + urlencode({'token': token})}
    with translation.override(saved.lang):
        text = render_to_string('search_watches/email.txt', context)
        html = render_to_string('search_watches/email.html', context)
        subject = translation.ngettext('%(n)s new paper for your saved search: %(name)s',
                                       '%(n)s new papers for your saved search: %(name)s', len(papers)) % {
            'n': len(papers), 'name': saved.name}
    message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [saved.user.email],
                                     headers={'List-Unsubscribe': '<%s>' % context['unsubscribe_url']})
    message.attach_alternative(html, 'text/html')
    try:
        return message.send() == 1
    except Exception:
        logger.exception('Saved search watch %s could not be emailed', saved.pk)
        return False


def eligible_ids(user):
    return set(user.saved_searches.filter(watch_enabled=True).order_by('pk').values_list('pk', flat=True)[:watch_limit(user)])


def claim(saved, now):
    """A five-minute lease on a due watch, or None when it is not due or another run has it."""
    delay = datetime.timedelta(days=1 if saved.frequency == 'daily' else 7)
    if saved.last_checked and saved.last_checked > now - delay:
        return None
    lease = now + datetime.timedelta(minutes=5)
    claimed = SavedSearch.objects.filter(pk=saved.pk, watch_enabled=True).filter(
        Q(watch_claim_until__isnull=True) | Q(watch_claim_until__lte=now)).filter(
        Q(last_checked__isnull=True) | Q(last_checked__lte=now - delay)).update(watch_claim_until=lease, last_attempt=now)
    return lease if claimed else None


def run_one(saved, now, base_url, mails, queue):
    lease = claim(saved, now)
    if lease is None:
        return False, None
    try:
        return True, check(saved, now, base_url, mails, queue)
    except Exception:
        logger.exception('Saved search watch %s failed', saved.pk)
        return True, None
    finally:
        SavedSearch.objects.filter(pk=saved.pk, watch_claim_until=lease).update(watch_claim_until=None)


def run(now, budget, base_url=digest.BASE_URL):
    """Check the due watches; email the new papers of those not carried by a digest. Returns emails sent."""
    from . import digest_bundle
    if not SavedSearch.objects.filter(watch_enabled=True).exists():
        return 0
    try:
        digest.ensure_batch(now)  # the announcements are the watches' main source
    except Exception:
        logger.exception('The announcements could not be read for the search watches')
    sent, checked, eligible = 0, 0, {}
    suppressed = mailcheck.suppressed()
    bundled = digest_bundle.bundled_users()
    rows = SavedSearch.objects.filter(watch_enabled=True, user__is_active=True).select_related('user').order_by(
        F('last_attempt').asc(nulls_first=True), 'last_checked', 'pk')
    started = time.monotonic()
    for saved in rows.iterator():
        if time.monotonic() - started > RUN_SECONDS:
            break
        asks = bool(scope(saved.params, saved.categories)[1])
        if asks and checked >= MAX_CHECKS_PER_RUN:
            continue
        user = saved.user
        mails = saved.email and user.pk not in bundled
        if mails and (sent >= budget or not user.email or user.email.strip().lower() in suppressed):
            continue
        if user.pk not in eligible:
            eligible[user.pk] = eligible_ids(user)
        if saved.pk not in eligible[user.pk]:
            continue
        ran, result = run_one(saved, now, base_url, mails, queue=saved.email and user.pk in bundled)
        checked += ran and asks
        sent += result == 'sent'
    return sent


def bundle_section(user, base_url, lang, now):
    """The new papers of the user's watches as a section of their digest (see digest_bundle)."""
    eligible = eligible_ids(user)
    for saved in user.saved_searches.filter(pk__in=eligible, email=True):
        # due watches are checked now, so that a paper of today's announcement is in today's digest
        run_one(saved, now, base_url, mails=False, queue=True)
    waiting = [s for s in user.saved_searches.filter(pk__in=eligible) if s.pending]
    if not waiting:
        return None
    items, more = [], 0
    for saved in waiting:
        shown = saved.pending[:5]
        more += len(saved.pending) - len(shown)
        items += [{'group': saved.name, 'title': r['title'], 'url': paper_url(base_url, lang, r['key']),
                   'note': r.get('blurb') or ', '.join(r.get('terms') or []), 'label': ''} for r in shown]

    def commit():
        for saved in waiting:
            SavedSearch.objects.filter(pk=saved.pk).update(pending=[], last_sent=now)
    return {'title': translation.gettext('New papers for your saved searches'), 'items': items, 'more': more,
            'manage_url': '%s/%s/saved-searches/' % (base_url.rstrip('/'), lang), 'commit': commit}


# --- pages ----------------------------------------------------------------------

def preview_key(saved, now):
    return 'saved-search-preview:%s' % hashlib.sha256(json.dumps(
        [saved.fingerprint, saved.categories, now.date().isoformat()]).encode()).hexdigest()


def preview(saved, now=None):
    """What the watch would have found over the announcements kept (two weeks), newest first, with one page of
    each search engine (truncated: there are more); cached an hour."""
    now = now or timezone.now()
    key = preview_key(saved, now)
    found = cache.get(key)
    if found is None:
        since = (now - datetime.timedelta(days=POOL_DAYS)).date()
        try:
            rows, failed, truncated = collect(saved, since, since, pages=1)
        except ValueError:
            rows, failed, truncated = [], True, False
        rows.sort(key=lambda r: str(r.get('published') or ''), reverse=True)
        venues = {}
        for row in rows:
            venue = row['venue'] if row['venue'] in ('arXiv', 'bioRxiv', 'medRxiv') else _('journals')
            venues[venue] = venues.get(venue, 0) + 1
        found = {'total': len(rows), 'per_week': round(len(rows) * 7 / POOL_DAYS), 'days': POOL_DAYS,
                 'venues': sorted(venues.items(), key=lambda v: -v[1]), 'failed': failed, 'truncated': truncated,
                 'rows': [{k: v for k, v in stored(r, now).items() if k != 'found'} for r in rows[:PREVIEW_SHOWN]]}
        cache.set(key, found, 60 if failed else 3600)
    return found


def chips(params, categories=''):
    """The filters of a saved search as labels; the years only shape the search page, never a watch."""
    state = QueryDict(urlencode(params, doseq=True))
    selected = {k: search_filters.selected_journals(state.getlist(k)) if state.getlist(k) else []
                for k in ('journals', 'exclude_journals')}
    labels = [c['label'] for c in search_filters.filter_chips(state, selected) if not c['label'].startswith(str(_('Sort by')))]
    if categories:
        labels.append(_('%(label)s: %(value)s') % {'label': _('Categories'), 'value': categories})
    return labels


def decorate(saved, now):
    saved.watch_available = False
    saved.labels = chips(saved.params, saved.categories)
    saved.years = bool(saved.params.get('year_from') or saved.params.get('year_to'))
    viewed = saved.last_viewed.isoformat() if saved.last_viewed else ''
    saved.new_count = sum(1 for r in saved.matches if r.get('found', '') > viewed)
    saved.just_checked = bool(saved.last_checked and now - saved.last_checked < datetime.timedelta(minutes=1))
    pool, engines = scope(saved.params, saved.categories)
    saved.reads_announcements = bool(pool)
    saved.reads_engines = bool(engines)
    return saved


def watch_options(request, saved):
    saved.frequency = 'daily' if request.POST.get('frequency') == 'daily' else 'weekly'
    saved.lang = request.POST.get('lang') if request.POST.get('lang') in ('en', 'fr') else get_language()
    saved.email = request.POST.get('delivery', 'email') != 'page'
    saved.mute = request.POST.get('mute') == '1'
    categories = clean_categories(request.POST.get('categories', ''))
    if not scope(saved.params)[0]:
        categories = ''  # journals, ORCID or arXiv syntax: the announcements cannot answer, nor keep to categories
    if categories != saved.categories:
        saved.categories = categories
        saved.last_checked = None  # what the new categories already hold is recorded, never sent
        saved.last_attempt = None
    return ['frequency', 'lang', 'email', 'mute', 'categories', 'last_checked', 'last_attempt']


def back(request, saved=None):
    if saved is not None and request.POST.get('next') == 'detail':
        return redirect('saved_search', pk=saved.pk)
    return redirect('saved_searches')


@never_cache
@login_required
@require_POST
def save(request):
    params = load_params(request.POST.get('state', ''))
    if params is None:
        return HttpResponseBadRequest(_('This search link has expired. Run the search again before saving it.'))
    replace = request.POST.get('replace', '')
    with transaction.atomic():
        User.objects.select_for_update().get(pk=request.user.pk)
        existing = SavedSearch.objects.filter(user=request.user, fingerprint=fingerprint(params)).first()
        if replace.isdigit():
            saved = get_object_or_404(SavedSearch.objects.select_for_update(), pk=replace, user=request.user)
            if existing and existing.pk != saved.pk:
                messages.error(request, _('This search is already saved under the name “%(name)s”.') % {'name': existing.name})
                return redirect('saved_search', pk=existing.pk)
            renamed = saved.name == default_name(saved.params)
            saved.params, saved.fingerprint = params, fingerprint(params)
            if renamed:
                saved.name = default_name(params)
            # the new search starts again from now: what it already finds is recorded, never sent
            saved.last_checked = saved.last_attempt = None
            saved.save(update_fields=['params', 'fingerprint', 'name', 'last_checked', 'last_attempt'])
            messages.success(request, _('Search updated. Its watch, its history and its feed are kept; the papers it already finds are not sent.'))
            return redirect('saved_search', pk=saved.pk)
        if not existing and request.user.saved_searches.count() >= MAX_SAVED:
            messages.error(request, _('You can keep up to 50 saved searches. Remove one before adding another.'))
            return redirect('saved_searches')
        saved, created = SavedSearch.objects.get_or_create(user=request.user, fingerprint=fingerprint(params), defaults={
            'params': params, 'name': default_name(params),
            'lang': get_language() if get_language() in ('en', 'fr') else 'en'})
    if created:
        request.session['saved_search_created'] = True
    messages.success(request, _('Search saved. Below: what it would have found over the last two weeks, and its email watch.'))
    return redirect('saved_search', pk=saved.pk)


@never_cache
@login_required
def manage(request):
    if request.method == 'POST':
        try:
            pk = int(request.POST.get('search', ''))
        except (ValueError, TypeError):
            raise Http404
        saved = get_object_or_404(SavedSearch, user=request.user, pk=pk)
        action = request.POST.get('action')
        if action == 'delete':
            saved.delete()
            messages.success(request, _('Saved search removed. Its watch is stopped.'))
            return redirect('saved_searches')
        elif action == 'pause':
            SavedSearch.objects.filter(pk=saved.pk, user=request.user).update(watch_enabled=False)
            messages.success(request, _('Search watch paused.'))
        elif action == 'rename':
            name = ' '.join(request.POST.get('name', '').split())[:120]
            if name:
                saved.name = name
                saved.save(update_fields=['name'])
        elif action == 'settings':
            saved.save(update_fields=watch_options(request, saved))
            messages.success(request, _('Watch settings saved.'))
        elif action == 'enable':
            page_only = request.POST.get('delivery') == 'page'
            if not page_only and (not request.user.email or request.user.email.strip().lower() in mailcheck.suppressed()):
                messages.error(request, _('A deliverable account email is required to enable a watch.'))
                return back(request, saved)
            if request.user.saved_searches.filter(watch_enabled=True).exclude(pk=saved.pk).count() >= watch_limit(request.user):
                messages.error(request, _('Watch limit reached. Pause another watch or see Premium for more watches.'))
                return back(request, saved)
            # Persist the opt-in immediately. The hourly worker establishes the baseline
            # after a complete source response, retrying outages without blocking the web request.
            with transaction.atomic():
                User.objects.select_for_update().get(pk=request.user.pk)
                saved = get_object_or_404(SavedSearch.objects.select_for_update(), pk=saved.pk, user=request.user)
                count = request.user.saved_searches.filter(watch_enabled=True).exclude(pk=saved.pk).count()
                if count >= watch_limit(request.user):
                    messages.error(request, _('Watch limit reached. Pause another watch or see Premium for more watches.'))
                else:
                    resumed = saved.last_checked is not None
                    saved.watch_enabled = True
                    fields = watch_options(request, saved)
                    saved.save(update_fields=['watch_enabled'] + fields)
                    request.session['search_watch_enabled'] = True
                    if resumed and saved.last_checked is not None:
                        messages.success(request, _('Watch resumed. The papers announced while it was paused (two weeks at most) come with the next check.'))
                    else:
                        messages.success(request, _('Watch enabled. Initialization will run automatically; no existing results will be emailed.'))
        return back(request, saved)
    now = timezone.now()
    rows = [decorate(row, now) for row in request.user.saved_searches.defer('seen', 'pending')]
    eligible = eligible_ids(request.user)
    for row in rows:
        row.watch_available = row.pk in eligible
    from . import digest_bundle
    return render(request, 'summarizer/saved_searches.html', {'searches': rows, 'watch_limit': watch_limit(request.user),
        'bundled': request.user.pk in digest_bundle.bundled_users(),
        'search_saved_event': request.session.pop('saved_search_created', False),
        'watch_enabled_event': request.session.pop('search_watch_enabled', False)})


def selection(request, saved):
    """The papers of the history the user ticked."""
    keys = {r['key'] for r in saved.matches}
    return [k for k in dict.fromkeys(request.POST.getlist('ids')) if k in keys]


def vote_token(saved, key, value):
    return signing.dumps([saved.pk, key, value], salt=VOTE_SALT, compress=True)


def record_vote(saved, key, value):
    """Relevant (1), not relevant (-1) or no opinion (None) on a paper the watch found; False if it found none such."""
    row = next((r for r in saved.matches if r['key'] == key), None)
    if row is None:
        return False
    if value is None:
        SavedSearchFeedback.objects.filter(saved=saved, key=key).delete()
        return True
    vector = digest.stored_vectors([key]).get(key)
    if vector is None and getattr(settings, 'OPENAI_KEY', ''):
        try:
            vector = digest.ensure_vectors([{'arxiv_id': key, 'title': row['title'], 'abstract': row.get('abstract') or ''}]).get(key)
        except Exception:
            vector = None
    SavedSearchFeedback.objects.update_or_create(saved=saved, key=key, defaults={
        'value': value, 'vector': vector.tobytes() if vector is not None else None})
    return True


@csrf_exempt  # the link comes from an email; the signed token is the authorization
@never_cache
def vote_link(request, token):
    """Relevant or not, from an email. The link opens a page that votes with a POST sent by the page itself,
    so the link checkers of mail providers, which only follow GET links, do not vote."""
    try:
        pk, key, value = signing.loads(token, salt=VOTE_SALT, max_age=365 * 86400)
        saved = SavedSearch.objects.defer('seen', 'pending').get(pk=pk)
    except (signing.BadSignature, SavedSearch.DoesNotExist, ValueError, TypeError):
        saved = key = value = None
    row = next((r for r in saved.matches if r['key'] == key), None) if saved else None
    done = False
    if row is not None and request.method == 'POST' and value in (1, -1):
        done = record_vote(saved, key, value)
    return render(request, 'summarizer/saved_search_vote.html', {'search': saved if row else None, 'paper': row,
                                                                   'value': value, 'done': done})


def act_on_papers(request, saved):
    from . import exports, library
    from .models import Collection
    action = request.POST.get('action')
    if action == 'vote':
        key, _sep, value = request.POST.get('vote', '').rpartition('|')
        if not record_vote(saved, key, {'1': 1, '-1': -1}.get(value)):
            raise Http404
        if request.headers.get('X-Requested-With') == 'fetch':
            return HttpResponse(status=204)
        return redirect(reverse('saved_search', kwargs={'pk': saved.pk}) + '#paper-' + re.sub(r'[^\w-]', '-', key))
    if action == 'feed':
        saved.feed_key = secrets.token_hex(16)
        saved.save(update_fields=['feed_key'])
        messages.success(request, _('A new feed address was made; the old one no longer works.'))
        return redirect('saved_search', pk=saved.pk)
    ids = selection(request, saved)
    if not ids:
        messages.error(request, _('Tick the papers first.'))
        return redirect('saved_search', pk=saved.pk)
    if action == 'chat':
        from . import assistant
        return redirect('/%s/chat/?ids=%s' % (get_language(), ','.join(ids[:assistant.MAX_PAPERS])))
    if action in ('bib', 'ris', 'md'):
        name = re.sub(r'[^A-Za-z0-9_-]+', '-', saved.name).strip('-')[:60] or 'saved-search'
        metas = [m for m in (exports.paper_meta(k) for k in ids) if m]
        return exports.export_many(metas, action, name, get_language())
    if action == 'collection':
        wanted = request.POST.get('collection', 'new')
        paid = access.has_tool(request.user, 'collections')
        if wanted == 'new':
            if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
                messages.error(request, _('You have reached the number of collections of your plan.'))
                return redirect('saved_search', pk=saved.pk)
            collection = Collection.objects.create(user=request.user, name=saved.name[:120])
        elif wanted.isdigit():
            collection = library.editable(request, wanted)
        else:
            raise Http404
        library.add_many(collection, ids, access.has_tool(collection.user, 'collections'))
        return redirect('/%s/collections/%s/' % (get_language(), collection.pk))
    raise Http404


@never_cache
@login_required
def detail(request, pk):
    from .models import Collection
    saved = get_object_or_404(SavedSearch.objects.defer('seen', 'pending'), user=request.user, pk=pk)
    if request.method == 'POST':
        return act_on_papers(request, saved)
    now = timezone.now()
    decorate(saved, now)
    saved.watch_available = saved.pk in eligible_ids(request.user)
    viewed = saved.last_viewed.isoformat() if saved.last_viewed else ''
    votes = dict(saved.feedback.values_list('key', 'value'))
    history = [dict(r, new=r.get('found', '') > viewed, vote=votes.get(r['key'], 0), found_on=found_on(r),
                    anchor=re.sub(r'[^\w-]', '-', r['key'])) for r in saved.matches]
    if not saved.feed_key:
        saved.feed_key = secrets.token_hex(16)
        saved.save(update_fields=['feed_key'])
    SavedSearch.objects.filter(pk=saved.pk).update(last_viewed=now)
    from . import digest_bundle
    query = saved.params.get('q') or ''
    shown = None
    if not history or request.GET.get('preview'):
        # the announcements answer at once; OpenAlex may take seconds: the page asks for it after loading
        shown = preview(saved, now) if request.GET.get('preview') else cache.get(preview_key(saved, now))
    context = {
        'search': saved, 'history': history, 'watch_limit': watch_limit(request.user),
        'preview': shown, 'preview_pending': shown is None and not history,
        'bundled': request.user.pk in digest_bundle.bundled_users(),
        'collections': Collection.objects.filter(user=request.user).order_by('name'),
        'feed_url': request.build_absolute_uri(reverse('saved_search_feed', kwargs={'key': saved.feed_key})),
        'edit_url': reverse('search_results') + '?' + urlencode(dict(saved.params, edit=saved.pk), doseq=True),
        'topic': query if saved.params.get('field', 'all') != 'author' else '',
        'pool_days': POOL_DAYS,
        'search_saved_event': request.session.pop('saved_search_created', False),
        'watch_enabled_event': request.session.pop('search_watch_enabled', False),
    }
    return render(request, 'summarizer/saved_search.html', context)


@never_cache
@login_required
def preview_part(request, pk):
    """The preview of a saved search, for its page to insert once it has loaded."""
    saved = get_object_or_404(SavedSearch.objects.defer('seen', 'pending', 'matches'), user=request.user, pk=pk)
    return render(request, 'summarizer/saved_search_preview.html', {'preview': preview(saved), 'search': saved})


def feed(request, key):
    """The new papers of a watch as an Atom feed, at an address only its user knows."""
    if not re.fullmatch(r'[0-9a-f]{32}', key):
        raise Http404
    saved = get_object_or_404(SavedSearch, feed_key=key)
    lang = saved.lang
    base = request.build_absolute_uri('/').rstrip('/')
    with translation.override(lang):
        atom = feedgenerator.Atom1Feed(
            title='%s · SummarizePaper' % saved.name, link='%s/%s/saved-searches/%d/' % (base, lang, saved.pk),
            description=_('New papers found by a saved search watch.'), language=lang)
        for row in [r for r in saved.matches if not r.get('muted')][:FEED_ITEMS]:
            found = datetime.datetime.fromisoformat(row['found']) if row.get('found') else saved.created
            atom.add_item(title=row['title'], link=paper_url(base, lang, row['key']), unique_id=row['key'],
                          description=row.get('blurb') or row.get('abstract') or '',
                          author_name=', '.join(row.get('authors') or [])[:200] or None, pubdate=found, updateddate=found,
                          categories=[row['venue']] if row.get('venue') else None)
    response = HttpResponse(atom.writeString('utf-8'), content_type='application/atom+xml; charset=utf-8')
    response['X-Robots-Tag'] = 'noindex'
    return response


def matching(user, key):
    """The user's saved searches whose watch found this paper (shown on its page)."""
    if not user.is_authenticated or not key:
        return []
    return [{'pk': pk, 'name': name} for pk, name, rows in user.saved_searches.values_list('pk', 'name', 'matches')
            if any(r.get('key') == key for r in rows)]


@never_cache
def unsubscribe(request):
    try:
        pk = signing.loads(request.GET.get('token', ''), salt=UNSUBSCRIBE_SALT, max_age=365 * 86400)
        saved = SavedSearch.objects.get(pk=pk)
    except (signing.BadSignature, SavedSearch.DoesNotExist, ValueError, TypeError):
        raise Http404
    paused = False
    if request.method == 'POST':
        SavedSearch.objects.filter(pk=saved.pk).update(watch_enabled=False)
        paused = True
    return render(request, 'summarizer/search_unsubscribe.html', {'paused': paused})
