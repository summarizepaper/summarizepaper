"""A researcher's own space: what they read, asked and made on the site, and what changed since.

- Overview: news about the papers they read or follow (retractions, publications, new versions), their results
  in progress, their to-read pile, papers close to their recent reading and the watches they keep.
- Readings: every paper opened while signed in, summarized at their request or given a reading status from
  another tool (PaperVisit), with its authors, year, venue, editorial status and collections; a reading status
  shared with the user's own collections; a private note; actions on several papers at once.
- Conversations and Results (literature reviews, bibliography checks...), each linking back to its tool.

Never changes generation quota records (PaperHistory): taking a paper out of the history only hides it.
Everything shown comes from stored data: the page never waits for an API. Once a day the hourly run reads the
latest arXiv version of the papers read lately and queues their editorial status checks (check_readings).
"""
import json
import re
import types
from datetime import datetime, timedelta, timezone as dt_timezone
from urllib.parse import urlencode, unquote, urlsplit

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.core.cache import cache
from django.db.models import DateTimeField, F, Max, OuterRef, Q, Subquery, Value
from django.db.models.functions import Coalesce, Greatest, Lower
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import get_language, gettext as _, gettext_lazy, ngettext
from django.views.decorators.http import require_POST

from . import sources

from .models import (ActivityState, AIassistant, ArxivPaper, ChatConversation, ChatShare, Collection, CollectionPaper,
                     DigestSubscription, FollowedPaper, Generated, LetterSubscription, PaperAuthor, PaperHistory,
                     PaperScore, PaperVisit, PublicationAlert, ReferenceWatch, SavedSearch, TopicMapWatch,
                     VersionAlert, WatchedProject)

PER_PAGE = 20
NEWS_DAYS = 90  # alerts of followed papers older than this are no longer news
NEWS_SCAN = 200  # recent readings whose stored facts are looked at for news
MATCHED_PAPERS = 5000  # papers matching a conversation search, at most
ALL_MAX = 200  # papers an action on "every matching paper" takes
STATUSES = ('to_read', 'read', 'key')
TABS = ('overview', 'readings', 'conversations', 'results')
SORTS = ('', 'published', 'title')
EPOCH = datetime(1970, 1, 1, tzinfo=dt_timezone.utc)
SEEN_AGAIN = timedelta(minutes=30)  # a visit of the overview this soon after the last one keeps what was new
MAX_DISMISSED = 1000
LATEST_KEY = 'reading-latest:%s'  # the latest arXiv version of a paper read lately (check_readings)
LATEST_TTL = 8 * 86400
CHECK_DAILY_KEY = 'readings-checked:%s'
CHECK_DAYS = 180  # papers read within this many days are checked
CHECK_MAX = 1000  # papers checked a day at most


def reading_labels():
    from .library import READING
    return {key: str(label) for key, label in READING}


# The results a user makes with the tools, kept as private Generated rows: (name, path under /<lang>/, title)
def _refcheck_title(data):
    n = data.get('entries') or 0
    files = ', '.join(data.get('files') or [])[:120]
    text = ngettext('%(n)s reference', '%(n)s references', n) % {'n': n}
    return '%s · %s' % (files, text) if files else text


RESULT_KINDS = {
    'literature': (gettext_lazy('Literature review'), 'literature-review', lambda d: d.get('question')),
    'refcheck': (gettext_lazy('Bibliography check'), 'refcheck', _refcheck_title),
    'missed': (gettext_lazy('What did I miss?'), 'missed', lambda d: d.get('title')),
    'teams': (gettext_lazy('Who works on it?'), 'teams', lambda d: d.get('title')),
    'students': (gettext_lazy('Find a research placement'), 'students', lambda d: d.get('title')),
    'conference': (gettext_lazy('Conference companion'), 'conference', lambda d: d.get('name')),
}


# --- recording ---------------------------------------------------------------------------

def base_of(arxiv_id):
    from .library import version_of
    return version_of(arxiv_id)[0]


def visit_of(user, arxiv_id):
    """The user's visit of this paper: of this id, else of another arXiv version of it."""
    rows = PaperVisit.objects.filter(user=user)
    return (rows.filter(arxiv_id=arxiv_id).first() or
            rows.filter(arxiv_id__regex=r'^%s(v[0-9]+)?$' % re.escape(base_of(arxiv_id))).order_by('-pk').first())


def save_visit(user, arxiv_id, title='', **fields):
    """Create or update the user's visit of a paper; it takes the id of the version last met."""
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if title:
        fields['title'] = ' '.join(str(title).split())[:500]
    visit = visit_of(user, arxiv_id)
    if visit is None:
        return PaperVisit.objects.create(user=user, arxiv_id=arxiv_id, paper=paper, **dict({'visited': None}, **fields))
    fields.update(arxiv_id=arxiv_id, paper=paper or (visit.paper if visit.arxiv_id == arxiv_id else None))
    PaperVisit.objects.filter(pk=visit.pk).update(**fields)
    return visit


def record_visit(user, arxiv_id, title=''):
    """A paper opened while signed in, summarized or not yet (then known by its id and the title its page showed)."""
    if not title and not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        return
    save_visit(user, arxiv_id, title, visited=timezone.now(), hidden=False)


def record_request(user, arxiv_id):
    """A summary requested while signed in: the paper is in the history even if its page was not opened."""
    if user is None or not user.is_authenticated:
        return
    save_visit(user, arxiv_id, requested=timezone.now(), hidden=False)


def sync_collections(user, ids, status):
    """The reading status of these papers, in any version, in the collections the user owns."""
    bases = {base_of(i) for i in ids}
    pks = [pk for pk, aid in CollectionPaper.objects.filter(collection__user=user).values_list('pk', 'arxiv_id')
           if base_of(aid) in bases]
    if pks:
        CollectionPaper.objects.filter(pk__in=pks).update(reading=status, changed=timezone.now())  # the next Zotero sync writes it


def set_status(user, visits, status):
    PaperVisit.objects.filter(pk__in=[v.pk for v in visits]).update(reading=status)
    sync_collections(user, [v.arxiv_id for v in visits], status)


def mark(user, arxiv_id, status, title='', sync=True):
    """A reading status given from anywhere (a paper page, the digest, a letter, a collection). A paper never
    opened enters the history then, dated by the marking."""
    visit = visit_of(user, arxiv_id)
    if visit is None:
        if not status:
            return None
        visit = PaperVisit.objects.create(user=user, arxiv_id=arxiv_id, reading=status, marked=timezone.now(),
                                          visited=None, title=' '.join(str(title or '').split())[:500],
                                          paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id).first())
    else:
        PaperVisit.objects.filter(pk=visit.pk).update(reading=status, hidden=False if status else visit.hidden)
    if sync:
        sync_collections(user, [arxiv_id], status)
    return visit


def statuses(user):
    """{base id: reading status} of the user's papers with one, for the buttons shown next to papers."""
    return {base_of(a): r for a, r in PaperVisit.objects.filter(user=user).exclude(reading='').values_list('arxiv_id', 'reading')}


# --- search ---------------------------------------------------------------------------------

def search_path(text):
    try:
        return urlsplit(text).path if '://' in text else text
    except ValueError:
        return text  # malformed pasted URLs remain ordinary search text


def paper_matches(query):
    """Local search only: pasted identifiers never trigger a remote lookup."""
    text = unquote(query).strip()
    path = search_path(text)
    paper_key = re.search(r'/paper/([^/?#]+)', path)
    if paper_key:
        text = paper_key.group(1)
    if text.startswith('doi_'):
        return ArxivPaper.objects.filter(arxiv_id__iexact=text)
    doi = sources.clean_doi(text)
    if doi and not doi.startswith('10.48550/arxiv.'):
        return ArxivPaper.objects.filter(Q(arxiv_id__iexact=sources.doi_key(doi)) | Q(link_doi__icontains=doi))
    identifier = arxiv_identifier(text)
    if identifier:
        return ArxivPaper.objects.filter(arxiv_id__iregex=r'^' + re.escape(identifier) + r'(v[0-9]+)?$')
    rows = ArxivPaper.objects.all()
    for word in text.split()[:12]:
        # Subquery avoids multiplying results when several authors match.
        authors = ArxivPaper.objects.filter(authors__name__icontains=word).values('pk')
        rows = rows.filter(Q(title__icontains=word) | Q(arxiv_id__icontains=word) |
                           Q(pk__in=Subquery(authors)))
    return rows


def arxiv_identifier(text):
    """'2101.00001' / 'hep-th--9901001' (no version) for an arXiv id or link, else None."""
    arxiv = re.search(r'(?:^|arxiv:|/abs/|/pdf/|arxiv\.)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:/|--)\d{7}(?:v\d+)?)(?:\.pdf)?/?$',
                      search_path(text), re.I)
    return re.sub(r'v\d+$', '', arxiv.group(1), flags=re.I).replace('/', '--') if arxiv else None


def visit_matches(visits, query):
    """The visits whose paper matches; papers not summarized yet are matched on their id and stored title."""
    text = unquote(query).strip()
    paper_key = re.search(r'/paper/([^/?#]+)', search_path(text))
    key = paper_key.group(1) if paper_key else text
    identifier = arxiv_identifier(key)
    if identifier:
        unsummarized = Q(arxiv_id__iregex=r'^' + re.escape(identifier) + r'(v[0-9]+)?$')
    elif key.startswith('doi_') or sources.clean_doi(key):
        doi = sources.clean_doi(key)
        unsummarized = Q(arxiv_id__iexact=sources.doi_key(doi) if doi else key)
    else:
        unsummarized, noted = Q(), Q()
        for word in text.split()[:12]:
            unsummarized &= Q(title__icontains=word)
            noted &= Q(note__icontains=word)
        return visits.filter(Q(paper__in=Subquery(paper_matches(query).values('pk'))) | Q(paper__isnull=True) & unsummarized | noted)
    return visits.filter(Q(paper__in=Subquery(paper_matches(query).values('pk'))) | Q(paper__isnull=True) & unsummarized)


def period_label(date):
    today = timezone.localdate()
    day = timezone.localtime(date).date()
    if day == today:
        return _('Today')
    if day == today - timedelta(days=1):
        return _('Yesterday')
    if day >= today - timedelta(days=today.weekday()):
        return _('This week')
    if day >= today.replace(day=1):
        return _('This month')
    return _('Earlier')


# --- readings ---------------------------------------------------------------------------------

def readings(user):
    """The user's visible readings, newest first, each with `date`: its last opening, summary request or marking."""
    epoch = Value(EPOCH, output_field=DateTimeField())
    return (PaperVisit.objects.filter(user=user, hidden=False)
            .annotate(date=Greatest(Coalesce('visited', epoch), Coalesce('requested', epoch), Coalesce('marked', epoch)))
            .filter(date__gt=EPOCH).order_by('-date', '-pk'))


def filtered_readings(user, query='', status='', sort=''):
    """(rows, status, sort) of the readings tab; an unknown status or sort is ignored."""
    rows = readings(user).select_related('paper')
    if status in STATUSES:
        rows = rows.filter(reading=status)
    elif status == 'unsorted':
        rows = rows.filter(reading='')
    elif status == 'unsummarized':
        rows = rows.filter(paper__isnull=True)
    elif status == 'noted':
        rows = rows.exclude(note='')
    else:
        status = ''
    if query:
        rows = visit_matches(rows, query)
    if sort == 'published':
        rows = rows.order_by(F('paper__published_arxiv').desc(nulls_last=True), '-date', '-pk')
    elif sort == 'title':
        rows = rows.order_by(Lower(Coalesce('paper__title', 'title')), '-pk')
    else:
        sort = ''
    return rows, status, sort


def latest_versions(ids):
    """{arXiv base id: latest version} read by check_readings."""
    from . import changes
    bases = {changes.base_id(i)[0] for i in ids if not sources.is_external(i)}
    found = cache.get_many([LATEST_KEY % b for b in bases])
    return {key[len(LATEST_KEY) - 2:]: version for key, version in found.items()}


def link_summarized(visits):
    """Papers summarized since they were opened get their row (a page of visits, one query)."""
    missing = [v for v in visits if v.paper_id is None]
    if not missing:
        return
    papers = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=[v.arxiv_id for v in missing])}
    for visit in missing:
        if visit.arxiv_id in papers:
            visit.paper = papers[visit.arxiv_id]
            PaperVisit.objects.filter(pk=visit.pk).update(paper=visit.paper)


def collections_of(user, ids):
    """{arxiv_id: [{name, url}]} of the user's collections (owned or shared with them) holding each paper, in any version."""
    from .library import version_of
    bases = {version_of(i)[0]: i for i in ids}
    found = {}
    rows = (CollectionPaper.objects.filter(Q(collection__user=user) | Q(collection__members__user=user))
            .values_list('arxiv_id', 'collection_id', 'collection__name').distinct())
    for aid, pk, name in rows:
        wanted = bases.get(version_of(aid)[0])
        if wanted and pk not in [c['pk'] for c in found.get(wanted, [])]:
            found.setdefault(wanted, []).append({'pk': pk, 'name': name, 'url': '/%s/collections/%s/' % (get_language(), pk)})
    return found


def describe_readings(user, visits):
    """What a card shows of each visit: authors, year, venue, editorial status, newer version, collections..."""
    from . import citations, collection_tools
    link_summarized(visits)
    facts = collection_tools.facts(visits, latest_versions([v.arxiv_id for v in visits]))
    authors = {}
    for paper_id, name in (PaperAuthor.objects.filter(paper_id__in=[v.paper_id for v in visits if v.paper_id])
                           .order_by('paper_id', 'author_order').values_list('paper_id', 'author__name')):
        authors.setdefault(paper_id, []).append(name)
    ids = [v.arxiv_id for v in visits]
    in_collections = collections_of(user, ids)
    followed = set(FollowedPaper.objects.filter(user=user).values_list('arxiv_id', flat=True))
    labels = reading_labels()
    items = []
    for visit in visits:
        paper, fact = visit.paper, facts.get(visit.arxiv_id, {})
        names = authors.get(visit.paper_id, [])
        url = reverse('arxividpage', kwargs={'arxiv_id': visit.arxiv_id})
        items.append({
            'pk': visit.pk, 'id': visit.arxiv_id, 'url': url, 'chat': url + '#assistant',
            'collection': url + '#paper-collections',
            'title': (paper.title if paper else '') or visit.title or visit.arxiv_id,
            'authors': ', '.join(names[:3]) + (' et al.' if len(names) > 3 else ''),
            'year': fact.get('year', ''), 'venue': fact.get('venue', ''),
            'category': (paper.category or '') if paper else '',
            'status_label': fact.get('label', ''), 'status_tone': fact.get('tone', ''),
            'notices': fact.get('notices', []), 'newer': fact.get('newer'),
            'changes': reverse('changes', kwargs={'arxiv_id': visit.arxiv_id}) if fact.get('newer') else '',
            'citations': fact.get('citations'),
            'summarized': paper is not None, 'collections': in_collections.get(visit.arxiv_id, []),
            'followed': citations.base_id(visit.arxiv_id) in followed if visit.arxiv_id else False,
            'reading': visit.reading, 'reading_label': labels.get(visit.reading, ''), 'note': visit.note,
            'date': visit.date, 'visited': visit.visited is not None and visit.visited == visit.date,
            'date_kind': ('visited' if visit.visited == visit.date else 'requested' if visit.requested == visit.date
                          else 'marked'),
        })
    return items


# --- overview ---------------------------------------------------------------------------------

def news(user):
    """[{id, title, url, tone, news: [{label, tone, link, link_label}]}] of what changed about the papers the user read or follows:
    retractions and corrections, journal publication since they read the preprint, newer arXiv versions."""
    from . import collection_tools, publication_status
    visits = list(readings(user).select_related('paper')[:NEWS_SCAN])
    facts = collection_tools.facts(visits, latest_versions([v.arxiv_id for v in visits]))
    published = {g.key: g.data() for g in Generated.objects.filter(
        kind=publication_status.KIND, lang='en', key__in=[v.arxiv_id for v in visits],
        content__contains='"published"')}
    lang = get_language()
    found, seen = [], set()

    def add(arxiv_id, title, kind, label, tone, date, link='', link_label=''):
        if (arxiv_id, kind) in seen:
            return
        seen.add((arxiv_id, kind))
        found.append({'id': arxiv_id, 'title': title, 'url': '/%s/paper/%s/' % (lang, arxiv_id), 'label': label,
                      'tone': tone, 'date': date, 'link': link, 'link_label': link_label,
                      'key': '%s:%s' % (base_of(arxiv_id), kind)})

    for visit in visits:
        fact = facts.get(visit.arxiv_id) or {}
        title = (visit.paper.title if visit.paper else '') or visit.title or visit.arxiv_id
        for notice in fact.get('notices', []):
            add(visit.arxiv_id, title, notice.get('type'), str(notice['label']), notice['tone'], visit.date,
                'https://doi.org/' + notice['doi'] if notice.get('doi') else '', _('Read the notice'))
        if fact.get('newer'):
            add(visit.arxiv_id, title, 'version', _('Version %(n)s is on arXiv') % {'n': fact['newer']}, 'info',
                visit.date, reverse('changes', kwargs={'arxiv_id': visit.arxiv_id}), _('See what changed'))
        report = published.get(visit.arxiv_id) or {}
        day = publication_status.iso_date(report.get('date'))[0] if report.get('status') == 'published' else None
        if day and day >= timezone.localtime(visit.date).date():
            add(visit.arxiv_id, title, 'published', _('Published in %(journal)s since you read it') % {
                'journal': report.get('journal') or _('a journal')}, 'good', visit.date,
                'https://doi.org/' + report['doi'] if report.get('doi') else '', _('Journal version'))
    since = timezone.now() - timedelta(days=NEWS_DAYS)
    for alert in PublicationAlert.objects.filter(user=user, created__gte=since).order_by('-created')[:20]:
        paper = alert.paper or alert.arxiv_id.replace('/', '--')
        if alert.kind == 'published':
            label, tone = (_('Published in %(journal)s') % {'journal': alert.journal} if alert.journal
                           else _('Published in a journal')), 'good'
        else:
            label, tone = collection_tools.NOTICE_LABELS.get(alert.kind, (alert.kind, 'info'))
        add(paper, alert.title or paper, alert.kind, str(label), tone, alert.created,
            'https://doi.org/' + alert.doi if alert.doi else '', _('Read the notice') if alert.kind != 'published' else _('Journal version'))
    titles = dict(FollowedPaper.objects.filter(user=user).values_list('arxiv_id', 'title'))
    for alert in VersionAlert.objects.filter(user=user, created__gte=since).order_by('-created')[:20]:
        paper = '%sv%d' % (alert.arxiv_id.replace('/', '--'), alert.old)
        add(paper, titles.get(alert.arxiv_id) or alert.arxiv_id, 'version',
            _('Version %(n)s is on arXiv') % {'n': alert.new}, 'info', alert.created,
            reverse('changes', kwargs={'arxiv_id': paper}) + '?' + urlencode({'from': alert.old, 'to': alert.new}),
            _('See what changed'))
    order = {'high': 0, 'medium': 1, 'good': 2, 'info': 3}
    found.sort(key=lambda n: (order.get(n['tone'], 3), -n['date'].timestamp()))
    papers = {}  # one line per paper, its most serious news first
    for n in found:
        paper = papers.setdefault(n['id'], {'id': n['id'], 'title': n['title'], 'url': n['url'], 'tone': n['tone'], 'news': []})
        paper['news'].append(n)
    return list(papers.values())


def describe_results(rows, grouped=False):
    """What the page shows of results; grouped: one line per tool and title, the latest, with how many times."""
    from . import generated
    lang = get_language()
    items, same = [], {}
    for row in rows:
        name, path, title = RESULT_KINDS[row.kind]
        data = row.data() if row.status == 'done' and row.content else {}
        status = row.status
        if status == 'pending' and row.updated < timezone.now() - generated.STALE:
            status = 'failed'  # lost with a restarted dyno: its tool starts it again when asked
        state = {'pending': _('In progress…'), 'done': ''}.get(status, _('Failed'))
        item = {'tool': name, 'title': (title(data) if data else '') or name, 'state': state, 'times': 1,
                'url': '/%s/%s/%s/' % (lang, path, row.key), 'date': row.created, 'status': status}
        key = (row.kind, item['title'])
        if grouped and status == 'done' and key in same:
            same[key]['times'] += 1
            continue
        same[key] = item
        items.append(item)
    return items


def results(user, kind=''):
    rows = Generated.objects.filter(user=user, kind__in=[kind] if kind in RESULT_KINDS else list(RESULT_KINDS))
    return rows.order_by('-created', '-pk')


def close_to_reading(user, read_ids, limit=5):
    """Papers close to the user's last readings (the neighbours computed for each paper), not read yet."""
    recent = list(readings(user).filter(paper__isnull=False).values_list('paper_id', 'paper__title')[:10])
    if not recent:
        return []
    source = dict(recent)
    rows = (PaperScore.objects.filter(from_paper_id__in=list(source), active=True)
            .exclude(to_paper__arxiv_id__in=read_ids).exclude(to_paper_id__in=list(source))
            .select_related('to_paper').order_by('-score')[:limit * 6])
    found, seen = [], set()
    for row in rows:
        if row.to_paper_id in seen or not row.to_paper.title:
            continue
        seen.add(row.to_paper_id)
        found.append({'title': row.to_paper.title, 'url': row.to_paper.get_absolute_url(), 'id': row.to_paper.arxiv_id,
                      'because': source[row.from_paper_id]})
        if len(found) == limit:
            break
    return found


def watches(user):
    """What the user keeps an eye on, tool by tool: [{name, count, detail, url}], the tools they use first."""
    lang = get_language()
    rows = [
        (_('Daily digest'), DigestSubscription.objects.filter(user=user).count(), 'digest/'),
        (_('Saved searches'), SavedSearch.objects.filter(user=user).count(), 'saved-searches/'),
        (_('Collections'), Collection.objects.filter(user=user).count(), 'collections/'),
        (_('Followed papers'), FollowedPaper.objects.filter(user=user).count(), 'citations/'),
        (_('Scoop watch'), WatchedProject.objects.filter(user=user).count(), 'scoops/'),
        (_('Weekly letters'), LetterSubscription.objects.filter(user=user, active=True).count(), 'letters/'),
        (_('Watched bibliographies'), ReferenceWatch.objects.filter(user=user, until__gte=timezone.now()).count(), 'refcheck/'),
        (_('Followed topics'), TopicMapWatch.objects.filter(user=user).count(), 'teams/'),
    ]
    return [{'name': name, 'count': count, 'url': '/%s/%s' % (lang, path)} for name, count, path in rows]


def with_state(user, found):
    """The news the user has not set aside, those found since their previous visit marked new; remembers this visit."""
    state, _created = ActivityState.objects.get_or_create(user=user)
    now = timezone.now()
    if state.seen is None or now - state.seen > SEEN_AGAIN:
        state.previous, state.seen = state.seen, now
    dismissed = set(state.dismissed)
    known, shown = {}, []
    for paper in found:
        paper['news'] = [n for n in paper['news'] if n['key'] not in dismissed]
        if not paper['news']:
            continue
        for n in paper['news']:
            first = state.known.get(n['key']) or now.isoformat()
            known[n['key']] = first
            n['new'] = bool(state.previous) and datetime.fromisoformat(first) > state.previous
        paper['new'] = any(n['new'] for n in paper['news'])
        paper['keys'] = ','.join(n['key'] for n in paper['news'])
        shown.append(paper)
    state.known = known  # news no longer found are forgotten
    state.save()
    return shown


def overview(request):
    user = request.user
    recent = list(readings(user).select_related('paper')[:5])
    read_ids = list(readings(user).values_list('arxiv_id', flat=True)[:500])
    to_read = readings(user).filter(reading='to_read')
    found = with_state(user, news(user))
    kept = watches(user)
    return {
        'news': found[:8], 'news_more': max(len(found) - 8, 0), 'news_new': sum(n['new'] for n in found),
        'recent_results': describe_results(results(user)[:20], grouped=True)[:4],
        'to_read': describe_readings(user, list(to_read.select_related('paper')[:5])), 'to_read_count': to_read.count(),
        'recent': describe_readings(user, recent),
        'suggestions': close_to_reading(user, read_ids),
        'watches': [w for w in kept if w['count']], 'unused': [w for w in kept if not w['count']][:4],
    }


# --- the page ---------------------------------------------------------------------------------

def conversations(request, query):
    own = AIassistant.objects.filter(user=request.user, active=True, response__isnull=False).exclude(response='')
    latest = own.order_by().values('arxiv_id').annotate(last=Max('pk')).values('last')
    names = ChatConversation.objects.filter(user=request.user, room=OuterRef('arxiv_id'))
    first = own.filter(arxiv_id=OuterRef('arxiv_id')).order_by('pk')
    rows = own.filter(pk__in=Subquery(latest)).annotate(name=Subquery(names.values('title')[:1]),
                                                        first=Subquery(first.values('query')[:1]))
    if query:
        questions = own
        for word in query.split()[:12]:
            questions = questions.filter(query__icontains=word)
        rooms = set(questions.values_list('arxiv_id', flat=True))
        # The papers matched once, then the user's own exchanges read in Python: a subquery per exchange
        # over every paper took seconds.
        papers = set(paper_matches(query).values_list('arxiv_id', flat=True)[:MATCHED_PAPERS])
        if papers:
            for room, discussed in own.values_list('arxiv_id', 'papers'):
                if room in papers or papers.intersection((discussed or '').split(',')):
                    rooms.add(room)
        named = Q()
        for word in query.split()[:12]:
            named &= Q(name__icontains=word)
        rows = rows.filter(Q(arxiv_id__in=rooms) | named)
    return rows.order_by('-created', '-pk')


def describe_conversations(request, rows):
    rows = list(rows)
    wanted = set()
    for row in rows:
        wanted.add(row.arxiv_id)
        wanted.update(i for i in (row.papers or '').split(',') if i)
    known = dict(ArxivPaper.objects.filter(arxiv_id__in=wanted).values_list('arxiv_id', 'title'))
    shared = set(ChatShare.objects.filter(user=request.user, active=True, room__in=[r.arxiv_id for r in rows])
                 .values_list('room', flat=True))
    items = []
    lang = get_language()
    for row in rows:
        on_paper = row.arxiv_id in known
        if on_paper:
            url = reverse('arxividpage', kwargs={'arxiv_id': row.arxiv_id}) + '#assistant'
            papers = [{'title': known[row.arxiv_id], 'url': '/%s/paper/%s/' % (lang, row.arxiv_id)}]
        else:
            url = reverse('chat') + '?' + urlencode({'c': row.arxiv_id})
            ids = [i for i in (row.papers or '').split(',') if i]
            papers = [{'title': known.get(i) or i, 'url': '/%s/paper/%s/' % (lang, i)} for i in ids]
        title = row.name or row.first or row.query or known.get(row.arxiv_id) or row.arxiv_id
        items.append({'title': title, 'last': row.query if row.query != title else '', 'date': row.created, 'url': url,
                      'room': row.arxiv_id, 'papers': papers[:4], 'more_papers': max(len(papers) - 4, 0),
                      'shared': row.arxiv_id in shared, 'paper_title': known.get(row.arxiv_id, '')})
    return items


def editable_collections(user):
    return list(Collection.objects.filter(Q(user=user) | Q(members__user=user)).distinct().order_by('-updated')[:50])


@login_required(login_url='login')
def page(request):
    tab = request.GET.get('tab', '')
    tab = tab if tab in TABS else 'overview'
    query = request.GET.get('q', '').strip()[:200]
    status = request.GET.get('status', '')
    kind = request.GET.get('kind', '')
    context = {'tab': tab, 'query': query, 'status': status, 'kind': kind, 'items': [],
               'reading_labels': reading_labels(), 'result_kinds': [(k, v[0]) for k, v in RESULT_KINDS.items()],
               'notice': request.GET.get('done', '')}
    if tab == 'overview':
        context.update(overview(request))
        return render(request, 'summarizer/history.html', context)
    sort = request.GET.get('sort', '')
    if tab == 'readings':
        rows, status, sort = filtered_readings(request.user, query, status, sort)
        context['collections'] = editable_collections(request.user)
        if status == 'key':
            context.update(key_papers_context(request.user, rows))
    elif tab == 'conversations':
        rows = conversations(request, query)
    else:
        kind = kind if kind in RESULT_KINDS else ''
        rows = results(request.user, kind)
    result = Paginator(rows, PER_PAGE).get_page(request.GET.get('page'))
    if tab == 'readings':
        items = describe_readings(request.user, list(result))
    elif tab == 'conversations':
        items = describe_conversations(request, result)
    else:
        items = describe_results(result)
    for item in items:
        item['period'] = period_label(item['date'])

    def page_url(number):
        return '?' + urlencode({k: v for k, v in (('tab', tab), ('q', query), ('status', status), ('kind', kind),
                                                   ('sort', sort), ('page', number)) if v})
    context.update({
        'items': items, 'pagination': result, 'status': status, 'kind': kind, 'sort': sort,
        'all_max': min(result.paginator.count, ALL_MAX),
        'clear_url': '?' + urlencode({'tab': tab}),
        'previous_url': page_url(result.previous_page_number()) if result.has_previous() else '',
        'next_url': page_url(result.next_page_number()) if result.has_next() else '',
        'here': request.get_full_path(),
    })
    return render(request, 'summarizer/history.html', context)


def back(request, done=''):
    """To the page the form was sent from (only this page), with what was done."""
    here = request.POST.get('next', '')
    path = reverse('history')
    if not here.startswith(path) or '//' in here:
        here = path + '?tab=readings'
    here = re.sub(r'([?&])done=[^&]*&?', r'\1', here).rstrip('?&')
    if done:
        here += ('&' if '?' in here else '?') + urlencode({'done': done})
    return redirect(here)


def fetched(request):
    return request.headers.get('x-requested-with') == 'fetch'


@login_required(login_url='login')
@require_POST
def selection(request):
    """What a reader does with papers of their history: one (a button of a card), the ones they ticked, or every
    paper matching the search and filter of the page (all=1, up to ALL_MAX). The page posts with fetch the buttons
    of a card (status, note, taking out): JSON then."""
    from . import access, exports, library
    one = request.POST.get('one', '')
    action, _sep, pk = one.partition(':') if one else (request.POST.get('action', ''), '', '')
    if pk:
        rows = PaperVisit.objects.filter(user=request.user, pk=pk if pk.isdigit() else 0)
    elif request.POST.get('all') == '1':
        rows, _status, _sort = filtered_readings(request.user, request.POST.get('q', '').strip()[:200],
                                                 request.POST.get('status', ''), request.POST.get('sort', ''))
        rows = rows[:ALL_MAX]
    else:
        rows = PaperVisit.objects.filter(user=request.user, pk__in=[p for p in request.POST.getlist('v')[:ALL_MAX] if p.isdigit()])
    visits = list(rows.select_related('paper'))
    if not visits:
        return JsonResponse({'status': 'none'}, status=404) if fetched(request) else back(request, 'none')
    ids = [v.arxiv_id for v in visits]
    if action in STATUSES or action == 'clear':
        status = '' if action == 'clear' else action
        set_status(request.user, visits, status)
        if fetched(request):
            return JsonResponse({'status': 'done', 'reading': status, 'label': reading_labels().get(status, '')})
        return back(request, 'status')
    if action == 'note':
        note = request.POST.get('note_%s' % visits[0].pk, request.POST.get('note', '')).strip()[:5000]
        PaperVisit.objects.filter(pk=visits[0].pk).update(note=note)
        return JsonResponse({'status': 'done', 'note': note}) if fetched(request) else back(request, 'note')
    if action == 'hide':
        PaperVisit.objects.filter(pk__in=[v.pk for v in visits]).update(hidden=True)
        return JsonResponse({'status': 'done'}) if fetched(request) else back(request, 'hidden')
    if action == 'chat':
        from .assistant import MAX_PAPERS
        return redirect(reverse('chat') + '?' + urlencode({'ids': ','.join(ids[:MAX_PAPERS])}))
    if action in ('bib', 'ris'):
        metas = [m for m in (exports.paper_meta(v.arxiv_id, types.SimpleNamespace(title=v.title or v.arxiv_id, authors=''))
                             for v in visits) if m]
        notes = {v.arxiv_id: (v.note, '') for v in visits if v.note}
        return exports.export_many(metas, action, 'my-readings', get_language(), notes=notes)
    if action == 'collection':
        paid = access.has_tool(request.user, 'collections')
        wanted = request.POST.get('collection', 'new')
        if wanted == 'new':
            if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
                return back(request, 'too_many')
            name = ' '.join(request.POST.get('name', '').split())[:120] or _('My readings')
            collection = Collection.objects.create(user=request.user, name=name)
        else:
            collection = library.editable(request, wanted)
        counts = library.add_many(collection, ids, access.has_tool(collection.user, 'collections'))
        added = {base_of(v.arxiv_id): v for v in visits}
        for paper in collection.papers.all():  # what the reader knows of each paper goes with it
            visit = added.get(base_of(paper.arxiv_id))
            if visit and (visit.reading and not paper.reading or visit.note and not paper.note):
                paper.reading = paper.reading or visit.reading
                paper.note = paper.note or visit.note
                paper.save(update_fields=['reading', 'note'])
        return redirect('/%s/collections/%s/?%s#papers' % (get_language(), collection.pk, urlencode(counts)))
    raise Http404


@login_required(login_url='login')
@require_POST
def mark_view(request):
    """A reading status given next to a paper anywhere on the site (reading_marks.py). JSON with fetch."""
    from .library import canonical_id
    arxiv_id = canonical_id(request.POST.get('arxiv_id', '')) or ''
    status = request.POST.get('status', '')
    if not arxiv_id or status not in STATUSES + ('',):
        raise Http404
    title = request.POST.get('title', '')[:500]
    if not title and not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        from .models import AnnouncedPaper
        title = AnnouncedPaper.objects.filter(arxiv_id=arxiv_id).values_list('title', flat=True).first() or ''
    mark(request.user, arxiv_id, status, title)
    if fetched(request):
        return JsonResponse({'status': 'done', 'reading': status, 'label': reading_labels().get(status, '')})
    here = request.POST.get('next', '')
    return redirect(here if here.startswith('/') and not here.startswith('//') else reverse('history') + '?tab=readings')


@login_required(login_url='login')
@require_POST
def dismiss_news(request):
    """Set aside news of the overview (keys of with_state); they do not come back."""
    keys = [k for k in request.POST.get('keys', '').split(',') if k][:20]
    state, _created = ActivityState.objects.get_or_create(user=request.user)
    state.dismissed = (state.dismissed + [k for k in keys if k not in state.dismissed])[-MAX_DISMISSED:]
    state.save(update_fields=['dismissed'])
    return JsonResponse({'status': 'done'}) if fetched(request) else redirect(reverse('history'))


def key_papers_context(user, rows):
    """For the key papers filter: the digests whose reference papers they can become (Premium)."""
    from . import access, digest
    return {'key_digests': list(DigestSubscription.objects.filter(user=user)),
            'key_digest_allowed': access.has_tool(user, 'digest'),
            'key_count': min(rows.count(), digest.MAX_REFERENCES), 'max_references': digest.MAX_REFERENCES}


@login_required(login_url='login')
@require_POST
def digest_references(request):
    """The latest key papers become the reference papers of a digest: its papers are ranked by closeness to them."""
    from . import access, digest
    subscription = DigestSubscription.objects.filter(user=request.user, pk=request.POST.get('digest') or 0).first()
    if subscription is None or not access.has_tool(request.user, 'digest'):
        raise Http404
    ids = [base_of(a).replace('--', '/') for a in readings(request.user).filter(reading='key').values_list('arxiv_id', flat=True)]
    refs = digest.clean_references(' '.join(ids))
    if refs:
        subscription.reference_papers = ', '.join(refs)
        subscription.save(update_fields=['reference_papers'])
    return redirect('/%s/digest/?d=%s&saved=1' % (get_language(), subscription.pk))


def check_readings(now):
    """From the hourly run, once a day: the latest arXiv version of the papers read lately (kept in the cache:
    PaperVersion is what version alerts compare with, so it is left to them), and their editorial status checks
    queued. Returns how many papers were checked."""
    from . import changes, digest, publication_status
    if not cache.add(CHECK_DAILY_KEY % now.date(), True, 86400):
        return 0
    since = now - timedelta(days=CHECK_DAYS)
    ids = list(PaperVisit.objects.filter(Q(visited__gte=since) | Q(requested__gte=since) | Q(marked__gte=since))
               .order_by('-pk').values_list('arxiv_id', flat=True)[:CHECK_MAX * 2])
    ids = list(dict.fromkeys(ids))[:CHECK_MAX]
    bases = sorted({changes.base_id(i)[0] for i in ids if not sources.is_external(i)})
    checked = 0
    for start in range(0, len(bases), changes.CHECK_BATCH):
        try:
            papers = digest.fetch_ids(bases[start:start + changes.CHECK_BATCH])
        except digest.ArxivUnavailable:
            continue
        found = {}
        for paper in papers:
            base, version = changes.base_id(paper['arxiv_id'])
            if version:
                found[LATEST_KEY % base] = version
        cache.set_many(found, LATEST_TTL)
        checked += len(found)
    reports = {g.key: g.data() for g in Generated.objects.filter(kind=publication_status.KIND, lang='en', key__in=ids)
               .exclude(content='')}
    publication_status.queue_many([i for i in ids if not sources.is_external(i) and publication_status.page_due(reports.get(i))])
    return checked


@login_required(login_url='login')
@require_POST
def remove_conversation(request, room):
    """Take a conversation out of the history, as the chat page's delete does: its shared link stops."""
    rows = AIassistant.objects.filter(user=request.user, arxiv_id=room, active=True)
    if not rows.exists():
        raise Http404
    rows.update(active=False)
    ChatShare.objects.filter(user=request.user, room=room, active=True).update(active=False)
    ChatConversation.objects.filter(user=request.user, room=room).delete()
    return back(request, 'removed')


@login_required(login_url='login')
@require_POST
def clear_readings(request):
    """Take every reading out of the history (the summary quota records stay: they are not shown)."""
    PaperVisit.objects.filter(user=request.user).update(hidden=True)
    return back(request, 'cleared')


@login_required(login_url='login')
def export_data(request):
    """Everything the site keeps about the user's activity, as JSON: the right to data portability."""
    user = request.user

    def when(value):
        return value.isoformat() if value else None
    data = {
        'account': {'username': user.username, 'email': user.email, 'joined': when(user.date_joined)},
        'readings': [{'paper': v.arxiv_id, 'title': (v.paper.title if v.paper else '') or v.title,
                      'last_opened': when(v.visited), 'summary_requested': when(v.requested), 'marked': when(v.marked),
                      'status': v.reading, 'note': v.note, 'hidden': v.hidden}
                     for v in PaperVisit.objects.filter(user=user).select_related('paper').order_by('pk')],
        'summary_requests': [{'paper': a, 'date': when(d)} for a, d in
                             PaperHistory.objects.filter(user=user).order_by('pk').values_list('arxiv_id', 'created')],
        'conversations': [{'room': a.arxiv_id, 'papers': a.papers or '', 'question': a.query, 'answer': a.response,
                           'date': when(a.created), 'deleted': not a.active}
                          for a in AIassistant.objects.filter(user=user).order_by('pk')],
        'conversation_names': list(ChatConversation.objects.filter(user=user).values('room', 'title')),
        'collections': [{'name': c.name, 'created': when(c.created),
                         'papers': [{'paper': p.arxiv_id, 'title': p.title, 'authors': p.authors, 'note': p.note,
                                     'tags': p.tags, 'status': p.reading, 'added': when(p.added)} for p in c.papers.all()]}
                        for c in Collection.objects.filter(user=user).prefetch_related('papers')],
        'saved_searches': [{'name': s.name, 'search': s.params, 'watch': s.watch_enabled, 'created': when(s.created)}
                           for s in SavedSearch.objects.filter(user=user)],
        'digests': [{'name': d.name, 'categories': d.categories, 'keywords': d.keywords, 'frequency': d.frequency,
                     'profile': d.profile, 'authors': d.authors} for d in DigestSubscription.objects.filter(user=user)],
        'followed_papers': [{'paper': f.arxiv_id, 'title': f.title, 'mine': f.mine, 'since': when(f.created)}
                            for f in FollowedPaper.objects.filter(user=user)],
        'scoop_watch': [{'title': p.title, 'description': p.description} for p in WatchedProject.objects.filter(user=user)],
        'letters': list(LetterSubscription.objects.filter(user=user).values('topic', 'lang', 'active')),
        'results': [{'tool': k, 'id': key, 'status': s, 'date': when(d)} for k, key, s, d in
                    Generated.objects.filter(user=user, kind__in=list(RESULT_KINDS)).order_by('pk')
                    .values_list('kind', 'key', 'status', 'created')],
    }
    response = HttpResponse(json.dumps(data, ensure_ascii=False, indent=1), content_type='application/json; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="summarizepaper-%s.json"' % timezone.localdate().isoformat()
    return response
