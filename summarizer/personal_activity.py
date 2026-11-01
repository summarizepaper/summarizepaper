"""A researcher's own space: what they read, asked and made on the site, and what changed since.

- Overview: news about the papers they read or follow (retractions, publications, new versions), their results
  in progress, their to-read pile, papers close to their recent reading and the watches they keep.
- Readings: every paper opened while signed in or summarized at their request (PaperVisit), with its authors,
  year, venue, editorial status and collections; a reading status; actions on several papers at once.
- Conversations and Results (literature reviews, bibliography checks...), each linking back to its tool.

Never changes generation quota records (PaperHistory): taking a paper out of the history only hides it.
Everything shown comes from stored data: the page never waits for an API.
"""
import json
import re
import types
from datetime import timedelta
from urllib.parse import urlencode, unquote, urlsplit

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Max, OuterRef, Q, Subquery
from django.db.models.functions import Coalesce, Greatest
from django.http import Http404, HttpResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import get_language, gettext as _, gettext_lazy, ngettext
from django.views.decorators.http import require_POST

from . import sources

from .models import (AIassistant, ArxivPaper, ChatConversation, ChatShare, Collection, CollectionPaper,
                     DigestSubscription, FollowedPaper, Generated, LetterSubscription, PaperAuthor, PaperHistory,
                     PaperScore, PaperVisit, PublicationAlert, ReferenceWatch, SavedSearch, TopicMapWatch,
                     VersionAlert, WatchedProject)

PER_PAGE = 20
NEWS_DAYS = 90  # alerts of followed papers older than this are no longer news
NEWS_SCAN = 200  # recent readings whose stored facts are looked at for news
MATCHED_PAPERS = 5000  # papers matching a conversation search, at most
STATUSES = ('to_read', 'read', 'key')
TABS = ('overview', 'readings', 'conversations', 'results')


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

def record_visit(user, arxiv_id, title=''):
    """A paper opened while signed in, summarized or not yet (then known by its id and the title its page showed)."""
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None and not title:
        return
    defaults = {'visited': timezone.now(), 'hidden': False, 'paper': paper}
    if title:
        defaults['title'] = ' '.join(str(title).split())[:500]
    PaperVisit.objects.update_or_create(user=user, arxiv_id=arxiv_id, defaults=defaults)


def record_request(user, arxiv_id):
    """A summary requested while signed in: the paper is in the history even if its page was not opened."""
    if user is None or not user.is_authenticated:
        return
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    visit, created = PaperVisit.objects.get_or_create(user=user, arxiv_id=arxiv_id, defaults={
        'visited': None, 'requested': timezone.now(), 'paper': paper})
    if not created:
        PaperVisit.objects.filter(pk=visit.pk).update(requested=timezone.now(), hidden=False,
                                                      paper=paper or visit.paper)


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
        unsummarized = Q()
        for word in text.split()[:12]:
            unsummarized &= Q(title__icontains=word)
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
    """The user's visible readings, newest first, each with `date`: its last opening or summary request."""
    return (PaperVisit.objects.filter(user=user, hidden=False)
            .annotate(date=Coalesce(Greatest('visited', 'requested'), 'visited', 'requested'))
            .filter(date__isnull=False).order_by('-date', '-pk'))


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
    facts = collection_tools.facts(visits)
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
            'reading': visit.reading, 'reading_label': labels.get(visit.reading, ''),
            'date': visit.date, 'visited': visit.visited is not None and visit.visited == visit.date,
        })
    return items


# --- overview ---------------------------------------------------------------------------------

def news(user):
    """[{id, title, url, tone, news: [{label, tone, link, link_label}]}] of what changed about the papers the user read or follows:
    retractions and corrections, journal publication since they read the preprint, newer arXiv versions."""
    from . import collection_tools, publication_status
    visits = list(readings(user).select_related('paper')[:NEWS_SCAN])
    facts = collection_tools.facts(visits)
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
                      'tone': tone, 'date': date, 'link': link, 'link_label': link_label})

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


def describe_results(rows):
    lang = get_language()
    items = []
    for row in rows:
        name, path, title = RESULT_KINDS[row.kind]
        data = row.data() if row.status == 'done' and row.content else {}
        if row.status == 'pending':
            state = _('In progress…')
        elif row.status == 'done':
            state = ''
        else:
            state = _('Failed')
        items.append({'tool': name, 'title': (title(data) if data else '') or name, 'state': state,
                      'url': '/%s/%s/%s/' % (lang, path, row.key), 'date': row.created, 'status': row.status})
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


def overview(request):
    user = request.user
    recent = list(readings(user).select_related('paper')[:5])
    read_ids = list(readings(user).values_list('arxiv_id', flat=True)[:500])
    to_read = readings(user).filter(reading='to_read')
    found = news(user)
    kept = watches(user)
    return {
        'news': found[:8], 'news_more': max(len(found) - 8, 0),
        'recent_results': describe_results(results(user)[:4]),
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
    if tab == 'readings':
        rows = readings(request.user).select_related('paper')
        if status in STATUSES:
            rows = rows.filter(reading=status)
        elif status == 'unsorted':
            rows = rows.filter(reading='')
        elif status == 'unsummarized':
            rows = rows.filter(paper__isnull=True)
        else:
            status = ''
        if query:
            rows = visit_matches(rows, query)
        context['collections'] = editable_collections(request.user)
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
                                                   ('page', number)) if v})
    context.update({
        'items': items, 'pagination': result, 'status': status, 'kind': kind,
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


@login_required(login_url='login')
@require_POST
def selection(request):
    """What a reader does with papers of their history: one (a button of a card) or the ones they ticked."""
    from . import access, exports, library
    one = request.POST.get('one', '')
    action, _sep, pk = one.partition(':') if one else (request.POST.get('action', ''), '', '')
    pks = [pk] if pk else request.POST.getlist('v')[:PER_PAGE]
    visits = list(PaperVisit.objects.filter(user=request.user, pk__in=[p for p in pks if p.isdigit()]).select_related('paper'))
    if not visits:
        return back(request, 'none')
    ids = [v.arxiv_id for v in visits]
    if action in STATUSES or action == 'clear':
        PaperVisit.objects.filter(pk__in=[v.pk for v in visits]).update(reading='' if action == 'clear' else action)
        return back(request, 'status')
    if action == 'hide':
        PaperVisit.objects.filter(pk__in=[v.pk for v in visits]).update(hidden=True)
        return back(request, 'hidden')
    if action == 'chat':
        from .assistant import MAX_PAPERS
        return redirect(reverse('chat') + '?' + urlencode({'ids': ','.join(ids[:MAX_PAPERS])}))
    if action in ('bib', 'ris'):
        metas = [m for m in (exports.paper_meta(v.arxiv_id, types.SimpleNamespace(title=v.title or v.arxiv_id, authors=''))
                             for v in visits) if m]
        return exports.export_many(metas, action, 'my-readings', get_language())
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
        return redirect('/%s/collections/%s/?%s#papers' % (get_language(), collection.pk, urlencode(counts)))
    raise Http404


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
                      'last_opened': when(v.visited), 'summary_requested': when(v.requested),
                      'status': v.reading, 'hidden': v.hidden}
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
