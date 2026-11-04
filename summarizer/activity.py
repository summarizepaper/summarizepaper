"""What happened on the site on a given day, for the staff page /admin/activite/: the day's numbers
against the day before and the week, every action in time order and per person, the pages they opened
(visits.py) and the tools they used, what needs a look, and how far the assistant's answers hold against
the papers. The team's own actions are left out unless asked."""
import collections
import datetime
import re
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.db.models import Count, Q
from django.db.models.functions import TruncDate
from django.urls import NoReverseMatch, reverse
from django.utils import timezone, translation

from . import visits
from .models import (AIPrice, AIUsage, AIassistant, Collection, DigestClick, DigestSubscription, FollowedPaper, Generated,
                     LetterSubscription, PageView, PaperHistory, PaperVisit, PremiumGrant, SavedSearch, Search, SummaryPaper,
                     ToolUse, UserPlan, Vote, WatchedProject)

PARIS = ZoneInfo('Europe/Paris')
FEED_LIMIT = 300  # of each kind
VISIT_LIMIT = 2000
SERIES_DAYS = 14
STUCK = datetime.timedelta(minutes=15)
ARXIV_ID = re.compile(r'^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?(/|--)\d{7})(v\d+)?$')  # old ids are stored with '--'

# Generated kinds whose result has its own page; the others are shown on the paper page
RESULT_URLS = {'insight': 'arxividpage', 'literature': 'literature_result', 'missed': 'missed_result', 'students': 'student_result',
               'conference': 'conference_result', 'teams': 'teams_result', 'refcheck': 'refcheck_result'}
TOOL_NAMES = {
    'presskit': 'Kit presse', 'presskit_status': 'Kit presse (statut)', 'presskit_share': 'Kit presse (partage)', 'presskit_experts': 'Kit presse (contacts par pays)', 'slides': 'Journal club',
    'slides_edit': 'Journal club (édition)', 'slides_v3': 'Journal club', 'slides_coll': 'Journal club (collection)',
    'slides_edit_v3': 'Journal club (édition)', 'slides_share': 'Journal club (partage)', 'literature': 'Revue de littérature', 'missed': 'Qu’ai-je manqué ?',
    'refcheck': 'Vérification de bibliographie', 'students': 'Stage ou thèse', 'teams': 'Qui travaille dessus ?',
    'conference': 'Compagnon de conférence', 'reading': 'Niveau de lecture', 'compare': 'Comparaison',
    'columns': 'Tableau de collection', 'questions': 'Questions ouvertes', 'changes': 'Changements de version',
    'insight': 'Éclairage', 'journals': 'Revues', 'reviews': 'Revues', 'review': 'Relecture', 'zotero': 'Zotero',
}

# key: (label, icon) — the icons are Material Symbols, as in the Unfold sidebar
KINDS = {
    'signup': ('Inscription', 'person_add'),
    'premium': ('Premium', 'workspace_premium'),
    'summary': ('Résumé demandé', 'description'),
    'new_summary': ('Nouveau résumé', 'note_add'),
    'question': ('Question', 'forum'),
    'search': ('Recherche', 'search'),
    'tool': ('Outil', 'build'),
    'vote': ('Vote', 'thumbs_up_down'),
    'follow': ('Abonnement', 'notifications'),
    'collection': ('Collection', 'folder'),
    'tool_open': ('Résultat rouvert', 'history'),
    'paper_open': ('Article ouvert', 'article'),
    'digest_click': ('Clic digest', 'ads_click'),
    'visit': ('Page vue', 'visibility'),
}


def day_bounds(day):
    """The start and end, as aware datetimes, of a calendar day in Paris."""
    start = datetime.datetime.combine(day, datetime.time.min, tzinfo=PARIS)
    return start, start + datetime.timedelta(days=1)


def today():
    return timezone.now().astimezone(PARIS).date()


def _paper(arxiv_id):
    return '/en/paper/%s/' % arxiv_id if arxiv_id and ARXIV_ID.match(arxiv_id) else ''


def _user_url(user):
    return reverse('admin:auth_user_change', args=[user.pk]) if user else ''


def _who(user):
    return (user.email or user.username) if user else ''


def _tool_url(row):
    name = RESULT_URLS.get(row.kind)
    if name:
        try:
            with translation.override(row.lang):
                return reverse(name, args=[row.key])
        except NoReverseMatch:
            return ''
    return _paper(row.key.split(':')[0])


STAFF_FIELDS = {User: 'is_staff', AIassistant: 'user__is_staff', DigestClick: 'subscription__user__is_staff'}


def _without_staff(query, staff=False):
    """The query without the team's own actions, unless staff (anonymous actions stay)."""
    if staff:
        return query
    field = STAFF_FIELDS.get(query.model, 'user__is_staff' if 'user' in {f.name for f in query.model._meta.fields} else None)
    return query.exclude(**{field: True}) if field else query


def _daily(query, field='created', distinct=None):
    """{Paris date: number of rows}, or of distinct values of `distinct`."""
    count = Count(distinct, distinct=True) if distinct else Count('pk')
    return {row['d']: row['n'] for row in query.annotate(d=TruncDate(field, tzinfo=PARIS)).values('d').annotate(n=count).order_by()}


def _series(start, end, staff=False):
    """{key: {Paris date: number}} of the actions of each kind between start and end."""
    span = {'created__gte': start, 'created__lt': end}
    rows = lambda model, **extra: _without_staff(model.objects.filter(**span, **extra), staff)
    views = _visits(start, end, staff)
    premium = collections.Counter(_daily(rows(UserPlan, paid=True)))
    premium.update(_daily(rows(PremiumGrant)))
    return {
        'visitors': _daily(views, distinct='visitor'),
        'robots': _daily(PageView.objects.filter(**span).exclude(visitor__in=_people(start, end)), distinct='visitor'),
        'views': _daily(views),
        'signup': _daily(_without_staff(User.objects.filter(date_joined__gte=start, date_joined__lt=end), staff), 'date_joined'),
        'summary': _daily(rows(PaperHistory)),
        'new_summary': _daily(SummaryPaper.objects.filter(**span)),
        'question': _daily(rows(AIassistant)),
        'search': _daily(rows(Search)),
        'tool': _daily(rows(Generated)),
        'premium': dict(premium),
    }


def _people(start, end):
    """The visitors who are people: they moved on a page (visits.seen) or were signed in."""
    return PageView.objects.filter(Q(human=True) | Q(user__isnull=False), created__gte=start, created__lt=end).values('visitor')


def _visits(start, end, staff=False):
    """The pages people opened between start and end, without the team's unless staff."""
    views = _without_staff(PageView.objects.filter(created__gte=start, created__lt=end), staff).filter(visitor__in=_people(start, end))
    return views if staff else views.exclude(visitor__in=_staff_visitors(start, end))


def _staff_visitors(start, end):
    """The visitors of the team: their pages while signed out, in the same browser and day, are theirs too."""
    return set(PageView.objects.filter(created__gte=start, created__lt=end, user__is_staff=True).values_list('visitor', flat=True))


def _cost(first, last):
    prices = {p.model: p for p in AIPrice.objects.all()}
    return sum(row.cost(prices) or 0 for row in AIUsage.objects.filter(day__gte=first, day__lte=last))


NUMBER_KEYS = ('visitors', 'views', 'robots', 'signup', 'summary', 'new_summary', 'question', 'search', 'tool', 'premium')
NUMBER_LABELS = {'visitors': ('Visiteurs', 'group'), 'views': ('Pages vues', 'visibility'), 'robots': ('Robots probables', 'smart_toy')}


def numbers(day, staff=False):
    """[{key, label, icon, value, yesterday, week, series}] for the day, the day before, the mean of the 7 days
    before, and the SERIES_DAYS days up to the day ([{day, value}], for the small bar chart of each card)."""
    start, end = day_bounds(day)
    first = day - datetime.timedelta(days=SERIES_DAYS - 1)
    series = _series(day_bounds(min(first, day - datetime.timedelta(days=7)))[0], end, staff)
    rows = []
    for key in NUMBER_KEYS:
        label, icon = NUMBER_LABELS.get(key) or KINDS[key]
        per_day = series[key]
        week = sum(per_day.get(day - datetime.timedelta(days=n), 0) for n in range(1, 8))
        days = [first + datetime.timedelta(days=n) for n in range(SERIES_DAYS)]
        top = max([per_day.get(d, 0) for d in days] + [1])
        rows.append({'key': key, 'label': label, 'icon': icon, 'value': per_day.get(day, 0),
                     'yesterday': per_day.get(day - datetime.timedelta(days=1), 0), 'week': round(week / 7, 1),
                     'series': [{'day': d, 'value': per_day.get(d, 0), 'height': round(100 * per_day.get(d, 0) / top)} for d in days]})
    failed = Generated.objects.filter(created__gte=start, created__lt=end, status='failed').count()
    next(r for r in rows if r['key'] == 'tool')['note'] = '%d en échec' % failed if failed else ''
    rows.append({'key': 'cost', 'label': 'Coût IA', 'icon': 'payments', 'value': '$%.2f' % _cost(day, day),
                 'yesterday': '$%.2f' % _cost(day - datetime.timedelta(days=1), day - datetime.timedelta(days=1)),
                 'week': '$%.2f' % (_cost(day - datetime.timedelta(days=7), day - datetime.timedelta(days=1)) / 7),
                 'note': 'jour UTC'})
    return rows


def _tool(kind):
    """The tool of a Generated kind, as visits.SECTIONS names it: Journal club (édition) -> Journal club."""
    return TOOL_NAMES.get(kind, kind).split(' (')[0]


def feed(day, kind=None, members_only=False, staff=False):
    """The day's actions, latest first: [{time, kind, label, icon, text, detail, href, who, who_url, person, alert,
    anonymous}], at most FEED_LIMIT of each kind (VISIT_LIMIT page views). `person` groups the actions of one
    member, or of one browser for the visitors without an account (u12, v<visitor hash>, '' when unknown)."""
    start, end = day_bounds(day)
    span = {'created__gte': start, 'created__lt': end}
    events = []
    # a browser that signs in during the day: its pages before belong to the member
    members = dict(PageView.objects.filter(**span, user__isnull=False).values_list('visitor', 'user_id'))

    def add(kind_key, when, text, href='', user=None, detail='', alert=False, person=None):
        label, icon = KINDS[kind_key]
        events.append({'time': when.astimezone(PARIS), 'kind': kind_key, 'label': label, 'icon': icon, 'text': text,
                       'detail': detail, 'href': href, 'who': _who(user), 'who_url': _user_url(user), 'alert': alert,
                       'person': person if person is not None else ('u%d' % user.pk if user else ''),
                       'anonymous': user is None and kind_key != 'new_summary'})

    def rows(model, limit=FEED_LIMIT, exclude=None, **extra):
        query = _without_staff(model.objects.filter(**span, **extra), staff).exclude(**exclude or {})
        if members_only and 'user' in {f.name for f in model._meta.fields}:
            query = query.filter(user__isnull=False)
        return query.order_by('-created')[:limit]

    wanted = lambda k: kind in (None, '', k)
    if wanted('signup'):
        for u in _without_staff(User.objects.filter(date_joined__gte=start, date_joined__lt=end), staff).order_by('-date_joined')[:FEED_LIMIT]:
            add('signup', u.date_joined, _who(u), _user_url(u), u, '' if u.is_active else 'non activé')
    if wanted('premium'):
        for p in rows(UserPlan, paid=True).select_related('user'):
            add('premium', p.created, 'Abonnement %s' % p.get_tier_display(), _user_url(p.user), p.user)
        for g in rows(PremiumGrant).select_related('user'):
            add('premium', g.created, 'Accès offert : %s' % g.get_tool_display(),
                reverse('admin:summarizer_premiumgrant_change', args=[g.pk]), g.user, g.note[:80])
    if wanted('summary'):
        for h in rows(PaperHistory).select_related('user'):
            add('summary', h.created, h.arxiv_id, _paper(h.arxiv_id), h.user, h.lang)
    if wanted('new_summary'):
        for s in rows(SummaryPaper).select_related('paper').defer('summary', 'notes', 'lay_summary', 'blog', 'keywords'):
            add('new_summary', s.created, s.paper.title or s.paper.arxiv_id, _paper(s.paper.arxiv_id), None,
                '%s · %s' % (s.paper.arxiv_id, s.lang))
    if wanted('question'):
        for q in rows(AIassistant).select_related('user').defer('response', 'evidence'):
            add('question', q.created, q.query or '', _paper(q.arxiv_id), q.user, q.arxiv_id)
    if wanted('search'):
        for s in rows(Search).select_related('user'):
            add('search', s.created, s.query or '', '/%s/search-results/' % s.lang, s.user)
    if wanted('tool'):
        for g in rows(Generated).select_related('user').defer('content'):
            state = {'failed': 'échec : %s' % g.error[:120], 'pending': 'en cours %s' % g.step, 'no_source': 'source indisponible'}.get(g.status, '')
            add('tool', g.created, '%s · %s' % (TOOL_NAMES.get(g.kind, g.kind), g.key[:60]), _tool_url(g), g.user,
                state, g.status == 'failed')
    if wanted('tool_open'):
        # the results someone opened again: those they started today are already a 'tool' action
        uses = _without_staff(ToolUse.objects.filter(updated__gte=start, updated__lt=end), staff) \
            .select_related('user', 'row').defer('row__content').order_by('-updated')[:FEED_LIMIT]
        for use in uses:
            if use.row.user_id == use.user_id and use.row.created >= start:
                continue
            add('tool_open', use.updated, '%s · %s' % (TOOL_NAMES.get(use.row.kind, use.row.kind), use.title[:80] or use.row.key[:60]),
                use.url if use.url.startswith('/') else _tool_url(use.row), use.user, 'fait le %s' % use.row.created.astimezone(PARIS).strftime('%d/%m'))
    if wanted('paper_open'):
        opened = _without_staff(PaperVisit.objects.filter(visited__gte=start, visited__lt=end), staff) \
            .select_related('user', 'paper').order_by('-visited')[:FEED_LIMIT]
        for v in opened:
            title = (v.paper.title if v.paper else '') or v.title or v.arxiv_id
            add('paper_open', v.visited, title, _paper(v.arxiv_id), v.user, v.arxiv_id)
    if wanted('digest_click'):
        clicks = _without_staff(DigestClick.objects.filter(**span, burst=False), staff) \
            .select_related('subscription__user').order_by('-created')[:FEED_LIMIT]
        for c in clicks:
            add('digest_click', c.created, c.arxiv_id, _paper(c.arxiv_id), c.subscription.user,
                '%s · rang %s' % (c.target, c.rank or 'hors liste'))
    if wanted('vote'):
        for v in _without_staff(Vote.objects.filter(created_at__gte=start, created_at__lt=end), staff).select_related('paper', 'user') \
                .order_by('-created_at')[:FEED_LIMIT]:
            if members_only and not v.user_id:
                continue
            add('vote', v.created_at, '%s %s' % ('👍' if v.vote > 0 else '👎', v.paper.arxiv_id), _paper(v.paper.arxiv_id), v.user)
    if wanted('follow'):
        for d in rows(DigestSubscription).select_related('user'):
            add('follow', d.created, 'Digest : %s' % (d.name or d.categories[:80]), '', d.user)
        for l in rows(LetterSubscription).select_related('user'):
            add('follow', l.created, 'Lettre : %s' % l.topic, '', l.user)
        for s in rows(SavedSearch).select_related('user'):
            add('follow', s.created, 'Recherche enregistrée : %s' % s.name, '', s.user)
        for w in rows(WatchedProject).select_related('user'):
            add('follow', w.created, 'Veille anti-scoop : %s' % w.title[:80], '', w.user)
        for f in rows(FollowedPaper).select_related('user'):
            add('follow', f.created, 'Alerte de citation : %s' % (f.title[:80] or f.arxiv_id), _paper(f.arxiv_id), f.user)
    if wanted('collection'):
        for c in rows(Collection).select_related('user'):
            add('collection', c.created, c.name, '', c.user)
    if wanted('visit'):
        for v in _visits(start, end, staff).filter(**({'user__isnull': False} if members_only else {})) \
                .select_related('user').order_by('-created')[:VISIT_LIMIT]:
            where = ' · '.join(filter(None, [v.referrer and 'depuis %s' % v.referrer, v.source and 'utm %s' % v.source,
                                             v.first and 'première page']))
            text = visits.section(v.view) + (' · « %s »' % v.query if v.query else '')
            href = v.path if v.path.startswith('/') and not v.path.startswith('//') else ''
            person = 'u%d' % v.user_id if v.user_id else ('u%d' % members[v.visitor] if v.visitor in members else 'v' + v.visitor)
            add('visit', v.created, text, href, v.user, where, person=person)
    events.sort(key=lambda e: e['time'], reverse=True)
    return events


def sources(day, staff=False):
    """Where the day's visitors came from: [(label, visitors)] for the other sites, the utm sources and the
    section of their first page."""
    start, end = day_bounds(day)
    views = _visits(start, end, staff)
    referrers, utm, entries, seen = collections.Counter(), collections.Counter(), collections.Counter(), set()
    for visitor, view, referrer, source in views.order_by('created').values_list('visitor', 'view', 'referrer', 'source'):
        if referrer:
            referrers[referrer] += 1
        if source:
            utm[source] += 1
        if visitor not in seen:
            seen.add(visitor)
            entries[visits.section(view)] += 1
            if not referrer and not source:
                referrers['(direct ou inconnu)'] += 1
    top = lambda counter: counter.most_common(12)
    return {'referrers': top(referrers), 'utm': top(utm), 'entries': top(entries)}


def tools(day, staff=False):
    """The use of each tool on the day and over the 7 days up to it: {'day': rows, 'week': rows}, rows
    [{tool, views, visitors, uses, failed, reopened, members}] with the most used first. A use is a result
    made (Generated), a question, a search or a summary request; reopened, a result opened again (ToolUse)."""
    def table(start, end):
        span = {'created__gte': start, 'created__lt': end}
        rows = collections.defaultdict(lambda: {'views': 0, 'visitors': set(), 'uses': 0, 'failed': 0, 'reopened': 0, 'members': set()})
        views = _visits(start, end, staff)
        for view, visitor, user in views.values_list('view', 'visitor', 'user_id').iterator():
            row = rows[visits.section(view)]
            row['views'] += 1
            row['visitors'].add(visitor)
            if user:
                row['members'].add(user)
        for kind, status, user in _without_staff(Generated.objects.filter(**span), staff).values_list('kind', 'status', 'user_id'):
            row = rows[_tool(kind)]
            row['uses'] += 1
            row['failed'] += status == 'failed'
            if user:
                row['members'].add(user)
        uses = _without_staff(ToolUse.objects.filter(updated__gte=start, updated__lt=end), staff)
        for kind, user, maker, made in uses.values_list('row__kind', 'user_id', 'row__user_id', 'row__created'):
            if maker == user and made >= start:
                continue
            rows[_tool(kind)]['reopened'] += 1
            rows[_tool(kind)]['members'].add(user)
        for model, tool in ((AIassistant, 'Assistant'), (Search, 'Recherche'), (PaperHistory, 'Article')):
            for user in _without_staff(model.objects.filter(**span), staff).values_list('user_id', flat=True).iterator():
                rows[tool]['uses'] += 1
                if user:
                    rows[tool]['members'].add(user)
        out = [dict(r, tool=tool, visitors=len(r['visitors']), members=len(r['members'])) for tool, r in rows.items()]
        return sorted(out, key=lambda r: (-(r['uses'] + r['reopened']), -r['views'], r['tool']))

    end = day_bounds(day)[1]
    return {'day': table(day_bounds(day)[0], end), 'week': table(day_bounds(day - datetime.timedelta(days=6))[0], end)}


def watch():
    """What needs a look now: tools that failed in the last 24 hours, and tools stuck in 'pending'."""
    now = timezone.now()
    failed = Generated.objects.filter(status='failed', updated__gte=now - datetime.timedelta(days=1)) \
        .select_related('user').defer('content').order_by('-updated')[:30]
    stuck = Generated.objects.filter(status='pending', updated__lt=now - STUCK, updated__gte=now - datetime.timedelta(days=2)) \
        .select_related('user').defer('content').order_by('-updated')[:30]
    as_rows = lambda query: [{'time': g.updated.astimezone(PARIS), 'tool': TOOL_NAMES.get(g.kind, g.kind), 'key': g.key[:60],
                              'href': _tool_url(g), 'who': _who(g.user), 'who_url': _user_url(g.user),
                              'detail': g.error[:200] or g.step} for g in query]
    unavailable = Generated.objects.filter(status='no_source', updated__gte=now - datetime.timedelta(days=1)) \
        .select_related('user').defer('content').order_by('-updated')[:30]
    no_source = as_rows(unavailable)
    for row in no_source:
        row['detail'] = 'Aucun abstract ni texte intégral autorisé disponible ; aucune génération effectuée.'
    return {'failed': as_rows(failed), 'stuck': as_rows(stuck), 'unavailable': no_source}


# an answer about several papers names each by its tag, such as [Smith et al. 2021] (assistant.MULTI_INSTRUCTIONS)
TAG = re.compile(r'\[[^\[\]\n]{2,80}?\d{4}[a-z]?\]')
QUALITY_DAYS = 14
CLAIM_STATUSES = ('verified', 'not_found', 'unsupported', 'general', 'visual')


def _tally():
    return collections.Counter()


def _count(tally, row):
    tally['answers'] += 1
    evidence = row.evidence or {}
    if evidence.get('status') == 'done':
        tally['checked'] += 1
        for claim in evidence.get('claims') or []:
            tally['claims'] += 1
            tally[claim.get('status') if claim.get('status') in CLAIM_STATUSES else 'unsupported'] += 1
            if claim.get('numbers_absent'):
                tally['numbers'] += 1
    if ',' in (row.papers or ''):
        tally['multi'] += 1
        tally['tagged'] += bool(TAG.search(row.response or ''))


def _shares(label, tally):
    def share(part, whole):
        return '%d %%' % round(100 * tally[part] / tally[whole]) if tally[whole] else '—'
    return {'label': label, 'answers': tally['answers'], 'checked': share('checked', 'answers'), 'claims': tally['claims'],
            'verified': share('verified', 'claims'), 'not_found': share('not_found', 'claims'),
            'unsupported': share('unsupported', 'claims'), 'general': share('general', 'claims'),
            'numbers': share('numbers', 'claims'), 'multi': tally['multi'], 'tagged': share('tagged', 'multi')}


def answer_quality(day, days=QUALITY_DAYS):
    """How far the answers of the assistant hold against the papers, over the `days` days up to `day`, per day
    and per model: the share of answers whose claims were checked, of claims found word for word in a paper,
    not found, unsupported or from general knowledge, of claims with numbers the papers never state, and of
    answers about several papers that name them by their tags."""
    start, end = day_bounds(day - datetime.timedelta(days=days - 1))[0], day_bounds(day)[1]
    per_day, per_model, total = collections.defaultdict(_tally), collections.defaultdict(_tally), _tally()
    rows = AIassistant.objects.filter(created__gte=start, created__lt=end, kind='answer') \
        .only('created', 'evidence', 'papers', 'response', 'model')
    for row in rows.iterator():
        for tally in (per_day[row.created.astimezone(PARIS).date()], per_model[row.model or '(non enregistré)'], total):
            _count(tally, row)
    return {'total': _shares('%d jours' % days, total),
            'days': [_shares(d.strftime('%d/%m'), per_day[d]) for d in sorted(per_day, reverse=True)],
            'models': [_shares(m, per_model[m]) for m in sorted(per_model, key=lambda m: -per_model[m]['answers'])]}
