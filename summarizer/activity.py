"""What happened on the site on a given day, for the staff page /admin/activite/: the day's numbers
against the day before and the week, every action in time order, what needs a look, and how far the
assistant's answers hold against the papers."""
import collections
import datetime
import re
from zoneinfo import ZoneInfo

from django.contrib.auth.models import User
from django.urls import NoReverseMatch, reverse
from django.utils import timezone, translation

from .models import (AIPrice, AIUsage, AIassistant, Collection, DigestSubscription, FollowedPaper, Generated,
                     LetterSubscription, PaperHistory, PremiumGrant, SavedSearch, Search, SummaryPaper, UserPlan,
                     Vote, WatchedProject)

PARIS = ZoneInfo('Europe/Paris')
FEED_LIMIT = 300
STUCK = datetime.timedelta(minutes=15)
ARXIV_ID = re.compile(r'^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?(/|--)\d{7})(v\d+)?$')  # old ids are stored with '--'

# Generated kinds whose result has its own page; the others are shown on the paper page
RESULT_URLS = {'insight': 'arxividpage', 'literature': 'literature_result', 'missed': 'missed_result', 'students': 'student_result',
               'conference': 'conference_result', 'teams': 'teams_result', 'refcheck': 'refcheck_result'}
TOOL_NAMES = {
    'presskit': 'Kit presse', 'presskit_status': 'Kit presse (statut)', 'presskit_share': 'Kit presse (partage)', 'presskit_experts': 'Kit presse (contacts par pays)', 'slides': 'Journal club',
    'slides_edit': 'Journal club (édition)', 'literature': 'Revue de littérature', 'missed': 'Qu’ai-je manqué ?',
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


def _counted(start, end):
    """Number of actions of each kind between start and end."""
    paid = UserPlan.objects.filter(paid=True, created__gte=start, created__lt=end).count()
    return {
        'signup': User.objects.filter(date_joined__gte=start, date_joined__lt=end).count(),
        'summary': PaperHistory.objects.filter(created__gte=start, created__lt=end).count(),
        'new_summary': SummaryPaper.objects.filter(created__gte=start, created__lt=end).count(),
        'question': AIassistant.objects.filter(created__gte=start, created__lt=end).count(),
        'search': Search.objects.filter(created__gte=start, created__lt=end).count(),
        'tool': Generated.objects.filter(created__gte=start, created__lt=end).count(),
        'premium': paid + PremiumGrant.objects.filter(created__gte=start, created__lt=end).count(),
    }


def _cost(first, last):
    prices = {p.model: p for p in AIPrice.objects.all()}
    return sum(row.cost(prices) or 0 for row in AIUsage.objects.filter(day__gte=first, day__lte=last))


def numbers(day):
    """[{key, label, icon, value, yesterday, week}] for the day, the day before and the mean of the 7 days before."""
    start, end = day_bounds(day)
    week_start = start - datetime.timedelta(days=7)
    now, before, week = _counted(start, end), _counted(start - datetime.timedelta(days=1), start), _counted(week_start, start)
    rows = []
    for key in ('signup', 'summary', 'new_summary', 'question', 'search', 'tool', 'premium'):
        label, icon = KINDS[key]
        rows.append({'key': key, 'label': label, 'icon': icon, 'value': now[key], 'yesterday': before[key],
                     'week': round(week[key] / 7, 1)})
    failed = Generated.objects.filter(created__gte=start, created__lt=end, status='failed').count()
    rows[5]['note'] = '%d en échec' % failed if failed else ''
    rows.append({'key': 'cost', 'label': 'Coût IA', 'icon': 'payments', 'value': '$%.2f' % _cost(day, day),
                 'yesterday': '$%.2f' % _cost(day - datetime.timedelta(days=1), day - datetime.timedelta(days=1)),
                 'week': '$%.2f' % (_cost(day - datetime.timedelta(days=7), day - datetime.timedelta(days=1)) / 7),
                 'note': 'jour UTC'})
    return rows


def feed(day, kind=None, members_only=False):
    """The day's actions, latest first: [{time, kind, label, icon, text, detail, href, who, who_url, alert, anonymous}]."""
    start, end = day_bounds(day)
    span = {'created__gte': start, 'created__lt': end}
    events = []

    def add(kind_key, when, text, href='', user=None, detail='', alert=False):
        label, icon = KINDS[kind_key]
        events.append({'time': when.astimezone(PARIS), 'kind': kind_key, 'label': label, 'icon': icon, 'text': text,
                       'detail': detail, 'href': href, 'who': _who(user), 'who_url': _user_url(user), 'alert': alert,
                       'anonymous': user is None and kind_key != 'new_summary'})

    def rows(model, **extra):
        query = model.objects.filter(**span, **extra)
        if members_only and 'user' in {f.name for f in model._meta.fields}:
            query = query.filter(user__isnull=False)
        return query.order_by('-created')[:FEED_LIMIT]

    wanted = lambda k: kind in (None, '', k)
    if wanted('signup'):
        for u in User.objects.filter(date_joined__gte=start, date_joined__lt=end).order_by('-date_joined')[:FEED_LIMIT]:
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
    if wanted('vote'):
        for v in Vote.objects.filter(created_at__gte=start, created_at__lt=end).select_related('paper', 'user') \
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
    events.sort(key=lambda e: e['time'], reverse=True)
    return events[:FEED_LIMIT]


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
