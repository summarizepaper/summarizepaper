"""Pages of the rising topics (see trends.py): the topics of a week, the history of terms, and their papers."""
import csv
import datetime
import json
from urllib.parse import urlencode

import numpy as np
from django.http import Http404, HttpResponse
from django.shortcuts import render
from django.utils import formats
from django.utils import timezone
from django.utils.html import escape
from django.utils.translation import get_language, gettext as _, gettext_lazy, ngettext

from . import access, trends
from .digest import THRESHOLDS, archive_of
from .models import DigestSubscription, RisingTopics, TermWeek

MAX_TERMS = 3  # compared at once; the first three hues of the palette stay apart for every reader
FREE_TERMS = 1
COLORS = ('#2a78d6', '#eb6834', '#1baf7a')
SMOOTH = 4  # weeks averaged on the chart
RECENT_WEEKS = 13  # the verdict compares the last three months with the three months before
MIN_CHANGE = 1.2
VERDICT_LLR = 10.8  # about p < 0.001
VERDICT_MIN = 10  # papers in the two periods, below which nothing can be told
EXAMPLES = ('sparse autoencoder', 'mixture of experts', 'state space model', 'JWST', 'quantum error correction')


def _week(value):
    try:
        return trends.week_of(datetime.date.fromisoformat(value or ''))
    except ValueError:
        return None


def _date(day):
    return formats.date_format(day, 'j F Y')


def sparkline(values, width=180, height=36):
    """A small SVG of a topic's weekly share, the last week marked."""
    if len(values) < 2:
        return ''
    top = max(values) or 1
    step = width / (len(values) - 1)
    points = [(round(i * step, 1), round(height - 3 - (v / top) * (height - 8), 1)) for i, v in enumerate(values)]
    dots = ''.join('<circle cx="%s" cy="%s" r="5" fill="transparent"><title>%s</title></circle>'
                   % (x, y, escape(_('%(n)s per 10,000 papers') % {'n': v})) for (x, y), v in zip(points, values))
    x, y = points[-1]
    return ('<svg class="spark" viewBox="-4 0 %d %d" width="%d" height="%d" role="img" aria-label="%s">'
            '<line x1="0" x2="%d" y1="%d" y2="%d" class="spark-base"/>'
            '<polyline points="%s" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
            '<circle cx="%s" cy="%s" r="4" fill="var(--accent)" stroke="var(--surface)" stroke-width="2"/>%s</svg>'
            % (width + 8, height, width + 8, height, escape(_('Share of papers using the term, week by week')),
               width, height - 1, height - 1, ' '.join('%s,%s' % p for p in points), x, y, dots))


KINDS = {
    'new': (gettext_lazy('New'), gettext_lazy('Almost absent from these papers before: first counted the week of %(since)s.')),
    'growing': (gettext_lazy('Rising for %(weeks)d weeks'), gettext_lazy('Up week after week, not a one-week event.')),
    'recurring': (gettext_lazy('Recurring'), gettext_lazy('Already this frequent the week of %(since)s, a year ago: perhaps a yearly cycle (a conference deadline, a season, an observing campaign).')),
    'comeback': (gettext_lazy('Comeback'), gettext_lazy('Already this frequent the week of %(since)s.')),
    'spike': (gettext_lazy('Sudden rise'), gettext_lazy('New this week: often an event, a release, a data set or a deadline. The coming weeks will tell if it lasts.')),
}


def decorate(topic):
    topic = dict(topic)
    kind = topic.get('kind') or 'spike'
    since = datetime.date.fromisoformat(topic['since']) if topic.get('since') else None
    label, note = KINDS.get(kind, KINDS['spike'])
    values = {'weeks': topic.get('weeks') or 1, 'since': _date(since) if since else ''}
    topic.update(kind=kind, kind_label=label % values, kind_note=note % values, spark=sparkline(topic.get('series') or []),
                 chat_ids=','.join(p['arxiv_id'] for p in topic.get('papers', [])),
                 query=topic.get('query') or topic['title'], explore=urlencode([('t', t) for t in topic['terms'][:MAX_TERMS]]))
    return topic


def scope_links(lang, day, scope):
    """The scopes with topics this week: all of arXiv, the archives and servers, then the categories of the open one."""
    written = set(RisingTopics.objects.filter(lang=lang, day=day).exclude(content__regex=r'"topics"\s*:\s*\[\s*\]').values_list('scope', flat=True))
    top = sorted((s for s in written if s != '*' and '.' not in s and s in trends_scopes_top()), key=trends.scope_name)
    archive = archive_of(scope) if scope != '*' else None
    inner = sorted((s for s in written if '.' in s and archive_of(s) == archive), key=trends.scope_name)
    return [{'scope': s, 'name': trends.scope_name(s)} for s in top], [{'scope': s, 'name': trends.scope_name(s)} for s in inner]


def trends_scopes_top():
    from .digest import ARCHIVES, BIO_ARCHIVES
    return set(ARCHIVES + BIO_ARCHIVES)


def for_you(user, lang):
    """[(score, scope, topic)] of the user's fields this week, closest to their research profile first."""
    subscriptions = list(DigestSubscription.objects.filter(user=user, active=True))
    categories = [c for s in subscriptions for c in s.categories.split(',') if c]
    profile = None
    for subscription in subscriptions:
        if subscription.profile_vector and subscription.profile_labels:
            labels = json.loads(subscription.profile_labels)
            rows = np.frombuffer(bytes(subscription.profile_vector), dtype=np.float32).reshape(len(labels), -1)
            profile = rows if profile is None else np.vstack([profile, rows])
    scopes = list(dict.fromkeys(categories + [archive_of(c) for c in categories] + ['*']))
    since = datetime.date.today() - 3 * trends.WEEK
    found = {}
    for scope in scopes:
        week, topics = trends.topics_of(scope, lang, since=since)
        for topic in topics:
            topic['week'] = week
            vector = trends.vector_of(topic)
            score = float(np.max(profile @ vector)) if profile is not None and vector is not None and \
                profile.shape[1] == vector.shape[0] else None
            if profile is not None and (score is None or score < THRESHOLDS['low']):
                continue
            if topic['main'] not in found:  # the most specific scope comes first
                found[topic['main']] = (score, scope, topic)
    ranked = sorted(found.values(), key=lambda f: (-(f[0] or 0), -f[2]['stats'][0]['llr'] if f[2].get('stats') else 0))
    return ranked[:12], bool(categories), profile is not None


def trends_page(request):
    paid = access.has_tool(request.user, 'trends')
    lang = get_language()
    scope = request.GET.get('scope', '*')
    if scope != '*' and not RisingTopics.objects.filter(scope=scope).exists():
        raise Http404
    locked = scope != '*' and not paid
    if locked:
        scope = '*'
    view = 'foryou' if request.GET.get('view') == 'foryou' else 'week'
    context = {'paid': paid, 'scope': scope, 'scope_name': trends.scope_name(scope), 'view': view, 'locked': locked, 'day': None}
    if view == 'foryou':
        if paid and request.user.is_authenticated:
            ranked, has_fields, has_profile = for_you(request.user, lang)
            context.update(personal=[dict(decorate(t), scope=s, scope_name=trends.scope_name(s), score=score)
                                     for score, s, t in ranked], has_fields=has_fields, has_profile=has_profile)
        return render(request, 'summarizer/trends.html', context)
    weeks = list(RisingTopics.objects.filter(scope=scope, lang=lang).order_by('-day').values_list('day', flat=True))
    week = _week(request.GET.get('week'))
    if week is not None and week not in weeks:
        raise Http404
    day, topics = trends.topics_of(scope, lang, week=week)
    shown = topics if paid else topics[:trends.FREE_TOPICS]
    first = TermWeek.objects.filter(scope='*').order_by('week').values_list('week', flat=True).first()
    top, inner = scope_links(lang, day, scope) if day else ([], [])
    index = weeks.index(day) if day in weeks else 0
    context.update({
        'result_status': trends.result_state(scope, lang, day).get('status', 'pending') if day else 'pending',
        'day': day, 'day_end': day + datetime.timedelta(days=6) if day else None,
        'topics': [decorate(t) for t in shown], 'hidden': len(topics) - len(shown), 'top_scopes': top, 'inner_scopes': inner,
        'archive': archive_of(scope) if scope != '*' else '', 'archive_name': trends.scope_name(archive_of(scope)) if scope != '*' else '',
        'older': weeks[index + 1] if index + 1 < len(weeks) else None, 'newer': weeks[index - 1] if index > 0 else None,
        'since': first, 'ready_on': first + trends.WEEK * (trends.MIN_BASE_WEEKS + 1) + trends.CLOSE if first else None,
    })
    from . import letters
    letter = letters.latest_for_scope(scope, lang)
    if letter is not None:
        context.update(letter=letter, letter_name=letters.localized_name(letter.topic, lang)[1])
    return render(request, 'summarizer/trends.html', context)


# --- the history of terms -------------------------------------------------------------------------------------

def _smooth(values):
    return [sum(values[max(0, i - SMOOTH + 1):i + 1]) / len(values[max(0, i - SMOOTH + 1):i + 1]) for i in range(len(values))]


def chart(weeks, series, width=760, height=300):
    """An SVG line chart of several terms' weekly share (4-week average), with month ticks and a hover readout."""
    left, right, top, bottom = 44, 12, 12, 28
    plot_w, plot_h = width - left - right, height - top - bottom
    peak = max([v for s in series for v in s['smooth']] + [0.1])
    magnitude = 10 ** int(np.floor(np.log10(peak)))
    step = next(m * magnitude for m in (1, 2, 2.5, 5, 10) if peak / (m * magnitude) <= 5)
    ymax = step * int(np.ceil(peak / step))

    def x(i):
        return round(left + (i / max(len(weeks) - 1, 1)) * plot_w, 1)

    def y(v):
        return round(top + plot_h - v / ymax * plot_h, 1)
    parts = ['<svg viewBox="0 0 %d %d" class="term-chart" role="img" aria-label="%s">'
             % (width, height, escape(_('Share of papers using each term, week by week')))]
    tick = 0
    while tick <= ymax + 1e-9:
        parts.append('<line x1="%d" x2="%d" y1="%s" y2="%s" class="grid"/><text x="%d" y="%s" class="axis" text-anchor="end">%s</text>'
                     % (left, width - right, y(tick), y(tick), left - 6, y(tick) + 4, ('%g' % round(tick, 2))))
        tick += step
    every = max(1, len(weeks) // 8)
    months = [i for i, w in enumerate(weeks) if i == 0 or w.month != weeks[i - 1].month]
    for i in months[::max(1, len(months) // 8 or 1)] if len(weeks) > 8 else range(0, len(weeks), every):
        parts.append('<text x="%s" y="%d" class="axis" text-anchor="middle">%s</text>'
                     % (x(i), height - 8, escape(formats.date_format(weeks[i], 'M Y'))))
    for s in series:
        parts.append('<polyline points="%s" fill="none" stroke="%s" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
                     % (' '.join('%s,%s' % (x(i), y(v)) for i, v in enumerate(s['smooth'])), s['color']))
        if len(series) > 1 and s['smooth']:
            parts.append('<text x="%s" y="%s" class="direct" text-anchor="end">%s</text>'
                         % (x(len(weeks) - 1) - 4, y(s['smooth'][-1]) - 6, escape(s['term'])))
    parts.append('<line class="cross" x1="0" x2="0" y1="%d" y2="%d" visibility="hidden"/>' % (top, top + plot_h))
    parts.append('<rect class="hit" x="%d" y="%d" width="%d" height="%d" fill="transparent"/></svg>' % (left, top, plot_w, plot_h))
    return ''.join(parts), {'left': left, 'width': width, 'plot_w': plot_w}


def verdict(raw):
    """Whether a term's share of papers [(papers, n)] rose over the last three months compared with the three before,
    as for the rising topics: a clear change (×1.2) too large to be chance; with the same weeks a year before."""
    weeks = min(RECENT_WEEKS, len(raw) // 2)
    if weeks < SMOOTH:
        return None
    recent, before = raw[-weeks:], raw[-2 * weeks:-weeks]
    p, n = sum(r[0] for r in recent), sum(r[1] for r in recent)
    p0, n0 = sum(r[0] for r in before), sum(r[1] for r in before)
    if not p or not p0:
        return None
    result = {'weeks': weeks, 'n': n, 'n0': n0, 'share': n * 10000 / p, 'before': n0 * 10000 / p0, 'kind': 'flat'}
    if n + n0 < VERDICT_MIN:
        result['kind'] = 'few'
        return result
    expected = max(n0, 0.5) / p0 * p
    growth = n / expected
    llr = 2 * ((n * np.log(n / expected) if n else 0) - (n - expected))
    if llr >= VERDICT_LLR and growth >= MIN_CHANGE:
        result.update(kind='up', growth=round(growth, 1))
    elif llr >= VERDICT_LLR and growth <= 1 / MIN_CHANGE:
        result['kind'] = 'down'
    if len(raw) >= 52 + weeks:
        year = raw[-52 - weeks:-52]
        papers = sum(r[0] for r in year)
        result['year'] = sum(r[1] for r in year) * 10000 / papers if papers else None
    return result


def trends_explore(request):
    paid = access.has_tool(request.user, 'trends')
    scope = request.GET.get('scope', '*') if paid else '*'
    if scope != '*' and not TermWeek.objects.filter(scope=scope).exists():
        raise Http404
    typed = [t.strip() for t in request.GET.getlist('t') if t.strip()][:MAX_TERMS if paid else FREE_TERMS]
    counted = [(t, trends.normalize(t)) for t in typed]
    wanted = list(dict.fromkeys(n for _, n in counted if n))
    context = {'paid': paid, 'scope': scope, 'scope_name': trends.scope_name(scope), 'typed': typed + [''] * (MAX_TERMS - len(typed)),
               'max_terms': MAX_TERMS if paid else FREE_TERMS, 'refused': [t for t, n in counted if not n],
               'renamed': [(t, n) for t, n in counted if n and n != t.lower()], 'more_terms': len(request.GET.getlist('t')) > len(typed),
               'examples': [(t, urlencode({'t': t})) for t in EXAMPLES], 'smooth': SMOOTH, 'recent_weeks': RECENT_WEEKS}
    if wanted:
        found = trends.history(scope, wanted)
        weeks = [w for w, _, _ in found[wanted[0]]]
        series = []
        for color, term in zip(COLORS, wanted):
            raw = [(p, n) for _, p, n in found[term]]
            shares = [n * 10000 / p if p else 0 for p, n in raw]
            series.append({'term': term, 'color': color, 'raw': raw, 'shares': shares, 'smooth': _smooth(shares),
                           'total': sum(n for _, n in raw), 'verdict': verdict(raw)})
        if request.GET.get('format') == 'csv' and paid:
            response = HttpResponse(content_type='text/csv; charset=utf-8')
            response['Content-Disposition'] = 'attachment; filename="rising-topics-%s.csv"' % scope.replace('*', 'arxiv')
            writer = csv.writer(response)
            writer.writerow(['# Weekly papers (first versions) using each term in their title or abstract, %s, '
                             'from summarizepaper.com/trends; a week with a single paper counts 0.' % scope])
            writer.writerow(['week_start', 'papers_in_scope'] + sum([[s['term'], s['term'] + ' per 10000'] for s in series], []))
            for i, week in enumerate(weeks):
                writer.writerow([week.isoformat(), series[0]['raw'][i][0]] +
                                sum([[s['raw'][i][1], round(s['shares'][i], 2)] for s in series], []))
            return response
        svg, geometry = chart(weeks, series) if len(weeks) > 1 else ('', {})
        context.update(series=series, weeks=weeks, svg=svg, geometry=geometry, rows=[
            {'week': w, 'values': [(s['raw'][i][1], round(s['shares'][i], 1)) for s in series]} for i, w in enumerate(weeks)][::-1],
            points=({'weeks': [_date(w) for w in weeks], 'unit': _('per 10,000 papers, averaged over 4 weeks'), 'series': [
                {'term': s['term'], 'color': s['color'], 'values': [formats.number_format(round(v, 1), 1) for v in s['smooth']],
                 'raw': [ngettext('%(n)s paper that week', '%(n)s papers that week', n) % {'n': n} for _, n in s['raw']]}
                for s in series]}),
            csv_query=request.GET.urlencode() + '&format=csv')
    return render(request, 'summarizer/trends_explore.html', context)


def trends_papers(request):
    """Every kept paper of a week (in a scope) using one of the terms."""
    paid = access.has_tool(request.user, 'trends')
    scope = request.GET.get('scope', '*')
    if scope != '*' and not paid:
        raise Http404
    week = _week(request.GET.get('week'))
    wanted = [trends.normalize(t) for t in request.GET.getlist('t')][:10]
    wanted = [t for t in wanted if t]
    if week is None or not wanted:
        raise Http404
    papers = []
    for paper in trends.stored_papers(week, scope):
        found = trends.terms('%s. %s' % (paper['title'], paper['abstract']))
        matched = [t for t in wanted if t in found]
        if matched:
            papers.append(dict(paper, matched=matched, shown_authors=', '.join(paper['authors'][:4]) +
                               (' et al.' if len(paper['authors']) > 4 else '')))
    papers.sort(key=lambda p: (-len(p['matched']), p['day']))
    return render(request, 'summarizer/trends_papers.html', {
        'scope': scope, 'scope_name': trends.scope_name(scope), 'week': week, 'week_end': week + datetime.timedelta(days=6),
        'terms': wanted, 'papers': papers, 'chat_ids': ','.join(p['key'] for p in papers[:10]), 'kept': week >= trends.closed_week(
            timezone.now()) - trends.WEEK * (trends.KEEP_PAPERS - 1)})
