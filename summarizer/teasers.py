"""The first results of a tool, shown to those who cannot open it yet, the rest greyed out.

Three places use them:
1. the page of a Premium tool shows the start of a real example, a result an administrator marked as one
   (Generated.showcase, only on results of staff accounts: nobody else's search is ever published);
2. a free account runs each tool of TRIAL_TOOLS once, and sees the start of its own result;
3. the paper page shows the people who sign the most papers around it (from its neighbourhood, already computed).

Greyed rows never carry the hidden content (not even in the HTML, nor blurred by CSS): only how many there are.
"""
import re
from collections import Counter

from django.utils import timezone
from django.utils.translation import gettext as _, ngettext

from . import access
from .models import Generated

SHOWN = 2  # results shown in full; the others are counted and greyed
GREYED_MAX = 3  # greyed rows drawn, whatever the count
TRIAL_TOOLS = ('teams', 'missed', 'refcheck')  # the least costly of the Premium tools that read a whole search
SHOWCASE_TOOLS = ('teams', 'missed', 'refcheck', 'literature')
PATHS = {'teams': 'teams', 'missed': 'missed', 'refcheck': 'refcheck', 'literature': 'literature-review'}


# --- who may see what --------------------------------------------------------------

def trial_open(user, kind):
    """A free account that has never run this tool may run it once."""
    if not (kind in TRIAL_TOOLS and user is not None and user.is_authenticated and not access.has_tool(user, kind)):
        return False
    from .generated import STALE
    spent = (Generated.objects.filter(kind=kind, user=user).exclude(status__in=('failed', 'no_source'))
             .exclude(status='pending', updated__lt=timezone.now() - STALE))  # a failed try is not spent
    return not spent.exists()


def mark_trial(user, kind):
    """The search just started by a free account is its free try (called by the tool's page after starting it)."""
    row = Generated.objects.filter(kind=kind, user=user).order_by('-created', '-pk').first()
    if row is not None:
        Generated.objects.filter(pk=row.pk).update(trial=True)


def trial_view(user, row):
    """The owner of a free try sees it through its teaser. A result of a lapsed subscription stays closed."""
    return row.trial and row.user_id == getattr(user, 'pk', None) and not access.has_tool(user, row.kind)


def audience(user, kind):
    """What the call to action offers: 'signup', 'open' (the tool is theirs), 'trial' (they may run it once),
    'premium' or 'plus'."""
    if user is None or not user.is_authenticated:
        return 'signup'
    if access.has_tool(user, kind):
        return 'open'
    if trial_open(user, kind):
        return 'trial'
    return 'plus' if kind in access.PLUS_TOOLS else 'premium'


# --- the examples ------------------------------------------------------------------

def showcase(kind, lang):
    """The example of a tool in this language (else in another), only from a staff account's result."""
    rows = Generated.objects.filter(kind=kind, showcase=True, status='done', user__is_staff=True).order_by('-updated')
    return rows.filter(lang=lang).first() or rows.first()


# --- the first results of each tool --------------------------------------------------

def _item(title, meta='', note=''):
    return {'title': title, 'meta': meta, 'note': note}


def _highlight(label, text, meta='', quote=False):
    """A passage of what the tool writes besides its list (a synthesis, where to cite, a contradicted citation...)."""
    return {'label': label, 'text': text, 'meta': meta, 'quote': quote}


def _teams(row):
    from . import teams
    result = row.data()
    if result.get('version') != teams.RESULT_VERSION:
        return None
    shown = teams.display(result, shared=True)
    items = [_item(g['institution'], ' · '.join(x for x in (g.get('country') or '', ngettext(
        '%(n)d paper', '%(n)d papers', g['papers']) % {'n': g['papers']}) if x)) for g in shown['institutions']]
    people, works = len(shown['people']), len(result.get('works') or [])
    headline = _('%(i)s, %(p)s and %(w)s on “%(title)s”.') % {
        'i': ngettext('%(n)d institution', '%(n)d institutions', len(items)) % {'n': len(items)},
        'p': ngettext('%(n)d person', '%(n)d people', people) % {'n': people},
        'w': ngettext('%(n)d paper', '%(n)d papers', works) % {'n': works},
        'title': result.get('title') or result.get('topic') or ''}
    approaches = [a for a in result.get('approaches') or [] if a.get('name')]
    leads = sum(1 for p in shown['people'] if p.get('lead'))
    contents = [x for x in (
        result.get('landscape') and _('The landscape of the topic'),
        len(approaches) > 1 and ngettext('%(n)d approach, named and described', '%(n)d approaches, each named and described', len(approaches)) % {'n': len(approaches)},
        leads and ngettext('%(n)d person who signs first, last or as corresponding author', '%(n)d people who sign first, last or as corresponding author', leads) % {'n': leads},
        shown['countries'] and ngettext('%(n)d country', '%(n)d countries', len(shown['countries'])) % {'n': len(shown['countries'])},
        _('Exports, a collection of the papers, a monthly follow-up')) if x]
    highlights = []
    if result.get('landscape'):
        highlights.append(_highlight(_('The landscape'), result['landscape']))
    if len(approaches) > 1:
        a = approaches[0]
        highlights.append(_highlight(_('An approach'), a.get('description') or '', '%s · %s' % (
            a['name'], ngettext('%(n)d paper', '%(n)d papers', a.get('papers') or 0) % {'n': a.get('papers') or 0})))
    return headline, items, contents, highlights


def _missed(row):
    from . import missed
    result = missed.shown(row)
    if not result:
        return None
    ordered = [i for _importance, group in result['groups'] for i in group]
    items = [_item(i['title'], ' · '.join(str(x) for x in (i.get('authors'), i.get('year')) if x), i.get('relation') or '')
             for i in ordered]
    high = sum(1 for i in result['active'] if i.get('importance') == 'high')
    headline = ngettext('%(n)d close paper the draft does not cite', '%(n)d close papers the draft does not cite',
                        len(items)) % {'n': len(items)}
    if high:
        headline += ', ' + ngettext('%(n)d that a referee would expect', '%(n)d that a referee would expect', high) % {'n': high}
    searched = result.get('searched') if isinstance(result.get('searched'), int) else 0
    probable = len(result.get('probable') or [])
    contents = [x for x in (
        any(i.get('section') for i in ordered) and _('Where to cite each one, section by section'),
        any(i.get('position') for i in ordered) and _('How to position your work against each one'),
        any(i.get('kind') for i in ordered) and _('What each one is to your draft: same question, competing method, earlier result, data...'),
        probable and ngettext('%(n)d paper probably already in your references, matched for you', '%(n)d papers probably already in your references, matched for you', probable) % {'n': probable},
        searched and ngettext('%(n)d paper screened in journals and preprints', '%(n)d papers screened in journals and preprints', searched) % {'n': searched}) if x]
    highlights = []
    best = next((i for i in ordered if i.get('position')), None)
    if best:  # never the sentences of the draft itself
        highlights.append(_highlight(_('Where to cite it and how to position yourself'), best['position'],
                                     ' · '.join(x for x in (best.get('section') or '', best['title']) if x)))
    return headline + '.', items, contents, highlights


def bib_changes(before, after):
    """'field: old → new' for the fields a correction changes or adds, from two BibTeX entries."""
    def fields(entry):
        return {m.group(1).lower(): ' '.join(m.group(2).split())
                for m in re.finditer(r'^\s*(\w+)\s*=\s*[{"](.*?)[}"]\s*,?\s*$', entry or '', re.M)}
    old, new = fields(before), fields(after)
    return ['%s: %s → %s' % (k, old[k], v) if k in old else '%s: + %s' % (k, v)
            for k, v in new.items() if old.get(k) != v]


def _refcheck(row):
    from . import refcheck
    result = refcheck.shown(row)
    if not result:
        return None
    items = [_item('%s · %s' % (p['key'], p.get('title') or ''), '', ', '.join(dict.fromkeys(i['label'] for i in p['active'])))
             for p in result['active']]
    refs = len(result.get('refs') or [])
    headline = ngettext('%(n)d reference to correct or check', '%(n)d references to correct or check', len(items)) % {'n': len(items)}
    if refs:
        headline += ' ' + ngettext('out of %(n)d', 'out of %(n)d', refs) % {'n': refs}
    claims = result.get('open_claims') or []
    if claims:
        headline += ', ' + ngettext('%(n)d sentence whose citation to check', '%(n)d sentences whose citation to check', len(claims)) % {'n': len(claims)}
    corrections = result.get('corrections') or {}
    checked = result.get('claims_checked') if isinstance(result.get('claims_checked'), int) else 0
    kinds = sorted((result.get('kinds') or {}).items(), key=lambda kv: refcheck.LEVELS.index(next(
        (i['level'] for p in result['active'] for i in p['active'] if i['kind'] == kv[0]), refcheck.LEVELS[-1])))
    contents = ['%s (%d)' % (refcheck.KIND_LABELS.get(kind, kind), n) for kind, n in kinds[:4]]
    contents += [x for x in (
        corrections and ngettext('%(n)d corrected BibTeX entry, ready to paste', '%(n)d corrected BibTeX entries, ready to paste', len(corrections)) % {'n': len(corrections)},
        checked and ngettext('%(n)d citation checked against the paper it cites', '%(n)d citations checked against the papers they cite', checked) % {'n': checked},
        result.get('undefined') and ngettext('%(n)d cited key missing from the .bib', '%(n)d cited keys missing from the .bib', len(result['undefined'])) % {'n': len(result['undefined'])},
        _('A follow-up of your references until your paper is published')) if x]
    highlights = []
    contradicted = next((c for c in claims if c.get('verdict') == 'contradicted' and c.get('reason')), None)
    if contradicted:
        highlights.append(_highlight(_('A citation the cited paper contradicts'), contradicted['sentence'], contradicted['reason'], quote=True))
    for key, correction in corrections.items():
        changed = bib_changes(correction.get('before'), correction.get('after'))
        if changed:
            highlights.append(_highlight(_('A corrected entry'), ' · '.join(changed[:3]), key))
            break
    return headline + '.', items, contents, highlights


def _literature(row):
    from . import literature
    result = literature._shown(row)
    if not result:
        return None
    items = [_item(p['title'], ' · '.join(str(x) for x in (p.get('authors'), p.get('year')) if x), p.get('reason') or '')
             for p in result.get('papers') or []]
    themes = len(result.get('themes') or [])
    headline = ngettext('%(n)d paper', '%(n)d papers', len(items)) % {'n': len(items)}
    if themes:
        headline += ' ' + ngettext('in %(n)d theme', 'in %(n)d themes', themes) % {'n': themes}
    question = result.get('title') or result.get('question') or ''
    overview, consensus = result.get('overview') or [], result.get('consensus') or []
    disagreements, gaps = result.get('disagreements') or [], result.get('gaps') or []
    contents = [x for x in (
        overview and _('A synthesis, each sentence with its papers'),
        themes and ngettext('%(n)d theme with its findings', '%(n)d themes with their findings', themes) % {'n': themes},
        consensus and ngettext('%(n)d point of consensus', '%(n)d points of consensus', len(consensus)) % {'n': len(consensus)},
        disagreements and ngettext('%(n)d disagreement, each side with its papers', '%(n)d disagreements, each side with its papers', len(disagreements)) % {'n': len(disagreements)},
        gaps and ngettext('%(n)d open question', '%(n)d open questions', len(gaps)) % {'n': len(gaps)},
        _('Each statement with the quoted sentence of its paper')) if x]
    highlights = []
    if overview:
        highlights.append(_highlight(_('The synthesis'), overview[0]['text']))
    if disagreements:
        highlights.append(_highlight(_('A disagreement between the papers'), disagreements[0]['text']))
    elif gaps:
        highlights.append(_highlight(_('An open question'), gaps[0]['text']))
    return headline + (' · ' + _('“%(title)s”') % {'title': question} if question else '') + '.', items, contents, highlights


READERS = {'teams': _teams, 'missed': _missed, 'refcheck': _refcheck, 'literature': _literature}


def greyed(count):
    return range(min(count, GREYED_MAX))


def teaser(row, shown=SHOWN):
    """{'kind', 'headline', 'contents' (what the result holds besides its list), 'highlights' (passages of it),
    'items' (shown in full), 'hidden', 'greyed', 'path', 'name', 'plus'} of a finished result, else None."""
    if row is None or row.status != 'done' or row.kind not in READERS:
        return None
    try:
        found = READERS[row.kind](row)
    except Exception:  # an example from an older version of a tool must never break its page
        return None
    if not found:
        return None
    headline, items, contents, highlights = found
    hidden = max(len(items) - shown, 0)
    from .premium_tools import TOOL_CHOICES
    return {'kind': row.kind, 'headline': headline, 'contents': contents, 'highlights': highlights, 'items': items[:shown], 'hidden': hidden, 'greyed': greyed(hidden),
            'path': PATHS[row.kind], 'name': dict(TOOL_CHOICES)[row.kind], 'plus': row.kind in access.PLUS_TOOLS}


def trial_page(request, row):
    """The owner's result seen through its teaser (trial_view): its first results, the others greyed."""
    from django.shortcuts import render
    from .generated import STALE
    from .premium_tools import TOOL_CHOICES
    if row.status == 'pending' and row.updated < timezone.now() - STALE:
        row.status = 'failed'  # lost in a restart
    return render(request, 'summarizer/tool_trial_result.html', {
        'row': row, 't': teaser(row), 'offer': audience(request.user, row.kind), 'path': PATHS[row.kind],
        'tool_name': dict(TOOL_CHOICES)[row.kind]})


def examples(lang):
    """The example of each tool that has one, its first result only, for the Premium page."""
    return [t for t in (teaser(showcase(kind, lang), shown=1) for kind in SHOWCASE_TOOLS) if t]


def page_context(request, kind):
    """What a tool's page needs for its example and its call to action."""
    from django.utils.translation import get_language
    return {'example': teaser(showcase(kind, get_language())) if kind in SHOWCASE_TOOLS else None,
            'offer': audience(request.user, kind), 'trial_open': trial_open(request.user, kind)}


# --- the paper page: who signs the papers around it ------------------------------------

ET_AL = re.compile(r'\s+et al\.?$')


def people_around(data, shown=1, least=2):
    """The authors named most often among a paper's neighbours (each lists its first three authors),
    those named at least `least` times: {'items', 'hidden', 'greyed', 'papers'} or None."""
    items = (data or {}).get('items') or []
    counts = Counter()
    for item in items:
        names = [n.strip() for n in ET_AL.sub('', item.get('authors') or '').split(',') if n.strip()]
        counts.update(dict.fromkeys(names, 1))
    frequent = [(name, n) for name, n in counts.most_common() if n >= least]
    if not frequent:
        return None
    hidden = len(frequent) - shown
    return {'items': [{'name': name, 'papers': n} for name, n in frequent[:shown]], 'hidden': max(hidden, 0),
            'greyed': greyed(max(hidden, 0)), 'papers': len(items)}


# --- the paper page: what changed in its latest version ---------------------------------

VERSION_TEASERS_PER_DAY = 30  # comparisons started for the teaser (the paid model reads both versions)
VERSION_TEASERS_KEY = 'version-teasers:%s'


def version_changes(request, arxiv_id):
    """The first change between the paper's version and the one before, the others counted and greyed:
    {'old', 'new', 'first', 'hidden', 'greyed', 'pending'} or None. A comparison is shared by every reader; one
    is started for a signed-in reader who is not a crawler, at most VERSION_TEASERS_PER_DAY a day."""
    from django.core.cache import cache
    from django.utils.translation import get_language
    from . import changes, generated, sources
    if not arxiv_id or sources.is_external(arxiv_id):
        return None
    base, shown = changes.base_id(arxiv_id)
    if shown < 2:
        return None
    old, new, lang = shown - 1, shown, get_language()
    key = changes.comparison_key(base, old, new)
    row = Generated.objects.filter(kind='changes', key=key, lang=lang).first()
    user = getattr(request, 'user', None)
    if row is None and user is not None and user.is_authenticated and not access.crawler(request):
        day = VERSION_TEASERS_KEY % timezone.now().date()
        cache.add(day, 0, 86400 * 2)
        if cache.incr(day) <= VERSION_TEASERS_PER_DAY:
            row = generated.start('changes', key, lang, changes.write(base, old, new, lang))
    if row is None or row.status not in ('done', 'pending'):
        return None
    found = {'old': old, 'new': new, 'pending': row.status == 'pending', 'first': None, 'hidden': 0, 'greyed': range(0)}
    if row.status == 'done':
        items = row.data().get('changes') or []
        if not items:
            return None
        hidden = len(items) - 1
        found.update(first=items[0], hidden=hidden, greyed=greyed(hidden))
    return found
