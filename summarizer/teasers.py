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
from django.utils.translation import gettext as _, gettext_lazy as _lazy, ngettext, pgettext_lazy

from . import access
from .models import Generated

SHOWN = 2  # results shown in full; the others are counted and greyed
GREYED_MAX = 3  # greyed rows drawn, whatever the count
TRIAL_TOOLS = ('teams', 'missed', 'refcheck')  # the least costly of the Premium tools that read a whole search
# The kinds of result an example can be, in the order of the Premium page; the tool of each (premium_tools)
SHOWCASE_TOOLS = ('teams', 'missed', 'refcheck', 'literature', 'slides_v3', 'compare', 'scoop_example', 'citation_example')
TOOL_OF = {'slides_v3': 'journal_club', 'compare': 'collections', 'scoop_example': 'scoops', 'citation_example': 'citations'}
KIND_OF = {tool: kind for kind, tool in TOOL_OF.items()}
PATHS = {'teams': 'teams', 'missed': 'missed', 'refcheck': 'refcheck', 'literature': 'literature-review',
         'slides_v3': 'journal-club', 'compare': 'collections', 'scoop_example': 'scoops', 'citation_example': 'citations'}
GUIDES = {'teams': 'find-research-teams-by-topic', 'missed': 'what-did-i-miss-uncited-papers',
          'refcheck': 'overleaf-bibliography-check', 'literature': 'literature-review-from-a-research-question',
          'slides_v3': 'journal-club-slides-from-a-paper', 'compare': 'organize-research-paper-collections',
          'scoop_example': 'scoop-watch-research-project', 'citation_example': 'track-research-paper-citations'}
# The fields an example can come from (Generated.showcase_field): a visitor sees the one of the papers they read
FIELDS = (('astro', pgettext_lazy('research field', 'Astrophysics')), ('cs', pgettext_lazy('research field', 'Computer science and AI')),
          ('bio', pgettext_lazy('research field', 'Biology and medicine')))
DEFAULT_FIELD = 'cs'  # most of the papers summarized on the site
FIELD_SESSION_KEY = 'example_field'
# The quality measured for a tool, with its date and its limits (management commands *_benchmark)
QUALITY = {
    'refcheck': _lazy('Measured in November 2026 on 52 citations annotated by hand: no false alarm and no miss, in English '
                      'and French. An easy test set, which does not replace a check on your own manuscript.'),
    'conference': _lazy('Measured in November 2026 on the program of ACAT 2024 (175 contributions): 96 to 99% of the talks '
                        'read, days and times right, and the 10 key talks of a test profile ranked first.'),
}


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

def field_of(category):
    """The example field of an arXiv category or a source ('astro-ph.EP', 'cs.CL', 'q-bio.NC', 'biorxiv'...), else ''."""
    head = (category or '').split('.')[0].lower()
    if head.startswith('astro'):
        return 'astro'
    if head in ('cs', 'stat', 'eess') or 'computer' in head:
        return 'cs'
    if head in ('q-bio', 'biorxiv', 'medrxiv') or 'medic' in head or 'bio' in head:
        return 'bio'
    return ''


def remember_field(request, category):
    """The paper page notes the field of the paper read, for the examples of the tools; never creates a session."""
    field = field_of(category)
    session = getattr(request, 'session', None)
    if field and session is not None and session.session_key and session.get(FIELD_SESSION_KEY) != field:
        session[FIELD_SESSION_KEY] = field


def visitor_field(request):
    """The field the visitor chose (?ex=), else of the last paper they read, else DEFAULT_FIELD."""
    names = dict(FIELDS)
    chosen = request.GET.get('ex', '')
    if chosen in names:
        return chosen
    session = getattr(request, 'session', None)
    remembered = session.get(FIELD_SESSION_KEY, '') if session is not None else ''
    return remembered if remembered in names else DEFAULT_FIELD


def examples_of(**filters):
    """The results marked as examples: a staff account's, or one shared by every reader (a journal club deck), never
    the private result of anybody else."""
    from django.db.models import Q
    return Generated.objects.filter(Q(user__is_staff=True) | Q(user__isnull=True), showcase=True, status='done', **filters)


def showcase(kind, lang, field=''):
    """The example of a tool: of this field and language first, then of this field, then of this language, then any."""
    rows = examples_of(kind=kind).order_by('-updated')
    for chosen in (rows.filter(showcase_field=field, lang=lang), rows.filter(showcase_field=field), rows.filter(lang=lang), rows):
        row = chosen.first()
        if row is not None:
            return row
    return None


def fields_shown(kinds):
    """[(code, name)] of the fields with an example of one of these tools."""
    have = set(examples_of(kind__in=kinds).values_list('showcase_field', flat=True))
    return [(code, name) for code, name in FIELDS if code in have]


# --- the first results of each tool --------------------------------------------------

def _item(title, meta='', note=''):
    return {'title': title, 'meta': meta, 'note': note}


def _highlight(label, text, meta='', quote=False, proof=None):
    """A passage of what the tool writes besides its list (a synthesis, where to cite, a contradicted citation...);
    proof: {'quote', 'title', 'paper'} the sentence of a paper it rests on, and that paper (its page when 'paper')."""
    return {'label': label, 'text': text, 'meta': meta, 'quote': quote, 'proof': proof}


def _proof(quote, title, paper=''):
    quote = ' '.join(str(quote or '').split())
    return {'quote': quote, 'title': title or '', 'paper': paper or ''} if quote else None


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
                                     best.get('section') or '',
                                     proof=_proof(best.get('abstract_quote'), best['title'], best.get('key'))))
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
    if refs and len(items) == refs and refs > 1:
        headline = _('All %(n)d references have a problem to fix or check') % {'n': refs}
    elif refs:
        headline = ngettext('%(n)d of the %(total)d references has a problem to fix or check',
                            '%(n)d of the %(total)d references have a problem to fix or check', len(items)) % {'n': len(items), 'total': refs}
    else:
        headline = ngettext('%(n)d reference has a problem to fix or check', '%(n)d references have a problem to fix or check',
                            len(items)) % {'n': len(items)}
    claims = result.get('open_claims') or []
    if claims:
        headline = _('%(found)s; %(claims)s') % {'found': headline, 'claims': ngettext(
            'in %(n)d sentence, the cited paper may not say what it is cited for',
            'in %(n)d sentences, the cited paper may not say what it is cited for', len(claims)) % {'n': len(claims)}}
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
        passage = contradicted.get('passage') if isinstance(contradicted.get('passage'), dict) else {}
        highlights.append(_highlight(_('A citation the cited paper contradicts'), contradicted['sentence'], contradicted['reason'],
                                     quote=True, proof=_proof(passage.get('quote'), contradicted.get('title'))))
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
    numbers = {p.get('n'): p for p in result.get('papers') or []}
    backed = next((c for c in consensus for s in c.get('support') or [] if s.get('quote') and s.get('n') in numbers), None)
    if backed:
        support = next(s for s in backed['support'] if s.get('quote') and s.get('n') in numbers)
        paper = numbers[support['n']]
        highlights.append(_highlight(_('A point of consensus, with its source'), backed['text'],
                                     proof=_proof(support['quote'], paper.get('title'), paper.get('key'))))
    if disagreements:
        highlights.append(_highlight(_('A disagreement between the papers'), disagreements[0]['text']))
    elif gaps:
        highlights.append(_highlight(_('An open question'), gaps[0]['text']))
    return headline + (' · ' + _('“%(title)s”') % {'title': question} if question else '') + '.', items, contents, highlights


def _slides(row):
    from . import slides
    deck = slides.normalized(row)
    pages = deck.get('pages') or []
    if not pages:
        return None
    paper = (deck.get('papers') or [{}])[0]
    headline = ngettext('%(n)d slide for a %(m)d-minute talk', '%(n)d slides for a %(m)d-minute talk', len(pages)) % {
        'n': len(pages), 'm': deck.get('duration') or 20} + ' · ' + _('“%(title)s”') % {'title': paper.get('title') or deck.get('title', '')}
    kinds = {p.get('kind') for p in pages}
    sourced = [p for p in pages if p.get('kind') == 'content' and any(c.get('verified') for c in p.get('sources') or [])]
    figures = [p for p in pages if p.get('figure')]
    contents = [x for x in (
        sourced and _('Each point with its passage of the paper'),
        'columns' in kinds and _('Earlier work and what came next, strengths and limits stated by the authors'),
        _('Where the paper stands: published, corrected, newer version'),
        figures and ngettext('%(n)d figure of the paper, with its credit', '%(n)d figures of the paper, with their credit', len(figures)) % {'n': len(figures)},
        deck.get('qa') and ngettext('%(n)d hard question to prepare', '%(n)d hard questions to prepare', len(deck['qa'])) % {'n': len(deck['qa'])},
        _('Presenter view, PowerPoint, Beamer and a link to share')) if x]
    highlights = []
    if sourced:
        page = sourced[0]
        check = next(c for c in page['sources'] if c.get('verified'))
        highlights.append(_highlight(_('A slide'), ' · '.join(page.get('bullets', [])[:3]), page['title'],
                                     proof=_proof(check.get('quote'), paper.get('title'), paper.get('id'))))
    items = [_item(p.get('title') or '', ' · '.join(p.get('bullets', [])[:1]) if p.get('kind') == 'content' else '') for p in pages]
    return headline, items, contents, highlights


# a "disagreement" that says there is none ("The table does not establish a direct contradiction...") makes a poor example
NO_DISAGREEMENT = re.compile(r"\b(no|not|neither|aucune?|pas|ni)\b[^.]{0,60}\b(contradict\w*|disagree\w*|conflict\w*|désaccord|divergen\w*)",
                             re.I)


def _compare(row):
    from . import comparison
    data = comparison.display(row.data(), row.lang)
    rows, cols = data['rows'], data['columns']
    if len(rows) < 2:
        return None
    headline = _('%(n)d papers compared on %(c)d columns') % {'n': len(rows), 'c': len(cols)} + '.'
    contents = [x for x in (
        _('Columns: %(names)s') % {'names': ', '.join(str(label) for _code, label in cols)},
        data['filled'] and _('%(q)d of the %(f)d cells with their passage of the paper') % {'q': data['quoted'], 'f': data['filled']},
        data['agreements'] and ngettext('%(n)d point where they agree', '%(n)d points where they agree', len(data['agreements'])) % {'n': len(data['agreements'])},
        data['disagreements'] and ngettext('%(n)d point where they disagree', '%(n)d points where they disagree', len(data['disagreements'])) % {'n': len(data['disagreements'])},
        data['gaps'] and ngettext('%(n)d question none of them addresses', '%(n)d questions none of them addresses', len(data['gaps'])) % {'n': len(data['gaps'])},
        _('Cells you correct by hand, CSV export, a chat with the papers')) if x]
    highlights = []
    for name, label in (('disagreements', _('Where the papers disagree')), ('agreements', _('Where the papers agree'))):
        finding = next((f for f in data[name] if f['passages'] and not NO_DISAGREEMENT.search(f['text'])), None)
        if finding:
            passage = finding['passages'][0]
            source = rows[passage['n'] - 1]
            highlights.append(_highlight(label, finding['text'], '', proof=_proof(
                passage['quote'], '%s · %s' % (source['title'], passage['label']), source['arxiv_id'])))
            break
    items = []
    for r in rows:
        cell = next((c for c in r['cells'] if not c['missing']), None)
        items.append(_item(r['title'], '%s: %s' % (dict(cols).get(cell['code'], ''), cell['text']) if cell else ''))
    return headline, items, contents, highlights


OVERLAP_LABELS = {'high': _lazy('Strong overlap'), 'partial': _lazy('Partial overlap')}


def _scoop(row):
    data = row.data()
    matches = data.get('matches') or []
    if not matches:
        return None
    high = sum(1 for m in matches if m.get('overlap') == 'high')
    headline = ngettext('%(n)d new paper close to the project “%(title)s”, found in the announcements of its first days',
                        '%(n)d new papers close to the project “%(title)s”, found in the announcements of its first days',
                        len(matches)) % {'n': len(matches), 'title': data.get('project', '')}
    contents = [x for x in (
        high and ngettext('%(n)d strong overlap, emailed the day it is announced', '%(n)d strong overlaps, emailed the day they are announced', high) % {'n': high},
        _('How close it is on each facet: question, method, data, results'),
        _('What it shares with your project and what stays your own'),
        _('What to do: cite it, set your project apart, contact the authors, hurry'),
        _('The prior art of the last three years, and the new papers of the teams you name')) if x]
    best = matches[0]
    highlights = [_highlight(str(OVERLAP_LABELS.get(best.get('overlap'), '')), best.get('shared') or '', best.get('title') or '')]
    if best.get('original'):
        highlights.append(_highlight(_('What stays your own'), best['original']))
    items = [_item(m.get('title') or '', str(OVERLAP_LABELS.get(m.get('overlap'), '')), m.get('shared') or '') for m in matches]
    return headline, items, contents, highlights, {'email': data.get('html', ''), 'email_subject': data.get('subject', '')}


def _citation(row):
    data = row.data()
    cites = data.get('items') or []
    if not cites:
        return None
    headline = ngettext('%(n)d new paper citing “%(title)s”', '%(n)d new papers citing “%(title)s”', len(cites)) % {
        'n': len(cites), 'title': data.get('paper', '')}
    contents = [_('What each one does with it: questions it, supports it, builds on it, uses it or mentions it'),
                _('The sentences that cite it, read in the citing paper'),
                _('Self-citations apart, and an email at once when a paper questions it'),
                _('Every day, week or month, for your papers and those you follow')]
    telling = ('questions', 'supports', 'builds_on', 'uses', 'mentions', '')  # what makes a citation worth reading first
    noted = [c for c in cites if c.get('note')]
    best = min(noted, key=lambda c: telling.index(c.get('stance') if c.get('stance') in telling else '')) if noted else None
    highlights = [_highlight(best.get('label') or '', best['note'], best.get('title') or '')] if best else []
    items = [_item(c.get('title') or '', c.get('label') or '', c.get('note') or '') for c in cites]
    return headline, items, contents, highlights, {'email': data.get('html', ''), 'email_subject': data.get('subject', '')}


READERS = {'teams': _teams, 'missed': _missed, 'refcheck': _refcheck, 'literature': _literature,
           'slides_v3': _slides, 'compare': _compare, 'scoop_example': _scoop, 'citation_example': _citation}


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
    headline, items, contents, highlights = found[:4]
    extras = found[4] if len(found) > 4 else {}
    lead = next((h for h in highlights if h.get('proof')), highlights[0] if highlights else None)
    hidden = max(len(items) - shown, 0)
    from .premium_tools import TOOL_CHOICES
    tool = TOOL_OF.get(row.kind, row.kind)
    return {'kind': row.kind, 'tool': tool, 'headline': headline, 'contents': contents, 'highlights': highlights[:2], 'lead': lead,
            'items': items[:shown], 'hidden': hidden, 'greyed': greyed(hidden), 'trial': tool in TRIAL_TOOLS,
            'email': extras.get('email', ''), 'email_subject': extras.get('email_subject', ''),
            'path': PATHS[row.kind], 'name': dict(TOOL_CHOICES)[tool], 'plus': tool in access.PLUS_TOOLS,
            'guide': GUIDES.get(row.kind, ''), 'quality': QUALITY.get(row.kind, ''), 'field': row.showcase_field,
            'demo': row.showcase}


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


def examples(request):
    """The example of each tool that has one, in the visitor's field, for the Premium page."""
    from django.utils.translation import get_language
    field = visitor_field(request)
    return {'examples': [t for t in (teaser(showcase(kind, get_language(), field), shown=1) for kind in SHOWCASE_TOOLS) if t],
            'example_field': field, 'example_fields': fields_shown(SHOWCASE_TOOLS)}


def page_context(request, tool):
    """What a tool's page needs for its example and its call to action (tool: a key of premium_tools)."""
    from django.utils.translation import get_language
    field = visitor_field(request)
    kind = KIND_OF.get(tool, tool)
    example = teaser(showcase(kind, get_language(), field)) if kind in SHOWCASE_TOOLS else None
    return {'example': example, 'example_field': field,
            'example_fields': fields_shown([kind]) if example else [], 'quality': QUALITY.get(tool, ''),
            'offer': audience(request.user, tool), 'trial_open': trial_open(request.user, tool)}


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
