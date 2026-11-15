"""A public page for each example of a Premium tool (teasers.SHOWCASE_TOOLS), to be found from a search engine:
"journal club slides on AlphaFold", "who cites the Planck results"... Each shows more of the example than the
tool's page (its first results, every passage it shows, the email it sent), the full result when it has a
public link, and the way to run the tool on one's own work. /<lang>/examples/ lists them by field.

The examples are the results an administrator marked (Generated.showcase, staff accounts or shared rows only:
teasers.examples_of), so a page never shows anybody's own search.
"""
import re

from django.http import Http404
from django.shortcuts import render
from django.utils.text import slugify
from django.utils.translation import get_language, gettext as _, gettext_lazy as _lazy

from . import teasers

SHOWN = 10  # results shown in full on an example's page (its tool's page shows teasers.SHOWN)
PREFIXES = {'teams': 'who-works-on', 'missed': 'missed-papers', 'refcheck': 'bibliography-check',
            'literature': 'literature-review', 'slides_v3': 'journal-club', 'compare': 'comparison-table',
            'scoop_example': 'scoop-watch', 'citation_example': 'citation-alert'}
TITLES = {
    'teams': _lazy('Who works on “%(s)s”? The teams, people and approaches'),
    'missed': _lazy('The close papers a draft on “%(s)s” does not cite'),
    'refcheck': _lazy('Bibliography check of a draft: “%(s)s”'),
    'literature': _lazy('Literature review: %(s)s'),
    'slides_v3': _lazy('Journal club slides on “%(s)s”, each point with its passage of the paper'),
    'compare': _lazy('%(s)s: the papers compared side by side'),
    'scoop_example': _lazy('Scoop watch of a research project: “%(s)s”'),
    'citation_example': _lazy('Who cites “%(s)s”, and what for'),
}
INTROS = {
    'teams': _lazy('From a topic, the tool read the papers on it and grouped who signs them: institutions, people, '
                   'the approaches they take and the countries where the work is done.'),
    'missed': _lazy('From a draft (here a demonstration manuscript), the tool searched journals and preprints for the close '
                    'papers it does not cite, and says where to cite each one and how to position the draft.'),
    'refcheck': _lazy('From a LaTeX draft and its .bib (here a demonstration manuscript), the tool checked every reference '
                      'against the records of the papers, then whether the cited papers say what they are cited for.'),
    'literature': _lazy('From a research question, the tool read the papers that answer it and wrote a synthesis, '
                        'the points of consensus and of disagreement and the open questions, each with its sources.'),
    'slides_v3': _lazy('From the paper alone, the tool wrote a journal club presentation: each point comes with the '
                       'passage of the paper it rests on, with the figures, the context and the questions to prepare.'),
    'compare': _lazy('From a collection of papers, the tool filled a table, one row per paper, each cell with its '
                     'passage of the paper, then said where the papers agree and where they disagree.'),
    'scoop_example': _lazy('From the description of a research project, the scoop watch reads the new papers of each '
                           'day and emails those that come close, and why. Here, the email of its first days.'),
    'citation_example': _lazy('Citation alerts read the papers that cite a paper you follow and say what each one does '
                              'with it. Here, the email of the latest papers citing this one.'),
}
EXAMPLE_TABLE = re.compile(r'^(example table|tableau d.exemple)\s*[:(-]?\s*', re.I)  # the demo collections' prefix
SHARED_PATHS = {'teams': 'teams', 'missed': 'missed', 'refcheck': 'refcheck', 'literature': 'literature-review'}


def subject(row):
    """What an example is about: the paper, the topic, the question, the project, the collection."""
    data = row.data()
    if row.kind == 'slides_v3':
        found = ((data.get('papers') or [{}])[0].get('title') or data.get('title') or '')
    elif row.kind == 'compare':
        from .models import Collection
        collection = Collection.objects.filter(pk=row.key.split(':')[0]).first()
        found = EXAMPLE_TABLE.sub('', collection.name) if collection else ''
    elif row.kind == 'scoop_example':
        found = data.get('project') or ''
    elif row.kind == 'citation_example':
        found = data.get('paper') or ''
    else:
        found = data.get('title') or data.get('topic') or data.get('question') or ''
    return ' '.join(str(found).split()).rstrip('.')


def slug(row, about=None):
    about = subject(row) if about is None else about
    return ('%s-%s' % (PREFIXES[row.kind], slugify(about)))[:90].rstrip('-') if about else ''


def listed(lang):
    """[(row, subject, slug)] of the examples in this language, newest first, one per page address."""
    found, seen = [], set()
    for row in teasers.examples_of(kind__in=teasers.SHOWCASE_TOOLS, lang=lang).order_by('-updated'):
        about = subject(row)
        address = slug(row, about)
        if address and address not in seen:
            seen.add(address)
            found.append((row, about, address))
    return found


def title(row, about):
    return TITLES[row.kind] % {'s': about[:1].upper() + about[1:] if row.kind == 'compare' else about}


def full_url(row, lang):
    """The public page of the whole result, when it has one: its shared link, or the shared collection."""
    if row.kind in SHARED_PATHS and row.share_token:
        return '/%s/%s/shared/%s/' % (lang, SHARED_PATHS[row.kind], row.share_token)
    if row.kind == 'compare':
        from .models import Collection
        collection = Collection.objects.filter(pk=row.key.split(':')[0]).exclude(share_token='').first()
        if collection is not None:
            return '/%s/collections/shared/%s/' % (lang, collection.share_token)
    if row.kind == 'slides_v3':
        paper = (row.data().get('papers') or [{}])[0].get('id') or ''
        return '/%s/paper/%s/' % (lang, paper) if paper else ''
    return ''


def _card(row, about, address, lang):
    names = dict(teasers.FIELDS)
    return {'url': '/%s/examples/%s/' % (lang, address), 'title': title(row, about), 'kind': row.kind,
            'tool': teasers.TOOL_OF.get(row.kind, row.kind), 'field': row.showcase_field,
            'field_name': names.get(row.showcase_field, '')}


def index(request):
    lang = get_language()
    cards = [_card(row, about, address, lang) for row, about, address in listed(lang)]
    order = [code for code, _name in teasers.FIELDS]
    groups = []
    for code, name in teasers.FIELDS:
        shown = sorted((c for c in cards if c['field'] == code), key=lambda c: teasers.SHOWCASE_TOOLS.index(c['kind']))
        if shown:
            groups.append((name, shown))
    other = [c for c in cards if c['field'] not in order]
    if other:
        groups.append((_('Other fields'), other))
    return render(request, 'summarizer/examples.html', {'groups': groups})


def detail(request, address):
    lang = get_language()
    rows = listed(lang)
    found = next(((row, about) for row, about, a in rows if a == address), None)
    if found is None:
        raise Http404
    row, about = found
    t = teasers.teaser(row, shown=SHOWN, highlights_shown=4)
    if t is None:
        raise Http404
    tool = teasers.TOOL_OF.get(row.kind, row.kind)
    other_lang = 'fr' if lang == 'en' else 'en'
    sibling = (teasers.examples_of(kind=row.kind, lang=other_lang, showcase_field=row.showcase_field).order_by('-updated').first()
               if row.showcase_field else None)
    alternates = [(lang, '/%s/examples/%s/' % (lang, address))]
    if sibling is not None and slug(sibling):
        alternates.append((other_lang, '/%s/examples/%s/' % (other_lang, slug(sibling))))
    more = [_card(r, a, s, lang) for r, a, s in rows if s != address and r.showcase_field == row.showcase_field][:6]
    return render(request, 'summarizer/example.html', {
        't': t, 'row': row, 'title': title(row, about), 'intro': INTROS[row.kind], 'about': about,
        'full_url': full_url(row, lang), 'offer': teasers.audience(request.user, tool), 'alternates': alternates,
        'example_page': True, 'more': more, 'field_name': dict(teasers.FIELDS).get(row.showcase_field, ''), 'address': address})
