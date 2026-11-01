"""Journal club (Premium+): slides to present a paper, or the selected papers of a collection, to colleagues.

A deck is written once per paper (or selection of a collection), language, audience and duration, from what
the license lets us read (presskit.read: the full text, or only the abstract), and shared by every reader;
each reader edits a private copy of it. What is checked rather than trusted to the model, as in the press kit:

- each point of a slide comes with the passage of the paper that supports it, kept only when the passage
  really is in the text the model was given (presskit.passage), with its section and a link to it; a point
  without one, or with a number the paper never states, is marked for the presenter to check;
- the limits of the paper are split: those the authors state (each with its passage) and the points to
  discuss, which are the model's own reading and say so;
- earlier work is compared from the neighbourhood of the paper (neighbours.py: the papers it cites and the
  papers that cite it), never from the model's memory;
- where the paper stands (publication, retraction or correction notices, a newer arXiv version) comes from
  publication_status and changes, without the model;
- figures come from the paper's own HTML, only under an open license, with their credit (chat_figures).

With only the abstract, the deck is short and says what the abstract does not tell. Decks written before
(kinds 'slides' and 'slides_v2') are still shown and edited, read through normalized().

The editor keeps the pages of the deck by origin: a page is a page of the deck (its passages and figure
choices stay trusted, server side) or a new one; the reader reorders, adds, removes and duplicates pages,
picks another figure of the paper, and asks the model to rewrite one slide (shorter, simpler, more technical,
or its speaker notes). A reader can share their version by a read-only link (result_share), present it with
a presenter window (notes, next slide, timer), or export it to PowerPoint or PDF.
"""
import copy
import hashlib
import json
import re
import urllib.parse

from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.core.cache import cache
from django.db import transaction
from django.http import Http404, HttpResponse, HttpResponseBadRequest, JsonResponse
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils import timezone, translation
from django.utils.translation import gettext
from django.utils.translation import gettext_lazy as _
from django.views.decorators.cache import never_cache
from django.views.decorators.http import require_POST

from . import access, generated, result_share, sources, usage
from .models import ArxivPaper, Collection, CollectionMember, Generated, PaperVersion
from .reading import OUTPUT_LANGUAGES

DECK_KIND = 'slides_v3'
LEGACY_KINDS = ('slides_v2', 'slides')  # the newest first
COLLECTION_KIND = 'slides_coll'
EDIT_KIND = 'slides_edit_v3'
BACKUP_KIND = 'slides_edit_backup'
SHARE_KIND = 'slides_share'
DECK_LANGUAGES = tuple(OUTPUT_LANGUAGES)
AUDIENCES = (
    ('field', _('Researchers of the field')),
    ('students', _('Students new to the topic')),
    ('other', _('Researchers of another field')),
)
AUDIENCE_PROMPTS = {
    'field': 'researchers of the field: technical, without defining the usual terms',
    'students': 'master and PhD students new to the topic: define the key terms, give more context, keep the '
                'method understandable',
    'other': 'researchers of another field: little jargon, say why the question matters before how it is answered',
}
DURATIONS = ((10, _('10 minutes')), (20, _('20 minutes')), (45, _('45 minutes')))
PLANS = {
    10: '4 to 6 slides: the context and the question, what was done, the main results with their key numbers, '
        'the take-home message',
    20: '7 to 10 slides: the context and the question, what was done (method, data), the main results with '
        'their key numbers, how it compares with earlier work, the take-home message',
    45: '12 to 16 slides: the context and the question, the method and the data in detail, each main result on '
        'its own slide with its numbers, the checks of robustness, how it compares with earlier work, the '
        'take-home message',
}
ABSTRACT_PLAN = '3 or 4 slides, from the abstract alone: the question, what was done, what is claimed'
DEFAULTS = {'audience': 'field', 'duration': 20}
MAX_SLIDES = 18
MAX_POINTS = 6
MAX_PAGES = 40
FIGURES_SHOWN = 20  # captions the model reads to choose the figures
RELATED = 6  # earlier papers whose abstracts it reads
FOLLOWUPS = 4
ABSTRACT_CHARS = 600
COLLECTION_PAPERS = 6
POINT_KINDS = ('result', 'interpretation', 'method', 'context')
REWRITES = {
    'shorter': 'shorter: fewer and shorter points, the essential only',
    'simpler': 'simpler: for listeners new to the topic, with plain words',
    'technical': 'more technical: precise terms, methods and numbers, for specialists',
    'notes': 'its speaker notes only: 3 to 5 sentences the presenter can say while showing it, keep the title '
             'and the points as they are',
}
REWRITES_PER_DAY = 150
REWRITE_KEY = 'slides-rewrite:%s:%s'
STEPS = [
    ('read', _('Reading the paper and its figures')),
    ('context', _('Finding the earlier papers, the citing papers and where the paper stands')),
    ('write', _('Writing the slides, each point with its passage of the paper')),
]
COLLECTION_STEPS = [
    ('read', _('Reading the selected papers and their figures')),
    ('write', _('Writing the slides, each point with its passage of the paper')),
]

SHARED_RULES = """Write in {language}: every title, point, note, strength, limitation, question and answer, and the
presentation title (translate the paper title). Do not mix languages. Keep proper names, acronyms, numbers and
claims accurate. Quotes stay in the ORIGINAL language of the paper, copied exactly: 1 to 3 consecutive sentences,
no ellipsis. Keep each point under 20 words. Plain text: no LaTeX, no Markdown.
Every point that says what a paper did or found carries in "quote" the exact passage that supports it, and says
whether it is a "result" (measured, computed or proved), an "interpretation" (concluded or suggested by the
authors), "method" or "context". A point you cannot support with a passage has an empty quote: write as few of
them as possible. Never invent a number, a result, a publication or a peer review.
Slide titles have no numbers ("Context", not "1. Context"). Questions are under 20 words too.
Write about the science only: no point on where or whether a paper is published or peer reviewed (a slide of its own
gives what was checked), nor on the documents or checks you were given."""

PROMPT = """You prepare a journal club presentation of the paper in source_documents, for {audience}, to be
given in about {minutes} minutes.
""" + SHARED_RULES + """
Make {plan}.
related_work lists the closest earlier papers we know and papers that cite this one, with their abstracts:
compare with them only, and never call the result the first beyond what they allow.
{figures}{abstract_only}Answer with a JSON object:
{{"title": "the paper title translated into {language}",
  "slides": [{{"title": "...", "points": [{{"text": "...", "quote": "...", "kind": "result|interpretation|method|context"}}],
              "figure": "the label of a figure of source_documents, or null", "notes": "2 or 3 sentences to say while showing it"}}],
  "strengths": [{{"point": "a strength, 1 sentence", "quote": "the passage that shows it"}}],
  "stated_limitations": [{{"point": "a limitation the authors themselves state", "quote": "the passage where they state it"}}],
  "discussion_points": ["a doubt or open point the authors do not state: your own reading, to be checked"],
  "questions": ["a question to start the discussion"],
  "related": [{{"paper": "the id of a paper of related_work", "relation": "what this paper adds to it or does differently, 1 sentence"}}],
  "hard_questions": [{{"question": "a hard question the audience is likely to ask the presenter",
                      "answer": "what the paper lets the presenter answer, 2 sentences", "quote": "the passage that supports the answer, or an empty string"}}],
  "not_in_abstract": ["only when only the abstract is available: something a listener would need that it does not say"]}}
Give 3 or 4 strengths, 0 to 4 stated limitations (none when the authors state none), 3 or 4 discussion points,
4 to 6 questions, at most 4 related papers and 4 to 6 hard questions."""

COLLECTION_PROMPT = """You prepare a journal club session on the {count} papers in source_documents, for {audience},
to be given in about {minutes} minutes.
""" + SHARED_RULES + """
Make {plan}. The first slide gives the question the papers share; then each paper in turn (its slides name it by
its id in "paper"); then how they compare; then the take-home message. A point's quote comes from the paper of its
slide.
{figures}Answer with a JSON object:
{{"title": "a title for the session, in {language}",
  "slides": [{{"title": "...", "paper": "the id of the paper presented, or null", "points": [{{"text": "...", "quote": "...", "kind": "result|interpretation|method|context"}}],
              "figure": "the label of a figure of that paper, or null", "notes": "2 or 3 sentences to say while showing it"}}],
  "agree": ["where the papers agree, naming them by their first author"],
  "differ": ["where they differ or report different results, with the numbers"],
  "discussion_points": ["a doubt or open point the papers do not state: your own reading, to be checked"],
  "questions": ["a question to start the discussion"],
  "hard_questions": [{{"question": "...", "answer": "what the papers let the presenter answer", "paper": "the id of the paper the answer rests on", "quote": "its passage, or an empty string"}}]}}
Give 2 to 4 agreements, 2 to 4 differences, 3 or 4 discussion points, 4 to 6 questions and 4 to 6 hard questions."""

COLLECTION_PLANS = {10: 'one or two slides per paper, 8 slides at most in all', 20: 'two slides per paper, 14 at most',
                    45: 'two to four slides per paper, 18 at most'}


# --- options and rows -------------------------------------------------------------------

def options(values, default_lang):
    """{'lang', 'audience', 'duration'} from a query or a form; ValueError when one is unknown."""
    lang = values.get('deck_lang') or default_lang
    if lang not in dict(DECK_LANGUAGES):
        lang = 'en' if 'deck_lang' not in values else lang
    audience = values.get('audience') or DEFAULTS['audience']
    try:
        duration = int(values.get('duration') or DEFAULTS['duration'])
    except (TypeError, ValueError):
        raise ValueError('Unsupported duration')
    if lang not in dict(DECK_LANGUAGES) or audience not in dict(AUDIENCES) or duration not in dict(DURATIONS):
        raise ValueError('Unsupported slide options')
    return {'lang': lang, 'audience': audience, 'duration': duration}


def paper_key(arxiv_id, opts):
    """The decks of the usual audience and duration keep the paper's id as key, as the earlier ones."""
    if opts['audience'] == DEFAULTS['audience'] and opts['duration'] == DEFAULTS['duration']:
        return arxiv_id
    return '%s|%s|%s' % (arxiv_id, opts['audience'], opts['duration'])


def collection_key(pk, ids, opts):
    signature = hashlib.sha256(json.dumps([ids, opts['audience'], opts['duration']]).encode()).hexdigest()[:16]
    return '%s:%s' % (pk, signature)


def query(opts, version=None):
    values = {'deck_lang': opts['lang'], 'audience': opts['audience'], 'duration': opts['duration']}
    if version:
        values['deck_version'] = version
    return '?' + urllib.parse.urlencode(values)


def paper_row(arxiv_id, opts, version):
    """The deck shown: the newest kind written for these options, or the version asked for ('1', '2', '3')."""
    if version not in (None, '', '1', '2', '3'):
        raise ValueError('Unsupported slide version')
    rows = Generated.objects.filter(lang=opts['lang'])
    current = rows.filter(kind=DECK_KIND, key=paper_key(arxiv_id, opts)).first()
    if version == '3' or (current is not None and not version):
        return current
    if paper_key(arxiv_id, opts) != arxiv_id:
        return None
    kinds = {'1': ['slides'], '2': ['slides_v2']}.get(version, LEGACY_KINDS)
    return next((row for row in (rows.filter(kind=kind, key=arxiv_id).first() for kind in kinds) if row), None)


def edit_location(user, row):
    """(kind, key) of the reader's private copy of a deck."""
    if row.kind == 'slides':
        return 'slides_edit', legacy_edit_key(user, row.key)
    if row.kind == 'slides_v2':
        return 'slides_edit_v2', legacy_edit_key(user, row.key)
    return EDIT_KIND, '%s:%s' % (user.pk, hashlib.sha256(('%s:%s' % (row.kind, row.key)).encode()).hexdigest())


def legacy_edit_key(user, arxiv_id):
    return '%s:%s' % (user.pk, hashlib.sha256(arxiv_id.encode()).hexdigest())


# --- reading a deck -------------------------------------------------------------------

def authors_line(paper):
    names = [a.name for a in paper.authors.all()[:7]]
    return ', '.join(names[:6]) + (' et al.' if len(names) > 6 else '')


def original_pages(paper, deck):
    """The pages of a deck written before version 3: figures as dicts, the strengths and limits, the questions."""
    pages = [dict(kind='title', title=deck.get('title') or paper.title, bullets=[authors_line(paper)], notes='', figure=None)]
    pages += [dict(s, kind='content') for s in deck.get('slides', [])]
    with translation.override(deck.get('language') or translation.get_language()):
        pages += [dict(kind='columns', title=gettext('Strengths and limits'), bullets=deck.get('strengths', []),
                       secondary=deck.get('limitations', []), notes='', figure=None,
                       headings=[gettext('Strengths'), gettext('Limits')]),
                  dict(kind='discussion', title=gettext('For the discussion'), bullets=deck.get('questions', []), notes='', figure=None)]
    return pages


def normalized(row, paper=None):
    """A deck in the form of version 3: {'title', 'language', 'pages', 'figures': {id: figure}, 'qa', ...}.
    Each page has its kind, title, bullets, notes, figure (an id or None), sources and layout."""
    data = row.data()
    if data.get('generation_version') == 3:
        deck = data
    else:
        lang = data.get('language') or row.lang
        license = data.get('license', '')
        pages, figures = [], {}
        with translation.override(lang):
            for page in original_pages(paper, data):
                figure = page.get('figure')
                if isinstance(figure, dict) and figure.get('url'):
                    fid = 'f%s' % figure.get('number')
                    figures[fid] = {'url': figure['url'], 'caption': figure.get('caption', ''),
                                    'label': gettext('Figure %(n)s') % {'n': figure.get('number')},
                                    'credit': gettext('Figure %(n)s of the paper (arXiv, %(license)s)') % {
                                        'n': figure.get('number'), 'license': license}}
                    page['figure'] = fid
                else:
                    page['figure'] = None
                pages.append(page)
        deck = {'generation_version': data.get('generation_version', 1), 'title': data.get('title') or (paper.title if paper else ''),
                'language': lang, 'license': license, 'abstract_only': data.get('abstract_only', False),
                'label': sources.label(row.key), 'pages': pages, 'figures': figures, 'qa': []}
    for index, page in enumerate(deck['pages']):
        page.setdefault('notes', '')
        page.setdefault('bullets', [])
        page.setdefault('sources', [])
        page.setdefault('headings', [])
        page.setdefault('figure', None)
        page.update(origin=index, layout=page.get('layout', 'right'), figure_size=page.get('figure_size', 50),
                    text_size=page.get('text_size', 30))
        if page['kind'] == 'columns':
            page.setdefault('secondary', [])
    return deck


def text(value, limit):
    if not isinstance(value, str) or len(value) > limit or '\x00' in value:
        raise ValueError
    return value.strip()


def validate_pages(value, originals, figures):
    """The reader's pages, each from a page of the deck (origin: its index) or new (origin: None). Only text,
    order, layout and the choice of a figure of the paper come from the reader: kinds, passages and figure
    addresses stay those of the deck."""
    if not isinstance(value, list) or not 0 < len(value) <= MAX_PAGES:
        raise ValueError
    pages = []
    for edit in value:
        if not isinstance(edit, dict):
            raise ValueError
        origin = edit.get('origin')
        if origin is None:
            page = {'kind': 'content', 'title': '', 'bullets': [], 'notes': '', 'figure': None, 'sources': [], 'headings': []}
        elif type(origin) is int and 0 <= origin < len(originals):
            page = copy.deepcopy(originals[origin])
        else:
            raise ValueError
        page['origin'] = origin
        page['title'] = text(edit.get('title'), 300)
        if not page['title']:
            raise ValueError
        for field in ('bullets', 'secondary'):
            if field == 'secondary' and page['kind'] != 'columns':
                continue
            lines = edit.get(field, [])
            if not isinstance(lines, list) or len(lines) > 8:
                raise ValueError
            page[field] = [text(line, 600) for line in lines if text(line, 600)]
        page['notes'] = text(edit.get('notes', ''), 3000)
        page['layout'] = edit.get('layout', 'right')
        page['figure_size'] = edit.get('figure_size', 50)
        page['text_size'] = edit.get('text_size', 30)
        if (page['layout'] not in ('left', 'right', 'top', 'bottom') or
                type(page['figure_size']) is not int or not 25 <= page['figure_size'] <= 70 or
                page['text_size'] not in (24, 30, 36)):
            raise ValueError
        figure = edit.get('figure', page.get('figure'))
        if isinstance(figure, str):  # a figure of the paper chosen by its id ('' for none); anything else keeps the deck's
            if figure and figure not in figures:
                raise ValueError
            page['figure'] = figure or None
        pages.append(page)
    return pages


EDITED = ('origin', 'title', 'bullets', 'secondary', 'notes', 'layout', 'figure_size', 'text_size', 'figure')


def revision_of(pages):
    return hashlib.sha256(json.dumps(pages, sort_keys=True).encode()).hexdigest()


def editable_pages(user, row, deck):
    """(the reader's pages, their revision, whether a saved copy no longer fits the deck)."""
    originals = deck['pages']
    kind, key = edit_location(user, row)
    saved = Generated.objects.filter(kind=kind, key=key, lang=row.lang, user=user).first() if user.is_authenticated else None
    pages, mismatch = copy.deepcopy(originals), False
    if saved:
        try:
            edits = saved.data()['pages']
            for index, edit in enumerate(edits):
                if isinstance(edit, dict) and 'origin' not in edit:
                    edit['origin'] = index  # saved before pages could move: one per page of the deck
            pages = validate_pages(edits, originals, deck['figures'])
        except (ValueError, KeyError, TypeError, AttributeError):
            mismatch = True
    return pages, revision_of(pages), mismatch


def shown_pages(pages, figures):
    """The pages with their figure as a dict, for the page and the exports."""
    return [dict(page, figure=figures.get(page['figure']) if page.get('figure') else None,
                 figure_id=page.get('figure') or '', checked=sum(1 for s in page.get('sources') or [] if s.get('verified')))
            for page in pages]


# --- writing a deck -------------------------------------------------------------------

def clean(value, limit):
    return ' '.join(str(value or '').split())[:limit]


def checked_points(points, source, source_text):
    """(bullets, their checks): each point with its passage when the paper has it, and the numbers the paper
    never states. source: the paper dict of presskit.passage."""
    from . import presskit
    bullets, checks = [], []
    for point in points if isinstance(points, list) else []:
        if isinstance(point, str):
            point = {'text': point}
        if not isinstance(point, dict) or not clean(point.get('text'), 10):
            continue
        claim = clean(point['text'], 400)
        found = presskit.passage(point.get('quote', ''), source)
        bullets.append(claim)
        checks.append({'text': claim, 'kind': point.get('kind') if point.get('kind') in POINT_KINDS else 'context',
                       'quote': found['quote'] if found else '', 'location': (found['location'] or '')[:160] if found else '',
                       'link': found['link'] if found else '', 'verified': bool(found),
                       'numbers': presskit.numbers_absent([claim], source_text),
                       'paper': source.get('arxiv_id', '')})
        if len(bullets) == MAX_POINTS:
            break
    return bullets, checks


def figure_list(figures, paper, prefix=''):
    """{id: figure} of the paper's figures with their image and credit, and their [{label, caption}] for the model."""
    from . import presskit
    found, shown = {}, []
    for index, figure in enumerate(figures[:FIGURES_SHOWN], 1):
        label = figure['label'].rstrip(':.| ')
        credit = ', '.join(part for part in [presskit.byline(paper), label,
                                             sources.label(figure.get('source_paper_id') or paper.arxiv_id),
                                             sources.license_label(figure.get('license_url') or paper.license or '')] if part)
        fid = '%sf%d' % (prefix, index)
        found[fid] = {'url': figure['images'][0], 'caption': clean(figure['caption'], 1500), 'label': (prefix.upper() + ' ' if prefix else '') + label,
                      'credit': credit, 'source_url': figure.get('source_url', '')}
        shown.append({'label': found[fid]['label'], 'caption': clean(figure['caption'], 300)})
    return found, shown


def figure_by_label(value, figures):
    from .presskit import normalized as plain
    wanted = plain(str(value or ''))
    return next((fid for fid, f in figures.items() if wanted and plain(f['label']) == wanted), None)


def neighbourhood(paper):
    """(earlier papers, papers that cite it) of the paper's neighbourhood, computed when missing. Never fatal."""
    from . import neighbours
    try:
        data = neighbours.stored(paper.arxiv_id) or neighbours.store(paper.arxiv_id) or {}
    except Exception:
        generated.logger.exception('Neighbourhood of %s failed', paper.arxiv_id)
        data = {}
    items = data.get('items') or []
    date = paper.published_arxiv.isoformat() if paper.published_arxiv else ''
    prior = [i for i in items if i.get('group') == 'prior']
    prior += [i for i in items if i.get('group') == 'close' and (not date or (i.get('date') or '9') <= date)]
    followups = [i for i in items if i.get('group') == 'followup']
    return prior[:RELATED], followups[:FOLLOWUPS]


def year(item):
    return (item.get('date') or '')[:4]


def short_ref(item):
    return '%s (%s)' % (item.get('authors') or item.get('title', '')[:40], year(item) or 'n.d.')


def standing(paper, report):
    """Where the paper stands, from the sources checked (no model): its publication, the notices about it and
    a newer arXiv version. ([lines], [warnings])."""
    from . import publication_status
    from .library import version_of
    lines, warnings = [str(publication_status.description(report))], []
    if (report or {}).get('journal') and report.get('status') in ('published', 'forthcoming'):
        lines.append(' · '.join(part for part in [report['journal'], report.get('date', ''),
                                                   'DOI: ' + report['doi'] if report.get('doi') else ''] if part))
    labels = {'retraction': gettext('Retraction notice'), 'withdrawal': gettext('Withdrawal notice'),
              'removal': gettext('Removal notice'), 'expression_of_concern': gettext('Expression of concern'),
              'correction': gettext('Correction published'), 'erratum': gettext('Erratum published'),
              'corrigendum': gettext('Corrigendum published')}
    for notice in (report or {}).get('updates') or []:
        label = labels.get(notice.get('type'))
        if label:
            lines.append(label + (' (DOI: %s)' % notice['doi'] if notice.get('doi') else ''))
            warnings.append(label)
    if not sources.is_external(paper.arxiv_id):
        base, shown = version_of(paper.arxiv_id)
        latest = PaperVersion.objects.filter(arxiv_id=base).values_list('version', flat=True).first()
        if latest and shown and latest > shown:
            lines.append(gettext('A newer version (v%(n)s) is on arXiv: check what changed before presenting.') % {'n': latest})
    return lines, warnings


def citation_line(*parts):
    """'Ann Lee et al. Clouds on a hot planet. arXiv: 2101.00001v1.' without doubled full stops."""
    return clean('. '.join(clean(p, 300).rstrip('. ') for p in parts if clean(p, 300)), 400) + '.'


def reference(item):
    # an unpublished paper's venue is its arXiv category ('Computation and Language'): its identifier says more
    venue = item.get('venue') if item.get('status') == 'published' else ''
    return citation_line(short_ref(item), item.get('title', ''), venue, sources.label(item['ref']) if item.get('ref') else '')


def hard_questions(value, papers_by_id, default):
    """[{question, answer, passage}] with the passage only when the paper has it."""
    from . import presskit
    found = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not clean(item.get('question'), 10):
            continue
        source = papers_by_id.get(str(item.get('paper') or '')) or default
        found.append({'question': clean(item['question'], 600), 'answer': clean(item.get('answer'), 900),
                      'passage': presskit.passage(item.get('quote'), source) if source else None,
                      'paper': source['arxiv_id'] if source else ''})
        if len(found) == 6:
            break
    return found


def texts(value, limit=6):
    return [clean(v, 400) for v in value if clean(v, 400)][:limit] if isinstance(value, list) else []


DISCUSSION_SHOWN, QUESTIONS_SHOWN = 3, 4  # what fits a slide; the rest goes to its notes


def discussion_page(points, questions, note):
    """The page of points to discuss and questions, the ones that do not fit in its notes."""
    extra = points[DISCUSSION_SHOWN:] + questions[QUESTIONS_SHOWN:]
    return {'kind': 'columns', 'title': gettext('For the discussion'), 'figure': None, 'sources': [],
            'headings': [gettext('Points to discuss (AI reading, to check)'), gettext('Questions')],
            'bullets': points[:DISCUSSION_SHOWN], 'secondary': questions[:QUESTIONS_SHOWN],
            'notes': '\n'.join([note] + ['– ' + line for line in extra])}


def content_pages(answer, figures, sources_of):
    """The model's slides, each point checked against the paper of its slide (sources_of(slide) -> (source, text))."""
    pages = []
    for slide in answer.get('slides', []) if isinstance(answer.get('slides'), list) else []:
        if not isinstance(slide, dict) or not clean(slide.get('title'), 3):
            continue
        source, source_text = sources_of(slide)
        bullets, checks = checked_points(slide.get('points') or slide.get('bullets'), source, source_text)
        figure = figure_by_label(slide.get('figure'), figures)
        if figure in [p['figure'] for p in pages]:
            figure = None
        pages.append({'kind': 'content', 'title': clean(slide['title'], 200), 'bullets': bullets, 'sources': checks,
                      'notes': clean(slide.get('notes'), 1500), 'figure': figure})
        if len(pages) == MAX_SLIDES:
            break
    if not pages:
        raise generated.GenerationError('no slides')
    return pages


# The labels of the pages we write ourselves. French has its catalog; for the other languages without one, the
# model translates them in the same call (labels_to_translate), with the lines of where_the_paper_stands.
CATALOG_LANGUAGES = ('en', 'fr')
FIXED_LABELS = (
    'Strengths and limits', 'Strengths', 'Limits stated by the authors', 'The authors state no limit in the text we read',
    'For the discussion', 'Points to discuss (AI reading, to check)', 'Questions',
    'The points to discuss are a reading by an AI, not statements of the paper.',
    'The points to discuss are a reading by an AI, not statements of the papers.',
    'What the abstract does not say', 'Only the abstract could be read: these points need the full paper.',
    'Earlier work and what came next', 'What it builds on', 'Papers that cite it', 'No earlier paper found in our index',
    'No citing paper found yet',
    "From the papers it cites and the papers that cite it (Semantic Scholar, OpenAlex and our index), not from the model's memory.",
    'Where the paper stands', 'Checked in Crossref and arXiv when the deck was written; check again before presenting.',
    'References', 'How the papers compare', 'Where they agree', 'Where they differ',
    'From the comparison table of the collection, whose cells carry their passages.',
    'Written by an AI from the papers: check it against them.', 'What the papers leave open',
    'From the open questions of the collection.',
)
LABELS_RULE = ('labels_to_translate lists labels of slides we write ourselves: translate each into {language} in '
               '"labels": {{"<the label exactly as given>": "<its translation>"}}; keep names, DOIs and dates.\n')


def to_translate(lang, extra=()):
    """The labels the model translates for a language without a catalog, [] otherwise."""
    return [] if lang in CATALOG_LANGUAGES else list(dict.fromkeys(list(FIXED_LABELS) + [l for l in extra if l]))


def translated(pages, answer, asked):
    """The pages with the labels we wrote in English replaced by the model's translation of them."""
    labels = answer.get('labels') if isinstance(answer.get('labels'), dict) else {}
    mapping = {k: clean(v, 400) for k, v in labels.items() if k in asked and isinstance(v, str) and clean(v, 400)}
    if not mapping:
        return pages
    for page in pages:
        page['title'] = mapping.get(page.get('title'), page.get('title'))
        if page.get('notes'):
            page['notes'] = '\n'.join(mapping.get(line, line) for line in page['notes'].split('\n'))
        for field in ('headings', 'bullets', 'secondary'):
            if isinstance(page.get(field), list):
                page[field] = [mapping.get(line, line) for line in page[field]]
    return pages


def metadata(paper, scope):
    """What the model knows of a paper besides its text: its authors, not its publication (a slide of its own)."""
    from . import presskit
    known = {k: v for k, v in presskit.provenance(paper, scope).items() if k not in ('doi', 'journal_ref', 'published')}
    return dict(known, authors=[a.name for a in paper.authors.all()[:20]])


def write(arxiv_id, opts):
    lang = opts['lang']

    def work(pk):
        from . import presskit, publication_status
        from .assistant import MAX_CONTEXT_CHARS, SOURCE_POLICY
        generated.progress(pk, 'read')
        if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
            presskit.import_paper(arxiv_id)
        paper = ArxivPaper.objects.prefetch_related('authors').get(arxiv_id=arxiv_id)
        available, origin, abstract_only = presskit.read(paper)
        scope = 'abstract' if abstract_only else 'excerpt' if len(available) > MAX_CONTEXT_CHARS else 'full_text'
        source_text = available[:MAX_CONTEXT_CHARS]
        source = {'text': source_text, 'source': origin, 'arxiv_id': paper.arxiv_id}
        figures, captions = figure_list(presskit.figures_of(paper), paper)
        generated.progress(pk, 'context')
        try:
            report = publication_status.for_generation(paper)
        except Exception:
            generated.logger.exception('Publication status of %s failed', arxiv_id)
            report = {}
        prior, followups = neighbourhood(paper)
        generated.progress(pk, 'write')
        related = [{'id': i['ref'], 'title': clean(i.get('title'), 300), 'authors': i.get('authors', ''), 'date': i.get('date', ''),
                    'relation': 'earlier' if i in prior else 'cites this paper', 'abstract': clean(i.get('abstract'), ABSTRACT_CHARS)}
                   for i in prior + followups]
        with translation.override(lang):
            lines, warnings = standing(paper, report)
        asked = to_translate(lang, lines + warnings)
        documents = {'metadata': metadata(paper, scope),
                     'text': source_text, 'related_work': related, 'figures': captions}
        if asked:
            documents['labels_to_translate'] = asked
        language = generated.language_name(lang)
        prompt = (LABELS_RULE.format(language=language) if asked else '') + PROMPT.format(
            language=language, audience=AUDIENCE_PROMPTS[opts['audience']], minutes=opts['duration'],
            plan=ABSTRACT_PLAN if abstract_only else PLANS[opts['duration']],
            figures='Put a figure on a slide when it helps, by its label, at most once each.\n' if captions else '',
            abstract_only='Only the abstract is available: write only what it says, and fill not_in_abstract.\n' if abstract_only else '')
        answer = generated.chat_json(json.dumps({'source_documents': documents}, ensure_ascii=False), paid=True,
                                     max_tokens=9000 if opts['duration'] == 45 else 7000,
                                     system=SOURCE_POLICY + '\n' + prompt)
        title = clean(answer.get('title'), 300)
        if not title:
            raise generated.GenerationError('missing presentation title in the requested language')
        with translation.override(lang):
            pages = translated(deck_pages(paper, title, answer, figures, source, source_text, abstract_only, report, prior, followups),
                               answer, asked)
        mapping = {line: translated([{'title': line}], answer, asked)[0]['title'] for line in lines + warnings}
        lines, warnings = [mapping[l] for l in lines], [mapping[w] for w in warnings]
        return {'generation_version': 3, 'title': title, 'language': lang, 'audience': opts['audience'],
                'duration': opts['duration'], 'abstract_only': abstract_only, 'scope': scope,
                'label': sources.label(arxiv_id), 'license': sources.license_label(paper.license or ''),
                'papers': [{'id': arxiv_id, 'title': clean(paper.title, 300), 'authors': presskit.byline(paper)}],
                'pages': pages, 'figures': figures, 'warnings': warnings, 'standing': lines,
                'qa': hard_questions(answer.get('hard_questions'), {}, source)}
    return work


def deck_pages(paper, title, answer, figures, source, source_text, abstract_only, report, prior, followups):
    """Every page of a paper's deck, in the deck's language (translation is active)."""
    from . import presskit
    venue = (report or {}).get('journal') if (report or {}).get('status') in ('published', 'forthcoming') else paper.journal_ref
    pages = [{'kind': 'title', 'title': title, 'notes': '', 'figure': None, 'sources': [],
              'bullets': [authors_line(paper), ' · '.join(part for part in [clean(venue, 200), sources.label(paper.arxiv_id)] if part)]}]
    pages += content_pages(answer, figures, lambda slide: (source, source_text))
    missing = texts(answer.get('not_in_abstract'))
    if abstract_only and missing:
        pages.append({'kind': 'content', 'title': gettext('What the abstract does not say'), 'bullets': missing,
                      'notes': gettext('Only the abstract could be read: these points need the full paper.'), 'figure': None, 'sources': []})
    known = {i['ref']: i for i in prior}
    relations = []
    for item in answer.get('related') or [] if isinstance(answer.get('related'), list) else []:
        other = known.get(str((item or {}).get('paper') or '')) if isinstance(item, dict) else None
        if other and clean(item.get('relation'), 10) and other['ref'] not in [r['ref'] for r, _t in relations]:
            relations.append((other, clean(item['relation'], 300)))
    earlier = [short_ref(r) + ': ' + relation for r, relation in relations[:4]] or [
        short_ref(i) + ': ' + clean(i.get('title'), 160) for i in prior[:3]]
    later = [short_ref(i) + ': ' + clean(i.get('title'), 160) for i in followups]
    if earlier or later:
        pages.append({'kind': 'columns', 'title': gettext('Earlier work and what came next'), 'figure': None, 'sources': [],
                      'headings': [gettext('What it builds on'), gettext('Papers that cite it')],
                      'bullets': earlier or [gettext('No earlier paper found in our index')],
                      'secondary': later or [gettext('No citing paper found yet')],
                      'notes': gettext('From the papers it cites and the papers that cite it (Semantic Scholar, OpenAlex and our index), not from the model\'s memory.')})
    lines, _warnings = standing(paper, report)
    pages.append({'kind': 'content', 'title': gettext('Where the paper stands'), 'bullets': lines, 'figure': None, 'sources': [],
                  'notes': gettext('Checked in Crossref and arXiv when the deck was written; check again before presenting.')})
    strengths, strength_checks = checked_points(
        [{'text': s.get('point'), 'quote': s.get('quote'), 'kind': 'result'} for s in answer.get('strengths') or [] if isinstance(s, dict)],
        source, source_text)
    stated, stated_checks, unsourced = [], [], []
    for item in answer.get('stated_limitations') or [] if isinstance(answer.get('stated_limitations'), list) else []:
        if not isinstance(item, dict) or not clean(item.get('point'), 10):
            continue
        bullets, checks = checked_points([{'text': item['point'], 'quote': item.get('quote'), 'kind': 'interpretation'}], source, source_text)
        if checks and checks[0]['verified']:
            stated += bullets
            stated_checks += checks
        else:
            unsourced += bullets  # a limit the authors were said to state, without their passage: the model's reading
    pages.append({'kind': 'columns', 'title': gettext('Strengths and limits'), 'figure': None,
                  'headings': [gettext('Strengths'), gettext('Limits stated by the authors')],
                  'bullets': strengths[:4], 'secondary': stated[:4] or [gettext('The authors state no limit in the text we read')],
                  'sources': [dict(c, column=0) for c in strength_checks[:4]] + [dict(c, column=1) for c in stated_checks[:4]], 'notes': ''})
    pages.append(discussion_page((texts(answer.get('discussion_points'), 4) + unsourced)[:5], texts(answer.get('questions'), 6),
                                 gettext('The points to discuss are a reading by an AI, not statements of the paper.')))
    cited = [r for r, _t in relations[:4]] + followups[:2]
    pages.append({'kind': 'content', 'title': gettext('References'), 'figure': None, 'sources': [], 'notes': '',
                  'bullets': [citation_line(presskit.byline(paper), paper.title, sources.label(paper.arxiv_id))]
                  + [reference(i) for i in cited][:6]})
    for page in pages:
        page.setdefault('notes', '')
        page.setdefault('headings', [])
    return pages


def collection_write(collection_pk, ids, opts):
    lang = opts['lang']

    def work(pk):
        from . import comparison, presskit
        from .assistant import MAX_CONTEXT_CHARS, SOURCE_POLICY
        generated.progress(pk, 'read')
        papers = {p.arxiv_id: p for p in ArxivPaper.objects.filter(arxiv_id__in=ids).prefetch_related('authors')}
        papers = [papers[i] for i in ids if i in papers]
        if len(papers) < 2:
            raise generated.SourceUnavailable('fewer than two known papers')
        budget = MAX_CONTEXT_CHARS // len(papers)
        read, figures, captions, documents = {}, {}, [], []
        for index, paper in enumerate(papers, 1):
            available, origin, abstract_only = presskit.read(paper)
            text_read = available[:budget]
            read[paper.arxiv_id] = ({'text': text_read, 'source': origin, 'arxiv_id': paper.arxiv_id}, text_read)
            found, shown = figure_list(presskit.figures_of(paper)[:8], paper, prefix='p%d' % index)
            figures.update(found)
            captions += [dict(c, paper=paper.arxiv_id) for c in shown]
            documents.append({'id': paper.arxiv_id, 'metadata': metadata(paper, 'abstract' if abstract_only else 'full_text'),
                              'text': text_read})
        generated.progress(pk, 'write')
        language = generated.language_name(lang)
        asked = to_translate(lang)
        prompt = (LABELS_RULE.format(language=language) if asked else '') + COLLECTION_PROMPT.format(
            count=len(papers), language=language, audience=AUDIENCE_PROMPTS[opts['audience']], minutes=opts['duration'],
            plan=COLLECTION_PLANS[opts['duration']],
            figures='Put a figure on a slide of its paper when it helps, by its label, at most once each.\n' if captions else '')
        answer = generated.chat_json(json.dumps({'source_documents': dict({'papers': documents, 'figures': captions},
                                                                          **({'labels_to_translate': asked} if asked else {}))}, ensure_ascii=False),
                                     paid=True, max_tokens=9000, system=SOURCE_POLICY + '\n' + prompt)
        title = clean(answer.get('title'), 300)
        if not title:
            raise generated.GenerationError('missing presentation title')
        first = read[papers[0].arxiv_id]

        def sources_of(slide):
            return read.get(str(slide.get('paper') or ''), first)

        collection = Collection.objects.get(pk=collection_pk)
        with translation.override(lang):
            pages = [{'kind': 'title', 'title': title, 'figure': None, 'sources': [], 'notes': '', 'headings': [],
                      'bullets': [presskit.byline(p) + ' · ' + clean(p.title, 120) for p in papers][:6]}]
            pages += content_pages(answer, figures, sources_of)
            _exact, table = comparison.shown_result(collection, 'compare', lang)
            table = table.data() if table is not None else {}
            agree = [clean(i.get('text'), 300) for i in table.get('agreements', []) if isinstance(i, dict)][:4] or texts(answer.get('agree'), 4)
            differ = [clean(i.get('text'), 300) for i in table.get('disagreements', []) if isinstance(i, dict)][:4] or texts(answer.get('differ'), 4)
            if agree or differ:
                pages.append({'kind': 'columns', 'title': gettext('How the papers compare'), 'figure': None, 'sources': [],
                              'headings': [gettext('Where they agree'), gettext('Where they differ')], 'bullets': agree, 'secondary': differ,
                              'notes': gettext('From the comparison table of the collection, whose cells carry their passages.') if table.get('agreements') or table.get('disagreements')
                              else gettext('Written by an AI from the papers: check it against them.')})
            _exact, open_row = comparison.shown_result(collection, 'questions', lang)
            open_questions = [clean(q.get('question'), 300) for q in (open_row.data() if open_row is not None else {}).get('questions', [])
                              if isinstance(q, dict)][:5]
            if open_questions:
                pages.append({'kind': 'content', 'title': gettext('What the papers leave open'), 'bullets': open_questions, 'figure': None,
                              'sources': [], 'notes': gettext('From the open questions of the collection.'), 'headings': []})
            pages.append(discussion_page(texts(answer.get('discussion_points'), 4), texts(answer.get('questions'), 6),
                                         gettext('The points to discuss are a reading by an AI, not statements of the papers.')))
            pages.append({'kind': 'content', 'title': gettext('References'), 'figure': None, 'sources': [], 'notes': '', 'headings': [],
                          'bullets': [citation_line(presskit.byline(p), p.title, sources.label(p.arxiv_id)) for p in papers]})
        pages = translated(pages, answer, asked)
        for page in pages:
            page.setdefault('notes', '')
            page.setdefault('headings', [])
        return {'generation_version': 3, 'title': title, 'language': lang, 'audience': opts['audience'], 'duration': opts['duration'],
                'abstract_only': False, 'label': collection.name[:80], 'collection': collection_pk,
                'papers': [{'id': p.arxiv_id, 'title': clean(p.title, 300), 'authors': presskit.byline(p)} for p in papers],
                'pages': pages, 'figures': figures, 'warnings': [], 'standing': [],
                'qa': hard_questions(answer.get('hard_questions'), {k: v[0] for k, v in read.items()}, first[0])}
    return work


# --- the subjects of a deck: a paper or a collection ----------------------------------------------

class PaperSubject:
    def __init__(self, request, arxiv_id):
        self.request, self.arxiv_id = request, arxiv_id
        self.paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
        if self.paper is None:
            raise Http404

    def row(self, opts, version):
        return paper_row(self.arxiv_id, opts, version)

    def start(self, opts):
        return generated.start(DECK_KIND, paper_key(self.arxiv_id, opts), opts['lang'], write(self.arxiv_id, opts))

    def url(self, name, opts, version=None):
        return reverse(name, args=[self.arxiv_id]) + query(opts, version)

    def footer(self, deck):
        return deck.get('label') or sources.label(self.arxiv_id)


class CollectionSubject:
    def __init__(self, request, pk):
        from . import comparison, library
        self.request = request
        self.collection = library.editable(request, pk)
        self.ids = [p.arxiv_id for p in comparison.analysed(self.collection)][:COLLECTION_PAPERS]
        self.paper = None

    def row(self, opts, version):
        if version not in (None, '', '3'):
            raise ValueError('Unsupported slide version')
        return Generated.objects.filter(kind=COLLECTION_KIND, key=collection_key(self.collection.pk, self.ids, opts), lang=opts['lang']).first()

    def start(self, opts):
        return generated.start(COLLECTION_KIND, collection_key(self.collection.pk, self.ids, opts), opts['lang'],
                               collection_write(self.collection.pk, self.ids, opts))

    def url(self, name, opts, version=None):
        return reverse('collection_' + name, args=[self.collection.pk]) + query(opts, version)

    def footer(self, deck):
        return deck.get('label') or self.collection.name


def paid(request):
    return access.has_tool(request.user, 'journal_club')


def deck_of(subject, opts, version):
    """(row, normalized deck) of a finished deck, or (row or None, None)."""
    row = subject.row(opts, version)
    if row is None or row.status != 'done' or not row.content:
        return row, None
    return row, normalized(row, subject.paper)


def render_deck(request, subject, template_context):
    try:
        opts = options(request.GET, request.LANGUAGE_CODE)
        version = request.GET.get('deck_version')
        row, deck = deck_of(subject, opts, version) if paid(request) else (None, None)
    except ValueError:
        return HttpResponseBadRequest('Unsupported slide language or version')
    pages, revision, mismatch = editable_pages(request.user, row, deck) if deck else ([], '', False)
    legacy = bool(row and row.kind in LEGACY_KINDS)
    previous = ''
    if subject.paper is not None and paid(request) and paper_key(subject.arxiv_id, opts) == subject.arxiv_id:
        # the newest deck written before the one shown, with the reader's edits of it
        kinds = (DECK_KIND,) + LEGACY_KINDS
        older = kinds[kinds.index(row.kind) + 1:] if row is not None and row.kind in kinds else LEGACY_KINDS
        old = next((k for k in older if Generated.objects.filter(kind=k, key=subject.arxiv_id, lang=opts['lang'], status='done').exists()), None)
        previous = subject.url('slides', opts, {'slides_v2': '2', 'slides': '1'}[old]) if old else ''
    version = {'slides': '1', 'slides_v2': '2'}.get(row.kind, '3') if row else None
    share = share_row(request.user, row) if deck and request.user.is_authenticated else None
    context = {
        'paid': paid(request), 'row': row, 'deck': deck, 'pages': shown_pages(pages, deck['figures']) if deck else [],
        'figures': deck['figures'] if deck else {}, 'deck_revision': revision, 'edits_mismatch': mismatch,
        'deck_lang': opts['lang'], 'deck_languages': DECK_LANGUAGES, 'audience': opts['audience'], 'audiences': AUDIENCES,
        'duration': opts['duration'], 'durations': DURATIONS, 'legacy_deck': legacy, 'previous_deck_url': previous,
        'deck_return_url': subject.url('slides', opts, version), 'save_url': subject.url('slides_save', opts, version),
        'powerpoint_url': subject.url('slides_powerpoint', opts, version), 'rewrite_url': subject.url('slides_rewrite', opts, version),
        'share_action_url': subject.url('slides_share', opts, version), 'start_url': subject.url('slides_start', opts),
        'share_url': result_share.shared_url(request, share, 'journal-club') if share else '',
        'share_timing': result_share.timing(share.share_expires) if share and share.share_token else '',
        'footer': subject.footer(deck) if deck else '', 'steps': STEPS if subject.paper is not None else COLLECTION_STEPS,
        'chat_ids': ','.join(p['id'] for p in deck.get('papers', [])) if deck else '',
        'duration_minutes': (deck or {}).get('duration') or opts['duration'],
    }
    context.update(template_context)
    return render(request, 'summarizer/slides.html', context)


@never_cache
def slides_page(request, arxiv_id):
    subject = PaperSubject(request, arxiv_id)
    paper = subject.paper
    return render_deck(request, subject, {
        'paper': paper, 'arxiv_id': arxiv_id, 'authors': authors_line(paper),
        'abstract_only_expected': not sources.is_open_license(paper.license)})


@never_cache
@login_required
def collection_slides_page(request, pk):
    subject = CollectionSubject(request, pk)
    return render_deck(request, subject, {'collection': subject.collection, 'selected': len(subject.ids),
                                          'max_papers': COLLECTION_PAPERS})


def start_deck(request, subject):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    try:
        opts = options(request.POST, request.LANGUAGE_CODE)
    except ValueError:
        return JsonResponse({'status': 'invalid_language'}, status=400)
    if getattr(subject, 'ids', None) is not None and len(subject.ids) < 2:
        return JsonResponse({'status': 'too_few'}, status=400)
    row = subject.start(opts)
    return JsonResponse({'status': row.status, 'id': row.pk, 'url': subject.url('slides', opts, '3')})


@require_POST
def slides_start(request, arxiv_id):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    return start_deck(request, PaperSubject(request, arxiv_id))


@require_POST
@login_required
def collection_slides_start(request, pk):
    return start_deck(request, CollectionSubject(request, pk))


def posted_deck(request, subject):
    """(row, deck, the reader's posted pages validated) of a save, an export or a share; raises Http404, ValueError."""
    if len(request.body) > 200000:
        raise ValueError
    opts = options(request.GET, request.LANGUAGE_CODE)
    row, deck = deck_of(subject, opts, request.GET.get('deck_version'))
    if deck is None:
        raise Http404
    return row, deck, json.loads(request.body)


def save_deck(request, subject):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    try:
        with transaction.atomic():
            User.objects.select_for_update().get(pk=request.user.pk)
            row, deck, payload = posted_deck(request, subject)
            current, revision, mismatch = editable_pages(request.user, row, deck)
            if payload.get('revision') != revision:
                return JsonResponse({'status': 'conflict'}, status=409)
            pages = validate_pages(payload.get('pages'), deck['pages'], deck['figures'])
            kind, key = edit_location(request.user, row)
            if mismatch:  # an earlier copy no longer fits the deck: kept aside rather than lost
                old = Generated.objects.filter(kind=kind, key=key, lang=row.lang, user=request.user).first()
                Generated.objects.update_or_create(kind=BACKUP_KIND, key=key, lang=row.lang,
                                                   defaults={'user': request.user, 'status': 'done', 'content': old.content})
            # Only text, order, layout and figure choices are kept: never client-supplied URLs or HTML.
            edits = [dict({k: v for k, v in p.items() if k in EDITED}, figure=p.get('figure') or '') for p in pages]
            Generated.objects.update_or_create(kind=kind, key=key, lang=row.lang,
                                               defaults={'user': request.user, 'status': 'done', 'content': json.dumps({'pages': edits})})
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({'status': 'invalid'}, status=400)
    return JsonResponse({'status': 'saved', 'revision': revision_of(pages)})


@never_cache
@require_POST
def slides_save(request, arxiv_id):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    return save_deck(request, PaperSubject(request, arxiv_id))


@never_cache
@require_POST
@login_required
def collection_slides_save(request, pk):
    return save_deck(request, CollectionSubject(request, pk))


def export_deck(request, subject, name):
    from . import slides_pptx
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    try:
        row, deck, payload = posted_deck(request, subject)
        _pages, revision, _mismatch = editable_pages(request.user, row, deck)
        if payload.get('revision') != revision:
            return JsonResponse({'status': 'conflict'}, status=409)
        pages = validate_pages(payload.get('pages'), deck['pages'], deck['figures'])
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({'status': 'invalid'}, status=400)
    try:
        with translation.override(row.lang):
            data = slides_pptx.render_pptx(shown_pages(pages, deck['figures']), subject.footer(deck), pages[0]['title'])
    except slides_pptx.ExportError as error:
        return JsonResponse({'status': 'unavailable', 'message': str(error)}, status=503)
    response = HttpResponse(data, content_type=slides_pptx.MIME)
    response['Content-Disposition'] = 'attachment; filename="SummarizePaper-JournalClub-%s.pptx"' % (
        re.sub(r'[^a-zA-Z0-9._-]', '-', name)[:80] + '-' + row.lang)
    return response


@never_cache
@require_POST
def slides_powerpoint(request, arxiv_id):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    return export_deck(request, PaperSubject(request, arxiv_id), arxiv_id)


@never_cache
@require_POST
@login_required
def collection_slides_powerpoint(request, pk):
    return export_deck(request, CollectionSubject(request, pk), 'collection-%s' % pk)


# --- rewriting one slide ------------------------------------------------------------------

REWRITE_PROMPT = """Rewrite this slide of a journal club presentation: {how}. Write in {language}.
Keep every fact, number and name of the slide; add none, except from the passages of the paper given with it.
2 to 6 points under 20 words each; plain text, no LaTeX, no Markdown.
Answer with a JSON object {{"title": "...", "bullets": ["..."], "notes": "..."}}."""


def rewrite_deck(request, subject):
    """A slide rewritten by the model, from the reader's text and the passages of its page; not saved."""
    from . import evidence
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    try:
        row, deck, payload = posted_deck(request, subject)
        how = payload.get('how')
        if how not in REWRITES:
            raise ValueError
        page = validate_pages([payload.get('page')], deck['pages'], deck['figures'])[0]
    except (ValueError, TypeError, AttributeError):
        return JsonResponse({'status': 'invalid'}, status=400)
    used = REWRITE_KEY % (request.user.pk, timezone.now().date().isoformat())
    if cache.get(used, 0) >= REWRITES_PER_DAY:
        return JsonResponse({'status': 'quota'}, status=429)
    cache.set(used, cache.get(used, 0) + 1, 86400)
    passages = [s['quote'] for s in page.get('sources', []) if s.get('quote')]
    slide = {'title': page['title'], 'bullets': page['bullets'] + page.get('secondary', []), 'notes': page['notes'],
             'passages_of_the_paper': passages}
    try:
        with usage.task('slides', request.user.pk):
            answer = generated.chat_json(json.dumps({'slide': slide}, ensure_ascii=False), paid=False, max_tokens=1200, timeout=25,
                                         system=REWRITE_PROMPT.format(how=REWRITES[how], language=generated.language_name(row.lang)))
    except Exception:
        generated.logger.exception('Slide rewrite failed')
        return JsonResponse({'status': 'failed'}, status=503)
    title = clean(answer.get('title'), 300) or page['title']
    bullets = texts(answer.get('bullets'), 8)
    notes = clean(answer.get('notes'), 3000)
    if how == 'notes':
        title, bullets = page['title'], page['bullets']
    before = evidence.with_roundings(evidence.numbers(' '.join([page['title'], page['notes']] + page['bullets'] + page.get('secondary', []) + passages)))
    added = sorted(set().union(*[evidence.numbers(t) for t in [title, notes] + bullets]) - before)
    return JsonResponse({'status': 'ok', 'title': title, 'bullets': bullets if page['kind'] != 'columns' or how == 'notes' else page['bullets'],
                         'notes': notes or page['notes'], 'new_numbers': added})


@never_cache
@require_POST
def slides_rewrite(request, arxiv_id):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    return rewrite_deck(request, PaperSubject(request, arxiv_id))


@never_cache
@require_POST
@login_required
def collection_slides_rewrite(request, pk):
    return rewrite_deck(request, CollectionSubject(request, pk))


# --- sharing a version ------------------------------------------------------------------

def share_key(user, row):
    return '%s:%s' % (user.pk, hashlib.sha256(('%s:%s' % (row.kind, row.key)).encode()).hexdigest())


def share_row(user, row):
    return Generated.objects.filter(kind=SHARE_KIND, key=share_key(user, row), lang=row.lang, user=user).first()


def share_deck(request, subject):
    """Share the reader's saved version by a read-only link ('share': 90 more days, the version saved now;
    'unshare': stop it). Participants see it before or after the session, without an account."""
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    action = request.POST.get('action')
    if action not in ('share', 'unshare'):
        return JsonResponse({'status': 'invalid'}, status=400)
    try:
        opts = options(request.GET, request.LANGUAGE_CODE)
        row, deck = deck_of(subject, opts, request.GET.get('deck_version'))
    except ValueError:
        return JsonResponse({'status': 'invalid'}, status=400)
    if deck is None:
        raise Http404
    shared = share_row(request.user, row)
    if action == 'share':
        pages, _revision, _mismatch = editable_pages(request.user, row, deck)
        snapshot = {k: deck.get(k) for k in ('title', 'language', 'label', 'license', 'abstract_only', 'papers', 'warnings', 'qa', 'duration')}
        snapshot.update(pages=shown_pages(pages, deck['figures']), footer=subject.footer(deck), shared_at=timezone.now().isoformat())
        shared, _created = Generated.objects.update_or_create(kind=SHARE_KIND, key=share_key(request.user, row), lang=row.lang,
                                                              defaults={'user': request.user, 'status': 'done', 'content': json.dumps(snapshot)})
    if shared is None:
        return JsonResponse({'status': 'ok', 'url': ''})
    result_share.act(shared, action)
    return JsonResponse({'status': 'ok', 'url': result_share.shared_url(request, shared, 'journal-club'),
                         'timing': result_share.timing(shared.share_expires) if shared.share_token else ''})


@require_POST
def slides_share(request, arxiv_id):
    if not paid(request):
        return JsonResponse({'status': 'premium'}, status=403)
    return share_deck(request, PaperSubject(request, arxiv_id))


@require_POST
@login_required
def collection_slides_share(request, pk):
    return share_deck(request, CollectionSubject(request, pk))


def slides_shared(request, token):
    """A reader's version shared by its link: read-only, as it was when shared, for anyone with the link."""
    row = result_share.shared_row(SHARE_KIND, token)
    deck = row.data()
    response = render(request, 'summarizer/slides.html', {
        'shared': True, 'paid': False, 'deck': deck, 'pages': deck.get('pages', []), 'figures': {}, 'deck_lang': row.lang,
        'footer': deck.get('footer', ''), 'shared_at': row.updated, 'duration_minutes': deck.get('duration') or 20,
        'chat_ids': ','.join(p['id'] for p in deck.get('papers', []))})
    response['X-Robots-Tag'] = 'noindex'
    return response


# --- proposing a paper for the next session ---------------------------------------------------------

def journal_club_collections(user):
    """The collections the user owns or edits, those named for a journal club first."""
    mine = list(Collection.objects.filter(user=user).order_by('-updated'))
    mine += [m.collection for m in CollectionMember.objects.filter(user=user).select_related('collection').order_by('-collection__updated')]
    return sorted(mine, key=lambda c: 'journal' not in c.name.lower() and 'club' not in c.name.lower())


@login_required
def propose(request):
    """Propose a paper for a journal club: add it to a collection the team shares (invite colleagues to it), from
    which the presenter makes the slides of the selected papers. GET: choose the collection; POST: add it."""
    from . import library
    arxiv_id = library.canonical_id(request.POST.get('arxiv_id') or request.GET.get('arxiv_id') or '') or ''
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    found = (paper.title, '') if paper is not None else library.lookup(arxiv_id) if arxiv_id else None
    if not found:  # a paper of the digest or a letter may not be summarized yet: arXiv or its source knows it
        raise Http404
    collections = journal_club_collections(request.user)
    if request.method == 'POST':
        wanted = request.POST.get('collection', 'new')
        paid_collections = access.has_tool(request.user, 'collections')
        if wanted == 'new':
            if Collection.objects.filter(user=request.user).count() >= library.limits(paid_collections)['collections']:
                return redirect(reverse('slides_propose') + '?' + urllib.parse.urlencode({'arxiv_id': arxiv_id, 'error': 'too_many'}))
            collection = Collection.objects.create(user=request.user, name=request.POST.get('name', '').strip()[:120] or gettext('Journal club'))
        else:
            collection = library.editable(request, wanted)
        counts = library.add_many(collection, [arxiv_id], access.has_tool(collection.user, 'collections'))
        return redirect('/%s/collections/%s/?%s#papers' % (translation.get_language(), collection.pk, urllib.parse.urlencode(counts)))
    return render(request, 'summarizer/slides_propose.html', {
        'title': found[0], 'arxiv_id': arxiv_id, 'collections': collections, 'error': request.GET.get('error', '')})
