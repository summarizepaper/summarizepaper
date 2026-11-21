"""Statistics guides written from data, in English and French, like the arXiv statistics guide.

- fastest-growing-research-topics: the terms whose share of arXiv, bioRxiv and medRxiv papers grew most, from
  the weekly term counts of the rising topics (TermWeek). Production data: written as Guide rows by the hourly
  run once a month (refresh_due), since files cannot be written on Heroku.
- authors-per-paper and arxiv-papers-published-in-journals: from arXiv's OAI-PMH metadata (arxiv_metadata.py),
  harvested on a laptop and committed as Markdown files (manage.py data_guides).

The counts decide what the pages say. The model only drops candidate terms that are not research topics and
writes their readable form, as for the rising topics.
"""
import datetime
import logging
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from urllib.parse import urlencode

import yaml
from django.utils import timezone, translation

from . import guides

logger = logging.getLogger(__name__)

SITE = guides.SITE
LANGS = ('en', 'fr')
MONTHS_FR = ['janvier', 'février', 'mars', 'avril', 'mai', 'juin', 'juillet', 'août', 'septembre', 'octobre',
             'novembre', 'décembre']
NBSP = ' '


# --- formatting ------------------------------------------------------------------------------------------------

def num(n, lang):
    text = '{:,}'.format(int(round(n)))
    return text.replace(',', NBSP) if lang == 'fr' else text


def dec(x, lang, digits=1):
    text = '%.*f' % (digits, x)
    return text.replace('.', ',') if lang == 'fr' else text


def pct(x, lang, digits=0):
    return dec(x, lang, digits) + (NBSP + '%' if lang == 'fr' else '%')


def times(x, lang):
    return '×' + dec(x, lang, 1)


def day(d, lang):
    if lang == 'fr':
        return '%d %s %d' % (d.day, MONTHS_FR[d.month - 1], d.year)
    return '%d %s %d' % (d.day, d.strftime('%B'), d.year)


def month(d, lang):
    return ('%s %d' % (MONTHS_FR[d.month - 1], d.year)) if lang == 'fr' else d.strftime('%B %Y')


def t(lang, en, fr):
    return fr if lang == 'fr' else en


def link(lang, path):
    return '%s/%s/%s' % (SITE, lang, path.lstrip('/'))


def cta(lang):
    return t(lang, '''<div class="cta-box">
<p>Follow your field without reading every listing</p>
<a class="btn btn-lg" href="%s">Get the daily paper alert</a>
<a class="btn btn-lg btn-ghost" href="%s">Summarize a paper</a>
</div>''', '''<div class="cta-box">
<p>Suivez votre domaine sans lire toutes les listes</p>
<a class="btn btn-lg" href="%s">Recevoir l'alerte quotidienne</a>
<a class="btn btn-lg btn-ghost" href="%s">Résumer un article</a>
</div>''') % (link(lang, 'digest/'), link(lang, ''))


def table(header, rows):
    lines = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    lines += ['| ' + ' | '.join(str(c).replace('|', '\\|') for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


DATA_PAGES = [('arxiv-statistics', 'arXiv statistics: submissions per year and field', 'Statistiques d’arXiv : soumissions par année et par domaine'),
              ('fastest-growing-research-topics', 'The fastest-growing research topics', 'Les sujets de recherche en plus forte croissance'),
              ('authors-per-paper', 'How many authors per paper, by field', 'Combien d’auteurs par article, par domaine'),
              ('arxiv-papers-published-in-journals', 'How many arXiv preprints are published in journals', 'Combien de preprints arXiv sont publiés en revue')]


def related(lang, slug):
    """The other statistics pages, linked from each one."""
    items = ['- [%s](%s)' % (t(lang, en, fr), link(lang, 'guides/%s/' % other)) for other, en, fr in DATA_PAGES if other != slug]
    return t(lang, '## More research publishing data', '## Plus de données sur la publication scientifique') + '\n\n' + '\n'.join(items)


def _esc(text):
    return str(text).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;').replace('"', '&quot;')


def bar_chart(title, caption, rows, aria, top=None):
    """A horizontal bar chart in the guides' SVG style (.viz in site.css; guide.html shows data-tip on hover).

    rows: (label, value, shown value, tooltip); bars scale to `top` (default: the largest value)."""
    top = top or max(v for _l, v, _s, _t in rows) or 1
    left, right, step, bar = 150, 690, 30, 18
    height = 16 + step * len(rows)
    parts = ['<figure class="viz">', '<figcaption><strong>%s</strong><span>%s</span></figcaption>' % (_esc(title), _esc(caption)),
             '<div class="viz-scroll"><svg viewBox="0 0 760 %d" role="img" aria-label="%s">' % (height, _esc(aria))]
    for i, (label, value, shown, tip) in enumerate(rows):
        y = 8 + i * step
        width = max(2.0, (right - left) * value / top)
        parts.append('<text class="row" x="%d" y="%.1f" dy="4" text-anchor="end"><tspan class="cat">%s</tspan></text>' % (left - 10, y + bar / 2, _esc(label)))
        parts.append('<g class="hit" data-tip="%s"><rect x="0" y="%d" width="760" height="%d" fill="transparent"/>'
                     '<rect class="bar" x="%d" y="%d" width="%.1f" height="%d" rx="3"/></g>' % (_esc(tip), y - 6, step, left, y, width, bar))
        parts.append('<text class="val" x="%.1f" y="%.1f" dy="4">%s</text>' % (left + width + 6, y + bar / 2, _esc(shown)))
    parts += ['</svg></div>', '</figure>']
    return '\n'.join(parts)


class _Dumper(yaml.SafeDumper):
    def ignore_aliases(self, data):  # the same date twice must not become '&id001' and '*id001'
        return True


def dump(meta):
    return yaml.dump(meta, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=1000)


def document(meta, body):
    return '---\n%s---\n\n%s\n' % (dump(meta), body.strip())


def write_file(lang, slug, meta, body):
    """A committed Markdown guide (content/guides/<lang>/<slug>.md); keeps the first publication date."""
    path = guides._path(lang, slug)
    if os.path.exists(path):
        old = guides._parse(path, os.path.getmtime(path))
        meta = dict(meta, date=old.get('date') or meta['date'])
    with open(path, 'w', encoding='utf-8') as f:
        f.write(document(meta, body))
    return path


def save_row(lang, slug, meta, body):
    """The guide as a Guide row (production data, written on Heroku); keeps the first publication date."""
    from .models import Guide
    meta = dict(meta)
    row = Guide.objects.filter(lang=lang, slug=slug).first()
    values = {'title': meta.pop('title'), 'description': meta.pop('description')[:300], 'body': body.strip() + '\n',
              'updated': meta.pop('updated'), 'draft': False, 'product': False}
    date = meta.pop('date')
    values['date'] = row.date if row else date
    values['extra'] = dump(meta) if meta else ''
    return Guide.objects.update_or_create(lang=lang, slug=slug, defaults=values)[0]


def scope_label(scope, lang):
    from . import trends
    with translation.override(lang):
        if scope == '*':
            return t(lang, 'All of arXiv', 'Tout arXiv')
        return str(trends.scope_name(scope))


# --- fastest-growing research topics (TermWeek) ------------------------------------------------------------

TOPICS_SLUG = 'fastest-growing-research-topics'
WINDOW = 13  # weeks compared: a quarter
YEAR = 52
# (scope, field in English, in French): the question each section answers
TOPIC_SCOPES = [
    ('cs', 'computer science and AI', 'informatique et IA'),
    ('stat', 'statistics', 'statistiques'),
    ('eess', 'electrical engineering and signal processing', 'génie électrique et traitement du signal'),
    ('math', 'mathematics', 'mathématiques'),
    ('quant-ph', 'quantum physics', 'physique quantique'),
    ('cond-mat', 'condensed matter physics', 'physique de la matière condensée'),
    ('astro-ph', 'astrophysics', 'astrophysique'),
    ('hep-ph', 'particle physics', 'physique des particules'),
    ('physics', 'physics (other)', 'physique (autres)'),
    ('q-bio', 'quantitative biology', 'biologie quantitative'),
    ('biorxiv', 'biology (bioRxiv)', 'biologie (bioRxiv)'),
    ('medrxiv', 'medicine (medRxiv)', 'médecine (medRxiv)'),
]
ROWS = 10
MIN_GROWTH = 2.0
MIN_LLR = 20.0
MIN_PAPERS = 25  # papers using a term in the latest weeks, below which a small field's counts are noise
CANDIDATES = 40
REFRESH_DAYS = 28


def _window(scope, weeks):
    from . import trends
    papers, counts, present = 0, {}, {}
    # one week at a time: 13 weeks of all of arXiv at once hold about a million terms, too much for a dyno
    for week in weeks:
        counted = trends.week_counts(scope, [week]).get(week)
        if not counted:
            continue
        total, found = counted
        papers += total
        for term, n in found.items():
            counts[term] = counts.get(term, 0) + n
            present[term] = present.get(term, 0) + 1
    return papers, counts, present


def topic_windows(today=None):
    """(recent weeks, base weeks, 'year' or 'before'), or None without enough history."""
    from . import trends
    from .models import TermWeek
    today = today or timezone.now().date()
    last = trends.closed_week(today)
    stored = sorted(set(TermWeek.objects.filter(scope='*', week__lte=last).values_list('week', flat=True)))
    if len(stored) < WINDOW + 4:
        return None
    recent = stored[-WINDOW:]
    have = set(stored)
    year = [w - datetime.timedelta(weeks=YEAR) for w in recent]
    if sum(w in have for w in year) >= WINDOW - 2:
        return recent, [w for w in year if w in have], 'year'
    before = [w for w in stored if w < recent[0]][-WINDOW:]
    return recent, before, 'before'


def _plain(term):
    from . import trends
    return not all(w in trends.STOP or w in trends.GENERIC for w in term.split()) and not re.fullmatch(r'[\d\s\-]+', term)


def spellings(term):
    """The other spellings of a term: 'language-model' <-> 'language model'."""
    found = set()
    if '-' in term:
        found.add(term.replace('-', ' '))
    if ' ' in term:
        found.add(term.replace(' ', '-'))
    return found - {term}


def variant(term, n, *tallies):
    """Whether another spelling of the term is at least as common in either window: abstracts hyphenate more and
    more compounds ('language-model agents'), which makes 'language-model' rise while 'language model' does not."""
    return any(tally.get(other, 0) >= n for other in spellings(term) for tally in tallies)


def ranked_terms(scope, recent, base):
    """{'papers', 'base_papers', 'growing': [...], 'fading': [...]} of a scope, before the model's check."""
    from . import trends
    papers, counts, _ = _window(scope, recent)
    base_papers, base_counts, base_present = _window(scope, base)
    if papers < 2000 or base_papers < 1000:
        return None
    floor = max(MIN_PAPERS, papers // 2000)
    growing, fading = [], []
    for term, n in counts.items():
        if n < floor or not _plain(term):
            continue
        if variant(term, n, counts, base_counts):
            continue
        # a stored week without the term had 0 or 1 paper using it: count 1, so that a rare term does not look new
        before = base_counts.get(term, 0) + 1.0 * (len(base) - base_present.get(term, 0))
        expected = before / base_papers * papers
        growth = n / expected
        llr = 2 * (n * math.log(n / expected) - (n - expected))
        if growth >= MIN_GROWTH and llr >= MIN_LLR:
            growing.append({'term': term, 'n': n, 'before': base_counts.get(term, 0), 'growth': growth})
    for term, before in base_counts.items():
        if before < 4 * floor or not _plain(term):
            continue
        n = counts.get(term, 0)
        if variant(term, before, base_counts, counts):
            continue
        expected = before / base_papers * papers
        if n and n / expected <= 1 / MIN_GROWTH:
            llr = 2 * (n * math.log(n / expected) - (n - expected))
            if llr >= MIN_LLR:
                fading.append({'term': term, 'n': n, 'before': before, 'growth': n / expected})

    def dedupe(items):
        kept = []
        for item in items:
            # 'gpt-6' and 'gpt-6 astra' with nearly the same count: keep one
            if any((trends.inside(item['term'], o['term']) or trends.inside(o['term'], item['term']))
                   and max(item['n'], o['n']) <= 1.25 * min(item['n'], o['n']) + 2 for o in kept):
                continue
            kept.append(item)
            if len(kept) == CANDIDATES:
                break
        return kept
    growing.sort(key=lambda i: (-i['growth'], -i['n']))
    fading.sort(key=lambda i: (i['growth'], -i['before']))
    return {'papers': papers, 'base_papers': base_papers, 'growing': dedupe(growing), 'fading': dedupe(fading)}


def check_terms(scope, items):
    """{term: {'label', 'gloss': {en, fr}}} for the candidates the model keeps as research topics."""
    from . import generated
    if not items:
        return {}
    where = {'*': 'arXiv', 'biorxiv': 'bioRxiv', 'medrxiv': 'medRxiv'}.get(scope, 'arXiv (%s)' % scope)
    listing = '\n'.join('- "%s"' % i['term'] for i in items)
    answer = generated.chat_json(
        'These terms come from counting the words of the titles and abstracts of %s papers; they are lowercased '
        'and plural words are cut to a stem ("time-sery" means "time series", "galaxy" may mean "galaxies"). Keep '
        'only the terms that name a specific research topic, method, model, object, instrument, dataset, material or '
        'phenomenon a researcher of that field would name as what a paper is about. Drop generic wording ("higher '
        'mean", "evaluated baseline"), general vocabulary of the field that is not a topic by itself ("median", '
        '"routing", "operating point", "absolute error", "magnetic field", "neural network", "gene expression"), '
        'statistical tests and metrics, a spelling variant of a common term, '
        'phrases of a template, formatting, a conference name, a fragment that names nothing on its own, and '
        'anything you are not sure is a real topic. For each kept term give "label": its readable form, as it would '
        'be written in a paper (restore plurals, capitals and acronyms; never add words), and "gloss": what it is, '
        '3 to 9 words, in English and in French, without inventing facts. Answer with a JSON object '
        '{"terms": [{"term": "copied exactly", "label": "...", "gloss": {"en": "...", "fr": "..."}}]} listing '
        'only the kept terms.\n\n%s' % (where, listing), paid=True, max_tokens=6000)
    known = {i['term'] for i in items}
    kept = {}
    for entry in answer.get('terms') or []:
        if not isinstance(entry, dict) or entry.get('term') not in known:
            continue
        gloss = entry.get('gloss') if isinstance(entry.get('gloss'), dict) else {}
        kept[entry['term']] = {'label': ' '.join(str(entry.get('label') or entry['term']).split())[:80],
                               'gloss': {lang: ' '.join(str(gloss.get(lang) or '').split())[:120] for lang in LANGS}}
    return kept


def _distinct(items):
    """One row per thing named: two stems can give the same label ('nuclear recoil'), and an acronym and its full
    name get the same description ('LZ' and 'LUX-ZEPLIN')."""
    seen, kept = set(), []
    for item in items:
        keys = {re.sub(r'[^a-z0-9]', '', item['label'].lower())}
        gloss = re.sub(r'[^a-z0-9]', '', (item.get('gloss') or {}).get('en', '').lower())
        if gloss:
            keys.add('gloss:' + gloss)
        if not keys & seen:
            kept.append(item)
        seen |= keys
    return kept


def topics_data(today=None, check=None):
    """Everything the topics guide shows, or None without enough history."""
    check = check or check_terms
    windows = topic_windows(today)
    if windows is None:
        return None
    recent, base, mode = windows
    sections = []
    for scope, en, fr in [('*', 'all of arXiv', 'tout arXiv')] + TOPIC_SCOPES:
        ranked = ranked_terms(scope, recent, base)
        if not ranked:
            continue
        candidates = ranked['growing'] + (ranked['fading'] if scope == '*' else [])
        kept = check(scope, candidates)
        growing = _distinct([dict(i, **kept[i['term']]) for i in ranked['growing'] if i['term'] in kept])[:ROWS]
        fading = _distinct([dict(i, **kept[i['term']]) for i in ranked['fading'] if i['term'] in kept])[:ROWS] if scope == '*' else []
        if growing:
            sections.append({'scope': scope, 'field': {'en': en, 'fr': fr}, 'papers': ranked['papers'],
                             'base_papers': ranked['base_papers'], 'growing': growing, 'fading': fading})
    if not sections or sections[0]['scope'] != '*':
        return None
    return {'recent': recent, 'base': base, 'mode': mode, 'sections': sections,
            'made': today or timezone.now().date()}


def _span(weeks, lang):
    start, end = weeks[0], weeks[-1] + datetime.timedelta(days=6)
    return t(lang, '%s to %s', 'du %s au %s') % (day(start, lang), day(end, lang))


def _explore(term, lang):
    return link(lang, 'trends/explore/?' + urlencode({'t': term}))


def _term_rows(items, lang):
    rows = []
    for i, item in enumerate(items, 1):
        name = '[%s](%s)' % (item['label'], _explore(item['term'], lang))
        rows.append((i, name, item['gloss'][lang] or '', num(item['n'], lang), num(item['before'], lang),
                     times(item['growth'], lang)))
    return rows


def topics_page(data, lang):
    """(meta, body) of the topics guide in one language."""
    made = data['made']
    recent, base = data['recent'], data['base']
    first = data['sections'][0]
    top = first['growing']
    recent_text, base_text = _span(recent, lang), _span(base, lang)
    yearly = data['mode'] == 'year'
    compare = t(lang, 'the same weeks a year earlier' if yearly else 'the previous weeks',
                'les mêmes semaines un an plus tôt' if yearly else 'les semaines précédentes')
    header = t(lang, ['#', 'Term', 'What it is', 'Papers, latest 13 weeks', 'Papers, ' + ('a year earlier' if yearly else 'before'), 'Share'],
               ['#', 'Terme', 'De quoi il s’agit', 'Articles, 13 dernières semaines', 'Articles, ' + ('un an plus tôt' if yearly else 'avant'), 'Part'])
    names = ', '.join('%s (%s)' % (i['label'], times(i['growth'], lang)) for i in top[:3])
    by_field = {s['scope']: s for s in data['sections']}

    takeaways = [t(lang, '**%s** papers submitted to arXiv from %s, compared with **%s** over %s.',
                   '**%s** articles soumis à arXiv %s, comparés à **%s** sur %s.')
                 % (num(first['papers'], lang), recent_text, num(first['base_papers'], lang), compare)]
    takeaways += [t(lang, '**%s**: %s papers, share of all papers %s.', '**%s** : %s articles, part de l’ensemble %s.')
                  % (i['label'], num(i['n'], lang), times(i['growth'], lang)) for i in top[:3]]
    for scope in ('cs', 'astro-ph', 'quant-ph', 'biorxiv'):
        section = by_field.get(scope)
        if section and section['growing']:
            item = section['growing'][0]
            takeaways.append(t(lang, 'Fastest-growing in %s: **%s** (%s).', 'Plus forte progression en %s : **%s** (%s).')
                             % (section['field'][lang], item['label'], times(item['growth'], lang)))
    body = ['<div class="takeaways" markdown="1">\n\n%s\n\n%s\n\n</div>' % (
        t(lang, 'Key numbers', 'Chiffres clés'), '\n'.join('- ' + line for line in takeaways))]
    body.append('<p class="note">%s</p>' % t(
        lang, 'Counted by SummarizePaper from the titles and abstracts of every new arXiv, bioRxiv and medRxiv paper. '
              'Latest 13 weeks: %s; compared with %s (%s). Updated every month; last update: %s.',
              'Calculé par SummarizePaper à partir des titres et résumés de chaque nouvel article arXiv, bioRxiv et medRxiv. '
              '13 dernières semaines : %s ; comparées à %s (%s). Mise à jour chaque mois ; dernière mise à jour : %s.')
        % (recent_text, compare, base_text, day(made, lang)))
    body.append(t(lang, '## What are the fastest-growing research topics on arXiv?',
                  '## Quels sujets de recherche progressent le plus sur arXiv ?'))
    body.append(t(lang, 'The terms whose share of all new arXiv papers grew most, comparing the latest 13 weeks with %s. '
                        '"Share" is how many times more often papers use the term, once arXiv\'s own growth is taken out.',
                  'Les termes dont la part dans l’ensemble des nouveaux articles arXiv a le plus augmenté, entre les 13 dernières '
                  'semaines et %s. La « part » indique combien de fois plus souvent les articles emploient le terme, '
                  'une fois retirée la croissance d’arXiv elle-même.') % compare)
    body.append(table(header, _term_rows(top, lang)))
    for section in data['sections'][1:]:
        field = section['field'][lang]
        body.append(t(lang, '## Which topics are growing fastest in %s?', '## Quels sujets progressent le plus en %s ?') % field)
        body.append(t(lang, '%s: %s papers in the latest 13 weeks, %s over %s.', '%s : %s articles sur les 13 dernières semaines, %s sur %s.')
                    % (scope_label(section['scope'], lang), num(section['papers'], lang), num(section['base_papers'], lang), compare))
        body.append(table(header, _term_rows(section['growing'], lang)))
    if first['fading']:
        body.append(t(lang, '## Which research topics are losing ground?', '## Quels sujets de recherche reculent ?'))
        body.append(t(lang, 'Terms that new arXiv papers use at most half as often as in %s.',
                      'Les termes que les nouveaux articles arXiv emploient au moins deux fois moins souvent que sur %s.') % compare)
        body.append(table(header, _term_rows(first['fading'], lang)))
    body.append(t(lang, '## How to follow a topic', '## Comment suivre un sujet'))
    body.append(t(lang, '''- Each term links to its weekly curve in the [rising topics explorer](%s), free for all of arXiv.
- The [rising topics](%s) page names the topics that jump each week in every arXiv category, bioRxiv and medRxiv; see [how to read it](%s).
- The [daily paper alert](%s) sends the new papers of your categories every weekday, each with a two-sentence summary.''',
                  '''- Chaque terme renvoie à sa courbe hebdomadaire dans l’[explorateur des sujets en hausse](%s), gratuit pour tout arXiv.
- La page des [sujets en hausse](%s) nomme chaque semaine les sujets qui bondissent dans chaque catégorie d’arXiv, bioRxiv et medRxiv ; voir [comment la lire](%s).
- L’[alerte quotidienne](%s) envoie chaque jour ouvré les nouveaux articles de vos catégories, chacun avec un résumé en deux phrases.''')
                % (link(lang, 'trends/explore/'), link(lang, 'trends/'), link(lang, 'guides/rising-research-topics/'), link(lang, 'digest/')))
    body.append(related(lang, TOPICS_SLUG))
    body.append(t(lang, '## Methodology and limits', '## Méthode et limites'))
    body.append(t(lang, '''- **Papers**: every new arXiv paper, dated by the submission of its first version (arXiv OAI-PMH), and the new bioRxiv and medRxiv preprints. A paper counts in each archive of its categories.
- **Terms**: words and phrases of 1 to 3 words of the title and abstract, lowercased, with plurals cut to a stem; the readable form shown is restored afterwards. Terms a single paper used in a week are not stored, so a term absent from a stored week counts as one paper. Spelling variants ("language-model" for "language model") are left out when the other spelling is at least as common.
- **Share**: papers using the term divided by all papers of the scope, in the latest 13 weeks, divided by the same ratio over %s. Listed terms at least doubled their share and pass a likelihood-ratio test (far from chance).
- **Filtering**: the counts choose the terms. A language model only removes candidates that are not research topics (generic wording, template phrases) and writes the readable form and the short description.
- Words are not topics: a new name for an old idea rises too, and a term can grow because of one large collaboration.''',
                  '''- **Articles** : chaque nouvel article arXiv, daté par la soumission de sa première version (OAI-PMH d’arXiv), et les nouveaux preprints bioRxiv et medRxiv. Un article compte dans chaque archive de ses catégories.
- **Termes** : mots et expressions de 1 à 3 mots du titre et du résumé, en minuscules, pluriels ramenés à une racine ; la forme lisible affichée est rétablie ensuite. Les termes qu’un seul article a employés dans une semaine ne sont pas conservés : un terme absent d’une semaine compte pour un article. Les variantes d’orthographe (« language-model » pour « language model ») sont écartées quand l’autre graphie est au moins aussi fréquente.
- **Part** : articles employant le terme divisés par l’ensemble des articles du périmètre sur les 13 dernières semaines, rapportés au même ratio sur %s. Les termes listés ont au moins doublé leur part et passent un test du rapport de vraisemblance (loin du hasard).
- **Filtrage** : les comptages choisissent les termes. Un modèle de langage retire seulement les candidats qui ne sont pas des sujets de recherche (formulations génériques, phrases toutes faites) et écrit la forme lisible et la courte description.
- Un mot n’est pas un sujet : un nouveau nom pour une vieille idée progresse aussi, et un terme peut grimper à cause d’une seule grande collaboration.''') % compare)
    body.append(cta(lang))

    faq = [{'q': t(lang, 'What are the fastest-growing research topics on arXiv?', 'Quels sont les sujets de recherche qui progressent le plus sur arXiv ?'),
            'a': t(lang, 'From %s, the terms whose share of new arXiv papers grew most compared with %s were %s.',
                   '%s, les termes dont la part dans les nouveaux articles arXiv a le plus augmenté par rapport à %s étaient %s.')
            % (recent_text[0].upper() + recent_text[1:] if lang == 'fr' else recent_text, compare, names)}]
    for scope in ('cs', 'astro-ph', 'quant-ph'):
        section = by_field.get(scope)
        if section:
            faq.append({'q': t(lang, 'Which topics are growing fastest in %s?', 'Quels sujets progressent le plus en %s ?') % section['field'][lang],
                        'a': t(lang, 'In the latest 13 weeks: %s.', 'Sur les 13 dernières semaines : %s.')
                        % ', '.join('%s (%s)' % (i['label'], times(i['growth'], lang)) for i in section['growing'][:3])})
    faq.append({'q': t(lang, 'How is the growth of a research topic measured?', 'Comment mesure-t-on la progression d’un sujet de recherche ?'),
                'a': t(lang, 'By the share of new papers whose title or abstract uses the term, in the latest 13 weeks compared with %s, so that the growth of arXiv itself does not count.',
                       'Par la part des nouveaux articles dont le titre ou le résumé emploie le terme, sur les 13 dernières semaines comparées à %s, pour que la croissance d’arXiv elle-même ne compte pas.') % compare})
    meta = {
        'title': t(lang, 'Fastest-growing research topics on arXiv (%s)', 'Les sujets de recherche qui progressent le plus sur arXiv (%s)') % month(made, lang),
        'description': t(lang, 'The terms whose share of new arXiv, bioRxiv and medRxiv papers grew most, by field, counted from every new paper. Updated monthly.',
                         'Les termes dont la part dans les nouveaux articles arXiv, bioRxiv et medRxiv a le plus augmenté, par domaine. Mis à jour chaque mois.'),
        'date': made, 'updated': made, 'faq': faq,
        'dataset': {'name': t(lang, 'Fastest-growing terms in arXiv, bioRxiv and medRxiv papers', 'Termes en plus forte progression dans les articles arXiv, bioRxiv et medRxiv'),
                    'description': t(lang, 'Weekly counts of the terms of the titles and abstracts of new papers, compared over 13 weeks.',
                                     'Comptages hebdomadaires des termes des titres et résumés des nouveaux articles, comparés sur 13 semaines.'),
                    'temporalCoverage': '%s/%s' % (base[0].isoformat(), (recent[-1] + datetime.timedelta(days=6)).isoformat())},
        'sources': [{'label': 'arXiv OAI-PMH', 'url': 'https://info.arxiv.org/help/oa/index.html'},
                    {'label': 'bioRxiv and medRxiv API', 'url': 'https://api.biorxiv.org/'}],
    }
    return meta, '\n\n'.join(body)


def write_topics(today=None, check=None):
    """Compute and save the topics guide in both languages; False without enough history."""
    data = topics_data(today, check)
    if data is None:
        logger.info('Topics guide: not enough history')
        return False
    for lang in LANGS:
        meta, body = topics_page(data, lang)
        save_row(lang, TOPICS_SLUG, meta, body)
    return True


def refresh_due(now=None):
    """The hourly run: rewrite the topics guide once its row is REFRESH_DAYS old (or missing)."""
    from .models import Guide
    now = now or timezone.now()
    row = Guide.objects.filter(lang='en', slug=TOPICS_SLUG).first()
    if row and row.updated and (now.date() - row.updated).days < REFRESH_DAYS:
        return False
    # without enough history, or after a failure, try again the next day rather than every hour
    from django.core.cache import cache
    if not cache.add('data-guides-topics-tried', 1, 86400):
        return False
    return write_topics(now.date())


# --- arXiv metadata: authors per paper, journal versions -----------------------------------------------------

AUTHORS_SLUG = 'authors-per-paper'
COLLABORATIONS = ('hep-ex', 'nucl')
JOURNALS_SLUG = 'arxiv-papers-published-in-journals'
# the archives shown, in this order; physics.* and the small ones are grouped as in arXiv's own statistics
FIELDS = [
    ('cs', 'Computer science', 'Informatique'),
    ('math', 'Mathematics', 'Mathématiques'),
    ('astro-ph', 'Astrophysics', 'Astrophysique'),
    ('cond-mat', 'Condensed matter', 'Matière condensée'),
    ('quant-ph', 'Quantum physics', 'Physique quantique'),
    ('hep-ph', 'High-energy physics, phenomenology', 'Physique des hautes énergies, phénoménologie'),
    ('hep-th', 'High-energy physics, theory', 'Physique des hautes énergies, théorie'),
    ('hep-ex', 'High-energy physics, experiment', 'Physique des hautes énergies, expérience'),
    ('gr-qc', 'General relativity and quantum cosmology', 'Relativité générale et cosmologie quantique'),
    ('nucl', 'Nuclear physics', 'Physique nucléaire'),
    ('physics', 'Other physics', 'Autres domaines de la physique'),
    ('math-ph', 'Mathematical physics', 'Physique mathématique'),
    ('nlin', 'Nonlinear sciences', 'Sciences non linéaires'),
    ('stat', 'Statistics', 'Statistiques'),
    ('eess', 'Electrical engineering and systems science', 'Génie électrique et science des systèmes'),
    ('q-bio', 'Quantitative biology', 'Biologie quantitative'),
    ('q-fin', 'Quantitative finance', 'Finance quantitative'),
    ('econ', 'Economics', 'Économie'),
]
FIELD_NAMES = {code: (en, fr) for code, en, fr in FIELDS}
SOURCES = [{'label': 'arXiv OAI-PMH interface', 'url': 'https://info.arxiv.org/help/oa/index.html'},
           {'label': 'arXiv metadata format (arXiv OAI schema)', 'url': 'https://oaipmh.arxiv.org/OAI/arXiv.xsd'}]


def field_of(categories):
    """The field of a paper: the archive of its primary (first) category, nucl-ex and nucl-th as 'nucl'."""
    if not categories:
        return None
    archive = categories[0].split('.')[0]
    if archive in ('nucl-ex', 'nucl-th'):
        return 'nucl'
    if archive in ('hep-lat',):
        return 'hep-th'
    return archive if archive in FIELD_NAMES else None


def year_of(arxiv_id):
    """2025 for '2501.01234'; None for old-style ids (before 2007), which these cohorts do not need."""
    match = re.match(r'(\d\d)(\d\d)\.\d{4,5}$', arxiv_id)
    return 2000 + int(match.group(1)) if match else None


def cohorts(records, years):
    """{year: [records submitted that year (first version, from the id)]}."""
    found = defaultdict(list)
    for record in records.values():
        year = year_of(record['id'])
        if year in years:
            found[year].append(record)
    return found


def author_stats(papers):
    papers = [p for p in papers if p['authors'] > 0]
    counts = [p['authors'] for p in papers]
    if not counts:
        return None
    n = len(counts)
    # a paper signed by one collaboration name ('groups' harvested since November 2026) is not a single-author paper
    solo = sum(p['authors'] == 1 and not p.get('groups') for p in papers)
    return {'papers': n, 'median': statistics.median(counts), 'mean': sum(counts) / n,
            'solo': 100 * solo / n, 'ten': 100 * sum(c >= 10 for c in counts) / n,
            'groups': 100 * sum(bool(p.get('groups')) for p in papers) / n,
            'hundred': 100 * sum(c >= 100 for c in counts) / n, 'max': max(counts),
            'bins': Counter(min(c, 11) if c <= 10 else (11 if c < 100 else 12) for c in counts)}


def authors_data(records, year, earlier):
    by_year = cohorts(records, {year, earlier})
    data = {'year': year, 'earlier': earlier, 'all': author_stats(by_year[year]), 'all_earlier': author_stats(by_year[earlier]),
            'fields': [], 'groups_known': all('groups' in p for p in by_year[year][:1000])}
    for code, _en, _fr in FIELDS:
        now = author_stats([p for p in by_year[year] if field_of(p['cats']) == code])
        before = author_stats([p for p in by_year[earlier] if field_of(p['cats']) == code])
        if now and now['papers'] >= 200:
            data['fields'].append({'code': code, 'now': now, 'before': before})
    biggest = sorted(by_year[year], key=lambda p: -p['authors'])[:5]
    data['biggest'] = [{'id': p['id'], 'authors': p['authors'], 'field': field_of(p['cats']), 'title': p.get('title', '')} for p in biggest]
    return data


def _field(code, lang):
    en, fr = FIELD_NAMES[code]
    shown = {'nucl': 'nucl-ex, nucl-th', 'physics': 'physics.*'}.get(code, code)
    return '%s (%s)' % (t(lang, en, fr), shown)


def _median(x, lang):
    return dec(x, lang, 1) if x != int(x) else num(x, lang)


def authors_page(data, lang, made):
    year, earlier = data['year'], data['earlier']
    a, b = data['all'], data['all_earlier']
    fields = sorted(data['fields'], key=lambda f: (-f['now']['median'], -f['now']['mean']))
    most, least = fields[0], fields[-1]
    # without the group flag, collaborations signing as one name make high-energy experiment look single-authored
    masked = () if data.get('groups_known') else COLLABORATIONS
    solo = sorted([f for f in fields if f['code'] not in masked], key=lambda f: -f['now']['solo'])[0]

    def solo_cell(f, key):
        return '—' if f['code'] in masked else pct(f[key]['solo'], lang)
    takeaways = [
        t(lang, 'An arXiv paper submitted in %d had **%s authors** at the median (mean %s), over %s papers.',
          'Un article arXiv soumis en %d comptait **%s auteurs** en médiane (moyenne %s), sur %s articles.')
        % (year, _median(a['median'], lang), dec(a['mean'], lang), num(a['papers'], lang)),
        t(lang, '**%s** of %d papers had a single author, against %s in %d.', '**%s** des articles de %d n’avaient qu’un seul auteur, contre %s en %d.')
        % (pct(a['solo'], lang, 1), year, pct(b['solo'], lang, 1), earlier),
        t(lang, 'Most authors: **%s**, median %s.', 'Le plus d’auteurs : **%s**, médiane %s.') % (_field(most['code'], lang), _median(most['now']['median'], lang)),
        t(lang, 'Fewest authors: **%s**, median %s.', 'Le moins d’auteurs : **%s**, médiane %s.') % (_field(least['code'], lang), _median(least['now']['median'], lang)),
        t(lang, 'Most single-author papers: **%s**, %s.', 'Le plus d’articles à un seul auteur : **%s**, %s.') % (_field(solo['code'], lang), pct(solo['now']['solo'], lang)),
    ]
    body = ['<div class="takeaways" markdown="1">\n\n%s\n\n%s\n\n</div>' % (t(lang, 'Key numbers', 'Chiffres clés'), '\n'.join('- ' + x for x in takeaways))]
    body.append('<p class="note">%s</p>' % t(lang, 'Counted by SummarizePaper from the metadata of every arXiv paper submitted in %d and %d (arXiv OAI-PMH, harvested on %s).',
                                              'Calculé par SummarizePaper à partir des métadonnées de chaque article arXiv soumis en %d et %d (OAI-PMH d’arXiv, récupérées le %s).')
                % (earlier, year, day(made, lang)))
    body.append(t(lang, '## How many authors does a research paper have?', '## Combien d’auteurs compte un article de recherche ?'))
    body.append(t(lang, 'Across the %s papers submitted to arXiv in %d, the median paper had %s authors and the mean was %s. The mean is pulled up by large collaborations: %s of papers had 10 authors or more, and %s had 100 or more.',
                  'Sur les %s articles soumis à arXiv en %d, l’article médian comptait %s auteurs et la moyenne était de %s. La moyenne est tirée vers le haut par les grandes collaborations : %s des articles avaient 10 auteurs ou plus, et %s en avaient 100 ou plus.')
                % (num(a['papers'], lang), year, _median(a['median'], lang), dec(a['mean'], lang), pct(a['ten'], lang, 1), pct(a['hundred'], lang, 2)))
    labels = [str(i) for i in range(1, 11)] + [t(lang, '11–99', '11 à 99'), t(lang, '100 or more', '100 ou plus')]
    rows = [(labels[i - 1], num(a['bins'].get(i, 0), lang), pct(100 * a['bins'].get(i, 0) / a['papers'], lang, 1)) for i in range(1, 13)]
    body.append(bar_chart(t(lang, 'arXiv papers by number of authors, %d', 'Articles arXiv selon le nombre d’auteurs, %d') % year,
                          t(lang, 'Share of the %s papers submitted in %d. Source: arXiv OAI-PMH metadata.', 'Part des %s articles soumis en %d. Source : métadonnées OAI-PMH d’arXiv.') % (num(a['papers'], lang), year),
                          [(labels[i - 1], a['bins'].get(i, 0), pct(100 * a['bins'].get(i, 0) / a['papers'], lang, 1),
                            '%s %s: %s %s' % (labels[i - 1], t(lang, 'authors', 'auteurs'), num(a['bins'].get(i, 0), lang), t(lang, 'papers', 'articles'))) for i in range(1, 13)],
                          t(lang, 'Bar chart of arXiv papers submitted in %d by number of authors: %s had 3 authors, the most common count, and %s had one.',
                            'Graphique en barres des articles arXiv soumis en %d selon le nombre d’auteurs : %s en avaient 3, le nombre le plus fréquent, et %s un seul.')
                          % (year, pct(100 * a['bins'].get(3, 0) / a['papers'], lang, 1), pct(100 * a['bins'].get(1, 0) / a['papers'], lang, 1))))
    body.append(t(lang, 'Papers submitted in %d by number of authors:', 'Articles soumis en %d selon le nombre d’auteurs :') % year)
    body.append(table(t(lang, ['Authors', 'Papers', 'Share'], ['Auteurs', 'Articles', 'Part']), rows))
    body.append(t(lang, '## Which fields have the most authors per paper?', '## Quels domaines comptent le plus d’auteurs par article ?'))
    rows = [(_field(f['code'], lang), num(f['now']['papers'], lang), _median(f['now']['median'], lang), dec(f['now']['mean'], lang),
             solo_cell(f, 'now'), pct(f['now']['ten'], lang)) for f in fields]
    body.append(t(lang, 'By the primary category of papers submitted in %d, ordered by median:', 'Selon la catégorie principale des articles soumis en %d, par médiane décroissante :') % year)
    body.append(table(t(lang, ['Field', 'Papers', 'Median authors', 'Mean', 'Single author', '10 authors or more'],
                        ['Domaine', 'Articles', 'Médiane d’auteurs', 'Moyenne', 'Un seul auteur', '10 auteurs ou plus']), rows))
    if masked:
        body.append(t(lang, '— High-energy experiment and nuclear physics collaborations often sign as one name on arXiv ("ATLAS Collaboration"), so their single-author share is not shown.',
                      '— Les collaborations de physique des hautes énergies expérimentale et de physique nucléaire signent souvent d’un seul nom sur arXiv (« ATLAS Collaboration ») : leur part d’articles à un seul auteur n’est pas indiquée.'))
    body.append(t(lang, '## Are papers written by more authors than before?', '## Les articles ont-ils plus d’auteurs qu’avant ?'))
    body.append(t(lang, 'Over all of arXiv, the median went from %s authors in %d to %s in %d, the share of papers with 10 authors or more went from %s to %s, and the share of single-author papers from %s to %s. Medians and shares are used here rather than means, which a few papers with thousands of authors can move: one gravitational-wave paper with 1,800 authors weighs as much as 900 two-author papers.',
                  'Sur tout arXiv, la médiane est passée de %s auteurs en %d à %s en %d, la part des articles à 10 auteurs ou plus de %s à %s, et celle des articles à un seul auteur de %s à %s. On compare ici des médianes et des parts plutôt que des moyennes, que quelques articles à plusieurs milliers d’auteurs suffisent à déplacer : un article sur les ondes gravitationnelles à 1 800 auteurs pèse autant que 900 articles à deux auteurs.')
                % (_median(b['median'], lang), earlier, _median(a['median'], lang), year, pct(b['ten'], lang, 1), pct(a['ten'], lang, 1), pct(b['solo'], lang, 1), pct(a['solo'], lang, 1)))
    rows = [(_field(f['code'], lang), _median(f['before']['median'], lang), _median(f['now']['median'], lang), pct(f['before']['ten'], lang), pct(f['now']['ten'], lang),
             solo_cell(f, 'before'), solo_cell(f, 'now'))
            for f in sorted(fields, key=lambda f: -f['now']['papers']) if f['before']]
    body.append(table(t(lang, ['Field', 'Median %d' % earlier, 'Median %d' % year, '10+ authors %d' % earlier, '10+ authors %d' % year, 'Single author %d' % earlier, 'Single author %d' % year],
                        ['Domaine', 'Médiane %d' % earlier, 'Médiane %d' % year, '10 auteurs ou plus %d' % earlier, '10 auteurs ou plus %d' % year, 'Un seul auteur %d' % earlier, 'Un seul auteur %d' % year]), rows))
    body.append(t(lang, '## Which papers had the most authors?', '## Quels articles avaient le plus d’auteurs ?'))
    body.append(t(lang, 'The %d arXiv papers with the longest author lists:', 'Les articles arXiv de %d avec les plus longues listes d’auteurs :') % year)
    body.append('\n'.join('- [%s](%s)%s — %s %s, %s' % (_esc(p['title']).replace('[', '(').replace(']', ')') if p.get('title') else 'arXiv:' + p['id'],
                                                        link(lang, 'paper/%s/' % p['id']), ' (arXiv:%s)' % p['id'] if p.get('title') else '',
                                                        num(p['authors'], lang), t(lang, 'authors', 'auteurs'),
                                                        _field(p['field'], lang) if p['field'] else '') for p in data['biggest']))
    body.append(t(lang, 'Read any of them in minutes: each link opens a summary on SummarizePaper. To see who works on a topic rather than who signed one paper, [Who works on it?](%s) maps the teams behind a research topic.',
                  'Lisez-les en quelques minutes : chaque lien ouvre un résumé sur SummarizePaper. Pour savoir qui travaille sur un sujet plutôt que qui a signé un article, [Qui travaille dessus ?](%s) cartographie les équipes derrière un sujet de recherche.')
                % link(lang, 'guides/find-research-teams-by-topic/'))
    body.append(related(lang, AUTHORS_SLUG))
    body.append(t(lang, '## Methodology and limits', '## Méthode et limites'))
    body.append(t(lang, '''- **Papers**: every arXiv paper whose identifier starts with %s or %s (submitted that year), from the arXiv metadata of arXiv's OAI-PMH interface. New versions of a paper do not count again. The yearly totals match arXiv's own monthly submission statistics.
- **Authors**: the names in the arXiv author list of the latest version. Some large collaborations sign as one name ("ATLAS Collaboration") on arXiv, so high-energy experiment counts are lower than in the journal versions.
- **Field**: the archive of the primary category; physics.* is "physics (other)", nucl-ex and nucl-th are nuclear physics, hep-lat is counted with hep-th. Fields with fewer than 200 papers are left out of the table, not of the totals.
- The median is the middle paper: half the papers have as many authors or fewer.''',
                  '''- **Articles** : chaque article arXiv dont l’identifiant commence par %s ou %s (soumis cette année-là), d’après les métadonnées de l’interface OAI-PMH d’arXiv. Les nouvelles versions d’un article ne comptent pas à nouveau. Les totaux annuels correspondent aux statistiques mensuelles de soumissions publiées par arXiv.
- **Auteurs** : les noms de la liste d’auteurs arXiv de la dernière version. Certaines grandes collaborations signent d’un seul nom (« ATLAS Collaboration ») sur arXiv : les comptes de physique des hautes énergies expérimentale sont donc plus bas que dans les versions publiées en revue.
- **Domaine** : l’archive de la catégorie principale ; physics.* forme « physique (autres) », nucl-ex et nucl-th la physique nucléaire, hep-lat est compté avec hep-th. Les domaines de moins de 200 articles sont absents du tableau, pas des totaux.
- La médiane est l’article du milieu : la moitié des articles ont autant d’auteurs ou moins.''') % (str(earlier)[2:] + 'xx', str(year)[2:] + 'xx'))
    body.append(cta(lang))
    faq = [
        {'q': t(lang, 'How many authors does a scientific paper have on average?', 'Combien d’auteurs compte un article scientifique en moyenne ?'),
         'a': t(lang, 'Papers submitted to arXiv in %d had %s authors on average and %s at the median; %s had a single author.',
                'Les articles soumis à arXiv en %d comptaient %s auteurs en moyenne et %s en médiane ; %s n’avaient qu’un seul auteur.')
         % (year, dec(a['mean'], lang), _median(a['median'], lang), pct(a['solo'], lang, 1))},
        {'q': t(lang, 'Which field has the most authors per paper?', 'Quel domaine compte le plus d’auteurs par article ?'),
         'a': t(lang, 'On arXiv in %d, %s, with a median of %s authors per paper. %s had the fewest, with a median of %s.',
                'Sur arXiv en %d, %s, avec une médiane de %s auteurs par article. %s en comptait le moins, avec une médiane de %s.')
         % (year, _field(most['code'], lang), _median(most['now']['median'], lang), _field(least['code'], lang), _median(least['now']['median'], lang))},
        {'q': t(lang, 'How common are single-author papers?', 'Les articles à un seul auteur sont-ils fréquents ?'),
         'a': t(lang, '%s of arXiv papers submitted in %d had a single author (%s in %d). They are most common in %s (%s).',
                '%s des articles soumis à arXiv en %d n’avaient qu’un seul auteur (%s en %d). Ils sont les plus fréquents en %s (%s).')
         % (pct(a['solo'], lang, 1), year, pct(b['solo'], lang, 1), earlier, _field(solo['code'], lang), pct(solo['now']['solo'], lang))},
    ]
    meta = {'title': t(lang, 'How many authors per paper? arXiv data by field (%d)', 'Combien d’auteurs par article ? Les données d’arXiv par domaine (%d)') % year,
            'description': t(lang, 'Median and mean number of authors of arXiv papers by field, single-author papers and large collaborations, from every paper submitted in %d.',
                             'Nombre médian et moyen d’auteurs des articles arXiv par domaine, articles à un seul auteur et grandes collaborations, sur tous les articles de %d.') % year,
            'date': made, 'updated': made, 'faq': faq,
            'dataset': {'name': t(lang, 'Authors per arXiv paper by field, %d and %d', 'Auteurs par article arXiv selon le domaine, %d et %d') % (earlier, year),
                        'description': t(lang, 'Number of authors of every arXiv paper submitted in %d and %d, by primary category.',
                                         'Nombre d’auteurs de chaque article arXiv soumis en %d et %d, par catégorie principale.') % (earlier, year),
                        'temporalCoverage': '%d-01-01/%d-12-31' % (earlier, year)},
            'sources': SOURCES}
    return meta, '\n\n'.join(body)


# journal names by their letters only ('Astrophys.J.' and 'The Astrophysical Journal' -> 'astrophysj', ...)
JOURNAL_ALIASES = {
    'MNRAS': ('mnras', 'monnotroyastronsoc', 'monthlynoticesoftheroyalastronomicalsociety', 'monthlynoticesroyalastronomicalsociety'),
    'Astrophys. J.': ('apj', 'astrophysj', 'astrophysicaljournal', 'theastrophysicaljournal'),
    'Astrophys. J. Lett.': ('apjl', 'apjlett', 'astrophysjlett', 'astrophysicaljournalletters', 'theastrophysicaljournalletters'),
    'Astrophys. J. Suppl.': ('apjs', 'astrophysjsuppl', 'astrophysjsupplser', 'astrophysicaljournalsupplementseries', 'theastrophysicaljournalsupplementseries'),
    'Astron. Astrophys.': ('aa', 'astronastrophys', 'astronomyastrophysics', 'astronomyandastrophysics'),
    'Astron. J.': ('aj', 'astronj', 'astronomicaljournal', 'theastronomicaljournal'),
    'JCAP': ('jcap', 'jcosmolastropartphys', 'journalofcosmologyandastroparticlephysics'),
    'JHEP': ('jhep', 'jhighenergyphys', 'journalofhighenergyphysics'),
    'Phys. Lett. B': ('plb', 'physletb', 'physicslettersb', 'physicslettersbsection'),
    'Phys. Lett. A': ('pla', 'physletta', 'physicslettersa'),
    'Nucl. Phys. B': ('nuclphysb', 'nuclearphysicsb'),
    'Eur. Phys. J. C': ('epjc', 'eurphysjc', 'europeanphysicaljournalc', 'theeuropeanphysicaljournalc'),
    'Class. Quantum Grav.': ('cqg', 'classquantgrav', 'classquantumgrav', 'classicalandquantumgravity'),
    'Nature': ('nature',), 'Science': ('science',),
    'Nature Astron.': ('natastron', 'natureastronomy', 'natureastron'),
    'Nature Commun.': ('natcommun', 'naturecommunications', 'naturecommun'),
    'Nature Phys.': ('natphys', 'naturephysics', 'naturephys'),
    'Sci. Rep.': ('scirep', 'scientificreports'),
    'IEEE Access': ('ieeeaccess',),
    'Quantum': ('quantum',),
    'J. Phys. Soc. Jpn.': ('jphyssocjpn', 'journalofthephysicalsocietyofjapan'),
    'Opt. Lett.': ('optlett', 'opticsletters'),
    'Opt. Express': ('optexpress', 'opticsexpress'),
    'Universe': ('universe',),
    'Entropy': ('entropy',),
    'EPTCS': ('eptcs', 'electronicproceedingsintheoreticalcomputerscience'),
    'SIGMA': ('sigma', 'symmetryintegrabilityandgeometrymethodsandapplications'),
    'JINST': ('jinst', 'journalofinstrumentation'),
    'SciPost Phys.': ('scipostphys', 'scipostphysics'),
}
JOURNAL_KEYS = {key: name for name, keys in JOURNAL_ALIASES.items() for key in keys}
PHYS_REV = re.compile(r'^(?:phys\.?\s*rev\.?|physical\s+review)\s*(letters|lett\b\.?|[a-e]\b|x\b|research|applied|materials|fluids|accel)?', re.I)
PHYS_REV_PARTS = {'letters': 'Lett.', 'lett': 'Lett.', 'lett.': 'Lett.', 'research': 'Research', 'applied': 'Applied',
                  'materials': 'Materials', 'fluids': 'Fluids', 'accel': 'Accel. Beams', 'accelbeams': 'Accel. Beams'}
# (?![a-z]) rather than \b, so that 'NeurIPS2022' matches too
CONFERENCES = [(re.compile(r'\b(?:%s)(?![a-z])' % pattern, re.I), name) for pattern, name in (
    ('neurips|nips', 'NeurIPS'), ('neural information processing systems', 'NeurIPS'), ('icml', 'ICML'),
    ('international conference on machine learning', 'ICML'), ('iclr', 'ICLR'),
    ('international conference on learning representations', 'ICLR'), ('cvpr', 'CVPR'),
    ('computer vision and pattern recognition', 'CVPR'), ('iccv', 'ICCV'), ('eccv', 'ECCV'), ('aaai', 'AAAI'),
    ('ijcai', 'IJCAI'), ('emnlp', 'EMNLP'), ('naacl', 'NAACL'), ('acl', 'ACL'), ('coling', 'COLING'),
    ('interspeech', 'Interspeech'), ('icassp', 'ICASSP'), ('icra', 'ICRA'), ('iros', 'IROS'), ('kdd', 'KDD'),
    ('miccai', 'MICCAI'), ('aistats', 'AISTATS'), ('uai', 'UAI'), ('colt', 'COLT'), ('stoc', 'STOC'), ('focs', 'FOCS'),
    ('soda', 'SODA'), ('chi', 'CHI'), ('www|the web conference', 'WWW'), ('sigir', 'SIGIR'))]
# journals read from a DOI when the journal reference is missing or unreadable (prefix of the lowercased DOI)
DOI_JOURNALS = [(re.compile(pattern), name) for pattern, name in (
    (r'^10\.1093/mnras', 'MNRAS'), (r'^10\.3847/1538-4357/', 'Astrophys. J.'), (r'^10\.3847/2041-8213/', 'Astrophys. J. Lett.'),
    (r'^10\.3847/1538-4365/', 'Astrophys. J. Suppl.'), (r'^10\.3847/1538-3881/', 'Astron. J.'), (r'^10\.3847/psj/', 'Planet. Sci. J.'),
    (r'^10\.1051/0004-6361/', 'Astron. Astrophys.'), (r'^10\.1088/1475-7516/', 'JCAP'), (r'^10\.1007/jhep', 'JHEP'),
    (r'^10\.1140/epjc/', 'Eur. Phys. J. C'), (r'^10\.1140/epja/', 'Eur. Phys. J. A'), (r'^10\.1140/epjp/', 'Eur. Phys. J. Plus'),
    (r'^10\.1016/j\.physletb', 'Phys. Lett. B'), (r'^10\.1016/j\.physleta', 'Phys. Lett. A'), (r'^10\.1016/j\.nuclphysb', 'Nucl. Phys. B'),
    (r'^10\.1016/j\.physa\.', 'Physica A'), (r'^10\.1016/j\.physd\.', 'Physica D'), (r'^10\.1016/j\.aim\.', 'Advances in Mathematics'),
    (r'^10\.1016/j\.dark\.', 'Phys. Dark Univ.'), (r'^10\.1016/j\.jcp\.', 'J. Comput. Phys.'), (r'^10\.1016/j\.patcog', 'Pattern Recognition'),
    (r'^10\.1038/s41467', 'Nature Commun.'), (r'^10\.1038/s41586', 'Nature'), (r'^10\.1038/s41550', 'Nature Astron.'),
    (r'^10\.1038/s41567', 'Nature Phys.'), (r'^10\.1038/s41598', 'Sci. Rep.'), (r'^10\.1038/s41534', 'npj Quantum Inf.'),
    (r'^10\.1038/s41566', 'Nature Photon.'), (r'^10\.1038/s42005', 'Commun. Phys.'), (r'^10\.1126/sciadv', 'Sci. Adv.'),
    (r'^10\.1126/science', 'Science'), (r'^10\.1073/pnas', 'PNAS'), (r'^10\.22331/q-', 'Quantum'), (r'^10\.21468/scipostphys\.', 'SciPost Phys.'),
    (r'^10\.1103/prxquantum', 'PRX Quantum'), (r'^10\.1364/oe\.', 'Opt. Express'), (r'^10\.1364/ol\.', 'Opt. Lett.'),
    (r'^10\.1109/lra\.', 'IEEE Robotics and Automation Letters'), (r'^10\.1109/access', 'IEEE Access'), (r'^10\.1109/icassp', 'ICASSP'),
    (r'^10\.1109/icra', 'ICRA'), (r'^10\.1109/tpami', 'IEEE TPAMI'), (r'^10\.1109/tsp\.', 'IEEE Trans. Signal Process.'),
    (r'^10\.1109/tip\.', 'IEEE Trans. Image Process.'), (r'^10\.1109/tgrs', 'IEEE Trans. Geosci. Remote Sens.'),
    (r'^10\.21437/interspeech', 'Interspeech'), (r'^10\.4204/eptcs', 'EPTCS'), (r'^10\.46298/lmcs', 'Logical Methods in Computer Science'),
    (r'^10\.3842/sigma', 'SIGMA'), (r'^10\.2140/agt', 'Algebr. Geom. Topol.'), (r'^10\.3390/universe', 'Universe'), (r'^10\.3390/e\d', 'Entropy'),
    (r'^10\.3390/sym', 'Symmetry'), (r'^10\.1088/1361-6382/', 'Class. Quantum Grav.'), (r'^10\.1088/1748-0221/', 'JINST'),
    (r'^10\.1088/1751-8121/', 'J. Phys. A'), (r'^10\.1088/1367-2630/', 'New J. Phys.'), (r'^10\.1088/1361-648x/', 'J. Phys.: Condens. Matter'),
    (r'^10\.1088/1742-5468/', 'J. Stat. Mech.'), (r'^10\.1088/1674-1137/', 'Chin. Phys. C'), (r'^10\.1088/1674-4527/', 'Res. Astron. Astrophys.'),
    (r'^10\.1209/0295-5075/', 'EPL'), (r'^10\.7566/jpsj', 'J. Phys. Soc. Jpn.'), (r'^10\.1017/jfm', 'J. Fluid Mech.'),
    (r'^10\.1371/journal\.pone', 'PLOS ONE'),
    (r'^10\.1098/rspa', 'Proc. R. Soc. A'), (r'^10\.4230/lipics', 'LIPIcs'), (r'^10\.3389/fphy', 'Front. Phys.'))]
DOI_PHYS_REV = re.compile(r'^10\.1103/physrev(lett|research|applied|materials|fluids|accelbeams|[a-ex])\b')


def journal_from_doi(doi):
    doi = (doi or '').lower()
    match = DOI_PHYS_REV.match(doi)
    if match:
        return 'Phys. Rev. ' + PHYS_REV_PARTS.get(match.group(1), match.group(1).upper())
    for pattern, name in DOI_JOURNALS:
        if pattern.match(doi):
            return name
    return ''


ENDS_BADLY = re.compile(r'\b(of|the|on|in|and|for|to|at|proceedings|proc|conference|conf|journal|vol|accepted|submitted|published)\.?$', re.I)


def journal_of(jref):
    """The journal or conference of a journal reference, short and normalized ('Phys. Rev. D', 'MNRAS', 'NeurIPS'),
    or '' when no name can be read."""
    text = ' '.join(jref.split())
    if not text:
        return ''
    match = PHYS_REV.match(text)
    if match:
        part = (match.group(1) or '').strip()
        return ('Phys. Rev. ' + PHYS_REV_PARTS.get(part.lower(), part.upper())).strip()
    for pattern, name in CONFERENCES:
        if pattern.search(text):
            return name
    # the name runs to the first volume number, year or parenthesis
    name = re.split(r'\s*(?:\(|\d|,|;|:(?=\s*\d)|\bvol\b|\bvolume\b)', text, maxsplit=1, flags=re.I)[0].strip(' -').rstrip(' ')
    if '. ' not in name:  # keep the final dot of an abbreviated name ('Pacific J. Math.'), not of 'MNRAS.'
        name = name.rstrip(' .')
    key = re.sub(r'[^a-z]', '', name.lower())
    if key in JOURNAL_KEYS:
        return JOURNAL_KEYS[key]
    if key.startswith('the') and key[3:] in JOURNAL_KEYS:
        return JOURNAL_KEYS[key[3:]]
    if len(key) < 2 or ENDS_BADLY.search(name) or name.lower().startswith(('accepted', 'submitted', 'to appear', 'in press', 'published')):
        return ''
    # 'Astrophys.J.' -> 'Astrophys. J.'
    return re.sub(r'\.(?=[A-Za-z])', '. ', name)[:80]


# the arXiv comments of a paper accepted somewhere ('Accepted at NeurIPS 2023', 'To appear in ...')
ACCEPTED = re.compile(r'\b(accepted|to appear|camera[- ]ready|published (?:in|at|by|as)|presented at|appears? in|appeared in)\b'
                      r'|^\W*(?:in )?proceedings of\b', re.I)
NOT_ACCEPTED = re.compile(r'\b(not (?:yet )?(?:been )?(?:accepted|published)|under review|submitted to|in submission)\b', re.I)


def accepted(comments):
    return bool(comments) and bool(ACCEPTED.search(comments)) and not NOT_ACCEPTED.search(comments)


def venue_of(record):
    """The journal or conference of a paper: journal reference, then DOI, then a conference named in accepted comments."""
    name = journal_of(record['jref']) or journal_from_doi(record['doi'])
    if name or not accepted(record.get('comments')):
        return name
    for pattern, conference in CONFERENCES:
        if pattern.search(record['comments']):
            return conference
    return ''


def _published(papers):
    """(papers, with a journal reference or DOI, with one or accepted per the comments)."""
    formal = sum(bool(p['jref'] or p['doi']) for p in papers)
    return len(papers), formal, sum(bool(p['jref'] or p['doi'] or accepted(p.get('comments'))) for p in papers)


def journal_data(records, years):
    by_year = cohorts(records, set(years))
    data = {'years': years, 'fields': [], 'all': {}, 'comments_known': all('comments' in p for p in by_year[years[0]][:1000])}
    for year in years:
        data['all'][year] = _published(by_year[year])
    newest_full = years[-1]
    cohort = years[0]
    for code, _en, _fr in FIELDS:
        shares = {}
        for year in years:
            papers = [p for p in by_year[year] if field_of(p['cats']) == code]
            shares[year] = _published(papers)
        if shares[cohort][0] < 300:
            continue
        journals = Counter(venue_of(p) for p in by_year[cohort] if field_of(p['cats']) == code)
        journals.pop('', None)
        data['fields'].append({'code': code, 'shares': shares, 'journals': [(name, c) for name, c in journals.most_common(5) if c >= 5]})
    data['newest'] = newest_full
    return data


def journals_page(data, lang, made):
    years = data['years']
    cohort, newest = years[0], years[-1]
    n, published, _with_comments = data['all'][cohort]
    share = 100 * published / n
    fields = sorted(data['fields'], key=lambda f: -f['shares'][cohort][1] / f['shares'][cohort][0])
    hi, lo = fields[0], fields[-1]

    def s(f, year, with_comments=False):
        total, done, done_or_accepted = f['shares'][year]
        return 100 * (done_or_accepted if with_comments else done) / total if total else 0
    commented = data.get('comments_known')
    takeaways = [
        t(lang, '**%s** of the arXiv papers submitted in %d now show a journal reference or a DOI on arXiv (%s of %s).',
          '**%s** des articles arXiv soumis en %d affichent aujourd’hui une référence de revue ou un DOI sur arXiv (%s sur %s).')
        % (pct(share, lang), cohort, num(published, lang), num(n, lang)),
        t(lang, 'Highest: **%s**, %s.', 'Le plus haut : **%s**, %s.') % (_field(hi['code'], lang), pct(s(hi, cohort), lang)),
        t(lang, 'Lowest: **%s**, %s.', 'Le plus bas : **%s**, %s.') % (_field(lo['code'], lang), pct(s(lo, cohort), lang)),
        t(lang, 'For papers submitted in %d, the share is still %s: publication takes time.', 'Pour les articles soumis en %d, la part n’est encore que de %s : la publication prend du temps.')
        % (newest, pct(100 * data['all'][newest][1] / max(data['all'][newest][0], 1), lang)),
    ]
    cs = next((f for f in fields if f['code'] == 'cs'), None)
    if commented and cs:
        takeaways.append(t(lang, 'Counting the papers whose arXiv comments say they were accepted (mostly at conferences), computer science goes from %s to **%s**.',
                           'En comptant les articles dont les commentaires arXiv indiquent une acceptation (surtout en conférence), l’informatique passe de %s à **%s**.')
                         % (pct(s(cs, cohort), lang), pct(s(cs, cohort, True), lang)))
    body = ['<div class="takeaways" markdown="1">\n\n%s\n\n%s\n\n</div>' % (t(lang, 'Key numbers', 'Chiffres clés'), '\n'.join('- ' + x for x in takeaways))]
    body.append('<p class="note">%s</p>' % t(lang, 'Counted by SummarizePaper from the metadata of every arXiv paper submitted from %d to %d (arXiv OAI-PMH, harvested on %s). A journal reference is added by the authors, or by arXiv from some publishers, so these shares are a lower bound.',
                                              'Calculé par SummarizePaper à partir des métadonnées de chaque article arXiv soumis de %d à %d (OAI-PMH d’arXiv, récupérées le %s). La référence de revue est ajoutée par les auteurs, ou par arXiv pour certains éditeurs : ces parts sont donc un minimum.')
                % (cohort, newest, day(made, lang)))
    body.append(t(lang, '## What share of arXiv preprints are published in journals?', '## Quelle part des preprints arXiv est publiée en revue ?'))
    body.append(t(lang, 'At least %s of the papers submitted to arXiv in %d point to a published version: their arXiv record carries a journal reference or a DOI. The share depends on how long papers have had to be published:',
                  'Au moins %s des articles soumis à arXiv en %d renvoient à une version publiée : leur notice arXiv porte une référence de revue ou un DOI. La part dépend du temps que les articles ont eu pour être publiés :')
                % (pct(share, lang), cohort))
    rows = [(year, num(data['all'][year][0], lang), num(data['all'][year][1], lang), pct(100 * data['all'][year][1] / max(data['all'][year][0], 1), lang)) for year in years]
    body.append(table(t(lang, ['Submitted in', 'Papers', 'With a journal reference or DOI', 'Share'],
                        ['Soumis en', 'Articles', 'Avec référence de revue ou DOI', 'Part']), rows))
    body.append(t(lang, '## Which fields publish their preprints in journals most?', '## Quels domaines publient le plus leurs preprints en revue ?'))
    body.append(bar_chart(t(lang, 'Share of %d arXiv papers with a journal reference or DOI, by field', 'Part des articles arXiv de %d avec référence de revue ou DOI, par domaine') % cohort,
                          t(lang, 'Primary category of each paper. A lower bound: many published papers have neither. Source: arXiv OAI-PMH metadata.',
                            'Catégorie principale de chaque article. Un minimum : beaucoup d’articles publiés n’ont ni l’un ni l’autre. Source : métadonnées OAI-PMH d’arXiv.'),
                          [(f['code'] if f['code'] not in ('nucl', 'physics') else {'nucl': 'nucl', 'physics': 'physics.*'}[f['code']], s(f, cohort), pct(s(f, cohort), lang),
                            '%s: %s' % (t(lang, *FIELD_NAMES[f['code']]), pct(s(f, cohort), lang))) for f in fields],
                          t(lang, 'Bar chart of the share of arXiv papers submitted in %d that show a journal reference or DOI, by field, from %s in %s to %s in %s.',
                            'Graphique en barres de la part des articles arXiv soumis en %d qui affichent une référence de revue ou un DOI, par domaine, de %s en %s à %s en %s.')
                          % (cohort, pct(s(hi, cohort), lang), t(lang, *FIELD_NAMES[hi['code']]), pct(s(lo, cohort), lang), t(lang, *FIELD_NAMES[lo['code']])), top=100))
    rows = [(_field(f['code'], lang),) + tuple(pct(s(f, y), lang) for y in years) for f in fields]
    body.append(table([t(lang, 'Field', 'Domaine')] + [t(lang, 'Submitted %d', 'Soumis en %d') % y for y in years], rows))
    body.append(t(lang, 'Physics and astronomy papers are usually submitted to arXiv when they go to a journal, and authors update the record once published. In computer science most papers go to conferences, whose proceedings are often not entered as a journal reference, so the low share there does not mean the papers stay unpublished.',
                  'En physique et en astronomie, les articles sont en général déposés sur arXiv au moment de la soumission à une revue, et les auteurs complètent la notice une fois l’article publié. En informatique, la plupart des articles vont en conférence, dont les actes sont rarement indiqués comme référence de revue : la part faible n’y signifie pas que les articles restent inédits.'))
    if commented:
        body.append(t(lang, '## What about conference papers?', '## Et les articles de conférence ?'))
        body.append(t(lang, 'Many authors write in the arXiv comments where a paper was accepted ("Accepted at NeurIPS 2022", "To appear in ..."), without adding a journal reference. Counting those papers too, for papers submitted in %d:',
                      'Beaucoup d’auteurs indiquent dans les commentaires arXiv où l’article a été accepté (« Accepted at NeurIPS 2022 », « To appear in ... »), sans ajouter de référence de revue. En comptant aussi ces articles, pour ceux soumis en %d :') % cohort)
        rows = [(_field(f['code'], lang), pct(s(f, cohort), lang), pct(s(f, cohort, True), lang))
                for f in sorted(fields, key=lambda f: -(s(f, cohort, True) - s(f, cohort)))]
        body.append(table(t(lang, ['Field', 'Journal reference or DOI', 'Including accepted per the comments'],
                            ['Domaine', 'Référence de revue ou DOI', 'Y compris acceptés selon les commentaires']), rows))
        body.append(t(lang, 'The comments are free text written by the authors; "accepted" is read from words such as accepted, to appear or camera-ready, and comments that say "submitted to" or "under review" are not counted.',
                      'Les commentaires sont un texte libre écrit par les auteurs ; l’acceptation est lue dans des mots comme accepted, to appear ou camera-ready, et les commentaires qui disent « submitted to » ou « under review » ne sont pas comptés.'))
    body.append(t(lang, '## Where are arXiv preprints published?', '## Où les preprints arXiv sont-ils publiés ?'))
    body.append((t(lang, 'The most frequent journals and conferences of papers submitted in %d, read from their journal reference, DOI or arXiv comments, by field:',
                   'Les revues et conférences les plus fréquentes des articles soumis en %d, lues dans leur référence de revue, leur DOI ou leurs commentaires arXiv, par domaine :') if commented else
                 t(lang, 'The most frequent journals of papers submitted in %d, read from their journal reference or DOI, by field:',
                   'Les revues les plus fréquentes des articles soumis en %d, lues dans leur référence de revue ou leur DOI, par domaine :')) % cohort)
    rows = []
    for f in sorted(fields, key=lambda f: -f['shares'][cohort][0]):
        if f['journals']:
            rows.append((_field(f['code'], lang), ', '.join('%s (%s)' % (name, num(c, lang)) for name, c in f['journals'])))
    body.append(table([t(lang, 'Field', 'Domaine'), t(lang, 'Most frequent journals and conferences (papers)', 'Revues et conférences les plus fréquentes (articles)') if commented
                       else t(lang, 'Most frequent journals (papers)', 'Revues les plus fréquentes (articles)')], rows))
    body.append(t(lang, '## How to know if a preprint was published', '## Comment savoir si un preprint a été publié'))
    body.append(t(lang, '''- On arXiv, look for the "Journal reference" and "DOI" lines under the abstract; they are often missing even when the paper is published.
- On SummarizePaper, a paper page shows its journal version when one is found, and Premium readers can be [alerted when a followed preprint is published](%s).
- For a whole reading list, Premium watches a [collection](%s) and emails you when one of its papers is published, retracted or corrected.''',
                  '''- Sur arXiv, cherchez les lignes « Journal reference » et « DOI » sous le résumé ; elles manquent souvent même quand l’article est publié.
- Sur SummarizePaper, la page d’un article indique sa version publiée quand elle est trouvée, et les lecteurs Premium peuvent être [alertés quand un preprint suivi est publié](%s).
- Pour toute une liste de lecture, Premium surveille une [collection](%s) et vous écrit quand l’un de ses articles est publié, rétracté ou corrigé.''')
                % (link(lang, 'guides/track-research-paper-citations/'), link(lang, 'guides/organize-research-paper-collections/')))
    body.append(related(lang, JOURNALS_SLUG))
    body.append(t(lang, '## Methodology and limits', '## Méthode et limites'))
    body.append(t(lang, '''- **Papers**: every arXiv paper whose identifier starts with the year (2201–2212 for %d, and so on), from the arXiv metadata of arXiv's OAI-PMH interface, harvested on %s.
- **Published**: the arXiv record has a journal reference or a DOI. Both are entered by the authors or added by arXiv for some publishers; many published papers have neither, so every share here is a lower bound, lowest in fields that publish in conference proceedings.
- **Field**: the archive of the primary category; physics.* is "physics (other)", nucl-ex and nucl-th are nuclear physics. Fields with fewer than 300 papers in %d are left out.
- **Journals**: names read from the journal reference, normalized (abbreviations and spacing), or else from the DOI prefix of the main publishers%s; a paper whose journal cannot be recognized is not counted in that table.''',
                  '''- **Articles** : chaque article arXiv dont l’identifiant commence par l’année (2201 à 2212 pour %d, etc.), d’après les métadonnées de l’interface OAI-PMH d’arXiv, récupérées le %s.
- **Publié** : la notice arXiv porte une référence de revue ou un DOI. Les deux sont saisis par les auteurs ou ajoutés par arXiv pour certains éditeurs ; beaucoup d’articles publiés n’ont ni l’un ni l’autre : chaque part est donc un minimum, plus encore dans les domaines qui publient dans des actes de conférence.
- **Domaine** : l’archive de la catégorie principale ; physics.* forme « physique (autres) », nucl-ex et nucl-th la physique nucléaire. Les domaines de moins de 300 articles en %d sont exclus.
- **Revues** : noms lus dans la référence de revue, normalisés (abréviations et espaces), ou à défaut dans le préfixe DOI des principaux éditeurs%s ; un article dont la revue n’est pas reconnue n’est pas compté dans ce tableau.''')
                % (cohort, day(made, lang), cohort, t(lang, ', or else from a conference named in comments that say the paper was accepted', ', ou à défaut dans une conférence citée par des commentaires qui disent l’article accepté') if commented else ''))
    body.append(cta(lang))
    faq = [
        {'q': t(lang, 'What percentage of arXiv papers get published?', 'Quel pourcentage des articles arXiv est publié ?'),
         'a': t(lang, 'At least %s of the papers submitted to arXiv in %d carry a journal reference or a DOI on arXiv. The true share is higher, since authors do not always add them.',
                'Au moins %s des articles soumis à arXiv en %d portent une référence de revue ou un DOI sur arXiv. La part réelle est plus élevée, car les auteurs ne les ajoutent pas toujours.')
         % (pct(share, lang), cohort)},
        {'q': t(lang, 'Which fields publish most of their arXiv preprints?', 'Quels domaines publient le plus leurs preprints arXiv ?'),
         'a': t(lang, '%s, with %s of %d papers showing a journal version; %s has the lowest share (%s). A low share also reflects authors who never add the reference, and conference papers.',
                '%s, avec %s des articles de %d affichant une version publiée ; %s a la part la plus faible (%s). Une part faible reflète aussi les auteurs qui n’ajoutent jamais la référence, et les articles de conférence.')
         % (_field(hi['code'], lang), pct(s(hi, cohort), lang), cohort, _field(lo['code'], lang), pct(s(lo, cohort), lang))},
        {'q': t(lang, 'Are arXiv papers peer reviewed?', 'Les articles arXiv sont-ils évalués par les pairs ?'),
         'a': t(lang, 'No: arXiv only moderates submissions. Peer review happens at the journal or conference; a journal reference on arXiv shows that a version was published there.',
                'Non : arXiv se contente de modérer les soumissions. L’évaluation par les pairs a lieu dans la revue ou la conférence ; une référence de revue sur arXiv montre qu’une version y a été publiée.')},
    ]
    if commented and cs:
        faq.insert(2, {'q': t(lang, 'How many computer science preprints on arXiv are published?', 'Combien de preprints d’informatique sur arXiv sont publiés ?'),
                       'a': t(lang, 'Of the computer science papers submitted to arXiv in %d, %s show a journal reference or DOI, but %s once papers whose arXiv comments say they were accepted, mostly at conferences such as NeurIPS, CVPR or AAAI, are counted too.',
                              'Des articles d’informatique soumis à arXiv en %d, %s affichent une référence de revue ou un DOI, mais %s si l’on compte aussi ceux dont les commentaires arXiv indiquent une acceptation, surtout en conférence (NeurIPS, CVPR, AAAI...).')
                       % (cohort, pct(s(cs, cohort), lang), pct(s(cs, cohort, True), lang))})
    meta = {'title': t(lang, 'How many arXiv preprints are published in journals? Data by field (%d–%d)', 'Combien de preprints arXiv sont publiés en revue ? Les données par domaine (%d–%d)') % (cohort, newest),
            'description': t(lang, 'Share of arXiv papers with a journal reference or DOI by field and year, and the journals they appear in, from every paper submitted since %d.',
                             'Part des articles arXiv avec référence de revue ou DOI par domaine et par année, et revues où ils paraissent, sur tous les articles depuis %d.') % cohort,
            'date': made, 'updated': made, 'faq': faq,
            'dataset': {'name': t(lang, 'Journal references of arXiv papers by field, %d–%d', 'Références de revue des articles arXiv par domaine, %d–%d') % (cohort, newest),
                        'description': t(lang, 'Whether each arXiv paper submitted from %d to %d carries a journal reference or DOI, by primary category.',
                                         'Présence d’une référence de revue ou d’un DOI pour chaque article arXiv soumis de %d à %d, par catégorie principale.') % (cohort, newest),
                        'temporalCoverage': '%d-01-01/%d-12-31' % (cohort, newest)},
            'sources': SOURCES}
    return meta, '\n\n'.join(body)
