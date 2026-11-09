"""'What did I miss?': papers close to a draft that the draft does not cite (Premium).

1. The draft's reference list is read: from a .bib file when the researcher gives one, else from the PDF or the
   pasted text, where the model copies each reference's first author, year, title and locator. Many styles (APS,
   AAS, MNRAS...) print no title, so a cited paper is recognized by its arXiv id, its DOI or its title, and, when
   the reference has no title, by its first author and year: those are set apart as probably cited, never
   proposed. The references without an identifier are resolved (Crossref, NASA ADS, OpenAlex, see bibliography.py)
   within a time limit, to find them in the citation graph.
2. The model tells the draft as an abstract would, with its question, method, data and results apart (as
   scoops.py does for a project), and writes search queries.
3. Candidates come from three places: arXiv and OpenAlex answer the queries; OpenAlex gives the works that cite
   several of the draft's references (bibliographic coupling, which finds what is written with other words);
   and the works the closest candidates cite most often (founding papers, see literature.snowball).
   When OpenAlex's daily budget is used up, the result says so.
4. Retracted papers, the draft itself and the papers it cites are left out; the others are ranked by their
   closest facet, raised a little when they share references with the draft, are founding papers or are well
   cited for their age.
5. The model reads the closest with the beginning of the draft: how important citing each one is, the kind of
   relation, how to position the draft, and where to cite it (a sentence of the draft). The sentence of the draft
   and the words of the abstract it quotes are kept only when they are found word for word.

The researcher marks each paper (already cited, not relevant, added), exports them, sends them to Zotero or to
a collection, follows the draft with Scoop watch, and, if they kept the draft's text (encrypted, in a second row
of kind 'missed_draft'), searches again later: the papers the previous search had not found are marked new.
"""
import contextvars
import datetime
import json
import logging
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from io import BytesIO, StringIO
from urllib.parse import urlencode

import numpy as np
import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.mail import EmailMultiAlternatives
from django.db import connections
from django.core.cache import cache
from django.http import Http404
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import (access, assistant, bibliography, digest, evidence, exports, generated, library, literature, mailcheck,
               neighbours, result_share, similar, sources)
from .models import Collection, Generated

logger = logging.getLogger(__name__)

RESULT_VERSION = 2
MAX_UPLOAD = 20 * 1024 * 1024
MAX_BIB_UPLOAD = 5 * 1024 * 1024
MAX_TEXT_CHARS = 300_000
KEPT_CHARS = 150_000  # the text kept to search again
HEAD_CHARS = 12_000  # read to describe the draft
EXCERPT_CHARS = 14_000  # read with the candidates: where to cite them
REFS_CHARS = 40_000
REFS_CHUNK = 7_000  # read apart: the model stops copying a long list near 100 references
QUERIES = 8
QUERY_CHARS = 120
ARXIV_PER_QUERY = 25
OPENALEX_PER_QUERY = 25
RESOLVED = 80  # references without an identifier looked up, for the citation graph
RESOLVE_SECONDS = 60
REFERENCE_WORKS = 200
CITES_MAX = 50  # references whose citing works are asked for in one OpenAlex filter
COUPLED_PER_CALL = 50
RECENT_YEARS = 3
JUDGED = 30
JUDGE_BATCH = 15
JUDGE_ROUNDS = 2  # the model sometimes answers for a few papers only: the others are asked again
THREADS = 4  # model calls made at once
COUPLING_BONUS = 0.02  # each shared reference, up to 4
COCITED_BONUS = 0.04  # a work the closest candidates cite often
STANDING_WEIGHT = 0.05
SAME_TITLE = 0.9
PROBABLE_SHOWN = 30
PER_DAY = 10
QUOTA_KEY = 'missed-quota:%s:%s'
DRAFT_KIND = 'missed_draft'
IMPORTANCES = ('high', 'medium', 'low', 'unjudged')
KINDS = ('same_question', 'competing_method', 'prior_result', 'method', 'data', 'review', 'context')
MARKS = ('cited', 'irrelevant', 'added')
EXPORTS = ('bib', 'ris')
REF_HEADING_RE = re.compile(
    r'^[ \t]*(?:[\dIVX]+\.?[ \t]*)?(?:references(?: and notes)?|bibliography|literature cited|works cited|'
    r'r[ée]f[ée]rences(?: bibliographiques)?|bibliographie|literatur(?:verzeichnis)?|referencias|bibliograf[ií]a)'
    r'[ \t]*:?[ \t]*$', re.I | re.M)
STEPS = [
    ('read', _('Reading your draft and its references')),
    ('search', _('Searching arXiv, the journals, and the papers that cite your references')),
    ('rank', _('Removing the papers you already cite and ranking the others')),
    ('review', _('An AI reads the closest papers and says where to cite them')),
]


def pdf_text(data):
    """All the text of a PDF, references included (they tell us what the draft cites)."""
    from pdfminer.converter import TextConverter
    from pdfminer.layout import LAParams
    from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager
    from pdfminer.pdfpage import PDFPage
    manager = PDFResourceManager(caching=False)
    out = StringIO()
    converter = TextConverter(manager, out, laparams=LAParams())
    interpreter = PDFPageInterpreter(manager, converter)
    for page in PDFPage.get_pages(BytesIO(data)):
        interpreter.process_page(page)
    converter.close()
    return out.getvalue().replace('\x00', '')


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def family(name):
    """'Ann-Marie de la Cruz' -> 'cruz', 'Lee, A.' -> 'lee': the last word of a family name, without accents."""
    name = bibliography.normalized(name.split(',')[0] if ',' in name else name)
    words = [w for w in name.split() if len(w) > 1 or not name.count(' ')]
    return words[-1] if words else ''


def year_of(value):
    found = re.search(r'(1[89]|20)\d\d', str(value or ''))
    return int(found.group(0)) if found else None


# --- the draft's references --------------------------------------------------

def split_references(text):
    """(body, reference list): the list starts at the first 'References' heading past the first third."""
    for match in REF_HEADING_RE.finditer(text):
        if match.start() > len(text) * 0.3:
            return text[:match.start()], text[match.end():]
    return text, ''


def at_once(calls):
    """The results of these functions, called in THREADS threads: the model answers them in parallel. Each counts
    its AI usage as the caller's task and closes its own database connection."""
    def run(context, call):
        try:
            return context.run(call)
        finally:
            connections.close_all()
    contexts = [contextvars.copy_context() for _ in calls]
    with ThreadPoolExecutor(max_workers=THREADS) as pool:
        return list(pool.map(run, contexts, calls))


def chunks(listing):
    """The reference list in pieces of about REFS_CHUNK characters, cut between lines."""
    pieces, piece = [], ''
    for line in listing[:REFS_CHARS].splitlines(keepends=True):
        if len(piece) + len(line) > REFS_CHUNK and piece:
            pieces.append(piece)
            piece = ''
        piece += line
    return pieces + ([piece] if piece.strip() else [])


def read_references(listing):
    """The references of a reference list, as the model copies them: [{'author', 'year', 'title', 'journal',
    'volume', 'page', 'doi', 'arxiv'}]."""
    if len(listing.strip()) < 50:
        return []
    return [ref for refs in at_once([lambda piece=piece: read_piece(piece) for piece in chunks(listing)]) for ref in refs]


def read_piece(piece):
    """The references of one piece of the list (see read_references)."""
    answer = generated.chat_json(
        'Above is the reference list of a research draft, or a part of it. Answer with a JSON object {"references": [["family name '
        'of the first author", "year", "title, or empty when the reference does not give it", "journal", "volume", '
        '"first page or article number", "DOI", "arXiv id"]]}: one array per reference, in order. Copy the values '
        'as they are written; leave empty what the reference does not give. A reference cut at the start or the end '
        'of the list is copied as far as it goes.',
        paid=False, max_tokens=8000, system='<references>\n%s\n</references>' % piece)
    refs = []
    for row in answer.get('references', []):
        if not isinstance(row, list) or not row:
            continue
        row = [str(v or '').strip() for v in row] + [''] * 8
        refs.append({'author': family(row[0]), 'year': year_of(row[1]), 'title': row[2], 'journal': row[3],
                     'volume': row[4], 'page': row[5], 'doi': sources.clean_doi(row[6]) or '',
                     'arxiv': re.sub(r'v\d+$', '', row[7].lower().replace('arxiv:', '').strip())})
    return refs


def bib_references(bib):
    """The references of a .bib file, as read_references gives them."""
    from . import refcheck
    refs = []
    for entry in refcheck.parse_bibtex(bib):
        f = entry['fields']
        arxiv, _version, doi = refcheck.identify(entry)
        authors = bibliography.families(f.get('author', ''))
        refs.append({'author': authors[0] if authors else '', 'year': year_of(f.get('year')),
                     'title': refcheck.plain(f.get('title', '')), 'journal': refcheck.plain(f.get('journal', '')),
                     'volume': f.get('volume', ''), 'page': bibliography.first(bibliography.pages(f.get('pages', ''))),
                     'doi': doi, 'arxiv': arxiv})
    return refs


def resolve(refs, limited, step=lambda: None):
    """Give the references without an identifier the DOI Crossref, NASA ADS or OpenAlex finds for them, RESOLVED
    at most and within RESOLVE_SECONDS: a match must agree on authors and title or locator (see bibliography.py)."""
    def openalex_allowed():
        allowed = sources.tool_search_allowed()
        if not allowed:
            limited.add('budget')
        return allowed
    resolver = bibliography.Resolver(openalex_allowed)
    resolver.deadline = time.monotonic() + RESOLVE_SECONDS
    todo = [r for r in refs if not r['doi'] and not r['arxiv'] and r['author'] and (r['title'] or r['volume'])][:RESOLVED]
    for ref in todo:
        if time.monotonic() > resolver.deadline:
            break
        entry = {'fields': {k: v for k, v in (('author', ref['author']), ('title', ref['title']),
                                              ('year', str(ref['year'] or '')), ('journal', ref['journal']),
                                              ('volume', ref['volume']), ('pages', ref['page'])) if v}}
        record = resolver.resolve(entry)
        if record and record.get('doi'):
            ref['doi'] = record['doi'].lower()
        step()


def identifiers(text, refs):
    """(arXiv ids without version, DOIs) the draft cites: written anywhere in it, or found for its references."""
    ids = {re.sub(r'v\d+$', '', m.replace('--', '/')).lower() for m in digest.ARXIV_ID_RE.findall(text)}
    dois = {m.group(1).rstrip('.,;)]').lower() for m in sources.DOI_RE.finditer(text)}
    for ref in refs:
        if ref['arxiv']:
            ids.add(ref['arxiv'])
        match = sources.ARXIV_DOI_RE.match(ref['doi'])
        if match:
            ids.add(match.group(1).lower())
        elif ref['doi']:
            dois.add(ref['doi'])
    return ids, dois


def citation_status(paper, cited_ids, cited_dois, cited_works, refs, draft_words):
    """('cited' | 'probable' | '', the reference it matches): cited when an identifier or the title is in the
    draft; probable when a reference without a title has its first author and year (and no other identifier)."""
    arxiv = re.sub(r'v\d+$', '', paper['key'].replace('--', '/')).lower() if not sources.is_external(paper['key']) else ''
    arxiv_doi = sources.ARXIV_DOI_RE.match(paper['doi'] or '')
    if arxiv and arxiv in cited_ids or arxiv_doi and arxiv_doi.group(1).lower() in cited_ids:
        return 'cited', None
    if paper['doi'] and paper['doi'].lower() in cited_dois or paper['openalex'] and paper['openalex'] in cited_works:
        return 'cited', None
    title = normalized(paper['title'])
    if len(title) > 20 and title in draft_words:
        return 'cited', None
    first = family(paper['authors'][0]) if paper['authors'] else ''
    doi = (paper['doi'] or '').lower()
    for ref in refs:
        if ref['arxiv'] and arxiv and ref['arxiv'] != arxiv or ref['doi'] and doi and ref['doi'] != doi:
            continue  # both are identified, and they differ: another paper of the same author and year
        if ref['title'] and len(ref['title']) > 15:
            if SequenceMatcher(None, normalized(ref['title']), title).ratio() >= SAME_TITLE:
                return 'cited', ref
        elif first and ref['author'] == first and ref['year'] and paper['year'] and abs(ref['year'] - paper['year']) <= 1:
            return 'probable', ref
    return '', None


# --- searches ----------------------------------------------------------------

def describe(head, lang):
    """The draft told as a paper's abstract, its facets and its queries, in English (papers are compared through
    their abstracts, see scoops.describe)."""
    answer = generated.chat_json(
        'Above is the beginning of a research draft. Answer in English with a JSON object {"title": "its title, or a '
        'short description", "title_local": "the same title in %s", "abstract": "the abstract of the draft, 100 to 200 '
        'words, no first person", "question": "the question it addresses, 1 or 2 sentences", "method": "its method, 1 '
        'or 2 sentences", "data": "the data, sample, system or object it studies, 1 or 2 sentences, empty if none", '
        '"results": "its results, 1 or 2 sentences", "core": "the 3 to 5 words that name its topic, as paper titles '
        'would", "queries": ["%d literature search queries of 2 to 5 keywords each, together covering its question, '
        'its method, its data and its closest problems"]}.' % (generated.language_name(lang), QUERIES),
        paid=False, max_tokens=1500, system='<draft>\n%s\n</draft>' % head)
    facets = {k: ' '.join(str(answer.get(k) or '').split()) for k in
              ('title', 'title_local', 'abstract', 'question', 'method', 'data', 'results', 'core')}
    facets['queries'] = list(dict.fromkeys(' '.join(str(q).split())[:QUERY_CHARS] for q in answer.get('queries', [])
                                           if str(q).strip()))[:QUERIES]
    if not facets['queries']:
        facets['queries'] = [facets['core'] or facets['title'] or head[:QUERY_CHARS]]
    facets['core'] = facets['core'] or facets['queries'][0]
    return facets


def search(pool, queries, limited):
    """arXiv and OpenAlex (journals, preprints, conference papers) for each query."""
    for query in queries:
        terms = ' AND '.join('all:%s' % w for w in re.findall(r'[\w\-]+', query)[:6])
        if terms and 'arxiv' not in limited:
            try:
                for paper in digest.arxiv_get({'search_query': terms, 'max_results': ARXIV_PER_QUERY, 'sortBy': 'relevance'},
                                              timeout=literature.ARXIV_TIMEOUT, retry=False):
                    pool.add(literature.from_arxiv(paper), 'search')
            except digest.ArxivUnavailable:
                logger.warning('arXiv search failed for %s', query)
                limited.add('arxiv')
        for work in openalex({'search': query, 'filter': 'has_abstract:true', 'per_page': OPENALEX_PER_QUERY}, limited):
            pool.add(literature.from_openalex(work), 'search')


def openalex(params, limited):
    """OpenAlex works for these parameters, within the tools' daily budget; `limited` (a set) is told why not."""
    return literature.works(params, reasons(limited))


def reasons(limited):
    """A list literature's helpers append their reasons to (budget, openalex, s2...), which end in `limited`."""
    class Reasons(list):
        def append(self, reason):
            limited.add(reason)
    return Reasons()


def reference_works(cited_ids, cited_dois, limited):
    """The OpenAlex ids of the draft's references, from their DOIs (an arXiv preprint has one too)."""
    dois = [d for d in list(cited_dois) + ['10.48550/arxiv.%s' % a for a in cited_ids] if not re.search(r'[|,\s]', d)]
    found = set()
    dois = dois[:REFERENCE_WORKS]
    for start in range(0, len(dois), 50):
        batch = dois[start:start + 50]
        for work in openalex({'filter': 'doi:%s' % '|'.join(batch), 'per_page': len(batch)}, limited):
            found.add((work.get('id') or '').rsplit('/', 1)[-1])
    return {w for w in found if re.match(r'^W\d+$', w)}


def coupled(pool, works, core, limited):
    """The works that cite several of the draft's references: those on its topic, the most cited, and the recent
    ones. The references most cited by the candidates already found are asked first (OpenAlex takes CITES_MAX)."""
    if not works:
        return
    counts = {w: 0 for w in works}
    for paper in pool.papers:
        for ref in paper['references']:
            if ref in counts:
                counts[ref] += 1
    asked = '|'.join(sorted(works, key=lambda w: -counts[w])[:CITES_MAX])
    since = (timezone.now() - datetime.timedelta(days=365 * RECENT_YEARS)).date().isoformat()
    need = min(2, len(works))
    for params in ({'search': core, 'filter': 'cites:%s,has_abstract:true' % asked},
                   {'filter': 'cites:%s,has_abstract:true' % asked, 'sort': 'cited_by_count:desc'},
                   {'search': core, 'filter': 'cites:%s,has_abstract:true,from_publication_date:%s' % (asked, since)}):
        for work in openalex(dict(params, per_page=COUPLED_PER_CALL), limited):
            paper = literature.from_openalex(work)
            if paper and len(set(paper['references']) & works) >= need:
                pool.add(paper, 'coupled')


class Ranker:
    """Closeness to the draft: to its closest facet (a paper sharing only the method or the data is as close to
    that facet as one on the whole draft), and to the whole draft, which sets the level shown."""

    def __init__(self, facets):
        texts = ['%s\n\n%s' % (facets['title'], facets['abstract'])] + [
            facets[k] for k in ('question', 'method', 'data', 'results') if facets[k]]
        self.rows = similar.embed([t[:similar.MAX_INPUT_CHARS] for t in texts])

    def __call__(self, papers):
        new = [p for p in papers if 'similarity' not in p]
        for start in range(0, len(new), similar.BATCH):
            batch = new[start:start + similar.BATCH]
            vectors = similar.embed([('%s\n\n%s' % (p['title'], p['abstract']))[:similar.MAX_INPUT_CHARS] for p in batch])
            scores = vectors @ self.rows.T
            for paper, row in zip(batch, scores):
                paper['whole'], paper['similarity'] = float(row[0]), float(np.max(row))
        return sorted(papers, key=lambda p: -p['similarity'])


def score(paper, now_year):
    return (paper['similarity'] + COUPLING_BONUS * min(paper.get('shared', 0), 4)
            + COCITED_BONUS * (paper.get('cocited', 0) >= literature.SNOWBALL_MIN)
            + STANDING_WEIGHT * literature.standing(paper, now_year))


# --- the model reads the closest ---------------------------------------------

JUDGE = """The draft above is summarized here: {summary}

The papers below come close to the draft, and the draft does not cite them. For each paper, from its abstract:
- importance: "high" if a referee would expect the draft to cite it (same question, a competing answer, a result
  the draft builds on, extends or contradicts), "medium" if it is worth citing or reading, "low" if only related,
  "none" if it does not concern the draft.
- kind: one of same_question, competing_method, prior_result, method (a method the draft could use), data (a data
  set, sample or tool), review, context.
- relation: one or two sentences of {language}: how the paper relates to the draft.
- position: one sentence of {language}: what to cite it for, and what sets the draft apart from it.
- section: the heading of the draft's section where to cite it, as the draft writes it ("" if unsure).
- draft_quote: one sentence of the draft, copied word for word from the excerpt above, next to which it would be
  cited ("" if none fits).
- abstract_quote: 10 to 30 words of the paper's abstract, copied word for word, that show the relation.
Answer with a JSON object {{"papers": [{{"id": "p0", "importance": "...", "kind": "...", "relation": "...",
"position": "...", "section": "...", "draft_quote": "...", "abstract_quote": "..."}}]}}."""


def listing(papers):
    rows = []
    for i, p in enumerate(papers):
        facts = [str(p['year'] or ''), p['venue'] or '', 'preprint, not peer reviewed' if p['status'] == 'preprint' else '']
        if p.get('cited_by'):
            facts.append('cited %d times' % p['cited_by'])
        if p.get('shared'):
            facts.append("cites %d of the draft's references" % p['shared'])
        rows.append('<paper id="p%d">\nTitle: %s\nAbout it: %s\nAbstract: %s\n</paper>'
                    % (i, p['title'], ', '.join(f for f in facts if f), p['abstract'][:1500]))
    return '\n\n'.join(rows)


def judge(papers, summary, excerpt, lang):
    """{index: note} of the model's notes on these papers, each quote kept only if found word for word."""
    def ask(batch):
        try:
            return generated.chat_json(
                JUDGE.format(summary=summary, language=generated.language_name(lang)), paid=True, max_tokens=6000,
                system='<draft>\n%s\n</draft>\n\n%s' % (excerpt, listing([papers[i] for i in batch])))
        except (generated.GenerationError, requests.RequestException, ValueError):
            logger.warning('The notes of %d papers could not be written', len(batch))
            return {}

    notes = {}
    for _round in range(JUDGE_ROUNDS):
        todo = [i for i in range(len(papers)) if i not in notes]
        batches = [todo[start:start + JUDGE_BATCH] for start in range(0, len(todo), JUDGE_BATCH)]
        for batch, answer in zip(batches, at_once([lambda batch=batch: ask(batch) for batch in batches])):
            for note in answer.get('papers', []):
                if not isinstance(note, dict) or not re.fullmatch(r'p\d+', str(note.get('id', ''))):
                    continue
                n = int(note['id'][1:])
                if n < len(batch) and note.get('importance') in ('high', 'medium', 'low', 'none'):
                    notes[batch[n]] = note
    for i, note in notes.items():
        draft_quote = str(note.get('draft_quote') or '').strip()
        abstract_quote = str(note.get('abstract_quote') or '').strip()
        found = evidence.find(draft_quote, excerpt) if len(draft_quote) >= 15 else None
        note['draft_quote'] = unbroken(' '.join(found.group().split())) if found else ''
        found = evidence.find(abstract_quote, papers[i]['abstract']) if len(abstract_quote) >= 15 else None
        note['abstract_quote'] = ' '.join(found.group().split()) if found else ''
    return notes


def unbroken(text):
    """A sentence of a PDF without the hyphens that cut its words at the end of a line ('predom- inantly'); 'pre-
    and post-' keeps its own."""
    return re.sub(r'(?<=[a-z])- (?=[a-z])(?!(?:and|or|to|und|et|ou)\b)', '', text)


def short_authors(authors, n=5):
    return ', '.join(authors[:n]) + (' et al.' if len(authors) > n else '')


def find(text, lang, step=lambda name: None, bib='', previous=None):
    """The result shown on the page, from the draft's text (and its .bib); step(name) reports progress.
    previous: {'key', 'searched', 'seen', 'marks', 'facets'} of the search this one repeats, whose description and
    queries it runs again: what it finds that the previous one had not is then new in the literature, not in the
    wording of the queries."""
    text = text[:MAX_TEXT_CHARS]
    body, reference_list = split_references(text)
    limited = set()
    facets = (previous or {}).get('facets') or describe(body[:HEAD_CHARS], lang)
    refs = bib_references(bib) if bib.strip() else read_references(reference_list)
    step('read')
    resolve(refs, limited, lambda: step('read'))
    cited_ids, cited_dois = identifiers(text, refs)

    step('search')
    pool = literature.Pool()
    search(pool, facets['queries'], limited)
    works = reference_works(cited_ids, cited_dois, limited)
    coupled(pool, works, facets['core'], limited)

    step('rank')
    now_year = timezone.now().year
    rank = Ranker(facets)
    closest = rank(pool.readable())
    literature.snowball(pool, closest, reasons(limited))
    literature.recover(pool, reasons(limited))
    for paper in pool.papers:
        paper['shared'] = len(set(paper['references']) & works)

    draft_words = normalized(text)
    marks = (previous or {}).get('marks', {})
    fresh, probable, cited, dismissed = [], [], 0, 0
    titles = [normalized(t) for t in (facets['title'], facets['title_local']) if len(t) > 20]
    for paper in pool.papers:
        if paper['retracted'] or any(SequenceMatcher(None, normalized(paper['title']), t).ratio() >= SAME_TITLE for t in titles):
            continue  # the draft itself, once it is on arXiv
        status, ref = citation_status(paper, cited_ids, cited_dois, works, refs, draft_words)
        if status == 'cited' or marks.get(paper['key']) == 'cited':
            cited += 1
        elif status == 'probable':
            probable.append((paper, ref))
        elif marks.get(paper['key']) == 'irrelevant':
            dismissed += 1
        elif paper['abstract']:
            fresh.append(paper)
    fresh = rank(fresh)
    for paper in fresh:
        paper['score'] = score(paper, now_year)
    chosen = sorted(fresh, key=lambda p: -p['score'])[:JUDGED]

    step('review')
    summary = facets['abstract'] or facets['title']
    notes = judge(chosen, summary, body[:EXCERPT_CHARS], lang) if chosen else {}
    seen = (previous or {}).get('seen')
    items, unrelated = [], 0
    for i, paper in enumerate(chosen):
        note = notes.get(i, {})
        if note.get('importance') == 'none':
            unrelated += 1
            continue
        items.append({
            'key': paper['key'], 'title': paper['title'], 'authors': short_authors(paper['authors']),
            'all_authors': paper['authors'][:30], 'year': paper['year'] or '', 'venue': paper['venue'] or '',
            'status': paper['status'], 'doi': paper['doi'] or '', 'cited_by': paper['cited_by'],
            'abstract': paper['abstract'][:2500], 'similarity': round(paper['whole'], 3),
            'level': neighbours.level(paper['whole'])[0], 'importance': note.get('importance', 'unjudged'),
            'kind': note.get('kind') if note.get('kind') in KINDS else '',
            'relation': str(note.get('relation') or '').strip()[:600],
            'position': str(note.get('position') or '').strip()[:400],
            'section': str(note.get('section') or '').strip()[:120],
            'draft_quote': note.get('draft_quote', '')[:400], 'abstract_quote': note.get('abstract_quote', '')[:400],
            'shared': paper['shared'], 'founding': paper.get('cocited', 0) >= literature.SNOWBALL_MIN,
            'found_by': [f for f in paper['found_by'] if f in ('search', 'coupled', 'cited')],
            'mark': marks.get(paper['key'], '') if marks.get(paper['key']) == 'added' else '',
            'new': seen is not None and paper['key'] not in seen and literature.title_key(paper['title']) not in seen,
        })
    order = {k: i for i, k in enumerate(IMPORTANCES)}
    items.sort(key=lambda i: order[i['importance']])
    return {
        'version': RESULT_VERSION, 'title': facets['title_local'] or facets['title'], 'english': facets['title'],
        'abstract': facets['abstract'], 'core': facets['core'], 'queries': facets['queries'], 'items': items,
        'probable': [{'key': p['key'], 'title': p['title'], 'authors': short_authors(p['authors'], 3), 'year': p['year'] or '',
                      'ref': '%s %s' % (ref['author'].title(), ref['year'] or '')}
                     for p, ref in sorted(probable, key=lambda pr: -pr[0].get('similarity', 0))[:PROBABLE_SHOWN]],
        'searched': len(pool.papers), 'cited': cited, 'unrelated': unrelated, 'dismissed': dismissed,
        'refs': {'read': len(refs), 'identified': len(cited_ids) + len(cited_dois), 'in_graph': len(works),
                 'from_bib': bool(bib.strip())},
        'limited': sorted(limited), 'searched_on': timezone.now().date().isoformat(),
        'carried': {k: v for k, v in marks.items() if v in ('cited', 'irrelevant')},  # for the searches after this one
        'facets': facets,
        'previous': {'key': previous['key'], 'searched': previous['searched']} if previous else None,
    }


# --- pages -------------------------------------------------------------------

def write(data, text, lang, bib='', keep=False, previous=None, refcheck_key=''):
    def work(pk):
        generated.progress(pk, 'read')
        draft = pdf_text(data) if data is not None else text
        if len(draft.strip()) < 200:
            raise generated.GenerationError('no text found in the draft')
        result = find(draft, lang, lambda name: generated.progress(pk, name), bib=bib, previous=previous)
        if refcheck_key:
            result['refcheck_key'] = refcheck_key
        if keep:
            from . import scoops
            row = Generated.objects.get(pk=pk)
            Generated.objects.update_or_create(kind=DRAFT_KIND, key=row.key, lang=row.lang, defaults={
                'user': row.user, 'status': 'done',
                'content': json.dumps({'text': scoops.seal(draft[:KEPT_CHARS]), 'bib': scoops.seal(bib)})})
            if previous:  # the new search keeps the draft from now on
                Generated.objects.filter(kind=DRAFT_KIND, key=previous['key'], user=row.user).delete()
        Generated.objects.filter(pk=pk).update(status='done', content=json.dumps(result), updated=timezone.now())
        ready_email(pk, result)  # once the link works
        return result
    return work


def ready_email(pk, result):
    """Tell the researcher their search is ready: it takes minutes, and few wait on the page."""
    row = Generated.objects.select_related('user').filter(pk=pk).first()
    user = row.user if row is not None else None
    if user is None or not user.email or mailcheck.undeliverable_reason(user.email):
        return
    with translation.override(row.lang):
        context = {'user': user, 'result': result, 'high': sum(1 for i in result['items'] if i['importance'] == 'high'),
                   'link': '%s/%s/missed/%s/' % (digest.BASE_URL, row.lang, row.key)}
        subject = render_to_string('summarizer/missed_email_subject.txt', context).strip()
        text = render_to_string('summarizer/missed_email.txt', context)
    try:
        EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email]).send()
    except Exception:
        logger.exception('Could not send the search %s to user %s', row.key, user.pk)


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def _start(request, work):
    row = generated.start('missed', uuid.uuid4().hex, get_language(), work, user=request.user)
    return redirect('/%s/missed/%s/' % (get_language(), row.key))


def missed_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/missed.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'missed')
    error = None
    if request.method == 'POST' and paid:
        upload = request.FILES.get('draft')
        bib_upload = request.FILES.get('bib')
        text = request.POST.get('text', '').strip()
        if upload is not None and (upload.size > MAX_UPLOAD or not upload.name.lower().endswith('.pdf')):
            error = 'file'
        elif bib_upload is not None and (bib_upload.size > MAX_BIB_UPLOAD or not bib_upload.name.lower().endswith('.bib')):
            error = 'bib'
        elif upload is None and len(text) < 200:
            error = 'empty'
        elif not _spend(request.user):
            error = 'quota'
        else:
            data = upload.read() if upload is not None else None
            bib = bib_upload.read().decode('utf-8', errors='replace') if bib_upload is not None else ''
            return _start(request, write(data, text, get_language(), bib, keep=bool(request.POST.get('keep')),
                                         refcheck_key=start_refcheck(request, bib)))
    searches = Generated.objects.filter(kind='missed', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/missed.html', {'paid': paid, 'error': error, 'searches': searches,
                                                      'refcheck_allowed': access.has_tool(request.user, 'refcheck'),
                                                      'per_day': PER_DAY, 'upload_max_bytes': MAX_UPLOAD, 'upload_max_mb': MAX_UPLOAD // (1024 * 1024),
                                                      'bib_max_bytes': MAX_BIB_UPLOAD})


def start_refcheck(request, bib):
    """The bibliography check of the same .bib, when asked for: the key of its report, else ''."""
    from . import refcheck
    if not (request.POST.get('refcheck') and refcheck.parse_bibtex(bib) and access.has_tool(request.user, 'refcheck')
            and refcheck._spend(request.user)):
        return ''
    load = refcheck.loader(bib, '', 'main.tex', None, None, '', None)
    return generated.start('refcheck', uuid.uuid4().hex, get_language(), refcheck.write(load, get_language(), request.user),
                           user=request.user).key


def kept_draft(row):
    return Generated.objects.filter(kind=DRAFT_KIND, key=row.key, user=row.user).first()


def shown(row):
    """The result as its page shows it; a result of the first version gets the fields of the second."""
    result = row.data() if row.status == 'done' else None
    if not result:
        return result
    if result.get('version', 1) < 2:
        for item in result.get('items', []):
            item['key'] = item.get('arxiv_id', '')
            item['level'] = neighbours.level(item.get('similarity', 0) / 100)[0]
            item.setdefault('importance', 'low')
    groups = {k: [] for k in IMPORTANCES}
    for item in result.get('items', []):
        if item.get('mark') not in ('cited', 'irrelevant'):
            groups[item['importance'] if item['importance'] in groups else 'low'].append(item)
    result['groups'] = [(k, groups[k]) for k in IMPORTANCES if groups[k]]
    result['marked'] = [i for i in result.get('items', []) if i.get('mark') in ('cited', 'irrelevant')]
    result['active'] = [i for i in result.get('items', []) if i.get('mark') not in ('cited', 'irrelevant')]
    result['new_count'] = sum(1 for i in result['active'] if i.get('new'))
    result['day'] = datetime.date.fromisoformat(result['searched_on']) if result.get('searched_on') else row.created.date()
    if result.get('previous'):
        result['previous']['day'] = datetime.date.fromisoformat(result['previous']['searched'])
    return result


def links(result):
    """Where the draft and its papers go next, in the other tools."""
    english = result.get('english') or result.get('title') or ''
    keys = [i['key'] for i in result['active'] if i['importance'] in ('high', 'medium')] or [i['key'] for i in result['active']]
    return {'scoops': urlencode({'title': english[:280], 'description': (result.get('abstract') or english)[:5000]}),
            'literature': urlencode({'question': english[:600]}),
            'teams': urlencode({'topic': (result.get('core') or english)[:600]}),
            'chat': ','.join(keys[:assistant.MAX_PAPERS]), 'chat_max': assistant.MAX_PAPERS}


def _row(request, key):
    if not access.has_tool(request.user, 'missed'):
        raise Http404
    row = Generated.objects.filter(kind='missed', key=key, user=request.user).first()
    if row is None:
        raise Http404
    return row


def mark(row, key, value):
    """Mark one paper of the result: already cited, not relevant, added to the references, or nothing."""
    data = row.data()
    for item in data.get('items', []):
        if item.get('key', item.get('arxiv_id')) == key:
            item['mark'] = value if value in MARKS else ''
            row.content = json.dumps(data)
            row.save(update_fields=['content', 'updated'])
            return True
    return False


def search_again(request, row):
    """The same draft searched again, from its kept text: the papers the previous search had not found are new,
    and the papers marked already cited or not relevant stay out."""
    from . import scoops
    draft = kept_draft(row)
    if draft is None:
        raise Http404
    if not _spend(request.user):
        return redirect('/%s/missed/%s/?m=quota' % (get_language(), row.key))
    kept = draft.data()
    data = row.data()
    items = data.get('items', [])
    previous = {'key': row.key, 'searched': data.get('searched_on') or row.created.date().isoformat(),
                'seen': [i.get('key', i.get('arxiv_id')) for i in items] + [literature.title_key(i['title']) for i in items]
                + [p['key'] for p in data.get('probable', [])],
                'marks': dict(data.get('carried') or {}, **{i.get('key', i.get('arxiv_id')): i['mark'] for i in items
                                                         if 'mark' in i})}
    previous['marks'] = {k: v for k, v in previous['marks'].items() if v}
    previous['facets'] = data.get('facets')
    previous['seen'] = set(previous['seen'])
    text = scoops.unseal(kept.get('text', ''))
    if not text:
        return redirect('/%s/missed/%s/?m=unreadable' % (get_language(), row.key))
    return _start(request, write(None, text, get_language(), scoops.unseal(kept.get('bib', '')), keep=True,
                                 previous=previous))


def save_collection(request, row, result):
    paid = access.has_tool(request.user, 'collections')
    if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
        return redirect('/%s/missed/%s/?m=too_many' % (get_language(), row.key))
    collection = Collection.objects.create(user=request.user, name=str(result.get('title') or _('What did I miss?'))[:120])
    known = {i['key']: (i['title'], i['authors']) for i in result['active']}
    for item in result['active']:
        library.add_paper(collection, item['key'], paid, known)
    return redirect('/%s/missed/%s/?m=saved&c=%s#next' % (get_language(), row.key, collection.pk))


@login_required
def missed_result(request, key):
    row = _row(request, key)
    if request.method == 'POST' and row.status == 'done':
        action = request.POST.get('action')
        if result_share.act(row, action):
            return redirect('/%s/missed/%s/' % (get_language(), row.key))
        if action == 'mark':
            mark(row, request.POST.get('key', ''), request.POST.get('mark', ''))
            return redirect('/%s/missed/%s/#%s' % (get_language(), row.key, request.POST.get('anchor', 'papers')))
        if action == 'again':
            return search_again(request, row)
        if action == 'forget':
            Generated.objects.filter(kind=DRAFT_KIND, key=row.key, user=request.user).delete()
            return redirect('/%s/missed/%s/?m=forgotten' % (get_language(), row.key))
        if action == 'save':
            return save_collection(request, row, shown(row))
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    result = shown(row)
    return render(request, 'summarizer/missed_result.html', {
        'row': row, 'result': result, 'paid': True, 'steps': STEPS, 'message': request.GET.get('m'),
        'links': links(result) if result else None, 'kept': result is not None and kept_draft(row) is not None,
        'per_day': PER_DAY, 'share_url': result_share.shared_url(request, row, 'missed'),
        'refcheck': Generated.objects.filter(kind='refcheck', key=result.get('refcheck_key'), user=request.user).first()
        if result and result.get('refcheck_key') else None,
        'collection': Collection.objects.filter(pk=request.GET.get('c'), user=request.user).first()
        if request.GET.get('c', '').isdigit() else None})


def missed_shared(request, token):
    """The read-only page of a shared search, for co-authors: the papers the draft does not cite, never who sent
    it, nor the sentences of the draft."""
    row = result_share.shared_row('missed', token)
    return render(request, 'summarizer/missed_result.html', {'row': row, 'result': shown(row), 'shared': True,
                                                             'token': token})


# --- exports -----------------------------------------------------------------

def metas(result):
    """The papers not marked already cited or not relevant, as the exports read them (no lookup)."""
    return [{'arxiv_id': i['key'], 'title': i['title'], 'authors': i.get('all_authors') or
             [a for a in i['authors'].replace(' et al.', '').split(', ') if a], 'year': i.get('year') or '', 'date': '',
             'category': '', 'abstract': i.get('abstract', ''), 'doi': i.get('doi') or '', 'journal': i.get('venue') or ''}
            for i in result['active']]


def export_result(row, fmt):
    result = shown(row)
    if fmt not in EXPORTS or not result or not result['active']:
        raise Http404
    name = 'missed-' + (re.sub(r'[^A-Za-z0-9]+', '-', result.get('english') or result.get('title') or '')[:50].strip('-') or 'papers')
    if fmt == 'bib':
        entries, _keys = literature.cite_keys([exports.bibtex_entry(m) for m in metas(result)])
        return exports._download('\n'.join(entries), name + '.bib', 'application/x-bibtex; charset=utf-8')
    with translation.override(row.lang):
        return exports.export_many(metas(result), 'ris', name, row.lang)


@login_required
def missed_export(request, key, fmt):
    row = _row(request, key)
    if row.status != 'done':
        raise Http404
    return export_result(row, fmt)


def missed_shared_export(request, token, fmt):
    return export_result(result_share.shared_row('missed', token), fmt)


def zotero_metas(user, key):
    """The papers of one of the user's searches, for 'Send to Zotero'."""
    row = Generated.objects.filter(kind='missed', key=key, user=user, status='done').first()
    return metas(shown(row)) if row is not None else []
