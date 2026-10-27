"""Press kit of a paper (Premium+): what a researcher needs to talk about a paper in public, or to answer for it.

Three readers use it, and the page shows the part of the reader's role (ROLES) above what they share:
- the authors: a draft release for their press office (with a place for their own quotes, never written for
  them), the questions journalists are likely to ask with what the paper lets them answer, the likely
  misreadings, a lay summary and a short thread;
- a researcher asked to comment on someone else's paper: what it shows and does not, its strengths and
  weaknesses each with its passage, what would settle the main claim;
- whoever writes about it: reporting angles, wording to check, questions for the authors.

One kit per paper and language, shared by every reader, written from the full text (or only the abstract when
the license does not let us read more) in two model calls made at the same time. What is checked rather than
trusted to the model:

- every passage it quotes must be in the text we gave it, word for word or up to what a PDF extraction changes
  (evidence.find), and is shown with its section, figure and a link to it on arXiv;
- the numbers of its explanation that the paper never states are listed;
- the novelty is set against the abstracts of the closest earlier papers we know, not the model's memory;
- researchers suggested for outside comment sign those papers first, last or as corresponding author, and are
  neither authors of this one nor their co-authors of the last five years (OpenAlex); their independence still
  needs checking;
- figures come from the paper's own HTML, only under an open license, with their credit.

evidence and insight import this module (normalized, quote_match): they are imported inside the functions here.
"""
import contextvars
import datetime
import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor

from django.db import connections
from django.http import Http404, HttpResponse, JsonResponse
from django.shortcuts import render
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_GET, require_POST

from . import access, generated, paper_source, publication_status, result_share, similar, sources, student_papers
from .assistant import SOURCE_POLICY, MAX_CONTEXT_CHARS
from .digest import name_key, same_author
from .models import AnnouncedPaper, ArxivPaper, Author, Generated, Letter, PaperScore, PaperVersion, PressKitReport
from .library import canonical_id

logger = logging.getLogger(__name__)

SCHEMA = 3
EXPERTS = 6
NEIGHBOURS = 30  # closest papers we know: candidates for outside comment
RELATED = 5  # of them, the earlier ones whose abstracts set the novelty in context
RELATED_CHARS = 1200
FIGURES_SHOWN = 20  # captions the model reads to choose the figures for the press
COAUTHOR_YEARS = 5
MAX_AUTHOR_IDS = 25  # authors of this paper whose co-authors are looked up (OpenAlex takes 100 values)
KINDS = ('result', 'interpretation')  # as insight.KINDS: measured, computed or proved / concluded or suggested
ROLES = [('author', _('It is my paper')), ('commentator', _('I am asked to comment')), ('writer', _('I write about it'))]
ROLE_KEYS = [key for key, _label in ROLES]
DEMO_KITS = {'2307.01574v1'}  # open to every visitor, so that a kit can be seen before subscribing
STEPS = [
    ('read', _('Reading the full paper')),
    ('publication', _('Checking publication records')),
    ('experts', _('Finding earlier papers and researchers for outside comment')),
    ('write', _('Writing the kit and checking every quote against the paper')),
]
LEVELS = ('ok', 'caution', 'warning')
LICENSE_TERMS = {
    'by': _('Reuse allowed with this credit.'),
    'by-sa': _('Reuse allowed with this credit, under the same license.'),
    'by-nc': _('Reuse with this credit for non-commercial use only: check with your outlet.'),
    'by-nc-sa': _('Reuse with this credit for non-commercial use only, under the same license: check with your outlet.'),
    'zero': _('Public domain.'), 'mark': _('Public domain.'),
}

PROMPT = """You prepare a press kit about the paper in source_documents, for researchers who will talk about it in public
and for the people who write about it. Write in {language}.
Be precise and sober: no hype, no claim the paper does not make. Flag what a careful reader must check.
Answer with a JSON object:
{{"headline": "a factual headline, 12 words at most",
  "plain_explanation": "3 to 5 sentences a general reader understands",
  "whats_new": "2 to 4 sentences: what is really new compared with earlier work, as the paper itself states it",
  "novelty": {{"assessment": "1 or 2 sentences: how the novelty the paper claims stands against related_work",
              "comparisons": [{{"paper": "the id of a related_work paper, exactly", "already": "what it already did, from its abstract",
                               "adds": "what this paper adds or does differently"}}]}},
  "hype_check": [{{"signal": "short name", "detail": "1 or 2 sentences", "level": "ok|caution|warning"}}],
  "verdict": "1 or 2 sentences on how solid the main claim is and how far it can be generalised",
  "evidence_cards": [{{"claim": "one precise main finding", "kind": "result|interpretation",
                      "quote": "an exact source passage supporting it", "why_it_matters": "its significance",
                      "limitation": "what this passage does not establish"}}],
  "key_sentences": ["sentences copied word for word from the paper, 3 to 5 of them, that state its main results"],
  "angles": [{{"angle": "a specific reporting angle", "why_now": "why it may matter to readers now",
              "reporting_step": "a concrete independent check or interview needed"}}],
  "wording_checks": [{{"avoid": "a tempting but misleading claim", "prefer": "a defensible formulation",
                      "reason": "why the distinction matters"}}],
  "questions_for_authors": [{{"question": "a question a journalist should ask the authors",
                             "in_paper": "what the paper already says about it, 1 or 2 sentences, or an empty string",
                             "quote": "the exact passage that says it, or an empty string"}}],
  "press_figures": [{{"label": "a figure label exactly as in figures", "why": "what it shows a general reader, 1 sentence"}}],
  "glossary": [{{"term": "...", "definition": "one sentence"}}]}}
Give 3 to 4 evidence cards, 2 to 3 distinct reporting angles, 2 to 3 wording checks and 4 to 6 questions.
"result" is what the authors measured, computed or proved; "interpretation" what they conclude, suggest or speculate.
For hype_check, look at least at: peer-review status (unknown unless established in the supplied source), size of the
sample or data set, simulation versus observation or experiment, statistical significance and error bars, whether the
claims in the abstract go beyond the results, conflicts of interest or funding if stated, and limitations the authors
themselves give. Use "ok" when a point is reassuring.
related_work holds the abstracts of the closest papers we know, chosen automatically: compare with them only, at most
3 comparisons, and say so when they do not allow a judgement. With no related_work, give an empty novelty.
press_figures: at most 2 of the supplied figures that a general reader understands; an empty list when none is supplied.
rising_topic, when supplied, is a measured rise of the paper's topic among new papers: use it for why_now, never invent
a trend. For questions, in_paper keeps a journalist from asking what the paper already answers; leave it empty when
the paper does not answer it.
Copy quotes exactly, in their ORIGINAL language: 1 to 3 consecutive sentences, no ellipsis; explanations use {language}.
Do not invent a quote, number, contact, publication status, peer review, independence, replication or section/page.
Use the publication_check metadata to establish what has already been checked, with its date and sources.
Do not ask a generic question about editorial/publication status if a publication has been found.
Instead ask about remaining uncertainty: the version to cite, subsequent corrections, data, or changes between versions.
For a declaration on arXiv, say it is author-declared, not independently confirmed. Missing records do not prove non-publication.
A journal reference alone does not establish peer review. Missing disclosure is unknown, not proof of no conflict.
Explain absolute effects, sample sizes, uncertainty, comparison groups and causal limits when the source supports them.
Distinguish measured results, simulations, associations and speculation. Never turn association into causation.
For abstract-only or partial sources, explicitly limit the analysis to those sources. Do not claim to have read more.
Use clear plain text suitable for a newsroom, with no Markdown or LaTeX commands."""

ROLE_PROMPT = """You prepare two parts of a press kit about the paper in source_documents. Write in {language}.
Be precise and sober: no hype, nothing the paper does not support. Answer with a JSON object:
{{"author": {{
    "press_release": {{"title": "a factual title, 12 words at most", "lede": "1 or 2 sentences: the result and why it matters",
                      "body": ["3 or 4 short paragraphs: the question, what was done, what was found with its numbers and
                               uncertainty, what it does not show yet"],
                      "quote_slot": "what a quote of an author could explain here, 1 sentence"}},
    "likely_questions": [{{"question": "a question a journalist is likely to ask the authors",
                          "answer": "what the paper itself lets the authors answer, 2 or 3 sentences",
                          "quote": "the exact passage of the paper that supports the answer, or an empty string"}}],
    "misreadings": [{{"misreading": "a likely misreading of the result in the press", "correction": "how to prevent it, 1 or 2 sentences"}}],
    "lay_summary": "about 100 words for an institution's news page or a grant report",
    "social_posts": ["3 or 4 posts of a short thread, 260 characters at most each, the first one standing alone"]}},
  "commentator": {{
    "in_brief": ["3 to 5 lines, each saying one thing the paper shows or does not show"],
    "strengths": [{{"point": "a strength, 1 sentence", "quote": "the exact passage that shows it"}}],
    "weaknesses": [{{"point": "a weakness or open question, 1 sentence", "quote": "the exact passage that shows it, or an empty string"}}],
    "what_would_convince": "the further evidence that would settle the main claim, 1 or 2 sentences",
    "comment_draft": "2 sentences a researcher asked for an outside comment could adapt in their own words: what is solid, what remains open"}}}}
Give 4 to 6 likely questions, 2 or 3 misreadings, 2 or 3 strengths and 2 or 3 weaknesses.
The authors write for their own paper: the release says "the team", never invents what an author said, and leaves the
quote to them (quote_slot only says what it could explain). The comment is a draft for someone else's paper, never
signed by anyone.
Copy quotes exactly, in their ORIGINAL language: 1 to 3 consecutive sentences, no ellipsis; everything else in {language}.
Do not invent a number, result, funding, publication status or peer review. For abstract-only or partial sources,
limit what you write to them. Plain text: no Markdown, no LaTeX commands, no hashtags, no emoji."""


def normalized(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()


def quote_match(sentence, source):
    sentence = ' '.join(str(sentence).split())
    if not 30 < len(sentence) <= 1800:
        return None
    # Preserve signs, decimal separators and punctuation: +5 is not -5.
    pattern = r'\s+'.join(re.escape(word) for word in sentence.split())
    return re.search(pattern, source, re.I)


def clean(value, limit):
    return ' '.join(str(value or '').split())[:limit]


def passage(quote, paper):
    """The quoted passage as the paper has it, with where it is (section, paragraph, figure caption, a link that
    opens arXiv's HTML on it), or None when the text we gave the model does not contain it."""
    from . import evidence
    quote = clean(quote, 1800)
    match = evidence.find(quote, paper['text']) if len(quote) > 30 else None
    if not match:
        return None
    place = evidence.location(match, paper)
    return {'quote': ' '.join(match.group().split()), 'location': place['section'], 'paragraph': place['paragraph'],
            'figure': place['caption'], 'link': evidence.passage_link(match.group(), paper)}


def verified_sentences(sentences, source):
    """Only the sentences that really are in the paper text (source: the text, or the paper dict of passage)."""
    paper = source if isinstance(source, dict) else {'text': source, 'source': 'text', 'arxiv_id': ''}
    kept = []
    for sentence in sentences if isinstance(sentences, list) else []:
        found = passage(sentence, paper)
        if found and found['quote'] not in kept:
            kept.append(found['quote'])
    return kept[:5]


def numbers_off(claim, quote):
    """Whether the claim states a number its passage does not (up to rounding)."""
    from . import evidence
    return bool(evidence.numbers(claim) - evidence.with_roundings(evidence.numbers(quote)))


def evidence_cards(value, paper, abstract_only):
    cards = []
    for card in value if isinstance(value, list) else []:
        if not isinstance(card, dict) or not card.get('claim'):
            continue
        found = passage(card.get('quote', ''), paper)
        claim = clean(card['claim'], 1200)
        cards.append({'claim': claim, 'kind': card.get('kind') if card.get('kind') in KINDS else 'interpretation',
                      'why_it_matters': clean(card.get('why_it_matters'), 1600),
                      'limitation': clean(card.get('limitation'), 1600),
                      'quote': found['quote'] if found else '', 'verified': bool(found),
                      'numeric_mismatch': bool(found and numbers_off(claim, found['quote'])),
                      'location': found['location'][:160] if found else '', 'paragraph': found['paragraph'] if found else None,
                      'figure': found['figure'] if found else '',
                      'link': found['link'] if found else '', 'abstract_only': abstract_only})
        if len(cards) == 4:
            break
    return cards


def numbers_absent(texts, source):
    """The values of our explanation that the paper text never states (rounded or not): to check first."""
    from . import evidence
    stated = evidence.with_roundings(evidence.numbers(source))
    return sorted(set().union(*[evidence.numbers(t) for t in texts if t]) - stated) if any(texts) else []


def records(value, fields, limit):
    return [{field: str(item.get(field, '')).strip()[:1800] for field in fields}
            for item in value if isinstance(item, dict) and item.get(fields[0])][:limit] if isinstance(value, list) else []


def quoted_records(value, fields, paper, limit):
    """records() whose 'quote' is kept, with its place, only when the paper has it."""
    found = []
    for item in value if isinstance(value, list) else []:
        if not isinstance(item, dict) or not clean(item.get(fields[0]), 10):
            continue
        record = {field: clean(item.get(field), 1800) for field in fields}
        record['passage'] = passage(item.get('quote'), paper)
        found.append(record)
        if len(found) == limit:
            break
    return found


def provenance(paper, scope):
    return {'paper_id': paper.arxiv_id, 'title': paper.title or paper.arxiv_id,
            'url': sources.ref_url(paper.arxiv_id), 'doi': paper.link_doi or '',
            'journal_ref': paper.journal_ref or '', 'license': paper.license or '',
            'published': paper.published_arxiv.isoformat() if paper.published_arxiv else '',
            'scope': scope}


# --- earlier papers and outside commentators ------------------------------------------------

def neighbours(paper):
    """The closest papers we know, closest first (computed when they are missing)."""
    if not PaperScore.objects.filter(from_paper=paper, active=True).exists():
        try:
            similar.find_and_store(paper.arxiv_id)
        except Exception:
            logger.exception('Similar papers of %s failed', paper.arxiv_id)
            return []
    scores = PaperScore.objects.filter(from_paper=paper, active=True).order_by('score').select_related('to_paper')
    return [score.to_paper for score in scores[:NEIGHBOURS] if score.to_paper.arxiv_id != paper.arxiv_id]


def related_work(paper, papers):
    """The RELATED closest papers posted before this one (later ones only when too few are), with their abstracts."""
    date = paper.published_arxiv
    earlier = [p for p in papers if p.abstract and p.published_arxiv and date and p.published_arxiv <= date]
    later = [p for p in papers if p.abstract and p not in earlier]
    return [{'id': p.arxiv_id, 'title': clean(p.title, 300), 'date': p.published_arxiv.isoformat() if p.published_arxiv else '',
             'abstract': clean(p.abstract, RELATED_CHARS)}
            for p in (earlier + later[:max(0, 3 - len(earlier))])[:RELATED]]


def novelty(value, related):
    value = value if isinstance(value, dict) else {}
    known = {item['id']: item for item in related}
    comparisons = []
    for item in value.get('comparisons') or []:
        if not isinstance(item, dict):
            continue
        other = known.get(str(item.get('paper') or '').strip())
        if other and clean(item.get('adds'), 10) and other['id'] not in [c['arxiv_id'] for c in comparisons]:
            comparisons.append({'arxiv_id': other['id'], 'title': other['title'], 'date': other['date'],
                                'already': clean(item.get('already'), 800), 'adds': clean(item.get('adds'), 800)})
    return {'assessment': clean(value.get('assessment'), 800) if related else '', 'comparisons': comparisons[:3],
            'compared': len(related)}


def paper_doi(key):
    if sources.is_external(key):
        return sources.key_doi(key).lower()
    return '10.48550/arxiv.%s' % re.sub(r'v\d+$', '', key.replace('--', '/')).lower()


def openalex(filters, select, per_page, sort=''):
    """OpenAlex works for these filters, counted against the tools' daily budget; None when unavailable."""
    if not sources.tool_search_allowed():
        return None
    params = {'filter': filters, 'per_page': per_page, 'select': select}
    if sort:
        params['sort'] = sort
    try:
        body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params(params))
    except sources.SourceUnavailable:
        return None
    return (body or {}).get('results') or []


def institution_ids(authorship):
    from .students import umbrella  # a national umbrella (CNRS) is not one team
    return {i['id'] for i in authorship.get('institutions') or [] if i.get('id') and not umbrella(i)}


def coauthors(author_ids):
    """OpenAlex ids of the people who signed a paper of the last COAUTHOR_YEARS years with one of these authors
    (large collaborations apart), or None when unavailable."""
    since = (timezone.now().date() - datetime.timedelta(days=365 * COAUTHOR_YEARS)).isoformat()
    works = openalex('author.id:%s,from_publication_date:%s' % ('|'.join(author_ids), since),
                     'id,authorships', 200, sort='publication_date:desc')
    if works is None:
        return None
    found = set()
    for work in works:
        authorships = work.get('authorships') or []
        if len(authorships) <= student_papers.BIG_TEAM:
            found.update((a.get('author') or {}).get('id') for a in authorships)
    found.discard(None)
    return found


def openalex_experts(paper, papers):
    """(experts, people left out because they sign with the authors or share their institution, whether the
    co-authors were checked: not when OpenAlex does not know this paper), or None.

    Candidates sign one of the closest papers first, last or as corresponding author (student_papers.author_roles),
    and are neither authors of this paper nor their co-authors of the last years."""
    dois = {paper_doi(p.arxiv_id): p for p in [paper] + papers if paper_doi(p.arxiv_id)}
    if not dois:
        return None
    works = openalex('doi:' + '|'.join(list(dois)[:50]),
                     'id,doi,display_name,authorships,publication_date,corresponding_author_ids', 50)
    if works is None:
        return None
    by_doi = {student_papers.doi(work): work for work in works}
    own = by_doi.get(paper_doi(paper.arxiv_id))
    own_names = [a.name for a in paper.authors.all()]
    own_ids, own_institutions = set(), set()
    if own:
        for authorship in own.get('authorships') or []:
            if (authorship.get('author') or {}).get('id'):
                own_ids.add(authorship['author']['id'])
            own_institutions |= institution_ids(authorship)
    linked = coauthors(sorted(own_ids)[:MAX_AUTHOR_IDS]) if own_ids else set()
    if linked is None:
        return None
    people, left_out = {}, set()
    for doi, work in by_doi.items():
        neighbour = dois.get(doi)
        if neighbour is None or neighbour is paper:
            continue
        roles = student_papers.author_roles(work)
        for authorship in work.get('authorships') or []:
            author = authorship.get('author') or {}
            aid, name = author.get('id'), display_name(author.get('display_name'))
            if not roles.get(aid) or not name:
                continue
            key = name_key(name)
            if aid in own_ids or aid in linked or any(key and same_author(key, n) for n in own_names):
                left_out.add(aid)
                continue
            if institution_ids(authorship) & own_institutions:
                left_out.add(aid)  # same institution as an author: not an outside view
                continue
            institution = next((i for i in authorship.get('institutions') or [] if i.get('display_name')), {})
            person = people.setdefault(aid, {'name': name, 'affiliation': institution.get('display_name') or '',
                                             'country': institution.get('country_code') or '',
                                             'orcid': author.get('orcid') or '', 'papers': [], 'checked': True})
            person['papers'].append({'arxiv_id': neighbour.arxiv_id, 'title': clean(neighbour.title, 300),
                                     'year': (work.get('publication_date') or '')[:4], 'roles': roles[aid]})
    ranked = sorted(people.values(), key=lambda p: (-len(p['papers']), -max(int(x['year'] or 0) for x in p['papers']), p['name']))
    experts = []
    for person in ranked[:EXPERTS]:
        latest = max(person['papers'], key=lambda x: x['year'])
        experts.append(dict({k: v for k, v in person.items() if k != 'papers'}, arxiv_id=latest['arxiv_id'],
                            title=latest['title'], year=latest['year'], roles=latest['roles'], count=len(person['papers'])))
    return experts, len(left_out - set(people)), bool(own_ids)


def display_name(name):
    from .teams import person_name  # 'Cassese, Ben' -> 'Ben Cassese'
    return person_name(name or '')


def local_experts(paper, papers):
    """Without OpenAlex: the first and last authors of the closest papers, not signing this one. Co-authorships unchecked."""
    own = [a.name for a in paper.authors.all()]
    found, seen = [], set()
    for other in papers:
        authors = list(other.authors.order_by('paperauthor__author_order'))
        for author in ([authors[0], authors[-1]] if len(authors) > 1 else authors):
            key = name_key(author.name)
            if not key or key in seen or any(same_author(key, n) for n in own):
                continue
            seen.add(key)
            found.append({'name': author.name, 'affiliation': author.affiliation or '', 'country': '', 'orcid': '',
                          'arxiv_id': other.arxiv_id, 'title': clean(other.title, 300),
                          'year': other.published_arxiv.year if other.published_arxiv else '', 'roles': [],
                          'count': 1, 'checked': False})
            if len(found) == EXPERTS:
                return found
    return found


def experts(paper, papers=None):
    """(researchers to approach for outside comment, whether their co-authorships were checked, people left out)."""
    papers = neighbours(paper) if papers is None else papers
    try:
        found = openalex_experts(paper, papers)
    except Exception:
        logger.exception('Outside commentators of %s failed', paper.arxiv_id)
        found = None
    if found is not None and found[0]:
        return found[0], found[2], found[1]
    return local_experts(paper, papers), False, 0


# --- figures and topic ------------------------------------------------------------------------

def figures_of(paper):
    from .chat_figures import paper_gallery
    try:
        data = paper_gallery(paper)
    except Exception:
        logger.exception('Figures of %s failed', paper.arxiv_id)
        return []
    return data.get('figures') or [] if data.get('status') == 'ready' else []


def byline(paper):
    """'Ann Lee et al.'"""
    names = [a.name for a in paper.authors.all()]
    return names[0] + (' et al.' if len(names) > 1 else '') if names else ''


def citation(paper):
    """How a copied passage is referenced: 'Ann Lee et al., arXiv: 2101.00001v1'."""
    return ', '.join(part for part in [byline(paper), sources.label(paper.arxiv_id)] if part) if isinstance(paper, ArxivPaper) else ''


def press_figures(value, figures, paper):
    """The figures the model chose, with the image of the paper, its caption and the credit its license asks."""
    by_label = {normalized(f['label']): f for f in figures}
    code, _version = sources.license_code(paper.license)
    chosen = []
    for item in value if isinstance(value, list) else []:
        figure = by_label.get(normalized(str((item or {}).get('label') or ''))) if isinstance(item, dict) else None
        if figure is None or figure['label'] in [c['label'] for c in chosen]:
            continue
        credit = ', '.join(part for part in [byline(paper), figure['label'], sources.label(paper.arxiv_id),
                                              sources.license_label(paper.license or '')] if part)
        chosen.append({'label': figure['label'], 'image': figure['images'][0], 'caption': figure['caption'][:1500],
                       'source_url': figure['source_url'], 'why': clean(item.get('why'), 400), 'credit': credit,
                       'license_code': code})
        if len(chosen) == 2:
            break
    return chosen


def rising(paper, lang):
    from . import trends
    try:
        found = trends.topic_of_paper(paper.arxiv_id.replace('--', '/'), lang)
    except Exception:
        return None
    if not found:
        return None
    scope, week, topic = found
    return {'topic': topic.get('title', ''), 'growth': topic.get('growth'), 'field': str(trends.scope_name(scope)),
            'week': week.isoformat() if hasattr(week, 'isoformat') else str(week)}


# --- writing ------------------------------------------------------------------------------------

def at_once(*calls):
    """The results of these calls, run at the same time, each in its own thread with the caller's AI usage task;
    an exception is returned in place of its result."""
    def run(context, call):
        try:
            return context.run(call)
        except Exception as error:
            return error
        finally:
            connections.close_all()
    with ThreadPoolExecutor(len(calls)) as pool:
        futures = [pool.submit(run, contextvars.copy_context(), call) for call in calls]
        return [future.result() for future in futures]


def author_part(value, paper):
    value = value if isinstance(value, dict) else {}
    release = value.get('press_release') if isinstance(value.get('press_release'), dict) else {}
    body = [clean(p, 1500) for p in release.get('body') or [] if clean(p, 1500)] if isinstance(release.get('body'), list) else []
    return {'press_release': {'title': clean(release.get('title'), 200), 'lede': clean(release.get('lede'), 600),
                              'body': body[:5], 'quote_slot': clean(release.get('quote_slot'), 400)},
            'likely_questions': quoted_records(value.get('likely_questions'), ['question', 'answer'], paper, 6),
            'misreadings': records(value.get('misreadings'), ['misreading', 'correction'], 3),
            'lay_summary': clean(value.get('lay_summary'), 1200),
            'social_posts': [clean(p, 300) for p in value.get('social_posts') or [] if clean(p, 300)][:5]
            if isinstance(value.get('social_posts'), list) else []}


def commentator_part(value, paper):
    value = value if isinstance(value, dict) else {}
    brief = value.get('in_brief') if isinstance(value.get('in_brief'), list) else []
    return {'in_brief': [clean(line, 400) for line in brief if clean(line, 400)][:5],
            'strengths': quoted_records(value.get('strengths'), ['point'], paper, 3),
            'weaknesses': quoted_records(value.get('weaknesses'), ['point'], paper, 3),
            'what_would_convince': clean(value.get('what_would_convince'), 600),
            'comment_draft': clean(value.get('comment_draft'), 600)}


def question_answers(value, paper):
    """{question: what the paper already says about it, with its passage} for the questions it answers."""
    found = {}
    for item in value if isinstance(value, list) else []:
        if isinstance(item, dict) and clean(item.get('question'), 600) and clean(item.get('in_paper'), 10):
            found[clean(item['question'], 600)] = {'answer': clean(item['in_paper'], 800),
                                                   'passage': passage(item.get('quote'), paper)}
    return found


def questions_of(value):
    """The questions as text, from the current answer ({question, in_paper, quote}) or an older kit's strings."""
    found = []
    for item in value if isinstance(value, list) else []:
        text = clean(item.get('question') if isinstance(item, dict) else item, 600)
        if text and text not in found:
            found.append(text)
    return found[:6]


def import_paper(arxiv_id):
    """Import only on explicit Premium generation, never on a public GET."""
    from .views import arxiv_metadata_or_none
    from .utils import updatearvixdatapaper
    from django.utils.dateparse import parse_datetime
    metadata = arxiv_metadata_or_none(arxiv_id)
    if not metadata or not metadata[0]:
        raise ValueError('Paper metadata is temporarily unavailable')
    keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract',
            'category', 'updated', 'published_arxiv', 'journal_ref', 'comments', 'license']
    fields = dict(zip(keys, metadata[1:]))
    fields['published_arxiv'] = parse_datetime(fields['published_arxiv']).date()
    updatearvixdatapaper(arxiv_id, fields)


def read(paper):
    """(text given to the model, where it comes from: 'html', 'pdf', 'index'... or 'abstract', abstract_only)."""
    from .models import PaperText
    if sources.is_open_license(paper.license):
        available = paper_source.paper_text(paper.arxiv_id)
        if not available or normalized(available) == normalized(paper.abstract or ''):
            full_text, origin = paper_source.read_again(paper.arxiv_id)
            if full_text:
                paper_source.store_text(paper.arxiv_id, full_text, origin)
                available = full_text
            else:
                available = ''
        if available:
            origin = PaperText.objects.filter(arxiv_id=paper.arxiv_id).values_list('source', flat=True).first() or 'text'
            return available, origin, False
    return paper.abstract or '', 'abstract', True


def write(arxiv_id, lang):
    def work(pk):
        generated.progress(pk, 'read')
        if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
            import_paper(arxiv_id)
        paper = ArxivPaper.objects.prefetch_related('authors').get(arxiv_id=arxiv_id)
        available, origin, abstract_only = read(paper)
        scope = 'abstract' if abstract_only else 'excerpt' if len(available) > MAX_CONTEXT_CHARS else 'full_text'
        source_text = available[:MAX_CONTEXT_CHARS]
        source = {'text': source_text, 'source': origin, 'arxiv_id': paper.arxiv_id}
        generated.progress(pk, 'publication')
        publication = publication_status.for_generation(paper)
        generated.progress(pk, 'experts')
        papers = neighbours(paper)
        related = related_work(paper, papers)
        people, checked, left_out = experts(paper, papers)
        figures = figures_of(paper)
        topic = rising(paper, lang)
        generated.progress(pk, 'write')
        documents = {'metadata': provenance(paper, scope), 'text': source_text, 'publication_check': publication,
                     'related_work': related,
                     'figures': [{'label': f['label'], 'caption': clean(f['caption'], 500)} for f in figures[:FIGURES_SHOWN]]}
        if topic:
            documents['rising_topic'] = topic
        context = json.dumps({'source_documents': documents}, ensure_ascii=False)
        language = generated.language_name(lang)
        answer, roles = at_once(
            lambda: generated.chat_json(context, paid=True, max_tokens=7000,
                                        system=SOURCE_POLICY + '\n' + PROMPT.format(language=language)),
            lambda: generated.chat_json(json.dumps({'source_documents': {k: documents[k] for k in ('metadata', 'text')}},
                                                   ensure_ascii=False), paid=True, max_tokens=5000,
                                        system=SOURCE_POLICY + '\n' + ROLE_PROMPT.format(language=language)))
        if isinstance(answer, Exception):
            raise answer
        if isinstance(roles, Exception):
            logger.warning('Press kit roles of %s failed: %r', arxiv_id, roles)
            roles = {}
        kit = {
            'headline': clean(answer.get('headline'), 300),
            'plain_explanation': clean(answer.get('plain_explanation'), 2000),
            'whats_new': clean(answer.get('whats_new'), 1600),
            'novelty': novelty(answer.get('novelty'), related),
            'hype_check': [{'signal': clean(h.get('signal'), 120), 'detail': clean(h.get('detail'), 600),
                            'level': h.get('level') if h.get('level') in LEVELS else 'caution'}
                           for h in answer.get('hype_check', []) if isinstance(h, dict) and h.get('signal')][:10],
            'verdict': clean(answer.get('verdict'), 800),
            'key_sentences': verified_sentences(answer.get('key_sentences', []), source),
            'questions_for_authors': questions_of(answer.get('questions_for_authors')),
            'question_answers': question_answers(answer.get('questions_for_authors'), source),
            'glossary': [{'term': clean(g.get('term'), 120), 'definition': clean(g.get('definition'), 600)}
                         for g in answer.get('glossary', []) if isinstance(g, dict) and g.get('term')][:10],
            'experts': people, 'experts_checked': checked, 'experts_left_out': left_out,
            'figures': press_figures(answer.get('press_figures'), figures, paper),
            'publication_status': publication,
            'abstract_only': abstract_only, 'schema_version': SCHEMA, 'source': provenance(paper, scope),
            'evidence_cards': evidence_cards(answer.get('evidence_cards'), source, abstract_only),
            'angles': records(answer.get('angles'), ['angle', 'why_now', 'reporting_step'], 3),
            'wording_checks': records(answer.get('wording_checks'), ['avoid', 'prefer', 'reason'], 3),
            'roles': {'author': author_part(roles.get('author'), source),
                      'commentator': commentator_part(roles.get('commentator'), source)},
        }
        kit['numbers_absent'] = numbers_absent([kit['headline'], kit['plain_explanation'], kit['whats_new']], source_text)
        return kit
    return work


# --- pages ----------------------------------------------------------------------------------------

def paper_id(value):
    key = canonical_id(value)
    if key and re.fullmatch(r'(?:\d{4}\.\d{4,5}|[a-z-]+--\d{7})', key):
        key += 'v1'
    return key


def editorial_context(letter_id, arxiv_id, lang):
    """Display-only context, restricted to a real selection in the requested language."""
    if not str(letter_id or '').isascii() or not str(letter_id or '').isdigit():
        return None
    if len(str(letter_id)) > 18:
        return None
    letter = Letter.objects.filter(pk=letter_id, lang=lang).first()
    if letter is None:
        return None
    try:
        content = json.loads(letter.content)
        for section in content['sections']:
            ids = section.get('papers', [])
            if arxiv_id not in [paper_id(a) for a in ids]:
                continue
            return {'letter': letter, 'heading': section.get('heading', ''),
                    'text': section.get('text', ''), 'others': [
                        {'arxiv_id': paper_id(a), 'title': content.get('papers', {}).get(a, a)}
                        for a in dict.fromkeys(ids) if paper_id(a) and paper_id(a) != arxiv_id][:5]}
    except (ValueError, TypeError, KeyError, AttributeError):
        return None
    return None


def announced_paper(arxiv_id):
    # RSS IDs can be versionless. Keep explicit versions intact.
    bare = re.sub(r'v\d+$', '', arxiv_id)
    return AnnouncedPaper.objects.filter(arxiv_id__in=[arxiv_id, bare]).first()


def can_read(user, arxiv_id):
    """Premium+ (or a press kit grant), or the example kit every visitor can read."""
    return access.has_tool(user, 'presskit') or arxiv_id in DEMO_KITS


def role_of(request, paper):
    """The role asked for (?role=), else 'author' for a signed-in reader who signs the paper, else 'commentator'."""
    asked = request.GET.get('role')
    if asked in ROLE_KEYS:
        return asked
    user = request.user
    key = name_key('%s %s' % (user.first_name, user.last_name)) if user.is_authenticated else None
    if key and isinstance(paper, ArxivPaper) and any(same_author(key, a.name) for a in paper.authors.all()):
        return 'author'
    return 'commentator'


def newer_version(arxiv_id):
    """The latest version we know of an arXiv paper when the kit is about an older one, else None."""
    from . import changes, library
    if sources.is_external(arxiv_id):
        return None
    base, version = changes.base_id(arxiv_id)
    site_base = base.replace('/', '--')
    known = [version] + list(PaperVersion.objects.filter(arxiv_id=base).values_list('version', flat=True))
    known += [library.version_of(key)[1] for key in
              ArxivPaper.objects.filter(arxiv_id__startswith=site_base + 'v').values_list('arxiv_id', flat=True)]
    latest = max(known)
    return {'version': latest, 'arxiv_id': '%sv%d' % (site_base, latest), 'from': version} if version and latest > version else None


def shown(kit, lang):
    """The kit as the page and the exports show it: key sentences already quoted by a card left out, and the
    questions with what the paper already answers."""
    quoted = {' '.join(card.get('quote', '').split()) for card in kit.get('evidence_cards', []) if card.get('verified')}
    kit['key_sentences'] = [s for s in kit.get('key_sentences', []) if ' '.join(s.split()) not in quoted]
    kit['questions_for_authors'] = questions_of(kit.get('questions_for_authors'))
    kit.setdefault('roles', {})
    return kit


def questions_shown(kit, report):
    answers = kit.get('question_answers') or {}
    return [{'text': q, **(answers.get(q) or {})} for q in publication_status.questions(kit, report)]


def kit_row(request, arxiv_id):
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=request.LANGUAGE_CODE).first()
    if row and row.user_id and row.user_id != request.user.pk:
        return None
    return row


def page_context(request, paper, arxiv_id, row, kit, shared=False):
    publication = publication_status.latest(arxiv_id, kit) if kit else None
    return {
        'paper': paper, 'arxiv_id': arxiv_id, 'row': row, 'kit': kit, 'steps': STEPS, 'shared': shared,
        **publication_status.display_context(publication),
        'author_questions': questions_shown(kit, publication) if kit else [],
        'roles': ROLES, 'role': role_of(request, paper),
        'newer': newer_version(arxiv_id) if kit else None,
        'source': provenance(paper, 'unknown') if isinstance(paper, ArxivPaper) else None,
        'license_terms': LICENSE_TERMS, 'citation': citation(paper),
    }


def presskit_page(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if paper is None:
        announced = announced_paper(arxiv_id)
        if announced is None or canonical_id(arxiv_id) != arxiv_id:
            raise Http404
        paper = {'title': announced.title, 'authors': Author.objects.none()}
    paid = access.has_tool(request.user, 'presskit')
    demo = arxiv_id in DEMO_KITS
    row = kit_row(request, arxiv_id) if paid or demo else None
    if demo and not paid and row is not None and row.status != 'done' and not row.content:
        row = None
    kit = shown(row.data(), request.LANGUAGE_CODE) if row is not None and row.content else None
    context = page_context(request, paper, arxiv_id, row, kit)
    context.update({
        'paid': paid, 'demo': demo and not paid,
        'publication_stale': paid and row.status != 'pending' and publication_status.stale(context['publication']) if row else False,
        'editorial': editorial_context(request.GET.get('letter'), arxiv_id, request.LANGUAGE_CODE),
        'can_read_full_text': bool(kit and isinstance(paper, ArxivPaper) and kit.get('source', {}).get('scope') == 'abstract'
                                   and sources.is_open_license(paper.license)),
        'share_url': result_share.shared_url(request, row, 'press-kit') if paid and row and kit else '',
        'demo_ids': sorted(DEMO_KITS),
    })
    return render(request, 'summarizer/presskit.html', context)


def presskit_shared(request, token):
    """A kit shared by a reader (a press office, co-authors): read-only, for anyone with the link."""
    row = result_share.shared_row('presskit', token)
    paper = ArxivPaper.objects.filter(arxiv_id=row.key).prefetch_related('authors').first()
    if paper is None:
        raise Http404
    kit = shown(row.data(), row.lang)
    context = page_context(request, paper, row.key, row, kit, shared=True)
    context.update({'paid': False, 'demo': False, 'publication_stale': False, 'editorial': None, 'share_token': token})
    response = render(request, 'summarizer/presskit.html', context)
    response['X-Robots-Tag'] = 'noindex'
    return response


@require_POST
def presskit_start(request, arxiv_id):
    if not access.has_tool(request.user, 'presskit'):
        return JsonResponse({'status': 'premium'}, status=403)
    if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists() and (
            canonical_id(arxiv_id) != arxiv_id or announced_paper(arxiv_id) is None):
        raise Http404
    lang = request.LANGUAGE_CODE
    existing = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=lang).first()
    if existing and existing.user_id and existing.user_id != request.user.pk:
        raise Http404
    row = generated.start('presskit', arxiv_id, lang, write(arxiv_id, lang), force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})


def readable_kit(request, arxiv_id):
    """(row, paper) of a finished kit this reader may export, or 403/404."""
    if not can_read(request.user, arxiv_id):
        return None, None
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=request.LANGUAGE_CODE).exclude(content='').first()
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if row is None or paper is None or (row.user_id and row.user_id != request.user.pk):
        raise Http404
    return row, paper


def export_name(arxiv_id, lang, extension):
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '-', arxiv_id)[:100]
    return 'attachment; filename="SummarizePaper-press-kit-%s-%s.%s"' % (safe_id, lang, extension)


def private(response):
    response['Cache-Control'] = 'private, no-store'
    response['X-Content-Type-Options'] = 'nosniff'
    return response


@require_GET
def presskit_pdf(request, arxiv_id):
    row, paper = readable_kit(request, arxiv_id)
    if row is None:
        return HttpResponse(status=403)
    from .presskit_pdf import document
    editorial = editorial_context(request.GET.get('letter'), arxiv_id, request.LANGUAGE_CODE)
    role = request.GET.get('role') if request.GET.get('role') in ROLE_KEYS else None
    response = HttpResponse(document(paper, row, editorial, publication=publication_status.latest(arxiv_id, row.data()),
                                     role=role), content_type='application/pdf')
    response['Content-Disposition'] = export_name(arxiv_id, request.LANGUAGE_CODE, 'pdf')
    return private(response)


@require_GET
def presskit_markdown(request, arxiv_id):
    """The kit as Markdown, to paste into a document, a newsroom tool or an email."""
    row, paper = readable_kit(request, arxiv_id)
    if row is None:
        return HttpResponse(status=403)
    from .presskit_pdf import markdown
    role = request.GET.get('role') if request.GET.get('role') in ROLE_KEYS else None
    text = markdown(paper, row, publication=publication_status.latest(arxiv_id, row.data()), role=role)
    response = HttpResponse(text, content_type='text/markdown; charset=utf-8')
    response['Content-Disposition'] = export_name(arxiv_id, request.LANGUAGE_CODE, 'md')
    return private(response)


def status_kit(request, arxiv_id):
    if not access.has_tool(request.user, 'presskit'):
        raise Http404
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=request.LANGUAGE_CODE).exclude(content='').first()
    if row is None or (row.user_id and row.user_id != request.user.pk):
        raise Http404
    return row


@require_POST
def presskit_share(request, arxiv_id):
    """Share the kit by a read-only link (again: 90 more days), or stop sharing it."""
    row = status_kit(request, arxiv_id)
    if not result_share.act(row, request.POST.get('action')):
        return JsonResponse({'status': 'invalid'}, status=400)
    return JsonResponse({'status': 'ok', 'url': result_share.shared_url(request, row, 'press-kit')})


REPORT_SECTIONS = ('evidence', 'novelty', 'hype', 'experts', 'figures', 'author', 'commentator', 'writer', 'questions',
                   'glossary', 'other')


@require_POST
def presskit_report(request, arxiv_id):
    """A reader reports a mistake in the kit; staff read the reports in the admin."""
    if not request.user.is_authenticated or not can_read(request.user, arxiv_id):
        return JsonResponse({'status': 'premium'}, status=403)
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=request.LANGUAGE_CODE).exclude(content='').first()
    text = clean(request.POST.get('text'), 2000)
    section = request.POST.get('section') if request.POST.get('section') in REPORT_SECTIONS else 'other'
    if row is None or not text:
        return JsonResponse({'status': 'invalid'}, status=400)
    if PressKitReport.objects.filter(user=request.user, created__gte=timezone.now() - datetime.timedelta(days=1)).count() >= 30:
        return JsonResponse({'status': 'busy'}, status=429)
    PressKitReport.objects.create(arxiv_id=arxiv_id, lang=row.lang, section=section, item=clean(request.POST.get('item'), 500),
                                  text=text, user=request.user, kit_updated=row.updated)
    return JsonResponse({'status': 'ok'})


@require_POST
def publication_start(request, arxiv_id):
    status_kit(request, arxiv_id)
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper is None:
        raise Http404
    previous = publication_status.latest(arxiv_id)
    # Share public publication metadata across languages; allow refresh at most once per minute.
    row = generated.start(publication_status.KIND, arxiv_id, 'en', lambda pk: publication_status.lookup(paper),
                          force=publication_status.stale(previous, datetime.timedelta(minutes=1)))
    return JsonResponse({'id': row.pk, 'status': row.status})


@require_GET
def publication_result(request, arxiv_id):
    from django.template.loader import render_to_string
    kit = status_kit(request, arxiv_id).data()
    report = publication_status.latest(arxiv_id, kit)
    html = render_to_string('summarizer/publication_status.html', publication_status.display_context(report), request=request)
    questions = questions_shown(kit, report)
    return JsonResponse({'html': html, 'questions': [q['text'] for q in questions],
                         'questions_html': render_to_string('summarizer/presskit_questions.html',
                                                            {'author_questions': questions, 'arxiv_id': arxiv_id}, request=request)},
                        headers={'Cache-Control': 'private, no-store'})
