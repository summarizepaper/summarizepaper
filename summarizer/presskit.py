"""Press kit of a paper, for journalists (Premium).

Written once per paper and language from the full text (or only the abstract
when the license does not let us read more). Two things are checked rather
than trusted to the model:

- the key sentences must appear word for word in the text we gave it, so a
  journalist never quotes a sentence the paper does not contain;
- suggested outside commentators come from our own data (authors of related
  papers who did not sign this one); their independence still needs checking.
"""
import json
import re

from django.http import Http404
from django.shortcuts import render
from django.views.decorators.http import require_GET, require_POST
from django.http import JsonResponse, HttpResponse

from . import access, generated, paper_source, similar, sources
from .assistant import SOURCE_POLICY, MAX_CONTEXT_CHARS
from .digest import name_key
from .models import AnnouncedPaper, ArxivPaper, Author, Generated, Letter, PaperScore
from .library import canonical_id

from django.utils.translation import gettext_lazy as _

EXPERTS = 6
STEPS = [
    ('read', _('Reading the full paper')),
    ('experts', _('Finding researchers for outside comment')),
    ('write', _('Writing the kit and checking the quotes against the paper')),
]
LEVELS = ('ok', 'caution', 'warning')

PROMPT = """You prepare a press kit for a science journalist about the paper above. Write in {language}.
Be precise and sober: no hype, no claim the paper does not make. Flag what a careful journalist must check.
Answer with a JSON object:
{{"headline": "a factual headline, 12 words at most",
  "plain_explanation": "3 to 5 sentences a general reader understands",
  "whats_new": "2 to 4 sentences: what is really new compared with earlier work, as the paper itself states it",
  "hype_check": [{{"signal": "short name", "detail": "1 or 2 sentences", "level": "ok|caution|warning"}}],
  "verdict": "1 or 2 sentences on how solid the main claim is and how far it can be generalised",
  "key_sentences": ["sentences copied word for word from the paper, 3 to 5 of them, that state its main results"],
  "questions_for_authors": ["4 to 6 questions a journalist should ask the authors"],
  "glossary": [{{"term": "...", "definition": "one sentence"}}]}}
For hype_check, look at least at: peer-review status (unknown unless established in the supplied source), size of the
sample or data set, simulation versus observation or experiment, statistical significance and error bars, whether the
claims in the abstract go beyond the results, conflicts of interest or funding if stated, and limitations the authors
themselves give. Use "ok" when a point is reassuring.
Also include:
"evidence_cards": [{{"claim": "one precise main finding", "quote": "an exact source passage supporting it",
                    "why_it_matters": "its significance", "limitation": "what this passage does not establish"}}],
"angles": [{{"angle": "a specific reporting angle", "why_now": "why it may matter to readers",
             "reporting_step": "a concrete independent check or interview needed"}}],
"wording_checks": [{{"avoid": "a tempting but misleading claim", "prefer": "a defensible formulation",
                      "reason": "why the distinction matters"}}].
Give 3 to 4 evidence cards, 2 to 3 distinct reporting angles and 2 to 3 wording checks.
Copy evidence quotes exactly in their ORIGINAL language; explanations must use {language}.
Do not invent a quote, number, contact, publication status, peer review, independence, replication or section/page.
A journal reference alone does not establish peer review. Missing disclosure is unknown, not proof of no conflict.
Explain absolute effects, sample sizes, uncertainty, comparison groups and causal limits when the source supports them.
Distinguish measured results, simulations, associations and speculation. Never turn association into causation.
For abstract-only or partial sources, explicitly limit the analysis to those sources. Do not claim to have read more.
Use clear plain text suitable for a newsroom, with no Markdown or LaTeX commands."""


def normalized(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()


def verified_sentences(sentences, source):
    """Only the sentences that really are in the paper text."""
    kept = []
    for sentence in sentences if isinstance(sentences, list) else []:
        match = quote_match(sentence, source)
        if match and match.group() not in kept:
            kept.append(match.group())
    return kept[:5]


def quote_match(sentence, source):
    sentence = ' '.join(str(sentence).split())
    if not 30 < len(sentence) <= 1800:
        return None
    # Preserve signs, decimal separators and punctuation: +5 is not -5.
    pattern = r'\s+'.join(re.escape(word) for word in sentence.split())
    return re.search(pattern, source, re.I)


def evidence_cards(value, source, abstract_only):
    cards = []
    for card in value if isinstance(value, list) else []:
        if not isinstance(card, dict) or not card.get('claim'):
            continue
        match = quote_match(card.get('quote', ''), source)
        headings = list(re.finditer(r'^#{1,6}\s+(.+)$', source[:match.start()], re.M)) if match else []
        numbers = lambda text: set(re.findall(r'(?<!\w)[+-]?\d+(?:[.,]\d+)?', text))
        cards.append({'claim': str(card['claim'])[:1200], 'why_it_matters': str(card.get('why_it_matters', ''))[:1600],
                      'limitation': str(card.get('limitation', ''))[:1600],
                      'quote': match.group() if match else '', 'verified': bool(match),
                      'numeric_mismatch': bool(match and not numbers(str(card['claim'])).issubset(numbers(match.group()))),
                      'location': headings[-1][1].strip()[:160] if headings else '',
                      'abstract_only': abstract_only})
        if len(cards) == 4:
            break
    return cards


def records(value, fields, limit):
    return [{field: str(item.get(field, '')).strip()[:1800] for field in fields}
            for item in value if isinstance(item, dict) and item.get(fields[0])][:limit] if isinstance(value, list) else []


def provenance(paper, scope):
    return {'paper_id': paper.arxiv_id, 'title': paper.title or paper.arxiv_id,
            'url': sources.ref_url(paper.arxiv_id), 'doi': paper.link_doi or '',
            'journal_ref': paper.journal_ref or '', 'license': paper.license or '',
            'published': paper.published_arxiv.isoformat() if paper.published_arxiv else '',
            'scope': scope}


def experts(paper):
    """Related-paper authors absent from the source byline; not a conflict-of-interest check."""
    if not PaperScore.objects.filter(from_paper=paper, active=True).exists():
        try:
            similar.find_and_store(paper.arxiv_id)
        except Exception:
            return []
    own = {name_key(a.name) for a in paper.authors.all()}
    found, seen = [], set()
    scores = PaperScore.objects.filter(from_paper=paper, active=True).order_by('score').select_related('to_paper')
    for score in scores[:30]:
        other = score.to_paper
        for author in other.authors.all()[:3]:  # bound the number of names per related paper
            key = name_key(author.name)
            if not key or key in own or key in seen:
                continue
            seen.add(key)
            found.append({'name': author.name, 'affiliation': author.affiliation or '',
                          'arxiv_id': other.arxiv_id, 'title': ' '.join((other.title or '').split())})
            if len(found) == EXPERTS:
                return found
    return found


def write(arxiv_id, lang):
    def work(pk):
        generated.progress(pk, 'read')
        if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
            # Import only on explicit Premium generation, never on a public GET.
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
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        abstract_only = not sources.is_open_license(paper.license)
        available = paper.abstract or ''
        if sources.is_open_license(paper.license):
            available = paper_source.paper_text(arxiv_id)
            if not available or normalized(available) == normalized(paper.abstract or ''):
                full_text, source = paper_source.read_again(arxiv_id)
                if full_text:
                    paper_source.store_text(arxiv_id, full_text, source)
                    available = full_text
                else:
                    abstract_only = True
        if abstract_only or not available:
            available = paper.abstract or ''
        scope = 'abstract' if abstract_only else 'excerpt' if len(available) > MAX_CONTEXT_CHARS else 'full_text'
        source_text = available[:MAX_CONTEXT_CHARS]
        context = json.dumps({'source_documents': {'metadata': provenance(paper, scope), 'text': source_text}}, ensure_ascii=False)
        generated.progress(pk, 'experts')
        people = experts(paper)
        generated.progress(pk, 'write')
        answer = generated.chat_json(context, paid=True, max_tokens=6500,
                                     system=SOURCE_POLICY + '\n' + PROMPT.format(language=generated.language_name(lang)))
        return {
            'headline': str(answer.get('headline', '')).strip(),
            'plain_explanation': str(answer.get('plain_explanation', '')).strip(),
            'whats_new': str(answer.get('whats_new', '')).strip(),
            'hype_check': [{'signal': str(h.get('signal', '')).strip(), 'detail': str(h.get('detail', '')).strip(),
                            'level': h.get('level') if h.get('level') in LEVELS else 'caution'}
                           for h in answer.get('hype_check', []) if isinstance(h, dict) and h.get('signal')][:10],
            'verdict': str(answer.get('verdict', '')).strip(),
            'key_sentences': verified_sentences(answer.get('key_sentences', []), source_text),
            'questions_for_authors': [str(q).strip() for q in answer.get('questions_for_authors', []) if str(q).strip()][:6],
            'glossary': [{'term': str(g.get('term', '')).strip(), 'definition': str(g.get('definition', '')).strip()}
                         for g in answer.get('glossary', []) if isinstance(g, dict) and g.get('term')][:10],
            'experts': people,
            'abstract_only': abstract_only, 'schema_version': 2, 'source': provenance(paper, scope),
            'evidence_cards': evidence_cards(answer.get('evidence_cards'), source_text, abstract_only),
            'angles': records(answer.get('angles'), ['angle', 'why_now', 'reporting_step'], 3),
            'wording_checks': records(answer.get('wording_checks'), ['avoid', 'prefer', 'reason'], 3),
        }
    return work


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


def presskit_page(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if paper is None:
        announced = announced_paper(arxiv_id)
        if announced is None or canonical_id(arxiv_id) != arxiv_id:
            raise Http404
        paper = {'title': announced.title, 'authors': Author.objects.none()}
    paid = access.is_paid(request.user)
    lang = request.LANGUAGE_CODE
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=lang).first() if paid else None
    if row and row.user_id and row.user_id != request.user.pk:
        row = None
    kit = row.data() if row is not None and row.content else None
    if kit:
        quoted = {' '.join(card.get('quote', '').split()) for card in kit.get('evidence_cards', []) if card.get('verified')}
        kit['key_sentences'] = [sentence for sentence in kit.get('key_sentences', []) if ' '.join(sentence.split()) not in quoted]
    return render(request, 'summarizer/presskit.html', {
        'paper': paper, 'arxiv_id': arxiv_id, 'paid': paid, 'row': row,
        'kit': kit, 'steps': STEPS,
        'editorial': editorial_context(request.GET.get('letter'), arxiv_id, lang),
        'source': provenance(paper, 'unknown') if isinstance(paper, ArxivPaper) else None,
    })


@require_POST
def presskit_start(request, arxiv_id):
    if not access.is_paid(request.user):
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


@require_GET
def presskit_pdf(request, arxiv_id):
    if not access.is_paid(request.user):
        return HttpResponse(status=403)
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=request.LANGUAGE_CODE).exclude(content='').first()
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if row is None or paper is None or (row.user_id and row.user_id != request.user.pk):
        raise Http404
    from .presskit_pdf import document
    editorial = editorial_context(request.GET.get('letter'), arxiv_id, request.LANGUAGE_CODE)
    response = HttpResponse(document(paper, row, editorial), content_type='application/pdf')
    safe_id = re.sub(r'[^a-zA-Z0-9._-]', '-', arxiv_id)[:100]
    response['Content-Disposition'] = 'attachment; filename="SummarizePaper-press-kit-%s-%s.pdf"' % (safe_id, request.LANGUAGE_CODE)
    response['Cache-Control'] = 'private, no-store'
    response['X-Content-Type-Options'] = 'nosniff'
    return response
