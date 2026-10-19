"""Press kit of a paper, for journalists (Premium).

Written once per paper and language from the full text (or only the abstract
when the license does not let us read more). Two things are checked rather
than trusted to the model:

- the key sentences must appear word for word in the text we gave it, so a
  journalist never quotes a sentence the paper does not contain;
- the independent researchers come from our own data (authors of the closest
  papers who did not sign this one), the model only reads them.
"""
import re

from django.http import Http404
from django.shortcuts import render
from django.views.decorators.http import require_POST
from django.http import JsonResponse

from . import access, generated, similar
from .assistant import build_context
from .digest import name_key
from .models import ArxivPaper, Generated, PaperScore

EXPERTS = 6
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
For hype_check, look at least at: preprint not yet peer reviewed (say so if no journal is mentioned), size of the
sample or data set, simulation versus observation or experiment, statistical significance and error bars, whether the
claims in the abstract go beyond the results, conflicts of interest or funding if stated, and limitations the authors
themselves give. Use "ok" when a point is reassuring."""


def normalized(text):
    return re.sub(r'\W+', ' ', text.lower()).strip()


def verified_sentences(sentences, source):
    """Only the sentences that really are in the paper text."""
    haystack = normalized(source)
    kept = []
    for sentence in sentences:
        sentence = ' '.join(str(sentence).split())
        if len(sentence) > 30 and normalized(sentence) in haystack and sentence not in kept:
            kept.append(sentence)
    return kept[:5]


def experts(paper):
    """Authors of the closest papers we know who are not authors of this one: [{name, arxiv_id, title}]."""
    if not PaperScore.objects.filter(from_paper=paper).exists():
        try:
            similar.find_and_store(paper.arxiv_id)
        except Exception:
            return []
    own = {name_key(a.name) for a in paper.authors.all()}
    found, seen = [], set()
    scores = PaperScore.objects.filter(from_paper=paper).order_by('score').select_related('to_paper')
    for score in scores[:30]:
        other = score.to_paper
        for author in other.authors.all()[:3]:  # the first authors did most of the work
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
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        context = build_context([arxiv_id])
        people = experts(paper)
        generated.release_connection()
        answer = generated.chat_json(PROMPT.format(language=generated.language_name(lang)), paid=True,
                                     max_tokens=4000, system=context)
        return {
            'headline': str(answer.get('headline', '')).strip(),
            'plain_explanation': str(answer.get('plain_explanation', '')).strip(),
            'whats_new': str(answer.get('whats_new', '')).strip(),
            'hype_check': [{'signal': str(h.get('signal', '')).strip(), 'detail': str(h.get('detail', '')).strip(),
                            'level': h.get('level') if h.get('level') in LEVELS else 'caution'}
                           for h in answer.get('hype_check', []) if isinstance(h, dict) and h.get('signal')][:10],
            'verdict': str(answer.get('verdict', '')).strip(),
            'key_sentences': verified_sentences(answer.get('key_sentences', []), context),
            'questions_for_authors': [str(q).strip() for q in answer.get('questions_for_authors', []) if str(q).strip()][:6],
            'glossary': [{'term': str(g.get('term', '')).strip(), 'definition': str(g.get('definition', '')).strip()}
                         for g in answer.get('glossary', []) if isinstance(g, dict) and g.get('term')][:10],
            'experts': people,
            'abstract_only': 'Only the title, authors and abstract are available' in context,
        }
    return work


def presskit_page(request, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).prefetch_related('authors').first()
    if paper is None:
        raise Http404
    paid = access.is_paid(request.user)
    lang = request.LANGUAGE_CODE
    row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=lang).first() if paid else None
    return render(request, 'summarizer/presskit.html', {
        'paper': paper, 'arxiv_id': arxiv_id, 'paid': paid, 'row': row,
        'kit': row.data() if row is not None and row.status == 'done' else None,
    })


@require_POST
def presskit_start(request, arxiv_id):
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    if not ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        raise Http404
    lang = request.LANGUAGE_CODE
    row = generated.start('presskit', arxiv_id, lang, write(arxiv_id, lang), force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})
