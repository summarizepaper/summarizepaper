"""The structured reading of a paper shown at the top of its page: one sentence, question, method, findings, limits.

Written once per paper and language when a synthesis runs, straight from the paper text (not from the
summary, so it does not repeat it). Pages summarized before it existed keep their former layout: nothing
here is written for them on a page view.

Every finding and limitation comes with a passage of the paper. A passage is shown only when it really
is in the text we gave the model (presskit.quote_match), and each finding says whether the authors
measured it or whether it is their interpretation.
"""
import json
import logging

from . import generated, paper_source, sources
from .assistant import SOURCE_POLICY
from .models import ArxivPaper, Generated, PaperText
from .presskit import normalized, quote_match

logger = logging.getLogger(__name__)

KIND = 'insight'
MAX_CHARS = 120_000  # about 30k tokens: introduction, method and results of nearly every paper
KINDS = ('result', 'interpretation')

PROMPT = """You write the structured reading of the research paper in source_documents for researchers. Write in {language}.
Be precise and sober. Use only what the paper states; never add a number, method or claim it does not contain.
Answer with a JSON object:
{{"tldr": "one sentence, 30 words at most: what the paper shows",
  "question": "1 or 2 sentences: the problem or question it addresses",
  "method": "2 or 3 sentences: data, approach, models or experiments, with the key numbers the paper gives",
  "findings": [{{"text": "one finding, one sentence", "kind": "result|interpretation",
                "quote": "an exact passage of the paper supporting it"}}],
  "limitations": [{{"text": "one limitation, one sentence", "quote": "an exact passage, or an empty string"}}],
  "significance": "1 or 2 sentences: what it changes for the field, as the authors argue it"}}
Give 3 to 5 findings and 1 to 3 limitations. "result" is what the authors measured, computed or proved;
"interpretation" is what they conclude, suggest or speculate from it. Limitations are those the paper states;
when it states none, give the most obvious one of its scope and an empty quote.
Copy each quote exactly, word for word, in the ORIGINAL language of the paper: 1 or 2 sentences, no ellipsis.
Write the other fields in {language}, in plain text: no Markdown, no LaTeX commands."""


def clean(value, limit):
    return ' '.join(str(value or '').split())[:limit]


def evidence(quote, source):
    match = quote_match(quote, source)
    return match.group() if match else ''


def structured(answer, source, abstract_only):
    """The model's answer as shown on the page: fields trimmed, quotes kept only when found in the source."""
    findings = []
    for item in answer.get('findings') or []:
        if isinstance(item, dict) and clean(item.get('text'), 600):
            findings.append({'text': clean(item['text'], 600),
                             'kind': item.get('kind') if item.get('kind') in KINDS else 'interpretation',
                             'quote': evidence(item.get('quote'), source)})
    limitations = [{'text': clean(item['text'], 600), 'quote': evidence(item.get('quote'), source)}
                   for item in answer.get('limitations') or [] if isinstance(item, dict) and clean(item.get('text'), 600)]
    return {'tldr': clean(answer.get('tldr'), 400), 'question': clean(answer.get('question'), 800),
            'method': clean(answer.get('method'), 1200), 'findings': findings[:5], 'limitations': limitations[:3],
            'significance': clean(answer.get('significance'), 800), 'abstract_only': abstract_only, 'schema_version': 1}


def source_text(paper):
    """(text given to the model, abstract_only)."""
    stored = PaperText.objects.filter(arxiv_id=paper.arxiv_id).only('source').first()
    if sources.is_open_license(paper.license) and not (stored and stored.source == 'abstract'):
        text = paper_source.paper_text(paper.arxiv_id)
        if text and normalized(text) != normalized(paper.abstract or ''):
            return text[:MAX_CHARS], False
    return (paper.abstract or '').strip(), True


def write(arxiv_id, lang, paid):
    def work(pk):
        paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
        text, abstract_only = source_text(paper)
        if not text:
            raise generated.SourceUnavailable('No abstract or permitted full text is available.')
        authors = [a.name for a in paper.authors.all()[:20]]
        generated.release_connection()
        context = json.dumps({'source_documents': {
            'metadata': {'title': paper.title, 'authors': authors, 'scope': 'abstract' if abstract_only else 'full_text'},
            'text': text}}, ensure_ascii=False)
        # the visitor waits for it before the synthesis is ready: a stuck call must not hold the page
        answer = generated.chat_json(context, paid=paid, max_tokens=2500, timeout=90,
                                     system=SOURCE_POLICY + '\n' + PROMPT.format(language=generated.language_name(lang)))
        return structured(answer, text, abstract_only)
    return work


def create(arxiv_id, lang, paid):
    """Write the reading of a paper whose synthesis just ran (in a worker thread). Never fatal for the synthesis."""
    try:
        Generated.objects.filter(kind=KIND, key=arxiv_id, lang=lang).delete()  # a new synthesis gets a new reading
        generated.run_now(KIND, arxiv_id, lang, write(arxiv_id, lang, paid))
    except Exception:
        logger.exception('Insight of %s (%s) failed', arxiv_id, lang)


def for_page(arxiv_id, lang):
    """The reading shown on the page, or None (older pages, or a failed generation)."""
    row = Generated.objects.filter(kind=KIND, key=arxiv_id, lang=lang, status='done').first()
    data = row.data() if row else None
    return data if isinstance(data, dict) and data.get('tldr') and data.get('findings') else None


def unavailable(arxiv_id, lang):
    return Generated.objects.filter(kind=KIND, key=arxiv_id, lang=lang, status='no_source').exists()
