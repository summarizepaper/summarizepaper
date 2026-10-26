"""The sources of an assistant answer: each claim with the passage of the paper that supports it.

After the answer, the model lists its claims with a quote of the paper for each one (assistant.check_answer).
A quote is accepted only when it is word for word in the text the model was given, never in our summary of
the paper: exactly (presskit.quote_match), or up to the typography a PDF extraction changes (find).
Each claim also names the end of the sentence of the answer that makes it, where the page puts its [n]. We then say where it is: the section, the paragraph when the text keeps
the paper's paragraphs (arXiv HTML), the figure or table whose caption it is, and a link that opens the
arXiv HTML page on that passage. Claims the model attributes to general knowledge, or that it admits the
paper does not support, are shown as such instead of being dropped.
"""
import difflib
import re
from decimal import ROUND_HALF_UP, Decimal
from urllib.parse import quote as url_quote

from . import sources
from .paper_source import ARXIV_HTML, arxiv_ref
from .presskit import normalized, quote_match

MAX_CLAIMS = 8  # for one paper; each further paper adds CLAIMS_PER_PAPER, up to MAX_CLAIMS_ALL
CLAIMS_PER_PAPER = 3
MAX_CLAIMS_ALL = 20
BASES = ('paper', 'figure_image', 'general', 'unsupported')

PROMPT = """New task: check your last answer against source_documents. Answer with a JSON object:
{{"claims": [{{"claim": "...", "anchor": "...", "paper": "...", "basis": "paper|figure_image|general|unsupported", "quote": "...", "figure": "..."}}]}}
List the factual statements your answer makes, in the order it makes them, {max_claims} at most; merge statements
that rest on the same passage. Leave out greetings, advice, questions to the user and the scope reminder.
When the answer makes no factual statement, answer {{"claims": []}}.
- claim: the statement as the answer makes it, one sentence, in {language}.
- anchor: the last 3 to 8 words of the sentence of your answer that makes the statement, copied exactly from your
  answer, without Markdown signs, equations or the final punctuation.
- paper: the identifier line of the paper it is about, exactly as in its <paper> block (e.g. "arXiv: 2406.08919v3").
- basis "paper": the paper text states it. quote: the passage that supports it, copied exactly, word for word,
  in the paper's original language: 1 to 3 consecutive sentences, no ellipsis, equations written as in source_documents.
- basis "figure_image": you read it on a figure image attached to the conversation. quote: the caption sentence, or "".
- basis "general": it comes from general knowledge, not from the paper. quote: "".
- basis "unsupported": no passage of the paper supports it as stated. quote: "". Never stretch a passage to fit.
- figure: the figure, table or equation the statement relies on, numbered as in the paper (e.g. "Figure 3"), or "".
Write nothing but the JSON object."""

CAPTION = re.compile(r'^\s*((?:Figure|Fig\.|Table|Tab\.)\s*\d+[a-z]?)\s*[:.|]', re.I)
FIGURE = re.compile(r'^(Figure|Fig\.?|Table|Tab\.?|Equation|Eq\.?)\s*\(?(\d+[a-z]?)\)?$', re.I)
# 170 000, 170,000 and 170000 are one number; 0,5 and 0.5 too (French answers, English papers)
NUMBER = re.compile(r'(?<![\w.,])\d{1,3}(?:[ \u00a0\u202f,]\d{3})+(?![\d])(?:\.\d+)?|(?<![\w.,])\d+(?:[.,]\d+)?')
REFERENCE = re.compile(r'\b(?:Fig(?:ure)?s?|Tab(?:le)?s?|Eq(?:uation)?s?|Sec(?:tion)?s?|§|Appendix|Annexe)\.?\s*\(?\d+(?:\.\d+)*[a-z]?\)?', re.I)


def numbers(text):
    """The values a text states, written the same way whatever the language; figure, section... numbers left out."""
    values = set()
    for number in NUMBER.findall(REFERENCE.sub(' ', text)):
        if re.search(r'[ \u00a0\u202f]|,\d{3}(?!\d)', number) and not re.search(r',\d{1,2}$|,\d{4,}', number):
            number = re.sub(r'[ \u00a0\u202f,]', '', number)  # thousands separators
        values.add(number.replace(',', '.'))
    return values


def with_roundings(values):
    """The values and what they round to: an answer may say 40.7% where the paper says 40.73%."""
    rounded = set(values)
    for value in values:
        if '.' in value:
            for places in range(len(value.split('.')[1])):
                rounded.add(str(Decimal(value).quantize(Decimal(1).scaleb(-places), ROUND_HALF_UP)))
    return rounded


def max_claims(papers):
    """How many claims of an answer are checked: an answer about more papers makes more of them."""
    return min(MAX_CLAIMS + CLAIMS_PER_PAPER * (papers - 1), MAX_CLAIMS_ALL)


def prompt(language, claims=MAX_CLAIMS):
    return PROMPT.format(language=language, max_claims=claims)


def clean(value, limit):
    return ' '.join(str(value or '').split())[:limit]


# what a PDF extraction or a model copying a passage changes without changing a word
EQUIVALENT = {c: "['’‘ʼ]" for c in "'’‘ʼ"} | {c: '["“”«»]' for c in '"“”«»'} | {c: r'[-‐‑–—−]\s*' for c in '-‐‑–—−'}
LIGATURES = (('ffi', '(?:ffi|ﬃ)'), ('ffl', '(?:ffl|ﬄ)'), ('ff', '(?:ff|ﬀ)'), ('fi', '(?:fi|ﬁ)'), ('fl', '(?:fl|ﬂ)'))
LINE_BREAK = r'(?:-\s*)?'  # a word cut at the end of a PDF line: "galax- ies"
MIN_TOLERANT = 15  # characters: shorter passages are too common to show where a claim comes from


def word_pattern(word):
    units, i = [], 0
    while i < len(word):
        for letters, pattern in LIGATURES:
            if word.startswith(letters, i):
                units.append(pattern)
                i += len(letters)
                break
        else:
            units.append(EQUIVALENT.get(word[i]) or re.escape(word[i]))
            i += 1
    return LINE_BREAK.join(units)


def tolerant_match(quote, text):
    words = quote.split()
    if len(words) < 3 or not MIN_TOLERANT <= len(quote) <= 1800:
        return None
    return re.search(r'\s+'.join(word_pattern(w) for w in words), text, re.I)


class Gapped:
    """A passage found with something the PDF put in its middle (page number, caption, link): the text from
    its first to its last words, read like a regex match."""
    gapped = True

    def __init__(self, text, start, end):
        self.text, self.span_start, self.span_end = text, start, end

    def start(self):
        return self.span_start

    def end(self):
        return self.span_end

    def group(self):
        return self.text[self.span_start:self.span_end]


def words_of(text):
    return normalized(re.sub(r'(\w)[-‐‑]\s+(\w)', r'\1\2', text)).split()  # "posi- tion" is one word


def end_pattern(words, reverse=False):
    """The first 3 words (or more, until 12 characters) of words, as a pattern; words are reversed for an end."""
    part = []
    for word in words:
        part.append(word)
        if len(part) >= 3 and len(' '.join(part)) >= 12:
            break
    return re.compile(r'\s+'.join(word_pattern(w) for w in (part[::-1] if reverse else part)), re.I)


def gapped_match(quote, text):
    """The passage when its first and last words are close together in the text, 90% of its words are there
    in order and every number it quotes is too: never a passage whose numbers the model changed.
    The ends may start a few words inside, when the interruption falls among the first or last words."""
    words = quote.split()
    if len(words) < 10:
        return None
    heads = [end_pattern(words[i:]) for i in range(4)]
    tails = [end_pattern(words[:len(words) - i][::-1], reverse=True) for i in range(4)]
    digits = lambda value: set(re.findall(r'\d+(?:\.\d+)?', value))
    quoted = words_of(quote)
    for head_pattern in heads:
        for head in head_pattern.finditer(text):
            for tail_pattern in tails:
                tail = tail_pattern.search(text, head.end(), head.start() + int(len(quote) * 1.6) + 200)
                if not tail:
                    continue
                passage = text[head.start():tail.end()]
                blocks = difflib.SequenceMatcher(None, quoted, words_of(passage), autojunk=False).get_matching_blocks()
                if sum(b.size for b in blocks) >= 0.9 * len(quoted) and digits(quote) <= digits(passage):
                    return Gapped(text, head.start(), tail.end())
    return None


def find(quote, text):
    """The passage as it is in the text, or None. A quote copied from the JSON of source_documents may keep
    its doubled backslashes."""
    candidates = [quote] + ([quote.replace('\\\\', '\\')] if '\\\\' in quote else [])
    for candidate in candidates:
        match = quote_match(candidate, text) or tolerant_match(candidate, text)
        if match:
            return match
    for candidate in candidates:
        match = gapped_match(candidate, text)
        if match:
            return match
    return None


def plain(text):
    """Answer text as the reader sees it: no Markdown emphasis or code signs, single spaces."""
    return ' '.join(re.sub(r'[*_`#>]', '', str(text or '')).split())


def anchor_in(value, answer_text):
    """The words that end the claim's sentence when they really are in the answer, else ''."""
    anchor = plain(value).rstrip('.,;:!?…')[:200]
    return anchor if len(anchor.split()) >= 2 and '$' not in anchor and anchor in plain(answer_text) else ''


def paper_of(name, supplied):
    """The supplied paper the model named, or None."""
    name = str(name or '').lower()
    for paper in supplied:
        if paper['label'].lower() in name or paper['arxiv_id'].lower() in name or arxiv_ref(paper['arxiv_id']).lower() in name:
            return paper
    return supplied[0] if len(supplied) == 1 else None


def figure_named(name, text):
    """'Figure 3' when the model names a figure, table or equation that the paper does number."""
    match = FIGURE.match(clean(name, 40))
    if not match:
        return ''
    first = match.group(1).lower()
    kind = 'Figure' if first.startswith('fig') else 'Table' if first.startswith('tab') else 'Equation'
    word = {'Figure': r'Fig(?:ure|\.)?', 'Table': r'Tab(?:le|\.)?', 'Equation': r'(?:Equation|Eq\.?)'}[kind]
    if re.search(r'\b%s\s*\(?%s\b' % (word, re.escape(match.group(2))), text, re.I):
        return '%s %s' % (kind, match.group(2))
    return ''


def location(match, paper):
    """Section, paragraph within it (when the text keeps the paper's paragraphs) and figure caption of a passage."""
    text = paper['text']
    if paper['source'] == 'abstract':
        return {'section': 'Abstract', 'paragraph': None, 'caption': ''}
    before = text[:match.start()]
    headings = list(re.finditer(r'^#{1,6}\s+(.+)$', before, re.M))
    section_start = headings[-1].end() if headings else 0
    paragraph_start = before.rfind('\n\n') + 2 if '\n\n' in before else 0
    caption = CAPTION.match(text[paragraph_start:match.end()])
    paragraph = None
    if paper['source'] == 'html' and not caption:
        paragraphs = [p for p in text[section_start:paragraph_start].split('\n\n') if p.strip()]
        paragraph = len(paragraphs) + 1
    return {'section': headings[-1][1].strip()[:160] if headings else '', 'paragraph': paragraph,
            'caption': caption.group(1) if caption else ''}


def fragment(words):
    return url_quote(' '.join(words), safe='').replace('-', '%2D')


def passage_link(passage, paper):
    """The arXiv HTML page opened on the passage (a text fragment), or ''.

    Equations are shown as LaTeX in our text but rendered on the page, so the link stops before the first one.
    """
    if paper['source'] != 'html' or sources.is_external(paper['arxiv_id']):
        return ''
    url = ARXIV_HTML % arxiv_ref(paper['arxiv_id'])
    words = passage.split('$', 1)[0].split()
    if '$' not in passage and len(words) > 12:
        return '%s#:~:text=%s,%s' % (url, fragment(words[:6]), fragment(words[-6:]))
    if len(words) < 4:
        return url
    return '%s#:~:text=%s' % (url, fragment(words[:10]))


def verify(answer, supplied, answer_text='', claims_max=MAX_CLAIMS):
    """The claims of the model's answer as shown under the answer.

    status: 'verified' (passage found in the paper), 'not_found' (the model gave a passage we cannot find),
    'visual' (read on an attached figure image), 'general' (general knowledge), 'unsupported'.
    numbers_absent: values of the claim the paper never states. A claim often takes a number from another
    passage or a table than the one it quotes: only a number found nowhere in the paper says something is wrong.
    """
    claims, stated = [], {}

    def stated_in(papers):
        values = set()
        for paper in papers:
            if paper['arxiv_id'] not in stated:
                stated[paper['arxiv_id']] = with_roundings(numbers(paper['text']))
            values |= stated[paper['arxiv_id']]
        return values

    for item in answer.get('claims') or [] if isinstance(answer, dict) else []:
        if not isinstance(item, dict) or not clean(item.get('claim'), 600):
            continue
        basis = item.get('basis') if item.get('basis') in BASES else 'unsupported'
        paper = paper_of(item.get('paper'), supplied)
        quote = clean(item.get('quote'), 1800)
        claim = {'claim': clean(item['claim'], 600), 'anchor': anchor_in(item.get('anchor'), answer_text),
                 'status': basis, 'paper': paper['label'] if paper else '',
                 'quote': '', 'section': '', 'paragraph': None, 'figure': '', 'link': '', 'gapped': False,
                 'numbers_absent': []}
        if basis in ('paper', 'figure_image'):
            found = None
            # the paper the model named first, then the others (a multi-paper answer can name the wrong one)
            for candidate in ([paper] if paper else []) + [p for p in supplied if p is not paper]:
                match = find(quote, candidate['text']) if quote else None
                if match:
                    found, paper = match, candidate
                    break
            if found:
                place = location(found, paper)
                claim.update({'status': 'verified', 'paper': paper['label'], 'quote': clean(found.group(), 1800),
                              'section': place['section'], 'paragraph': place['paragraph'],
                              'figure': place['caption'], 'link': passage_link(found.group(), paper),
                              'gapped': getattr(found, 'gapped', False)})
            else:
                claim['status'] = 'visual' if basis == 'figure_image' else 'not_found'
        if paper and not claim['figure']:
            claim['figure'] = figure_named(item.get('figure'), paper['text'])
        if claim['status'] != 'general':
            claim['numbers_absent'] = sorted(numbers(claim['claim']) - stated_in([paper] if paper else supplied))
        claims.append(claim)
        if len(claims) == claims_max:
            break
    return claims
