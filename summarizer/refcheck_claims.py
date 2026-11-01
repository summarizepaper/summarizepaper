"""What each citation of a manuscript is used for, checked against the cited papers (see refcheck.py).

1. The LaTeX is cut into sentences, each with its file and line. A sentence ends at . ! ? before a capital (not
   after e.g., et al., Fig. or an initial), at a blank line, a heading, an \\item or a caption. Every citation
   command counts (\\citep, \\Citet, \\citeyear, \\parencite, \\cites{a}{b}...), \\nocite excepted. Tables are
   left out: their cells are no sentences.
2. A long manuscript has more citing sentences than are read: those that attribute a result, a value or a
   method to one or two papers come first, lists of background references last. The report says how many
   sentences were read out of how many.
3. Each sentence is judged once, with ALL the papers it cites and the sentence before it: a statement cited to
   two papers is supported when one of them says it. A paper without an abstract never makes a sentence flagged.
4. The sentences flagged on the abstracts are read again against the full text of the papers concerned, when
   their rights let us read it (tdm.py): the model must copy the passage that supports or contradicts the
   sentence, and a passage the paper does not contain word for word is never shown (evidence.find). The values
   a sentence attributes to a paper whose full text we read and that the paper never states are reported too.
"""
import logging
import re
from bisect import bisect_right

from . import evidence, generated, presskit, sources, tdm

logger = logging.getLogger(__name__)

MAX_CLAIMS = 60  # sentences read next to the abstracts, CLAIMS_PER_CALL per model call, in parallel
CLAIMS_PER_CALL = 12
PAPERS_PER_CLAIM = 5
ABSTRACT_CHARS = 1800
FULL_TEXTS = 12  # cited papers read in full, for the flagged sentences
DEEP_PER_CALL = 4
PASSAGES = 6
PASSAGE_CHARS = 1200

CITE_RE = re.compile(r'\\(?!nocite\b)([A-Za-z]*[cC]ite[a-zA-Z]*)\*?')
ARGS_RE = re.compile(r'\s*(?:\[[^\]]*\]\s*){0,2}\{([^}]*)\}')
NOCITE_RE = re.compile(r'\\nocite\s*\{([^}]*)\}')
TABLE_RE = re.compile(r'\\begin\{(tabular\*?|tabularx|longtable|tabulary|array)\}.*?\\end\{\1\}', re.S)
BREAK_RE = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z\\$(\[])'
    r'|\n[ \t]*\n\s*'
    r'|\\(?:part|chapter|(?:sub)*section|(?:sub)?paragraph)\*?(?:\[[^\]]*\])?\{[^}]*\}'
    r'|\\item\b|\\caption(?=\s*[\[{])|\\(?:begin|end)\{[^}]*\}(?:\[[^\]]*\])?')
ABBREVIATION = re.compile(r'(?:\b(?:e\.g|i\.e|et al|cf|Figs?|Eqs?|Sects?|Secs?|Refs?|Tabs?|vs|approx|resp|Ch|No|Vol|'
                          r'pp|Dr|Prof|Mr|Ms|St|Jr)\.|(?<![A-Za-z])[A-Z]\.)$')
RESULT_WORDS = re.compile(
    r'\b(?:show[sn]?|showed|find|finds|found|demonstrat\w*|report\w*|measur\w*|observ\w*|detect\w*|estimat\w*|'
    r'predict\w*|prove[dns]?|reveal\w*|establish\w*|conclud\w*|confirm\w*|obtain\w*|deriv\w*|indicat\w*|'
    r'suggest\w*|argu\w*|claim\w*|propos\w*|introduc\w*|achiev\w*|outperform\w*|improv\w*|first)\b', re.I)
BACKGROUND = re.compile(r'\b(?:e\.g\.|see|reviews?|references therein|among others|for instance|such as)\b', re.I)
VALUE = re.compile(r'\d+(?:[.,]\d+)?\s*(?:\\?%|\\sigma|σ|\\pm|±)|\d+\.\d+')
WORD = re.compile(r'[A-Za-z][A-Za-z\-]{3,}')
STOP = frozenset('this that with from have were which their these those also than then there been into such '
                 'more most other only over some very when where while about after before between both each '
                 'however using used based shown show showed found paper work study studies recent results'.split())
VERDICTS = ('supported', 'unclear', 'contradicted')
USES = ('background', 'method', 'data', 'result', 'comparison')


# --- reading the LaTeX ---------------------------------------------------------

def citations(text):
    """[(start, end, [keys])] of the citation commands of a LaTeX text."""
    found = []
    for match in CITE_RE.finditer(text):
        name, i, keys = match.group(1), match.end(), []
        while True:
            args = ARGS_RE.match(text, i)
            if not args:
                break
            keys += [k.strip() for k in args.group(1).split(',') if k.strip()]
            i = args.end()
            if not name.lower().endswith('cites'):
                break
        if keys:
            found.append((match.start(), i, list(dict.fromkeys(keys))))
    return found


def cited_keys(texts):
    """(keys cited anywhere, whether \\nocite{*} cites the whole .bib)."""
    keys, everything = [], False
    for _name, tex in texts:
        tex = re.sub(r'(?<!\\)%.*', '', tex)
        keys += [k for _s, _e, found in citations(tex) for k in found]
        for group in NOCITE_RE.findall(tex):
            everything |= '*' in group
            keys += [k.strip() for k in group.split(',') if k.strip() and k.strip() != '*']
    return list(dict.fromkeys(keys)), everything


def body_range(text):
    start = text.find('\\begin{document}')
    start = start + len('\\begin{document}') if start >= 0 else 0
    end = text.find('\\end{document}', start)
    end = end if end >= 0 else len(text)
    bibliography = re.search(r'\\begin\{thebibliography\}|\\bibliography\{|\\printbibliography', text[start:end])
    return start, start + bibliography.start() if bibliography else end


def sentences(tex):
    """[(line, LaTeX text)] of the sentences of a LaTeX file, comments and preamble left out."""
    text = '\n'.join(re.sub(r'(?<!\\)%.*', '', line) for line in tex.split('\n'))
    # a table is no sentence: its cells read as noise (its caption stays); lines keep their numbers
    text = TABLE_RE.sub(lambda m: re.sub(r'[^\n]', ' ', m.group(0)), text)
    start, end = body_range(text)
    newlines = [i for i, c in enumerate(text) if c == '\n']
    pieces, i = [], start
    for found in BREAK_RE.finditer(text, start, end):
        pieces.append([i, found.start(), found.group(0)])
        i = found.end()
    pieces.append([i, end, ''])
    merged = []
    for piece in pieces:
        if merged and not merged[-1][2].strip() and '\n\n' not in re.sub(r'[ \t]', '', merged[-1][2]) and \
                ABBREVIATION.search(text[merged[-1][0]:merged[-1][1]].rstrip()):
            merged[-1][1:] = piece[1:]
        else:
            merged.append(piece)
    found = []
    for a, b, _separator in merged:
        chunk = text[a:b]
        if chunk.strip():
            a += len(chunk) - len(chunk.lstrip())
            found.append((bisect_right(newlines, a - 1) + 1, chunk.strip()))
    return found


def readable(latex):
    """A LaTeX sentence as a reader sees it, each citation shown as [key, key]; mathematics kept for MathJax."""
    parts, last = [], 0
    for start, end, keys in citations(latex):
        parts += [latex[last:start], '[%s]' % ', '.join(keys)]
        last = end
    parts.append(latex[last:])
    text = ''.join(parts)
    pieces = re.split(r'(\$\$?[^$]+\$\$?)', text)
    for i in range(0, len(pieces), 2):
        piece = re.sub(r'\\(?:label|nocite)\s*\{[^}]*\}', '', pieces[i])
        piece = re.sub(r'\\(?:ref|eqref|autoref|cref|Cref|pageref)\s*\{[^}]*\}', '…', piece)
        piece = re.sub(r'(?<!\\)~', ' ', piece)
        piece = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^\]]*\])?', '', piece)
        pieces[i] = re.sub(r'\\([%&#_$])', r'\1', piece.replace('{', '').replace('}', ''))
    return ' '.join(''.join(pieces).split())


def citing_sentences(texts):
    """[{'file', 'line', 'sentence', 'previous', 'keys', 'order'}] of the sentences with citations, in reading order."""
    found = []
    for name, tex in texts:
        previous = ''
        for line, latex in sentences(tex):
            keys = list(dict.fromkeys(k for _s, _e, ks in citations(latex) for k in ks))
            sentence = readable(latex)
            if keys and 30 < len(sentence) < 1500:
                found.append({'file': name, 'line': line, 'sentence': sentence, 'previous': previous[-600:],
                              'keys': keys, 'order': len(found)})
            previous = sentence
    return found


def priority(claim):
    """Sentences that attribute a result, a value or a method to one or two papers first; background lists last."""
    sentence, keys = claim['sentence'], claim['keys']
    return (3 * bool(RESULT_WORDS.search(sentence)) + 2 * bool(VALUE.search(sentence)) + (len(keys) <= 2)
            - 2 * (len(keys) > 4) - 2 * bool(BACKGROUND.search(sentence)))


def chosen(claims, limit=MAX_CLAIMS):
    """The claims read, at most limit of them, by priority, given back in reading order."""
    picked = sorted(claims, key=lambda c: (-priority(c), c['order']))[:limit]
    return sorted(picked, key=lambda c: c['order'])


# --- judging on the abstracts ----------------------------------------------------

def listing(claim, papers, offset):
    lines = ['<claim id="%d">' % offset, 'Sentence of the draft: %s' % claim['sentence'][:1200]]
    if claim['previous']:
        lines.append('Sentence before it: %s' % claim['previous'][:500])
    for key in claim['keys'][:PAPERS_PER_CLAIM]:
        paper = papers.get(key)
        if paper and paper.get('abstract'):
            lines.append('Cited paper [%s]: %s\nIts abstract: %s' % (key, paper['title'], paper['abstract'][:ABSTRACT_CHARS]))
        else:
            lines.append('Cited paper [%s]: %s (abstract not available)' % (key, (paper or {}).get('title', '')))
    return '\n'.join(lines + ['</claim>'])


def judge(batch, papers, start, lang):
    answer = generated.chat_json(
        'Each claim above is a sentence of a research draft, its citations shown as [key], the sentence before it, '
        'and the abstracts of the papers it cites. Judge each sentence as a whole: does what it attributes to the '
        'cited papers agree with them? "supported" when it does, or when the papers are cited as background on a '
        'topic they are about (a list of papers cited for one general statement only needs each to be about it); '
        '"unclear" when the abstracts do not say it (the full text may); "contradicted" when an abstract says '
        'otherwise or the sentence clearly misrepresents a paper (another result, sign, object or conclusion). What '
        'the sentence says in its own name, without attributing it to the papers, is not judged. For "unclear" and '
        '"contradicted", list in "keys" the cited papers concerned and explain in %s in one sentence. A paper given '
        'without its abstract cannot be judged: never flag a sentence for what such a paper may say. Say also how the '
        'sentence uses its citations: "background", "method", "data", "result" or "comparison". Answer with a JSON '
        'object {"items": [{"id": "claim id", "verdict": "supported|unclear|contradicted", "use": "...", "keys": '
        '["key"], "reason": "..."}]}.' % generated.language_name(lang),
        paid=True, max_tokens=4000, system='\n\n'.join(listing(c, papers, start + i) for i, c in enumerate(batch)))
    return answer.get('items') if isinstance(answer.get('items'), list) else []


def check_claims(claims, papers, lang):
    """The claims the abstracts do not clearly support, each with its verdict, the papers concerned and a reason."""
    from .missed import at_once
    batches = [(start, claims[start:start + CLAIMS_PER_CALL]) for start in range(0, len(claims), CLAIMS_PER_CALL)]
    answers = at_once([lambda start=start, batch=batch: judge(batch, papers, start, lang) for start, batch in batches])
    flagged = []
    for items in answers:
        for item in items:
            if not isinstance(item, dict) or item.get('verdict') not in ('unclear', 'contradicted'):
                continue
            try:
                claim = claims[int(item.get('id'))]
            except (TypeError, ValueError, IndexError):
                continue
            keys = [k for k in (item.get('keys') or []) if k in claim['keys']] or claim['keys']
            keys = [k for k in keys if (papers.get(k) or {}).get('abstract')]  # nothing is concluded from no abstract
            if not keys:
                continue
            flagged.append(dict(claim, verdict=item['verdict'], use=item.get('use') if item.get('use') in USES else '',
                                concerned=keys, reason=str(item.get('reason', '')).strip()))
    return flagged


# --- the full text -------------------------------------------------------------

def full_text(paper):
    """{'text', 'source', 'arxiv_id'} of a cited paper when its rights let us read it, else None."""
    try:
        if paper.get('arxiv'):
            text, source = sources.arxiv_text(paper['arxiv'])
            if len(text) > sources.MIN_TEXT:
                return {'text': text, 'source': source, 'arxiv_id': paper['arxiv']}
        if paper.get('doi'):
            key = sources.doi_key(paper['doi'])
            meta = sources.metadata(key)
            if meta and tdm.may_read(meta.get('license', ''), key):
                text, source = sources.full_text(key)
                if source != 'abstract' and len(text) > sources.MIN_TEXT:
                    return {'text': text, 'source': source, 'arxiv_id': key}
    except Exception:
        logger.info('Full text of %s could not be read', paper.get('arxiv') or paper.get('doi'), exc_info=True)
    return None


def words(text):
    return {w.lower() for w in WORD.findall(text)} - STOP


def passages(claim, text):
    """The passages of a paper closest to a sentence (shared words), in the paper's order."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 80]
    if len(paragraphs) < 5:
        paragraphs = [text[i:i + PASSAGE_CHARS] for i in range(0, len(text), PASSAGE_CHARS)]
    wanted = words(claim['sentence'] + ' ' + claim['previous'][-200:])
    scored = sorted(range(len(paragraphs)), key=lambda i: -len(wanted & words(paragraphs[i])))[:PASSAGES]
    return [paragraphs[i][:PASSAGE_CHARS * 2] for i in sorted(scored)]


def significant_values(sentence):
    """The values a sentence states that a paper should state too: decimals and numbers with a unit sign, no years."""
    values = evidence.numbers(re.sub(r'\[[^\]]*\]', ' ', sentence))
    return {v for v in values if '.' in v or (v.isdigit() and len(v) >= 2 and not re.fullmatch(r'(1[89]|20)\d\d', v))}


def deepen(flagged, papers, lang):
    """Read again the flagged claims against the full text of the papers concerned, when it can be read."""
    from .missed import at_once
    wanted = list(dict.fromkeys(k for c in sorted(flagged, key=lambda c: c['verdict'] != 'contradicted')
                                for k in c['concerned'] if k in papers))[:FULL_TEXTS]
    texts = dict(zip(wanted, at_once([lambda key=key: full_text(papers[key]) for key in wanted])))
    readable_claims = []
    for claim in flagged:
        read = [k for k in claim['concerned'] if texts.get(k)]
        if read:
            readable_claims.append((claim, read))
            claim['full_text'] = read
            values = significant_values(claim['sentence'])
            if values and all(texts.get(k) for k in claim['keys']):
                stated = set()
                for key in claim['keys']:
                    stated |= evidence.with_roundings(evidence.numbers(texts[key]['text']))
                claim['numbers_absent'] = sorted(values - stated)

    def ask(batch):
        blocks = []
        for i, (claim, read) in enumerate(batch):
            block = ['<claim id="%d">' % i, 'Sentence of the draft: %s' % claim['sentence'][:1200]]
            for key in read:
                block.append('Passages of the cited paper [%s] %s:\n%s' % (
                    key, papers[key]['title'], '\n[...]\n'.join(passages(claim, texts[key]['text']))))
            blocks.append('\n'.join(block + ['</claim>']))
        answer = generated.chat_json(
            'Each claim above is a sentence of a research draft, its citations shown as [key], and passages of the full '
            'text of the cited papers. Say whether the papers support what the sentence attributes to them: '
            '"supported", "unclear" (the passages do not say it) or "contradicted" (a passage says otherwise). For '
            '"supported" and "contradicted", give in "paper" the key of the paper and copy in "quote" the passage that '
            'shows it, word for word from the passages, one or two sentences. Explain in %s in one sentence. Answer '
            'with a JSON object {"items": [{"id": "claim id", "verdict": "supported|unclear|contradicted", "paper": '
            '"key", "quote": "...", "reason": "..."}]}.' % generated.language_name(lang),
            paid=True, max_tokens=3000, system='\n\n'.join(blocks))
        return answer.get('items') if isinstance(answer.get('items'), list) else []

    batches = [readable_claims[i:i + DEEP_PER_CALL] for i in range(0, len(readable_claims), DEEP_PER_CALL)]
    for batch, items in zip(batches, at_once([lambda batch=batch: ask(batch) for batch in batches])):
        for item in items:
            try:
                claim, read = batch[int(item.get('id'))]
            except (TypeError, ValueError, IndexError, AttributeError):
                continue
            verdict = item.get('verdict')
            key = item.get('paper') if item.get('paper') in read else read[0]
            found = presskit.passage(str(item.get('quote') or ''), texts[key]) if verdict in ('supported', 'contradicted') else None
            reason = str(item.get('reason') or '').strip()
            if verdict == 'supported' and found:  # a value the paper never states keeps the sentence flagged
                claim.update(verdict='unclear' if claim.get('numbers_absent') else 'supported',
                             passage=dict(found, key=key), reason=reason)
            elif verdict == 'contradicted' and found:
                claim.update(verdict='contradicted', passage=dict(found, key=key), reason=reason)
            elif verdict in VERDICTS:
                claim.update(verdict='contradicted' if claim['verdict'] == 'contradicted' else 'unclear',
                             reason=reason or claim['reason'], not_in_full_text=True)
    return flagged


def run(texts, papers, lang, step=lambda: None):
    """{'claims': flagged, 'confirmed': supported by the full text, 'checked', 'citing'}."""
    every = citing_sentences(texts)
    readable_ones = [c for c in every if any((papers.get(k) or {}).get('abstract') for k in c['keys'])]
    claims = chosen(readable_ones)
    flagged = check_claims(claims, papers, lang) if claims else []
    if flagged:
        step()
        flagged = deepen(flagged, papers, lang)
    flagged.sort(key=lambda c: (c['verdict'] != 'contradicted', c['order']))
    return {'claims': [c for c in flagged if c['verdict'] != 'supported'],
            'confirmed': [c for c in flagged if c['verdict'] == 'supported'],
            'checked': len(claims), 'citing': len(every), 'readable': len(readable_ones)}
