"""Bibliography check (Premium): what is wrong with the references of a paper before it is submitted.

From a BibTeX file, and optionally the LaTeX source that cites it:

1. Each entry is identified: its DOI, its arXiv id, or, for the entries with
   neither, a conservative Crossref/ADS/OpenAlex cascade using titles, authors and locators.
   Matched metadata is compared field by field; an unresolved entry is never called incorrect.
2. Retracted papers come from OpenAlex (is_retracted, fed by Retraction Watch);
   papers withdrawn from arXiv from their arXiv comment.
3. A preprint cited as such may have been published since: arXiv's own DOI and
   journal reference, then Semantic Scholar's. The corrected entry gets the
   journal, year and DOI from Crossref, and a corrected .bib is offered.
4. An arXiv version older than the latest one, and the same paper cited under
   two keys, are reported too.
5. With the LaTeX source, the model reads each sentence with a citation next
   to the abstract of the cited paper, and reports the sentences the abstract
   does not support. Only abstracts are read: it is a warning, not a verdict.
"""
import contextvars
import html
import re
import uuid

import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import Http404, HttpResponse
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, bibliography, digest, generated, sources
from .models import Generated

MAX_UPLOAD = 5 * 1024 * 1024
MAX_ENTRIES = 400
TITLE_SEARCHES = 40  # bounded cascades for entries without identifiers; remaining entries are disclosed
OPENALEX_BATCH = 50
ARXIV_BATCH = 100
S2_BATCH = 'https://api.semanticscholar.org/graph/v1/paper/batch'
MAX_CLAIMS = 60  # sentences checked against the abstracts, in one or two model calls
CLAIMS_PER_CALL = 30
PER_DAY = 10
QUOTA_KEY = 'refcheck-quota:%s:%s'
CITE_RE = re.compile(r'\\(?:cite|citep|citet|citealp|citealt|citeauthor|parencite|textcite|autocite|footcite)\*?'
                     r'(?:\[[^\]]*\]){0,2}\{([^}]+)\}')
LEVELS = ('error', 'warning', 'info')
LIMITED = contextvars.ContextVar('openalex_limited', default=False)  # set when OpenAlex calls were skipped
STEPS = [
    ('read', _('Reading your references')),
    ('identify', _('Matching references with Crossref, NASA ADS and OpenAlex')),
    ('published', _('Looking for retractions and for preprints published since')),
    ('claims', _('An AI checks each citation against the abstract of the cited paper')),
]


# --- BibTeX ------------------------------------------------------------------

def _value(text, i):
    """The value starting at text[i] ({...}, "..." or a bare word) and the index after it."""
    if text[i] == '{':
        depth, j = 0, i
        while j < len(text):
            depth += {'{': 1, '}': -1}.get(text[j], 0)
            if depth == 0:
                return text[i + 1:j], j + 1
            j += 1
        return text[i + 1:], len(text)
    if text[i] == '"':
        j = i + 1
        while j < len(text) and (text[j] != '"' or text[j - 1] == '\\'):
            j += 1
        return text[i + 1:j], j + 1
    match = re.match(r'[^,}\s]*', text[i:])
    return match.group(0), i + len(match.group(0))


def parse_bibtex(text):
    """[{'key', 'type', 'fields': {name: value}, 'raw'}] of the entries (not @string, @comment, @preamble)."""
    entries = []
    for match in re.finditer(r'@(\w+)\s*([{(])', text):
        kind = match.group(1).lower()
        if kind in ('string', 'comment', 'preamble'):
            continue
        start, i = match.start(), match.end()
        key_match = re.match(r'\s*([^,\s]+)\s*,', text[i:])
        if not key_match:
            continue
        i += key_match.end()
        fields = {}
        while i < len(text):
            field = re.match(r'\s*([\w\-:.]+)\s*=\s*', text[i:])
            if not field:
                break
            i += field.end()
            if i >= len(text):
                break
            value, i = _value(text, i)
            fields[field.group(1).lower()] = ' '.join(value.split())
            rest = re.match(r'\s*(#\s*[^,}]+)?\s*,?', text[i:])  # string concatenations are dropped
            i += rest.end()
            if i < len(text) and text[i] in '})':
                break
        end = text.find('\n@', i)
        raw = text[start:i + 1] if i < len(text) and text[i] in '})' else text[start:end if end > 0 else len(text)]
        entries.append({'key': key_match.group(1), 'type': kind, 'fields': fields, 'raw': raw.strip()})
        if len(entries) >= MAX_ENTRIES:
            break
    return entries


def plain(text):
    """BibTeX braces and accents commands out, for showing and comparing titles."""
    text = re.sub(r'\\[a-zA-Z]+\s*(?=\{)|[{}]', '', text or '')
    return ' '.join(re.sub(r'\\[\'"`^~=.]', '', text).split())


def identify(entry):
    """(arxiv id without version or '', the version cited or '', DOI or '') from the entry's fields."""
    f = entry['fields']
    doi = sources.clean_doi(f.get('doi', '')) or ''
    arxiv, version = '', ''
    match = sources.ARXIV_DOI_RE.match(doi)
    if match:
        arxiv, doi = match.group(1), ''
    # the eprint field is an arXiv id; elsewhere, an id only counts next to the word arXiv
    candidates = [(f.get('eprint', ''), True)] + [(f.get(k, ''), False) for k in ('url', 'journal', 'note', 'howpublished')]
    for value, trusted in candidates:
        found = re.search(r'(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?', value)
        if found and (trusted or 'arxiv' in value.lower()):
            arxiv, version = arxiv or found.group(1), found.group(2) or ''
            break
    if not doi:
        found = sources.DOI_RE.search(f.get('url', ''))
        if found and not sources.ARXIV_DOI_RE.match(found.group(1)):
            doi = found.group(1).rstrip('.,;)').lower()
    return arxiv, version, doi.lower()


def is_preprint_entry(entry):
    f = entry['fields']
    journal = f.get('journal', '').lower()
    return not f.get('doi') or sources.ARXIV_DOI_RE.match(f.get('doi', '')) or 'arxiv' in journal or \
        (entry['type'] in ('misc', 'unpublished', 'online') and not journal)


# --- lookups -----------------------------------------------------------------

def arxiv_records(ids):
    """{arxiv id: arXiv API record} (latest version), by batches of ARXIV_BATCH."""
    found = {}
    ids = list(dict.fromkeys(ids))
    for start in range(0, len(ids), ARXIV_BATCH):
        try:
            papers = digest.fetch_ids(ids[start:start + ARXIV_BATCH])
        except digest.ArxivUnavailable:
            continue
        for paper in papers:
            versioned = paper['arxiv_id'].replace('--', '/')
            found[re.sub(r'v\d+$', '', versioned)] = dict(paper, version=re.search(r'v(\d+)$', versioned).group(1)
                                                          if re.search(r'v\d+$', versioned) else '')
    return found


def openalex_works(dois):
    """{doi: {'title', 'retracted', 'abstract'}} from OpenAlex, by batches of OPENALEX_BATCH."""
    found = {}
    dois = list(dict.fromkeys(d for d in dois if d))
    for start in range(0, len(dois), OPENALEX_BATCH):
        batch = dois[start:start + OPENALEX_BATCH]
        if not sources.tool_search_allowed():
            LIMITED.set(True)
            break
        try:
            body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
                'filter': 'doi:%s' % '|'.join(batch), 'per_page': len(batch),
                'select': 'doi,display_name,is_retracted,abstract_inverted_index'}))
        except sources.SourceUnavailable:
            continue
        for work in (body or {}).get('results', []):
            doi = (work.get('doi') or '').split('doi.org/')[-1].lower()
            found[doi] = {'title': work.get('display_name') or '', 'retracted': bool(work.get('is_retracted')),
                          'abstract': sources.inverted_abstract(work.get('abstract_inverted_index'))}
    return found


def by_title(title):
    """(doi, arxiv id) of the OpenAlex work with this title, ('', '') when no title matches closely."""
    if not sources.tool_search_allowed():
        LIMITED.set(True)
        return '', ''
    try:
        body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
            'search': title[:300], 'per_page': 3, 'select': 'doi,display_name,locations'}))
    except sources.SourceUnavailable:
        return '', ''
    wanted = normalized(title)
    for work in (body or {}).get('results', []):
        if normalized(work.get('display_name') or '') == wanted:
            doi = (work.get('doi') or '').split('doi.org/')[-1].lower()
            match = sources.ARXIV_DOI_RE.match(doi)
            return ('', match.group(1)) if match else (doi, sources.arxiv_of(work))
    return '', ''


def normalized(text):
    return re.sub(r'[^a-z0-9]+', ' ', plain(text).lower()).strip()


def s2_published(arxiv_ids):
    """{arxiv id: journal DOI} from Semantic Scholar, for preprints arXiv does not link to a journal."""
    if not arxiv_ids:
        return {}
    headers = {'x-api-key': settings.S2_API_KEY} if getattr(settings, 'S2_API_KEY', '') else {}
    try:
        response = requests.post(S2_BATCH, params={'fields': 'externalIds,journal,publicationTypes'}, headers=headers,
                                 json={'ids': ['ARXIV:%s' % a for a in arxiv_ids[:500]]}, timeout=60)
        response.raise_for_status()
        rows = response.json()
    except (requests.RequestException, ValueError):
        return {}
    found = {}
    for arxiv_id, row in zip(arxiv_ids, rows):
        doi = ((row or {}).get('externalIds') or {}).get('DOI') or ''
        if doi and not sources.ARXIV_DOI_RE.match(doi) and ((row.get('journal') or {}).get('name')):
            found[arxiv_id] = doi.lower()
    return found


def crossref_fields(doi):
    """BibTeX fields of the published version from Crossref: journal, year, volume, pages, doi."""
    try:
        body = sources._json(sources.CROSSREF % doi, {'mailto': 'contact@summarizepaper.com'})
    except sources.SourceUnavailable:
        return {'doi': doi}
    work = (body or {}).get('message') or {}
    year = ((work.get('published') or work.get('issued') or {}).get('date-parts') or [[None]])[0][0]
    fields = {'journal': html.unescape((work.get('container-title') or [''])[0]), 'year': str(year or ''),
              'volume': work.get('volume') or '', 'pages': work.get('page') or work.get('article-number') or '',
              'doi': doi}
    return {k: v for k, v in fields.items() if v}


def with_fields(entry, fields, kind=None, published=False):
    """Rewrite only reviewed fields, preserving the key and the remaining source fields."""
    merged = {k: v for k, v in entry['fields'].items()
              if not published or k not in ('archiveprefix', 'primaryclass', 'howpublished')}
    merged.update(fields)
    lines = ['@%s{%s,' % (kind or entry['type'], entry['key'])]
    lines += ['  %s = {%s},' % (name, re.sub(r'(?<!\\)&', r'\\&', value) if name == 'journal' else value)
              for name, value in merged.items()]
    return '\n'.join(lines) + '\n}'


# --- citation sentences ------------------------------------------------------

def citing_sentences(tex):
    """[(sentence, [keys])] of the LaTeX text, comments and preamble left out."""
    tex = re.sub(r'(?<!\\)%.*', '', tex)
    tex = tex.split('\\begin{document}', 1)[-1]
    tex = re.sub(r'\\(?:begin|end)\{[^}]*\}', ' ', tex)
    found = []
    for sentence in re.split(r'(?<=[.!?])\s+(?=[A-Z\\])', ' '.join(tex.split())):
        keys = [k.strip() for group in CITE_RE.findall(sentence) for k in group.split(',') if k.strip()]
        if keys and 30 < len(sentence) < 1500:
            found.append((sentence, list(dict.fromkeys(keys))))
    return found


def check_claims(claims, lang):
    """[{'key', 'sentence', 'verdict', 'reason'}] of the citations the abstracts do not clearly support."""
    flagged = []
    for start in range(0, len(claims), CLAIMS_PER_CALL):
        batch = claims[start:start + CLAIMS_PER_CALL]
        listing = '\n\n'.join('<claim id="%d">\nSentence of the draft: %s\nCited paper: %s\nIts abstract: %s\n</claim>'
                              % (start + i, c['sentence'][:1200], c['title'], c['abstract'][:2500])
                              for i, c in enumerate(batch))
        answer = generated.chat_json(
            'Each claim above is a sentence of a draft that cites a paper, and the abstract of that paper. Say whether '
            'the abstract supports what the sentence attributes to the cited paper: "supported", "unclear" when the '
            'abstract is silent on it (the full text may say it), or "contradicted" when the abstract says otherwise '
            'or the sentence clearly misrepresents the paper. A citation given as background for a topic the paper '
            'is about is "supported". For "unclear" and "contradicted", explain in %s in one sentence. Answer with a '
            'JSON object {"items": [{"id": "claim id", "verdict": "supported|unclear|contradicted", "reason": "..."}]}.'
            % generated.language_name(lang), paid=True, max_tokens=4000, system=listing)
        for item in answer.get('items', []):
            if not isinstance(item, dict) or item.get('verdict') not in ('unclear', 'contradicted'):
                continue
            try:
                claim = claims[int(item.get('id'))]
            except (TypeError, ValueError, IndexError):
                continue
            flagged.append(dict(claim, verdict=item['verdict'], reason=str(item.get('reason', '')).strip()))
    flagged.sort(key=lambda c: c['verdict'] != 'contradicted')
    return flagged


# --- the check ---------------------------------------------------------------

def check(bib, tex, lang, step=lambda name: None):
    LIMITED.set(False)
    entries = parse_bibtex(bib)
    if not entries:
        raise generated.GenerationError('no BibTeX entry found')
    step('identify')
    ids = {}
    for entry in entries:
        ids[entry['key']] = identify(entry)
    def openalex_allowed():
        allowed = sources.tool_search_allowed()
        if not allowed:
            LIMITED.set(True)
        return allowed
    resolver = bibliography.Resolver(openalex_allowed)
    metadata, skipped, searched = {}, set(), 0
    for entry in entries:
        arxiv, version, doi = ids[entry['key']]
        matched = None
        if doi:
            matched = resolver.resolve(entry, doi)
        elif not arxiv:
            if searched < TITLE_SEARCHES:
                searched += 1
                matched = resolver.resolve(entry)
            else:
                skipped.add(entry['key'])
        step('identify')  # heartbeat while a long bibliography is resolved
        if matched:
            metadata[entry['key']] = matched
            ids[entry['key']] = (arxiv or matched.get('arxiv', ''), version, doi or matched['doi'])
    records = arxiv_records([a for a, _, _ in ids.values() if a])
    step('published')
    works = openalex_works([d for _, _, d in ids.values() if d] +
                           [r['doi'].lower() for r in records.values() if r.get('doi')])
    unlinked = [a for a, r in records.items() if not r.get('doi')]
    s2 = s2_published([a for a in unlinked if any(ids[e['key']][0] == a and is_preprint_entry(e) for e in entries)])

    items, corrected, seen = [], {}, {}
    for entry in entries:
        arxiv, version, doi = ids[entry['key']]
        record = records.get(arxiv) if arxiv else None
        title = plain(entry['fields'].get('title', '')) or (record or {}).get('title', '') or entry['key']
        issues = []
        for ref in (('doi:' + doi) if doi else '', ('arxiv:' + arxiv) if arxiv else ''):
            if ref and ref in seen and seen[ref] != entry['key']:
                issues.append({'kind': 'duplicate', 'level': 'warning', 'other': seen[ref]})
            elif ref:
                seen.setdefault(ref, entry['key'])
        matched = metadata.get(entry['key'])
        if not matched and not record:
            issues.append({'kind': 'unknown', 'level': 'info',
                           'reason': 'limit' if entry['key'] in skipped or resolver.limited else
                                     'unavailable' if resolver.unavailable else 'no_match'})
        if arxiv and record is None:
            issues.append({'kind': 'arxiv_missing', 'level': 'warning', 'arxiv': arxiv})
        if record and re.search(r'\bwithdrawn\b', record.get('comment', ''), re.I):
            issues.append({'kind': 'withdrawn', 'level': 'error', 'detail': record['comment'][:300]})
        published = (record or {}).get('doi', '').lower() or s2.get(arxiv, '')
        for d in (doi, published):
            if d and works.get(d, {}).get('retracted'):
                issues.append({'kind': 'retracted', 'level': 'error', 'doi': d})
                break
        if record and published and is_preprint_entry(entry) and published != doi:
            fields = crossref_fields(published)
            corrected[entry['key']] = with_fields(entry, fields, 'article', published=True)
            issues.append({'kind': 'published', 'level': 'warning', 'doi': published,
                           'journal': fields.get('journal') or record.get('journal_ref', ''), 'bibtex': corrected[entry['key']]})
        changes = bibliography.differences(entry, matched) if matched else []
        if changes:
            fields = {c['field']: c['after'] for c in changes if c['field'] != 'type'}
            if matched['doi']:
                fields['doi'] = matched['doi']
            kind = matched['type'] if any(c['field'] == 'type' for c in changes) else entry['type']
            if kind == 'book' and matched['fields'].get('isbn') and not entry['fields'].get('isbn'):
                fields['isbn'] = matched['fields']['isbn']
            corrected[entry['key']] = with_fields(entry, fields, kind)
            issues.append({'kind': 'metadata', 'level': 'warning', 'changes': changes,
                           'source': matched['source'], 'url': matched['url'], 'bibtex': corrected[entry['key']]})
        if record and version and record.get('version') and int(version[1:]) < int(record['version']):
            issues.append({'kind': 'old_version', 'level': 'info', 'cited': version, 'latest': 'v%s' % record['version']})
        items.append({'key': entry['key'], 'title': title, 'arxiv': arxiv, 'doi': doi or published, 'issues': issues,
                      'abstract': (record or {}).get('abstract') or works.get(doi, {}).get('abstract', ''),
                      'source': matched['source'] if matched else 'arXiv' if record else '',
                      'source_url': matched['url'] if matched else 'https://arxiv.org/abs/' + arxiv if record else '',
                      'verification': 'incorrect' if changes else 'published' if any(i['kind'] == 'published' for i in issues) else
                                      'identified' if matched or record else 'unverified'})

    claims = []
    if tex:
        step('claims')
        by_key = {i['key']: i for i in items}
        for sentence, keys in citing_sentences(tex):
            for key in keys:
                item = by_key.get(key)
                if item and item['abstract'] and len(claims) < MAX_CLAIMS:
                    claims.append({'key': key, 'title': item['title'], 'abstract': item['abstract'],
                                   'sentence': re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])*', '', sentence).strip()})
        flagged = check_claims(claims, lang) if claims else []
    else:
        flagged = []
    for item in items:
        item.pop('abstract')
    rank = {level: i for i, level in enumerate(LEVELS)}
    problems = [i for i in items if i['issues']]
    problems.sort(key=lambda i: min(rank[x['level']] for x in i['issues']))
    fixed_bib = bib
    for entry in entries:
        if entry['key'] in corrected and entry['raw'] in fixed_bib:
            fixed_bib = fixed_bib.replace(entry['raw'], corrected[entry['key']], 1)
    return {
        'entries': len(entries), 'problems': problems, 'claims': [
            {k: c[k] for k in ('key', 'title', 'sentence', 'verdict', 'reason')} for c in flagged],
        'claims_checked': len(claims), 'with_tex': bool(tex),
        'counts': {level: sum(any(x['level'] == level for x in i['issues']) for i in items) for level in LEVELS},
        'fixed_bib': fixed_bib if corrected else '', 'corrected': len(corrected), 'limited': LIMITED.get(),
        'summary': {status: sum(i['verification'] == status for i in items)
                    for status in ('incorrect', 'published', 'identified', 'unverified')},
        'identified': [{k: i[k] for k in ('key', 'title', 'source', 'source_url')} for i in items if i['verification'] == 'identified' and not i['issues']],
        'sources_unavailable': sorted(resolver.unavailable), 'lookup_limited': resolver.limited or bool(skipped),
        'metadata_checked': len(metadata),
    }


def write(bib, tex, lang):
    def work(pk):
        generated.progress(pk, 'read')
        return check(bib, tex, lang, lambda name: generated.progress(pk, name))
    return work


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def _text(upload, pasted):
    if upload is None:
        return pasted
    if upload.size > MAX_UPLOAD:
        return None
    return upload.read().decode('utf-8', errors='replace')


def refcheck_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/refcheck.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.is_paid(request.user)
    error = None
    if request.method == 'POST' and paid:
        bib = _text(request.FILES.get('bib'), request.POST.get('bibtext', ''))
        tex = _text(request.FILES.get('tex'), request.POST.get('textext', ''))
        if bib is None or tex is None:
            error = 'file'
        elif not parse_bibtex(bib):
            error = 'empty'
        elif not _spend(request.user):
            error = 'quota'
        else:
            row = generated.start('refcheck', uuid.uuid4().hex, get_language(), write(bib, tex.strip(), get_language()),
                                  user=request.user)
            return redirect('/%s/refcheck/%s/' % (get_language(), row.key))
    checks = Generated.objects.filter(kind='refcheck', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/refcheck.html', {'paid': paid, 'error': error, 'checks': checks,
                                                        'per_day': PER_DAY, 'upload_max_bytes': MAX_UPLOAD, 'upload_max_mb': MAX_UPLOAD // (1024 * 1024)})


def _row(request, key):
    row = Generated.objects.filter(kind='refcheck', key=key, user=request.user).first()
    if row is None:
        raise Http404
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    return row


@login_required
def refcheck_result(request, key):
    row = _row(request, key)
    return render(request, 'summarizer/refcheck_result.html', {
        'row': row, 'result': row.data() if row.status == 'done' else None, 'steps': STEPS})


@login_required
def refcheck_bib(request, key):
    row = _row(request, key)
    bib = row.data().get('fixed_bib') if row.status == 'done' else ''
    if not bib:
        raise Http404
    response = HttpResponse(bib, content_type='application/x-bibtex; charset=utf-8')
    response['Content-Disposition'] = 'attachment; filename="references-corrected.bib"'
    return response
