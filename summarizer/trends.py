"""Rising topics: the terms papers use much more this week than in the weeks before.

Every paper is dated by the submission of its first version (arXiv's OAI-PMH, see trend_sources.py), so
we count a week of submissions from the following Tuesday at 06:00 UTC, after Monday's normal
arXiv announcement. For each week and scope (all of arXiv '*', an archive like cs, a category like cs.LG, bioRxiv,
medRxiv and their categories), we count how many papers use each term of 1 to 3 words of their title
and abstract (TermWeek). A term rises when this week's count is both at least twice what its share over
the previous eight weeks predicts and very unlikely by chance (a Poisson log-likelihood ratio), and when
it comes from several independent teams, so that one group's series of papers is not a trend. Its
history then says what kind of rise it is: new, rising for several weeks, recurring at this time of
year, a comeback, or a sudden spike. The model groups the rising terms into topics and names them;
it never decides what rises: the counts do.
"""
import base64
import datetime
import hashlib
import json
import logging
import math
import re
import time
import zlib
from functools import reduce
from operator import or_

import numpy as np
import requests
from django.core.cache import cache
from django.db import transaction
from django.db.models import F, Q, Value
from django.db.models.functions import Mod
from django.utils import timezone

from . import generated, sources
from .digest import ARCHIVES, BIO_ARCHIVES, BIO_CODES, SERVER_NAMES, archive_of

logger = logging.getLogger(__name__)

WEEK = datetime.timedelta(days=7)
CLOSE = datetime.timedelta(days=1)  # Tuesday after the Monday-to-Sunday submission week
CLOSE_TIME = datetime.timedelta(hours=6)  # after Monday's 20:00 US Eastern announcement, in either DST season
BASE_WEEKS = 8  # the weeks a rise is measured against
MIN_BASE_WEEKS = 4
SERIES_WEEKS = 26  # the history drawn on each topic
HISTORY_WEEKS = 160  # counts kept, about three years
KEEP_PAPERS = 3  # weeks of papers kept for examples, teams and lists
STORE_FLOOR = 2  # terms used by a single paper in a week are not stored
PART_TERMS = 2000  # terms per stored row: the history of one term reads one small row per week
MIN_WEEK_PAPERS = 60  # smaller scopes have too few papers for a weekly trend
MIN_COUNT = {'*': 12}
MIN_COUNT_SCOPE = 6
MIN_GROWTH = 2.0
MIN_LLR = 15.0  # about p < 1e-4 for one term; checked on real weeks (trends_calibrate): the model then drops the noise left
MIN_TEAMS = 3
MAX_TEAM_SHARE = 0.5  # no team may sign more than half of a rising term's papers
CANDIDATES = 30
SCOPES_PER_RUN = 8
FREE_TOPICS = 3
LANGS = ('en', 'fr')
LOCK = 'trends-run'
ARXIV_API = 'https://export.arxiv.org/api/query'

STOP = set('''a about above across after again against all almost along also although always am among an and another any
are around as at available based be because been before being below between both but by can cannot could did do does
done due during each either else enough especially et etc even ever every few first for found from further furthermore
general given has have having here however i if in including into is it its itself just known last least less like
made main many may me more moreover most much must near nearly new no nor not novel now of often on once one only onto or
other others otherwise our ours out over overall own paper papers per perhaps present presented proposed propose proposes
provide provides rather recent recently respectively result results same several shall show shows shown significant
significantly since so some such than that the their them then there thereby therefore these they this those though
three through thus to together too toward towards two under unlike until up upon us use used uses using various very via
was way we well were what when where whereas whether which while who whom whose why will with within without work works
would yet demonstrate demonstrates obtain obtained find finds second third also existing introduce introduces able
achieve achieves ii iii iv vi vii viii ix xi xii how should largely better performs perform subsequent january february
march april may june july august september october november december'''.split())
# words too common to be a term alone, though they end many ('diffusion model', 'large language model')
GENERIC = set('''model data method analysis study framework performance task approach problem high low large different
important effective efficient state art system result network learning'''.split())
TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9\-]*[A-Za-z0-9]")
KEEP_ENDINGS = ('ss', 'us', 'is', 'ics', 'ous', 'as', 'ys')
NOT_PLURAL = {'lens', 'series', 'species', 'news', 'means', 'mathematics', 'physics', 'always', 'perhaps', 'whereas',
              'various', 'previous', 'sometimes', 'nucleus', 'radius', 'focus', 'status', 'corpus', 'census', 'class'}


def fold(word):
    """The singular of a term's last word: 'networks' -> 'network', 'galaxies' -> 'galaxy', 'physics' stays."""
    if len(word) <= 4 or word in NOT_PLURAL or not word.endswith('s') or word.endswith(KEEP_ENDINGS):
        return word
    if word.endswith('ies') and len(word) > 5:
        return word[:-3] + 'y'
    if word.endswith(('sses', 'ches', 'shes')):
        return word[:-2]
    return word[:-1]


def _words(chunk):
    """[(word, acronym)] of a clause: 'LLMs' -> ('llm', True); acronyms and names like LoRA or AlphaFold count alone."""
    found = []
    for match in TOKEN.finditer(chunk):
        word = match.group(0)
        acronym = len(word) <= 12 and sum(c.isupper() for c in word) >= 2 and not word.isupper() or \
            word.isupper() and 2 <= len(word) <= 8
        if acronym and len(word) > 2 and word.endswith('s') and word[-2].isupper():
            word = word[:-1]  # LLMs
        found.append((word.lower(), acronym))
    return found


def terms(text):
    """The distinct 1- to 3-word terms of a title and abstract, the last word in the singular."""
    text = re.sub(r'\$[^$]*\$|\\[A-Za-z]+', ' ', text)
    found = set()
    # a term never spans two sentences or clauses
    for chunk in re.split(r'[.;:!?()\[\],]+(?:\s|$)|[;:!?()\[\]]', text):
        words = _words(chunk)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                first, last = gram[0][0], gram[-1][0]
                if first in STOP or last in STOP:
                    continue
                if n == 1 and not gram[0][1] and (len(first) < 4 or '-' not in first and len(first) < 5):
                    continue
                term = [w for w, _ in gram[:-1]] + [last if gram[-1][1] else fold(last)]
                if not all(w in GENERIC for w in term):
                    found.add(' '.join(term))
    return found


def normalize(text):
    """A term typed by a visitor, as it is counted ('Large Language Models' -> 'large language model'), or ''."""
    words = _words(' '.join(text.split())[:80])
    if not 1 <= len(words) <= 3 or words[0][0] in STOP or words[-1][0] in STOP:
        return ''
    if len(words) == 1 and not words[0][1] and (len(words[0][0]) < 4 or '-' not in words[0][0] and len(words[0][0]) < 5):
        return ''
    folded = [w for w, _ in words[:-1]] + [words[-1][0] if words[-1][1] else fold(words[-1][0])]
    return '' if all(w in GENERIC for w in folded) else ' '.join(folded)


def week_of(day):
    return day - datetime.timedelta(days=day.weekday())


def closed_week(today):
    """Latest eligible submission week, from Tuesday 06:00 UTC.

    Datetimes observe the UTC cutoff; date-only callers select by calendar day.
    Delayed moderation can still make papers public after this first count.
    """
    if isinstance(today, datetime.datetime):
        today = (today.astimezone(datetime.timezone.utc) - CLOSE_TIME).date()
    week = week_of(today) - WEEK
    return week if week + WEEK + CLOSE <= today else week - WEEK


def scopes_of(categories):
    """The scopes a paper counts in: bioRxiv and medRxiv papers only in their own, so '*' stays all of arXiv."""
    categories = {c for c in categories if c}
    archives = {archive_of(c) for c in categories}
    if not categories:
        return set()
    if archives & set(BIO_ARCHIVES):
        return categories | archives
    return {'*'} | archives | categories


def scope_name(scope):
    from .arxiv_taxonomy import GROUPS, NAMES
    from django.utils.translation import gettext as _
    if scope == '*':
        return _('All of arXiv')
    if scope in SERVER_NAMES:
        return SERVER_NAMES[scope]
    if scope in BIO_CODES:
        return '%s · %s' % (SERVER_NAMES[BIO_CODES[scope][0]], BIO_CODES[scope][1])
    if scope in NAMES:
        return '%s (%s)' % (NAMES[scope], scope)
    for title, categories in GROUPS:
        if {archive_of(c) for c, _ in categories} == {scope}:
            return '%s (%s)' % (title, scope)
    return scope


def author_key(name):
    """'Ann B. Lee' -> 'ann b lee': the full name, so that two people sharing a surname and an initial stay apart."""
    return ' '.join(re.sub(r'[^\w\s-]', ' ', name.lower()).split())


# --- counting -------------------------------------------------------------------------------------------------

def count(papers, only=None):
    """{scope: (papers, {term: papers using it})} of a week's papers (dicts with title, abstract, categories),
    for the scopes `only` accepts (all by default)."""
    scopes = {}
    for paper in papers:
        own = [s for s in scopes_of(paper['categories']) if only is None or only(s)]
        if not own:
            continue
        found = terms('%s. %s' % (paper['title'], paper['abstract']))
        for scope in own:
            totals = scopes.setdefault(scope, [0, {}])
            totals[0] += 1
            counts = totals[1]
            for term in found:
                counts[term] = counts.get(term, 0) + 1
    return {scope: tuple(totals) for scope, totals in scopes.items()}


# a week of all of arXiv holds millions of (scope, term) pairs: they are counted and stored a group of scopes at a time
GROUPS = [lambda s: s == '*'] + [lambda s, i=i: s != '*' and '.' not in s and zlib.crc32(s.encode()) % 2 == i for i in range(2)] + \
    [lambda s, i=i: '.' in s and zlib.crc32(s.encode()) % 3 == i for i in range(3)]


def is_bio(scope):
    return archive_of(scope) in BIO_ARCHIVES


def count_week(week, papers, source=None):
    """Count and store a week, a group of scopes at a time; the rows stored. source 'arxiv' or 'bio' replaces only
    the scopes of that source (bioRxiv and medRxiv papers never count in arXiv's), so each can be read apart."""
    from .models import TermWeek
    rows = TermWeek.objects.filter(week=week)
    bio = Q(scope__startswith='biorxiv') | Q(scope__startswith='medrxiv')
    rows = rows.filter(bio) if source == 'bio' else rows.exclude(bio) if source == 'arxiv' else rows
    rows.delete()
    wanted = (lambda s: True) if source is None else (lambda s: is_bio(s) == (source == 'bio'))
    return sum(store(week, count(papers, lambda s, g=group: g(s) and wanted(s)), replace=False) for group in GROUPS)


def part_of(term, parts):
    return zlib.crc32(term.encode()) % parts


def store(week, counted, replace=True):
    """Store the counts of a week, replacing all of its stored counts unless replace=False."""
    from .models import TermWeek
    rows = []
    for scope, (papers, counts) in counted.items():
        kept = {t: n for t, n in counts.items() if n >= STORE_FLOOR}
        parts = 1 + len(kept) // PART_TERMS
        split = [{} for _ in range(parts)]
        for term, n in kept.items():
            split[part_of(term, parts)][term] = n
        rows += [TermWeek(scope=scope, week=week, part=i, parts=parts, papers=papers,
                          counts=json.dumps(chunk, separators=(',', ':'), sort_keys=True)) for i, chunk in enumerate(split)]
    with transaction.atomic():
        if replace:
            TermWeek.objects.filter(week=week).delete()
        TermWeek.objects.bulk_create(rows, batch_size=100)
    return len(rows)


def paper_rows(papers):
    from .models import TrendPaper
    return [TrendPaper(key=p['key'], week=week_of(p['day']), day=p['day'], title=p['title'][:2000],
                       abstract=p['abstract'][:6000], authors=json.dumps(p['authors'][:100]),
                       categories=','.join(p['categories'])[:300]) for p in papers]


def keep_papers(papers):
    from .models import TrendPaper
    papers = list({p['key']: p for p in papers}.values())
    with transaction.atomic():
        for start in range(0, len(papers), 500):
            TrendPaper.objects.filter(key__in=[p['key'] for p in papers[start:start + 500]]).delete()
        TrendPaper.objects.bulk_create(paper_rows(papers), batch_size=500)


def stored_papers(week, scope=None):
    """The kept papers of a week (in a scope), as dicts."""
    from .models import TrendPaper
    found = []
    for key, day, title, abstract, authors, categories in TrendPaper.objects.filter(week=week).values_list(
            'key', 'day', 'title', 'abstract', 'authors', 'categories').iterator():
        categories = [c for c in categories.split(',') if c]
        if scope is None or scope in scopes_of(categories):
            found.append({'key': key, 'day': day, 'title': title, 'abstract': abstract,
                          'authors': json.loads(authors or '[]'), 'categories': categories})
    return found


def harvest(weeks, today):
    """Read the papers of these weeks again from arXiv, bioRxiv and medRxiv, keep them and count each week."""
    from . import trend_sources
    first = min(weeks)
    wanted = set(weeks)
    papers = [p for p in trend_sources.arxiv_papers(first) if week_of(p['day']) in wanted]
    for week in weeks:
        papers += trend_sources.bio_papers(week, min(week + WEEK - datetime.timedelta(days=1), today))
    keep_papers(papers)
    for week in sorted(weeks):
        count_week(week, stored_papers(week))
    return len(papers)


# --- reading the counts ---------------------------------------------------------------------------------------

def _value(text, term):
    """The count of a term in a stored JSON row, without parsing the whole row."""
    key = '"%s":' % term
    start = text.find(key)
    while start > 0 and text[start - 1] not in '{,':
        start = text.find(key, start + 1)
    if start < 0:
        return 0
    end = start + len(key)
    stop = end
    while stop < len(text) and text[stop].isdigit():
        stop += 1
    return int(text[end:stop] or 0)


def week_counts(scope, weeks):
    """{week: (papers, {term: n})} of whole weeks of a scope."""
    from .models import TermWeek
    found = {}
    for week, papers, counts in TermWeek.objects.filter(scope=scope, week__in=weeks).values_list('week', 'papers', 'counts'):
        entry = found.setdefault(week, [papers, {}])
        entry[1].update(json.loads(counts))
    return {week: tuple(entry) for week, entry in found.items()}


def history(scope, wanted, weeks=None):
    """{term: [(week, papers, n)]} over the stored weeks (or these weeks), oldest first; reads one row per week and term."""
    from .models import TermWeek
    wanted = list(dict.fromkeys(wanted))
    rows = TermWeek.objects.filter(scope=scope)
    if weeks is not None:
        rows = rows.filter(week__in=weeks)
    crcs = {term: zlib.crc32(term.encode()) for term in wanted}
    if wanted:
        rows = rows.filter(reduce(or_, [Q(part=Mod(Value(c), F('parts'))) for c in set(crcs.values())]))
    papers, found = {}, {term: {} for term in wanted}
    for week, part, parts, total, counts in rows.values_list('week', 'part', 'parts', 'papers', 'counts').iterator():
        papers[week] = total
        for term in wanted:
            if crcs[term] % parts == part:
                found[term][week] = _value(counts, term)
    if weeks is None:
        weeks = sorted(set(TermWeek.objects.filter(scope=scope).values_list('week', flat=True)))
    return {term: [(w, papers[w], found[term].get(w, 0)) for w in sorted(weeks) if w in papers] for term in wanted}


def inside(short, long):
    return short != long and (' %s ' % short) in (' %s ' % long)


def rising_terms(scope, week, verbose=False):
    """The terms rising in a scope in a week, most significant first, as dicts; None without enough history.
    verbose also returns the terms left out and why."""
    base_weeks = [week - WEEK * i for i in range(1, BASE_WEEKS + 1)]
    counted = week_counts(scope, [week] + base_weeks)
    base = [w for w in base_weeks if w in counted]
    if week not in counted or len(base) < MIN_BASE_WEEKS:
        return None
    papers, counts = counted[week]
    if papers < MIN_WEEK_PAPERS:
        return None
    base_papers = sum(counted[w][0] for w in base)
    floor = MIN_COUNT.get(scope, MIN_COUNT_SCOPE)
    rising = []
    for term, n in counts.items():
        if n < floor:
            continue
        # a week without the term had 0 or 1 paper using it (single uses are not stored): count it as half a paper
        before = sum(counted[w][1].get(term, 0.5) for w in base)
        expected = before / base_papers * papers
        growth = n / expected
        if growth < MIN_GROWTH:
            continue
        llr = 2 * (n * math.log(n / expected) - (n - expected))
        if llr >= MIN_LLR:
            rising.append({'term': term, 'n': n, 'expected': round(expected, 1), 'growth': round(growth, 1), 'llr': round(llr, 1)})
    rising.sort(key=lambda t: -t['llr'])
    kept = []
    for item in rising:
        # 'large language' and 'language model' when 'large language model' is there with nearly the same count
        if any(inside(item['term'], other['term']) and item['n'] <= other['n'] * 1.25 for other in kept):
            continue
        kept = [o for o in kept if not (inside(o['term'], item['term']) and o['n'] <= item['n'] * 1.25)]
        kept.append(item)
        if len(kept) == CANDIDATES * 2:
            break
    left = []
    week_papers = stored_papers(week, scope)
    if week_papers:
        by_key = {p['key']: p for p in week_papers}
        texts = {p['key']: terms('%s. %s' % (p['title'], p['abstract'])) for p in week_papers}
        checked = []
        for item in kept:
            using = [k for k, found in texts.items() if item['term'] in found]
            teams = team_of([by_key[k] for k in using])
            sizes = {}
            for key in using:
                sizes[teams[key]] = sizes.get(teams[key], 0) + 1
            item['teams'] = len(sizes)
            largest = max(sizes.values()) if sizes else 0
            if len(sizes) < MIN_TEAMS or largest > MAX_TEAM_SHARE * max(len(using), 1):
                left.append(dict(item, reason='%d teams, the largest signs %d of %d papers' % (len(sizes), largest, len(using))))
                continue
            checked.append(item)
        kept = checked
    kept = kept[:CANDIDATES]
    return (kept, left) if verbose else kept


def team_of(papers):
    """{paper key: team}: among these papers (those using a term), papers that share an author are in the same team,
    however long the chain. Only the papers of one term are linked, so the week's whole co-author network does not
    join everyone through common names."""
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for paper in papers:
        keys = ['p:' + paper['key']] + ['a:' + author_key(a) for a in paper['authors'] if author_key(a)]
        for key in keys[1:]:
            parent[find(key)] = find(keys[0])
    return {paper['key']: find('p:' + paper['key']) for paper in papers}


def kind(points):
    """What kind of rise the last point of a term's weekly history is: {'kind', 'weeks', 'since'}.

    new: almost absent before; recurring: as frequent at this time last year; growing: up for at least three
    weeks; comeback: as frequent some months ago; spike: a sudden rise, often an event, a release or a deadline."""
    if not points:
        return {'kind': 'spike', 'weeks': 1, 'since': None}
    shares = [(w, max(n, 0.5) / p if p else 0) for w, p, n in points]
    week, current = shares[-1]
    before = shares[:-1]
    reference = before[-12:-4] if len(before) >= 12 else before[:-3] or before
    ref = sum(s for _, s in reference) / len(reference) if reference else current
    weeks = 0
    for _, share in reversed(shares):
        if share < 1.5 * ref:
            break
        weeks += 1
    result = {'kind': 'spike', 'weeks': weeks, 'since': None}
    old = points[:-4]
    if len(points) >= 13 and sum(n for _, _, n in old) <= 2:
        first = next((w for w, _, n in points if n), week)
        return dict(result, kind='new', since=first.isoformat())
    year = [(w, s) for w, s in before if datetime.timedelta(weeks=49) <= week - w <= datetime.timedelta(weeks=55)]
    if year and max(s for _, s in year) >= 0.6 * current:
        return dict(result, kind='recurring', since=max(year, key=lambda x: x[1])[0].isoformat())
    if weeks >= 3:
        return dict(result, kind='growing')
    earlier = [(w, s) for w, s in before if datetime.timedelta(weeks=13) <= week - w < datetime.timedelta(weeks=49)]
    if earlier and max(s for _, s in earlier) >= 0.6 * current:
        return dict(result, kind='comeback', since=max(earlier, key=lambda x: x[1])[0].isoformat())
    return result


# --- writing the topics ---------------------------------------------------------------------------------------

def first_use(term):
    """The oldest arXiv paper whose title or abstract uses a term, {'arxiv_id', 'title', 'year'}, or None; cached.
    arXiv's search matches stems and loose phrases, so its oldest answers are checked against the term itself."""
    key = 'trends-first:%s' % zlib.crc32(term.encode())
    found = cache.get(key)
    if found is not None:
        return found or None
    found = {}
    try:
        time.sleep(3)  # arXiv's API asks for one request every three seconds
        response = requests.get(ARXIV_API, timeout=60, headers={'User-Agent': sources.USER_AGENT}, params={
            'search_query': 'abs:"%s"' % term, 'sortBy': 'submittedDate', 'sortOrder': 'ascending', 'max_results': 50})
        if response.status_code != 200:
            return None
        for entry in re.findall(r'<entry>(.*?)</entry>', response.text, re.S):
            arxiv_id = re.search(r'<id>https?://arxiv.org/abs/([^<]+?)(v\d+)?</id>', entry)
            title = re.search(r'<title>(.*?)</title>', entry, re.S)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.S)
            year = re.search(r'<published>(\d{4})', entry)
            if arxiv_id and title and year and term in terms('%s. %s' % (title.group(1), summary.group(1) if summary else '')):
                found = {'arxiv_id': arxiv_id.group(1), 'title': ' '.join(title.group(1).split()), 'year': int(year.group(1))}
                break
    except requests.RequestException:
        logger.info('First use of %r not found', term)
        return None
    cache.set(key, found, 90 * 86400)
    return found or None


def _vector(vector):
    return base64.b64encode(vector.astype(np.float16).tobytes()).decode()


def vector_of(topic):
    return np.frombuffer(base64.b64decode(topic['vector']), dtype=np.float16).astype(np.float32) if topic.get('vector') else None


def result_state(scope, lang, week):
    from .models import RisingTopics
    content = RisingTopics.objects.filter(scope=scope, lang=lang, day=week).values_list('content', flat=True).first()
    return json.loads(content) if content else {}


def _save_results(scope, week, content_by_language, metadata):
    from .models import RisingTopics
    with transaction.atomic():
        for lang in LANGS:
            content = dict(content_by_language.get(lang, {'topics': []}), **metadata)
            RisingTopics.objects.update_or_create(scope=scope, lang=lang, day=week,
                                                 defaults={'content': json.dumps(content)})
    _RECENT.clear()


def write(scope, week, rising=None):
    """Publish a validated result, recording why empty results need another check.

    Empty statistical results are checked daily without an AI call. A model that
    rejects every candidate gets at most three attempts for the same signals.
    """
    now = timezone.now()
    previous = result_state(scope, LANGS[0], week)
    rising = rising if rising is not None else rising_terms(scope, week)
    signature = hashlib.sha256(json.dumps(rising, sort_keys=True).encode()).hexdigest()
    attempts = previous.get('attempts', 0) if previous.get('signature') == signature else 0
    metadata = {'checked_at': now.isoformat(), 'candidate_count': len(rising or []),
                'signature': signature, 'attempts': attempts,
                'next_retry_at': (now + datetime.timedelta(days=1)).isoformat()}
    if not rising:
        metadata['status'] = 'insufficient_data' if rising is None else 'no_signal'
        _save_results(scope, week, {}, metadata)
        return None
    if attempts >= 3 and previous.get('status') == 'filtered':
        _save_results(scope, week, {}, dict(metadata, status='filtered'))
        return None
    metadata['attempts'] = attempts + 1
    try:
        content = _write_topics(scope, week, rising)
    except Exception as exc:
        # A failed refresh must not erase an already published result.
        if not previous.get('topics'):
            _save_results(scope, week, {}, dict(metadata, status='failed', error_type=type(exc).__name__,
                next_retry_at=(now + datetime.timedelta(hours=min(24, 2 ** min(attempts, 5)))).isoformat()))
        raise
    if not content:
        _save_results(scope, week, {}, dict(metadata, status='filtered',
            next_retry_at=(now + datetime.timedelta(hours=6 if attempts < 2 else 24)).isoformat()))
        return None
    _save_results(scope, week, content, dict(metadata, status='ready', next_retry_at=None))
    return content[LANGS[0]]


def _write_topics(scope, week, rising):
    """Name only statistically selected signals; an empty model answer is not a no-signal result."""
    papers = stored_papers(week, scope)
    texts = {p['key']: terms('%s. %s' % (p['title'], p['abstract'])) for p in papers}
    by_key = {p['key']: p for p in papers}
    stats = {item['term']: item for item in rising}
    using = {term: [k for k, found in texts.items() if term in found] for term in stats}
    listing = '\n'.join('- "%s": %d papers this week (%d teams), x%.1f compared with the previous weeks. Examples: %s'
                        % (term, s['n'], s.get('teams', 0), s['growth'],
                           ' | '.join('%s [%s]' % (by_key[k]['title'], k) for k in using[term][:4]))
                        for term, s in stats.items())
    where = SERVER_NAMES.get(scope) or (SERVER_NAMES[archive_of(scope)] + ' ' + scope if archive_of(scope) in BIO_ARCHIVES
                                        else 'arXiv' + ('' if scope == '*' else ' (%s)' % scope))
    answer = generated.chat_json(
        'These terms are used in many more %s papers this week than usual. Group the ones that belong to the same '
        'research topic into 1 to 7 topics (fewer when there are few candidates); never put unrelated terms together, and leave out terms that are only noise '
        '(formatting, generic words, a conference or a template). For each topic write, in English and in French, a title '
        'a curious non-specialist understands and 2 or 3 sentences saying what it is about and why researchers may be '
        'publishing more on it now, without inventing events. Also give "query": the topic as a researcher would type it '
        'in a search engine, in English, 4 to 12 words. Give the terms it groups, copied exactly, and up to 3 of the '
        'example IDs. Never write IDs in the texts. Answer with a JSON object {"topics": [{"title": {"en": "...", '
        '"fr": "..."}, "text": {"en": "...", "fr": "..."}, "query": "...", "terms": ["..."], "papers": ["ID"]}]}.\n\n%s'
        % (where, listing), paid=True, max_tokens=5000)
    if not isinstance(answer.get('topics'), list):
        raise generated.GenerationError('Rising topics response has no topics list')
    topics = []
    for topic in answer['topics']:
        if not isinstance(topic, dict) or not isinstance(topic.get('title'), dict):
            continue
        grouped = sorted({t for t in topic.get('terms', []) if t in stats}, key=lambda t: -stats[t]['llr'])
        titles = {lang: str(topic['title'].get(lang) or '').strip() for lang in LANGS}
        if not grouped or not all(titles.values()):
            continue
        query = ' '.join(str(topic.get('query') or titles['en']).split())[:200]
        words = {fold(w) for w, _ in _words(query) if w not in STOP and w not in GENERIC and len(w) > 2}

        def closeness(key):
            # the papers using most of the topic's terms, then most of the words of its query (twice in the title)
            title = {fold(w) for w, _ in _words(by_key[key]['title'])}
            body = {fold(w) for w, _ in _words(by_key[key]['abstract'])}
            return (sum(t in texts[key] for t in grouped), sum(2 * (w in title) + (w in body) for w in words))
        keys = sorted({k for t in grouped for k in using[t]}, key=closeness, reverse=True)
        topics.append({
            'titles': titles, 'texts': {lang: str((topic.get('text') or {}).get(lang) or '').strip() for lang in LANGS},
            'query': query, 'terms': grouped, 'main': grouped[0],
            'stats': [stats[t] for t in grouped], 'keys': keys, 'chosen': [k for k in topic.get('papers', []) if k in by_key],
        })
    if not topics:
        if answer['topics']:
            raise generated.GenerationError('No valid bilingual topic matched the selected signals')
        return None
    main_history = history(scope, [t['main'] for t in topics], weeks=[week - WEEK * i for i in range(60)][::-1])
    for topic in topics:
        teams = team_of([by_key[k] for k in topic['keys']])
        points = main_history.get(topic['main'], [])
        topic.update(kind(points))
        topic['series'] = [round(n * 10000 / p, 1) for _, p, n in points[-(SERIES_WEEKS + 1):]]
        main = topic['stats'][0]
        topic['growth'], topic['count'] = main['growth'], len(topic['keys']) or main['n']
        topic['teams'] = len({teams[k] for k in topic['keys']}) if topic['keys'] else main.get('teams', 0)
        # examples: the ones the model chose, then the papers using most of the terms, one per team
        examples, seen = [], set()
        for key in topic.pop('chosen') + topic['keys']:
            if key in by_key and teams.get(key) not in seen and key not in [e['arxiv_id'] for e in examples]:
                seen.add(teams.get(key))
                examples.append({'arxiv_id': key, 'title': by_key[key]['title']})
            if len(examples) == 4:
                break
        topic['papers'] = examples
        topic['ids'] = topic.pop('keys')[:200]
        multiword = [t for t in topic['terms'] if ' ' in t]
        topic['first'] = first_use(multiword[0]) if multiword and archive_of(scope) not in BIO_ARCHIVES else None
    try:
        from . import similar
        vectors = similar.embed(['%s. %s %s' % (t['titles']['en'], t['texts']['en'], ', '.join(t['terms'])) for t in topics])
        for topic, vector in zip(topics, vectors):
            topic['vector'] = _vector(vector)
    except Exception:
        logger.exception('Rising topics of %s not embedded', scope)
    topics.sort(key=lambda t: -t['stats'][0]['llr'])
    return {lang: {'topics': [dict({k: v for k, v in t.items() if k not in ('titles', 'texts')},
                                    title=t['titles'][lang], text=t['texts'][lang]) for t in topics]}
            for lang in LANGS}


def due_scopes(week, now=None):
    """Unwritten or due empty results, including old empty rows without a status."""
    from .models import RisingTopics, TermWeek
    now = now or timezone.now()
    done = set()
    languages = {}
    for scope, lang, content in RisingTopics.objects.filter(day=week).values_list('scope', 'lang', 'content'):
        result = json.loads(content)
        retry_at = result.get('next_retry_at')
        waiting = bool(retry_at and datetime.datetime.fromisoformat(retry_at) > now)
        if result.get('topics') or waiting:
            languages.setdefault(scope, set()).add(lang)
    done = {scope for scope, langs in languages.items() if set(LANGS) <= langs}
    sizes = dict(TermWeek.objects.filter(week=week, part=0).values_list('scope', 'papers'))
    scopes = sorted((s for s, n in sizes.items() if n >= MIN_WEEK_PAPERS and s != '*'), key=lambda s: -sizes[s])
    return [s for s in (['*'] if '*' in sizes else []) + scopes if s not in done]


def update(now=None):
    """From the hourly digest run: count the weeks of submissions just completed, then write their topics."""
    from .models import RisingTopics, TermWeek, TrendPaper
    if not cache.add(LOCK, True, 3 * 3600):
        return 0
    try:
        now = now or timezone.now()
        today = now.astimezone(datetime.timezone.utc).date()
        target = closed_week(now)
        stored = set(TermWeek.objects.filter(scope='*', week__gt=target - 4 * WEEK).values_list('week', flat=True))
        missing = [target - WEEK * i for i in range(4) if target - WEEK * i not in stored]
        if missing:
            harvest(missing, today)
        TrendPaper.objects.filter(week__lt=target - WEEK * (KEEP_PAPERS - 1)).delete()
        TermWeek.objects.filter(week__lt=target - WEEK * HISTORY_WEEKS).delete()
        written, tried = 0, 0
        for scope in due_scopes(target, now):
            if tried >= SCOPES_PER_RUN:
                break
            tried += 1
            try:
                if write(scope, target):
                    written += 1
            except Exception:
                logger.exception('Rising topics of %s failed', scope)
        return written
    finally:
        cache.delete(LOCK)


# --- reading the topics ---------------------------------------------------------------------------------------

def topics_of(scope, lang, week=None, since=None):
    """(week, topics) of a scope: of that week, or the latest one written (from `since` on)."""
    from .models import RisingTopics
    rows = RisingTopics.objects.filter(scope=scope, lang=lang)
    if week is not None:
        rows = rows.filter(day=week)
    if since is not None:
        rows = rows.filter(day__gte=since)
    row = rows.order_by('-day').first()
    return (row.day, json.loads(row.content)['topics']) if row else (None, [])


def latest(scope, lang, since=None):
    return topics_of(scope, lang, since=since)


def for_digest(categories, lang, today):
    """For a Premium Monday digest: the first rising topics of its categories (or their archives) this week."""
    found, seen = [], set()
    since = today - 3 * WEEK
    for category in dict.fromkeys(categories):
        for scope in (category, archive_of(category)):
            _, topics = topics_of(scope, lang, since=since)
            if topics:
                for topic in topics[:2]:
                    if topic['title'] not in seen:
                        seen.add(topic['title'])
                        found.append(dict(topic, archive=scope))
                break
    return found[:4]


_RECENT = {}


def recent(lang):
    """(topics, papers) of the last weeks, for pages that look up many: [(scope, week, topic, vector)] and
    {paper key: (scope, week, topic)}; read again every 15 minutes."""
    from .models import RisingTopics
    entry = _RECENT.get(lang)
    if entry is None or time.time() - entry[0] > 900:
        topics, papers = [], {}
        since = timezone.now().date() - 4 * WEEK
        rows = RisingTopics.objects.filter(lang=lang, day__gte=since).order_by('day').values_list('scope', 'day', 'content')
        for scope, day, content in rows.iterator():
            for topic in json.loads(content).get('topics', []):
                short = {k: topic.get(k) for k in ('title', 'growth', 'kind', 'main', 'terms', 'count', 'teams')}
                topics.append((scope, day, short, vector_of(topic)))
                for paper in topic.get('ids', []):
                    # the latest week, then the most specific scope: a category over its archive over all of arXiv
                    previous = papers.get(paper)
                    if previous is None or previous[1] < day or len(scope) > len(previous[0]):
                        papers[paper] = (scope, day, short)
        entry = _RECENT[lang] = (time.time(), topics, papers)
    return entry[1], entry[2]


def topic_of_paper(arxiv_id, lang):
    """(scope, week, topic) of a recent rising topic this paper belongs to, or None."""
    return recent(lang)[1].get(re.sub(r'v\d+$', '', arxiv_id))


def close_topics(vector, lang, threshold=0.5, limit=3):
    """[(similarity, scope, week, topic)] of the recent topics closest to an embedding (a project, a research profile)."""
    found = {}
    for scope, week, topic, v in recent(lang)[0]:
        if v is None or v.shape != vector.shape:
            continue
        score = float(v @ vector)
        if score >= threshold and score > found.get(topic['main'], (0,))[0]:
            found[topic['main']] = (score, scope, week, topic)
    return sorted(found.values(), key=lambda f: -f[0])[:limit]
