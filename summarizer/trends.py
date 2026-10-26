"""Rising topics: the terms arXiv papers use much more this week than in the weeks before.

Every daily announcement is counted once: for each term (1 to 3 words) of the
titles and abstracts, how many papers use it, over all of arXiv ('*') and per
archive (cs, astro-ph...). Once a week, a term rises when this week's share of
papers using it is well above its share over the previous four weeks. The
model then groups the rising terms into a few topics and names them in plain
language, with example papers. It never decides what rises: the counts do.
"""
import datetime
import json
import logging
import math
import re

from django.core.cache import cache
from django.http import Http404
from django.shortcuts import render
from django.utils import timezone
from django.utils.translation import get_language

from . import access, generated
from .digest import ARCHIVES, BIO_ARCHIVES, JOURNALS, SERVER_NAMES, archive_of
from .models import AnnouncedPaper, ArxivBatch, RisingTopics, TermDay

logger = logging.getLogger(__name__)

WEEK = datetime.timedelta(days=7)
BASELINE = datetime.timedelta(days=28)
MIN_BASELINE_DAYS = 8  # announcements needed before the week, so a rise means something
MIN_WEEK_PAPERS = 60  # smaller archives have too few papers for a weekly trend
MIN_COUNT = {'*': 15}
MIN_COUNT_ARCHIVE = 6
MIN_GROWTH = 2.0
CANDIDATES = 30
KEEP = datetime.timedelta(days=45)  # about 0.7 MB of counts per announcement
SCOPES_PER_RUN = 6
FREE_TOPICS = 3
LANGS = ('en', 'fr')
LOCK = 'trends-run'

STOP = set('''a about above across after again against all almost along also although always am among an and another any
are around as at available based be because been before being below between both but by can cannot could did do does
done due during each either else enough especially et etc even ever every few first for found from further furthermore
general given has have having here however i if in including into is it its itself just known large last least less like
made main many may me more moreover most much must near nearly new no nor not novel now of often on once one only onto or
other others otherwise our ours out over overall own paper papers per perhaps present presented proposed propose proposes
provide provides rather recent recently respectively result results same several shall show shows shown significant
significantly since so some such than that the their them then there thereby therefore these they this those though
three through thus to together too toward towards two under unlike until up upon us use used uses using various very via
was way we well were what when where whereas whether which while who whom whose why will with within without work works
would yet approach approaches method methods study studies demonstrate demonstrates obtain obtained find finds analysis
model models data first second third high low different important problem problems also existing state art based
framework present novel effective efficient performance task tasks introduce introduces able achieve achieves'''.split())
WORD = re.compile(r"[a-z][a-z0-9\-]*[a-z0-9]")


def terms(text):
    """The distinct 1- to 3-word terms of a title and abstract."""
    text = re.sub(r'\$[^$]*\$', ' ', text.lower())
    found = set()
    # a term never spans two sentences or clauses
    for chunk in re.split(r'[.;:!?()\[\],]+(?:\s|$)|[;:!?()\[\]]', text):
        words = WORD.findall(chunk)
        for n in (1, 2, 3):
            for i in range(len(words) - n + 1):
                gram = words[i:i + n]
                if gram[0] in STOP or gram[-1] in STOP:
                    continue
                if n == 1 and (len(gram[0]) < 4 or '-' not in gram[0] and len(gram[0]) < 5):
                    continue
                found.add(' '.join(gram))
    return found


def count_day(day):
    """Store the term counts of one announcement, if its papers are still kept."""
    papers = list(AnnouncedPaper.objects.filter(announced=day).values_list('title', 'abstract', 'categories'))
    if not papers:
        return False
    scopes = {}
    for title, abstract, categories in papers:
        own = {archive_of(c) for c in categories.split(',') if c}
        if own == {JOURNALS}:
            continue  # the rising topics are arXiv's, bioRxiv's and medRxiv's
        found = terms('%s. %s' % (title, abstract))
        # '*' stays all of arXiv: bioRxiv and medRxiv papers only count in their own scope
        for scope in own | ({'*'} if not own or not own.issubset(BIO_ARCHIVES) else set()):
            totals = scopes.setdefault(scope, [0, {}])
            totals[0] += 1
            for term in found:
                totals[1][term] = totals[1].get(term, 0) + 1
    for scope, (count, counts) in scopes.items():
        floor = 3 if scope == '*' else 2
        kept = {t: n for t, n in counts.items() if n >= floor}
        TermDay.objects.update_or_create(day=day, scope=scope, defaults={'papers': count, 'counts': json.dumps(kept)})
    return True


def count_missing():
    from .digest import KEEP_DAYS
    recent = timezone.now().date() - datetime.timedelta(days=KEEP_DAYS + 2)  # older papers are not kept
    stored = set(TermDay.objects.filter(scope='*', day__gte=recent).values_list('day', flat=True))
    days = [d for d in ArxivBatch.objects.filter(day__gte=recent).values_list('day', flat=True) if d not in stored]
    return [d for d in sorted(days) if count_day(d)]


def inside(short, long):
    return short != long and (' %s ' % short) in (' %s ' % long)


def _sum(rows):
    papers, counts = 0, {}
    for row in rows:
        papers += row.papers
        for term, n in json.loads(row.counts).items():
            counts[term] = counts.get(term, 0) + n
    return papers, counts


def rising_terms(scope, day):
    """[(term, this week's papers, growth)] of the terms rising in the week up to `day`; None without history."""
    rows = list(TermDay.objects.filter(scope=scope, day__gt=day - WEEK - BASELINE, day__lte=day))
    week = [r for r in rows if r.day > day - WEEK]
    base = [r for r in rows if r.day <= day - WEEK]
    if len(base) < MIN_BASELINE_DAYS or not week:
        return None
    week_papers, week_counts = _sum(week)
    base_papers, base_counts = _sum(base)
    if week_papers < MIN_WEEK_PAPERS:
        return None
    floor = MIN_COUNT.get(scope, MIN_COUNT_ARCHIVE)
    rising = []
    for term, n in week_counts.items():
        if n < floor:
            continue
        expected = base_counts.get(term, 0) / base_papers * week_papers
        growth = (n + 1) / (expected + 1)
        if growth >= MIN_GROWTH:
            rising.append((term, n, growth))
    rising.sort(key=lambda t: -t[2] * math.log(t[1]))
    kept = []
    for term, n, growth in rising:
        # 'large language' and 'language models' when 'large language models' is there with nearly the same count
        if any(inside(term, longer) and n <= other * 1.25 for longer, other, _ in kept):
            continue
        kept = [(t, m, g) for t, m, g in kept if not (inside(t, term) and m <= n * 1.25)]
        kept.append((term, n, growth))
        if len(kept) == CANDIDATES:
            break
    return kept


def examples(scope, day, term, limit=4):
    papers = AnnouncedPaper.objects.filter(announced__gt=day - WEEK, announced__lte=day)
    found = []
    pattern = re.compile(r'\b%s\b' % re.escape(term))
    for arxiv_id, title, abstract, categories in papers.values_list('arxiv_id', 'title', 'abstract', 'categories').iterator():
        if scope != '*' and scope not in {archive_of(c) for c in categories.split(',') if c}:
            continue
        if pattern.search(re.sub(r'\s+', ' ', ('%s. %s' % (title, abstract)).lower())):
            found.append({'arxiv_id': arxiv_id, 'title': title})
            if len(found) == limit:
                break
    return found


def write(scope, day, lang, rising=None):
    """Name the rising topics of a scope in one language and store them."""
    rising = rising if rising is not None else rising_terms(scope, day)
    if not rising:
        return None
    samples = {term: examples(scope, day, term) for term, _, _ in rising}
    listing = '\n'.join('- "%s": %d papers this week, x%.1f compared with the previous weeks. Examples: %s'
                        % (term, n, growth, ' | '.join('%s [%s]' % (p['title'], p['arxiv_id']) for p in samples[term]))
                        for term, n, growth in rising)
    answer = generated.chat_json(
        'These terms are used in many more %s papers this week than usual%s. Group the ones that belong together '
        'into 3 to 7 topics; leave out terms that are only noise (formatting, generic words). For each topic, in %s, '
        'write a title a curious non-specialist understands and 2 or 3 sentences saying what it is about and why '
        'researchers may be publishing more on it now, without inventing events. Give the terms it groups, copied '
        'exactly, and up to 3 of the example IDs. Never write IDs in the texts. Answer with a JSON object '
        '{"topics": [{"title": "...", "text": "...", "terms": ["..."], "papers": ["ID"]}]}.\n\n%s'
        % (SERVER_NAMES.get(scope, 'arXiv'), '' if scope == '*' or scope in BIO_ARCHIVES else ' in the arXiv archive %s' % scope,
           generated.language_name(lang), listing),
        paid=True, max_tokens=3000)
    stats = {term: (n, growth) for term, n, growth in rising}
    titles = {p['arxiv_id']: p['title'] for ps in samples.values() for p in ps}
    topics = []
    for topic in answer.get('topics', []):
        if not isinstance(topic, dict) or not topic.get('title'):
            continue
        grouped = [t for t in topic.get('terms', []) if t in stats]
        if not grouped:
            continue
        topics.append({
            'title': str(topic['title']).strip(), 'text': str(topic.get('text', '')).strip(), 'terms': grouped,
            'count': max(stats[t][0] for t in grouped), 'growth': round(max(stats[t][1] for t in grouped), 1),
            'papers': [{'arxiv_id': a, 'title': titles[a]} for a in topic.get('papers', []) if a in titles][:3],
        })
    topics.sort(key=lambda t: -t['growth'])
    if not topics:
        return None
    content = {'topics': topics}
    RisingTopics.objects.update_or_create(scope=scope, lang=lang, day=day, defaults={'content': json.dumps(content)})
    return content


def due_scopes(day):
    """Scopes whose topics of the week up to `day` are not written yet, all of arXiv first."""
    done = set(RisingTopics.objects.filter(day=day).values_list('scope', 'lang'))
    sizes = {r.scope: r.papers for r in TermDay.objects.filter(day=day)}
    scopes = ['*'] + sorted((a for a in ARCHIVES + BIO_ARCHIVES if sizes.get(a, 0) * 5 >= MIN_WEEK_PAPERS),
                            key=lambda a: -sizes[a])
    return [(s, lang) for s in scopes for lang in LANGS if (s, lang) not in done]


def update(now=None):
    """From the hourly digest run: count new announcements and, once a week, write the week's topics."""
    if not cache.add(LOCK, True, 3600):
        return 0
    try:
        count_missing()
        TermDay.objects.filter(day__lt=(now or timezone.now()).date() - KEEP).delete()
        day = TermDay.objects.filter(scope='*').order_by('-day').values_list('day', flat=True).first()
        if day is None:
            return 0
        # once a week: finish the last week's scopes, or start a new week seven days after it
        last = RisingTopics.objects.filter(scope='*').order_by('-day').values_list('day', flat=True).first()
        target = day if last is None or day - last >= WEEK else last
        if TermDay.objects.filter(scope='*', day__gt=target - WEEK - BASELINE, day__lte=target - WEEK).count() < MIN_BASELINE_DAYS:
            return 0
        written, cached = 0, {}
        for scope, lang in due_scopes(target):
            if written >= SCOPES_PER_RUN:
                break
            if scope not in cached:
                cached[scope] = rising_terms(scope, target)
            if not cached[scope]:
                continue
            try:
                if write(scope, target, lang, cached[scope]):
                    written += 1
            except Exception:
                logger.exception('Rising topics of %s failed', scope)
        return written
    finally:
        cache.delete(LOCK)


def latest(scope, lang, since=None):
    rows = RisingTopics.objects.filter(scope=scope, lang=lang)
    if since is not None:
        rows = rows.filter(day__gte=since)
    row = rows.order_by('-day').first()
    return (row.day, json.loads(row.content)['topics']) if row else (None, [])


def for_digest(categories, lang, today):
    """For a Premium Monday digest: the first rising topics of its archives this week."""
    found = []
    for archive in dict.fromkeys(archive_of(c) for c in categories):
        _, topics = latest(archive, lang, since=today - WEEK)
        found += [dict(t, archive=archive) for t in topics[:2]]
    return found[:4]


def trends_page(request):
    paid = access.has_tool(request.user, 'trends')
    lang = get_language()
    scope = request.GET.get('scope', '*')
    if scope != '*' and scope not in ARCHIVES + BIO_ARCHIVES:
        raise Http404
    if scope != '*' and not paid:
        scope = '*'
    day, topics = latest(scope, lang)
    available = sorted(set(RisingTopics.objects.filter(lang=lang).exclude(scope='*').values_list('scope', flat=True)))
    first = TermDay.objects.filter(scope='*').order_by('day').values_list('day', flat=True).first()
    return render(request, 'summarizer/trends.html', {
        'paid': paid, 'scope': scope, 'day': day, 'topics': topics if paid else topics[:FREE_TOPICS],
        'hidden': 0 if paid else max(0, len(topics) - FREE_TOPICS), 'scopes': available,
        'since': first, 'ready_on': first + WEEK + datetime.timedelta(days=14) if first else None,
    })
