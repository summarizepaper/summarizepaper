"""Whether the language model may read a paper's full text, apart from whether we may show it again.

Reading a paper to summarize it and answer questions about it is text and data
mining. Showing the paper's own figures or passages again is redistribution,
which only an open license allows (sources.is_open_license). A license that
does not let us redistribute (arXiv's own, CC BY-ND...) does not by itself
forbid mining: the EU exception (directive 2019/790 art. 4, CPI L122-5-3)
allows it, commercial use included, on a copy lawfully accessed, unless the
rights holder reserved it in a machine-readable way.

Four answers:

    allowed                    the license lets us build on the paper (CC BY, BY-SA, BY-NC, BY-NC-SA, CC0, public domain)
    eligible_under_exception   no such license, but the exception applies: the license is known, the copy is open to
                               anyone (never behind a login or a paywall), the terms of the site it comes from were read,
                               and neither the site (TDMRep file, robots.txt) nor the document (TDMRep header or meta)
                               reserves mining
    blocked                    the site or the document reserves mining
    review_required            the rights are not established: no license, a publisher's own terms, a copy elsewhere
                               than the site whose terms we read, or the site could not be asked. Not reading until
                               someone decides: finding no reservation is not a permission.

Someone decides in the admin, site by site, after reading its terms (TdmDecision): its copies without
an established license are then read under the exception (reservations still checked), or never read.
Each summary counts the answer it got (TdmCount), shown in the admin next to what the AI costs.

Only the first two are read. A text read under the exception is never shown
back beyond short quotes (assistant.QUOTE_NOTE), and is forgotten KEEP_DAYS
after it was read (forget_texts): the paper is read again, from its own
source, when it is asked about. PDFs are read in memory and never written
anywhere. What the robot does is told publicly at /tdm/ (templates/summarizer/tdm.html).
"""
import datetime
import fnmatch
import logging
import re
import urllib.parse
import urllib.robotparser

import requests
from django.conf import settings
from django.core.cache import cache
from django.utils import timezone

logger = logging.getLogger(__name__)

ALLOWED = 'allowed'
EXCEPTION = 'eligible_under_exception'
BLOCKED = 'blocked'
REVIEW = 'review_required'
READABLE = (ALLOWED, EXCEPTION)
RANK = {ALLOWED: 0, EXCEPTION: 1, REVIEW: 2, BLOCKED: 3}  # which copy of a paper to read first

# sites whose terms of use we read and which do not forbid mining (checked 2026-10-31): arXiv's API terms let
# anyone retrieve and use e-prints and forbid storing and serving them again, which we never do
TERMS_READ = ('arxiv.org',)
ARXIV_URLS = ('https://arxiv.org/html/', 'https://arxiv.org/pdf/')
ROBOTS_AGENT = 'SummarizePaper'
POLICY_KEY = 'tdm-policy:%s'
POLICY_SECONDS = 2 * 86400  # arXiv's is asked again every hour (run): page loads, which never ask, always know it
MAX_ROBOTS = 20000  # characters kept of the groups of a robots.txt that apply to us (the cache is small)
TIMEOUT = 10
KEEP_DAYS = 30
DECISIONS_KEY = 'tdm-decisions'
DECISIONS_SECONDS = 600  # TdmDecision.save clears it; a bulk delete in the admin is seen within that time
META_RE = re.compile(r'<meta\s[^>]*name=["\']tdm-reservation["\'][^>]*>', re.I)
CONTENT_RE = re.compile(r'content=["\']\s*1\s*["\']', re.I)


def license_status(url):
    """What the license alone says: allowed, eligible_under_exception (if the source agrees) or review_required."""
    from . import sources
    if sources.is_open_license(url):
        return ALLOWED
    lowered = (url or '').lower()
    if 'arxiv.org/licenses/' in lowered:
        return EXCEPTION  # distribution by arXiv only: says nothing against reading
    if 'creativecommons.org' in lowered and sources.license_code(url)[0] in ('by-nd', 'by-nc-nd'):
        return EXCEPTION  # copies allowed, adaptations not: our summary is our own text, not an adaptation
    return REVIEW


def _host_in(host, domains):
    return any(host == d or host.endswith('.' + d) for d in domains)


def _policy(origin, ask=True, refresh=False):
    """{'tdmrep': rules of /.well-known/tdmrep.json, 'robots': robots.txt} of a site; None when it could not be asked
    (or, without ask, when its answer is not in the cache). refresh asks again, keeping the cached answer on failure."""
    from . import sources
    key = POLICY_KEY % origin
    found = None if refresh else cache.get(key)
    if found is not None:
        return found
    if not ask or not settings.TDM_CHECK_NETWORK:
        return None
    try:
        tdmrep = sources._session.get(origin + '/.well-known/tdmrep.json', timeout=TIMEOUT)
        robots = sources._session.get(origin + '/robots.txt', timeout=TIMEOUT)
    except requests.RequestException:
        logger.info('TDM policy of %s could not be read', origin, exc_info=True)
        return None
    if tdmrep.status_code >= 500 or robots.status_code >= 500 or 429 in (tdmrep.status_code, robots.status_code):
        logger.info('TDM policy of %s: HTTP %s, %s', origin, tdmrep.status_code, robots.status_code)
        return None
    rules = []
    if tdmrep.status_code == 200:
        try:
            rules = [rule for rule in tdmrep.json() if isinstance(rule, dict)]
        except (ValueError, TypeError):
            logger.warning('TDM policy of %s: tdmrep.json is not a list of rules', origin)
            return None  # a file we cannot understand may say no
    # a robots.txt that is missing or refused (4xx) puts no limit (RFC 9309)
    found = {'tdmrep': rules, 'robots': _our_rules(robots.text) if robots.status_code == 200 else ''}
    cache.set(key, found, POLICY_SECONDS)
    return found


def _our_rules(robots):
    """The groups of a robots.txt that apply to us, as urllib.robotparser picks them (our agent's, or every agent's):
    publishers' files list hundreds of other robots, and the cache is small."""
    kept, group, ours, agents = [], [], False, False
    for line in robots.splitlines():
        field = line.split('#', 1)[0].split(':', 1)
        name = field[0].strip().lower() if len(field) == 2 else ''
        if name == 'user-agent':
            if not agents:  # a new group starts
                kept += group if ours else []
                group, ours = [], False
            agents = True
            agent = field[1].strip().lower()
            ours = ours or agent == '*' or (agent != '' and agent in ROBOTS_AGENT.lower())
        elif name:
            agents = False
        group.append(line)
    kept += group if ours else []
    return '\n'.join(kept)[:MAX_ROBOTS]


def site_reserves(url, ask=True):
    """True when the site reserves mining of this URL (TDMRep, or robots.txt keeps us out), False when it does not,
    None when it could not be asked."""
    parts = urllib.parse.urlsplit(url)
    policy = _policy('%s://%s' % (parts.scheme, parts.netloc), ask)
    if policy is None:
        return None
    path = parts.path or '/'
    for rule in policy['tdmrep']:  # the first rule whose location matches applies
        location = urllib.parse.urlsplit(str(rule.get('location') or '')).path or '/'
        if fnmatch.fnmatchcase(path, location if location.endswith('*') else location + '*'):
            return str(rule.get('tdm-reservation', '')).strip() == '1'
    robots = urllib.robotparser.RobotFileParser()
    robots.parse(policy['robots'].splitlines())
    return not robots.can_fetch(ROBOTS_AGENT, url)


def document_reserves(headers, html=''):
    """Whether a document reserves mining itself: a 'tdm-reservation: 1' header, or the same in an HTML meta tag."""
    try:
        if str(headers.get('tdm-reservation', '')).strip() == '1':
            return True
    except AttributeError:
        pass
    return any(CONTENT_RE.search(tag) for tag in META_RE.findall((html or '')[:200000]))


def decision(host):
    """What was decided in the admin for a site (TdmDecision): eligible_under_exception, blocked or None."""
    from .models import TdmDecision
    decisions = cache.get(DECISIONS_KEY)
    if decisions is None:
        decisions = dict(TdmDecision.objects.values_list('domain', 'status'))
        cache.set(DECISIONS_KEY, decisions, DECISIONS_SECONDS)
    host = (host or '').lower().removeprefix('www.')
    # the most specific domain first: a decision about a subdomain wins over its parent's
    return next((decisions[d] for d in sorted(decisions, key=len, reverse=True) if _host_in(host, (d,))), None)


def status(license_url, url, ask=True):
    """The answer for reading the copy at url of a paper under that license. Without ask, a site whose answer is not
    in the cache is not asked (page loads): review_required."""
    level = license_status(license_url)
    host = (urllib.parse.urlsplit(url).hostname or '').lower()
    decided = decision(host)
    if decided == BLOCKED:
        return BLOCKED
    if level == REVIEW and decided == EXCEPTION:
        level = EXCEPTION  # someone read the site's terms
    if level != EXCEPTION:
        return level
    if 'arxiv.org/licenses/' in (license_url or '').lower() and not _host_in(host, TERMS_READ) and not decided:
        return REVIEW  # arXiv's license is about arXiv's copy
    reserved = site_reserves(url, ask)
    if reserved is None:
        return REVIEW
    return BLOCKED if reserved else EXCEPTION


def arxiv_status(license_url, ask=True):
    """The answer for an arXiv paper read from arXiv (its HTML, else its PDF): the strictest of the two."""
    answers = [status(license_url, url, ask) for url in ARXIV_URLS]
    return next((a for a in (BLOCKED, REVIEW, EXCEPTION) if a in answers), ALLOWED)


def paper_status(license_url, key, ask=True):
    """The answer for reading the full text of a paper (an arXiv id or a sources key) under that license. A paper from
    another source has several copies, each checked where it is read (sources.full_text): without a license that
    lets us read it, the best of its copies (its arXiv version, a site someone decided about)."""
    from . import sources
    if not sources.is_external(key):
        return arxiv_status(license_url, ask)
    level = license_status(license_url)
    if level in READABLE:
        return level
    copies = sources.copy_statuses(key, ask)
    return min(copies, key=RANK.get) if copies else level


def may_read(license_url, key, ask=True):
    """Whether the model may read the full text of a paper (see paper_status)."""
    return paper_status(license_url, key, ask) in READABLE


def record(answer, day=None):
    """Count a summary under the answer it got (TdmCount). Never breaks the summary."""
    from django.db.models import F
    from .models import TdmCount
    day = day or timezone.localdate()
    try:
        if not TdmCount.objects.filter(day=day, status=answer).update(count=F('count') + 1):
            TdmCount.objects.get_or_create(day=day, status=answer)
            TdmCount.objects.filter(day=day, status=answer).update(count=F('count') + 1)
    except Exception:
        logger.exception('TDM count of %s could not be saved', answer)


def run(now):
    """From the hourly digest run: ask arXiv again before its cached answer expires, and forget the texts kept long
    enough. Returns how many were forgotten."""
    _policy('https://arxiv.org', refresh=True)
    return forget_texts(now)


def forget_texts(now=None):
    """Forget the full texts of papers without an open license KEEP_DAYS after they were read, whatever let us read
    them (their license, a site decision): the stored text, and the chunks of an index from before PaperText, which
    hold the paper as well (paper_source.paper_text reads the paper again when it is asked about). Returns how many
    papers were forgotten."""
    from .models import ArxivPaper, PaperText, PickledData
    cutoff = (now or timezone.now()) - datetime.timedelta(days=KEEP_DAYS)
    read = set(PaperText.objects.filter(updated__lt=cutoff).exclude(source__in=('abstract', 'forgotten'))
               .values_list('arxiv_id', flat=True))
    indexed = set(PickledData.objects.values_list('arxiv_id', flat=True))  # none was made in the last KEEP_DAYS
    keys = sorted(read | indexed)
    licenses = {}
    for start in range(0, len(keys), 500):
        licenses.update(ArxivPaper.objects.filter(arxiv_id__in=keys[start:start + 500]).values_list('arxiv_id', 'license'))
    stale = [key for key in keys if license_status(licenses.get(key)) != ALLOWED]
    forgotten = 0
    for start in range(0, len(stale), 500):
        chunk = stale[start:start + 500]
        texts = set(PaperText.objects.filter(arxiv_id__in=chunk).values_list('arxiv_id', flat=True))
        forgotten += PaperText.objects.filter(arxiv_id__in=chunk, updated__lt=cutoff).exclude(
            source__in=('abstract', 'forgotten')).update(text='', source='forgotten')
        # a paper known only by its index stays known as read (utils.has_paper_text), and is read again when asked about
        alone = [key for key in chunk if key not in texts]
        PaperText.objects.bulk_create([PaperText(arxiv_id=key, text='', source='forgotten') for key in alone])
        forgotten += len(alone)
        PickledData.objects.filter(arxiv_id__in=chunk).delete()
    return forgotten
