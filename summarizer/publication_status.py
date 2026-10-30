"""Sourced editorial checks, without model inference or publisher-page scraping."""
import datetime
import html
import json
import logging
import re
import time
import unicodedata
import requests
from urllib.parse import quote
from xml.etree import ElementTree

from django.core.cache import cache
from django.utils import timezone
from django.utils.translation import gettext as _
from pylatexenc.latex2text import LatexNodes2Text

from . import digest, generated, sources
from .models import ArxivPaper, FollowedPaper, Generated, PaperText

logger = logging.getLogger(__name__)

KIND = 'presskit_status'
TTL = datetime.timedelta(days=1)


def normalized(value):
    text = LatexNodes2Text().latex_to_text(html.unescape(str(value or '')))
    text = text.replace('_', '').replace('^', '')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def latest(paper_id, kit=None):
    row = Generated.objects.filter(kind=KIND, key=paper_id, lang='en').first()
    report = (row.data() if row and row.content else None) or (kit or {}).get('publication_status')
    return with_peer_review(paper_id, report)


# Verified policy, limited to article types explicitly covered by this journal.
# Editorials, News & Views and a journal name declared on arXiv are insufficient.
NATURE_ASTRONOMY_POLICY = 'https://www.nature.com/natastron/editorial-policies/peer-review'


def with_peer_review(paper_id, report):
    if not report or report.get('status') != 'published':
        return report
    doi = sources.clean_doi(report.get('doi', ''))
    if normalized(report.get('journal')) != 'nature astronomy' or not re.fullmatch(r'10\.1038/s41550-\d{3,4}-\d{4,6}-[a-z0-9]+', doi):
        return report
    row = PaperText.objects.filter(arxiv_id=paper_id, source='pdf').first()
    if row is None:
        return report
    # Match the publisher's masthead, not a type word in our metadata or prose.
    text = row.text
    marker = re.search(r'(?im)^\s*nature astronomy\s*\n+\s*(Article|Analysis|Review|Perspective|Matters Arising)\s*\n+\s*https://doi\.org/([^\s]+)', text[:3000])
    if not marker or sources.clean_doi(marker.group(2)) != doi:
        return report
    report = dict(report, peer_review='journal_policy', article_type=marker.group(1),
                  peer_review_policy=NATURE_ASTRONOMY_POLICY)
    report['sources'] = list(report.get('sources') or [])
    if not any(source.get('kind') == 'peer_review_policy' for source in report['sources']):
        report['sources'].append({'name': 'Nature Astronomy · peer-review policy',
            'url': NATURE_ASTRONOMY_POLICY, 'kind': 'peer_review_policy',
            'detail': peer_review_text(report)})
    return report


def peer_review_text(report):
    if (report or {}).get('peer_review') == 'journal_policy':
        return _("Peer-reviewed article, according to the journal's policy for this article type.")
    return _('Peer-review status could not be determined from the available information.')


def stale(report, age=TTL):
    try:
        return timezone.now() - datetime.datetime.fromisoformat(report['checked_at']) > age
    except (TypeError, KeyError, ValueError):
        return True


def crossref(doi):
    body = sources._json(sources.CROSSREF % quote(doi, safe=''), {'mailto': 'contact@summarizepaper.com'})
    work = (body or {}).get('message') if isinstance(body, dict) else None
    return work if isinstance(work, dict) else {}


def publication_date(work):
    for field in ['published-online', 'published-print', 'published', 'issued']:
        parts = (work.get(field) or {}).get('date-parts') or []
        if parts and parts[0]:
            try:
                datetime.date(*[int(n) for n in (parts[0] + [1, 1])[:3]])
            except (ValueError, TypeError):
                continue
            return '-'.join(str(n).zfill(4 if i == 0 else 2) for i, n in enumerate(parts[0][:3]))
    return ''


def accepted_date(work):
    """'2023-05-25' when Crossref has the acceptance date, from the field (A&A) or a publisher assertion (AAS, Springer)."""
    parts = (work.get('accepted') or {}).get('date-parts') or []
    values = ['-'.join(str(n) for n in parts[0])] if parts and parts[0] and len(parts[0]) == 3 else []
    values += [str(a.get('value') or '') for a in work.get('assertion') or [] if a.get('name') in ('accepted', 'date_accepted')]
    for value in values:
        for form in ('%Y-%m-%d', '%d %B %Y'):
            try:
                return datetime.datetime.strptime(value.strip(), form).date().isoformat()
            except ValueError:
                continue
    return ''


def lookup(paper):
    report = {'checked_at': timezone.now().isoformat(), 'status': 'unknown', 'doi': '', 'journal': '',
              'date': '', 'sources': [], 'checked_sources': [], 'unavailable': [], 'peer_review': 'not_independently_verified'}
    doi = sources.clean_doi(paper.link_doi or '') or sources.key_doi(paper.arxiv_id)
    comment = ''
    if not sources.is_external(paper.arxiv_id):
        base = re.sub(r'v\d+$', '', paper.arxiv_id).replace('--', '/')
        report['checked_sources'].append('arXiv')
        try:
            response = sources._get('https://export.arxiv.org/api/query', {'id_list': base, 'max_results': 1}, timeout=12)
            response.raise_for_status()
            entries = digest.parse_feed(response.text)
            record = next((p for p in entries if re.sub(r'v\d+$', '', p['arxiv_id']).replace('--', '/') == base), None)
            if record:
                doi = sources.clean_doi(record.get('doi', '')) or doi
                comment = record.get('comment', '')[:2000]
                report['arxiv_doi'], report['journal_ref'] = doi or '', record.get('journal_ref', '')[:200]
                report['sources'].append({'name': 'arXiv', 'url': 'https://arxiv.org/abs/' + base,
                    'detail': ' · '.join(filter(None, [record.get('journal_ref', '')[:600], comment])) or _('Preprint record found.'), 'kind': 'author_metadata'})
                report['status'] = 'journal_reference' if record.get('journal_ref') else 'preprint'
                report['journal'] = record.get('journal_ref', '')
        except (sources.SourceUnavailable, ElementTree.ParseError, ValueError, requests.RequestException):
            report['unavailable'].append('arXiv')
    # A repository DOI identifies the preprint; it does not prove journal publication.
    if doi and not sources.ARXIV_DOI_RE.match(doi):
        report['checked_sources'].append('Crossref')
        try:
            work = crossref(doi)
            if work.get('type') == 'posted-content':
                report['status'] = 'preprint'
                report['sources'].append({'name': 'Crossref', 'url': 'https://doi.org/' + doi,
                    'detail': _('Preprint record found.'), 'kind': 'preprint_metadata'})
            # Follow only an explicit preprint-to-publication relation (at most one extra lookup).
            related = (work.get('relation') or {}).get('is-preprint-of') or []
            linked = next((sources.clean_doi(r.get('id', '')) for r in related if r.get('id-type') == 'doi'), '')
            if linked:
                report['sources'].append({'name': 'Crossref', 'url': 'https://doi.org/' + doi,
                    'detail': _('Explicit link from the preprint to its published version.'), 'kind': 'relation'})
                work = crossref(linked)
            if apply_work(report, work):
                return report
        except (sources.SourceUnavailable, ValueError, TypeError, AttributeError, KeyError):
            report['unavailable'].append('Crossref')
    # Strict title + author match only. Similar titles must never become a publication claim.
    authors = list(paper.authors.values_list('name', flat=True))
    if paper.title and authors:
        if 'Crossref' not in report['checked_sources']:
            report['checked_sources'].append('Crossref')
        try:
            body = sources._json('https://api.crossref.org/works', {'query.title': paper.title[:300],
                'query.author': authors[0][:120], 'rows': 3, 'mailto': 'contact@summarizepaper.com'})
            for work in ((body or {}).get('message') or {}).get('items', []):
                title = (work.get('title') or [''])[0]
                families = {normalized(a.get('family', '')).split()[-1] for a in work.get('author', []) if normalized(a.get('family', ''))}
                own = {normalized(a).split()[-1] for a in authors if normalized(a)}
                wanted = normalized(paper.title)
                if len(wanted) >= 25 and normalized(title) == wanted and families & own and apply_work(report, work):
                    return report
        except (sources.SourceUnavailable, ValueError, TypeError, AttributeError, KeyError):
            if 'Crossref' not in report['unavailable']:
                report['unavailable'].append('Crossref')
    # These are declarations on the preprint, explicitly labelled as such, not independent checks.
    if re.search(r'\bwithdrawn\b|retir[ée]', comment, re.I) and not re.search(r'not withdrawn|non retir', comment, re.I):
        report['status'] = 'withdrawn_declared'
    elif re.search(r'\baccepted\b|accept[ée]', comment, re.I) and not re.search(r'not accepted|non accept|not yet accepted', comment, re.I):
        report['status'] = 'accepted_declared'
    elif re.search(r'\bsubmitted\b|\bunder review\b|soumis', comment, re.I):
        report['status'] = 'submitted_declared'
    return report


NOTICES = ('retraction', 'withdrawal', 'removal', 'expression_of_concern', 'correction', 'erratum', 'corrigendum')


def notices(work):
    """[{type, doi}] of the retractions, corrections and expressions of concern Crossref links to an article."""
    found = []
    for update in work.get('updated-by') or []:
        kind = str((update or {}).get('type') or '').lower().replace('-', '_').replace(' ', '_')
        if kind in NOTICES:
            found.append({'type': kind, 'doi': sources.clean_doi(str(update.get('DOI') or '')) or ''})
    return found


def apply_work(report, work):
    doi = sources.clean_doi(work.get('DOI', ''))
    journal = html.unescape((work.get('container-title') or [''])[0])
    date = publication_date(work)
    if not doi or not journal or work.get('type') not in ('journal-article', 'proceedings-article') or not date:
        return False
    report.update(status='forthcoming' if date > timezone.now().date().isoformat()[:len(date)] else 'published',
                  doi=doi, journal=journal, date=date, accepted=accepted_date(work), updates=notices(work))
    report['sources'].append({'name': 'Crossref', 'url': 'https://doi.org/' + doi,
                             'detail': journal + ' · ' + date, 'kind': 'publisher_metadata'})
    return True


def for_generation(paper):
    report = latest(paper.arxiv_id)
    if stale(report):
        report = with_peer_review(paper.arxiv_id, lookup(paper))
        Generated.objects.update_or_create(kind=KIND, key=paper.arxiv_id, lang='en',
            defaults={'status': 'done', 'content': json.dumps(report)})
    return report


def description(report):
    status = (report or {}).get('status')
    labels = {
        'published': _('Publication found'),
        'forthcoming': _('Publication announced for a future date'),
        'journal_reference': _('Journal reference declared on arXiv; publication not independently confirmed'),
        'withdrawn_declared': _('Withdrawal declared on arXiv by the authors'),
        'accepted_declared': _('Acceptance declared on arXiv by the authors'),
        'submitted_declared': _('Submission or review declared on arXiv by the authors'),
        'preprint': _('Preprint found; no journal publication confirmed'),
        'unknown': _('Editorial status not confirmed by the sources checked'),
    }
    return labels.get(status, _('Editorial status has not been checked yet'))


def questions(kit, report):
    result = []
    for question in (kit or {}).get('questions_for_authors', []):
        clean = normalized(question)
        generic = re.search(r'statut (?:editorial|de publication)|editorial status|publication status|status of (?:the |this )?manuscript', clean)
        if generic and (report or {}).get('status') == 'published':
            question = _('Which version of the text and data should be cited, and have any corrections appeared since publication?')
        if question not in result:
            result.append(question)
    return result


def display_context(report):
    try:
        checked = datetime.datetime.fromisoformat(report['checked_at'])
    except (TypeError, KeyError, ValueError):
        checked = None
    return {'publication': report, 'publication_label': description(report), 'publication_checked_at': checked,
            'publication_peer_review': peer_review_text(report)}


# The paper page shows the journal, DOI and dates found by lookup. Its views queue the papers whose check is
# missing or old, and the hourly run checks a few of them at arXiv's pace (refresh_queued).
QUEUE_KEY = 'publication-status-queue'
MAX_QUEUED = 500
MAX_REFRESHED_PER_RUN = 30
ARXIV_SECONDS = 3
SHOWN_EVERY = datetime.timedelta(days=30)
PUBLISHED_SHOWN_EVERY = datetime.timedelta(days=180)


def page_due(report):
    if not report:
        return True
    return stale(report, PUBLISHED_SHOWN_EVERY if report.get('status') == 'published' else SHOWN_EVERY)


def queue(arxiv_id):
    ids = cache.get(QUEUE_KEY) or []
    if arxiv_id not in ids and len(ids) < MAX_QUEUED:
        cache.set(QUEUE_KEY, ids + [arxiv_id], 7 * 86400)


def iso_date(value):
    """('2013-06' ->) (datetime.date(2013, 6, 1), 'F Y'): the date and the format that shows only what is known."""
    parts = [int(n) for n in re.findall(r'\d+', value or '')][:3]
    try:
        return datetime.date(*(parts + [1, 1])[:3]), {1: 'Y', 2: 'F Y'}.get(len(parts), 'd M. Y')
    except (TypeError, ValueError):
        return None, ''


def for_page(paper):
    """What the paper page shows of the paper's publication: {doi, journal, published, published_format, accepted,
    forthcoming}. The journal comes from Crossref when it confirmed the publication, else the arXiv journal reference."""
    row = Generated.objects.filter(kind=KIND, key=paper.arxiv_id, lang='en').first()
    report = row.data() if row and row.content else None
    if page_due(report) and not sources.is_external(paper.arxiv_id):
        queue(paper.arxiv_id)
    report = report or {}
    confirmed = report.get('status') in ('published', 'forthcoming')
    doi = (confirmed and report.get('doi')) or report.get('arxiv_doi') or sources.clean_doi(paper.link_doi or '') or ''
    published, published_format = iso_date(report.get('date')) if confirmed else (None, '')
    accepted = iso_date(report.get('accepted'))[0] if confirmed else None
    return {'doi': '' if sources.ARXIV_DOI_RE.match(doi) else doi,
            'journal': report.get('journal', '') if confirmed else (paper.journal_ref or ''),
            'confirmed': confirmed, 'forthcoming': report.get('status') == 'forthcoming',
            'published': published, 'published_format': published_format, 'accepted': accepted}


def remember(paper, report):
    """Keep on the paper the DOI and journal reference that arXiv now has: they are often added after publication."""
    fields = {}
    doi = report.get('arxiv_doi') or (report.get('status') in ('published', 'forthcoming') and report.get('doi'))
    if doi and not sources.ARXIV_DOI_RE.match(doi) and sources.clean_doi(paper.link_doi or '') != doi:
        fields['link_doi'] = 'https://doi.org/' + doi
    if report.get('journal_ref') and report['journal_ref'] != (paper.journal_ref or ''):
        fields['journal_ref'] = report['journal_ref']
    if fields:  # update() leaves ArxivPaper.updated alone: it dates the summary
        ArxivPaper.objects.filter(pk=paper.pk).update(**fields)


def refresh_queued(limit=MAX_REFRESHED_PER_RUN):
    """From the hourly run: check the queued papers. Followed papers are left to publication_alerts, whose
    alerts come from comparing its own checks. Returns how many papers were checked."""
    ids = cache.get(QUEUE_KEY) or []
    cache.set(QUEUE_KEY, ids[limit:], 7 * 86400)
    checked = 0
    for arxiv_id in ids[:limit]:
        paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
        base = re.sub(r'v\d+$', '', arxiv_id).replace('--', '/')
        if paper is None or FollowedPaper.objects.filter(arxiv_id=base).exists():
            continue
        row = Generated.objects.filter(kind=KIND, key=arxiv_id, lang='en').first()
        if not page_due(row.data() if row and row.content else None):
            continue
        if checked:
            time.sleep(ARXIV_SECONDS)
        try:
            report = lookup(paper)
        except Exception:
            logger.warning('Publication check failed for %s', arxiv_id)
            continue
        checked += 1
        Generated.objects.update_or_create(kind=KIND, key=arxiv_id, lang='en',
                                           defaults={'status': 'done', 'content': json.dumps(report)})
        remember(paper, report)
    return checked
