"""Sourced editorial checks, without model inference or publisher-page scraping."""
import datetime
import html
import json
import re
import unicodedata
import requests
from urllib.parse import quote
from xml.etree import ElementTree

from django.utils import timezone
from django.utils.translation import gettext as _
from pylatexenc.latex2text import LatexNodes2Text

from . import digest, generated, sources
from .models import Generated

KIND = 'presskit_status'
TTL = datetime.timedelta(days=1)


def normalized(value):
    text = LatexNodes2Text().latex_to_text(html.unescape(str(value or '')))
    text = text.replace('_', '').replace('^', '')
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode().lower()
    return re.sub(r'[^a-z0-9]+', ' ', text).strip()


def latest(paper_id, kit=None):
    row = Generated.objects.filter(kind=KIND, key=paper_id, lang='en').first()
    return (row.data() if row and row.content else None) or (kit or {}).get('publication_status')


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
    journal = (work.get('container-title') or [''])[0]
    date = publication_date(work)
    if not doi or not journal or work.get('type') not in ('journal-article', 'proceedings-article') or not date:
        return False
    report.update(status='forthcoming' if date > timezone.now().date().isoformat()[:len(date)] else 'published',
                  doi=doi, journal=journal, date=date, updates=notices(work))
    report['sources'].append({'name': 'Crossref', 'url': 'https://doi.org/' + doi,
                             'detail': journal + ' · ' + date, 'kind': 'publisher_metadata'})
    return True


def for_generation(paper):
    report = latest(paper.arxiv_id)
    if stale(report):
        report = lookup(paper)
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
    return {'publication': report, 'publication_label': description(report), 'publication_checked_at': checked}
