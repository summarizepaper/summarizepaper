"""The references of a manuscript watched until it is published (Premium, from a bibliography check).

Months go by between submission and the proofs: a cited preprint is published, a cited article is retracted
or corrected. Once a day the hourly run (publication_alerts.run) reads again, for the watches due, the arXiv
record of each cited preprint (its journal DOI, a withdrawal) and the Crossref record of each DOI (the
retractions, corrections and expressions of concern it links to; for a preprint server's DOI, its published
version), WATCH_LOOKUPS references at most a day, each at most once a day. The first check of a reference only
records what it is: what changes afterwards becomes a PublicationAlert, emailed with the alerts of the followed
papers. A watch ends after WATCH_DAYS days, or when its owner stops it.
"""
import datetime
import json
import logging
import re
import time

from django.db.models import F, Q
from django.utils import timezone

from . import publication_status, refcheck, sources
from .models import PublicationAlert, ReferenceWatch

logger = logging.getLogger(__name__)

WATCH_DAYS = 548  # eighteen months: most papers are published by then
WATCH_LOOKUPS = 300
EVERY = datetime.timedelta(days=1)
CROSSREF_SECONDS = 0.2


def start(user, row, result):
    """Watch the identified references of a check (again, when it was stopped)."""
    refs = [{'key': r['key'], 'title': r['title'][:300], 'arxiv': r.get('arxiv', ''), 'doi': r.get('doi', '')}
            for r in result.get('refs', []) if r.get('arxiv') or r.get('doi')]
    watch, _created = ReferenceWatch.objects.update_or_create(user=user, check_key=row.key, defaults={
        'refs': json.dumps(refs), 'checked': None, 'name': str(result.get('name') or '')[:200],
        'until': timezone.now() + datetime.timedelta(days=WATCH_DAYS)})
    return watch


def state_of(ref, records):
    """What a reference is today: {'published', 'journal', 'withdrawn', 'notices'}; None when a source did not answer."""
    state = {'published': '', 'journal': '', 'withdrawn': False, 'notices': []}
    if ref.get('arxiv'):
        record = records.get(ref['arxiv'])
        if record is None:
            return None
        state['withdrawn'] = bool(re.search(r'\bwithdrawn\b', record.get('comment', ''), re.I))
        doi = sources.clean_doi(record.get('doi', '')) or ''
        if doi and not sources.ARXIV_DOI_RE.match(doi):
            state['published'], state['journal'] = doi.lower(), record.get('journal_ref', '')[:200]
    doi = ref.get('doi') or state['published']
    if doi:
        try:
            work = publication_status.crossref(doi)
            if work.get('type') == 'posted-content':
                linked = next((sources.clean_doi(r.get('id', '')) for r in (work.get('relation') or {}).get('is-preprint-of') or []
                               if r.get('id-type') == 'doi'), '')
                linked = linked or refcheck.biorxiv_published(doi)
                if linked:
                    state['published'] = linked.lower()
                    work = publication_status.crossref(linked)
        except (sources.SourceUnavailable, ValueError, TypeError, AttributeError):
            return None
        if state['published'] and work.get('container-title'):
            state['journal'] = str(work['container-title'][0])[:200]
        state['notices'] = sorted([n['type'], n['doi']] for n in publication_status.notices(work))
    return state


def news(before, after):
    """[(kind, journal, doi)] of what changed since the previous check."""
    found = []
    if after['published'] and not before.get('published'):
        found.append(('published', after['journal'], after['published']))
    if after['withdrawn'] and not before.get('withdrawn'):
        found.append(('withdrawal', '', ''))
    seen = {tuple(n) for n in before.get('notices') or []}
    found += [(kind, after['journal'], doi) for kind, doi in after['notices'] if (kind, doi) not in seen]
    return found


def check(now, limit=WATCH_LOOKUPS):
    """Check the watches due, the least recently checked first; queue the alerts. Returns how many were queued."""
    ReferenceWatch.objects.filter(until__lt=now).delete()
    spent = queued = 0
    due = ReferenceWatch.objects.filter(Q(checked__isnull=True) | Q(checked__lt=now - EVERY)).order_by(
        F('checked').asc(nulls_first=True))
    for watch in due:
        if spent >= limit:
            break
        refs = json.loads(watch.refs or '[]')
        todo = [r for r in refs if not r.get('checked') or r['checked'] < (now - EVERY).isoformat()][:limit - spent]
        records = refcheck.arxiv_records([r['arxiv'] for r in todo if r.get('arxiv')])
        for ref in todo:
            spent += 1
            state = state_of(ref, records)
            if ref.get('doi') or (state or {}).get('published'):
                time.sleep(CROSSREF_SECONDS)
            if state is None:
                continue
            if 'state' in ref:
                for kind, journal, doi in news(ref['state'], state):
                    paper = ref['arxiv'] or sources.doi_key(ref['doi'])
                    _alert, created = PublicationAlert.objects.get_or_create(
                        user=watch.user, arxiv_id=paper, kind=kind, doi=doi or '',
                        defaults={'paper': paper, 'title': ref['title'], 'journal': journal or ''})
                    queued += created
            ref['state'], ref['checked'] = state, now.isoformat()
        watch.refs = json.dumps(refs)
        if all(r.get('checked', '') >= (now - EVERY).isoformat() for r in refs):
            watch.checked = now
        watch.save(update_fields=['refs', 'checked'])
    return queued
