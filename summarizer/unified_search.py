"""Cached search across arXiv and the existing journal/preprint index."""
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from itertools import zip_longest
from xml.etree import ElementTree

import requests
from django.core.cache import cache
from django.db import close_old_connections

from . import digest, sources


def arxiv_identifier(text):
    value = text.strip()
    value = re.sub(r'^https?://(?:www\.|export\.)?arxiv\.org/(?:abs|pdf|html)/', '', value, flags=re.I)
    value = re.sub(r'^arxiv:\s*', '', value, flags=re.I)
    value = re.sub(r'\.pdf(?:[?#].*)?$', '', value, flags=re.I)
    value = re.split(r'[?#]', value)[0]
    if digest.ARXIV_ID_RE.fullmatch(value):
        return digest.site_id(value if re.search(r'v\d+$', value) else value + 'v1')
    return None


def arxiv_results(query, field='all'):
    if field in ('author', 'title'):
        text = re.sub(r'["\\]', ' ', query)
        query = '%s:"%s"' % ('au' if field == 'author' else 'ti', text)
    response = requests.get(digest.ARXIV_API, params={
        'search_query': query if re.search(r'\b(?:all|au|ti|abs|cat|co|jr|id):', query) else 'all:' + query,
        'start': 0, 'max_results': 25, 'sortBy': 'relevance', 'sortOrder': 'descending',
    }, headers={'User-Agent': sources.USER_AGENT}, timeout=8)
    response.raise_for_status()
    return [{
        'key': p['arxiv_id'], 'arxiv_id': p['arxiv_id'], 'title': p['title'],
        'authors': p['authors'][:6], 'more_authors': len(p['authors']) > 6,
        'venue': 'arXiv', 'published': p['published'], 'arxiv_result': True,
        'doi_key': sources.doi_key(p['doi']) if p['doi'] else '',
    } for p in digest.parse_feed(response.text)]


def _fetch(provider, query, field):
    # External-paper lookups may use a thread-local database connection.
    close_old_connections()
    try:
        return (provider(query) if field == 'all' else provider(query, field=field)), False
    except (sources.SourceUnavailable, requests.RequestException, ElementTree.ParseError):
        return [], True
    finally:
        close_old_connections()


def search(query, field='all', source='all', kind=''):
    field = field if field in ('author', 'title') else 'all'
    source = source if source in ('open', 'arxiv') else 'all'
    kind = kind if kind in sources.SEARCH_TYPES else ''
    key = 'unified-search:v2:' + ':'.join((field, source, kind)) + ':' + hashlib.sha256(query.lower().encode()).hexdigest()
    cached = cache.get(key)
    if cached is not None:
        return cached
    with ThreadPoolExecutor(max_workers=2) as pool:
        arxiv_job = pool.submit(_fetch, arxiv_results, query, field) if source != 'open' else None
        other_provider = partial(sources.search, kind=kind) if kind else sources.search
        other_job = pool.submit(_fetch, other_provider, query, field) if source != 'arxiv' else None
        arxiv, arxiv_failed = arxiv_job.result() if arxiv_job else ([], False)
        other, other_failed = other_job.result() if other_job else ([], False)
    results, seen = [], set()
    for pair in zip_longest(arxiv, other):
        for paper in pair:
            if paper is None:
                continue
            aliases = {paper['key'].lower()}
            if paper.get('doi_key'):
                aliases.add(paper['doi_key'].lower())
            if paper.get('arxiv_id'):
                aliases.add(re.sub(r'v\d+$', '', paper['arxiv_id'].replace('/', '--').lower()))
            duplicate = bool(aliases & seen)
            seen.update(aliases)
            if not duplicate:
                results.append(paper)
    result = (results, arxiv_failed or other_failed)
    cache.set(key, result, 60 if result[1] else sources.SEARCH_SECONDS)
    return result
