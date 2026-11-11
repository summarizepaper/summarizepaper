"""Zotero (Premium): the researcher's library, read and written through the Zotero web API.

- Connection: one click through Zotero's OAuth when ZOTERO_CLIENT_KEY and ZOTERO_CLIENT_SECRET are set, or a
  private key pasted on the integrations page. The key is stored encrypted (models.zotero_cipher). The library
  read and written is the personal one or a group library the key may write to; single papers go to the
  collection the researcher chose.
- send(): papers written without duplicates. An item already in the library (same DOI, arXiv id or title) gets
  the collection, the tags and the notes it lacks; the others are created, as their published version when
  there is one, with the citation key of our BibTeX pinned for Better BibTeX. The summary note says it was
  written by a model, from which version of the paper and when; the reader's own note is a note apart.
- sync_collection(): a collection of SummarizePaper and a Zotero collection kept together. Papers added on
  either side reach the other, a paper removed on one side leaves the other (the Zotero item stays in the
  library), notes, tags, reading status and editorial notices (retraction, correction, published version) go
  to Zotero as notes and tags.
- apply_refcheck(): the findings of a bibliography check written back: tags and a note on each item concerned,
  and the published version of a preprint added next to it, related to it.
- send_review(): a literature review as a collection with a subcollection per approach, each paper with a note
  holding the statements of the review it supports and its sentence that supports them.
"""
import base64
import datetime
import hashlib
import hmac
import html
import logging
import re
import secrets
import time
import unicodedata
import urllib.parse
import uuid

import requests
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import Http404, JsonResponse
from django.shortcuts import redirect, render
from django.utils import timezone, translation
from django.utils.html import escape
from django.utils.translation import get_language
from django.utils.translation import gettext as _
from django.utils.translation import gettext_lazy
from django.views.decorators.debug import sensitive_post_parameters
from django.views.decorators.http import require_POST

from . import access, digest, exports, generated, keypoints, sources
from .models import Collection, CollectionPaper, Generated, ZoteroAccount

logger = logging.getLogger(__name__)

API = 'https://api.zotero.org'
OAUTH = 'https://www.zotero.org/oauth'
TAG = 'SummarizePaper'
BATCH = 50  # objects per write
MAX_ITEMS = 500  # items read from one collection
DAILY = datetime.timedelta(hours=20)  # a collection kept in sync is synced again after this, by the hourly run
DAILY_PER_RUN = 5
MAX_PDF = 40 * 1024 * 1024
SEARCH_LIMIT = 100  # items read for each search of the library
NOTICE_TAGS = {'retracted': 'Retracted', 'withdrawn': 'Withdrawn', 'concern': 'Expression of concern',
               'corrected': 'Correction published', 'published': 'Published version available'}
SKIPPED_MARKS = ('intended', 'ignored')
SYNC_STEPS = [('link', gettext_lazy('Finding the Zotero collection')), ('pull', gettext_lazy('Adding the papers new in Zotero')),
              ('push', gettext_lazy('Writing the papers, notes and tags into Zotero'))]


class Unavailable(Exception):
    """Zotero refused the request or did not answer."""


# --- the API ------------------------------------------------------------------------------------------------

def call(account, method, path, **kwargs):
    """A request to the account's library ('items', 'collections/KEY/items/top'...); waits when Zotero asks to."""
    headers = {'Zotero-API-Key': account.api_key, 'Zotero-API-Version': '3', **kwargs.pop('headers', {})}
    url = '%s/%s/%s' % (API, account.prefix(), path)
    for attempt in range(3):
        try:
            response = requests.request(method, url, headers=headers, timeout=30, **kwargs)
        except requests.RequestException as e:
            raise Unavailable(str(e))
        if response.status_code not in (429, 503) or attempt == 2:
            break
        time.sleep(min(_seconds(response.headers.get('Retry-After')) or 5, 30))
    if response.status_code >= 400:
        raise Unavailable('%s on %s %s' % (response.status_code, method, path))
    backoff = _seconds(response.headers.get('Backoff'))
    if backoff:
        time.sleep(min(backoff, 30))
    return response


def _seconds(value):
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def get(account, path, params=None):
    try:
        return call(account, 'GET', path, params=params or {}).json()
    except ValueError:
        raise Unavailable('not JSON')


def create(account, objects):
    """Creates items, notes or collections; {index: key} of those Zotero accepted."""
    created = {}
    for start in range(0, len(objects), BATCH):
        try:
            result = call(account, 'POST', 'items' if 'itemType' in objects[0] else 'collections',
                          json=objects[start:start + BATCH]).json()
        except ValueError:
            continue
        for index, value in (result.get('successful') or result.get('success') or {}).items():
            created[start + int(index)] = value.get('key') if isinstance(value, dict) else value
    return created


def patch(account, item, changes):
    """Changes some fields of an item; False when it was modified meanwhile or refused."""
    try:
        call(account, 'PATCH', 'items/%s' % item['key'], json=changes,
             headers={'If-Unmodified-Since-Version': str(item.get('version') or item['data'].get('version', 0))})
        return True
    except Unavailable:
        logger.warning('Zotero item %s could not be updated', item['key'])
        return False


def check_key(library_id, api_key):
    """What Zotero says of a key ({userID, username, access}), or None when it does not exist."""
    try:
        response = requests.get('%s/keys/%s' % (API, api_key), timeout=15)
        return response.json() if response.status_code == 200 else None
    except (requests.RequestException, ValueError):
        return None


def may_write(info, library_id):
    return bool(info) and str(info.get('userID')) == str(library_id) and \
        bool((info.get('access') or {}).get('user', {}).get('write'))


def groups(account):
    """[(prefix, name)] of the group libraries this key may write to."""
    info = check_key(account.library_id, account.api_key) or {}
    rights = (info.get('access') or {}).get('groups') or {}
    try:
        found = requests.get('%s/users/%s/groups' % (API, account.library_id), timeout=15,
                             headers={'Zotero-API-Key': account.api_key, 'Zotero-API-Version': '3'}).json()
    except (requests.RequestException, ValueError):
        return []
    out = []
    for group in found if isinstance(found, list) else []:
        gid = str(group.get('id'))
        if (rights.get(gid) or rights.get('all') or {}).get('write'):
            out.append(('groups/%s' % gid, (group.get('data') or {}).get('name') or gid))
    return out


def collections(account):
    """[{key, name, parent, items}] of the library's collections."""
    found = []
    for start in range(0, 1000, 100):
        page = get(account, 'collections', {'limit': 100, 'start': start})
        found += [{'key': c.get('key'), 'name': (c.get('data') or {}).get('name', ''),
                   'parent': (c.get('data') or {}).get('parentCollection') or '',
                   'items': (c.get('meta') or {}).get('numItems', 0)} for c in page]
        if len(page) < 100:
            break
    return found


def ensure_collection(account, name, parent='', known=None):
    """The key of the collection with this name (under parent), created when there is none."""
    name = ' '.join(name.split())[:200] or 'SummarizePaper'
    for c in known if known is not None else collections(account):
        if c['name'] == name and (c['parent'] or '') == (parent or ''):
            return c['key']
    created = create(account, [{'name': name, **({'parentCollection': parent} if parent else {})}])
    if not created.get(0):
        raise Unavailable('collection not created')
    if known is not None:
        known.append({'key': created[0], 'name': name, 'parent': parent, 'items': 0})
    return created[0]


def fetch(account, path, params=None):
    """(JSON, version of the library) of a read."""
    response = call(account, 'GET', path, params=params or {})
    try:
        return response.json(), _seconds(response.headers.get('Last-Modified-Version'))
    except ValueError:
        raise Unavailable('not JSON')


def collection_items(account, key):
    """The top items of a collection (at most MAX_ITEMS), whether that is all of them, and the library's version."""
    items, version = [], 0
    for start in range(0, MAX_ITEMS, 100):
        page, read = fetch(account, 'collections/%s/items/top' % key, {'format': 'json', 'limit': 100, 'start': start})
        version = version or read
        items += [i for i in page if (i.get('data') or {}).get('itemType') not in ('note', 'attachment', 'annotation')]
        if len(page) < 100:
            return items, True, version
    return items, False, version


def changes_since(account, version):
    """(top items changed since this version of the library, trashed ones included; keys of the items deleted
    since; the library's version now), or None when more changed than MAX_ITEMS (the whole collection is read)."""
    changed, now = [], 0
    for start in range(0, MAX_ITEMS, 100):
        page, read = fetch(account, 'items/top', {'since': version, 'includeTrashed': 1, 'limit': 100, 'start': start})
        now = now or read
        changed += page
        if len(page) < 100:
            break
    else:
        return None
    deleted, _read = fetch(account, 'deleted', {'since': version})
    return changed, set((deleted or {}).get('items') or []), now


def open_link(account, key, kind='items'):
    """Opens the item or the collection in the Zotero app."""
    return 'zotero://select/%s/%s/%s' % (account.library or 'library', kind, key)


def web_link(account, key, kind='items'):
    if account.library:
        return 'https://www.zotero.org/%s/%s/%s' % (account.library, kind, key)
    return 'https://www.zotero.org/%s/%s/%s' % (account.username, kind, key) if account.username else ''


def item_uri(account, key):
    return 'http://zotero.org/%s/items/%s' % (account.prefix(), key)


# --- the same paper -----------------------------------------------------------------------------------------

ARXIV_IN_ZOTERO = re.compile(r'(?:arxiv\.org/(?:abs|pdf)/|arxiv:\s*)((?:\d{4}\.\d{4,5}|[a-z-]+(?:\.[a-z]{2})?/\d{7})(?:v\d+)?)', re.I)


def zotero_ref(data):
    """The arXiv id or DOI of a Zotero item, '' when it has neither."""
    for field in ('archiveID', 'url', 'extra'):
        match = ARXIV_IN_ZOTERO.search(str(data.get(field) or ''))
        if match:
            return match.group(1)
    doi = sources.clean_doi(str(data.get('DOI') or '')) or sources.clean_doi(
        (re.search(r'^DOI:\s*(\S+)', str(data.get('extra') or ''), re.I | re.M) or [None, ''])[1])
    return doi or ''


def _title(text):
    from .publication_status import normalized
    title = normalized(text)
    return title if len(title) >= 20 else ''


def identifiers(arxiv_id='', dois=(), title=''):
    """What tells a paper apart: its DOIs, its arXiv id without version, its title."""
    found = {('doi', d.lower()) for d in dois if d}
    if arxiv_id and sources.is_external(arxiv_id):
        doi = sources.key_doi(arxiv_id)
        if doi:
            found.add(('doi', doi.lower()))
    elif arxiv_id:
        found.add(('arxiv', exports.bare_ref(arxiv_id).lower()))
    if _title(title):
        found.add(('title', _title(title)))
    return found


def meta_identifiers(meta):
    published = meta.get('published') or {}
    return identifiers(meta['arxiv_id'], [meta.get('doi'), published.get('doi')], meta['title'])


def item_identifiers(data):
    found = set()
    for field in ('archiveID', 'url', 'extra'):
        for match in ARXIV_IN_ZOTERO.finditer(str(data.get(field) or '')):
            found.add(('arxiv', re.sub(r'v\d+$', '', match.group(1)).lower()))
    for text in [str(data.get('DOI') or '')] + re.findall(r'^DOI:\s*(\S+)', str(data.get('extra') or ''), re.I | re.M):
        doi = sources.clean_doi(text)
        if doi:
            found.add(('doi', doi.lower()))
    if _title(data.get('title', '')):
        found.add(('title', _title(data['title'])))
    return found


def find(account, wanted, title, items=None, author=''):
    """The item of the library (or of these items) that is this paper, or None. Zotero's quick search reads the
    titles, creators and years only (not DOIs nor archive ids), and one word at a time does not depend on how it
    combines words: the longest word of the title, then the first author's family name (a title changed between
    the preprint and the journal); the DOI, arXiv id or title of each item found decides."""
    def match(found):
        for item in found:
            data = item.get('data') or {}
            if data.get('itemType') not in ('note', 'attachment', 'annotation') and wanted & item_identifiers(data):
                return item
        return None
    if items is not None:
        return match(items)
    words = sorted(re.findall(r'[^\W\d_]{4,}', re.sub(r'\$[^$]*\$', ' ', title or '')), key=len, reverse=True)
    family = exports.split_name(author)[1].split()[-1] if author and exports.split_name(author)[1] else ''
    for query in [q for q in dict.fromkeys(words[:1] + [family]) if q]:
        found = match(get(account, 'items/top', {'q': query, 'qmode': 'titleCreatorYear', 'limit': SEARCH_LIMIT}))
        if found is not None:
            return found
    return None


# --- what is written ----------------------------------------------------------------------------------------

def creator(name):
    given, family = exports.split_name(name)
    if given:
        return {'creatorType': 'author', 'firstName': given, 'lastName': family}
    return {'creatorType': 'author', 'name': family}


def tag_list(tags):
    return [{'tag': t} for t in dict.fromkeys(t.strip() for t in tags if t and t.strip())]


def item_for(meta, citekey=''):
    """The Zotero item of a paper: its published version when there is one, the arXiv id kept."""
    journal, doi, year = exports.journal_of(meta)
    ref = exports.bare_ref(meta['arxiv_id'])
    external = sources.is_external(meta['arxiv_id'])
    item = {'itemType': 'journalArticle' if journal else 'preprint', 'title': meta['title'],
            'creators': [creator(a) for a in meta['authors'][:100]], 'abstractNote': meta['abstract'],
            'date': year if meta.get('published') else meta['date'], 'DOI': doi or '',
            'url': 'https://doi.org/%s' % doi if journal and doi else exports.paper_url(meta)}
    extra = []
    if journal:
        item['publicationTitle'] = journal
        if not external:
            extra.append('arXiv: %s' % ref)
    elif external:
        item['repository'] = (meta['journal'] or '').replace(' preprint', '').split(',')[0]
    else:
        item.update(repository='arXiv', archiveID='arXiv:%s' % ref)
    if citekey:  # the key of our .bib files: Zotero's own field, and the line Better BibTeX before Zotero 7 pins
        item['citationKey'] = citekey
        extra.append('Citation Key: %s' % citekey)
    item['extra'] = '\n'.join(extra)
    return item


def note(body, marker):
    """A child note; marker: a text that, found in a note the item already has, means it is there already."""
    return {'html': body, 'marker': marker}


def with_math(text):
    """Escaped text whose LaTeX formulas Zotero's note editor displays: $$...$$ as a block, $...$ inline."""
    text = re.sub(r'\$\$(.+?)\$\$', lambda m: '<pre class="math">$$%s$$</pre>' % m.group(1), escape(text), flags=re.S)
    return re.sub(r'(?<![$\\])\$([^$\n]+?)\$(?!\$)', lambda m: '<span class="math">$%s$</span>' % m.group(1), text)


def summary_note(meta, lang):
    row = exports.summary_of(meta['arxiv_id'], lang)
    if row is None:
        return None
    url = exports.page_url(meta, lang)
    with translation.override(lang):
        points = ''.join('<li>%s</li>' % with_math(p) for p in keypoints.stored_points(row.notes))
        body = '<h1>SummarizePaper</h1><p><em>%s</em></p>%s<h2>%s</h2><p>%s</p><p><a href="%s">%s</a></p>' % (
            escape(exports.provenance(meta['arxiv_id'], row)),
            '<h2>%s</h2><ul>%s</ul>' % (escape(_('Key points')), points) if points else '',
            escape(_('Summary')), with_math(row.summary.strip()).replace('\n', '<br>'), url, url)
    return note(body, url)


def own_note(text, lang):
    with translation.override(lang):
        title = _('My note')
    return note('<h1>%s</h1><p>%s</p>' % (escape(title), escape(text.strip()).replace('\n', '<br>')), plain(escape(text)))


def plain(text):
    return ' '.join(re.sub(r'<[^>]+>', ' ', html.unescape(text or '')).split())


def paper_entry(meta, lang, note_text='', tags='', reading='', notices=(), key=''):
    """What send() writes for a paper: its summary, the reader's note, their tags and reading status, notices."""
    from .library import READING
    notes = [n for n in (summary_note(meta, lang), own_note(note_text, lang) if note_text.strip() else None) if n]
    with translation.override(lang):
        status = str(dict(READING).get(reading, '')) if reading else ''
    found = [t.strip() for t in (tags or '').split(',') if t.strip()] + ([status] if status else []) + list(notices)
    return {'meta': meta, 'notes': notes, 'tags': found, 'key': key}


def add_notes(account, key, notes):
    """The notes the item does not have yet."""
    if not notes:
        return 0
    children = get(account, 'items/%s/children' % key, {'itemType': 'note'})
    there = [(c.get('data') or {}).get('note', '') for c in children]
    missing = [n for n in notes if not any(n['marker'] in t or n['marker'] in plain(t) for t in there)]
    create(account, [{'itemType': 'note', 'parentItem': key, 'note': n['html'], 'tags': tag_list([TAG])} for n in missing])
    return len(missing)


def complete(account, item, collections_wanted=(), tags=(), relations=()):
    """Adds collections, tags and related items an item lacks."""
    data = item.get('data') or {}
    changes = {}
    have = list(data.get('collections') or [])
    if any(c not in have for c in collections_wanted if c):
        changes['collections'] = have + [c for c in dict.fromkeys(collections_wanted) if c and c not in have]
    present = {t.get('tag', '').lower() for t in data.get('tags') or []}
    if any(t.lower() not in present for t in tags if t):
        changes['tags'] = list(data.get('tags') or []) + [t for t in tag_list(tags) if t['tag'].lower() not in present]
    if relations:
        related = dict(data.get('relations') or {})
        current = related.get('dc:relation') or []
        current = [current] if isinstance(current, str) else list(current)
        if any(r not in current for r in relations):
            related['dc:relation'] = current + [r for r in relations if r not in current]
            changes['relations'] = related
    return patch(account, item, changes) if changes else True


def send(account, entries, collection='', step=lambda: None):
    """Writes papers into the library without duplicating them. entries: paper_entry() dicts; collection: the
    key of the Zotero collection they go to. {'saved', 'existing', 'failed', 'keys': item key per entry}."""
    keys, new, existing = [''] * len(entries), [], 0
    citekeys = exports.unique_keys([exports.cite_key(e['meta']) for e in entries])
    for index, entry in enumerate(entries):
        if index % 10 == 9:
            step()
        item = None
        if entry.get('key'):
            try:
                item = get(account, 'items/%s' % entry['key'])
            except Unavailable:
                item = None  # deleted in Zotero since
            if item and (item.get('data') or {}).get('deleted'):
                item = None
        item = item or find(account, meta_identifiers(entry['meta']), entry['meta']['title'],
                            author=(entry['meta']['authors'] or [''])[0])
        if item is None:
            new.append(index)
            continue
        existing += 1
        # a key only once the item is in the collection: a sync takes out of a collection the papers whose item left it
        keys[index] = item['key'] if complete(account, item, [collection], entry['tags']) or not collection else ''
        add_notes(account, item['key'], entry['notes'])
    objects = []
    for index in new:
        obj = item_for(entries[index]['meta'], citekeys[index])
        obj['tags'] = tag_list([TAG] + entries[index]['tags'])
        if collection:
            obj['collections'] = [collection]
        objects.append(obj)
    created = create(account, objects) if objects else {}
    children = []
    for position, index in enumerate(new):
        keys[index] = created.get(position) or ''
        if keys[index]:
            children += [{'itemType': 'note', 'parentItem': keys[index], 'note': n['html'], 'tags': tag_list([TAG])}
                         for n in entries[index]['notes']]
    if children:
        step()
        create(account, children)
    return {'saved': sum(1 for i in new if keys[i]), 'existing': existing, 'failed': sum(1 for k in keys if not k),
            'keys': keys}


# --- collections kept together ------------------------------------------------------------------------------

def row_identifiers(row):
    return identifiers(row.arxiv_id, [], row.title)


def notice_tags(facts):
    """The tags of what the editorial checks found on a paper (collection_tools.facts)."""
    found = []
    for notice in facts.get('notices') or []:
        kind = notice.get('type') or ''
        found.append(NOTICE_TAGS['retracted'] if kind in ('retraction', 'removal') else
                     NOTICE_TAGS['withdrawn'] if kind == 'withdrawal' else
                     NOTICE_TAGS['concern'] if kind == 'expression_of_concern' else NOTICE_TAGS['corrected'])
    if facts.get('status') == 'published':
        found.append(NOTICE_TAGS['published'])
    elif facts.get('status') == 'withdrawn_declared':
        found.append(NOTICE_TAGS['withdrawn'])
    return found


def link_rows(rows, items):
    """Gives each paper of the collection the key of its item; the items left are those not in the collection."""
    left = list(items)
    for row in rows:
        wanted = row_identifiers(row)
        item = next((i for i in left if wanted & item_identifiers(i.get('data') or {})), None)
        if item is not None:
            left.remove(item)
            if row.zotero_key != item['key']:
                row.zotero_key = item['key']
                row.save(update_fields=['zotero_key'])
    return left


def link_collection(account, collection, key=''):
    """Links the collection to a Zotero collection (this one, or one of the same name, created if needed)."""
    moved = collection.zotero_library != account.prefix() or (key and key != collection.zotero_key)
    if moved or not collection.zotero_key:
        if collection.zotero_key:
            collection.papers.update(zotero_key='')  # items of another collection or library
        collection.zotero_key = key or ensure_collection(account, collection.name)
        collection.zotero_library = account.prefix()
        collection.zotero_removed, collection.zotero_synced = '', None
        collection.save(update_fields=['zotero_key', 'zotero_library', 'zotero_removed', 'zotero_synced'])


def pull(account, collection, paid, step=lambda: None, full=True):
    """Adds to the collection the papers of its Zotero collection it lacks, and takes out those removed from Zotero
    (or put in its trash) since the last sync. Not full: only what changed in the library since the last sync is
    read. (counts as library.add_many gives them, with 'skipped': items without DOI nor arXiv id, and 'removed';
    the library's version read)."""
    from .library import add_many, canonical_id
    rows = list(collection.papers.all())
    changes = changes_since(account, collection.zotero_version) \
        if not full and collection.zotero_version and collection.zotero_synced else None
    gone = []
    if changes is None:
        items, complete_list, version = collection_items(account, collection.zotero_key)
        left = link_rows(rows, items)
        if complete_list and collection.zotero_synced:
            present = {i['key'] for i in items}
            gone = [r for r in rows if r.zotero_key and r.zotero_key not in present]
    else:
        changed, deleted, version = changes
        inside = [i for i in changed if collection.zotero_key in ((i.get('data') or {}).get('collections') or [])
                  and not (i.get('data') or {}).get('deleted')]
        out = ({i['key'] for i in changed} - {i['key'] for i in inside}) | deleted
        gone = [r for r in rows if r.zotero_key and r.zotero_key in out]
        linked = {r.zotero_key for r in rows if r.zotero_key}
        left = link_rows([r for r in rows if not r.zotero_key], [i for i in inside if i['key'] not in linked])
    CollectionPaper.objects.filter(pk__in=[r.pk for r in gone]).delete()
    refs = [zotero_ref(i.get('data') or {}) for i in left]
    ids = [i for i in dict.fromkeys(canonical_id(r) for r in refs if r) if i]
    step()
    counts = add_many(collection, ids, paid) if ids else {}
    link_rows(list(collection.papers.filter(zotero_key='')), left)
    counts['skipped'] = counts.get('skipped', 0) + sum(1 for r in refs if not r)
    counts['removed'] = len(gone)
    return counts, version


def forget(collection, keys):
    """Papers removed from a synced collection: their items leave the Zotero collection at the next sync."""
    if collection.zotero_key and keys:
        known = [k for k in collection.zotero_removed.split(',') if k]
        collection.zotero_removed = ','.join(dict.fromkeys(known + list(keys)))
        collection.save(update_fields=['zotero_removed'])


def take_out(account, collection):
    for key in [k for k in collection.zotero_removed.split(',') if k]:
        try:
            item = get(account, 'items/%s' % key)
        except Unavailable:
            continue
        have = (item.get('data') or {}).get('collections') or []
        if collection.zotero_key in have:
            patch(account, item, {'collections': [c for c in have if c != collection.zotero_key]})
    collection.zotero_removed = ''
    collection.save(update_fields=['zotero_removed'])


def collection_entries(collection, lang, rows=None):
    from . import collection_tools
    rows = list(collection.papers.all()) if rows is None else rows
    facts = collection_tools.facts(rows)
    entries, kept = [], []
    for row in rows:
        meta = exports.paper_meta(row.arxiv_id, row)
        if meta is None:
            continue
        entries.append(paper_entry(meta, lang, row.note, row.tags, row.reading, notice_tags(facts.get(row.arxiv_id, {})),
                                   key=row.zotero_key))
        kept.append(row)
    return entries, kept


def synchronize(collection, account, lang, paid, key='', full=True, step=lambda name: None):
    """Keeps a collection and its Zotero collection together (key: a Zotero collection to link it to first). Not
    full (the daily sync): only what changed on either side since the last sync is read and written."""
    started = timezone.now()
    link_collection(account, collection, key)
    take_out(account, collection)
    step('pull')
    counts, version = pull(account, collection, paid, step=lambda: step('pull'), full=full)
    step('push')
    rows = list(collection.papers.all())
    if not full and collection.zotero_synced:
        since = collection.zotero_synced
        rows = [r for r in rows if not r.zotero_key or r.added >= started or (r.changed and r.changed > since)]
    entries, rows = collection_entries(collection, lang, rows)
    result = send(account, entries, collection.zotero_key, step=lambda: step('push'))
    for row, item_key in zip(rows, result.pop('keys')):
        if item_key and row.zotero_key != item_key:
            row.zotero_key = item_key
            row.save(update_fields=['zotero_key'])
    Collection.objects.filter(pk=collection.pk).update(zotero_synced=started, zotero_version=version, zotero_lang=lang)
    return dict(counts, **result, open=open_link(account, collection.zotero_key, 'collections'),
                web=web_link(account, collection.zotero_key, 'collections'))


def sync_collection(collection_pk, account_pk, lang, paid, key=''):
    """The sync asked from a page, in the background: everything is read and written."""
    def work(pk):
        collection = Collection.objects.get(pk=collection_pk)
        account = ZoteroAccount.objects.get(pk=account_pk)
        generated.progress(pk, 'link')
        try:
            return synchronize(collection, account, lang, paid, key, step=lambda name: generated.progress(pk, name))
        except Unavailable as e:
            logger.warning('Zotero sync of collection %s failed: %s', collection_pk, e)
            raise generated.SourceUnavailable('zotero')
    return work


def run(now):
    """From the hourly run: the collections kept in sync with Zotero that were not synced for a day, a few at a
    time: what changed on either side, the notices found since (retractions, published versions) as tags, and
    their papers queued for a new editorial check. Returns how many were synced."""
    from django.db.models import Q
    from . import publication_status
    due = Collection.objects.exclude(zotero_key='').filter(
        Q(zotero_synced__isnull=True) | Q(zotero_synced__lt=now - DAILY)).select_related('user').order_by('zotero_synced')
    done = 0
    for collection in due[:DAILY_PER_RUN * 4]:
        if done >= DAILY_PER_RUN:
            break
        account = ZoteroAccount.objects.filter(user=collection.user).first()
        if account is None or not account.api_key or collection.zotero_library != account.prefix() \
                or not access.has_tool(collection.user, 'zotero'):
            continue
        for arxiv_id in collection.papers.values_list('arxiv_id', flat=True):
            publication_status.queue(arxiv_id)
        try:
            synchronize(collection, account, collection.zotero_lang or 'en', access.has_tool(collection.user, 'collections'),
                        full=False)
            write_notices(collection)
            done += 1
        except Exception:
            logger.warning('Daily Zotero sync of collection %s failed', collection.pk, exc_info=True)
            Collection.objects.filter(pk=collection.pk).update(zotero_synced=now)  # tried again tomorrow
    return done


def write_notices(collection):
    """After the papers of a synced collection were checked again: their notices as tags in Zotero."""
    account = ZoteroAccount.objects.filter(user=collection.user).first()
    if account is None or not collection.zotero_key or collection.zotero_library != account.prefix() \
            or not access.has_tool(collection.user, 'zotero'):
        return 0
    from . import collection_tools
    rows = [r for r in collection.papers.all() if r.zotero_key]
    facts = collection_tools.facts(rows)
    written = 0
    for row in rows:
        tags = notice_tags(facts.get(row.arxiv_id, {}))
        if tags:
            try:
                written += complete(account, get(account, 'items/%s' % row.zotero_key), tags=tags)
            except Unavailable:
                continue
    return written


# --- a check read from Zotero ------------------------------------------------------------------------------

SHORT_TITLE_SKIP = {'a', 'an', 'the', 'some', 'from', 'on', 'in', 'to', 'of', 'do', 'with', 'and', 'or', 'for', 'at',
                    'by', 'as', 'via', 'der', 'die', 'das', 'ein', 'eine', 'un', 'une', 'la', 'le', 'les', 'el', 'los',
                    'las', 'de', 'des', 'del', 'du', 'et', 'und'}


def _ascii(text):
    return unicodedata.normalize('NFKD', text or '').encode('ascii', 'ignore').decode()


def citation_keys(data):
    """The keys a manuscript may cite a Zotero item with, the likeliest first: the one pinned in Zotero (Better
    BibTeX's 'Citation Key:'), Better BibTeX's default (lee + CloudsHotPlanet + 2021), and ours (lee2021clouds)."""
    keys = []
    pinned = data.get('citationKey') or (re.search(r'^\s*(?:Citation Key|bibtex)\s*:\s*(\S+)', data.get('extra') or '',
                                                   re.I | re.M) or [None, ''])[1]
    if pinned:
        keys.append(pinned)
    creators = [c for c in data.get('creators') or [] if c.get('creatorType', 'author') == 'author'] or \
        list(data.get('creators') or [])
    first = creators[0] if creators else {}
    family = first.get('lastName') or first.get('name') or ''
    year = (re.search(r'\d{4}', str(data.get('date') or '')) or [None])[0] or ''
    words = [w for w in re.findall(r'[A-Za-z0-9]+', _ascii(data.get('title', ''))) if w.lower() not in SHORT_TITLE_SKIP][:3]
    short = ''.join(w[:1].upper() + w[1:] for w in words)
    for name in (family, family.split()[-1] if family.split() else ''):
        name = re.sub(r'[^A-Za-z0-9]', '', _ascii(name)).lower()
        if name:
            keys.append(name + short + year)
    full = ' '.join(filter(None, [first.get('firstName', ''), family]))
    if full:
        keys.append(exports.cite_key({'authors': [full], 'year': year, 'title': data.get('title', '')}))
    return list(dict.fromkeys(keys))


def keyed(found, cited):
    """The BibTeX of a Zotero collection with each entry under the key the manuscript cites it with (or the key
    pinned in Zotero), and {key: Zotero item key}. found: [{'bibtex', 'item', 'keys'}]."""
    entries, library, used = [], {}, set()
    for entry in found:
        head = re.match(r'\s*@\w+\s*\{\s*([^,\s]+)\s*,', entry['bibtex'])
        if not head:
            continue
        key = next((k for k in entry['keys'] if k in cited and k not in used), None) or \
            (entry['keys'][0] if entry['pinned'] and entry['keys'][0] not in used else head.group(1))
        used.add(key)
        library[key] = entry['item']
        entries.append(entry['bibtex'][:head.start(1)] + key + entry['bibtex'][head.end(1):])
    return '\n\n'.join(entries), library


def own_full_text(account, item_key):
    """The text of the PDF a researcher keeps in Zotero for an item: Zotero's full-text index, else the file itself.
    Read for their own check only, never stored."""
    from .missed import pdf_text
    children = get(account, 'items/%s/children' % item_key)
    for child in children:
        data = child.get('data') or {}
        if data.get('itemType') != 'attachment' or data.get('contentType') != 'application/pdf':
            continue
        try:
            text = (get(account, 'items/%s/fulltext' % child['key']) or {}).get('content') or ''
        except Unavailable:
            text = ''
        if len(text) > sources.MIN_TEXT:
            return text
        if data.get('linkMode') in ('imported_file', 'imported_url'):
            try:
                pdf = call(account, 'GET', 'items/%s/file' % child['key']).content
                text = pdf_text(pdf) if len(pdf) <= MAX_PDF else ''
            except Exception:
                text = ''
            if len(text) > sources.MIN_TEXT:
                return text
    return ''


def reader(account, library):
    """For the bibliography check: key -> the full text of the paper from the researcher's own PDF in Zotero."""
    def read(key):
        if key not in library:
            return None
        try:
            text = own_full_text(account, library[key])
        except Exception:
            logger.info('Zotero full text of %s could not be read', key, exc_info=True)
            return None
        return {'text': text, 'source': 'zotero', 'arxiv_id': key} if text else None
    return read


# --- a bibliography check written back ----------------------------------------------------------------------

def check_note(issues, report_url, lang, published_key=''):
    from .refcheck import KIND_LABELS
    with translation.override(lang):
        lines = []
        for issue in issues:
            text = escape(str(KIND_LABELS.get(issue['kind'], issue['kind'])))
            if issue.get('journal'):
                text += ': %s' % escape(issue['journal'])
            for field in ('doi', 'notice'):
                if issue.get(field):
                    text += ' <a href="https://doi.org/%s">doi:%s</a>' % (escape(issue[field]), escape(issue[field]))
            lines.append('<li>%s</li>' % text)
        body = '<h1>%s</h1><p><em>%s</em></p><ul>%s</ul>%s<p><a href="%s">%s</a></p>' % (
            escape(_('Bibliography check')),
            escape(_('Found by SummarizePaper on %(date)s in Crossref, OpenAlex, Retraction Watch and arXiv.') % {
                'date': timezone.localdate().isoformat()}),
            ''.join(lines),
            '<p>%s</p>' % escape(_('Its published version was added to your library next to it, as a related item.'))
            if published_key else '', report_url, report_url)
    return note(body, report_url)


def published_item(fields, preprint_data):
    """The journal version of a preprint, from the corrected BibTeX entry of the check."""
    from .refcheck import plain as bib_plain
    def value(name):
        return bib_plain(fields.get(name, ''))
    item = {'itemType': 'journalArticle', 'title': value('title') or preprint_data.get('title', ''),
            'creators': [creator(a) for a in re.split(r'\s+and\s+', value('author')) if a.strip()][:100]
            or preprint_data.get('creators', []),
            'publicationTitle': value('journal'), 'volume': value('volume'), 'issue': value('number'),
            'pages': value('pages'), 'date': value('year'), 'DOI': value('doi'),
            'url': 'https://doi.org/%s' % value('doi') if value('doi') else '', 'tags': tag_list([TAG])}
    arxiv = re.search(r'\d{4}\.\d{4,5}', str(preprint_data.get('archiveID', '')) + str(preprint_data.get('url', '')))
    if arxiv:
        item['extra'] = 'arXiv: %s' % arxiv.group()
    return item


def apply_refcheck(account_pk, row_pk):
    """Writes the findings of a check into the library: tags and a note on each item concerned; for a preprint
    published since, its journal version added and related to it. Marks 'intended' and 'ignored' are respected."""
    def work(pk):
        from .refcheck import parse_bibtex
        account = ZoteroAccount.objects.get(pk=account_pk)
        row = Generated.objects.get(pk=row_pk)
        result = row.data()
        report_url = '%s/%s/refcheck/%s/' % (digest.BASE_URL, row.lang, row.key)
        tagged, added, missing = 0, 0, []
        try:
            items = collection_items(account, result['zotero_collection'])[0] if result.get('zotero_collection') else None
            for index, problem in enumerate(result.get('problems', [])):
                if index % 5 == 4:
                    generated.progress(pk, 'write')
                issues = [i for i in problem['issues'] if i['kind'] in NOTICE_TAGS and i.get('mark') not in SKIPPED_MARKS]
                if not issues:
                    continue
                arxiv = problem.get('arxiv') or ''
                wanted = identifiers(arxiv, [problem.get('doi')], problem['title'])
                item = find(account, wanted, problem['title'], items) or (find(account, wanted, problem['title'])
                                                                         if items is not None else None)
                if item is None:
                    missing.append(problem['title'])
                    continue
                published_key = ''
                issue = next((i for i in issues if i['kind'] == 'published' and i.get('bibtex')), None)
                if issue is not None:
                    parsed = parse_bibtex(issue['bibtex'])
                    fields = parsed[0]['fields'] if parsed else {}
                    doi = sources.clean_doi(fields.get('doi', '') or issue.get('doi', '')) or ''
                    there = find(account, identifiers('', [doi]), fields.get('title') or problem['title']) if doi else None
                    if there is None and fields.get('journal'):
                        new = published_item(fields, item.get('data') or {})
                        new['collections'] = (item.get('data') or {}).get('collections') or []
                        new['relations'] = {'dc:relation': [item_uri(account, item['key'])]}
                        published_key = create(account, [new]).get(0, '')
                        added += bool(published_key)
                    elif there is not None:
                        published_key = there['key']
                        complete(account, there, relations=[item_uri(account, item['key'])])
                complete(account, item, tags=[NOTICE_TAGS[i['kind']] for i in issues],
                         relations=[item_uri(account, published_key)] if published_key else ())
                add_notes(account, item['key'], [check_note(issues, report_url, row.lang, published_key)])
                tagged += 1
        except Unavailable as e:
            logger.warning('Writing check %s into Zotero failed: %s', row.key, e)
            raise generated.SourceUnavailable('zotero')
        return {'tagged': tagged, 'added': added, 'missing': missing[:30]}
    return work


# --- a literature review ------------------------------------------------------------------------------------

def review_statements(result):
    """{paper number: [(section, statement, the paper's sentence supporting it)]}."""
    found = {}

    def add(section, item):
        quotes = {s['n']: s['quote'] for s in item.get('support') or []}
        for n in item.get('papers') or []:
            found.setdefault(n, []).append((section, item.get('text', ''), quotes.get(n, '')))
    for theme in result.get('themes', []):
        for item in theme.get('findings', []):
            add(theme['name'], item)
    for name, title in (('consensus', 'Where the papers agree'), ('gaps', 'What they leave open')):
        for item in result.get(name, []):
            add(_(title), item)
    for item in result.get('disagreements', []):
        for side in item.get('sides', []):
            add(_('Where they disagree'), side)
    return found


def review_note(result, n, url, lang):
    with translation.override(lang):
        lines = ''.join('<li><strong>%s</strong>: %s%s</li>' % (
            escape(section), escape(text), '<blockquote>%s</blockquote>' % escape(quote) if quote else
            ' <em>(%s)</em>' % escape(_('no sentence of this paper quoted'))) for section, text, quote in
            review_statements(result).get(n, []))
        body = '<h1>%s</h1><p><em>%s</em></p><ul>%s</ul><p><a href="%s">%s</a></p>' % (
            escape(_('Literature review: %(title)s') % {'title': result.get('title', '')}),
            escape(_('Exploratory review written with SummarizePaper on %(date)s; check each statement in the paper.') % {
                'date': result.get('searched', '') or timezone.localdate().isoformat()}),
            lines or '<li>%s</li>' % escape(_('Included in the review, not cited by a statement.')), url, url)
    return note(body, url)


def send_review(account_pk, row_pk):
    """A review as a Zotero collection, a subcollection per approach, each paper with its statements."""
    def work(pk):
        from . import literature
        account = ZoteroAccount.objects.get(pk=account_pk)
        row = Generated.objects.get(pk=row_pk)
        result = row.data()
        url = '%s/%s/literature-review/%s/' % (digest.BASE_URL, row.lang, row.key)
        try:
            known = collections(account)
            with translation.override(row.lang):
                name = result.get('title') or _('Literature review')
            parent = ensure_collection(account, name, '', known)
            metas = literature.metas(result)
            entries = [{'meta': m, 'notes': [review_note(result, p['n'], url, row.lang)], 'tags': [], 'key': ''}
                       for m, p in zip(metas, result['papers'])]
            generated.progress(pk, 'write')
            sent = send(account, entries, parent, step=lambda: generated.progress(pk, 'write'))
            keys = {p['n']: k for p, k in zip(result['papers'], sent['keys'])}
            for theme in result.get('themes', []):
                sub = ensure_collection(account, theme['name'], parent, known)
                for n in theme.get('papers') or []:
                    if keys.get(n):
                        complete(account, get(account, 'items/%s' % keys[n]), [sub])
        except Unavailable as e:
            logger.warning('Review %s could not be sent to Zotero: %s', row.key, e)
            raise generated.SourceUnavailable('zotero')
        return dict({k: v for k, v in sent.items() if k != 'keys'}, open=open_link(account, parent, 'collections'),
                    web=web_link(account, parent, 'collections'))
    return work


# --- OAuth --------------------------------------------------------------------------------------------------

def oauth_ready():
    return bool(getattr(settings, 'ZOTERO_CLIENT_KEY', '') and getattr(settings, 'ZOTERO_CLIENT_SECRET', ''))


def _quote(value):
    return urllib.parse.quote(str(value), safe='~')


def oauth_post(url, token_secret='', **params):
    """A request signed with OAuth 1.0a (HMAC-SHA1); the answer's form fields."""
    params = dict(params, oauth_consumer_key=settings.ZOTERO_CLIENT_KEY, oauth_nonce=secrets.token_hex(16),
                  oauth_signature_method='HMAC-SHA1', oauth_timestamp=str(int(time.time())))
    base = '&'.join(['POST', _quote(url), _quote('&'.join('%s=%s' % (_quote(k), _quote(v)) for k, v in sorted(params.items())))])
    key = '%s&%s' % (_quote(settings.ZOTERO_CLIENT_SECRET), _quote(token_secret))
    params['oauth_signature'] = base64.b64encode(hmac.new(key.encode(), base.encode(), hashlib.sha1).digest()).decode()
    header = 'OAuth ' + ', '.join('%s="%s"' % (_quote(k), _quote(v)) for k, v in sorted(params.items()))
    response = requests.post(url, headers={'Authorization': header}, timeout=20)
    response.raise_for_status()
    return dict(urllib.parse.parse_qsl(response.text))


@login_required
def oauth_start(request):
    if not oauth_ready() or not access.has_tool(request.user, 'zotero'):
        raise Http404
    callback = request.build_absolute_uri('/%s/integrations/zotero/callback/' % get_language())
    try:
        token = oauth_post(OAUTH + '/request', oauth_callback=callback)
    except requests.RequestException:
        logger.warning('Zotero OAuth request token failed')
        return redirect('/%s/integrations/?error=oauth' % get_language())
    request.session['zotero_oauth'] = {'token': token.get('oauth_token', ''), 'secret': token.get('oauth_token_secret', '')}
    query = urllib.parse.urlencode({'oauth_token': token.get('oauth_token', ''), 'library_access': 1, 'notes_access': 1,
                                    'write_access': 1, 'all_groups': 'write', 'name': 'SummarizePaper'})
    return redirect('%s/authorize?%s' % (OAUTH, query))


@login_required
def oauth_callback(request):
    pending = request.session.pop('zotero_oauth', None)
    if not oauth_ready() or not pending or request.GET.get('oauth_token') != pending.get('token'):
        return redirect('/%s/integrations/?error=oauth' % get_language())
    try:
        answer = oauth_post(OAUTH + '/access', pending['secret'], oauth_token=pending['token'],
                            oauth_verifier=request.GET.get('oauth_verifier', ''))
    except requests.RequestException:
        logger.warning('Zotero OAuth access token failed')
        return redirect('/%s/integrations/?error=oauth' % get_language())
    library_id = answer.get('userID', '')
    for candidate in (answer.get('oauth_token_secret', ''), answer.get('oauth_token', '')):
        info = check_key(library_id, candidate) if candidate else None
        if may_write(info, library_id):
            save_account(request.user, library_id, candidate, answer.get('username') or info.get('username', ''))
            return redirect('/%s/integrations/?saved=1' % get_language())
    return redirect('/%s/integrations/?error=oauth' % get_language())


def save_account(user, library_id, api_key, username=''):
    account = ZoteroAccount.objects.filter(user=user).first() or ZoteroAccount(user=user)
    if account.library_id != library_id:
        account.library, account.library_name, account.inbox, account.inbox_name = '', '', '', ''
    account.library_id, account.api_key, account.username = library_id, api_key, username
    account.save()
    return account


# --- pages --------------------------------------------------------------------------------------------------

def account_of(request):
    """(account, refusal): the visitor's Zotero account, or the JSON answer to give instead."""
    if not access.has_tool(request.user, 'zotero'):
        return None, JsonResponse({'status': 'premium'}, status=403)
    account = ZoteroAccount.objects.filter(user=request.user).first()
    if account is None or not account.api_key:
        return None, JsonResponse({'status': 'setup', 'url': '/%s/integrations/' % get_language()})
    return account, None


@sensitive_post_parameters('api_key')
def integrations_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/integrations.html', {'anonymous': True})
    paid = access.has_tool(request.user, 'zotero')
    account = ZoteroAccount.objects.filter(user=request.user).first()
    error, error_code, library_id = False, request.GET.get('error', ''), ''
    if request.method == 'POST' and paid:
        action = request.POST.get('action')
        if action == 'disconnect':
            ZoteroAccount.objects.filter(user=request.user).delete()
            return redirect('/%s/integrations/' % get_language())
        if action == 'library' and account is not None:
            wanted = request.POST.get('library', '')
            names = dict(groups(account))
            if wanted == '' or wanted in names:
                account.library, account.library_name = wanted, names.get(wanted, '')
                account.inbox, account.inbox_name = '', ''
                account.save(update_fields=['library', 'library_name', 'inbox', 'inbox_name'])
            return redirect('/%s/integrations/?saved=1' % get_language())
        if action == 'inbox' and account is not None:
            account.inbox = request.POST.get('inbox', '') if re.fullmatch(r'[A-Z0-9]{8}', request.POST.get('inbox', '')) else ''
            account.inbox_name = request.POST.get('inbox_name', '')[:200] if account.inbox else ''
            account.save(update_fields=['inbox', 'inbox_name'])
            return redirect('/%s/integrations/?saved=1' % get_language())
        library_id = request.POST.get('library_id', '').strip()
        api_key = request.POST.get('api_key', '').strip()
        if not re.fullmatch(r'[0-9]{1,40}', library_id):
            error_code = 'user_id'
        elif not re.fullmatch(r'[A-Za-z0-9]{1,100}', api_key):
            error_code = 'api_key'
        else:
            info = check_key(library_id, api_key)
            if may_write(info, library_id):
                save_account(request.user, library_id, api_key, info.get('username', ''))
                return redirect('/%s/integrations/?saved=1' % get_language())
            error_code = ''
        error = True
    connected = account is not None and bool(account.api_key)
    return render(request, 'summarizer/integrations.html', {
        'paid': paid, 'account': account if connected else None, 'expired': account is not None and not connected,
        'error': error, 'error_code': error_code, 'library_id': library_id[:40], 'saved': request.GET.get('saved') == '1',
        'oauth': oauth_ready(), 'libraries': groups(account) if connected and paid else []})


@login_required
def collections_view(request):
    """The library's collections, for the pickers (import, check, digest, scoop watch, inbox). JSON."""
    account, refused = account_of(request)
    if refused:
        return refused
    try:
        found = collections(account)
    except Unavailable:
        return JsonResponse({'status': 'unavailable'})
    names = {c['key']: c['name'] for c in found}

    def path(c):
        return '%s / %s' % (names[c['parent']], c['name']) if c['parent'] in names else c['name']
    return JsonResponse({'status': 'done', 'collections': sorted(
        [{'key': c['key'], 'name': path(c), 'items': c['items']} for c in found], key=lambda c: c['name'].lower())})


@login_required
def references_view(request):
    """The arXiv ids and DOIs of a Zotero collection, the last added first (?n= at most), for the digest's and
    the scoop watch's reference papers. JSON."""
    account, refused = account_of(request)
    if refused:
        return refused
    key = request.GET.get('key', '')
    if not re.fullmatch(r'[A-Z0-9]{8}', key):
        return JsonResponse({'status': 'unknown'}, status=400)
    try:
        items = get(account, 'collections/%s/items/top' % key, {'sort': 'dateAdded', 'direction': 'desc', 'limit': 100})
    except Unavailable:
        return JsonResponse({'status': 'unavailable'})
    refs = [r for r in dict.fromkeys(zotero_ref(i.get('data') or {}) for i in items) if r]
    limit = max(1, min(int(request.GET.get('n', '5')) if request.GET.get('n', '').isdigit() else 5, 50))
    return JsonResponse({'status': 'done', 'refs': refs[:limit], 'without': len(items) - len(refs)})


@require_POST
def send_view(request):
    """Sends one paper (arxiv_id), a collection (collection: kept in sync), a literature review (literature) or
    the papers a draft misses (missed) to Zotero, in the background."""
    account, refused = account_of(request)
    if refused:
        return refused
    lang = get_language()
    if request.POST.get('literature'):
        row = Generated.objects.filter(kind='literature', key=request.POST['literature'], user=request.user,
                                       status='done').first()
        if row is None:
            raise Http404
        work = send_review(account.pk, row.pk)
    elif request.POST.get('collection'):
        from .library import editable
        collection = editable(request, request.POST['collection'])
        if collection.user_id == request.user.id:
            work = sync_collection(collection.pk, account.pk, lang, access.has_tool(request.user, 'collections'))
            row = generated.start('zotero_sync', uuid.uuid4().hex, lang, work, user=request.user)
            return JsonResponse({'status': row.status, 'id': row.pk})
        # edited with its owner: copied once into the editor's library, only the owner keeps it in sync
        metas = [m for m in (exports.paper_meta(r.arxiv_id, r) for r in collection.papers.all()) if m]
        if not metas:
            raise Http404
        work = send_papers(account.pk, metas, lang, collection.name)
    else:
        if request.POST.get('missed'):
            from . import missed
            metas = missed.zotero_metas(request.user, request.POST['missed'])
            with translation.override(lang):
                name = '%s · %s' % (_('What did I miss?'), timezone.localdate().isoformat())
        else:
            metas = [m for m in [exports.paper_meta(request.POST.get('arxiv_id', ''))] if m]
            name = ''
        if not metas:
            raise Http404
        work = send_papers(account.pk, metas, lang, name)
    row = generated.start('zotero', uuid.uuid4().hex, lang, work, user=request.user)
    return JsonResponse({'status': row.status, 'id': row.pk})


def send_papers(account_pk, metas, lang, name=''):
    """Papers into a new collection with this name, or into the account's collection for single papers."""
    def work(pk):
        account = ZoteroAccount.objects.get(pk=account_pk)
        try:
            target = ensure_collection(account, name) if name else account.inbox
            result = send(account, [paper_entry(m, lang) for m in metas], target,
                          step=lambda: generated.progress(pk, 'write'))
        except Unavailable as e:
            logger.warning('Papers could not be sent to Zotero: %s', e)
            raise generated.SourceUnavailable('zotero')
        keys = result.pop('keys')
        if len(metas) == 1 and keys[0]:
            result.update(open=open_link(account, keys[0]), web=web_link(account, keys[0]))
        elif target:
            result.update(open=open_link(account, target, 'collections'), web=web_link(account, target, 'collections'))
        return result
    return work


@login_required
@require_POST
def open_view(request):
    """A Zotero collection opened as a collection of SummarizePaper, kept in sync with it."""
    from . import library
    account, refused = account_of(request)
    if refused:
        return refused
    key, name = request.POST.get('key', ''), request.POST.get('name', '').strip()[:120] or 'Zotero'
    if not re.fullmatch(r'[A-Z0-9]{8}', key):
        return JsonResponse({'status': 'unknown'}, status=400)
    paid = access.has_tool(request.user, 'collections')
    linked = Collection.objects.filter(user=request.user, zotero_key=key, zotero_library=account.prefix()).first()
    if linked is None:
        if Collection.objects.filter(user=request.user).count() >= library.limits(paid)['collections']:
            return JsonResponse({'status': 'too_many'})
        linked = Collection.objects.create(user=request.user, name=name)
    row = generated.start('zotero_sync', uuid.uuid4().hex, get_language(),
                          sync_collection(linked.pk, account.pk, get_language(), paid, key), user=request.user)
    return JsonResponse({'status': row.status, 'id': row.pk, 'url': '/%s/collections/%d/' % (get_language(), linked.pk)})


@login_required
@require_POST
def refcheck_view(request, key):
    """Writes a bibliography check's findings into the library."""
    account, refused = account_of(request)
    if refused:
        return refused
    row = Generated.objects.filter(kind='refcheck', key=key, user=request.user, status='done').first()
    if row is None:
        raise Http404
    job = generated.start('zotero_refcheck', uuid.uuid4().hex, get_language(), apply_refcheck(account.pk, row.pk),
                          user=request.user)
    return JsonResponse({'status': job.status, 'id': job.pk})
