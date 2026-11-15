"""The site's Bluesky accounts: each theme of the weekly letters (letters.TOPICS) posts its letter once it is written,
then the papers the letter picked, one a day, each with a link card to its page (the picture of social.py).

Nothing here calls a language model: the letters and the summaries of their papers are already written (letters.py
summarizes the papers of the week's letters so that their pages are ready). Posts are in English, from the English
letters, and each link carries utm_source=bluesky-bot, which the conversions page counts.

The accounts are in the BLUESKY_ACCOUNTS setting (an environment variable holding JSON): {"ai": {"handle":
"ai.summarizepaper.com", "password": "<app password>"}, "*": {...}}. "*" posts the themes that have no account of
their own; without "*", those themes are not posted. A theme posts at most once a day, after START_HOUR (UTC), and
an account at most once an hourly run, so that a shared account spreads its posts over the afternoon.

The same posts go to Mastodon (mastodon.py, MASTODON_ACCOUNTS) with run(network='mastodon'), tracked apart.
"""
import datetime
import io
import json
import logging
import re

import requests
from django.conf import settings
from PIL import Image

from . import letters, social
from .models import BlueskyPost, Letter, SummaryPaper

logger = logging.getLogger(__name__)

PDS = 'https://bsky.social/xrpc/'
START_HOUR = 13  # UTC: morning in America, afternoon in Europe
LETTER_AGE = datetime.timedelta(days=8)  # a letter older than this is not posted, nor are its papers
MAX_TEXT = 290  # Bluesky counts 300 graphemes; characters are at least as many
MAX_THUMB = 950_000  # bytes; Bluesky refuses blobs over 1,000,000
UTM = 'utm_source=bluesky-bot&utm_medium=social&utm_campaign=%s'
TAG = re.compile(r'(?<!\S)#(\w+)')
TIMEOUT = 30


def accounts(network='bluesky'):
    if network == 'mastodon':
        from . import mastodon
        return mastodon.accounts()
    found = getattr(settings, 'BLUESKY_ACCOUNTS', None) or {}
    return {k: v for k, v in found.items() if isinstance(v, dict) and v.get('handle') and v.get('password')}


def account_for(topic, found=None):
    found = accounts() if found is None else found
    return found.get(topic) or found.get('*')


def account_name(account):
    return account.get('handle') or account.get('instance', '')


# --- what to post next --------------------------------------------------------------------------------------

def _posted(topic, kind, key, network='bluesky'):
    return BlueskyPost.objects.filter(topic=topic, kind=kind, key=key, network=network).exists()


def next_item(topic, today, network='bluesky'):
    """('letter', key, letter) for a letter not posted yet, else ('paper', arxiv_id, letter) for its next paper
    whose English summary is written, else None."""
    from .presskit import paper_id
    letter = (Letter.objects.filter(topic=topic, lang='en', day__gte=today - LETTER_AGE)
              .order_by('-day').first())
    if letter is None:
        return None
    key = '%s/%s' % (topic, letter.day.isoformat())
    if not _posted(topic, 'letter', key, network):
        return 'letter', key, letter
    content = json.loads(letter.content)
    for raw in content.get('order') or list(content.get('papers', {})):
        arxiv_id = paper_id(raw) or raw
        if _posted(topic, 'paper', arxiv_id, network):
            continue
        if SummaryPaper.objects.filter(paper__arxiv_id=arxiv_id, lang='en').exists():
            return 'paper', arxiv_id, letter
    return None


def _with_utm(url, topic, network='bluesky'):
    return '%s%s%s' % (url, '&' if '?' in url else '?', UTM.replace('bluesky', network) % topic)


def letter_post(letter, network='bluesky'):
    """(text, link, card title, card description, picture bytes) of a letter."""
    share = social.letter_share(letter, [], 'en')
    response = social.letter_image(None, 'en', letter.topic, letter.day.isoformat())
    return (share['short'], _with_utm(share['url'], letter.topic, network), social.plain(letter.title),
            social.shorten(social.plain(letter.dek), 280), response.content)


def paper_post(arxiv_id, topic, network='bluesky'):
    from . import insight
    from .models import ArxivPaper
    paper = ArxivPaper.objects.get(arxiv_id=arxiv_id)
    sumpaper = SummaryPaper.objects.filter(paper=paper, lang='en').first()
    share = social.paper_share(paper, sumpaper, insight.for_page(arxiv_id, 'en'), 'en')
    response = social.paper_image(None, 'en', arxiv_id)
    hook = share['short'].split('\n\n')[0]
    return (share['short'], _with_utm(share['url'], topic, network), social.shorten(social.plain(paper.title), 200),
            social.shorten(hook, 280), getattr(response, 'content', b''))


# --- the Bluesky API ------------------------------------------------------------------------------------------

def facets(text):
    """The hashtags of the text as Bluesky tag facets (offsets in UTF-8 bytes)."""
    found = []
    for match in TAG.finditer(text):
        start = len(text[:match.start()].encode('utf-8'))
        end = start + len(match.group(0).encode('utf-8'))
        found.append({'index': {'byteStart': start, 'byteEnd': end},
                      'features': [{'$type': 'app.bsky.richtext.facet#tag', 'tag': match.group(1)}]})
    return found


def thumbnail(png_bytes):
    """A JPEG of the picture under Bluesky's size limit, or None."""
    if not png_bytes:
        return None
    try:
        image = Image.open(io.BytesIO(png_bytes)).convert('RGB')
    except Exception:
        return None
    for quality in (85, 70, 55):
        out = io.BytesIO()
        image.save(out, 'JPEG', quality=quality, optimize=True)
        if out.tell() <= MAX_THUMB:
            return out.getvalue()
    return None


def session(account):
    response = requests.post(PDS + 'com.atproto.server.createSession', timeout=TIMEOUT,
                             json={'identifier': account['handle'], 'password': account['password']})
    response.raise_for_status()
    data = response.json()
    return data['accessJwt'], data['did']


def post(account, text, link, title, description, picture, now):
    """Publish one post with a link card; its at:// URI."""
    token, did = session(account)
    headers = {'Authorization': 'Bearer %s' % token}
    external = {'uri': link, 'title': title[:300], 'description': description[:1000]}
    jpeg = thumbnail(picture)
    if jpeg:
        blob = requests.post(PDS + 'com.atproto.repo.uploadBlob', data=jpeg, timeout=TIMEOUT,
                             headers=dict(headers, **{'Content-Type': 'image/jpeg'}))
        blob.raise_for_status()
        external['thumb'] = blob.json()['blob']
    text = text if len(text) <= MAX_TEXT else social.shorten(text, MAX_TEXT)
    record = {'$type': 'app.bsky.feed.post', 'text': text, 'langs': ['en'],
              'createdAt': now.astimezone(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.000Z'),
              'embed': {'$type': 'app.bsky.embed.external', 'external': external}}
    tags = facets(text)
    if tags:
        record['facets'] = tags
    response = requests.post(PDS + 'com.atproto.repo.createRecord', headers=headers, timeout=TIMEOUT,
                             json={'repo': did, 'collection': 'app.bsky.feed.post', 'record': record})
    response.raise_for_status()
    return response.json().get('uri', '')


# --- the hourly step ------------------------------------------------------------------------------------------

def run(now, dry_run=False, network='bluesky'):
    """Post what is due on one network ('bluesky' or 'mastodon', whose posts are told apart in BlueskyPost.network);
    the number of posts (with dry_run, of posts that would be made)."""
    found = accounts(network)
    if not found or now.astimezone(datetime.timezone.utc).hour < START_HOUR:
        return 0
    if network == 'mastodon':
        from .mastodon import post as publish
    else:
        publish = post
    today = now.date()
    used = set()  # one post per account and run
    done = 0
    for topic in letters.TOPICS:
        account = account_for(topic, found)
        if account is None or account_name(account) in used:
            continue
        if BlueskyPost.objects.filter(topic=topic, network=network, created__date=today).exists():
            continue
        item = next_item(topic, today, network)
        if item is None:
            continue
        kind, key, letter = item
        try:
            content = letter_post(letter, network) if kind == 'letter' else paper_post(key, topic, network)
            if dry_run:
                logger.info('%s %s (%s): %s', network, topic, kind, content[0])
                done += 1
                continue
            uri = publish(account, *content, now=now)
        except Exception:
            logger.exception('%s post %s %s failed', network, topic, key)
            used.add(account_name(account))
            continue
        BlueskyPost.objects.create(topic=topic, kind=kind, key=key, network=network, account=account_name(account), uri=uri)
        used.add(account_name(account))
        done += 1
    return done
