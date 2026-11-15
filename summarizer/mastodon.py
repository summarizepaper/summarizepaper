"""The site's Mastodon account: the posts of bluesky.py (a weekly letter, then its papers one a day), published with
the Mastodon API. The picture is attached with a description (Mastodon would fetch the link's own card from its
server, which Cloudflare may challenge), and the link is in the text.

MASTODON_ACCOUNTS (an environment variable holding JSON) gives the accounts by theme like BLUESKY_ACCOUNTS:
{"*": {"instance": "https://fediscience.org", "token": "<access token with write:statuses and write:media>"}}.
"""
import requests
from django.conf import settings

from . import social

MAX_TEXT = 490  # Mastodon's default limit is 500 characters, a link counted as 23
TIMEOUT = 60


def accounts():
    found = getattr(settings, 'MASTODON_ACCOUNTS', None) or {}
    return {k: v for k, v in found.items() if isinstance(v, dict) and v.get('instance') and v.get('token')}


def status_text(text, link):
    room = MAX_TEXT - 23 - 2
    return '%s\n\n%s' % (text if len(text) <= room else social.shorten(text, room), link)


def post(account, text, link, title, description, picture, now=None):
    """Publish one status with the picture attached; its URL."""
    base = account['instance'].rstrip('/')
    headers = {'Authorization': 'Bearer %s' % account['token']}
    data = {'status': status_text(text, link), 'language': 'en', 'visibility': 'public'}
    if picture:
        media = requests.post(base + '/api/v2/media', headers=headers, timeout=TIMEOUT,
                              files={'file': ('card.png', picture, 'image/png')},
                              data={'description': ('%s. %s' % (title, description))[:1500]})
        if media.ok:
            data['media_ids[]'] = [media.json()['id']]
    response = requests.post(base + '/api/v1/statuses', headers=headers, data=data, timeout=TIMEOUT)
    if response.status_code == 422 and 'media_ids[]' in data:  # the picture still processing: post without it
        data.pop('media_ids[]')
        response = requests.post(base + '/api/v1/statuses', headers=headers, data=data, timeout=TIMEOUT)
    response.raise_for_status()
    return response.json().get('url', '')
