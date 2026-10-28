"""The Premium digest read aloud.

Each paper of a digest is read once per language (title, then its 2-sentence
blurb) and the clip is shared by every digest that lists it. The email links
to a page that plays the clips of that digest one after the other: MP3 frames
can be joined end to end, so the 'episode' is the clips put together.
"""
import logging
import re

import requests
from django.conf import settings
from django.core import signing
from django.http import Http404, HttpResponse
from django.shortcuts import render
from django.utils import timezone, translation
from django.utils.translation import gettext as _

from . import usage
from .models import DigestItem, PaperAudio, SiteSettings

logger = logging.getLogger(__name__)

ENDPOINT = 'https://api.openai.com/v1/audio/speech'
SALT = 'digest-audio'
MAX_CLIPS = 8  # papers read in one episode
MAX_NEW_PER_RUN = 40
KEEP_DAYS = 10  # clips are a few hundred kB each
INTRO = '_intro'
CHUNK = 1500  # characters read in one request: long texts outlast the read timeout
TIMEOUT = (10, 180)
TRIES = 2


def chunks(text):
    """The text cut at paragraphs, then sentences, into pieces of at most CHUNK characters."""
    pieces, current = [], ''
    for sentence in re.split(r'(?<=[.!?…])\s+|\n{2,}', text.strip()):
        if current and len(current) + len(sentence) + 1 > CHUNK:
            pieces.append(current)
            current = ''
        current = ('%s %s' % (current, sentence)).strip()
    if current:
        pieces.append(current)
    return [p[:4000] for p in pieces]


def _speak_once(cfg, text):
    for attempt in range(TRIES):
        try:
            response = requests.post(ENDPOINT, timeout=TIMEOUT, headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
                                     json={'model': cfg.model_tts, 'voice': cfg.tts_voice, 'input': text,
                                           'response_format': 'mp3'})
            response.raise_for_status()
        except (requests.ConnectionError, requests.Timeout):
            if attempt + 1 == TRIES:
                raise
            logger.warning('Text to speech timed out, trying again')
            continue
        usage.add(cfg.model_tts, len(text), 0, 'audio')  # speech is priced by character
        return response.content


def speak(text):
    """The text read aloud as MP3, one request per chunk: MP3 frames can be joined end to end."""
    cfg = SiteSettings.current()
    return b''.join(_speak_once(cfg, piece) for piece in chunks(text))


_BITRATES = {1: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
             2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160]}
_RATES = {3: [44100, 48000, 32000], 2: [22050, 24000, 16000], 0: [11025, 12000, 8000]}


def mp3_seconds(data):
    """The length of MPEG layer III audio, from its frame headers."""
    data, i, seconds = bytes(data), 0, 0.0
    if data[:3] == b'ID3' and len(data) > 10:
        i = 10 + ((data[6] & 0x7f) << 21 | (data[7] & 0x7f) << 14 | (data[8] & 0x7f) << 7 | (data[9] & 0x7f))
    while i + 4 <= len(data):
        b1, b2 = data[i + 1], data[i + 2]
        version, layer, bitrate, rate = (b1 >> 3) & 3, (b1 >> 1) & 3, b2 >> 4, (b2 >> 2) & 3
        if data[i] != 0xff or b1 & 0xe0 != 0xe0 or version == 1 or layer != 1 or bitrate in (0, 15) or rate == 3:
            i += 1
            continue
        sample_rate = _RATES[version][rate]
        samples = 1152 if version == 3 else 576
        size = samples // 8 * _BITRATES[1 if version == 3 else 2][bitrate] * 1000 // sample_rate + ((b2 >> 1) & 1)
        seconds += samples / sample_rate
        i += size
    return seconds


def mp3_response(request, data):
    """The MP3, or the byte range the player asked for: without ranges, browsers cannot seek in it."""
    match = re.match(r'bytes=(\d*)-(\d*)$', request.headers.get('Range', ''))
    if match and (match.group(1) or match.group(2)):
        if match.group(1):
            start, end = int(match.group(1)), int(match.group(2) or len(data) - 1)
        else:
            start, end = max(0, len(data) - int(match.group(2))), len(data) - 1
        end = min(end, len(data) - 1)
        if start > end:
            response = HttpResponse(status=416)
            response['Content-Range'] = 'bytes */%d' % len(data)
            return response
        response = HttpResponse(data[start:end + 1], content_type='audio/mpeg', status=206)
        response['Content-Range'] = 'bytes %d-%d/%d' % (start, end, len(data))
    else:
        response = HttpResponse(data, content_type='audio/mpeg')
    response['Accept-Ranges'] = 'bytes'
    response['Cache-Control'] = 'private, max-age=86400'
    return response


def clip_text(item):
    return '%s.\n\n%s' % (item.title.rstrip('.'), item.blurb)


def ensure(items, lang, budget):
    """Read aloud the items that have no clip yet; budget is a one-element list of clips this run may make."""
    have = set(PaperAudio.objects.filter(arxiv_id__in=[i.arxiv_id for i in items] + [INTRO], lang=lang)
               .values_list('arxiv_id', flat=True))
    todo = [] if INTRO in have else [(INTRO, None)]
    todo += [(i.arxiv_id, i) for i in items[:MAX_CLIPS] if i.arxiv_id not in have and i.blurb]
    made = []
    for arxiv_id, item in todo:
        if budget[0] <= 0:
            break
        budget[0] -= 1
        if item is None:
            with translation.override(lang):
                text = _('Here are the new papers of your SummarizePaper digest.')
        else:
            text = clip_text(item)
        try:
            data = speak(text)
        except requests.RequestException:
            logger.exception('Text to speech failed for %s', arxiv_id)
            break  # the model or the network is down: no use trying the next clips
        PaperAudio.objects.get_or_create(arxiv_id=arxiv_id, lang=lang, defaults={'mp3': data})
        made.append(arxiv_id)
    return made


def token(subscription, arxiv_ids):
    return signing.dumps([subscription.pk, subscription.lang, list(arxiv_ids)[:MAX_CLIPS]], salt=SALT, compress=True)


def _episode(token_value):
    try:
        pk, lang, ids = signing.loads(token_value, salt=SALT)
    except (signing.BadSignature, ValueError, TypeError):
        raise Http404
    clips = {a.arxiv_id: a for a in PaperAudio.objects.filter(arxiv_id__in=ids + [INTRO], lang=lang)}
    return lang, ids, clips


def listen_page(request, token_value):
    lang, ids, clips = _episode(token_value)
    items = {i.arxiv_id: i for i in DigestItem.objects.filter(arxiv_id__in=ids, lang=lang)}
    tracks = [items[a] for a in ids if a in items and a in clips]
    return render(request, 'summarizer/listen.html', {'tracks': tracks, 'token': token_value, 'lang': lang})


def episode_mp3(request, token_value):
    lang, ids, clips = _episode(token_value)
    order = ([INTRO] if INTRO in clips else []) + [a for a in ids if a in clips]
    if len(order) <= (1 if INTRO in clips else 0):
        raise Http404
    return mp3_response(request, b''.join(bytes(clips[a].mp3) for a in order))


def cleanup():
    PaperAudio.objects.filter(created__lt=timezone.now() - timezone.timedelta(days=KEEP_DAYS)).exclude(arxiv_id=INTRO).delete()
