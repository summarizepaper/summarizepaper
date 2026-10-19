"""The Premium digest read aloud.

Each paper of a digest is read once per language (title, then its 2-sentence
blurb) and the clip is shared by every digest that lists it. The email links
to a page that plays the clips of that digest one after the other: MP3 frames
can be joined end to end, so the 'episode' is the clips put together.
"""
import logging

import requests
from django.conf import settings
from django.core import signing
from django.http import Http404, HttpResponse
from django.shortcuts import render
from django.utils import timezone, translation
from django.utils.translation import gettext as _

from .models import DigestItem, PaperAudio, SiteSettings

logger = logging.getLogger(__name__)

ENDPOINT = 'https://api.openai.com/v1/audio/speech'
SALT = 'digest-audio'
MAX_CLIPS = 8  # papers read in one episode
MAX_NEW_PER_RUN = 40
KEEP_DAYS = 10  # clips are a few hundred kB each
INTRO = '_intro'


def speak(text):
    cfg = SiteSettings.current()
    response = requests.post(ENDPOINT, timeout=120, headers={'Authorization': 'Bearer %s' % settings.OPENAI_KEY},
                             json={'model': cfg.model_tts, 'voice': cfg.tts_voice, 'input': text[:4000],
                                   'response_format': 'mp3'})
    response.raise_for_status()
    return response.content


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
    response = HttpResponse(b''.join(bytes(clips[a].mp3) for a in order), content_type='audio/mpeg')
    response['Cache-Control'] = 'private, max-age=86400'
    return response


def cleanup():
    PaperAudio.objects.filter(created__lt=timezone.now() - timezone.timedelta(days=KEEP_DAYS)).exclude(arxiv_id=INTRO).delete()
