"""Conference companion (Premium): the talks and posters of a conference program that matter for one's research.

1. The program, pasted or as a PDF, is cut into pieces; the model lists the
   talks and posters of each piece (title, speakers, session, time, abstract).
2. They are ranked by closeness to the researcher's interests (their digest
   research profile, a scoop watch project, or a few sentences), with
   embeddings, so a program of hundreds of talks costs a few model calls.
3. The model reads the closest ones and says why each one matters and how
   much (do not miss, worth it, if you have time); their speakers are the
   people to meet.
"""
import re
import unicodedata
import uuid
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, generated, missed, similar, sources
from .models import Generated, WatchedProject

MAX_PROGRAM = 120000  # characters of the program read
CHUNK = 12000
PARALLEL = 4
SHORTLIST = 30
MIN_PROGRAM = 300
MIN_INTERESTS = 20
MAX_INTERESTS = 4000
PRIORITIES = ('must', 'should', 'maybe')
PER_DAY = 5
QUOTA_KEY = 'conference-quota:%s:%s'
STEPS = [
    ('read', _('Reading the program')),
    ('list', _('Listing the talks and posters')),
    ('rank', _('Ranking them by closeness to your research')),
    ('write', _('An AI says why the closest ones matter to you')),
]

PDF_ACCENTS = {'´': '\u0301', 'ˊ': '\u0301', '`': '\u0300', 'ˋ': '\u0300',
               '¨': '\u0308', 'ˆ': '\u0302', '˜': '\u0303', '¸': '\u0327'}
PDF_ACCENT_RE = re.compile(r'(?<!\\)([' + re.escape(''.join(PDF_ACCENTS)) + r'])([A-Za-z])')


def normalize_accents(text):
    """Recompose detached PDF accent glyphs; preserve apostrophes and math symbols."""
    text = unicodedata.normalize('NFC', text)

    def compose(match):
        combined = unicodedata.normalize('NFC', match[2] + PDF_ACCENTS[match[1]])
        return combined if len(combined) == 1 else match[0]

    # Backticks used as inline code delimiters are not detached grave accents.
    parts = re.split(r'(?<!\w)(`[^`\n]+`)(?!\w)', text)
    return ''.join(part if index % 2 else PDF_ACCENT_RE.sub(compose, part)
                   for index, part in enumerate(parts))


def result_data(data):
    """Repair cached results at display time without regenerating scientific content."""
    result = dict(data)
    if isinstance(result.get('name'), str):
        result['name'] = normalize_accents(result['name'])
    result['talks'] = [dict(talk, **{key: normalize_accents(value) for key, value in talk.items()
                                   if isinstance(value, str)}) for talk in result.get('talks', []) if isinstance(talk, dict)]
    result['people'] = list(dict.fromkeys(normalize_accents(person) for person in result.get('people', [])
                                         if isinstance(person, str)))
    return result


def chunks(text):
    """Pieces of about CHUNK characters, cut at blank lines when possible."""
    text = text[:MAX_PROGRAM]
    pieces, start = [], 0
    while start < len(text):
        end = min(len(text), start + CHUNK)
        if end < len(text):
            cut = text.rfind('\n\n', start + CHUNK // 2, end)
            end = cut if cut > 0 else end
        pieces.append(text[start:end])
        start = end
    return pieces


def list_talks(piece):
    answer = generated.chat_json(
        'Above is a piece of a conference program. List every talk, poster or contributed presentation in it (not the '
        'breaks, sessions headers alone, or social events). Keep the title and names as written. Answer with a JSON '
        'object {"talks": [{"title": "...", "speakers": "names as written", "session": "session name or empty", '
        '"time": "day and time as written, or empty", "abstract": "its abstract if given, at most 600 characters, '
        'or empty"}]}.', paid=False, max_tokens=6000, system='<program>\n%s\n</program>' % piece)
    return [{k: normalize_accents(str(t.get(k, '')).strip()) for k in ('title', 'speakers', 'session', 'time', 'abstract')}
            for t in answer.get('talks', []) if isinstance(t, dict) and str(t.get('title', '')).strip()]


def find(program, interests, lang, step=lambda name: None):
    step('list')
    with ThreadPoolExecutor(PARALLEL) as pool:
        listed = [t for found in pool.map(sources.closing(list_talks), chunks(normalize_accents(program))) for t in found]
    unique = {}
    for talk in listed:  # a talk cut between two pieces is listed twice
        unique.setdefault(talk['title'].lower(), talk)
    talks = list(unique.values())
    if not talks:
        return {'talks': [], 'listed': 0, 'people': []}
    step('rank')
    vectors = similar.embed([interests[:similar.MAX_INPUT_CHARS]] +
                            ['%s\n\n%s' % (t['title'], t['abstract']) for t in talks])
    scores = vectors[1:] @ vectors[0]
    shortlist = [dict(talks[i], id=str(n), similarity=float(scores[i]))
                 for n, i in enumerate(list(np.argsort(-scores))[:SHORTLIST])]
    step('write')
    listing = '\n\n'.join('<talk id="%s">\nTitle: %s\nSpeakers: %s\nAbstract: %s\n</talk>' % (
        t['id'], t['title'], t['speakers'], t['abstract'] or '(none)') for t in shortlist)
    notes = generated.chat_json(
        'A researcher goes to the conference whose talks are above. Their research: %s\n\nIn %s, for each talk that '
        'matters for this research, say in one sentence why, and how much: "must" (do not miss it: same question or '
        'method), "should" (worth it), "maybe" (if you have time). Leave out the talks that do not matter. Answer with '
        'a JSON object {"talks": [{"id": "talk id", "priority": "must|should|maybe", "why": "..."}]}.'
        % (interests[:MAX_INTERESTS], generated.language_name(lang)), paid=True, max_tokens=4000, system=listing)
    by_id = {str(n.get('id')): n for n in notes.get('talks', [])
             if isinstance(n, dict) and n.get('priority') in PRIORITIES}
    chosen = [dict(t, priority=by_id[t['id']]['priority'], why=str(by_id[t['id']].get('why', '')).strip())
              for t in shortlist if t['id'] in by_id]
    chosen.sort(key=lambda t: (PRIORITIES.index(t['priority']), -t['similarity']))
    people = []
    for t in chosen:
        if t['priority'] != 'maybe' and t['speakers'] and t['speakers'] not in people:
            people.append(t['speakers'])
    for t in chosen:
        t.pop('id')
        t['similarity'] = round(100 * t['similarity'])
    return result_data({'talks': chosen, 'listed': len(talks), 'people': people[:15]})


def write(data, program, interests, name, lang):
    def work(pk):
        generated.progress(pk, 'read')
        text = missed.pdf_text(data) if data is not None else program
        if len(text.strip()) < MIN_PROGRAM:
            raise generated.GenerationError('no text found in the program')
        return dict(find(text, interests, lang, lambda step: generated.progress(pk, step)), name=normalize_accents(name))
    return work


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def default_interests(user):
    """The research profile of the user's digest, the first one written."""
    return user.digests.exclude(profile='').values_list('profile', flat=True).first() or ''


def conference_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/conference.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'conference')
    projects = WatchedProject.objects.filter(user=request.user).exclude(description='')
    error = None
    form = {}
    if request.method == 'POST' and paid:
        form = request.POST
        upload = request.FILES.get('program_pdf')
        program = request.POST.get('program', '').strip()
        project = projects.filter(pk=request.POST.get('project') or 0).first()
        interests = request.POST.get('interests', '').strip()[:MAX_INTERESTS]
        if project:
            interests = ('%s\n\n%s' % (interests, project.search_text or project.description)).strip()
        name = request.POST.get('name', '').strip()[:200]
        if upload is not None and (upload.size > missed.MAX_UPLOAD or not upload.name.lower().endswith('.pdf')):
            error = 'file'
        elif upload is None and len(program) < MIN_PROGRAM:
            error = 'program'
        elif len(interests) < MIN_INTERESTS:
            error = 'interests'
        elif not _spend(request.user):
            error = 'quota'
        else:
            data = upload.read() if upload is not None else None
            row = generated.start('conference', uuid.uuid4().hex, get_language(),
                                  write(data, program, interests, name, get_language()), user=request.user)
            return redirect('/%s/conference/%s/' % (get_language(), row.key))
    searches = Generated.objects.filter(kind='conference', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/conference.html', {
        'paid': paid, 'error': error, 'searches': searches, 'projects': projects, 'per_day': PER_DAY, 'form': form,
        'upload_max_bytes': missed.MAX_UPLOAD, 'upload_max_mb': missed.MAX_UPLOAD // (1024 * 1024),
        'interests': form.get('interests') if form else default_interests(request.user)})


@login_required
def conference_result(request, key):
    if not access.has_tool(request.user, 'conference'):
        raise Http404
    row = Generated.objects.filter(kind='conference', key=key, user=request.user).first()
    if row is None:
        raise Http404
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    return render(request, 'summarizer/conference_result.html', {
        'row': row, 'result': result_data(row.data()) if row.status == 'done' else None, 'steps': STEPS})
