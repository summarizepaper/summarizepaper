"""Scoop watch (Premium): an email when a new paper comes close to a project still in progress.

A user describes an ongoing project in a few paragraphs, or gives the PDF of a
draft, from which the model writes that description (the PDF is not stored).
The description gets one embedding. After each arXiv announcement (with its
bioRxiv and medRxiv papers, see digest.py), the hourly run ranks the new papers
by closeness to each project, using the embeddings the digests already keep, and
the model reads the closest abstracts: it says whether each one overlaps the
project strongly (same question, same or very close approach) or partly, what
it shares with the project and what stays the project's own. Papers signed by
the user's own name are left out.

When a project is added, the announcements of the last days are checked at
once and shown on the page, without an email; after that, new close papers are
emailed the day they are announced.
"""
import datetime
import hashlib
import json
import logging

import numpy as np
import requests
from django.conf import settings
from django.core.mail import EmailMultiAlternatives
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import get_language

from . import access, digest, generated, mailcheck, missed, similar
from .models import ArxivBatch, ScoopMatch, WatchedProject

logger = logging.getLogger(__name__)

MAX_PROJECTS = 5
MIN_CHARS = 300
MAX_CHARS = 6000
CANDIDATES = 8  # closest new papers the model reads, per project and check
FLOOR = 0.45  # cosine below which a paper is not even shown to the model
MISSING_VECTORS = 0.05  # a check waits for the next run while more papers than this lack their embedding
MAX_CHECKS_PER_RUN = 30  # one model call each
MAX_PER_EMAIL = 20
OVERLAPS = ('high', 'partial')
SHOWN = 30


def project_text(project):
    return '%s\n\n%s' % (project.title, project.description)


def search_abstract(project):
    """The description rewritten in English as a paper's abstract. Papers are compared through their abstracts,
    and a description in another language, or informal, is much farther from them than the same project told as
    an abstract (on an exomoon project in French, close papers went from 0.48 to 0.68 cosine)."""
    answer = generated.chat_json(
        'Above is the description of a research project, written by its author, possibly informally or in another '
        'language. Rewrite it in English as the abstract of a paper on this research would read: the question, the '
        'method, the data, and the other approaches to the same question it mentions. Describe the research itself, '
        'not the author\'s wishes or worries. 100 to 200 words, no first person. Answer with a JSON object '
        '{"abstract": "..."}.', paid=False, max_tokens=600, system=project_text(project)[:MAX_CHARS])
    return str(answer.get('abstract', '')).strip()


def project_vector(project):
    """Embedding of the title and of the description told as an abstract, recomputed only when they change."""
    key = hashlib.sha256(json.dumps([similar.MODEL, 'abstract-1', project_text(project)]).encode()).hexdigest()
    if project.vector_key == key and project.vector:
        return np.frombuffer(bytes(project.vector), dtype=np.float32)
    project.search_text = search_abstract(project) or project.description
    vector = similar.embed([('%s\n\n%s' % (project.title, project.search_text))[:similar.MAX_INPUT_CHARS]])[0]
    project.vector, project.vector_key = vector.tobytes(), key
    project.save(update_fields=['search_text', 'vector', 'vector_key'])
    return vector


def own_paper(project, paper):
    """True if the user signs it (their own preprint is not a scoop)."""
    key = digest.name_key(project.author_name) if project.author_name else None
    return bool(key) and any(digest.same_author(key, name) for name in paper['authors'])


def days_to_check(project, batches):
    """The stored announcements the project has not been compared with. The latest one waits while an
    archive is still missing from it; an older one still incomplete is checked with what it has."""
    if not batches:
        return []
    latest = max(batches)
    return sorted(day for day, complete in batches.items()
                  if (project.last_day is None or day > project.last_day) and (complete or day < latest))


def stored_batches(now):
    """{day: complete} of the announcements still kept (see digest.KEEP_DAYS)."""
    since = now.date() - datetime.timedelta(days=digest.KEEP_DAYS)
    return {day: not missing for day, missing in ArxivBatch.objects.filter(day__gte=since).values_list('day', 'missing')}


def judge(project, closest, lang):
    """{arxiv_id: {'overlap', 'shared', 'original'}} for the papers the model finds overlapping."""
    listing = '\n\n'.join('<paper id="%s">\nTitle: %s\nAbstract: %s\n</paper>' % (p['arxiv_id'], p['title'], p['abstract'])
                          for p, _ in closest)
    answer = generated.chat_json(
        'Above are a researcher\'s ongoing, unpublished project and papers that have just been announced. For each '
        'paper, say how much it overlaps the project: "high" if it addresses the same question with the same or a '
        'very close approach, so that it could take away the novelty of the project; "partial" if it shares the '
        'question with another approach, or the approach on another question, or part of the expected results; '
        '"none" otherwise. For "high" and "partial", write in %s one or two sentences on what the paper shares with '
        'the project ("shared"), and one or two on what stays original in the project and how to set it apart '
        '("original"). Only the abstracts are known: be factual and do not exaggerate the overlap. Answer with a '
        'JSON object {"items": [{"id": "paper id", "overlap": "high|partial|none", "shared": "...", "original": "..."}]}.'
        % generated.language_name(lang),
        paid=True, max_tokens=3000,
        system='<project>\n%s\n</project>\n\n%s' % (project_text(project)[:MAX_CHARS], listing))
    ids = {p['arxiv_id'] for p, _ in closest}
    return {i['id']: {'overlap': i['overlap'], 'shared': str(i.get('shared', '')).strip(),
                      'original': str(i.get('original', '')).strip()}
            for i in answer.get('items', [])
            if isinstance(i, dict) and i.get('id') in ids and i.get('overlap') in OVERLAPS}


def check(project, days, papers, vectors, quiet=False):
    """Compare the project with the papers announced on `days` and store the close ones; returns how many.
    quiet: they are shown on the page but not emailed (the first check, over the last days)."""
    announced = [aid for aid, p in papers.items() if p['announced'] in days]
    if sum(aid not in vectors for aid in announced) > MISSING_VECTORS * len(announced):
        raise ValueError('embeddings missing for the papers of %s' % days)  # the day is not marked as checked
    reference = project_vector(project)
    known = set(project.matches.values_list('arxiv_id', flat=True))
    scored = [(p, float(vectors[aid] @ reference)) for aid, p in papers.items()
              if p['announced'] in days and aid in vectors and aid not in known and not own_paper(project, p)]
    closest = sorted([s for s in scored if s[1] >= FLOOR], key=lambda s: -s[1])[:CANDIDATES]
    verdicts = judge(project, closest, project.lang) if closest else {}
    rows = [ScoopMatch(project=project, arxiv_id=p['arxiv_id'], title=p['title'], authors=json.dumps(p['authors'][:8]),
                       announced=p['announced'], similarity=score, emailed=quiet, **verdicts[p['arxiv_id']])
            for p, score in closest if p['arxiv_id'] in verdicts]
    ScoopMatch.objects.bulk_create(rows, ignore_conflicts=True)
    project.last_day = max(days)
    project.save(update_fields=['last_day'])
    return len(rows)


def papers_of(days, cache):
    """(papers, vectors) announced on these days, loaded once per run for the same days."""
    key = tuple(days)
    if key not in cache:
        papers = digest.load_papers(days)
        cache[key] = (papers, digest.ensure_vectors(list(papers.values())))
    return cache[key]


def check_due(now, limit=MAX_CHECKS_PER_RUN):
    """Compare the Premium users' projects with the announcements they have not seen; returns how many were checked."""
    batches = stored_batches(now)
    loaded, done = {}, 0
    for project in WatchedProject.objects.exclude(description='').select_related('user'):
        if done >= limit:
            break
        days = days_to_check(project, batches)
        if not days or not access.is_paid(project.user):
            continue
        papers, vectors = papers_of(days, loaded)
        try:
            check(project, days, papers, vectors, quiet=project.last_day is None)
        except (requests.RequestException, KeyError, ValueError, generated.GenerationError):
            logger.exception('Scoop check failed for project %s', project.pk)  # tried again next hour
        done += 1
    return done


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """Email up to `budget` users their new close papers; returns how many emails were sent."""
    from django.contrib.auth.models import User
    suppressed = mailcheck.suppressed()
    waiting = ScoopMatch.objects.filter(emailed=False).values_list('project__user_id', flat=True).distinct()
    sent = 0
    for user in User.objects.filter(pk__in=list(waiting)):
        if sent >= budget:
            break
        if not user.email or user.email.strip().lower() in suppressed or not access.is_paid(user):
            continue
        matches = list(ScoopMatch.objects.filter(project__user=user, emailed=False).select_related('project'))
        lang = matches[0].project.lang
        projects = {}
        for match in sorted(matches, key=lambda m: (OVERLAPS.index(m.overlap), -m.similarity))[:MAX_PER_EMAIL]:
            match.url = '%s/%s/arxiv-id/%s/?autorun=1' % (base_url, lang, match.arxiv_id)
            projects.setdefault(match.project, []).append(match)
        context = {'projects': list(projects.items()), 'count': len(matches),
                   'high': sum(m.overlap == 'high' for m in matches), 'more': max(0, len(matches) - MAX_PER_EMAIL),
                   'manage_url': '%s/%s/scoops/' % (base_url, lang), 'base_url': base_url}
        with translation.override(lang):
            subject = render_to_string('scoops/subject.txt', context).strip()
            text = render_to_string('scoops/email.txt', context)
            html = render_to_string('scoops/email.html', context)
        message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
        message.attach_alternative(html, 'text/html')
        try:
            message.send()
        except Exception:
            logger.exception('Scoop alert to %s failed', user.username)
            continue
        ScoopMatch.objects.filter(pk__in=[m.pk for m in matches]).update(emailed=True)
        sent += 1
    return sent


def run(now, budget):
    """From the hourly digest run. It uses the announcements the digests stored, and fetches none itself."""
    check_due(now)
    return send_alerts(now, budget) if budget > 0 else 0


def prepare(project_pk, data=None):
    """In the background, after a project is added: its description from the draft (if a PDF was given),
    then the first check over the announcements of the last days."""
    project = WatchedProject.objects.get(pk=project_pk)
    if data is not None:
        draft = missed.pdf_text(data)[:missed.MAX_TEXT_CHARS]
        if len(draft.strip()) < MIN_CHARS:
            return  # no text in the PDF: the page says so after a while (see scoops_page)
        generated.release_connection()
        written = generated.chat_json(
            'Above is the beginning of a research draft. Answer with a JSON object {"title": "its title or a short '
            'description", "description": "in English, 150 to 250 words: the question it addresses, its method, its '
            'data, and the results it expects or reaches"}.',
            paid=False, max_tokens=800, system='<draft>\n%s\n</draft>' % draft[:12000])
        project.title = project.title or str(written.get('title', '')).strip()[:300] or 'Draft'
        project.description = str(written.get('description', '')).strip()[:MAX_CHARS]
        project.save(update_fields=['title', 'description'])
    days = days_to_check(project, stored_batches(timezone.now()))
    if days and project.description:
        check(project, days, *papers_of(days, {}), quiet=True)


# --- views -------------------------------------------------------------------

def scoops_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/scoops.html', {'anonymous': True, 'max_projects': MAX_PROJECTS})
    paid = access.is_paid(request.user)
    error = None
    projects = WatchedProject.objects.filter(user=request.user)
    if request.method == 'POST' and paid:
        action = request.POST.get('action')
        if action == 'delete':
            projects.filter(pk=request.POST.get('pk')).delete()
            return redirect('/%s/scoops/' % get_language())
        upload = request.FILES.get('draft')
        title = request.POST.get('title', '').strip()[:300]
        description = request.POST.get('description', '').strip()[:MAX_CHARS]
        if projects.count() >= MAX_PROJECTS:
            error = 'too_many'
        elif upload is not None and (upload.size > missed.MAX_UPLOAD or not upload.name.lower().endswith('.pdf')):
            error = 'file'
        elif upload is None and (not title or len(description) < MIN_CHARS):
            error = 'empty'
        else:
            project = WatchedProject.objects.create(
                user=request.user, title=title, description='' if upload is not None else description,
                author_name=request.POST.get('author_name', '').strip()[:200], lang=get_language())
            generated.background(prepare, project.pk, upload.read() if upload is not None else None)
            return redirect('/%s/scoops/#project-%d' % (get_language(), project.pk))
    projects = list(projects)
    for project in projects:
        project.shown = list(project.matches.all()[:SHOWN])
        # a draft still unread after a while was not readable, or its reading was lost in a restart
        project.failed = not project.description and project.created < timezone.now() - generated.STALE
    return render(request, 'summarizer/scoops.html', {
        'paid': paid, 'projects': projects, 'error': error, 'max_projects': MAX_PROJECTS, 'min_chars': MIN_CHARS,
        'author_name': projects[0].author_name if projects else '', 'form': request.POST if error else {}})
