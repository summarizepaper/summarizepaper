"""Citation alerts (Premium): who cites the papers I follow, and what they say about them.

Semantic Scholar gives the citing papers of an arXiv paper and, when it has
their full text, the sentences around the citation. Once a day the hourly
digest run refreshes the papers somebody follows; a new citing paper is
queued for each follower, and once a week a follower with queued citations
gets one email. The model reads the citation sentences to say whether the
new paper builds on the followed one, supports it, questions it, or only
mentions it.

Following a paper records its current citations as known, so the first
email only has citations that are really new.
"""
import datetime
import json
import logging
import re
import time

import requests
from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import get_language
from django.views.decorators.http import require_POST

from . import access, digest, generated, mailcheck
from .models import ArxivPaper, CitationAlertLog, CitationCache, FollowedPaper

logger = logging.getLogger(__name__)

S2 = 'https://api.semanticscholar.org/graph/v1/paper/arXiv:%s/citations'
FIELDS = 'title,externalIds,publicationDate,authors,contexts,intents'
PAGE = 1000
MAX_PAGES = 10
REFRESH = datetime.timedelta(hours=20)
MAX_REFRESH_PER_RUN = 150
SECONDS_BETWEEN_CALLS = 1.1  # Semantic Scholar's public rate is about one request per second
EVERY = datetime.timedelta(days=7)
MAX_FOLLOWED = 100
MAX_PER_EMAIL = 40
DAILY_KEY = 'citations-refreshed:%s'
STANCES = ('builds_on', 'supports', 'questions', 'mentions')


def base_id(text):
    """'2101.00001v2', 'hep-th--9901001', or an arXiv link -> '2101.00001' / 'hep-th/9901001'."""
    match = digest.ARXIV_ID_RE.search(text.replace('--', '/'))
    return re.sub(r'v\d+$', '', match.group(1)) if match else None


def fetch(arxiv_id):
    """{paperId: citing paper} from Semantic Scholar; None when it does not know the paper."""
    citing, offset = {}, 0
    headers = {'x-api-key': settings.S2_API_KEY} if getattr(settings, 'S2_API_KEY', '') else {}
    for _ in range(MAX_PAGES):
        for wait in (5, 20, None):
            response = requests.get(S2 % arxiv_id, params={'fields': FIELDS, 'limit': PAGE, 'offset': offset},
                                    headers=headers, timeout=60)
            if response.status_code != 429 or wait is None:
                break
            time.sleep(wait)
        if response.status_code == 404:
            return None
        response.raise_for_status()
        body = response.json()
        for row in body.get('data') or []:
            paper = row.get('citingPaper') or {}
            if not paper.get('paperId') or not paper.get('title'):
                continue
            citing[paper['paperId']] = {
                'title': ' '.join(paper['title'].split()),
                'arxiv_id': (paper.get('externalIds') or {}).get('ArXiv') or '',
                'doi': (paper.get('externalIds') or {}).get('DOI') or '',
                'date': paper.get('publicationDate') or '',
                'authors': [a.get('name', '') for a in (paper.get('authors') or [])[:6]],
                'contexts': [' '.join(c.split()) for c in (row.get('contexts') or [])[:3]],
                'intents': row.get('intents') or [],
            }
        if body.get('next') is None:
            break
        offset = body['next']
    return citing


def refresh(arxiv_id, now):
    """Update the cached citations of a paper and queue the new ones for its followers."""
    try:
        citing = fetch(arxiv_id)
    except (requests.RequestException, ValueError):
        logger.warning('Semantic Scholar failed for %s', arxiv_id)
        return False
    cached, _ = CitationCache.objects.get_or_create(arxiv_id=arxiv_id)
    cached.checked = now
    if citing is None:
        cached.save(update_fields=['checked'])
        return False
    cached.citing, cached.total = json.dumps(citing), len(citing)
    cached.save()
    for followed in FollowedPaper.objects.filter(arxiv_id=arxiv_id):
        if not followed.title:
            followed.title = titles([arxiv_id]).get(arxiv_id, '')
        if followed.known == '':
            # first check since the user followed it: what it already has is not news
            followed.known = json.dumps(list(citing))
            followed.save(update_fields=['known', 'title'])
            continue
        known = set(json.loads(followed.known))
        pending = json.loads(followed.pending or '[]')
        new = [pid for pid in citing if pid not in known and pid not in pending]
        followed.pending = json.dumps(pending + new)
        followed.save(update_fields=['pending', 'title'])
    return True


def refresh_due(now, limit=MAX_REFRESH_PER_RUN):
    """Refresh the followed papers not checked for a day (at most `limit`, spaced for the public API)."""
    followed = set(FollowedPaper.objects.values_list('arxiv_id', flat=True))
    fresh = set(CitationCache.objects.filter(arxiv_id__in=followed, checked__gte=now - REFRESH)
                .values_list('arxiv_id', flat=True))
    done = 0
    for arxiv_id in sorted(followed - fresh)[:limit]:
        if done:
            time.sleep(SECONDS_BETWEEN_CALLS)
        refresh(arxiv_id, now)
        done += 1
    return done


def titles(arxiv_ids):
    """{arxiv_id: title} from our tables, then one arXiv query for the rest; unknown ids are left out."""
    from .library import lookup_many
    found = lookup_many([a.replace('/', '--') for a in arxiv_ids])
    return {a: found[a.replace('/', '--')][0] for a in arxiv_ids if a.replace('/', '--') in found}


def follow(user, arxiv_id, title=''):
    """Start following. Nothing slow here: the first check, in the background, records the citations it
    already has as known, so they are not reported."""
    paper = ArxivPaper.objects.filter(arxiv_id__startswith=arxiv_id.replace('/', '--') + 'v').first()
    if paper and paper.title:
        title = ' '.join(paper.title.split())
    followed, created = FollowedPaper.objects.get_or_create(user=user, arxiv_id=arxiv_id, defaults={'title': title})
    if created:
        cached = CitationCache.objects.filter(arxiv_id=arxiv_id, checked__isnull=False).first()
        if cached is not None:
            followed.known = json.dumps(list(json.loads(cached.citing or '{}')))
            followed.save(update_fields=['known'])
    return followed, created


def refresh_new(arxiv_ids):
    for index, arxiv_id in enumerate(arxiv_ids):
        if index:
            time.sleep(SECONDS_BETWEEN_CALLS)
        refresh(arxiv_id, timezone.now())


def stances(followed_title, citations, lang):
    """{paperId: {'stance', 'note'}} from the citation sentences, one model call for the email."""
    with_context = {pid: c for pid, c in citations.items() if c['contexts']}
    if not with_context:
        return {}
    listing = '\n\n'.join('<citation id="%s">\nCiting paper: %s\nSentences: %s\n</citation>'
                          % (pid, c['title'], ' | '.join(c['contexts'])) for pid, c in with_context.items())
    try:
        answer = generated.chat_json(
            'The citations above all cite the paper "%s". For each one, from the sentences around the citation, say '
            'whether the citing paper builds on it, supports or confirms it, questions or contradicts it, or only '
            'mentions it, and write in %s one short sentence on what it says about it. Answer with a JSON object '
            '{"items": [{"id": "...", "stance": "builds_on|supports|questions|mentions", "note": "..."}]}.'
            % (followed_title, generated.language_name(lang)), paid=False, max_tokens=3000, system=listing)
    except Exception:
        logger.exception('Citation stances failed')
        return {}
    return {i['id']: {'stance': i.get('stance') if i.get('stance') in STANCES else 'mentions',
                      'note': str(i.get('note', '')).strip()}
            for i in answer.get('items', []) if isinstance(i, dict) and i.get('id') in with_context}


def due_users(now):
    queued = FollowedPaper.objects.exclude(pending='').exclude(pending='[]').values_list('user_id', flat=True).distinct()
    recent = set(CitationAlertLog.objects.filter(sent__gte=now - EVERY).values_list('user_id', flat=True))
    return [uid for uid in queued if uid not in recent]


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """Email up to `budget` users their new citations; returns how many emails were sent."""
    from django.contrib.auth.models import User
    suppressed = mailcheck.suppressed()
    sent = 0
    for user in User.objects.filter(pk__in=due_users(now)):
        if sent >= budget:
            break
        if not user.email or user.email.strip().lower() in suppressed or not access.is_paid(user):
            continue
        lang = user.digests.values_list('lang', flat=True).first() or 'en'
        papers = []
        followed_rows = list(FollowedPaper.objects.filter(user=user).exclude(pending='').exclude(pending='[]'))
        for followed in followed_rows:
            cached = CitationCache.objects.filter(arxiv_id=followed.arxiv_id).first()
            citing = json.loads(cached.citing or '{}') if cached else {}
            new = {pid: citing[pid] for pid in json.loads(followed.pending) if pid in citing}
            if not new:
                continue
            notes = stances(followed.title, dict(list(new.items())[:MAX_PER_EMAIL]), lang)
            items = []
            for pid, c in list(new.items())[:MAX_PER_EMAIL]:
                link = ('%s/%s/arxiv-id/%s/' % (base_url, lang, c['arxiv_id'].replace('/', '--'))) if c['arxiv_id'] else \
                    ('https://doi.org/%s' % c['doi'] if c['doi'] else 'https://www.semanticscholar.org/paper/%s' % pid)
                items.append(dict(c, url=link, **notes.get(pid, {'stance': '', 'note': ''})))
            papers.append({'followed': followed, 'items': items, 'more': max(0, len(new) - MAX_PER_EMAIL)})
        if not papers:
            continue
        context = {'papers': papers, 'count': sum(len(p['items']) + p['more'] for p in papers),
                   'manage_url': '%s/%s/citations/' % (base_url, lang), 'base_url': base_url}
        with translation.override(lang):
            subject = render_to_string('citations/subject.txt', context).strip()
            text = render_to_string('citations/email.txt', context)
            html = render_to_string('citations/email.html', context)
        message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
        message.attach_alternative(html, 'text/html')
        try:
            message.send()
        except Exception:
            logger.exception('Citation alert to %s failed', user.username)
            continue
        for followed in followed_rows:
            followed.known = json.dumps(json.loads(followed.known or '[]') + json.loads(followed.pending or '[]'))
            followed.pending = ''
            followed.save(update_fields=['known', 'pending'])
        CitationAlertLog.objects.update_or_create(user=user, defaults={'sent': now})
        sent += 1
    return sent


def run(now, budget):
    """From the hourly digest run: refresh once a day, then send what is due within the email budget."""
    if cache.add(DAILY_KEY % now.date(), True, 86400):
        refresh_due(now)
    return send_alerts(now, budget) if budget > 0 else 0


# --- views -------------------------------------------------------------------

def citations_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/citations.html', {'anonymous': True, 'max_followed': MAX_FOLLOWED})
    paid = access.is_paid(request.user)
    error = None
    if request.method == 'POST' and paid:
        if request.POST.get('action') == 'unfollow':
            FollowedPaper.objects.filter(user=request.user, pk=request.POST.get('pk')).delete()
            return redirect('/%s/citations/' % get_language())
        ids = []
        for text in re.split(r'[\s,;]+', request.POST.get('ids', '')):
            arxiv_id = base_id(text) if text else None
            if arxiv_id and arxiv_id not in ids:
                ids.append(arxiv_id)
        room = MAX_FOLLOWED - FollowedPaper.objects.filter(user=request.user).count()
        if len(ids) > room:
            error = 'too_many'
        ids = ids[:max(0, room)]
        known = titles([a for a in ids if not FollowedPaper.objects.filter(user=request.user, arxiv_id=a).exists()])
        new = [a for a in ids if follow(request.user, a, known.get(a, ''))[1] and not
               CitationCache.objects.filter(arxiv_id=a, checked__isnull=False).exists()]
        if new:
            generated.background(refresh_new, new)
        if not error:
            return redirect('/%s/citations/' % get_language())
    followed = list(FollowedPaper.objects.filter(user=request.user))
    untitled = [f for f in followed if not f.title]
    if untitled:
        # followed before their title was looked up, or unknown to Semantic Scholar
        found = titles([f.arxiv_id for f in untitled])
        for f in untitled:
            if found.get(f.arxiv_id):
                f.title = found[f.arxiv_id]
                f.save(update_fields=['title'])
    totals =dict(CitationCache.objects.filter(arxiv_id__in=[f.arxiv_id for f in followed]).values_list('arxiv_id', 'total'))
    for f in followed:
        f.total = totals.get(f.arxiv_id)
        f.queued = len(json.loads(f.pending or '[]'))
    return render(request, 'summarizer/citations.html', {
        'paid': paid, 'followed': followed, 'error': error, 'max_followed': MAX_FOLLOWED})


@require_POST
def follow_view(request):
    """From a paper page. JSON."""
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    arxiv_id = base_id(request.POST.get('arxiv_id', ''))
    if not arxiv_id:
        return JsonResponse({'status': 'unknown'}, status=400)
    if request.POST.get('action') == 'unfollow':
        FollowedPaper.objects.filter(user=request.user, arxiv_id=arxiv_id).delete()
        return JsonResponse({'status': 'unfollowed'})
    if FollowedPaper.objects.filter(user=request.user).count() >= MAX_FOLLOWED:
        return JsonResponse({'status': 'too_many'})
    followed, created = follow(request.user, arxiv_id)
    if created and not followed.title:
        followed.title = titles([arxiv_id]).get(arxiv_id, '')
        followed.save(update_fields=['title'])
    if created and followed.known == '':
        generated.background(refresh_new, [arxiv_id])
    return JsonResponse({'status': 'following'})


def is_following(user, arxiv_id):
    if not user.is_authenticated:
        return False
    return FollowedPaper.objects.filter(user=user, arxiv_id=base_id(arxiv_id) or '').exists()
