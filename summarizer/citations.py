"""Citation alerts (Premium): who cites the papers I follow, and what they say about them.

NASA ADS gives the citing papers of an arXiv paper: it links preprints to
their journal version, so its counts are the ones astronomers and physicists
know. Semantic Scholar adds, when it has their full text, the sentences around
the citation, and is the source for papers ADS does not know. Once a day the hourly
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
from django.core.paginator import Paginator
from django.db import transaction
from django.core.mail import EmailMultiAlternatives
from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import get_language
from django.views.decorators.http import require_POST

from . import access, digest, generated, mailcheck, sources
from .models import ArxivPaper, CitationAlertHistory, CitationAlertLog, CitationCache, FollowedPaper

logger = logging.getLogger(__name__)

ADS = 'https://api.adsabs.harvard.edu/v1/search/query'
ADS_FIELDS = 'bibcode,title,author,pubdate,identifier,doi'
ADS_PAGE = 2000
S2 = 'https://api.semanticscholar.org/graph/v1/paper/%s/citations'
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
    """'2101.00001v2', 'hep-th--9901001', or an arXiv link -> '2101.00001' / 'hep-th/9901001';
    a DOI or a doi_ key -> the doi_ key (Semantic Scholar and ADS know papers by DOI too)."""
    key = text.strip().lower()
    if sources.is_external(key):
        return key if key.startswith('doi_') and sources.valid_key(key) and sources.key_doi(key) else None
    doi = sources.clean_doi(text)
    if doi and not sources.ARXIV_DOI_RE.match(doi):
        return sources.doi_key(doi)
    match = digest.ARXIV_ID_RE.search(text.replace('--', '/'))
    return re.sub(r'v\d+$', '', match.group(1)) if match else None


def fetch_s2(arxiv_id):
    """{paperId: citing paper} from Semantic Scholar; None when it does not know the paper."""
    citing, offset = {}, 0
    headers = {'x-api-key': settings.S2_API_KEY} if getattr(settings, 'S2_API_KEY', '') else {}
    for _ in range(MAX_PAGES):
        for wait in (5, 20, None):
            response = requests.get(S2 % s2_ref(arxiv_id), params={'fields': FIELDS, 'limit': PAGE, 'offset': offset},
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
        time.sleep(SECONDS_BETWEEN_CALLS)
    return citing


def s2_ref(paper_id):
    """How Semantic Scholar names a followed paper: 'arXiv:2101.00001' or 'DOI:10.1093/mnras/stad1234'."""
    if sources.is_external(paper_id):
        return 'DOI:%s' % sources.key_doi(paper_id)
    return 'arXiv:%s' % paper_id


def ads_get(params):
    response = requests.get(ADS, params=params, timeout=60,
                            headers={'Authorization': 'Bearer %s' % settings.ADS_TOKEN})
    response.raise_for_status()
    return response.json()['response']


def fetch_ads(arxiv_id):
    """{'ads:<bibcode>': citing paper} from ADS; None when it does not know the paper."""
    if sources.is_external(arxiv_id):
        target = 'doi:"%s"' % sources.key_doi(arxiv_id)
    else:
        target = 'identifier:"arXiv:%s"' % arxiv_id
    if not ads_get({'q': target, 'fl': 'bibcode', 'rows': 1})['numFound']:
        return None
    citing = {}
    for page in range(MAX_PAGES):
        body = ads_get({'q': 'citations(%s)' % target, 'fl': ADS_FIELDS, 'rows': ADS_PAGE,
                        'start': page * ADS_PAGE, 'sort': 'date desc'})
        for doc in body['docs']:
            if not doc.get('title'):
                continue
            arxiv = next((i[6:] for i in doc.get('identifier') or [] if i.startswith('arXiv:')), '')
            citing['ads:' + doc['bibcode']] = {
                'title': ' '.join(doc['title'][0].split()),
                'arxiv_id': arxiv,
                'doi': (doc.get('doi') or [''])[0],
                'date': (doc.get('pubdate') or '').replace('-00', ''),
                'authors': (doc.get('author') or [])[:6],
                'contexts': [],
                'intents': [],
            }
        if (page + 1) * ADS_PAGE >= body['numFound']:
            break
    return citing


class Citing(dict):
    """{id: citing paper}, and the source it came from ('ads' or 's2'), known even when it is empty."""

    def __init__(self, papers, source):
        super().__init__(papers)
        self.source = source


def same_paper(c):
    return [k for k in (c['arxiv_id'] and 'arxiv:' + re.sub(r'v\d+$', '', c['arxiv_id']).lower(),
                        c['doi'] and 'doi:' + c['doi'].lower()) if k]


def fetch(arxiv_id):
    """{id: citing paper}, from ADS when it knows the paper, with the citation sentences of Semantic
    Scholar; from Semantic Scholar alone otherwise. None when neither knows it."""
    citing = fetch_ads(arxiv_id) if getattr(settings, 'ADS_TOKEN', '') else None
    if citing is None:
        s2 = fetch_s2(arxiv_id)
        return None if s2 is None else Citing(s2, 's2')
    try:
        sentences = fetch_s2(arxiv_id) or {}
    except (requests.RequestException, ValueError):
        sentences = {}  # the counts do not depend on it; tomorrow's check adds the sentences
    by_key = {k: c for c in sentences.values() if c['contexts'] for k in same_paper(c)}
    for c in citing.values():
        match = next((by_key[k] for k in same_paper(c) if k in by_key), None)
        if match:
            c['contexts'], c['intents'] = match['contexts'], match['intents']
    return Citing(citing, 'ads')


def source_of(citing):
    """'ads' or 's2' from the ids of the citing papers, '' when there are none."""
    if not citing:
        return ''
    return 'ads' if any(k.startswith('ads:') for k in citing) else 's2'


def rekey(ids, old, new):
    """The ids of `new` for the papers `ids` of `old`, when the source of a paper's citations changed."""
    index = {k: pid for pid, c in new.items() for k in same_paper(c)}
    found = (next((index[k] for k in same_paper(old[pid]) if k in index), None) for pid in ids if pid in old)
    return [pid for pid in dict.fromkeys(found) if pid]


def refresh(arxiv_id, now):
    """Update the cached citations of a paper and queue the new ones for its followers."""
    try:
        citing = fetch(arxiv_id)
    except (requests.RequestException, ValueError, KeyError):
        logger.warning('Citations failed for %s', arxiv_id)
        return False
    cached, _ = CitationCache.objects.get_or_create(arxiv_id=arxiv_id)
    cached.checked = now
    if citing is None:
        cached.save(update_fields=['checked'])
        return False
    before = json.loads(cached.citing or '{}')
    # ids are bibcodes ('ads:...') or Semantic Scholar ids: when the source changes, carry over what
    # followers knew and had queued, by arXiv id or DOI, instead of reporting every citation as new.
    # The source is stored, so a paper with no citation in the old source switches too.
    previous = cached.source or source_of(before)
    current = getattr(citing, 'source', '') or source_of(citing) or previous
    switched = bool(previous) and current != previous
    cached.citing, cached.total, cached.source = json.dumps(citing), len(citing), current
    cached.save()
    for followed in FollowedPaper.objects.filter(arxiv_id=arxiv_id):
        if not followed.title:
            followed.title = titles([arxiv_id]).get(arxiv_id, '')
        if followed.known == '':
            # first check since the user followed it: what it already has is not news
            followed.known = json.dumps(list(citing))
            followed.save(update_fields=['known', 'title'])
            continue
        if switched:
            pending = rekey(json.loads(followed.pending or '[]'), before, citing)
            # a citation the new source has and the old one missed is not news either
            followed.known = json.dumps([pid for pid in citing if pid not in pending])
            followed.pending = json.dumps(pending) if pending else ''
            followed.save(update_fields=['known', 'pending', 'title'])
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
    if sources.is_external(arxiv_id):
        paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    else:
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


def pending(user, base_url, lang):
    """(email context, commit(now)) of the new citations not emailed yet to this user, or None."""
    papers, archive, delivered = [], [], {}
    followed_rows = list(FollowedPaper.objects.filter(user=user).exclude(pending='').exclude(pending='[]'))
    for followed in followed_rows:
        cached = CitationCache.objects.filter(arxiv_id=followed.arxiv_id).first()
        citing = json.loads(cached.citing or '{}') if cached else {}
        new = {pid: citing[pid] for pid in json.loads(followed.pending) if pid in citing}
        if not new:
            continue
        delivered[followed.pk] = list(new)
        notes = stances(followed.title, dict(list(new.items())[:MAX_PER_EMAIL]), lang)
        items = []
        for pid, c in new.items():
            link = ('%s/%s/paper/%s/' % (base_url, lang, c['arxiv_id'].replace('/', '--'))) if c['arxiv_id'] else \
                ('https://doi.org/%s' % c['doi'] if c['doi'] else
                 'https://ui.adsabs.harvard.edu/abs/%s/abstract' % pid[4:] if pid.startswith('ads:') else
                 'https://www.semanticscholar.org/paper/%s' % pid)
            items.append(dict(c, url=link, **notes.get(pid, {'stance': '', 'note': ''})))
        papers.append({'followed': followed, 'items': items[:MAX_PER_EMAIL], 'more': max(0, len(new) - MAX_PER_EMAIL)})
        archive.append({'arxiv_id': followed.arxiv_id, 'title': followed.title, 'items': items})
    if not papers:
        return None
    context = {'papers': papers, 'count': sum(len(p['items']) + p['more'] for p in papers),
               'manage_url': '%s/%s/citations/' % (base_url, lang), 'base_url': base_url}

    def commit(now):
        with transaction.atomic():
            CitationAlertHistory.objects.get_or_create(user=user, sent=now, defaults={
                'lang': lang, 'count': context['count'], 'papers': archive})
            for followed in followed_rows:
                current = FollowedPaper.objects.select_for_update().filter(pk=followed.pk, user=user).first()
                if current is None or followed.pk not in delivered:
                    continue
                ids = delivered[followed.pk]
                current.known = json.dumps(list(dict.fromkeys(json.loads(current.known or '[]') + ids)))
                pending = [pid for pid in json.loads(current.pending or '[]') if pid not in set(ids)]
                current.pending = json.dumps(pending) if pending else ''
                current.save(update_fields=['known', 'pending'])
            CitationAlertLog.objects.update_or_create(user=user, defaults={'sent': now})
    return context, commit


STANCE_LABELS = {'builds_on': 'builds on it', 'supports': 'supports it', 'questions': 'questions it', 'mentions': 'mentions it'}


def bundle_section(user, base_url, lang, now):
    """The new citations as a section of the user's digest (see digest_bundle), in the active language."""
    if not access.has_tool(user, 'citations'):
        return None
    found = pending(user, base_url, lang)
    if found is None:
        return None
    context, commit = found
    return {'title': translation.gettext('New citations of the papers you follow'), 'manage_url': context['manage_url'],
            'more': sum(p['more'] for p in context['papers']), 'commit': lambda: commit(now),
            'items': [{'group': p['followed'].title or p['followed'].arxiv_id, 'title': c['title'], 'url': c['url'],
                       'label': translation.gettext(STANCE_LABELS[c['stance']]) if c.get('stance') in STANCE_LABELS else '',
                       'note': c.get('note', '')} for p in context['papers'] for c in p['items']]}


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """Email up to `budget` users their new citations; returns how many emails were sent."""
    from django.contrib.auth.models import User
    from . import digest_bundle
    suppressed = mailcheck.suppressed()
    sent = 0
    for user in User.objects.filter(pk__in=due_users(now)).exclude(pk__in=digest_bundle.bundled_users()):
        if sent >= budget:
            break
        if not user.email or user.email.strip().lower() in suppressed or not access.has_tool(user, 'citations'):
            continue
        lang = user.digests.values_list('lang', flat=True).first() or 'en'
        found = pending(user, base_url, lang)
        if found is None:
            continue
        context, commit = found
        with translation.override(lang):
            subject = render_to_string('citations/subject.txt', context).strip()
            text = render_to_string('citations/email.txt', context)
            html = render_to_string('citations/email.html', context)
        message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
        message.attach_alternative(html, 'text/html')
        try:
            if not message.send():
                continue
        except Exception:
            logger.exception('Citation alert to %s failed', user.username)
            continue
        commit(now)
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
    paid = access.has_tool(request.user, 'citations')
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
    history = (Paginator(CitationAlertHistory.objects.filter(user=request.user), 10)
               .get_page(request.GET.get('alerts_page'))) if paid else None
    last_sent = (CitationAlertLog.objects.filter(user=request.user).values_list('sent', flat=True).first()
                 if paid and not history.paginator.count else None)
    return render(request, 'summarizer/citations.html', {
        'paid': paid, 'followed': followed, 'error': error, 'max_followed': MAX_FOLLOWED,
        'alert_history': history, 'last_unarchived_alert': last_sent})


@require_POST
def follow_view(request):
    """From a paper page. JSON."""
    if not access.has_tool(request.user, 'citations'):
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
