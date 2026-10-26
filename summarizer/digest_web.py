"""The digest on the web: every paper of an email ranked with why it is there, the links of the email
(counted, so the ranking can be measured and learn from what its reader opens), saving a paper to a
collection, the live preview of the settings, and a profile started from the reader's own papers.
"""
import collections
import datetime
import json
import logging
import re

import requests
from django.core import signing
from django.core.cache import cache
from django.db.models import Q
from django.http import Http404, HttpResponse, HttpResponseNotAllowed
from django.shortcuts import redirect, render
from django.utils import timezone, translation
from django.utils.translation import get_language
from django.views.decorators.csrf import csrf_exempt

from . import access, digest, library, sources
from .models import (AnnouncedPaper, ArxivBatch, Collection, DigestClick, DigestEdition, DigestFeedback, DigestItem,
                     DigestSubscription)

logger = logging.getLogger(__name__)

PAGE_MAX = 300  # papers listed on the page of one email
BURST_SECONDS = 10
BURST_CLICKS = 4  # this many links of one digest opened within BURST_SECONDS: a mail scanner, not a reader
BOT_RE = re.compile(r'bot|crawl|spider|preview|scan|proofpoint|barracuda|mimecast|safelinks|slurp|python|curl', re.I)
PREVIEWS = (30, 600)  # live previews per user and 10 minutes
ORCID_RE = re.compile(r'(\d{4}-\d{4}-\d{4}-\d{3}[\dX])', re.I)
ARXIV_AUTHOR_RE = re.compile(r'^(?:https?://arxiv\.org/a/)?([a-z][a-z0-9_]*_\d+)(?:\.(?:atom2?|html))?/?$', re.I)
IMPORT_PAPERS = 25
IMPORT_CATEGORIES = 3
IMPORT_COAUTHORS = 5


# --- the papers of an email -------------------------------------------------------------

def edition_from_token(token):
    try:
        pk, edition_pk = signing.loads(token, salt=digest.EDITION_SALT)
    except (signing.BadSignature, ValueError, TypeError):
        return None
    return DigestEdition.objects.filter(pk=edition_pk, subscription_id=pk).select_related('subscription__user').first()


def papers_of(days, categories, extra_ids=()):
    """{arxiv_id: paper} of these announcements in these categories (and these papers), from the stored batches."""
    query = Q()
    for category in categories:
        query |= Q(categories__contains=category)
    if extra_ids:
        query |= Q(arxiv_id__in=list(extra_ids))
    if not query:
        return {}
    papers = {}
    for row in AnnouncedPaper.objects.filter(query, announced__in=days).iterator():
        papers[row.arxiv_id] = {
            'arxiv_id': row.arxiv_id, 'title': row.title, 'abstract': row.abstract,
            'authors': json.loads(row.authors or '[]'), 'categories': row.categories.split(','),
            'announced': row.announced,
            'published': datetime.datetime.combine(row.announced, datetime.time(), tzinfo=digest.ANNOUNCE_TZ),
        }
    return papers


def describe(entries, subscription, base_url='', in_email=()):
    """The entries as the page shows them: links, why each paper is there, and its summary when one exists."""
    lang = subscription.lang
    ids = [e['paper']['arxiv_id'] for e in entries]
    items = {i.arxiv_id: i for i in DigestItem.objects.filter(arxiv_id__in=ids, lang=lang)}
    wanted = subscription.category_list()
    rows = []
    for rank, entry in enumerate(entries, 1):
        paper = entry['paper']
        aid = paper['arxiv_id']
        ident, abs_url, pdf_url = digest.source_links(aid)
        item = items.get(aid)
        points = json.loads(item.points or '{}') if item is not None and item.points else {}
        rows.append({
            'rank': rank, 'arxiv_id': aid, 'title': paper['title'], 'ident': ident, 'abs_url': abs_url, 'pdf_url': pdf_url,
            'authors': ', '.join(paper['authors'][:6]) + (' et al.' if len(paper['authors']) > 6 else ''),
            'category': paper['categories'][0] if paper['categories'] else '',
            'primary': entry.get('primary', True), 'score': entry.get('score'),
            'reasons': entry.get('why') or digest.reasons(entry, wanted),
            'blurb': item.blurb if item is not None else '',
            'points': [(label, points[key]) for key, label in (
                ('contribution', translation.gettext('Contribution')), ('method', translation.gettext('Method')),
                ('result', translation.gettext('Result'))) if points.get(key)] if subscription.level == 'expert' else [],
            'abstract': paper['abstract'],
            'in_email': aid in in_email,
            'url': '%s/%s/paper/%s/?autorun=1' % (base_url, lang, aid),
            'save_url': '%s/%s/digest/save/%s/' % (base_url, lang, digest.save_token(subscription, aid)),
            'more_url': '%s/%s/digest/feedback/%s/' % (base_url, lang, digest.feedback_token(subscription, aid, 1)),
            'less_url': '%s/%s/digest/feedback/%s/' % (base_url, lang, digest.feedback_token(subscription, aid, -1)),
        })
    return rows


def edition_page(request, token):
    """Every paper of the announcements an email covered, ranked as the email was, with why each one is there."""
    edition = edition_from_token(token)
    if edition is None:
        raise Http404
    subscription = edition.subscription
    paid = access.has_tool(subscription.user, 'digest')
    categories = subscription.category_list()[:digest.limits(paid)['categories']]
    sent = json.loads(edition.papers or '[]')
    days = edition.day_list()
    papers = papers_of(days, categories, sent)
    rows, expired = [], not papers
    if papers:
        ranking = digest.ranking_for(subscription, paid, quick=True)
        vectors = digest.stored_vectors(list(papers))
        entries = digest.rank_papers(subscription, papers, set(), set(days), categories, ranking, vectors)
        listed = {e['paper']['arxiv_id'] for e in entries}
        with translation.override(subscription.lang):
            # papers of followed authors outside the categories, as the email put them: first
            via = translation.gettext('An author you follow')
            first = [{'paper': papers[aid], 'score': None, 'primary': True, 'why': [via]}
                     for aid in sent if aid in papers and aid not in listed]
            rows = describe(first + entries[:PAGE_MAX], subscription, in_email=set(sent))
    return render(request, 'summarizer/digest_edition.html', {
        'edition': edition, 'subscription': subscription, 'rows': rows, 'expired': expired,
        'total': len(rows), 'shown': len(sent), 'days': days, 'paid': paid,
        'categories': [(c, digest.category_label(c)) for c in categories],
        'threshold': digest.THRESHOLDS.get(subscription.volume) if paid else None,
    })


# --- the links of an email ------------------------------------------------------------------

def open_link(request, token):
    """A link of a digest email: counted (rank, target), then on to the paper. Mail scanners, which open
    every link within seconds, are told apart so they neither count nor teach the ranking."""
    if request.method not in ('GET', 'HEAD'):
        return HttpResponseNotAllowed(['GET', 'HEAD'])
    try:
        pk, edition_pk, arxiv_id, rank, target = signing.loads(token, salt=digest.OPEN_SALT)
    except (signing.BadSignature, ValueError, TypeError):
        raise Http404
    subscription = DigestSubscription.objects.filter(pk=pk).first()
    lang = subscription.lang if subscription else get_language()
    _, abs_url, pdf_url = digest.source_links(arxiv_id)
    campaign = 'weekly' if subscription and subscription.frequency == 'weekly' else 'daily'
    url = {'abs': abs_url, 'pdf': pdf_url}.get(target) or digest.tracked(
        '/%s/paper/%s/?autorun=1' % (lang, arxiv_id), campaign, 'r%d' % rank if rank else 'paper')
    if subscription is not None and request.method == 'GET' and not BOT_RE.search(request.META.get('HTTP_USER_AGENT', '')):
        try:
            record_click(subscription, edition_pk, arxiv_id, rank, target)
        except Exception:
            logger.exception('Digest click %s could not be recorded', arxiv_id)
    return redirect(url)


def record_click(subscription, edition_pk, arxiv_id, rank, target, now=None):
    now = now or timezone.now()
    window = DigestClick.objects.filter(subscription=subscription, created__gte=now - datetime.timedelta(seconds=BURST_SECONDS))
    burst = window.count() + 1 >= BURST_CLICKS
    if burst:
        # the clicks just before were the scanner too: they are not a reader's interest
        DigestFeedback.objects.filter(subscription=subscription, source='click',
                                      arxiv_id__in=list(window.values_list('arxiv_id', flat=True))).delete()
        window.update(burst=True)
    edition = DigestEdition.objects.filter(pk=edition_pk, subscription=subscription).first() if edition_pk else None
    DigestClick.objects.create(subscription=subscription, edition=edition, arxiv_id=arxiv_id, rank=rank or 0,
                               target=str(target)[:10], burst=burst)
    if not burst and not DigestFeedback.objects.filter(subscription=subscription, arxiv_id=arxiv_id).exists():
        vector = digest.paper_vector(arxiv_id)
        if vector is not None:
            DigestFeedback.objects.get_or_create(subscription=subscription, arxiv_id=arxiv_id,
                                                 defaults={'value': 1, 'source': 'click', 'vector': vector})


@csrf_exempt  # the link comes from an email; the signed token is the authorization
def save_paper(request, token):
    """Save a paper of a digest to one of its reader's collections, or to a new one."""
    try:
        pk, arxiv_id = signing.loads(token, salt=digest.SAVE_SALT)
    except (signing.BadSignature, ValueError, TypeError):
        raise Http404
    subscription = DigestSubscription.objects.filter(pk=pk).select_related('user').first()
    if subscription is None:
        raise Http404
    user = subscription.user
    item = DigestItem.objects.filter(arxiv_id=arxiv_id).first()
    status, chosen = None, None
    if request.method == 'POST':
        wanted = request.POST.get('collection', '')
        if wanted == 'new':
            if Collection.objects.filter(user=user).count() >= library.limits(access.has_tool(user, 'collections'))['collections']:
                status = 'too_many'
            else:
                name = request.POST.get('name', '').strip()[:120] or (item.title[:120] if item else arxiv_id)
                chosen = Collection.objects.create(user=user, name=name)
        else:
            chosen = Collection.objects.filter(Q(user=user) | Q(members__user=user), pk=wanted if wanted.isdigit() else 0).first()
        if chosen is not None:
            counts = library.add_many(chosen, [library.canonical_id(arxiv_id) or arxiv_id],
                                      access.has_tool(chosen.user, 'collections'))
            status = next((k for k in counts if k in library.ADD_RESULTS), 'unknown')
            status = 'added' if status == 'updated' else status
    return render(request, 'summarizer/digest_save.html', {
        'subscription': subscription, 'item': item, 'arxiv_id': arxiv_id, 'status': status, 'chosen': chosen,
        'collections': library.paper_collections(user, arxiv_id),
    })


# --- the settings ---------------------------------------------------------------------------

def preview_subscription(request, subscription, paid):
    """An unsaved copy of the digest with the settings of the form, to show what it would send."""
    preview = DigestSubscription(user=request.user)
    if subscription is not None:
        preview.pk = subscription.pk  # its votes and clicks rank the preview too (it is never saved)
        for field in ('profile_vector', 'profile_key', 'profile_labels'):
            setattr(preview, field, getattr(subscription, field))
    lim = digest.limits(paid)
    preview.categories = ','.join(digest.clean_categories(
        request.POST.getlist('categories') + request.POST.get('other', '').split(','), lim['categories']))
    preview.keywords = request.POST.get('keywords', '')[:500]
    preview.lang = request.POST.get('lang') if request.POST.get('lang') in ('en', 'fr') else get_language()
    preview.level = 'general' if request.POST.get('level') == 'general' else 'expert'
    preview.frequency = 'weekly' if request.POST.get('frequency') == 'weekly' else 'daily'
    if paid:
        preview.profile = request.POST.get('profile', '').strip()[:2000]
        preview.reference_papers = ', '.join(digest.clean_references(request.POST.get('reference_papers', '')))
        preview.volume = request.POST.get('volume') if request.POST.get('volume') in dict(DigestSubscription.VOLUMES) else 'top'
    return preview


def live_preview(request):
    """What the settings of the form, saved or not, would have sent from the latest announcement. HTML fragment."""
    if not request.user.is_authenticated:
        return HttpResponse(status=403)
    if request.method != 'POST':
        return HttpResponseNotAllowed(['POST'])
    key = 'digest-try:%s' % request.user.pk
    cache.add(key, 0, PREVIEWS[1])
    if cache.incr(key) > PREVIEWS[0]:
        return render(request, 'summarizer/digest_try.html', {'busy': True})
    paid = access.has_tool(request.user, 'digest')
    wanted = request.POST.get('digest')
    subscription = request.user.digests.filter(pk=wanted).first() if wanted and wanted.isdigit() else None
    preview = preview_subscription(request, subscription, paid)
    categories = preview.category_list()
    days = list(ArxivBatch.objects.order_by('-day').values_list('day', flat=True)[:5 if preview.frequency == 'weekly' else 1])
    if not categories or not days:
        return render(request, 'summarizer/digest_try.html', {'empty': not categories, 'no_batch': not days})
    papers = papers_of(days, categories)
    ranking = digest.ranking_for(preview, paid, save=False, quick=True)
    entries = digest.rank_papers(preview, papers, set(), set(days), categories, ranking, digest.stored_vectors(list(papers)))
    limit = digest.limits(paid)['papers']
    threshold = paid and preview.volume in digest.THRESHOLDS and ranking is not None and ranking.has_profile
    chosen = digest.above_threshold(entries, preview.volume)[:digest.THRESHOLD_MAX_EMAIL] if threshold else entries[:limit]
    with translation.override(preview.lang):
        rows = describe(chosen, preview)
    return render(request, 'summarizer/digest_try.html', {
        'rows': rows, 'total': len(entries), 'papers': len(papers), 'day': days[0], 'weekly': preview.frequency == 'weekly',
        'threshold': threshold, 'ranked': bool(ranking), 'profile_pending': paid and preview.profile and ranking is None,
    })


def import_publications(text):
    """What a researcher's own papers suggest for a digest, from an ORCID iD or an arXiv author identifier:
    {'papers': n, 'categories': [...], 'references': [...], 'coauthors': [...]} or {'error': ...}."""
    text = (text or '').strip()
    orcid, author = ORCID_RE.search(text), ARXIV_AUTHOR_RE.match(text)
    try:
        if orcid:
            arxiv_ids, dois = orcid_works(orcid.group(1).upper())
            papers = digest.fetch_ids([i.replace('--', '/') for i in arxiv_ids[:IMPORT_PAPERS]], timeout=20, retry=False)
            references = [p['arxiv_id'] for p in papers][:digest.MAX_REFERENCES]
            references += [sources.doi_key(d) for d in dois][:digest.MAX_REFERENCES - len(references)]
        elif author:
            response = requests.get('https://arxiv.org/a/%s.atom' % author.group(1).lower(), timeout=20,
                                    headers={'User-Agent': sources.USER_AGENT})
            if response.status_code != 200:
                return {'error': 'unknown'}
            papers = digest.parse_feed(response.text)[:IMPORT_PAPERS]
            references = [p['arxiv_id'] for p in papers][:digest.MAX_REFERENCES]
        else:
            return {'error': 'format'}
    except (requests.RequestException, digest.ArxivUnavailable, ValueError, KeyError):
        logger.exception('Import of publications %s failed', text)
        return {'error': 'unavailable'}
    if not papers and not references:
        return {'error': 'empty'}
    primary = collections.Counter(p['categories'][0] for p in papers if p['categories'])
    names, spelling = collections.Counter(), {}
    for paper in papers:
        for name in {digest.name_key(n): n for n in paper['authors'] if digest.name_key(n)}.items():
            names[name[0]] += 1
            spelling.setdefault(name[0], name[1])
    own = names.most_common(1)[0][0] if names else None  # the researcher signs all their papers
    coauthors = [spelling[k] for k, count in names.most_common() if k != own and count >= 2][:IMPORT_COAUTHORS]
    return {'papers': len(papers) or len(references), 'categories': [c for c, _ in primary.most_common(IMPORT_CATEGORIES)],
            'references': references, 'coauthors': coauthors}


def orcid_works(orcid):
    """(arXiv ids, DOIs) of the works of an ORCID record, newest first."""
    response = requests.get('https://pub.orcid.org/v3.0/%s/works' % orcid, timeout=15,
                            headers={'Accept': 'application/json', 'User-Agent': sources.USER_AGENT})
    if response.status_code != 200:
        raise ValueError('ORCID %s: HTTP %s' % (orcid, response.status_code))
    works = []
    for group in response.json().get('group') or []:
        summary = (group.get('work-summary') or [{}])[0]
        year = int((((summary.get('publication-date') or {}).get('year') or {}).get('value')) or 0)
        ids = {}
        for ext in ((group.get('external-ids') or {}).get('external-id') or []):
            ids.setdefault((ext.get('external-id-type') or '').lower(), (ext.get('external-id-value') or '').strip())
        works.append((year, ids))
    works.sort(key=lambda w: w[0], reverse=True)
    arxiv_ids, dois = [], []
    for _, ids in works:
        match = digest.ARXIV_ID_RE.search(ids.get('arxiv', '').replace('arXiv:', ''))
        if match:
            arxiv_ids.append(digest.site_id(match.group(1)))
        elif ids.get('doi'):
            doi = sources.clean_doi(ids['doi'])
            if doi:
                dois.append(doi)
    return arxiv_ids, dois


def apply_import(subscription, found, paid):
    """The digest with what an import suggests added to it (not saved)."""
    lim = digest.limits(paid)
    subscription.categories = ','.join(digest.clean_categories(subscription.category_list() + found['categories'],
                                                               lim['categories']))
    if paid:
        refs = subscription.reference_list() + found['references']
        subscription.reference_papers = ', '.join(list(dict.fromkeys(refs))[:digest.MAX_REFERENCES])
        subscription.authors = ', '.join(digest.clean_authors(', '.join(subscription.author_list() + found['coauthors'])))
    return subscription
