"""What changed between two versions of an arXiv paper (Premium), and an email when a followed paper gets a new one.

arXiv gives each version's abstract and comment ("v2: corrected Fig. 3"). When
the paper's license lets us read it (see sources.is_open_license), the HTML of
both versions is read too and only the paragraphs that differ are given to the
model, which says what changed: results, method, data, conclusions, an error
fixed, or only the writing. A comparison is written once per paper, pair of
versions and language, and shared by every reader.

Once a day, the hourly run reads the latest version of the papers that Premium
users follow (citation alerts, see citations.py). A new version is compared
with the previous one and emailed, with what changed, to their followers.
Papers kept in collections are left out: a collection can hold 200 papers,
and nobody asked to hear about each of them.
"""
import difflib
import logging
import re

import requests
from django.conf import settings
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.http import Http404, JsonResponse
from django.shortcuts import render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_POST

from . import access, digest, generated, mailcheck, paper_source, sources
from .models import ArxivPaper, FollowedPaper, Generated, PaperVersion, VersionAlert

logger = logging.getLogger(__name__)

OAI = 'https://oaipmh.arxiv.org/oai'
MAX_DIFF_CHARS = 40000
SAME_PARAGRAPH = 0.97  # a paragraph this close to its old self only had typos fixed
CHECK_BATCH = 100  # ids per arXiv query
MAX_CHECKED_PER_DAY = 3000
MAX_COMPARISONS_PER_RUN = 4  # about a minute each
DAILY_KEY = 'versions-checked:%s'
KINDS = ('results', 'conclusions', 'method', 'data', 'error_fix', 'figures', 'references', 'writing')
STEPS = [
    ('read', _('Reading both versions')),
    ('diff', _('Finding the paragraphs that changed')),
    ('write', _('An AI explains what changed')),
]


def base_id(arxiv_id):
    """'2101.00001v2' or 'hep-th--9901001v1' -> ('2101.00001', 2) / ('hep-th/9901001', 1); version 0 when not given."""
    ref = arxiv_id.replace('--', '/')
    match = re.match(r'^(.*?)(?:v(\d+))?$', ref)
    return match.group(1), int(match.group(2) or 0)


def latest_version(base):
    """The number of arXiv's latest version of a paper (0 when arXiv does not know it), cached a few hours."""
    key = 'arxiv-latest:%s' % base
    found = cache.get(key)
    if found is None:
        papers = digest.fetch_ids([base])
        found = base_id(papers[0]['arxiv_id'])[1] if papers else 0
        cache.set(key, found, 6 * 3600)
    return found


def version_record(base, version):
    """arXiv's record of one version: title, abstract, comment, dates."""
    papers = digest.fetch_ids(['%sv%d' % (base, version)])
    if not papers:
        raise generated.GenerationError('arXiv does not know %sv%d' % (base, version))
    return papers[0]


def day(record):
    """'2026-10-20', the date this version was posted."""
    date = record.get('updated') or record.get('published')
    return date.date().isoformat() if date else ''


def arxiv_license(base):
    """The license URL arXiv gives a paper (OAI-PMH), '' when it gives none; cached a month."""
    key = 'arxiv-license:%s' % base
    found = cache.get(key)
    if found is None:
        try:
            response = requests.get(OAI, timeout=30, params={
                'verb': 'GetRecord', 'identifier': 'oai:arXiv.org:%s' % base, 'metadataPrefix': 'arXiv'})
            match = re.search(r'<license>\s*([^<\s]+)\s*</license>', response.text) if response.status_code == 200 else None
        except requests.RequestException:
            return ''  # not cached: tried again next time
        found = match.group(1) if match else ''
        cache.set(key, found, 30 * 86400)
    return found


def can_read(base):
    """True when the paper's license lets us read its full text (the license is the same for every version):
    the license we stored when the paper was summarized, or the one arXiv gives."""
    paper = ArxivPaper.objects.filter(arxiv_id__startswith=base.replace('/', '--') + 'v').exclude(license=None).first()
    return sources.is_open_license(paper.license if paper else arxiv_license(base))


def version_html(base, version):
    """Paragraphs of the arXiv HTML of one version, [] when it has none."""
    try:
        response = requests.get(paper_source.ARXIV_HTML % ('%sv%d' % (base, version)), timeout=60,
                                headers={'User-Agent': 'SummarizePaper (https://www.summarizepaper.com)'})
    except requests.RequestException:
        return []
    if response.status_code != 200 or '/html/' not in response.url:
        return []
    text = paper_source.html_to_text(response.text)
    return [' '.join(p.split()) for p in text.split('\n\n') if len(p.strip()) > 40]


def changed_paragraphs(old, new):
    """'- old paragraph' / '+ new paragraph' lines of what differs, typo fixes left out, cut at MAX_DIFF_CHARS."""
    out, size = [], 0
    for op, a1, a2, b1, b2 in difflib.SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if op == 'equal':
            continue
        before, after = old[a1:a2], new[b1:b2]
        if op == 'replace' and len(before) == len(after) and all(
                difflib.SequenceMatcher(None, x, y).ratio() > SAME_PARAGRAPH for x, y in zip(before, after)):
            continue
        for line in ['- ' + p for p in before] + ['+ ' + p for p in after]:
            if size + len(line) > MAX_DIFF_CHARS:
                out.append('[... more changes left out]')
                return '\n\n'.join(out)
            out.append(line)
            size += len(line)
    return '\n\n'.join(out)


def compare(base, old, new, lang, step=lambda name: None):
    """The comparison shown on the page and in the alerts."""
    before, after = version_record(base, old), version_record(base, new)
    full = can_read(base)
    diff = ''
    if full:
        old_text, new_text = version_html(base, old), version_html(base, new)
        step('diff')
        if old_text and new_text:
            diff = changed_paragraphs(old_text, new_text)
        else:
            full = False
    step('write')
    material = '\n\n'.join('<version number="%d" date="%s">\nTitle: %s\nComment: %s\nAbstract: %s\n</version>' % (
        number, day(record), record['title'], record.get('comment', ''), record['abstract'])
        for number, record in ((old, before), (new, after)))
    if full:
        material += '\n\n<changed_paragraphs>\n%s\n</changed_paragraphs>' % (diff or 'none: the full texts are the same')
    answer = generated.chat_json(
        'Above are two versions of the same arXiv paper: %s. Tell a researcher who read version %d what changed in '
        'version %d. Write in %s. List the changes that matter, most important first: changed results or numbers '
        '(give before and after), changed conclusions, method, data, an error fixed, figures, references, or only the '
        'writing. Say plainly whether the main results changed. Do not invent changes you cannot see. No LaTeX, no '
        'markdown. Answer with a JSON object {"verdict": "one or two sentences", "results_changed": true|false, '
        '"changes": [{"kind": "results|conclusions|method|data|error_fix|figures|references|writing", "importance": '
        '"major|minor", "detail": "one or two sentences"}]}.' % (
            'their abstracts, comments and the paragraphs of the full text that differ' if full else
            'their abstracts and comments only (the full text could not be compared)',
            old, new, generated.language_name(lang)),
        paid=True, max_tokens=3000, system=material)
    changes = [{'kind': c.get('kind') if c.get('kind') in KINDS else 'writing',
                'importance': 'major' if c.get('importance') == 'major' else 'minor',
                'detail': str(c.get('detail', '')).strip()}
               for c in answer.get('changes', []) if isinstance(c, dict) and str(c.get('detail', '')).strip()][:15]
    changes.sort(key=lambda c: c['importance'] != 'major')
    return {'title': after['title'], 'old': old, 'new': new, 'full_text': full,
            'old_date': day(before), 'new_date': day(after),
            'old_comment': before.get('comment', ''), 'new_comment': after.get('comment', ''),
            'verdict': str(answer.get('verdict', '')).strip(), 'results_changed': bool(answer.get('results_changed')),
            'changes': changes}


def comparison_key(base, old, new):
    return '%s:%d:%d' % (base.replace('/', '--'), old, new)


def write(base, old, new, lang):
    def work(pk):
        generated.progress(pk, 'read')
        return compare(base, old, new, lang, lambda name: generated.progress(pk, name))
    return work


# --- daily check of followed papers -----------------------------------------

def watched():
    """{arxiv base id: set of user ids} of the arXiv papers followed."""
    found = {}
    for user_id, arxiv_id in FollowedPaper.objects.values_list('user_id', 'arxiv_id'):
        if not sources.is_external(arxiv_id):
            found.setdefault(base_id(arxiv_id)[0], set()).add(user_id)
    return found


def check_versions(now, limit=MAX_CHECKED_PER_DAY):
    """Read the latest version of the watched papers; queue an alert for each new one. Returns how many are new."""
    users = watched()
    known = dict(PaperVersion.objects.filter(arxiv_id__in=list(users)).values_list('arxiv_id', 'version'))
    ids = sorted(users)[:limit]
    new = 0
    for start in range(0, len(ids), CHECK_BATCH):
        batch = ids[start:start + CHECK_BATCH]
        try:
            papers = digest.fetch_ids(batch)
        except digest.ArxivUnavailable:
            logger.warning('arXiv versions check failed')
            continue
        for paper in papers:
            base, version = base_id(paper['arxiv_id'])
            if base not in users or not version:
                continue
            previous = known.get(base)
            PaperVersion.objects.update_or_create(arxiv_id=base, defaults={'version': version, 'checked': now})
            if previous and version > previous:
                new += 1
                VersionAlert.objects.bulk_create(
                    [VersionAlert(user_id=uid, arxiv_id=base, old=previous, new=version) for uid in users[base]],
                    ignore_conflicts=True)
    return new


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """Write the comparisons the queued alerts need (a few per run), then email them. Returns the emails sent."""
    from django.contrib.auth.models import User
    comparisons = MAX_COMPARISONS_PER_RUN
    suppressed = mailcheck.suppressed()
    sent = 0
    for user in User.objects.filter(pk__in=list(VersionAlert.objects.filter(emailed=False)
                                                 .values_list('user_id', flat=True).distinct())):
        if sent >= budget:
            break
        alerts = list(VersionAlert.objects.filter(user=user, emailed=False))
        if not user.email or user.email.strip().lower() in suppressed or not access.is_paid(user):
            VersionAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True)
            continue
        lang = user.digests.values_list('lang', flat=True).first() or 'en'
        items, waiting = [], False
        for alert in alerts:
            key = comparison_key(alert.arxiv_id, alert.old, alert.new)
            row = Generated.objects.filter(kind='changes', key=key, lang=lang).first()
            if row is None or row.status != 'done':
                if comparisons <= 0:
                    waiting = True
                    continue
                comparisons -= 1
                row = generated.run_now('changes', key, lang, write(alert.arxiv_id, alert.old, alert.new, lang))
            if row.status == 'done':
                items.append(dict(row.data(), url='%s/%s/changes/%sv%d/?from=%d' % (
                    base_url, lang, alert.arxiv_id.replace('/', '--'), alert.new, alert.old)))
        if waiting or not items:
            continue  # the rest of the comparisons are written in the next runs, then one email has them all
        context = {'items': items, 'count': len(items), 'base_url': base_url,
                   'manage_url': '%s/%s/citations/' % (base_url, lang)}
        with translation.override(lang):
            subject = render_to_string('changes/subject.txt', context).strip()
            text = render_to_string('changes/email.txt', context)
            html = render_to_string('changes/email.html', context)
        message = EmailMultiAlternatives(subject, text, settings.DEFAULT_FROM_EMAIL, [user.email])
        message.attach_alternative(html, 'text/html')
        try:
            message.send()
        except Exception:
            logger.exception('Version alert to %s failed', user.username)
            continue
        VersionAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True)
        sent += 1
    return sent


def run(now, budget):
    """From the hourly digest run: read the versions once a day, then send what is ready within the email budget."""
    if cache.add(DAILY_KEY % now.date(), True, 86400):
        check_versions(now)
    return send_alerts(now, budget) if budget > 0 else 0


# --- views -------------------------------------------------------------------

def _versions(arxiv_id, request):
    if sources.is_external(arxiv_id):
        raise Http404
    base, shown = base_id(arxiv_id)
    try:
        latest = latest_version(base)
    except digest.ArxivUnavailable:
        latest = shown
    if not latest:
        raise Http404
    new = min(int(request.GET.get('to') or request.POST.get('to') or latest), latest)
    old = int(request.GET.get('from') or request.POST.get('from') or new - 1)
    return base, latest, max(1, min(old, new - 1)), new


def changes_page(request, arxiv_id):
    base, latest, old, new = _versions(arxiv_id, request)
    paid = access.is_paid(request.user)
    row = Generated.objects.filter(kind='changes', key=comparison_key(base, old, new),
                                   lang=request.LANGUAGE_CODE).first() if paid and latest > 1 else None
    title = ArxivPaper.objects.filter(arxiv_id__startswith=base.replace('/', '--') + 'v').values_list('title', flat=True).first()
    return render(request, 'summarizer/changes.html', {
        'arxiv_id': arxiv_id, 'base': base, 'site_base': base.replace('/', '--'), 'latest': latest, 'old': old,
        'new': new, 'versions': range(1, latest + 1), 'paid': paid, 'row': row, 'title': title or base,
        'result': row.data() if row is not None and row.status == 'done' else None, 'steps': STEPS})


@require_POST
def changes_start(request, arxiv_id):
    if not access.is_paid(request.user):
        return JsonResponse({'status': 'premium'}, status=403)
    base, latest, old, new = _versions(arxiv_id, request)
    if latest < 2:
        return JsonResponse({'status': 'one_version'})
    lang = request.LANGUAGE_CODE
    row = generated.start('changes', comparison_key(base, old, new), lang, write(base, old, new, lang),
                          force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})
