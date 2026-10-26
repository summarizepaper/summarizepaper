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
Papers kept in collections are left out unless their owner watches the
collection (library.follow_papers then follows each of them): a collection can
hold 200 papers, and nobody asked to hear about each of them otherwise.
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


def latest_version(base, **options):
    """The number of arXiv's latest version of a paper (0 when arXiv does not know it), cached a few hours."""
    key = 'arxiv-latest:%s' % base
    found = cache.get(key)
    if found is None:
        papers = digest.fetch_ids([base], **options)
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
    paper = (ArxivPaper.objects.filter(arxiv_id__startswith=base.replace('/', '--') + 'v')
             .exclude(license=None).exclude(license='').first())
    return sources.is_open_license(paper.license if paper else arxiv_license(base))


def license_preview(base):
    """Use known licenses on page loads; never block the page on an OAI request."""
    license = (ArxivPaper.objects.filter(arxiv_id__startswith=base.replace('/', '--') + 'v')
               .exclude(license__isnull=True).exclude(license='')
               .values_list('license', flat=True).first())
    license = license or cache.get('arxiv-license:%s' % base)
    return 'unknown' if not license else ('allowed' if sources.is_open_license(license) else 'restricted')


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
    scope = 'full_text' if full else 'abstract_license'
    diff = ''
    if full:
        old_text, new_text = version_html(base, old), version_html(base, new)
        step('diff')
        if old_text and new_text:
            diff = changed_paragraphs(old_text, new_text)
        else:
            full = False
            scope = 'abstract_unavailable'
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
        'writing. Say plainly whether the main results changed. When only abstracts and comments are provided, '
        'explicitly limit your verdict to those sources; never conclude that the full paper or its results are unchanged. '
        'Do not invent changes you cannot see. No LaTeX, no '
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
    return {'title': after['title'], 'old': old, 'new': new, 'full_text': full, 'source_scope': scope,
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


def ready_items(alerts, base_url, lang, comparisons):
    """(items, waiting): the comparisons of these alerts, written within the budget `comparisons` (a one-element
    list); waiting is True when some are left for the next runs."""
    items, waiting = [], False
    for alert in alerts:
        key = comparison_key(alert.arxiv_id, alert.old, alert.new)
        row = Generated.objects.filter(kind='changes', key=key, lang=lang).first()
        if row is None or row.status != 'done':
            if comparisons[0] <= 0:
                waiting = True
                continue
            comparisons[0] -= 1
            row = generated.run_now('changes', key, lang, write(alert.arxiv_id, alert.old, alert.new, lang))
        if row.status == 'done':
            items.append(dict(row.data(), url='%s/%s/changes/%sv%d/?from=%d' % (
                base_url, lang, alert.arxiv_id.replace('/', '--'), alert.new, alert.old)))
    return items, waiting


def bundle_section(user, base_url, lang):
    """The new versions of followed papers as a section of the user's digest (see digest_bundle)."""
    alerts = list(VersionAlert.objects.filter(user=user, emailed=False))
    if not alerts or not access.has_tool(user, 'changes'):
        return None
    items, waiting = ready_items(alerts, base_url, lang, [MAX_COMPARISONS_PER_RUN])
    if waiting or not items:
        return None  # in the next digest, with all of them
    return {'title': translation.gettext('New versions of the papers you follow'), 'more': 0,
            'manage_url': '%s/%s/citations/' % (base_url, lang),
            'commit': lambda: VersionAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True),
            'items': [{'title': '%s (v%s → v%s)' % (i.get('title', ''), i.get('old', ''), i.get('new', '')), 'url': i['url'],
                       'label': translation.gettext('The results changed') if i.get('results_changed') else
                       translation.gettext('Same main results'), 'note': i.get('verdict', '')} for i in items]}


def send_alerts(now, budget, base_url=digest.BASE_URL):
    """Write the comparisons the queued alerts need (a few per run), then email them. Returns the emails sent.
    The alerts of a user whose digest carries them are left for it (see digest_bundle)."""
    from django.contrib.auth.models import User
    from . import digest_bundle
    comparisons = [MAX_COMPARISONS_PER_RUN]
    suppressed = mailcheck.suppressed()
    bundled = digest_bundle.bundled_users()
    sent = 0
    for user in User.objects.filter(pk__in=list(VersionAlert.objects.filter(emailed=False)
                                                 .values_list('user_id', flat=True).distinct())):
        if sent >= budget:
            break
        alerts = list(VersionAlert.objects.filter(user=user, emailed=False))
        if not user.email or user.email.strip().lower() in suppressed or not access.has_tool(user, 'changes'):
            VersionAlert.objects.filter(pk__in=[a.pk for a in alerts]).update(emailed=True)
            continue
        lang = user.digests.values_list('lang', flat=True).first() or 'en'
        items, waiting = ready_items(alerts, base_url, lang, comparisons)
        if waiting or not items or user.pk in bundled:
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

def _number(value, default):
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def _versions(arxiv_id, request, ask_arxiv=True):
    """(base id, latest version, old, new). Without ask_arxiv, the latest version is the one we know: crawlers
    follow the link of every paper page, and arXiv answers one query every 3 seconds for the whole site."""
    if sources.is_external(arxiv_id):
        raise Http404
    base, shown = base_id(arxiv_id)
    latest = max(shown, PaperVersion.objects.filter(arxiv_id=base).values_list('version', flat=True).first() or 0)
    if ask_arxiv:
        try:
            # one short try: a slow arXiv used to hold the page past Heroku's 30 seconds (H12)
            latest = latest_version(base, timeout=8, retry=False) or latest
        except digest.ArxivUnavailable:
            pass
    if not latest:
        raise Http404
    new = max(min(2, latest), min(_number(request.GET.get('to') or request.POST.get('to'), latest), latest))
    old = _number(request.GET.get('from') or request.POST.get('from'), new - 1)
    return base, latest, max(1, min(old, new - 1)), new


def changes_page(request, arxiv_id):
    paid = access.has_tool(request.user, 'changes')
    base, latest, old, new = _versions(arxiv_id, request, ask_arxiv=paid)
    rows = []
    if paid and latest > 1:
        prefix = base.replace('/', '--') + ':'
        for saved in Generated.objects.filter(kind='changes', key__startswith=prefix,
                                               lang=request.LANGUAGE_CODE).order_by('-updated'):
            parts = saved.key[len(prefix):].split(':')
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                a, b = map(int, parts)
                if 1 <= a < b <= latest:
                    rows.append({'row': saved, 'old': a, 'new': b})
        if not request.GET.get('from') and not request.GET.get('to'):
            recent = next((item for item in rows if item['row'].status in ('done', 'pending')), None)
            if recent:
                old, new = recent['old'], recent['new']
    row = next((item['row'] for item in rows if (item['old'], item['new']) == (old, new)), None)
    title = ArxivPaper.objects.filter(arxiv_id__startswith=base.replace('/', '--') + 'v').values_list('title', flat=True).first()
    return render(request, 'summarizer/changes.html', {
        'arxiv_id': arxiv_id, 'base': base, 'site_base': base.replace('/', '--'), 'latest': latest, 'old': old,
        'new': new, 'versions': range(1, latest + 1), 'paid': paid, 'row': row, 'title': title or base,
        'result': row.data() if row is not None and row.status == 'done' else None, 'steps': STEPS,
        'history': rows, 'license_preview': license_preview(base) if paid and latest > 1 else None})


@require_POST
def changes_start(request, arxiv_id):
    if not access.has_tool(request.user, 'changes'):
        return JsonResponse({'status': 'premium'}, status=403)
    base, latest, old, new = _versions(arxiv_id, request)
    if latest < 2:
        return JsonResponse({'status': 'one_version'})
    if (str(old) != request.POST.get('from', str(old)) or str(new) != request.POST.get('to', str(new))):
        return JsonResponse({'status': 'invalid_versions'}, status=400)
    lang = request.LANGUAGE_CODE
    row = generated.start('changes', comparison_key(base, old, new), lang, write(base, old, new, lang),
                          force=request.POST.get('again') == '1')
    return JsonResponse({'status': row.status, 'id': row.pk})
