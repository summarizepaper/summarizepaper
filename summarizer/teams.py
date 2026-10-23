"""Who works on a topic (Premium): the research groups that published the most on it in the last two years.

1. The model writes a few literature searches from the topic (or from a project
   of the scoop watch, see scoops.py).
2. OpenAlex answers them with the papers of the last two years, journals and
   preprints, with their authors' institutions; the papers are kept when their
   title is close enough to the topic (embeddings), so a broad search word does
   not bring in another field.
3. The papers are grouped by institution: papers, citations, the people who
   sign them, and how often they sign last (often the head of the group).
4. The model reads the titles of each group and says what it focuses on and
   how, and sums up the landscape.
"""
import datetime
import logging
import uuid

import numpy as np
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.http import Http404
from django.shortcuts import redirect, render
from django.utils import timezone
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _

from . import access, generated, similar, sources
from .models import Generated, WatchedProject

logger = logging.getLogger(__name__)

QUERIES = 4
PER_QUERY = 100
YEARS = 2
CLOSE = 0.35  # cosine of a paper title to the topic, below which it is about something else
MIN_CLOSE = 30  # when fewer papers pass CLOSE, the closest ones are kept anyway
GROUPS = 12
PEOPLE = 5
TITLES = 6
MIN_CHARS = 20
LARGE_TEAM = 25  # papers with more authors count only for the institutions of their first and last authors
MAX_CHARS = 4000
PER_DAY = 10
QUOTA_KEY = 'teams-quota:%s:%s'
FIELDS = 'id,doi,display_name,authorships,publication_date,cited_by_count,primary_location'
STEPS = [
    ('plan', _('Writing the searches')),
    ('search', _('Searching the papers of the last two years on OpenAlex')),
    ('group', _('Grouping the papers by institution')),
    ('write', _('An AI describes what each group does')),
]


def works(query, since):
    """OpenAlex works matching a query, published since `since`; None when today's budget is used up."""
    if not sources.tool_search_allowed():
        return None
    try:
        body = sources._json('%s/works' % sources.OPENALEX, sources._openalex_params({
            'search': query, 'filter': 'from_publication_date:%s' % since.isoformat(), 'per_page': PER_QUERY,
            'select': FIELDS}))
    except sources.SourceUnavailable:
        logger.warning('OpenAlex search failed for %s', query)
        return []
    return (body or {}).get('results', [])


def paper_link(work):
    doi = (work.get('doi') or '').split('doi.org/')[-1]
    arxiv = sources.ARXIV_DOI_RE.match(doi)
    if arxiv:
        return 'https://arxiv.org/abs/%s' % arxiv.group(1)
    return 'https://doi.org/%s' % doi if doi else ((work.get('primary_location') or {}).get('landing_page_url') or '')


def person_name(name):
    """'Cassese, Ben' -> 'Ben Cassese', so one person is counted once."""
    if ',' in name:
        last, _, first = name.partition(',')
        name = '%s %s' % (first.strip(), last.strip())
    return ' '.join(name.split())


def group(papers):
    """[{institution, country, papers, citations, people, titles}] by number of papers, the biggest first."""
    groups = {}
    for work in papers:
        seen = set()
        authorships = work.get('authorships') or []
        if len(authorships) > LARGE_TEAM:  # a large collaboration or a review would land in every group
            authorships = [a for a in authorships if a.get('author_position') in ('first', 'last')]
        for authorship in authorships:
            name = person_name((authorship.get('author') or {}).get('display_name') or '')
            for institution in (authorship.get('institutions') or [])[:1]:  # an author's first affiliation
                key = institution.get('id') or institution.get('display_name')
                if not key:
                    continue
                g = groups.setdefault(key, {'institution': institution.get('display_name') or '',
                                            'country': institution.get('country_code') or '',
                                            'works': {}, 'people': {}})
                g['works'][work['id']] = work
                if name:
                    person = g['people'].setdefault(name, {'name': name, 'papers': 0, 'last': 0})
                    if (key, name) not in seen:
                        person['papers'] += 1
                        person['last'] += authorship.get('author_position') == 'last'
                        seen.add((key, name))
    found = []
    for key, g in groups.items():
        listed = sorted(g['works'].values(), key=lambda w: -(w.get('cited_by_count') or 0))
        people = sorted(g['people'].values(), key=lambda p: (-p['papers'], -p['last']))[:PEOPLE]
        found.append({'id': key, 'institution': g['institution'], 'country': g['country'], 'papers': len(listed),
                      'citations': sum(w.get('cited_by_count') or 0 for w in listed), 'people': people,
                      'titles': [{'title': w.get('display_name') or '', 'url': paper_link(w),
                                  'year': (w.get('publication_date') or '')[:4]} for w in listed[:TITLES]]})
    found.sort(key=lambda g: (-g['papers'], -g['citations']))
    return found


def find(topic, lang, step=lambda name: None):
    plan = generated.chat_json(
        'Above is a research topic or project. Answer with a JSON object {"title": "a short name for the topic, in %s", '
        '"queries": ["%d literature search queries of 2 to 5 keywords each, in English, covering the topic and its '
        'main approaches"]}.' % (generated.language_name(lang), QUERIES),
        paid=False, max_tokens=500, system='<topic>\n%s\n</topic>' % topic[:MAX_CHARS])
    queries = [str(q) for q in plan.get('queries', []) if str(q).strip()][:QUERIES]
    step('search')
    since = timezone.now().date() - datetime.timedelta(days=365 * YEARS)
    found, limited = {}, False
    for query in queries:
        answer = works(query, since)
        limited = limited or answer is None
        for work in answer or []:
            if work.get('id') and work.get('display_name'):
                found.setdefault(work['id'], work)
    papers = list(found.values())
    if not papers:
        return {'title': str(plan.get('title', '')).strip(), 'groups': [], 'papers': 0, 'landscape': '', 'limited': limited}
    step('group')
    vectors = similar.embed([topic[:similar.MAX_INPUT_CHARS]] + [w['display_name'][:500] for w in papers])
    scores = vectors[1:] @ vectors[0]
    order = list(np.argsort(-scores))
    kept = [papers[i] for i in order if scores[i] >= CLOSE] or []
    if len(kept) < MIN_CLOSE:
        kept = [papers[i] for i in order[:MIN_CLOSE]]
    groups = group(kept)[:GROUPS]
    step('write')
    listing = '\n\n'.join('<group id="%d">\nInstitution: %s (%s)\nPaper titles: %s\n</group>' % (
        i, g['institution'], g['country'], ' | '.join(t['title'] for t in g['titles'])) for i, g in enumerate(groups))
    notes = generated.chat_json(
        'The groups above published on this topic: %s. In %s, for each group, say in one sentence what it focuses on '
        'and in one sentence its approach (method, instrument, data, theory or experiment), from its paper titles '
        'only. Then sum up in 3 sentences the landscape: the main approaches and who leads each. Do not invent what '
        'the titles do not say. Answer with a JSON object {"groups": [{"id": "group id", "focus": "...", '
        '"approach": "..."}], "landscape": "..."}.' % (topic[:600], generated.language_name(lang)),
        paid=True, max_tokens=3000, system=listing)
    by_id = {str(n.get('id')): n for n in notes.get('groups', []) if isinstance(n, dict)}
    for i, g in enumerate(groups):
        note = by_id.get(str(i), {})
        g['focus'], g['approach'] = str(note.get('focus', '')).strip(), str(note.get('approach', '')).strip()
    return {'title': str(plan.get('title', '')).strip(), 'groups': groups, 'papers': len(kept),
            'landscape': str(notes.get('landscape', '')).strip(), 'since': since.year, 'limited': limited}


def write(topic, lang):
    def work(pk):
        generated.progress(pk, 'plan')
        return find(topic, lang, lambda name: generated.progress(pk, name))
    return work


def _spend(user):
    key = QUOTA_KEY % (user.pk, timezone.now().date())
    cache.add(key, 0, 86400)
    return cache.incr(key) <= PER_DAY


def teams_page(request):
    if not request.user.is_authenticated:
        return render(request, 'summarizer/teams.html', {'anonymous': True, 'per_day': PER_DAY})
    paid = access.has_tool(request.user, 'teams')
    projects = WatchedProject.objects.filter(user=request.user).exclude(description='')
    error = None
    if request.method == 'POST' and paid:
        project = projects.filter(pk=request.POST.get('project') or 0).first()
        topic = ('%s\n\n%s' % (project.title, project.search_text or project.description) if project
                 else request.POST.get('topic', '').strip()[:MAX_CHARS])
        if len(topic) < MIN_CHARS:
            error = 'empty'
        elif not _spend(request.user):
            error = 'quota'
        else:
            row = generated.start('teams', uuid.uuid4().hex, get_language(), write(topic, get_language()), user=request.user)
            return redirect('/%s/teams/%s/' % (get_language(), row.key))
    searches = Generated.objects.filter(kind='teams', user=request.user).order_by('-created')[:20]
    return render(request, 'summarizer/teams.html', {'paid': paid, 'error': error, 'searches': searches,
                                                     'projects': projects, 'per_day': PER_DAY})


@login_required
def teams_result(request, key):
    if not access.has_tool(request.user, 'teams'):
        raise Http404
    row = Generated.objects.filter(kind='teams', key=key, user=request.user).first()
    if row is None:
        raise Http404
    if row.status == 'pending' and row.updated < timezone.now() - generated.STALE:
        row.status = 'failed'  # lost in a restart
    return render(request, 'summarizer/teams_result.html', {
        'row': row, 'result': row.data() if row.status == 'done' else None, 'steps': STEPS})
