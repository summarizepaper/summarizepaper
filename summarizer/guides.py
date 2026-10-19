"""Editorial guides written in Markdown, one file per language.

A guide lives in content/guides/<lang>/<slug>.md and starts with a YAML
front matter block:

    ---
    title: Best AI tools to summarize arXiv papers (2026)
    description: One sentence, 160 characters at most.
    date: 2026-10-20
    updated: 2026-10-20          # optional
    draft: true                  # drafts are only shown to staff, never listed
    product: true                # optional, adds SummarizePaper as SoftwareApplication (comparisons)
    dataset:                     # optional, first-party data behind a statistics guide
      name: ...
      description: ...
      temporalCoverage: 2023-02-24/2026-10-18
    faq:                         # optional, rendered and marked up as FAQPage
      - q: Question?
        a: Answer.
    sources:                     # optional, rendered under the article
      - label: arXiv submission statistics (2026)
        url: https://...
    ---

The same slug in another language directory is linked with hreflang.

Guides can also be written in the admin (models.Guide): a row replaces the file
with the same language and slug, since files cannot be written on Heroku.
"""
import datetime
import json
import logging
import os
from functools import lru_cache

import markdown
import yaml
from django.conf import settings
from django.http import Http404
from django.shortcuts import render
from django.utils.translation import gettext as _

from .models import Guide

logger = logging.getLogger(__name__)

SITE = 'https://www.summarizepaper.com'
CONTENT_DIR = os.path.join(settings.BASE_DIR, 'content', 'guides')
AUTHOR_URL = SITE + '/en/author/quentin-kral/'


def _path(lang, slug):
    return os.path.join(CONTENT_DIR, lang, slug + '.md')


@lru_cache(maxsize=256)
def _render(text):
    html = markdown.markdown(text, extensions=['extra', 'sane_lists', 'toc'])
    # a wrapper lets wide tables scroll on phones instead of widening the page
    html = html.replace('<table>', '<div class="table-wrap"><table>').replace('</table>', '</table></div>')
    return {'minutes': max(1, round(len(text.split()) / 220)), 'html': html}


@lru_cache(maxsize=256)
def _parse(path, mtime):
    with open(path, encoding='utf-8') as f:
        text = f.read()
    meta = {}
    if text.startswith('---'):
        _, front, text = text.split('---', 2)
        meta = yaml.safe_load(front) or {}
    meta.update(_render(text))
    return meta


def _from_row(row):
    meta = row.extra_meta()
    meta.update(title=row.title, description=row.description, date=row.date, draft=row.draft)
    if row.updated:
        meta['updated'] = row.updated
    if row.product:
        meta['product'] = True
    meta.update(_render(row.body))
    return meta


def load(lang, slug):
    """The guide as a dict, or None when that language has no such guide."""
    if not slug.replace('-', '').isalnum():
        return None
    row = Guide.objects.filter(lang=lang, slug=slug).first()
    path = _path(lang, slug)
    if row is None and not os.path.isfile(path):
        return None
    try:
        guide = _from_row(row) if row else dict(_parse(path, os.path.getmtime(path)))
    except (ValueError, yaml.YAMLError):
        # one broken guide must not take down the sitemap and llms.txt
        logger.exception('Could not parse guide %s/%s', lang, slug)
        return None
    guide.update(slug=slug, lang=lang, url='%s/%s/guides/%s/' % (SITE, lang, slug))
    guide.setdefault('updated', guide.get('date'))
    return guide


def published(lang=None, include_drafts=False):
    """Published guides, newest first, in one language or all of them."""
    langs = [lang] if lang else [l for l, _ in settings.LANGUAGES]
    guides = []
    for l in langs:
        folder = os.path.join(CONTENT_DIR, l)
        slugs = {name[:-3] for name in os.listdir(folder) if name.endswith('.md')} if os.path.isdir(folder) else set()
        slugs.update(Guide.objects.filter(lang=l).values_list('slug', flat=True))
        for slug in slugs:
            guide = load(l, slug)
            if guide and (include_drafts or not guide.get('draft')):
                guides.append(guide)
    return sorted(guides, key=lambda g: str(g.get('date') or ''), reverse=True)


def _iso(value):
    return value.isoformat() if isinstance(value, (datetime.date, datetime.datetime)) else value


def _json_ld(guide):
    graph = [{
        '@type': 'Article',
        'headline': guide.get('title'),
        'description': guide.get('description'),
        'inLanguage': guide['lang'],
        'datePublished': _iso(guide.get('date')),
        'dateModified': _iso(guide.get('updated')),
        'mainEntityOfPage': guide['url'],
        'author': {'@id': AUTHOR_URL + '#person'},
        'publisher': {'@id': SITE + '/#organization'},
    }, {
        '@type': 'BreadcrumbList',
        'itemListElement': [
            {'@type': 'ListItem', 'position': 1, 'name': 'SummarizePaper', 'item': '%s/%s/' % (SITE, guide['lang'])},
            {'@type': 'ListItem', 'position': 2, 'name': _('Guides'), 'item': '%s/%s/guides/' % (SITE, guide['lang'])},
            {'@type': 'ListItem', 'position': 3, 'name': guide.get('title'), 'item': guide['url']},
        ],
    }]
    if guide.get('product'):
        graph.append({
            '@type': 'SoftwareApplication',
            'name': 'SummarizePaper',
            'url': SITE + '/',
            'applicationCategory': 'EducationalApplication',
            'operatingSystem': 'Web',
            'publisher': {'@id': SITE + '/#organization'},
            'offers': [
                {'@type': 'Offer', 'name': 'Free', 'price': '0', 'priceCurrency': 'EUR'},
                {'@type': 'Offer', 'name': 'Premium', 'price': '7.00', 'priceCurrency': 'EUR',
                 'url': SITE + '/en/premium/'},
            ],
        })
    if guide.get('dataset'):
        dataset = guide['dataset']
        graph.append({
            '@type': 'Dataset',
            'name': dataset.get('name'),
            'description': dataset.get('description'),
            'temporalCoverage': dataset.get('temporalCoverage'),
            'url': guide['url'],
            'creator': {'@id': SITE + '/#organization'},
            'isAccessibleForFree': True,
        })
    if guide.get('faq'):
        graph.append({
            '@type': 'FAQPage',
            'mainEntity': [{'@type': 'Question', 'name': item['q'],
                            'acceptedAnswer': {'@type': 'Answer', 'text': item['a']}}
                           for item in guide['faq']],
        })
    # "</" would end the script tag early
    return json.dumps({'@context': 'https://schema.org', '@graph': graph}, ensure_ascii=False).replace('</', '<\\/')


def guide_index(request):
    guides = published(request.LANGUAGE_CODE, include_drafts=request.user.is_staff)
    return render(request, 'summarizer/guides.html', {'guides': guides})


def guide_detail(request, slug):
    lang = request.LANGUAGE_CODE
    guide = load(lang, slug)
    if guide is None or (guide.get('draft') and not request.user.is_staff):
        raise Http404
    alternates = [load(l, slug) for l, _ in settings.LANGUAGES]
    alternates = [g for g in alternates if g and not g.get('draft')]
    return render(request, 'summarizer/guide.html', {
        'guide': guide,
        'alternates': alternates if len(alternates) > 1 else [],
        'json_ld': _json_ld(guide),
    })
