"""Weekly letters: 'AI this week', 'Space this week'... written like an article for curious readers.

Once a week, for each theme, the model picks the papers of the last seven
announcements that matter most to a general reader, then writes an article
about them, in each language. The letters are public pages (and in the
sitemap); Premium accounts can also get them by email.
"""
import datetime
import json
import logging
import random

from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.core.cache import cache
from django.core.mail import EmailMultiAlternatives
from django.http import Http404
from django.shortcuts import redirect, render
from django.template.loader import render_to_string
from django.utils import timezone, translation
from django.utils.translation import get_language
from django.utils.translation import gettext_lazy as _
from django.views.decorators.csrf import csrf_exempt

from . import access, digest, generated, mailcheck
from .models import AnnouncedPaper, ArxivBatch, Letter, LetterSubscription

logger = logging.getLogger(__name__)

TOPICS = {
    'ai': {'name': _('Artificial intelligence'), 'categories': {'cs.AI', 'cs.LG', 'cs.CL', 'cs.CV', 'cs.RO', 'stat.ML'}},
    'space': {'name': _('Astronomy and space'), 'archives': {'astro-ph'}},
    'quantum': {'name': _('Quantum physics'), 'categories': {'quant-ph'}},
    'life': {'name': _('Biology and medicine'), 'archives': {'q-bio'}, 'categories': {'physics.bio-ph', 'physics.med-ph'}},
    'climate': {'name': _('Climate and Earth'), 'categories': {'physics.ao-ph', 'physics.geo-ph', 'astro-ph.EP'}},
    'economy': {'name': _('Economics and finance'), 'archives': {'econ', 'q-fin'}},
}
LANGS = ('en', 'fr')
WEEK = datetime.timedelta(days=7)
MAX_LISTED = 400  # titles the model chooses from
PICKED = 6
MIN_PAPERS = 20
LETTERS_PER_RUN = 4
LOCK = 'letters-run'
UNSUBSCRIBE_SALT = 'letter-unsubscribe'


def in_topic(topic, categories):
    spec = TOPICS[topic]
    return any(c in spec.get('categories', ()) or c.split('.')[0] in spec.get('archives', ()) for c in categories)


def week_papers(topic, day):
    papers = []
    for row in AnnouncedPaper.objects.filter(announced__gt=day - WEEK, announced__lte=day).order_by('arxiv_id'):
        categories = [c for c in row.categories.split(',') if c]
        if in_topic(topic, categories):
            papers.append({'arxiv_id': row.arxiv_id, 'title': row.title, 'abstract': row.abstract,
                           'primary': in_topic(topic, categories[:1])})
    return papers


def pick(topic, day, papers, lang):
    """The papers of the letter: the ones already chosen for this week in another language, or the model's choice."""
    other = Letter.objects.filter(topic=topic, day=day).exclude(lang=lang).first()
    by_id = {p['arxiv_id']: p for p in papers}
    if other is not None:
        chosen = [by_id[a] for a in json.loads(other.content).get('papers', {}) if a in by_id]
        if chosen:
            return chosen
    listed = [p for p in papers if p['primary']] or papers
    if len(listed) > MAX_LISTED:
        listed = random.Random('%s%s' % (topic, day)).sample(listed, MAX_LISTED)
    answer = generated.chat_json(
        'Here are arXiv papers of the last week in "%s" (ID | title). Choose the %d that a curious general reader '
        'would find most interesting or important, favouring concrete results over incremental work, and varied '
        'subjects. Answer with a JSON object {"ids": ["ID", ...]}.\n\n%s'
        % (TOPICS[topic]['name'], PICKED, '\n'.join('%s | %s' % (p['arxiv_id'], p['title']) for p in listed)),
        paid=False, max_tokens=500)
    return [by_id[a] for a in dict.fromkeys(answer.get('ids', [])) if a in by_id][:PICKED]


def write(topic, day, lang):
    papers = week_papers(topic, day)
    if len(papers) < MIN_PAPERS:
        return None
    chosen = pick(topic, day, papers, lang)
    if len(chosen) < 3:
        return None
    with translation.override(lang):
        name = str(TOPICS[topic]['name'])
    listing = '\n\n'.join('<paper id="%s">\nTitle: %s\nAbstract: %s\n</paper>' % (p['arxiv_id'], p['title'], p['abstract'])
                          for p in chosen)
    answer = generated.chat_json(
        'Write, in %s, the weekly letter "%s this week" for curious, educated readers who are not specialists, from '
        'the new arXiv papers above (%d papers were announced in this field this week). Write it like a good science '
        'magazine article: a title, a one-sentence subtitle, then one short section per paper or group of related '
        'papers, each with a heading and 1 or 2 paragraphs saying what was found and why it matters, and a closing '
        'paragraph on what links them. Stay faithful to the abstracts: say "preprint" or "not yet peer reviewed" '
        'where useful, no hype, no LaTeX, no markdown, never write IDs in the texts. Answer with a JSON object '
        '{"title": "...", "dek": "...", "sections": [{"heading": "...", "text": "...", "papers": ["ID"]}], '
        '"closing": "..."}.' % (generated.language_name(lang), name, len(papers)),
        paid=True, max_tokens=5000, system=listing)
    titles = {p['arxiv_id']: p['title'] for p in chosen}
    sections = [{'heading': str(s.get('heading', '')).strip(), 'text': str(s.get('text', '')).strip(),
                 'papers': [a for a in s.get('papers', []) if a in titles]}
                for s in answer.get('sections', []) if isinstance(s, dict) and s.get('text')]
    if not sections or not answer.get('title'):
        return None
    letter, _ = Letter.objects.update_or_create(topic=topic, lang=lang, day=day, defaults={
        'title': str(answer['title']).strip()[:300], 'dek': str(answer.get('dek', '')).strip(),
        'content': json.dumps({'sections': sections, 'closing': str(answer.get('closing', '')).strip(),
                               'papers': titles, 'count': len(papers)})})
    return letter


def update(now=None):
    """From the hourly digest run: write the letters of the week, a few per run."""
    if not cache.add(LOCK, True, 3600):
        return 0
    try:
        day = ArxivBatch.objects.order_by('-day').values_list('day', flat=True).first()
        if day is None:
            return 0
        last = Letter.objects.order_by('-day').values_list('day', flat=True).first()
        target = day if last is None or day - last >= WEEK else last
        done = set(Letter.objects.filter(day=target).values_list('topic', 'lang'))
        tried = cache.get('letters-tried:%s' % target) or set()
        written = 0
        for topic in TOPICS:
            for lang in LANGS:
                if (topic, lang) in done or (topic, lang) in tried or written >= LETTERS_PER_RUN:
                    continue
                try:
                    if write(topic, target, lang):
                        written += 1
                    else:
                        tried.add((topic, lang))  # too few papers this week: not tried again
                except Exception:
                    logger.exception('Letter %s %s failed', topic, lang)
        cache.set('letters-tried:%s' % target, tried, 8 * 86400)
        return written
    finally:
        cache.delete(LOCK)


def send(now, budget, base_url=digest.BASE_URL):
    """Email the new letters to Premium subscribers, within the email budget."""
    from django.core import signing
    suppressed = mailcheck.suppressed()
    sent = 0
    for subscription in LetterSubscription.objects.filter(active=True).select_related('user'):
        if sent >= budget:
            break
        letter = Letter.objects.filter(topic=subscription.topic, lang=subscription.lang).first()
        if letter is None or (subscription.last_letter and letter.day <= subscription.last_letter):
            continue
        user = subscription.user
        if not user.email or user.email.strip().lower() in suppressed or not access.is_paid(user):
            continue
        token = signing.dumps(subscription.pk, salt=UNSUBSCRIBE_SALT)
        context = dict(letter_context(letter), base_url=base_url,
                       url='%s/%s/letters/%s/%s/' % (base_url, letter.lang, letter.topic, letter.day.isoformat()),
                       unsubscribe_url='%s/%s/letters/unsubscribe/%s/' % (base_url, letter.lang, token))
        with translation.override(letter.lang):
            text = render_to_string('letters/email.txt', context)
            html = render_to_string('letters/email.html', context)
        message = EmailMultiAlternatives(letter.title, text, settings.DEFAULT_FROM_EMAIL, [user.email],
                                         headers={'List-Unsubscribe': '<%s>' % context['unsubscribe_url'],
                                                  'List-Unsubscribe-Post': 'List-Unsubscribe=One-Click'})
        message.attach_alternative(html, 'text/html')
        try:
            message.send()
        except Exception:
            logger.exception('Letter email to %s failed', user.username)
            continue
        subscription.last_letter = letter.day
        subscription.save(update_fields=['last_letter'])
        sent += 1
    return sent


# --- views -------------------------------------------------------------------

def letter_context(letter):
    from .presskit import paper_id
    content = json.loads(letter.content)
    titles = content.get('papers', {})
    for section in content['sections']:
        section['links'] = [{'arxiv_id': a, 'title': titles.get(a, a), 'presskit_id': paper_id(a)}
                            for a in section['papers']]
    with translation.override(letter.lang):
        name = str(TOPICS[letter.topic]['name'])
    return {'letter': letter, 'sections': content['sections'], 'closing': content.get('closing', ''),
            'count': content.get('count'), 'topic_name': name}


def letters_page(request):
    lang = get_language()
    paid = access.is_paid(request.user)
    subscribed = set()
    if request.user.is_authenticated:
        subscribed = set(LetterSubscription.objects.filter(user=request.user, active=True).values_list('topic', flat=True))
        if request.method == 'POST' and paid:
            wanted = {t for t in request.POST.getlist('topics') if t in TOPICS}
            for topic in TOPICS:
                LetterSubscription.objects.update_or_create(user=request.user, topic=topic, defaults={
                    'active': topic in wanted, 'lang': lang})
            return redirect('/%s/letters/?saved=1' % lang)
    latest = {}
    for letter in Letter.objects.filter(lang=lang).order_by('-day')[:60]:
        latest.setdefault(letter.topic, letter)
    topics = [{'slug': slug, 'name': spec['name'], 'letter': latest.get(slug), 'subscribed': slug in subscribed}
              for slug, spec in TOPICS.items()]
    return render(request, 'summarizer/letters.html', {'topics': topics, 'paid': paid,
                                                       'saved': request.GET.get('saved') == '1'})


def topic_page(request, topic):
    if topic not in TOPICS:
        raise Http404
    letters = Letter.objects.filter(topic=topic, lang=get_language()).order_by('-day')[:52]
    return render(request, 'summarizer/letter_topic.html', {'topic': topic, 'name': TOPICS[topic]['name'], 'letters': letters})


def letter_page(request, topic, day):
    try:
        day = datetime.date.fromisoformat(day)
    except ValueError:
        raise Http404
    letter = Letter.objects.filter(topic=topic, day=day, lang=get_language()).first()
    if letter is None or topic not in TOPICS:
        raise Http404
    other = Letter.objects.filter(topic=topic, day=day).exclude(lang=letter.lang).values_list('lang', flat=True).first()
    return render(request, 'summarizer/letter.html', dict(letter_context(letter), other_lang=other,
                                                         paid=access.is_paid(request.user)))


@csrf_exempt  # mail clients send the one-click unsubscribe POST without a CSRF token
def unsubscribe(request, token):
    from django.core import signing
    try:
        pk = signing.loads(token, salt=UNSUBSCRIBE_SALT)
    except signing.BadSignature:
        raise Http404
    subscription = LetterSubscription.objects.filter(pk=pk).first()
    done = False
    if subscription is not None and request.method == 'POST':
        subscription.active = False
        subscription.save(update_fields=['active'])
        done = True
    return render(request, 'summarizer/letter_unsubscribe.html', {'subscription': subscription, 'done': done,
                                                                  'name': TOPICS.get(getattr(subscription, 'topic', ''), {}).get('name', '')})
