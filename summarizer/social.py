"""Sharing a paper, the rising topics of the week or a weekly letter on LinkedIn, X or Bluesky.

A shared link works through two things. The picture a network shows when the link is pasted (the page's og:image):
drawn here as a 1200x630 PNG with the title and what the paper found, rather than the logo. And a post ready to
publish, built from what the page already holds, so a click costs no AI call. Authors are the first audience:
researchers post their own papers, and the paper page tells them a post is ready.

The pictures are drawn on request and cached by Cloudflare and the networks (Cache-Control), not in Redis, whose
30 MB are shared by the cache and the websockets. Each link carries utm_source=<network>, which Umami keeps.
"""
import datetime
import functools
import io
import logging
import os
import re
import threading
from urllib.parse import quote, urlencode

from django.conf import settings
from django.http import Http404, HttpResponse
from django.utils import formats, translation
from django.utils.translation import gettext as _
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

SITE = 'https://www.summarizepaper.com'
LANGS = ('en', 'fr')
NETWORKS = ('linkedin', 'x', 'bluesky', 'email', 'copy', 'native')
NEW_PAPER_DAYS = 90  # an author's post says "our new paper" within this many days of the first version
SHORT = 200  # characters of the post for X (280, the link counted as 23) and Bluesky (300, the whole link counted)
CACHE = 'public, max-age=86400, s-maxage=604800'
CACHE_SOON = 'public, max-age=60, s-maxage=60'  # drawn without the figure being fetched: asked again soon
RENDER_SLOTS = threading.BoundedSemaphore(2)
FIGURE_LICENSES = {'by', 'by-sa', 'zero', 'mark'}
FIGURE_KEY = 'og-figure:v1:%s'
FIGURE_WIDTH = 440
MAX_FIGURE_BYTES = 8_000_000

WIDTH, HEIGHT = 1200, 630
MARGIN = 72
FOOTER = HEIGHT - 100  # the line above the logo
TEXT, MUTED, ACCENT, BORDER = (22, 22, 29), (91, 91, 107), (79, 70, 229), (230, 230, 239)
GRADIENT = ((249, 115, 22), (236, 72, 153), (99, 102, 241), (16, 185, 129))  # --brand-gradient of site.css
FONT_DIR = os.path.join(settings.BASE_DIR, 'static', 'summarizer', 'fonts')
FALLBACK = os.path.join(settings.BASE_DIR, 'font', 'DejaVuSansCondensed.ttf')  # math signs, scripts Inter lacks
FALLBACK_BOLD = os.path.join(settings.BASE_DIR, 'font', 'DejaVuSansCondensed-Bold.ttf')
LOGO = os.path.join(settings.BASE_DIR, 'static', 'summarizer', 'images', 'logo.png')

# English hashtags: the ones people follow on LinkedIn and X, whatever the language of the post
CATEGORY_TAGS = {
    'cs.LG': ('#MachineLearning', '#AI'), 'stat.ML': ('#MachineLearning', '#AI'), 'cs.AI': ('#AI',),
    'cs.CL': ('#NLP', '#AI'), 'cs.CV': ('#ComputerVision', '#AI'), 'cs.RO': ('#Robotics',), 'cs.CR': ('#Cybersecurity',),
    'astro-ph.EP': ('#PlanetaryScience', '#Astronomy'), 'astro-ph.CO': ('#Cosmology', '#Astrophysics'),
    'quant-ph': ('#QuantumPhysics', '#QuantumComputing'), 'gr-qc': ('#GeneralRelativity', '#Physics'),
    'q-bio.NC': ('#Neuroscience',), 'biorxiv.neuroscience': ('#Neuroscience', '#bioRxiv'),
}
ARCHIVE_TAGS = {
    'astro-ph': ('#Astrophysics', '#Astronomy'), 'cs': ('#ComputerScience',), 'cond-mat': ('#CondensedMatter', '#Physics'),
    'physics': ('#Physics',), 'math': ('#Mathematics',), 'math-ph': ('#MathematicalPhysics',), 'stat': ('#Statistics',),
    'hep-ph': ('#ParticlePhysics', '#Physics'), 'hep-th': ('#TheoreticalPhysics', '#Physics'),
    'hep-ex': ('#ParticlePhysics', '#Physics'), 'hep-lat': ('#ParticlePhysics', '#Physics'),
    'nucl-th': ('#NuclearPhysics', '#Physics'), 'nucl-ex': ('#NuclearPhysics', '#Physics'), 'nlin': ('#Physics',),
    'q-bio': ('#Biology',), 'q-fin': ('#Finance',), 'econ': ('#Economics',), 'eess': ('#Engineering',),
    'biorxiv': ('#Biology', '#bioRxiv'), 'medrxiv': ('#Medicine', '#medRxiv'),
}
LETTER_TAGS = {
    'ai': ('#AI', '#MachineLearning'), 'space': ('#Astronomy', '#Astrophysics'), 'quantum': ('#QuantumPhysics', '#QuantumComputing'),
    'matter': ('#Physics', '#Materials'), 'particles': ('#ParticlePhysics', '#Physics'), 'brain': ('#Neuroscience',),
    'life': ('#Biology', '#Medicine'), 'climate': ('#Climate', '#EarthScience'), 'math': ('#Mathematics', '#Statistics'),
    'computing': ('#ComputerScience',), 'economy': ('#Economics', '#Finance'),
}
SERVER_SLUGS = {'arxiv': '*', 'biorxiv': 'biorxiv', 'medrxiv': 'medrxiv'}  # '*' is no path segment


# --- the post ----------------------------------------------------------------------------------------------------

def plain(text):
    """One line of plain text from stored summaries: no tags, Markdown stars, LaTeX dollars or runs of spaces."""
    from . import sources
    from .templatetags.seo import plaintext
    text = re.sub(r'\s+', ' ', plaintext(sources.plain_math(str(text or ''))).replace('**', '')).strip()
    return re.sub(r'^[\s,;:.·•*\-–—]+', '', text)  # older summaries can start with ", , , ," left by stripped tags


def shorten(text, limit):
    """At most `limit` characters, cut after a word with an ellipsis."""
    text = plain(text)
    if len(text) <= limit:
        return text
    cut = text[:limit - 1].rsplit(' ', 1)[0].rstrip(' ,;:.-–—')
    return cut + '…'


def first_sentence(text, limit=280):
    text = plain(text)
    match = re.match(r'(.{20,}?[.!?])\s+[A-ZÀ-Ý]', text)
    return shorten(match.group(1) if match else text, limit)


def hashtags(*tags):
    return ' '.join(dict.fromkeys(t for t in tags if t))


def category_tags(category, arxiv=True):
    category = category or ''
    tags = CATEGORY_TAGS.get(category) or ARCHIVE_TAGS.get(category.split('.')[0]) or ()
    return hashtags(*tags[:2], '#arXiv' if arxiv else '#Research')


def tagged(url, network):
    return '%s%sutm_source=%s' % (url, '&' if '?' in url else '?', network)


def share(url, text, short, tool, subject, key='share', image=''):
    """What share_post.html needs: the post, a shorter one for X and Bluesky (whose limits count the link), the subject
    of an email, and the page link with its utm_source for each network. The post shows the bare link; the script
    swaps in the tagged one. `key` tells apart several boxes of one page; `image` is the picture to draw in advance."""
    links = {n: tagged(url, n) for n in NETWORKS}
    hrefs = {  # without the script; with it, the edited post goes instead
        'linkedin': 'https://www.linkedin.com/feed/?' + urlencode({'shareActive': 'true', 'text': text.replace(url, links['linkedin'])}),
        'x': 'https://x.com/intent/post?' + urlencode({'text': short, 'url': links['x']}),
        'bluesky': 'https://bsky.app/intent/compose?' + urlencode({'text': '%s\n%s' % (short, links['bluesky'])}),
        'email': 'mailto:?' + urlencode({'subject': subject, 'body': text.replace(url, links['email'])}, quote_via=quote),
    }
    return {'url': url, 'text': text, 'short': short, 'tool': tool, 'subject': subject, 'links': links, 'hrefs': hrefs,
            'key': key, 'image': image}


def is_author(user, paper):
    """Whether a signed-in reader signs the paper: the name of their account (Google gives it) among its authors."""
    from .digest import name_key, same_author
    if not getattr(user, 'is_authenticated', False):
        return False
    key = name_key('%s %s' % (user.first_name, user.last_name))
    return bool(key) and any(same_author(key, a.name) for a in paper.authors.all())


def paper_share(paper, sumpaper, insight, lang, author=False, image=''):
    """The post for a paper page: its title, what it shows in one sentence, three findings, the link. An author's
    post speaks in the first person: "Our new paper: …"."""
    from . import keypoints, sources
    url = '%s/%s/paper/%s/' % (SITE, lang, paper.arxiv_id)
    title = plain(paper.title)
    if insight:
        hook, points = insight['tldr'], [f['text'] for f in insight.get('findings', [])[:3]]
    else:
        hook = first_sentence(sumpaper.summary or sumpaper.lay_summary or paper.abstract)
        points = [shorten(p, 240) for p in keypoints.stored_points(sumpaper.notes)[:3]]
    tags = category_tags(paper.category, arxiv=not sources.is_external(paper.arxiv_id))
    if author:
        published = paper.published_arxiv
        recent = published is None or (datetime.date.today() - published).days <= NEW_PAPER_DAYS
        lead = (_('Our new paper: %(title)s') if recent else _('Our paper: %(title)s')) % {'title': title}
        lines = [lead, '', plain(hook)]
        if points:
            lines += ['', _('What we found:')] + ['• ' + plain(p) for p in points]
        lines += ['', '%s %s' % (_('A summary for every reader, and the paper answers your questions:'), url), '', tags]
        short = shorten('%s %s' % ((_('Our new paper:') if recent else _('Our paper:')), plain(hook)), SHORT - len(tags) - 2)
    else:
        lines = [title, '', plain(hook)]
        if points:
            lines += ['', _('Key findings:')] + ['• ' + plain(p) for p in points]
        lines += ['', '%s %s' % (_('AI summary, key points and questions to the paper:'), url), '', tags]
        short = shorten(hook, SHORT - len(tags) - 2)
    return share(url, '\n'.join(lines), short + '\n\n' + tags, 'paper', title, image=image)


def trends_url(server, day, lang, topic=None):
    slug = next(s for s, scope in SERVER_SLUGS.items() if scope == server)
    return '%s/%s/trends/?%sweek=%s%s' % (SITE, lang, '' if server == '*' else 'scope=%s&' % slug, day.isoformat(),
                                         '&topic=%d' % topic if topic else '')


def week_label(day):
    return '%s – %s' % (formats.date_format(day, 'j F'), formats.date_format(day + datetime.timedelta(days=6), 'j F Y'))


def topic_figures(topic):
    """'×4.5, 12 papers from 12 teams', with what the topic has."""
    figures = []
    if topic.get('growth'):
        figures.append('×%s' % formats.localize(topic['growth']))
    if topic.get('count') and topic.get('teams'):
        figures.append(_('%(n)s papers from %(t)s teams') % {'n': topic['count'], 't': topic['teams']})
    return ', '.join(figures)


def trends_share(server, server_name, day, topics, lang):
    """The post for the free topics of a server's week: each with its rise and its number of papers and teams."""
    url = trends_url(server, day, lang)
    lines = [_('Rising research topics this week on %(server)s (papers of %(week)s):') % {'server': server_name, 'week': week_label(day)}, '']
    for i, topic in enumerate(topics, 1):
        figures = topic_figures(topic)
        lines.append('%d. %s' % (i, plain(topic['title'])) + (' — ' + figures if figures else ''))
    tags = hashtags('#' + server_name, '#Research', '#Science')
    lines += ['', '%s %s' % (_('Counted in the title and abstract of every new paper, explained in plain language:'), url), '', tags]
    lead = _('Rising this week on %(server)s:') % {'server': server_name}
    short = shorten('%s %s' % (lead, ' · '.join(plain(t['title']) for t in topics)), SHORT - len(tags) - 2) + '\n\n' + tags
    return share(url, '\n'.join(lines), short, 'trends', lead.rstrip(' :'))


def topic_share(server, server_name, day, topic, index, lang):
    """The post for one free topic of the week: its name, its rise, what it is about, the link to its card."""
    url = trends_url(server, day, lang, index)
    figures = topic_figures(topic)
    lead = _('Rising this week on %(server)s: %(title)s') % {'server': server_name, 'title': plain(topic['title'])}
    lines = [lead, '']
    if figures:
        lines += ['%s (%s)' % (figures, _('papers of %(week)s') % {'week': week_label(day)}), '']
    if topic.get('text'):
        lines += [shorten(topic['text'], 600), '']
    tags = hashtags('#' + server_name, '#Research', '#Science')
    lines += ['%s %s' % (_('Example papers, the history of the term and who works on it:'), url), '', tags]
    short = shorten('%s%s' % (lead, ' — ' + figures if figures else ''), SHORT - len(tags) - 2) + '\n\n' + tags
    slug = next(s for s, scope in SERVER_SLUGS.items() if scope == server)
    return share(url, '\n'.join(lines), short, 'trends', plain(topic['title']), key='share-topic-%d' % index,
                 image=image_url('topic', lang, slug, day.isoformat(), str(index)))


def letter_share(letter, sections, lang):
    """The post for a weekly letter: its title, its standfirst and the headings of its sections."""
    url = '%s/%s/letters/%s/%s/' % (SITE, lang, letter.topic, letter.day.isoformat())
    tags = hashtags(*LETTER_TAGS.get(letter.topic, ()), '#Research')
    lines = [plain(letter.title), '', plain(letter.dek)]
    headings = [plain(s['heading']) for s in sections[:4] if s.get('heading')]
    if headings:
        lines += ['', _('In this letter:')] + ['• ' + h for h in headings]
    lines += ['', '%s %s' % (_('Each finding links to its paper:'), url), '', tags]
    short = shorten('%s — %s' % (plain(letter.title), plain(letter.dek)), SHORT - len(tags) - 2) + '\n\n' + tags
    return share(url, '\n'.join(lines), short, 'letters', plain(letter.title))


# --- the picture -------------------------------------------------------------------------------------------------

@functools.lru_cache(None)
def inter_faces():
    """[(path, [(first, last)])] of the Inter subsets, read from the unicode-range of inter.css, Latin first."""
    with open(os.path.join(FONT_DIR, 'inter.css')) as f:
        css = f.read()
    faces = []
    for block in re.findall(r'@font-face\s*{(.*?)}', css, re.S):
        src, ranges = re.search(r"url\('([^']+)'\)", block), re.search(r'unicode-range:\s*([^;]+);', block)
        if src and ranges:
            spans = []
            for part in ranges.group(1).split(','):
                first, _sep, last = part.strip()[2:].partition('-')
                spans.append((int(first, 16), int(last or first, 16)))
            faces.append((os.path.join(FONT_DIR, src.group(1)), spans))
    return faces[::-1]


@functools.lru_cache(maxsize=96)
def font(path, size, weight):
    try:
        face = ImageFont.truetype(path, size)
        if path.endswith('.woff2'):
            face.set_variation_by_axes([weight])
        return face
    except OSError:  # a FreeType without WOFF2 support: DejaVu for everything
        return None


def runs(text, size, weight):
    """[[font, piece]]: each character drawn with the first Inter subset that has it, else with DejaVu."""
    found = []
    for char in text:
        code = ord(char)
        path = next((p for p, spans in inter_faces() if any(a <= code <= b for a, b in spans)), None)
        face = font(path, size, weight) if path else None
        face = face or font(FALLBACK_BOLD if weight >= 600 else FALLBACK, size, weight)
        if found and found[-1][0] is face:
            found[-1][1] += char
        else:
            found.append([face, char])
    return found


def measure(text, size, weight):
    return sum(face.getlength(piece) for face, piece in runs(text, size, weight))


def write(draw, x, baseline, text, size, weight, fill):
    for face, piece in runs(text, size, weight):
        draw.text((x, baseline), piece, font=face, fill=fill, anchor='ls')
        x += face.getlength(piece)
    return x


def wrap(text, size, weight, width, lines_max):
    """(lines, complete): the text in lines of at most `width` pixels; the last one ends with an ellipsis when cut."""
    lines, line = [], ''
    for word in plain(text).split():
        while measure(word, size, weight) > width and len(word) > 1:  # a long formula or URL: cut it
            word = word[:-2] + '…'
        trial = (line + ' ' + word).strip()
        if line and measure(trial, size, weight) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    if line:
        lines.append(line)
    if len(lines) <= lines_max:
        return lines, True
    lines = lines[:lines_max]
    last = lines[-1]
    while ' ' in last and measure(last + '…', size, weight) > width:
        last = last.rsplit(' ', 1)[0]
    lines[-1] = last.rstrip(' ,;:.-–—') + '…'
    return lines, False


def fit(text, sizes, weight, width, lines_max):
    """(size, lines): the largest size at which the whole text fits, else the smallest one, cut."""
    for size in sizes:
        lines, complete = wrap(text, size, weight, width, lines_max)
        if complete:
            return size, lines
    return sizes[-1], lines


def gradient_at(t):
    span = t * (len(GRADIENT) - 1)
    i = min(int(span), len(GRADIENT) - 2)
    a, b, f = GRADIENT[i], GRADIENT[i + 1], span - i
    return tuple(round(a[k] + (b[k] - a[k]) * f) for k in range(3))


@functools.lru_cache(None)
def logo(height):
    image = Image.open(LOGO).convert('RGBA')
    return image.resize((round(image.width * height / image.height), height), Image.LANCZOS)


def card(eyebrow, title, body='', items=(), note='', figure=None, credit=''):
    """A 1200x630 PNG: the brand band, a small line above the title, the title, then a paragraph or a short list of
    (figure, text) rows, and the logo with the address at the bottom. With `figure` (a PIL image of the paper), the
    text takes the left and the figure the right, its credit below it."""
    image = Image.new('RGB', (WIDTH, HEIGHT), 'white')
    draw = ImageDraw.Draw(image)
    for x in range(WIDTH):
        draw.line([(x, 0), (x, 11)], fill=gradient_at(x / (WIDTH - 1)))
    inner = WIDTH - 2 * MARGIN
    titles, body_size = (60, 54, 48, 42), 32
    if figure is not None:
        box = (WIDTH - MARGIN - FIGURE_WIDTH, 44, WIDTH - MARGIN, FOOTER - (52 if credit else 24))
        shown = figure.copy()
        shown.thumbnail((box[2] - box[0], box[3] - box[1]), Image.LANCZOS)
        image.paste(shown, (box[0] + (box[2] - box[0] - shown.width) // 2, box[1] + (box[3] - box[1] - shown.height) // 2))
        if credit:
            line = wrap(credit, 20, 400, FIGURE_WIDTH, 1)[0][0]
            write(draw, box[2] - measure(line, 20, 400), FOOTER - 20, line, 20, 400, MUTED)
        inner -= FIGURE_WIDTH + 40
        titles, body_size = (50, 44, 38, 34), 28
    y = 96
    write(draw, MARGIN, y, wrap(eyebrow, 26, 600, inner, 1)[0][0] if eyebrow else '', 26, 600, ACCENT)
    y += 26
    size, lines = fit(title, titles, 800, inner, 2 if items else (3 if body else 5) + (1 if figure is not None else 0))
    step = round(size * 1.2)
    for line in lines:
        y += step
        write(draw, MARGIN, y, line, size, 800, TEXT)
    y += 26
    if items:
        figure_width = max(measure(f, 32, 700) for f, _t in items) + 22
        for figure, text in items:
            if y + 56 > FOOTER - 12:
                break
            y += 56
            write(draw, MARGIN, y, figure, 32, 700, ACCENT)
            write(draw, MARGIN + figure_width, y, wrap(text, 32, 500, inner - figure_width, 1)[0][0], 32, 500, TEXT)
    elif body:
        leading = round(body_size * 1.38)
        room = int((FOOTER - (64 if note else 24) - y) // leading)
        if room > 0:
            for line in wrap(body, body_size, 400, inner, room)[0]:
                y += leading
                write(draw, MARGIN, y, line, body_size, 400, MUTED)
    if note:
        write(draw, MARGIN, FOOTER - 28, wrap(note, 24, 400, inner, 1)[0][0], 24, 400, MUTED)
    draw.line([(MARGIN, FOOTER), (WIDTH - MARGIN, FOOTER)], fill=BORDER, width=2)
    mark = logo(52)
    image.paste(mark, (MARGIN, FOOTER + 24), mark)
    write(draw, MARGIN + mark.width + 16, FOOTER + 62, 'SummarizePaper', 30, 700, TEXT)
    address = 'summarizepaper.com'
    write(draw, WIDTH - MARGIN - measure(address, 26, 500), FOOTER + 61, address, 26, 500, MUTED)
    out = io.BytesIO()
    image.save(out, 'PNG', optimize=True)
    return out.getvalue()


def png(content, cache=CACHE):
    response = HttpResponse(content, content_type='image/png')
    response['Cache-Control'] = cache
    return response


def rendered(draw_card):
    """The PNG drawn by `draw_card`, at most RENDERS at once: a burst of new addresses gets a short 503 rather than
    tying up the process (each picture takes about 0.1 s of CPU)."""
    if not RENDER_SLOTS.acquire(timeout=5):
        response = HttpResponse(status=503)
        response['Retry-After'] = '10'
        return response
    try:
        return draw_card()
    finally:
        RENDER_SLOTS.release()


# --- a figure of the paper in its picture ---------------------------------------------------------------------------

def figure_license(paper):
    """Whether a figure may illustrate the paper's picture: CC BY, BY-SA, CC0 or public domain (no NC: the picture
    advertises the site; no ND: the figure is resized)."""
    from . import sources
    code, _version = sources.license_code(paper.license)
    return code in FIGURE_LICENSES and 'creativecommons.org' in (paper.license or '').lower()


def figure_of(paper):
    """(PIL image, credit) of the paper's first figure, None when it has none we may show, or 'pending' while it is
    being fetched (in a thread: the gallery reads the paper's HTML, which can take seconds)."""
    from django.core.cache import cache
    if not figure_license(paper):
        return None
    found = cache.get(FIGURE_KEY % paper.arxiv_id)
    if found is None:
        if cache.add(FIGURE_KEY % paper.arxiv_id + ':lock', 1, 120):
            threading.Thread(target=fetch_figure, args=(paper.arxiv_id,), daemon=True).start()
        return 'pending'
    if not found:
        return None
    data, credit = found
    return Image.open(io.BytesIO(data)), credit


def fetch_figure(arxiv_id):
    """Store the first figure of the paper, shrunk to the picture's size, for a week (or that it has none, for a day)."""
    import requests
    from django.core.cache import cache
    from django.db import connection
    from . import chat_figures, sources
    from .models import ArxivPaper
    try:
        paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
        connection.close()  # Postgres allows 20: none held during the downloads below
        gallery = chat_figures.paper_gallery(paper) if paper else {'figures': []}
        connection.close()  # the gallery of a journal paper reads the database too
        for figure in gallery.get('figures', [])[:4]:
            url = figure['images'][0]
            if not chat_figures.safe_image(url):
                continue
            try:
                response = requests.get(url, timeout=(4, 10), headers={'User-Agent': sources.USER_AGENT})
                response.raise_for_status()
                if len(response.content) > MAX_FIGURE_BYTES:
                    continue
                picture = Image.open(io.BytesIO(response.content))
                picture.load()
            except (requests.RequestException, OSError, Image.DecompressionBombError):
                continue  # an SVG or a broken image: the next figure
            if picture.width < 200 or picture.height < 120:
                continue
            flat = Image.new('RGB', picture.size, 'white')
            picture = picture.convert('RGBA')
            flat.paste(picture, mask=picture.getchannel('A'))
            flat.thumbnail((FIGURE_WIDTH * 2, (FOOTER - 96) * 2), Image.LANCZOS)
            out = io.BytesIO()
            flat.save(out, 'JPEG', quality=82, optimize=True)
            credit = '%s · %s' % ((figure.get('label') or 'Figure').rstrip(' :.'), sources.license_label(paper.license))
            cache.set(FIGURE_KEY % arxiv_id, (out.getvalue(), credit), 7 * 86400)
            return
        cache.set(FIGURE_KEY % arxiv_id, b'', 86400)
    except Exception:
        logger.exception('Figure of %s for its picture failed', arxiv_id)
        cache.set(FIGURE_KEY % arxiv_id, b'', 3600)
    finally:
        connection.close()


def _lang(lang):
    if lang not in LANGS:
        raise Http404
    return lang


# --- the picture of each page (og:image) -------------------------------------------------------------------------

def paper_image(request, lang, arxiv_id):
    """A paper's picture, with its first figure when its license allows; only for papers summarized here."""
    from . import insight, sources
    from .models import ArxivPaper, SummaryPaper
    from .templatetags.dash_slash import category_label
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    _lang(lang)
    if paper is None or not SummaryPaper.objects.filter(paper=paper).exists():
        raise Http404
    with translation.override(lang):
        sumpaper = SummaryPaper.objects.filter(paper=paper, lang=lang).first()
        reading = insight.for_page(arxiv_id, lang) if sumpaper else None
        body = reading['tldr'] if reading else first_sentence(
            (sumpaper and (sumpaper.summary or sumpaper.lay_summary)) or paper.abstract, 400)
        found = figure_of(paper)
        figure, credit = found if isinstance(found, tuple) else (None, '')
        eyebrow = ' · '.join(filter(None, [_('AI summary').upper(), sources.label(arxiv_id),
                                           category_label(paper.category) if paper.category and figure is None else '']))
        return rendered(lambda: png(card(eyebrow, plain(paper.title) or sources.label(arxiv_id), body, figure=figure, credit=credit),
                                    CACHE_SOON if found == 'pending' else CACHE))


def _week_topics(server, day, lang):
    from . import trends
    if server not in SERVER_SLUGS:
        raise Http404
    try:
        week = datetime.date.fromisoformat(day)
    except ValueError:
        raise Http404
    found, topics = trends.topics_of(SERVER_SLUGS[server], _lang(lang), week=week)
    if found is None:
        raise Http404
    return week, topics[:trends.FREE_TOPICS]


def trends_image(request, lang, server, day):
    from .trends_web import SERVERS
    week, topics = _week_topics(server, day, lang)
    with translation.override(lang):
        eyebrow = '%s · %s' % (_('Rising topics in research').upper(), week_label(week))
        title = _('Rising this week on %(server)s') % {'server': dict(SERVERS)[SERVER_SLUGS[server]]}
        items = [('×%s' % formats.localize(t['growth']) if t.get('growth') else '•', plain(t['title'])) for t in topics]
        note = _('×: papers using the topic this week, compared with the eight weeks before.')
        return rendered(lambda: png(card(eyebrow, title, items=items, note=note)))


def topic_image(request, lang, server, day, index):
    """One free topic of the week: its rise and number of teams above its name, what it is about below."""
    from .trends_web import SERVERS
    week, topics = _week_topics(server, day, lang)
    if not 1 <= index <= len(topics):
        raise Http404
    topic = topics[index - 1]
    with translation.override(lang):
        name = dict(SERVERS)[SERVER_SLUGS[server]]
        eyebrow = ' · '.join(filter(None, [(_('Rising this week on %(server)s') % {'server': name}).upper(), topic_figures(topic)]))
        note = '%s · %s' % (_('Rising topics in research'), week_label(week))
        return rendered(lambda: png(card(eyebrow, plain(topic['title']), plain(topic.get('text', '')), note=note)))


def letter_image(request, lang, topic, day):
    from . import letters
    with translation.override(_lang(lang)):
        letter = letters._letter(topic, day, lang)
        eyebrow = '%s · %s' % (letters.localized_name(topic, lang)[1].upper(), formats.date_format(letter.day, 'j F Y'))
        return rendered(lambda: png(card(eyebrow, plain(letter.title), plain(letter.dek))))


def image_url(kind, lang, *parts, version=None):
    """The address of a page's picture; `version` changes when its text does, so that caches take the new one."""
    url = '%s/og/%s/%s/%s.png' % (SITE, lang, kind, '/'.join(parts))
    return '%s?v=%s' % (url, version) if version else url


def paper_image_url(paper, sumpaper, lang):
    version = int(sumpaper.updated.timestamp()) if sumpaper and sumpaper.updated else None
    return image_url('p', lang, paper.arxiv_id, version=version)

