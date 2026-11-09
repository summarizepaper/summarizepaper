"""The journal club deck as a LaTeX Beamer project (a zip: main.tex and the figures), to open in Overleaf.

The same validated pages as the PowerPoint export (slides.shown_pages), with the same figures (slides_pptx.figure_bytes),
the speaker notes and the passages of the paper in \\note. XeLaTeX with DejaVu Sans Condensed, which TeX Live and
Overleaf ship, sets Greek, Cyrillic and the symbols of scientific text (μ, ±, ≈, →) without math mode; Chinese and Japanese
use Fandol (ctex) and Harano Aji (xeCJK). Arabic, Hindi and Korean would need other fonts and shaping: those decks have no
Beamer export (LANGUAGES).
"""
import io
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

from django.utils.translation import gettext as _

from . import overleaf, slides_pptx, sources
from .slides_pptx import EXPORT_WORKERS, ExportError, notes_text

LANGUAGES = ('en', 'fr', 'es', 'de', 'it', 'pt', 'nl', 'pl', 'tr', 'ru', 'uk', 'zh-hans', 'ja')
SIZES = {36: '\\normalsize', 30: '\\small', 24: '\\footnotesize'}
# The fonts of LaTeX are not those of the page, where the PowerPoint export checks that the text fits: a slide longer
# than the page is scaled down to fit, and LaTeX names it in a warning of the log (Overleaf's Logs and outputs) when it
# loses more than a tenth of its size.
FIT = '\\slidebody{\n%s\n}'
FIT_MACRO = r"""\newsavebox{\spbody}
\newcommand{\slidebody}[1]{%
  \sbox{\spbody}{\begin{minipage}[t]{\textwidth}#1\end{minipage}}%
  \ifdim\dimexpr\ht\spbody+\dp\spbody\relax>0.89\textheight
    \PackageWarningNoLine{SummarizePaper}{Slide \insertframenumber\space is scaled down by more than a tenth to fit the page: shorten it for a larger text}%
  \fi
  \adjustbox{max totalheight=0.8\textheight}{\usebox{\spbody}}}
"""
LARGE_FIGURE = 200 * 1024  # bytes: a larger PNG goes as a JPEG, so that the project stays light for Overleaf

PREAMBLE = r"""% {comment}
\documentclass[aspectratio=169,11pt]{{beamer}}
\usepackage{{fontspec}}
{fonts}
\usepackage{{graphicx}}
\usepackage{{adjustbox}}
\usetheme{{default}}
\setbeamertemplate{{navigation symbols}}{{}}
\setbeamercolor{{frametitle}}{{fg=black!85}}
\setbeamercolor{{title}}{{fg=black!85}}
\setbeamercolor{{itemize item}}{{fg=black!60}}
\setbeamertemplate{{itemize item}}{{\textbullet}}
\setbeamertemplate{{footline}}{{\begin{{beamercolorbox}}[wd=\paperwidth,ht=2.6ex,dp=1.4ex,leftskip=1em,rightskip=1em]{{footline}}\tiny\color{{black!55}}SummarizePaper · {footer}\hfill\insertframenumber\,/\,\inserttotalframenumber\end{{beamercolorbox}}}}
% \setbeameroption{{show notes on second screen=right}} % {notes}
"""
SANS = r"""\setsansfont{DejaVuSansCondensed}[Extension=.ttf, BoldFont=DejaVuSansCondensed-Bold, ItalicFont=DejaVuSansCondensed-Oblique, BoldItalicFont=DejaVuSansCondensed-BoldOblique, Scale=0.95]"""
CJK_FONTS = {
    'zh-hans': r"""\usepackage[fontset=fandol]{ctex}
\renewcommand{\familydefault}{\sfdefault}
""" + SANS,
    # Fandol lacks Japanese kanji forms (強, 証, ー): Harano Aji, TeX Live's Japanese fonts
    'ja': r"""\usepackage{xeCJK}
\setCJKmainfont{HaranoAjiGothic-Regular.otf}[BoldFont=HaranoAjiGothic-Bold.otf]
\setCJKsansfont{HaranoAjiGothic-Regular.otf}[BoldFont=HaranoAjiGothic-Bold.otf]
""" + SANS,
}


def itemize(lines):
    if not lines:
        return ''
    return '\\begin{itemize}\n%s\n\\end{itemize}' % '\n'.join('  \\item %s' % overleaf.tex(line) for line in lines)


def figure_file(data):
    """(extension, bytes) of a figure: as it is, or a large one as a JPEG on white."""
    if len(data) <= LARGE_FIGURE:
        return 'png', data
    with Image.open(io.BytesIO(data)) as image:
        image = image.convert('RGBA')
        flat = Image.new('RGB', image.size, 'white')
        flat.paste(image, mask=image.getchannel('A'))
        out = io.BytesIO()
        flat.save(out, format='JPEG', quality=88, optimize=True)
    return ('jpg', out.getvalue()) if out.tell() < len(data) else ('png', data)


def figure(page, name, height):
    caption = page['figure'].get('credit') or page['figure'].get('label') or ''
    return ('{\\centering\\includegraphics[width=\\linewidth,height=%s\\textheight,keepaspectratio]{%s}\\\\[2pt]\n'
            '{\\tiny\\color{black!55}%s}\\par}' % (height, name, overleaf.tex(caption)))


def frame(page, name):
    """One slide; name: the file of its figure in the project, if it has one."""
    size = SIZES.get(page.get('text_size'), '\\small')
    notes = notes_text(page)
    note = '\n\\note{%s}' % '\\\\ '.join(overleaf.tex(part) for part in notes.split('\n') if part.strip()) if notes else ''
    if page['kind'] == 'title':
        lines = '\\\\[0.6em]\n'.join(overleaf.tex(line) for line in page['bullets'])
        return ('\\begin{frame}\n\\centering\\vfill\n{\\Large\\bfseries %s\\par}\\vspace{1.4em}\n{\\normalsize %s\\par}\\vfill\n'
                '\\end{frame}%s\n' % (overleaf.tex(page['title']), lines, note))
    body = []
    if page['kind'] == 'columns':
        headings = (page.get('headings') or [])[:2]
        headings = headings if len(headings) == 2 else [_('Strengths'), _('Limits')]
        body.append('\\begin{columns}[T]')
        for heading, lines in ((headings[0], page['bullets']), (headings[1], page.get('secondary') or [])):
            body.append('\\begin{column}{0.48\\textwidth}\n\\textbf{%s}\n%s\n\\end{column}' % (overleaf.tex(heading), itemize(lines)))
        body.append('\\end{columns}')
    elif name:
        share = page['figure_size'] / 100
        if page['layout'] in ('left', 'right'):
            text = '\\begin{column}{%.2f\\textwidth}\n%s\n\\end{column}' % (0.96 - share, itemize(page['bullets']))
            picture = '\\begin{column}{%.2f\\textwidth}\n%s\n\\end{column}' % (share, figure(page, name, 0.68))
            body.append('\\begin{columns}[c]\n%s\n%s\n\\end{columns}' % ((picture, text) if page['layout'] == 'left' else (text, picture)))
        else:
            picture = figure(page, name, '%.2f' % (share * 0.8))
            text = itemize(page['bullets'])
            body.append('%s\n\\par\\vspace{0.5em}\n%s' % ((picture, text) if page['layout'] == 'top' else (text, picture)))
    else:
        body.append(itemize(page['bullets']))
    return '\\begin{frame}[t]{%s}\n%s\n\\end{frame}%s\n' % (overleaf.tex(page['title']), FIT % (size + '\n' + '\n'.join(body)), note)


def render_project(pages, footer, title, lang):
    """{name: content} of the project; pages: as slides.shown_pages gives them."""
    if lang not in LANGUAGES:
        raise ExportError(_('The LaTeX export is not available for this language yet.'))
    urls = list(dict.fromkeys(p['figure']['url'] for p in pages if p.get('figure')))
    try:
        with ThreadPoolExecutor(max_workers=EXPORT_WORKERS) as pool:
            images = dict(zip(urls, pool.map(sources.closing(slides_pptx.figure_bytes), urls)))
    except ExportError:
        raise ExportError(_('A figure could not be included. Please try the export again later.'))
    names, files = {}, {}
    for i, url in enumerate(urls, 1):
        extension, data = figure_file(images[url])
        names[url] = 'figures/figure-%d.%s' % (i, extension)
        files[names[url]] = data
    tex = PREAMBLE.format(comment=_('Journal club slides written with SummarizePaper; check each point in the paper.'),
                          fonts=CJK_FONTS.get(lang, SANS), footer=overleaf.tex(footer),
                          notes=_('uncomment to show the speaker notes and the passages of the paper'))
    tex += FIT_MACRO + '\\title{%s}\n\\date{}\n\n\\begin{document}\n\n' % overleaf.tex(title)
    tex += '\n'.join(frame(page, names.get((page.get('figure') or {}).get('url'))) for page in pages)
    tex += '\n\\end{document}\n'
    files['main.tex'] = tex
    if sum(len(c) for c in files.values()) > overleaf.MAX_ZIP:
        raise ExportError(_('The figures are too large for Overleaf. Remove one or two, then export again.'))
    return files
