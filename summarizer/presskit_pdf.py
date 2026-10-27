"""The press kit as a file: a self-contained, searchable PDF (no browser or remote assets) or Markdown to paste.

Both read the same outline: the reader's role first (all three when none is given), then what every role shares.
"""
import re
from pathlib import Path

from django.conf import settings
from django.utils.formats import date_format
from django.utils.translation import gettext as _
from fpdf import FPDF

from . import sources, publication_status
from .latex_pdf import prose_html


def where(found):
    """'Results · ¶2 · Figure 3' for a verified passage."""
    location = found.get('location') or ''
    parts = [location if len(location) <= 70 else location[:69].rstrip() + '…', '¶%s' % found['paragraph'] if found.get('paragraph') else '', found.get('figure') or '']
    return ' · '.join(p for p in parts if p)


def outline(paper, row, publication=None, role=None, editorial=None):
    """[(kind, text, extra)]: 'title', 'meta', 'h', 'h3', 'p', 'note', 'warn', 'quote' (extra: where), 'item', 'link' (extra: url)."""
    kit = row.data()
    publication = publication or kit.get('publication_status')
    out = []
    add = lambda kind, text, extra='': out.append((kind, str(text or ''), extra)) if text else None

    add('title', paper.title or paper.arxiv_id)
    add('meta', ', '.join(paper.authors.values_list('name', flat=True)[:12]))
    add('meta', paper.arxiv_id + '  ·  ' + date_format(row.updated, 'j F Y'))
    url = sources.ref_url(paper.arxiv_id)
    add('link', url, url)
    source = kit.get('source') or {}
    scope = source.get('scope', 'abstract' if kit.get('abstract_only') else 'unknown')
    scopes = {'full_text': _('Source read: available full text.'), 'abstract': _('Source read: abstract only.'),
              'excerpt': _('Source read: partial text. The analysis does not cover the entire paper.'),
              'unknown': _('Legacy kit: the exact source scope was not recorded. Enrich the kit to record it.')}
    add('warn', scopes.get(scope, scopes['unknown']))
    add('meta', _('Peer-review status is not independently confirmed. A journal reference alone does not establish it.'))
    add('meta', source.get('journal_ref'))
    add('meta', 'DOI: ' + source['doi'] if source.get('doi') else '')
    add('note', _('AI-assisted brief. Verify facts, interpretation and quotations against the original paper before publication.'))

    add('h', kit.get('headline') or _('Plain-language explanation'))
    add('p', kit.get('plain_explanation'))
    if kit.get('verdict'):
        add('warn', kit['verdict'])
    if kit.get('numbers_absent'):
        add('warn', _('Numbers of this explanation that the paper text does not state: %s. Check them first.') % ', '.join(kit['numbers_absent']))

    roles = kit.get('roles') or {}
    for key in ([role] if role else ['author', 'commentator', 'writer']):
        role_part(add, key, roles.get(key) or {}, kit, publication)

    add('h', _("What's new"))
    add('p', kit.get('whats_new'))
    novelty = kit.get('novelty') or {}
    if novelty.get('comparisons') or novelty.get('assessment'):
        add('h3', _('Against the closest earlier papers'))
        add('p', novelty.get('assessment'))
        for item in novelty.get('comparisons', []):
            add('item', '%s (%s)' % (item['title'], item.get('date', '')[:4]))
            add('p', _('Already done:') + ' ' + item.get('already', ''))
            add('p', _('This paper adds:') + ' ' + item.get('adds', ''))
        add('note', _('Compared with the abstracts of the closest papers we know, chosen automatically: not a literature review.'))

    if kit.get('evidence_cards'):
        add('h', _('Findings and source evidence'))
        add('note', _('A matching quote confirms that the passage exists, not that the finding is independently validated.'))
        for index, card in enumerate(kit['evidence_cards'], 1):
            kind = {'result': _('Result'), 'interpretation': _('Interpretation')}.get(card.get('kind'), '')
            add('h3', '%s. %s%s' % (index, card['claim'], ' [%s]' % kind if kind else ''))
            add('p', card.get('why_it_matters'))
            if card.get('numeric_mismatch'):
                add('warn', _('Numbers in this claim do not all match the cited passage. Check them before publication.'))
            if card.get('verified') and card.get('quote'):
                add('quote', card['quote'], where(card))
            else:
                add('note', _('Supporting passage not verified. Check this claim in the paper before using it.'))
            add('p', _('Limit of this finding:') + ' ' + card.get('limitation', ''))

    add('h', _('Hype check'))
    for item in kit.get('hype_check', []):
        mark = {'ok': '✓', 'warning': '⚠'}.get(item.get('level'), '•')
        add('item', '%s %s: %s' % (mark, item['signal'], item.get('detail', '')))

    if kit.get('figures'):
        add('h', _('Figures for the press'))
        for figure in kit['figures']:
            add('h3', figure['label'])
            add('p', figure.get('why'))
            add('note', figure.get('caption'))
            add('meta', _('Credit:') + ' ' + figure['credit'])
            add('link', figure['source_url'], figure['source_url'])

    if kit.get('key_sentences'):
        add('h', _('Key sentences from the paper'))
        add('note', _('Copied word for word from the paper (checked against its text). Quote the paper, not us.'))
        for sentence in kit['key_sentences']:
            add('quote', sentence)

    add('h', _('Researchers to approach for outside comment'))
    if kit.get('experts_checked'):
        add('note', _('They sign close papers first, last or as corresponding author, and neither sign this paper nor have signed with its authors in the last five years (OpenAlex). Their independence and possible conflicts of interest still need to be checked.'))
    else:
        add('note', _('These researchers are not listed as coauthors in our metadata. Their independence, affiliations and possible conflicts of interest still need to be checked.'))
    for person in kit.get('experts', []):
        add('item', person['name'] + (' — ' + person['affiliation'] if person.get('affiliation') else ''))
        add('meta', '%s%s' % (person.get('title', ''), ' (%s)' % person['year'] if person.get('year') else ''))
        link = sources.ref_url(person['arxiv_id']) if person.get('arxiv_id') else person.get('url', '')
        add('link', link, link)
    if not kit.get('experts'):
        add('p', _('We do not know enough papers close to this one yet.'))

    if kit.get('glossary'):
        add('h', _('Glossary'))
        for item in kit['glossary']:
            add('item', item['term'] + ': ' + item.get('definition', ''))

    if publication:
        add('h', _('Editorial status: automatic verification'))
        add('warn', publication_status.description(publication))
        add('meta', _('Checked at:') + ' ' + publication.get('checked_at', ''))
        if publication.get('journal'):
            add('p', publication['journal'] + (' · ' + publication['date'] if publication.get('date') else ''))
        if publication.get('unavailable'):
            add('note', _('Some sources were unavailable:') + ' ' + ', '.join(publication['unavailable']))
        for reference in publication.get('sources', []):
            add('meta', reference['name'] + ': ' + reference['detail'])
            add('link', reference['url'], reference['url'])
        add('note', _('A publication record does not independently establish peer review. No publication found does not mean the manuscript is unpublished; private submission and review decisions may be inaccessible.'))
    if editorial:
        add('h', _('From the weekly letter'))
        add('h3', editorial['heading'])
        add('p', editorial['text'])
        add('note', _('Editorial context written by AI from abstracts, not independent scientific validation. The kit and its verified quotes rely on the source paper, not on this letter.'))
    return out


def numbers_warning(add, part):
    if part.get('numbers_absent'):
        add('warn', _('Numbers of these texts that the paper text does not state: %s. Check them before you send them.')
            % ', '.join(part['numbers_absent']))


def role_part(add, key, part, kit, publication):
    if key == 'author' and ((part.get('press_release') or {}).get('lede') or part.get('likely_questions')):
        release = part.get('press_release') or {}
        add('h', _('For the authors: draft press release'))
        numbers_warning(add, part)
        add('note', _('A draft for your press office, written by AI from the paper: check every sentence, and add your own quote.'))
        add('h3', release.get('title'))
        add('warn', release.get('lede'))
        for paragraph in release.get('body', []):
            add('p', paragraph)
        if release.get('quote_slot'):
            add('p', _('[Your quote: %(slot)s]') % {'slot': release['quote_slot']})
        if part.get('likely_questions'):
            add('h3', _('Questions journalists are likely to ask'))
            for item in part['likely_questions']:
                add('item', item['question'])
                add('p', item.get('answer'))
                if item.get('passage'):
                    add('quote', item['passage']['quote'], where(item['passage']))
        if part.get('misreadings'):
            add('h3', _('Likely misreadings'))
            for item in part['misreadings']:
                add('item', item['misreading'])
                add('p', item.get('correction'))
        if part.get('lay_summary'):
            add('h3', _('Lay summary (institution news, grant report)'))
            add('p', part['lay_summary'])
        if part.get('social_posts'):
            add('h3', _('A short thread'))
            for post in part['social_posts']:
                add('item', post)
    elif key == 'commentator' and (part.get('in_brief') or part.get('strengths')):
        add('h', _('Asked to comment: what this paper shows'))
        numbers_warning(add, part)
        for line in part.get('in_brief', []):
            add('item', line)
        for title, items in ((_('Strengths'), part.get('strengths')), (_('Weaknesses and open questions'), part.get('weaknesses'))):
            if items:
                add('h3', title)
                for item in items:
                    add('item', item['point'])
                    if item.get('passage'):
                        add('quote', item['passage']['quote'], where(item['passage']))
        if part.get('what_would_convince'):
            add('h3', _('What would settle the main claim'))
            add('p', part['what_would_convince'])
        if part.get('comment_draft'):
            add('h3', _('A comment to adapt in your own words'))
            add('p', part['comment_draft'])
    elif key == 'writer':
        if kit.get('angles'):
            add('h', _('Reporting angles'))
            for item in kit['angles']:
                add('h3', item['angle'])
                add('p', item.get('why_now'))
                add('p', _('Next reporting step:') + ' ' + item.get('reporting_step', ''))
        if kit.get('wording_checks'):
            add('h', _('Wording to check before publication'))
            for item in kit['wording_checks']:
                add('warn', _('Avoid:') + ' ' + item['avoid'])
                add('p', _('Prefer:') + ' ' + item.get('prefer', ''))
                add('note', item.get('reason'))
        add('h', _('Questions to ask the authors'))
        answers = kit.get('question_answers') or {}
        for index, question in enumerate(publication_status.questions(kit, publication), 1):
            add('item', '%s. %s' % (index, question))
            if question in answers:
                add('note', _('The paper already says:') + ' ' + answers[question]['answer'])


class PressPDF(FPDF):
    def header(self):
        self.set_xy(self.l_margin, self.t_margin)
        self.set_font('Brief', 'B', 11)
        self.set_text_color(65, 58, 130)
        self.cell(0, 8, 'SummarizePaper  /  ' + _('Press kit'), new_x='LMARGIN', new_y='NEXT')
        self.set_draw_color(215, 218, 228)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

    def footer(self):
        self.set_y(-14)
        self.set_font('Brief', '', 8)
        self.set_text_color(100, 106, 120)
        self.cell(0, 6, 'SummarizePaper  ·  %s %s' % (_('Page'), self.page_no()), align='R')


def document(paper, row, editorial=None, publication=None, role=None):
    pdf = PressPDF(format='A4')
    pdf.set_margins(18, 17, 18)
    pdf.set_auto_page_break(True, margin=21)
    font = Path(settings.BASE_DIR) / 'font'
    for style, name in [('', 'DejaVuSansCondensed.ttf'), ('B', 'DejaVuSansCondensed-Bold.ttf'),
                        ('I', 'DejaVuSansCondensed-Oblique.ttf')]:
        pdf.add_font('Brief', style, str(font / name))
    pdf.set_title(_('Press kit') + ' — ' + (paper.title or paper.arxiv_id))
    pdf.set_author('SummarizePaper')
    pdf.set_creator('SummarizePaper')
    pdf.set_lang(row.lang)
    pdf.add_page()

    def text(value, style='', size=10, color=(35, 43, 58), link=''):
        value = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', str(value or ''))
        if not value:
            return
        pdf.set_font('Brief', style, size)
        pdf.set_text_color(*color)
        math_prose = prose_html(value) if not link else None
        if math_prose is None:
            pdf.multi_cell(0, 5.4 if size <= 11 else 7.5, value, new_x='LMARGIN', new_y='NEXT', link=link)
        else:
            # Only escaped text and our own sub/sup tags enter the offline HTML renderer.
            if style:
                tag = 'b' if style == 'B' else 'i'
                math_prose = '<%s>%s</%s>' % (tag, math_prose, tag)
            pdf.write_html(math_prose, font_family='Brief')
            pdf.set_x(pdf.l_margin)
        pdf.ln(2)

    for kind, value, extra in outline(paper, row, publication, role, editorial):
        if kind == 'title':
            text(value, 'B', 18)
        elif kind == 'meta':
            text(value, size=9)
        elif kind == 'link':
            text(value, size=9, color=(65, 58, 130), link=extra)
        elif kind == 'h':
            if pdf.will_page_break(26):
                pdf.add_page()
            pdf.ln(4)
            text(value, 'B', 13, (65, 58, 130))
        elif kind == 'h3':
            if pdf.will_page_break(20):
                pdf.add_page()
            text(value, 'B')
        elif kind == 'warn':
            text(value, 'B')
        elif kind == 'note':
            text(value, 'I', size=9)
        elif kind == 'quote':
            if extra:
                text(_('Source passage verified') + ' · ' + extra, size=9)
            text('“%s”' % value, 'I', color=(65, 70, 80))
        elif kind == 'item':
            text('•  ' + value if not re.match(r'\d+\. ', value) else value)
        else:
            text(value)
    return bytes(pdf.output())


def markdown(paper, row, publication=None, role=None, editorial=None):
    lines = []
    for kind, value, extra in outline(paper, row, publication, role, editorial):
        value = ' '.join(value.split())
        if kind == 'title':
            lines += ['# ' + value, '']
        elif kind == 'h':
            lines += ['', '## ' + value, '']
        elif kind == 'h3':
            lines += ['', '### ' + value, '']
        elif kind == 'warn':
            lines += ['**%s**' % value, '']
        elif kind == 'note':
            lines += ['_%s_' % value, '']
        elif kind == 'quote':
            lines += ['> ' + value] + (['> — %s' % extra] if extra else []) + ['']
        elif kind == 'item':
            lines.append(value if re.match(r'\d+\. ', value) else '- ' + value)
        elif kind == 'link':
            lines += ['<%s>' % extra, '']
        else:
            lines += [value, '']
    text = '\n'.join(lines)
    return re.sub(r'\n{3,}', '\n\n', text).strip() + '\n'
