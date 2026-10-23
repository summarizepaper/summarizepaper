"""A self-contained, searchable newsroom brief; no browser or remote assets."""
import re
from pathlib import Path

from django.conf import settings
from django.utils.formats import date_format
from django.utils.translation import gettext as _
from fpdf import FPDF

from . import sources, publication_status
from .latex_pdf import prose_html


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


def document(paper, row, editorial=None, publication=None):
    kit = row.data()
    publication = publication or kit.get("publication_status")
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

    def heading(value):
        if pdf.will_page_break(26):
            pdf.add_page()
        pdf.ln(4)
        text(value, 'B', 13, (65, 58, 130))

    def quote(value):
        text('“%s”' % value, 'I', color=(65, 70, 80))

    text(paper.title or paper.arxiv_id, 'B', 18)
    text(', '.join(paper.authors.values_list('name', flat=True)[:12]), size=9)
    text(paper.arxiv_id + '  ·  ' + date_format(row.updated, 'j F Y'), size=9)
    url = sources.ref_url(paper.arxiv_id)
    text(url, size=9, color=(65, 58, 130), link=url)
    source = kit.get('source') or {}
    scope = source.get('scope', 'abstract' if kit.get('abstract_only') else 'unknown')
    scopes = {'full_text': _('Source read: available full text.'), 'abstract': _('Source read: abstract only.'),
              'excerpt': _('Source read: partial text. The analysis does not cover the entire paper.'),
              'unknown': _('Legacy kit: the exact source scope was not recorded. Enrich the kit to record it.')}
    text(scopes.get(scope, scopes['unknown']), 'B', size=9)
    text(_('Peer-review status is not independently confirmed. A journal reference alone does not establish it.'), size=9)
    if source.get('journal_ref'):
        text(source['journal_ref'], size=9)
    if source.get('doi'):
        text('DOI: ' + source['doi'], size=9)
    text(_('AI-assisted reporting brief. Verify facts, interpretation and quotations against the original paper before publication.'), 'I', size=9)

    if publication:
        heading(_('Editorial status: automatic verification'))
        text(publication_status.description(publication), 'B')
        text(_('Checked at') + ': ' + publication.get('checked_at', ''), size=9)
        if publication.get('journal'):
            text(publication['journal'] + (' · ' + publication['date'] if publication.get('date') else ''))
        if publication.get('unavailable'):
            text(_('Some sources were unavailable') + ': ' + ', '.join(publication['unavailable']), 'I', size=9)
        for reference in publication.get('sources', []):
            text(reference['name'] + ': ' + reference['detail'], size=9)
            text(reference['url'], size=9, link=reference['url'])
        text(_('A publication record does not independently establish peer review. No publication found does not mean the manuscript is unpublished; private submission and review decisions may be inaccessible.'), 'I', size=9)

    heading(kit.get('headline') or _('Plain-language explanation'))
    text(kit.get('plain_explanation'))
    heading(_("What's new"))
    text(kit.get('whats_new'))

    if kit.get('evidence_cards'):
        heading(_('Findings and source evidence'))
        text(_('A matching quote confirms that the passage exists, not that the finding is independently validated.'), 'I', size=9)
        for index, card in enumerate(kit['evidence_cards'], 1):
            if pdf.will_page_break(35):
                pdf.add_page()
            text('%s. %s' % (index, card['claim']), 'B')
            text(card.get('why_it_matters'))
            if card.get('numeric_mismatch'):
                text(_('Numbers in this claim do not all match the cited passage. Check them before publication.'), 'B')
            if card.get('verified') and card.get('quote'):
                text(_('Source passage verified') + (' · ' + card['location'] if card.get('location') else ''), size=9)
                quote(card['quote'])
            else:
                text(_('Supporting passage not verified. Check this claim in the paper before using it.'), 'I')
            text(_('Limit of this finding') + ': ' + card.get('limitation', ''))

    if kit.get('angles'):
        heading(_('Reporting angles'))
        for item in kit['angles']:
            text(item['angle'], 'B')
            text(item.get('why_now'))
            text(_('Next reporting step') + ': ' + item.get('reporting_step', ''))
    if kit.get('wording_checks'):
        heading(_('Wording to check before publication'))
        for item in kit['wording_checks']:
            text(_('Avoid') + ': ' + item['avoid'], 'B')
            text(_('Prefer') + ': ' + item.get('prefer', ''))
            text(item.get('reason'), 'I', size=9)

    heading(_('Hype check'))
    text(kit.get('verdict'), 'B')
    for item in kit.get('hype_check', []):
        text(item['signal'], 'B')
        text(item.get('detail'))
    used_quotes = {' '.join(card.get('quote', '').split()) for card in kit.get('evidence_cards', []) if card.get('verified')}
    key_sentences = [sentence for sentence in kit.get('key_sentences', []) if ' '.join(sentence.split()) not in used_quotes]
    if key_sentences:
        heading(_('Key sentences from the paper'))
        text(_('Copied word for word from the paper (checked against its text). Quote the paper, not us.'), 'I', size=9)
        for item in key_sentences:
            quote(item)
    heading(_('Questions to ask the authors'))
    for index, item in enumerate(publication_status.questions(kit, publication), 1):
        text('%s. %s' % (index, item))
    heading(_('Researchers to approach for outside comment'))
    text(_('These researchers are not listed as coauthors in our metadata. Their independence, affiliations and possible conflicts of interest still need to be checked.'), 'I', size=9)
    for person in kit.get('experts', []):
        text(person['name'] + (' — ' + person['affiliation'] if person.get('affiliation') else ''), 'B')
        text(person.get('title'))
        source_url = sources.ref_url(person['arxiv_id'])
        text(source_url, size=9, color=(65, 58, 130), link=source_url)
    if not kit.get('experts'):
        text(_('We do not know enough papers close to this one yet.'))
    if kit.get('glossary'):
        heading(_('Glossary'))
        for item in kit['glossary']:
            text(item['term'] + ': ' + item.get('definition', ''))
    if editorial:
        heading(_('From the weekly letter'))
        text(editorial['heading'], 'B')
        text(editorial['text'])
        text(_('Editorial context written by AI from abstracts, not independent scientific validation. The kit and its verified quotes rely on the source paper, not on this letter.'), 'I', size=9)
    return bytes(pdf.output())
