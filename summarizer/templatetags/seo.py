import html
import re

from django import template
from django.utils.safestring import mark_safe

from summarizer import keypoints, summary_text

register = template.Library()


@register.filter
def oneline(value):
    """Collapse the newlines and runs of spaces found in arXiv titles and abstracts."""
    return re.sub(r'\s+', ' ', str(value or '')).strip()


@register.filter
def plaintext(value):
    """Readable text from stored summary HTML: tags become spaces, entities are decoded."""
    text = re.sub(r'<[^>]+>', ' ', str(value or ''))
    text = re.sub(r'\s+', ' ', html.unescape(text)).strip()
    text = summary_text.PREFIX.sub('', text, count=1).strip()
    #some generated summaries start with a bare "Summary"/"Résumé" label glued to the text
    return re.sub(r'^(Summary|Résumé)\s*:?\s*(?=[A-ZÀ-Ý])', '', text)


@register.filter
def first_point(value):
    """The first point of a numbered summary, without its number; other summaries unchanged."""
    text = plaintext(value)
    match = re.match(r'1[.)]\s+(.+?)(?:\s+2[.)]\s|$)', text)
    return match.group(1) if match else text


@register.filter
def keypoint(value):
    """A key point as safe HTML: no markdown, short leading label in bold."""
    return mark_safe(keypoints.point_html(str(value or '')))


@register.filter
def lay_summary(value):
    return mark_safe(summary_text.lay_html(value))
