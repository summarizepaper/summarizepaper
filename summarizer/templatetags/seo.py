import html
import re

from django import template

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
    #some generated summaries start with a bare "Summary"/"Résumé" label glued to the text
    return re.sub(r'^(Summary|Résumé)\s*:?\s*(?=[A-ZÀ-Ý])', '', text)
