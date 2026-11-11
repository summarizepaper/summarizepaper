from django import template
from django.utils.translation import get_language

register = template.Library()


@register.simple_tag
def colon():
    """A colon after a link or a tag, where the text cannot carry it: French puts a space before it."""
    return ' :' if (get_language() or '').startswith('fr') else ':'


@register.filter
def quoted(value):
    """A quotation between the quotation marks of the page's language."""
    return ('« %s »' if (get_language() or '').startswith('fr') else '“%s”') % value
