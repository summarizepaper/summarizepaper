"""{% weekly_letter arxiv_id as letter %}: the latest weekly letter that presents a paper (see letters.letter_of_paper)."""
from django import template
from django.utils.translation import get_language

register = template.Library()


@register.simple_tag
def weekly_letter(arxiv_id):
    from summarizer import letters
    try:
        letter = letters.letter_of_paper(str(arxiv_id or ''), get_language())
    except Exception:
        return None
    if letter is None:
        return None
    return {'letter': letter, 'name': letters.localized_name(letter.topic, letter.lang)[1]}
