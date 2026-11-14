"""{% weekly_letter arxiv_id category as letter %}: the latest weekly letter that presents a paper (see
letters.letter_of_paper), else the latest letter of the paper's field, with `field` set."""
from django import template
from django.utils.translation import get_language

register = template.Library()


@register.simple_tag
def weekly_letter(arxiv_id, category=''):
    from summarizer import letters
    from summarizer.models import Letter
    lang = get_language()
    try:
        letter = letters.letter_of_paper(str(arxiv_id or ''), lang)
        field = False
        if letter is None and category:
            topic = next((t for t in letters.TOPICS if letters.in_topic(t, [str(category)])), None)
            letter = topic and (Letter.objects.filter(topic=topic, lang=lang).order_by('-day')
                                .only('topic', 'day', 'title', 'lang').first())
            field = True
    except Exception:
        return None
    if not letter:
        return None
    return {'letter': letter, 'name': letters.localized_name(letter.topic, letter.lang)[1], 'field': field}
