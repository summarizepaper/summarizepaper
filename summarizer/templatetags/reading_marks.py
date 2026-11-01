"""{% reading_mark arxiv_id title %}: the reading status buttons shown next to a paper, for a signed-in reader.
Their statuses are read once per page (personal_activity.statuses)."""
from django import template
from django.utils.translation import get_language

register = template.Library()


@register.inclusion_tag('summarizer/reading_mark.html', takes_context=True)
def reading_mark(context, arxiv_id, title='', full=False):
    from summarizer import personal_activity
    request = context.get('request')
    user = getattr(request, 'user', None)
    if not arxiv_id or user is None or not user.is_authenticated:
        return {'show': False}
    known = getattr(request, '_reading_statuses', None)
    if known is None:
        known = request._reading_statuses = personal_activity.statuses(user)
    labels = personal_activity.reading_labels()
    return {'show': True, 'arxiv_id': arxiv_id, 'title': title, 'current': known.get(personal_activity.base_of(arxiv_id), ''),
            'labels': labels.items() if full else [('to_read', labels['to_read'])], 'full': full,
            'csrf_token': context.get('csrf_token'), 'next': request.get_full_path(), 'LANGUAGE_CODE': get_language()}
