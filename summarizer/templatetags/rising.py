"""{% rising_topic arxiv_id as topic %}: the recent rising topic a paper belongs to (see trends.topic_of_paper)."""
from django import template
from django.utils.translation import get_language

register = template.Library()


@register.simple_tag
def rising_topic(arxiv_id):
    from summarizer import trends
    try:
        found = trends.topic_of_paper(str(arxiv_id or '').replace('--', '/'), get_language())
    except Exception:
        return None
    if not found:
        return None
    scope, week, topic = found
    return dict(topic, scope=scope, week=week, scope_name=trends.scope_name(scope))
