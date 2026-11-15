"""{% paper_around arxiv_id as around %}: the people who sign the most papers around a paper, and how many papers
cite it, from its stored neighbourhood (neighbours.py): nothing is computed for the page. See teasers.py."""
from django import template

register = template.Library()


@register.simple_tag(takes_context=True)
def paper_around(context, arxiv_id, title=''):
    from urllib.parse import quote
    from django.utils.translation import get_language
    from summarizer import neighbours, teasers
    teams_next = '/%s/teams/?topic=%s' % (get_language(), quote(str(title or '')[:300]))
    try:
        data = neighbours.stored(str(arxiv_id or '')) or {}
    except Exception:
        data = {}
    request = context.get('request')
    user = getattr(request, 'user', None)
    citing = (data.get('counts') or {}).get('cites') or 0
    return {'people': teasers.people_around(data), 'citing': citing, 'citing_more': citing >= neighbours.LINKED,
            'offer': teasers.audience(user, 'teams'), 'teams_next': teams_next,
            'paid_citations': teasers.access.has_tool(user, 'citations')}
