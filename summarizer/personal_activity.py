"""Private reading and conversation history. Never changes generation quota records."""
import re
from datetime import timedelta
from urllib.parse import urlencode, unquote, urlsplit

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Exists, F, Max, OuterRef, Q, Subquery, TextField, Value
from django.db.models.functions import Coalesce, Concat, Greatest, StrIndex
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone
from django.utils.translation import gettext as _

from . import sources

from .models import AIassistant, ArxivPaper, ChatConversation, PaperHistory, PaperVisit


def record_visit(user, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper:
        PaperVisit.objects.update_or_create(user=user, paper=paper, defaults={'visited': timezone.now()})


def search_path(text):
    try:
        return urlsplit(text).path if '://' in text else text
    except ValueError:
        return text  # malformed pasted URLs remain ordinary search text


def paper_matches(query):
    """Local search only: pasted identifiers never trigger a remote lookup."""
    text = unquote(query).strip()
    path = search_path(text)
    paper_key = re.search(r'/paper/([^/?#]+)', path)
    if paper_key:
        text = paper_key.group(1)
    if text.startswith('doi_'):
        return ArxivPaper.objects.filter(arxiv_id__iexact=text)
    doi = sources.clean_doi(text)
    if doi and not doi.startswith('10.48550/arxiv.'):
        return ArxivPaper.objects.filter(Q(arxiv_id__iexact=sources.doi_key(doi)) | Q(link_doi__icontains=doi))
    arxiv_text = search_path(text)
    arxiv = re.search(r'(?:^|arxiv:|/abs/|/pdf/|arxiv\.)(\d{4}\.\d{4,5}(?:v\d+)?|[a-z-]+(?:/|--)\d{7}(?:v\d+)?)(?:\.pdf)?/?$', arxiv_text, re.I)
    if arxiv:
        identifier = re.sub(r'v\d+$', '', arxiv.group(1), flags=re.I).replace('/', '--')
        return ArxivPaper.objects.filter(arxiv_id__iregex=r'^' + re.escape(identifier) + r'(v[0-9]+)?$')
    rows = ArxivPaper.objects.all()
    for word in text.split()[:12]:
        # Subquery avoids multiplying results when several authors match.
        authors = ArxivPaper.objects.filter(authors__name__icontains=word).values('pk')
        rows = rows.filter(Q(title__icontains=word) | Q(arxiv_id__icontains=word) |
                           Q(pk__in=Subquery(authors)))
    return rows


def period_label(date):
    today = timezone.localdate()
    day = timezone.localtime(date).date()
    if day == today:
        return _('Today')
    if day == today - timedelta(days=1):
        return _('Yesterday')
    if day >= today - timedelta(days=today.weekday()):
        return _('This week')
    if day >= today.replace(day=1):
        return _('This month')
    return _('Earlier')


@login_required(login_url='login')
def page(request):
    tab = 'conversations' if request.GET.get('tab') == 'conversations' else 'readings'
    query = request.GET.get('q', '').strip()[:200]
    if tab == 'readings':
        visits = PaperVisit.objects.filter(user=request.user, paper=OuterRef('pk'))
        requests = PaperHistory.objects.filter(user=request.user, arxiv_id=OuterRef('arxiv_id')).order_by('-created')
        rows = ArxivPaper.objects.filter(
            Q(pk__in=PaperVisit.objects.filter(user=request.user).values('paper_id')) |
            Q(arxiv_id__in=PaperHistory.objects.filter(user=request.user).values('arxiv_id'))
        ).annotate(
            last_visit=Subquery(visits.values('visited')[:1]),
            last_request=Subquery(requests.values('created')[:1]),
        ).annotate(date=Coalesce(Greatest('last_visit', 'last_request'), 'last_visit', 'last_request'))
        rows = rows.filter(date__isnull=False)
        if query:
            rows = rows.filter(pk__in=Subquery(paper_matches(query).values('pk')))
        rows = rows.order_by('-date', '-pk')
    else:
        own = AIassistant.objects.filter(user=request.user, active=True, response__isnull=False).exclude(response='')
        latest = own.order_by().values('arxiv_id').annotate(last=Max('pk')).values('last')
        names = ChatConversation.objects.filter(user=request.user, room=OuterRef('arxiv_id'))
        rows = own.filter(pk__in=Subquery(latest)).annotate(name=Subquery(names.values('title')[:1]))
        if query:
            questions = own
            for word in query.split()[:12]:
                questions = questions.filter(query__icontains=word)
            # CSV membership uses delimiters: a short ID must not match a different paper.
            papers = paper_matches(query).annotate(position=StrIndex(
                Concat(Value(','), OuterRef('papers'), Value(','), output_field=TextField()),
                Concat(Value(','), F('arxiv_id'), Value(','), output_field=TextField())))
            linked = papers.filter(Q(arxiv_id=OuterRef('arxiv_id')) | Q(position__gt=0))
            matched = own.annotate(has_paper=Exists(linked)).filter(
                Q(arxiv_id__in=Subquery(questions.values('arxiv_id'))) | Q(has_paper=True)).values('arxiv_id')
            named = Q()
            for word in query.split()[:12]:
                named &= Q(name__icontains=word)
            rows = rows.filter(Q(arxiv_id__in=Subquery(matched)) | named)
        rows = rows.order_by('-created', '-pk')
    result = Paginator(rows, 20).get_page(request.GET.get('page'))
    items = []
    if tab == 'readings':
        for row in result:
            url = reverse('arxividpage', kwargs={'arxiv_id': row.arxiv_id})
            items.append({'title': row.title or row.arxiv_id, 'id': row.arxiv_id, 'date': row.date,
                          'visited': row.last_visit == row.date, 'url': url, 'chat': url + '#assistant',
                          'collection': url + '#paper-collections'})
    else:
        known = dict(ArxivPaper.objects.filter(arxiv_id__in=[r.arxiv_id for r in result]).values_list('arxiv_id', 'title'))
        for row in result:
            if row.arxiv_id in known:
                url = reverse('arxividpage', kwargs={'arxiv_id': row.arxiv_id}) + '#assistant'
            else:
                url = reverse('chat') + '?' + urlencode({'c': row.arxiv_id})
            items.append({'title': row.name or row.query or known.get(row.arxiv_id) or row.arxiv_id,
                          'date': row.created, 'url': url, 'paper_title': known.get(row.arxiv_id, '')})
    for item in items:
        item['period'] = period_label(item['date'])
    def page_url(number):
        return '?' + urlencode({'tab': tab, 'q': query, 'page': number})
    return render(request, 'summarizer/history.html', {
        'tab': tab, 'query': query, 'items': items, 'pagination': result,
        'clear_url': '?' + urlencode({'tab': tab}),
        'previous_url': page_url(result.previous_page_number()) if result.has_previous() else '',
        'next_url': page_url(result.next_page_number()) if result.has_next() else '',
        'readings_url': '?' + urlencode({'tab': 'readings', 'q': query}),
        'conversations_url': '?' + urlencode({'tab': 'conversations', 'q': query}),
    })
