"""Private reading and conversation history. Never changes generation quota records."""
from urllib.parse import urlencode

from django.contrib.auth.decorators import login_required
from django.core.paginator import Paginator
from django.db.models import Max, OuterRef, Q, Subquery
from django.db.models.functions import Coalesce, Greatest
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone

from .models import AIassistant, ArxivPaper, ChatConversation, PaperHistory, PaperVisit


def record_visit(user, arxiv_id):
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper:
        PaperVisit.objects.update_or_create(user=user, paper=paper, defaults={'visited': timezone.now()})


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
            rows = rows.filter(Q(title__icontains=query) | Q(arxiv_id__icontains=query))
        rows = rows.order_by('-date', '-pk')
    else:
        own = AIassistant.objects.filter(user=request.user, active=True, response__isnull=False).exclude(response='')
        latest = own.order_by().values('arxiv_id').annotate(last=Max('pk')).values('last')
        names = ChatConversation.objects.filter(user=request.user, room=OuterRef('arxiv_id'))
        rows = own.filter(pk__in=Subquery(latest)).annotate(name=Subquery(names.values('title')[:1]))
        if query:
            matched = own.filter(query__icontains=query).values('arxiv_id')
            papers = ArxivPaper.objects.filter(title__icontains=query).values('arxiv_id')
            rows = rows.filter(Q(arxiv_id__in=Subquery(matched)) | Q(name__icontains=query) |
                               Q(arxiv_id__in=Subquery(papers)))
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
    def page_url(number):
        return '?' + urlencode({'tab': tab, 'q': query, 'page': number})
    return render(request, 'summarizer/history.html', {
        'tab': tab, 'query': query, 'items': items, 'pagination': result,
        'previous_url': page_url(result.previous_page_number()) if result.has_previous() else '',
        'next_url': page_url(result.next_page_number()) if result.has_next() else '',
        'readings_url': '?' + urlencode({'tab': 'readings', 'q': query}),
        'conversations_url': '?' + urlencode({'tab': 'conversations', 'q': query}),
    })
