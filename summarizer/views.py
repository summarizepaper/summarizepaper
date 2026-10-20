# Create your views here.
from django.shortcuts import render, redirect, HttpResponseRedirect, get_object_or_404
from django.urls import reverse
from django.http import HttpResponse
import requests
import time
import re
import hashlib
from .models import (AnnouncedPaper, ArxivPaper, Vote, PaperHistory, PDFHistory, SummaryPaper, AIassistant, Search,
                     PaperScore, UserPlan)
from .forms import RegistrationForm
from django.conf import settings
from django.core.mail import send_mail, EmailMessage
import summarizer.utils as utils
from summarizer import access, accounts, citations, digest, library, mailcheck, reading, similar, sources, exports
from datetime import datetime, timedelta
import ast
from django.core.cache import cache
from django.utils import timezone
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.models import User
from django.contrib.sites.shortcuts import get_current_site
from django.template.loader import render_to_string
from django.utils.encoding import force_bytes
from django.utils.http import urlsafe_base64_encode, urlsafe_base64_decode
from django.views import View
import six
import json
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.contrib.auth import authenticate
from django.core.exceptions import PermissionDenied
from django.contrib.auth import views as auth_views
from django.urls import reverse_lazy
from django.core.exceptions import ValidationError
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout
from django.contrib import messages
import os
from django.utils.translation import get_language, get_language_info, gettext as _
from django.db.models import Sum
from django.core.paginator import Paginator
from urllib.parse import urlencode
import urllib, urllib.request
from xml.etree import ElementTree
import asyncio
import aiohttp
from django.views.decorators.http import require_GET
from django.http import JsonResponse
from django.utils.safestring import mark_safe
import logging

logger = logging.getLogger(__name__)

@require_GET
def robots_txt(request):
    lines = [
        "User-Agent: *",
        "Disallow: /admin",
        "Sitemap: https://www.summarizepaper.com/sitemap.xml",
    ]
    return HttpResponse("\n".join(lines), content_type="text/plain")


def llms_txt(request):
    """Plain-text overview of the site for LLM crawlers, see llmstxt.org."""
    from summarizer.guides import published
    return render(request, 'llms.txt', {'guides': published('en')}, content_type='text/plain; charset=utf-8')



class CustomAuthenticationForm(AuthenticationForm):
    """Accepts the email address as well as the username, and tells apart accounts not activated yet."""
    inactive_user = None

    def clean(self):
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')
        if username and password and '@' in username:
            for candidate in User.objects.filter(email__iexact=username.strip()).order_by('-is_active', '-last_login'):
                if candidate.check_password(password):
                    self.cleaned_data['username'] = candidate.username
                    break
            username = self.cleaned_data['username']
        if username and password:
            inactive = User.objects.filter(username=username, is_active=False).first()
            if inactive is not None and inactive.check_password(password):
                self.inactive_user = inactive
                raise ValidationError("Your account is not activated yet.", code='inactive')
        return super().clean()


class CustomLoginView(auth_views.LoginView):
    authentication_form = CustomAuthenticationForm

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        form = context.get('form')
        context['inactive_user'] = getattr(form, 'inactive_user', None)
        context.update(accounts.auth_context())
        return context

    def get_success_url(self):
        #honour a safe ?next= (e.g. coming back to the Premium page)
        next_url = self.get_redirect_url()
        if next_url:
            return next_url
        user = self.request.user
        if user.is_staff:
            return reverse_lazy('about')
        else:
            return reverse_lazy('summarize')


class TokenGenerator(PasswordResetTokenGenerator):
    def _make_hash_value(self, user, timestamp):
        toka=six.text_type(user.pk)
        print('toka',toka)
        tokb=six.text_type(timestamp)
        print('tokb',tokb)
        tokc=six.text_type(user.is_active)
        print('tokc',tokc)
        tok=toka+tokb+tokc
        print('tok',tok)

        return (tok)


generate_token = TokenGenerator()

class ActivateView(View):
    def get(self, request, uidb64, token):
        try:
            uid = int(urlsafe_base64_decode(uidb64))
            print('uidaaa',uid)
            user = User.objects.get(pk=uid)
            print('useraaa',user)
        except (TypeError, ValueError, OverflowError, User.DoesNotExist):
            user = None
            print('none')

        if user is not None and generate_token.check_token(user, token):
            user.is_active = True
            print('aqyui')
            user.save()
            login(request, user)
            #return redirect('summarize/?activated=True')
            response = redirect('summarize')
            response['Location'] += '?activated=True'
            return response

            #return render(request, 'summarizer/home.html', {'activated': True})

        return render(request, 'account_activation_invalid.html')


class RegisterView(View):
    def get(self, request):
        form = RegistrationForm()
        return render(request, 'register.html', {'form': form, **accounts.auth_context()})

    def post(self, request):
        form = RegistrationForm(request.POST)
        context = {'form': form, **accounts.auth_context()}
        ip = access.hash_ip(access.client_ip(request))
        if accounts.looks_like_bot(request.POST) or accounts.signups_exhausted(ip):
            # no account and no email, but the same page a person would see
            return redirect('account_activation_sent')
        if form.is_valid():
            username = form.cleaned_data.get('username')
            if User.objects.filter(username=username).exists():
                form.add_error('username', 'This username is already taken')
                return render(request, 'register.html', context)

            email = form.cleaned_data.get('email')
            if User.objects.filter(email__iexact=email).exists():
                context['email_taken'] = email
                return render(request, 'register.html', context)
            if mailcheck.undeliverable_reason(email):
                form.add_error('email', _('This address cannot receive our emails. Please check it or use another one.'))
                return render(request, 'register.html', context)

            password1 = form.cleaned_data.get('password1')
            user = User.objects.create_user(username=username, email=email, password=password1)
            user.is_active = False
            user.save()
            accounts.count_signup(ip)
            accounts.send_login_link(request, user)
            return redirect('account_activation_sent')

        return render(request, 'register.html', context)


def logout_view(request):
    logout(request)
    # Redirect the user to the login page or any other page of your choice
    return redirect('login')

def search_results(request):
    query = (request.GET.get('q') or '').strip()
    page_num = request.GET.get('page', 1)
    try:
        page_num = int(page_num)
    except (TypeError, ValueError):
        page_num = 1

    src = 'open' if request.GET.get('src') == 'open' else 'arxiv'
    kind = request.GET.get('type') if request.GET.get('type') in sources.SEARCH_TYPES else ''
    if not query:
        # arXiv answers 400 to an empty search_query: show the empty search page instead.
        return render(request, 'summarizer/search_results.html', {'query': '', 'search': [], 'page_obj': None,
                                                                    'src': src, 'kind': kind})
    if src == 'open':
        return open_search_results(request, query, kind, page_num)

    # perform search and pagination logic here
    search_results = []
    #total_results = 0
    items_per_page = 10
    #start = (int(page_num) - 1) * items_per_page
    max_results=30
    print('query',query)

    '''
    query1 = query# urllib.parse.quote(query)
    print('query1',query1)

    # Define the regular expressions for each search sequence
    author_regex = re.compile(r'au')
    title_regex = re.compile(r'ti:')
    abstract_regex = re.compile(r'abs')
    comments_regex = re.compile(r'co')
    journal_regex = re.compile(r'jr')
    category_regex = re.compile(r'cat')
    id_regex = re.compile(r'id_list')
    rn_regex = re.compile(r'rn')

    # Split the query into individual search terms and boolean operators
    #terms = re.findall(r'\w+:\"[^\"]+\"|\w+', query1)
    #terms = re.findall(r'(\w+:\"[^\"]+\"|\w+:\w+|\w+:?\w+):?"?[^\s"]+"?|\w+', query1)
    #terms = re.findall(r'\w+:\"[^\"]+\"|\w+:\w+|\w+', query1)
    terms = re.findall(r'\w+:\"[^\"]+\"|\w+:\w+|\w+|\(|\)', query1)
    print('terms',terms)
    # Process each search term and boolean operator
    new_terms = []
    for i, term in enumerate(terms):
        print('term',i,term)
        if ':' in term:
            # This is a search term
            prefix, value = term.split(':')
            value = value.strip(' "\t')
            print('prefix, value',prefix, value)
            if 1==1:#prefix == 'all':
                # Replace 'all:' with the appropriate sequence
                if author_regex.search(value):
                    prefix = 'au'
                    print('ok')
                elif title_regex.search(value):
                    prefix = 'ti'
                elif abstract_regex.search(value):
                    prefix = 'abs'
                elif comments_regex.search(value):
                    prefix = 'co'
                elif journal_regex.search(value):
                    prefix = 'jr'
                elif category_regex.search(value):
                    prefix = 'cat'
                elif id_regex.search(value):
                    prefix = 'id_list'
                elif rn_regex.search(value):
                    prefix = 'rn'
                #value = value.replace('all:', f'{prefix}:')
            #new_terms.append(f'{prefix}:{value}')
            if  prefix=='au':
                value=value.replace(" ","_")
            else:
                value=value.replace(" ","%20")


            print('value',value,urllib.parse.quote(value))
            new_terms.append(f'{prefix}:"{value}"')
        elif term in ('AND', 'OR', 'ANDNOT'):
            # This is a boolean operator
            #new_terms.append(term.upper())
            new_terms.append(f"+{term}+")
        elif term in ('(',')'):
            new_terms.append(f"{term}")

    if new_terms==[]:
        query1=query1.replace(" ","%20")
        new_terms.append('all:'+query1)
        print('new terms',new_terms)

    # Construct the final URL for the API request
    query2 = ' '.join(new_terms)
    query2=query2.replace(" ","")
    print('query2',query2)
    #query2=urllib.parse.urlencode(query2)
    #print('query2',query2)

    '''
    url = f'https://export.arxiv.org/api/query?search_query={query}&start=0&max_results={max_results}&sortBy=relevance&sortOrder=descending'
    #url = f'https://export.arxiv.org/api/query?search_query={query2}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending'

    # Define the API endpoint URL
    #url = f"https://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=25"

    #response = urllib.request.urlopen(url)
    #data = response.read()
    #print('data',data.decode('utf-8'))

    # find and modify the value of an element
    ns = {'ns0': 'http://www.w3.org/2005/Atom','ns1':'http://a9.com/-/spec/opensearch/1.1/','ns2':'http://arxiv.org/schemas/atom'} # add more as needed
    #tit=root.find('ns0:title', ns).text

    #url = 'https://export.arxiv.org/api/query?search_query=all:'+query1+'&start='+str(start)+'&max_results='+str(items_per_page)
    #url = 'https://export.arxiv.org/api/query?search_query=all:"'+query1+'"&start=0&max_results='+str(max_results)+'&sortBy=submittedDate&sortOrder=descending'

    print('url::',url)
    #input('jjk')
    url = urllib.parse.quote(url, safe=':/=&?+')
    print('url::2',url)

    try:
        data = urllib.request.urlopen(url, timeout=20).read().decode('utf-8')
        root = ElementTree.fromstring(data)
        entries = root.findall("ns0:entry",ns)
    except (urllib.error.URLError, ElementTree.ParseError, TimeoutError) as e:
        # Malformed query (arXiv 400) or arXiv unavailable: no results rather than a 500.
        print('arXiv search failed', query, e)
        entries = []

    for entry in entries:
        if entry.find("ns0:title",ns) is not None:
            authors = []
            affiliation = []
            title = ""
            link_hp=""
            cat=""
            published=""
            for author in entry.findall("ns0:author",ns):
                authors.append(author.find("ns0:name",ns).text)
                print('test',authors)
                if author.find("ns2:affiliation",ns) is not None:
                    print('aff',author.find("ns2:affiliation",ns).text)
                    affiliation.append(author.find("ns2:affiliation",ns).text)
                else:
                    affiliation.append('')

            link_hp = entry.find("ns0:id",ns).text
            title = entry.find("ns0:title",ns).text
            if entry.find("ns2:primary_category",ns) is not None:
                cat = entry.find("ns2:primary_category",ns).attrib['term']
            if entry.find("ns0:published",ns) is not None:
                published = entry.find("ns0:published",ns).text
                published = datetime.strptime(published, '%Y-%m-%dT%H:%M:%SZ')


            print('link',link_hp)
            pattern1 = r'http://arxiv.org/abs/([\d\.]+v\d)'
            pattern2 = r'http://arxiv.org/abs/([\w\-/]+)'

            match1 = re.search(pattern1, link_hp)
            match2 = re.search(pattern2, link_hp)

            # Check which pattern was matched
            if match1:
                print('Matched pattern 1:', match1.group(1))
                arxiv_id=match1.group(1)
            elif match2:
                print('Matched pattern 2:', match2.group(1))
                arxiv_id=match2.group(1)
            else:
                print('No pattern matched')
                arxiv_id=''

            #arxiv_id = re.search(r'/(\d+\.\d+)', link_hp).group(1)
            #https://arxiv.org/abs/cond-mat/0609158v1
            #http://arxiv.org/abs/1905.06628v1


            search_results.append({'arxiv_id':arxiv_id,'title': title, 'authors': authors, 'link':link_hp,'category':cat,'published':published})


    paginator = Paginator(search_results, items_per_page)
    page_obj = paginator.get_page(page_num)

    #total_pages = math.ceil(total_results / items_per_page)

    context = {'query': query, 'search': search_results,'page_obj': page_obj, 'src': src, 'kind': kind}

    print('page_num',page_num)
    if page_num==1:
        if request.user.is_authenticated:
            #if User.objects.filter(username=user).exists():
            userinst = request.user#User.objects.get(username=user)
            # Do something with the admin_user instance
        else:
            userinst = None# AnonymousUser()

        lang = get_language()

        Search.objects.create(
            query=query,
            user=userinst,
            lang=lang
        )

    return render(request, 'summarizer/search_results.html', context)

OPEN_SEARCH_SECONDS = 6 * 3600  # an OpenAlex search costs; pages and reloads of the same search reuse it


def open_search_results(request, query, kind, page_num):
    """Search of the open-access journals and preprint servers (sources.search)."""
    cache_key = 'open-search:%s' % hashlib.sha256(json.dumps([query.lower(), kind]).encode()).hexdigest()
    results, failed = cache.get(cache_key), False
    if results is None:
        try:
            results = sources.search(query, kind)
            cache.set(cache_key, results, OPEN_SEARCH_SECONDS)
        except sources.SourceUnavailable:
            logger.warning('Open-access search failed for %r', query, exc_info=True)
            results, failed = [], True
    page_obj = Paginator(results, 10).get_page(page_num) if results else None
    if page_num == 1:
        Search.objects.create(query=query, user=request.user if request.user.is_authenticated else None,
                              lang=get_language())
    return render(request, 'summarizer/search_results.html', {
        'query': query, 'search': results, 'page_obj': page_obj, 'src': 'open', 'kind': kind, 'failed': failed})


def summarize(request):
    stuff_for_frontend = {}
    lang = get_language()

    if request.method == 'POST':
        print('request.POST',request.POST)
        arxiv_id = request.POST['arxiv_id']
        arxiv_id = arxiv_id.strip()

        regex_pattern1=r'^\d{4}\.\d{4,5}(v\d+)?$'
        pattern1 = re.compile(regex_pattern1)
        #pattern = 'cond-mat/0609158v1'

        # Regular expression pattern to match the expected format
        regex_pattern2 = r'^[\w\-/]+v\d+$'

        # Compile the regular expression pattern into a regex object
        pattern2 = re.compile(regex_pattern2)

        # a DOI, or a bioRxiv, HAL, OSF, PubMed or Europe PMC link or id: straight to its page
        # an old arXiv id has its archive before a slash; 'hal-01234567v2' or 'dz6ut_v1' only look like one
        arxiv_like = bool(pattern1.match(arxiv_id) or (pattern2.match(arxiv_id) and '/' in arxiv_id))
        found = None
        if not arxiv_like:
            try:
                found = sources.resolve(arxiv_id)
            except sources.SourceUnavailable:
                logger.warning('Could not resolve %r', arxiv_id, exc_info=True)
        if found:
            Search.objects.create(query=arxiv_id[:2000], user=request.user if request.user.is_authenticated else None,
                                  lang=lang)
            kind, value = found
            target = digest.site_id(value) if kind == 'arxiv' else value
            return HttpResponseRedirect(reverse('arxividpage', args=(target,)))

        # Match the pattern against the input string
        if not arxiv_like:
            # Return an error message if the format is incorrect
            #search_results=utils.arxiv_search(arxiv_id)

            print('searchresults',search_results)
            #paginator = Paginator(search_results, 10)  # 10 results per page
            #page_number = request.GET.get('page')
            #page_obj = paginator.get_page(page_number)

            #context = {'query': arxiv_id, 'search': search_results,'page_obj': page_obj}
            query = arxiv_id
            page = 1

            # perform search and pagination logic here

            # redirect to search results view with query and page as query strings
            #return redirect('search_results', q=query, page=page)
            query_params = {'q': query, 'page': page}
            if request.POST.get('src') == 'open':
                query_params['src'] = 'open'
            url = f'/{lang}/search-results/?{urlencode(query_params)}'
            return redirect(url)
            #return redirect('search_results?q={}&page={}'.format(query, page))


            #return render(request, 'summarizer/search_results.html', context)
            #return render(request, 'summarizer/home.html', {'error': 'Invalid arXiv ID format. It should be a four-digit number, a dot, a four or five-digit number, and an optional version number consisting of the letter "v" and one or more digits. For example, 2101.1234, 2101.12345, and 2101.12345v2 are valid identifiers.'})
        else:
            if request.user.is_authenticated:
                #if User.objects.filter(username=user).exists():
                userinst = request.user#User.objects.get(username=user)
                # Do something with the admin_user instance
            else:
                userinst = None# AnonymousUser()

            print('here....')
            Search.objects.create(
                query=arxiv_id,
                user=userinst,
                lang=lang
            )

            if pattern1.match(arxiv_id):
                print('arxiv_id1',arxiv_id)
                if re.search(r'v\d{1,2}$', arxiv_id):
                    print("Arxiv ID is valid.")
                else:
                    # if the arxiv_id does not end with "v1" to "v99", add "v1" to the end
                    arxiv_id += "v1"
                    print("Arxiv ID has been updated to:", arxiv_id)

                return HttpResponseRedirect(reverse('arxividpage', args=(arxiv_id,)))

            if pattern2.match(arxiv_id):
                print('arxiv_id2',arxiv_id)
                #arxiv_id = arxiv_id.replace('/', '%2F')
                #print('arxiv_id3',arxiv_id)
                cat,arxiv_id_old=arxiv_id.split('/')
                print('cat,arxiv',cat,arxiv_id_old)
                return HttpResponseRedirect(reverse('arxividpage', kwargs={'cat': cat,'arxiv_id':arxiv_id_old}))



            #url = reverse('arxividpage', args=['hep-ph/9411346v1'])
            #return HttpResponseRedirect(reverse('arxividpage', args=(arxiv_id,)))

    activated = request.GET.get('activated', False)
    latestpapers=SummaryPaper.objects.filter(lang=lang).exclude(summary__exact='', notes__exact='', lay_summary__exact='', blog__exact='', keywords__exact='', summary__isnull=False, notes__isnull=False, lay_summary__isnull=False, blog__isnull=False, keywords__isnull=False).order_by('-updated')[:7]
    print('latestpapers',latestpapers)

    stuff_for_frontend.update({
        'activated':activated,
        'latestpapers':latestpapers
    })


    return render(request, 'summarizer/home.html', stuff_for_frontend)

def legal(request):
    stuff_for_frontend = {}

    return render(request, "summarizer/legal-notice.html", stuff_for_frontend)

def about(request):
    stuff_for_frontend = {}

    return render(request, "summarizer/about.html", stuff_for_frontend)

def faq(request):
    stuff_for_frontend = {}
    stuff_for_frontend.update ({
        'sofauthor': 'CarbonFreeConf.com',
        'softitle': 'SummarizePaper: AI Powered Article Summarization and Virtual Assistant',
        'sofurl': 'https://summarizepaper.com',
        'sofyear': '2023',
        'sofdate': '2023-03-30',
        'runs_per_day': access.runs_per_day(),
        'chat_anonymous': access.chat_messages_per_day()['anonymous'],
        'chat_free': access.chat_messages_per_day()['user'],
        'chat_paid': access.chat_messages_per_day()['paid'],
        'papers_per_email': digest.PAPERS_PER_EMAIL,
        'max_categories': digest.MAX_CATEGORIES,
    })
    return render(request, "summarizer/faq.html", stuff_for_frontend)


def contact(request):
    stuff_for_frontend = {}

    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        message = request.POST['message']

        # Send an email to the specified email address
        subject = 'Paper Summarization Contact Form: ' + name
        emailto = ['contact@summarizepaper.com']
        #emailto.append(email)
        emailsend = EmailMessage(
            subject,
            message+'\n\n\nFrom: '+name+' ('+email+')',
            'SummarizePaper <communication@summarizepaper.com>',  # from
            emailto,  # to
            # getemails,  # bcc
            # reply_to=replylist,
            headers={'Message-From': 'www.summarizepaper.com'},
        )
        #emailsend.content_subtype = "html"

        sent=emailsend.send(fail_silently=False)

        if sent == 1:
            stuff_for_frontend.update({'sent':1})
        else:
            stuff_for_frontend.update({'sent':0})

            print('Failed to send the email.')

        return render(request, 'summarizer/contact.html',stuff_for_frontend)


    return render(request, "summarizer/contact.html", stuff_for_frontend)

def privacy(request):
    stuff_for_frontend = {}

    return render(request, "summarizer/privacy.html", stuff_for_frontend)

def escape_latex(abstract):
    while "$" in abstract:
        start = abstract.index("$")
        end = abstract.index("$", start + 1)
        abstract = abstract[:start] + "\[" + abstract[start + 1:end] + "\]" + abstract[end + 1:]
    return abstract

def arxiv_metadata_or_none(arxiv_id):
    try:
        if sources.is_external(arxiv_id):
            return sources.metadata_list(arxiv_id)
        return asyncio.run(utils.get_arxiv_metadata(arxiv_id))
    # arXiv sometimes answers with an empty or non-XML body when it is rate limiting
    except (asyncio.TimeoutError, aiohttp.ClientError, ElementTree.ParseError, sources.SourceUnavailable):
        return None


def arxiv_unavailable(request, arxiv_id):
    if sources.is_external(arxiv_id):
        message = _('The source of this paper is not responding right now. Please try again in a moment.')
    else:
        message = 'arXiv is not responding right now. Please try again in a moment.'
    return render(request, 'summarizer/arxividpage.html', {'arxiv_id': arxiv_id, 'errormess': message}, status=503)


def license_display(url):
    """(label shown, whether the full text may be read) of a paper's license."""
    if not url:
        return '', False
    return sources.license_label(url), sources.is_open_license(url)


def arxiv_version_redirect(arxiv_id):
    """For a DOI first opened here: its arXiv page, when there is an arXiv version and the DOI's own
    license does not let us read the text (the arXiv version may, and its page is the one people share)."""
    try:
        paper = sources.metadata(arxiv_id)
    except sources.SourceUnavailable:
        return None
    if paper and paper['arxiv_id'] and not sources.is_open_license(paper['license']):
        return redirect(reverse('arxividpage', kwargs={'arxiv_id': digest.site_id(paper['arxiv_id'])}))
    return None


def tree(request, arxiv_id):
    arxiv_id = arxiv_id.strip()
    lang = get_language()
    li = get_language_info(lang)
    language = li['name']

    stuff_for_frontend = {"arxiv_id": arxiv_id,"language":lang}

    paper=''
    cc_format=''

    if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        print('dejabbbbbbbbbbbb')
        paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

        cc_format, _public = license_display(paper.license)

        alpaper=True
        relpapers=''
        scores=[]
        max_score=0
        if PaperScore.objects.filter(from_paper=paper,active=True).exists():
            relpapers = (PaperScore.objects.filter(from_paper=paper,active=True).order_by('score')
                         .select_related('to_paper').prefetch_related('to_paper__authors'))
            relpapersData = []
            import numpy as np
            import random

            MAX_AUTHORS = 5

            first_author = paper.authors.first()
            author_name = first_author.name if first_author else 'Unknown Author'

            paper_authors = paper.authors.all()
            author_names = ', '.join([author.name for author in paper_authors[:MAX_AUTHORS]])
            if len(paper_authors) > MAX_AUTHORS:
                author_names += '...'

            relpapersData.append({
                    'arxiv_id': paper.arxiv_id,
                    'title': paper.title,
                    'score': 0.2*10,
                    'lang':lang,
                    'distance':0,
                    'x': 0,
                    'y': 0,
                    'first_author': author_name,
                    'authors': author_names,
                    'main':True
                })

            # spread the papers evenly around the main one: searching random free spots took seconds
            start_angle = random.random() * 2 * np.pi
            for k, relpaper in enumerate(relpapers):
                paper_authors = list(relpaper.to_paper.authors.all())
                first_author = min(paper_authors, key=lambda a: a.pk) if paper_authors else None
                author_name = first_author.name if first_author else 'Unknown Author'
                author_names = ', '.join([author.name for author in paper_authors[:MAX_AUTHORS]])
                if len(paper_authors) > MAX_AUTHORS:
                    author_names += ', ...'
                #author_names = ', '.join([author.name for author in paper_authors])
                #author_names = ', '.join([a.name for a in relpaper.to_paper.authors])
                print('aut',author_names)
                distance = 100 * (relpaper.score) + 50.
                # add x and y positions based on distance
                #x = distance * np.cos(random.random() * np.pi * 2)
                #y = distance * np.sin(random.random() * np.pi * 2)
                angle = start_angle + 2 * np.pi * k / len(relpapers)
                x = distance * np.cos(angle)
                y = distance * np.sin(angle)

                relpapersData.append({
                    'arxiv_id': relpaper.to_paper.arxiv_id,
                    'title': relpaper.to_paper.title,
                    'score': 100 * (1 - relpaper.score),
                    'lang':lang,
                    'distance':distance,
                    'x': x,
                    'y': y,
                    'first_author': author_name,
                    'authors': author_names,
                    'main':False
                })

            # Get maximum x and y values
            max_d = max(item['distance'] for item in relpapersData)
            #min_d = min(item['distance'] for item in relpapersData)
            min_d = min(filter(lambda item: item['distance'] != 0, relpapersData), key=lambda item: item['distance'])['distance']
            print('max_d',max_d)
            print('min_d',min_d)

            for item in relpapersData:
                print("item['distance']1",item['distance'])
                if item['distance'] != 0:
                    newdist = ((item['distance'] - min_d+5) / (max_d - min_d+5)) * 100
                    item['x']=item['x']/item['distance']*newdist
                    item['y']=item['y']/item['distance']*newdist
                print("item['distance']2",item['distance'])
            # Renormalize x and y values
            #for item in relpapersData:
            #   item['x'] = (item['x'] / max_x) * 100
            #   item['y'] = (item['y'] / max_y) * 100

            # Convert the list of dictionaries to a JSON string
            relpapers_json = json.dumps(relpapersData)

            max_score = min(rel.score for rel in relpapers)
            #for rel in relpapers:
            #    rel.left_pos = 100 * (relpapers.index(rel) % 10) / 10
            for pap in relpapers:
                scores.append(similar.percent(pap.score))
            stuff_for_frontend.update({
                'relpapers':relpapers,
                'max_score':str(max_score*10),
                'relpapers_json':relpapers_json,
                'scores':scores,
            })

    else:
        alpaper=False
        arxiv_detailsf=arxiv_metadata_or_none(arxiv_id)
        if arxiv_detailsf is None:
            return arxiv_unavailable(request, arxiv_id)
        exist=arxiv_detailsf[0]
        #exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments
        if arxiv_detailsf[0] == 0:
            stuff_for_frontend.update({
                'exist':exist,
            })
        else:
            arxiv_details=arxiv_detailsf[1:]

            #[authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published_datetime, journal_ref, comments]
            keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments', 'license']

            arxiv_dict = dict(zip(keys, arxiv_details))
            print('jfgh',arxiv_detailsf)
            published_datetime = datetime.strptime(arxiv_dict['published_arxiv'], '%Y-%m-%dT%H:%M:%SZ')

            arxiv_dict['published_arxiv']=published_datetime

            url = arxiv_dict['license']
            cc_format=''
            license=''
            if url != '':
                parts = url.split('/')
                print('parts',parts)
                license = parts[-3]
                version = parts[-2]
                if license.upper() != "NONEXCLUSIVE-DISTRIB":
                    cc_format = 'CC ' + license.upper() + ' ' + version
                else:
                    cc_format = license.upper() + ' ' + version

            public=False
            #print('lo',license.upper())
            if (license.upper().strip() == "BY" or license.upper().strip() == "BY-SA" or license.upper().strip() == "BY-NC-SA" or license.upper().strip() == "ZERO"):
                public=True
                print('pub2')

            print(cc_format) # Output: CC BY-NC-SA 4.0
            paper=arxiv_dict


    stuff_for_frontend.update({
        'alpaper':alpaper,
        'cc_format':cc_format,
        'paper':paper,
    })

    return render(request, "summarizer/tree.html", stuff_for_frontend)

def create_embed(request):
    stuff_for_frontend = {}

    if request.method == 'POST':
        paper_id = request.POST.get('paper_id')
        print('embed',paper_id)
        
        # Create embeddings for paper with the given ID
        return JsonResponse({'paper_id': paper_id})

    stuff_for_frontend.update({
        'test':1,
    })

    return redirect(reverse('home'))


ARXIV_API_URL = 'https://export.arxiv.org/api/query'


ARXIV_ID_FULL = re.compile(r'^(\d{4}\.\d{4,5}|[a-z\-]+(\.[A-Z]{2})?/\d{7})(v\d+)?$', re.I)


def open_feed(request, search_query, id_list):
    """The other sources in arXiv's Atom form (sources.atom_feed): a search of the open-access journals and
    preprints, or one paper by DOI, key or link."""
    try:
        if id_list:
            found = sources.resolve(id_list) if not sources.is_external(id_list.lower()) else ('key', id_list.lower())
            if found and found[0] == 'arxiv':
                return None  # an arXiv DOI: arXiv answers
            paper = sources.metadata(found[1]) if found and sources.valid_key(found[1]) else None
            papers = [paper] if paper else []
        else:
            words = re.sub(r'\b(all|ti|au|abs|cat|id):', ' ', search_query).replace(' AND ', ' ')
            words = ' '.join(re.sub(r'[()"]', ' ', words).split())
            papers = sources.search_papers(words, per_page=10) if words else []
    except sources.SourceUnavailable:
        return HttpResponse('The journal search is not reachable right now.', status=502, content_type='text/plain')
    return HttpResponse(sources.atom_feed(papers), content_type='application/atom+xml; charset=utf-8')


def arxiv_api(request):
    """Relay the chat page's arXiv lookups: arXiv sends no CORS header, so browsers block direct calls.

    With src=open, or an identifier that is not arXiv's (a DOI, a HAL or PubMed link...), the answer comes
    from the other sources, in the same Atom form."""
    search_query = request.GET.get('search_query', '').strip()[:300]
    id_list = request.GET.get('id_list', '').strip()[:300]
    if bool(search_query) == bool(id_list):
        return HttpResponse('Give either search_query or id_list.', status=400, content_type='text/plain')
    if id_list and ARXIV_ID_FULL.match(id_list.replace('--', '/')):
        id_list = id_list.replace('--', '/')  # our URL form of old ids, from a collection's "chat with these"
    if request.GET.get('src') == 'open' or (id_list and not ARXIV_ID_FULL.match(id_list)):
        answer = open_feed(request, search_query, id_list)
        if answer is not None:
            return answer
        id_list = sources.ARXIV_DOI_RE.match(sources.clean_doi(id_list)).group(1)
    if id_list:
        params = {'id_list': id_list}
    else:
        try:
            max_results = min(max(int(request.GET.get('max_results', 10)), 1), 20)
        except ValueError:
            max_results = 10
        params = {'search_query': search_query, 'max_results': max_results}

    cache_key = 'arxiv-api:' + hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()
    body = cache.get(cache_key)
    if body is None:
        try:
            r = requests.get(ARXIV_API_URL, params=params, timeout=15)
            r.raise_for_status()
        except requests.RequestException:
            return HttpResponse('arXiv is not reachable right now.', status=502, content_type='text/plain')
        body = r.content
        cache.set(cache_key, body, 600)
    return HttpResponse(body, content_type='application/atom+xml; charset=utf-8')


def chat(request):
    stuff_for_frontend = {}
    lang = get_language()
    li = get_language_info(lang)
    language = li['name']


    if 'ON_HEROKU' in os.environ:
        onhero=True
    else:
        onhero=False

    max_nb_paper = 10

    client_ip = utils.get_client_ip(request)#request.META['REMOTE_ADDR']
    #print('clientip',client_ip)
    #print('clientip2',client_ip.encode('utf-8'))

    # Check if this IP address has already voted on this post
    hashed_ip_address = hashlib.sha256(client_ip.encode('utf-8')).hexdigest()
    
    stuff_for_frontend.update({
        'onhero':onhero,
        'language':lang,
        'hashed_ip_address':hashed_ip_address,
        'max_nb_paper':max_nb_paper,
    })

    return render(request, "summarizer/chat.html", stuff_for_frontend)

def paper_structured_data(paper, sumpaper, lang):
    """schema.org JSON-LD for a paper page, already escaped for a <script> tag."""
    arxiv_url = sources.ref_url(paper.arxiv_id)
    title = re.sub(r'\s+', ' ', paper.title or paper.arxiv_id).strip()
    data = {
        '@context': 'https://schema.org',
        '@type': 'ScholarlyArticle',
        'headline': title[:110],
        'name': title,
        'identifier': sources.label(paper.arxiv_id).split(': ', 1)[-1],
        'url': 'https://www.summarizepaper.com/%s/arxiv-id/%s/' % (lang, paper.arxiv_id),
        'sameAs': arxiv_url,
        'inLanguage': lang,
        'author': [{'@type': 'Person', 'name': a.name} for a in paper.authors.all()[:20]],
    }
    if paper.published_arxiv:
        data['datePublished'] = paper.published_arxiv.isoformat()
    if paper.category:
        data['about'] = paper.category
    if sources.is_external(paper.arxiv_id) and paper.journal_ref:
        data['isPartOf'] = {'@type': 'Periodical', 'name': paper.journal_ref}
    if sumpaper and sumpaper.summary:
        data['abstract'] = sumpaper.summary.strip()
    elif paper.abstract:
        data['abstract'] = re.sub(r'\s+', ' ', paper.abstract).strip()
    if sumpaper and sumpaper.keywords:
        data['keywords'] = sumpaper.keywords
    escaped = json.dumps(data).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return mark_safe(escaped)

def citation_meta(paper):
    """Highwire Press tags, read by the Zotero, Mendeley and Paperpile browser buttons."""
    ref = re.sub(r'v\d+$', '', paper.arxiv_id.replace('--', '/'))
    tags = [('citation_title', ' '.join((paper.title or '').split()))]
    tags += [('citation_author', a.name) for a in paper.authors.all()[:100]]
    if paper.published_arxiv:
        tags.append(('citation_publication_date', paper.published_arxiv.strftime('%Y/%m/%d')))
    if sources.is_external(paper.arxiv_id):
        if paper.journal_ref:
            tags.append(('citation_journal_title', paper.journal_ref))
    else:
        tags += [('citation_arxiv_id', ref), ('citation_publisher', 'arXiv'),
                 ('citation_pdf_url', 'https://arxiv.org/pdf/%s' % ref)]
    if paper.abstract:
        tags.append(('citation_abstract', ' '.join(paper.abstract.split())))
    if paper.link_doi and 'doi.org' in paper.link_doi:
        tags.append(('citation_doi', paper.link_doi.split('doi.org/')[-1]))
    return tags


def waiting_details(arxiv_id):
    """Title, authors and abstract shown while a summary is being written, when we already know them."""
    announced = AnnouncedPaper.objects.filter(arxiv_id=arxiv_id).first()
    if announced:
        authors = json.loads(announced.authors or '[]')
        return {'title': announced.title, 'abstract': announced.abstract,
                'authors': ', '.join(authors[:8]) + (' et al.' if len(authors) > 8 else '')}
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id).first()
    if paper and paper.title:
        return {'title': paper.title, 'abstract': paper.abstract or '', 'authors': ''}
    return None


def arxividpage(request, arxiv_id, error_message=None, cat=None):
    arxiv_id = arxiv_id.strip()
    

    print('cat',cat)
    print('id',arxiv_id)
    print('err',error_message)
    if cat is not None:
        arxiv_id=cat+'--'+arxiv_id
        print('comp',arxiv_id)

    external = sources.is_external(arxiv_id.lower())
    if external and arxiv_id != arxiv_id.lower():
        return redirect(reverse('arxividpage', kwargs={'arxiv_id': arxiv_id.lower()}))

    if not external and not re.search(r'v\d{1,2}$', arxiv_id):
        arxiv_id += "v1"
        #arxiv_id=arxiv_id.replace('--','/')
        # redirect to the new URL
        return redirect(reverse('arxividpage', kwargs={'arxiv_id': arxiv_id}))

    #if '--' in arxiv_id:
        #arxiv_id=arxiv_id.replace('--','/')
        #return redirect(reverse('arxividpage', kwargs={'arxiv_id': arxiv_id}))

    if 'ON_HEROKU' in os.environ:
        onhero=True
    else:
        onhero=False

    #import sys
    #print('syspath',sys.path)

    #print('base dir',settings.BASE_DIR)
    lang = get_language()
    li = get_language_info(lang)
    language = li['name']
    #input("Press Enter to continue...")

    client_ip = utils.get_client_ip(request)#request.META['REMOTE_ADDR']
    print('clientip',client_ip)
    # Check if this IP address has already voted on this post
    hashed_ip_address = hashlib.sha256(client_ip.encode('utf-8')).hexdigest()

    paid = access.is_paid(request.user)

    stuff_for_frontend = {"arxiv_id": arxiv_id,"onhero":onhero,"language":lang,"hashed_ip_address":hashed_ip_address,'paid':paid}

    print('jk')
    regex_pattern1=r'^\d{4}\.\d{4,5}(v\d+)?$'
    pattern1 = re.compile(regex_pattern1)
    #pattern = 'cond-mat/0609158v1'

    # Regular expression pattern to match the expected format
    regex_pattern2 = r'^[\w\-/]+v\d+$'

    # Compile the regular expression pattern into a regex object
    pattern2 = re.compile(regex_pattern2)

    # Match the pattern against the input string

    #pattern = re.compile(r'^\d{4}\.\d{4,5}(v\d+)?$')
    #if not pattern.match(arxiv_id):
    if external and not sources.valid_key(arxiv_id):
        stuff_for_frontend.update({'errormess': _('Wrong url: this is not a paper identifier we know.')})
        return render(request, "summarizer/arxividpage.html", stuff_for_frontend, status=404)

    if not external and not pattern1.match(arxiv_id) and not pattern2.match(arxiv_id):

        print('not')
        errormess='Wrong url. '+arxiv_id+' is an invalid arXiv ID format. It should be a four-digit number, a dot, a four or five-digit number, and an optional version number consisting of the letter "v" and one or more digits. For example, 2101.1234, 2101.12345, and 2101.12345v2 are valid identifiers.'
        # Return an error message if the format is incorrect
        #return render(request, 'summarizer/home.html', {'error': 'Invalid arXiv ID format. It should be a four-digit number, a dot, a four or five-digit number, and an optional version number consisting of the letter "v" and one or more digits. For example, 2101.1234, 2101.12345, and 2101.12345v2 are valid identifiers.'})
        stuff_for_frontend.update({
            'errormess':errormess,
        })
        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)

    page_running = cache.get(utils.run_group(arxiv_id, lang))
    print('page_running',page_running)
    if page_running:
        print('in page_running')
        # Page is already running, do not start it again
        stuff_for_frontend.update({
            'run2':True,
            'waiting':waiting_details(arxiv_id),
        })
        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)


    #lang = get_language()
    #print('lang',lang)
    #if lang =='fr':
    if error_message:
        print('rrr',error_message)
        if error_message=="vote":
            error_message=_('You have already voted')

        else:
            error_message=""

        stuff_for_frontend.update({
            'error_message':error_message,
        })

    if request.method == 'POST':
        print('in here',request.POST)
        if 'close_button' in request.POST:
            print('ok close')
            tree = request.POST.get('tree', '1')
            print('tree',tree)
            access.grant_run(request.session, arxiv_id, 'close')
            stuff_for_frontend.update({
                'close':True,
                'treeval':tree
            })

        if 'download_pdf' in request.POST:
            print('download')
            #pdf_bytes=utils.summary_pdf(arxiv_id,lang)
            local_date = request.POST.get('local_date')
            print('loloooooooooo',local_date)
            res=utils.generate_pdf(request,arxiv_id,lang,local_date)
            #response = HttpResponse(pdf_bytes, content_type="application/pdf")
            #response['Content-Disposition'] = 'attachment; filename=%s' % filename  # force browser to download file

            #response = HttpResponse(bytes(pdf_bytes), content_type='application/pdf')
            client_ip = utils.get_client_ip(request)#request.META['REMOTE_ADDR']
            print('clientip',client_ip)
            # Check if this IP address has already voted on this post
            hashed_ip_address = hashlib.sha256(client_ip.encode('utf-8')).hexdigest()
            if request.user.is_authenticated:
                #if User.objects.filter(username=user).exists():
                userinst = request.user#User.objects.get(username=user)
                # Do something with the admin_user instance
            else:
                userinst = None# AnonymousUser()

            PDFHistory.objects.create(
                arxiv_id=arxiv_id,
                user=userinst,
                lang=lang,
                ip_address=hashed_ip_address
            )

            '''
            response = HttpResponse(content_type="application/pdf")
            filename="SummarizePaper-"+str(arxiv_id)+".pdf"
            response['Content-Disposition'] = 'attachment; filename=%s' % filename  # force browser to download file
            #response.write(resp)
            #print('sumpdf',sumpdf)
            #from io import BytesIO

            #buffer = BytesIO(pdf_bytes)
            #response.write(buffer.getvalue())
            #print('pdf_bytes',pdf_bytes)
            response.write(pdf_bytes)
            '''
            return res#response

        if 'run_button' in request.POST:
            print('ok run')
            hashed_ip_address = access.hash_ip(access.client_ip(request))
            allowed, refusal = access.run_quota(request.user, hashed_ip_address)

            if allowed:

                if request.user.is_authenticated:
                    #if User.objects.filter(username=user).exists():
                    userinst = request.user#User.objects.get(username=user)
                    # Do something with the admin_user instance
                else:
                    userinst = None# AnonymousUser()

                PaperHistory.objects.create(
                    arxiv_id=arxiv_id,
                    user=userinst,
                    lang=lang,
                    ip_address=hashed_ip_address
                )

                if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
                    print('deja')
                    paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]
                    #if SummaryPaper.objects.filter(paper=paper,lang=lang).exists():
                    #    sumpaper=SummaryPaper.objects.filter(paper=paper,lang=lang)[0]

                    previous_votes = Vote.objects.filter(paper=paper,lang=lang)

                    if SummaryPaper.objects.filter(paper=paper,lang=lang).exists() and previous_votes.exists():
                        #we cancel the votes because we rerun the process
                        print('rerun cancel votes')
                        for pv in previous_votes:
                            pv.active=False
                            pv.save()
                        #paper.total_votes = 0
                        #paper.save()

                #arxiv_id=arxiv_id.replace('--','/')

                access.grant_run(request.session, arxiv_id, 'run')
                stuff_for_frontend.update({
                    'run':True,
                })
            else:
                #refusal is 'robot' (runs too close together) or 'toomany' (daily limit)
                stuff_for_frontend.update({
                    refusal:True,
                })
        print('renderrrrrrrrrrrrr')
        #return redirect(reverse('arxividpage', kwargs={'arxiv_id': arxiv_id,'stuff_for_frontend':stuff_for_frontend}))

        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)
        

    else:
        print('arxivjjj',arxiv_id)
        if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
            print('dejatttt')
            paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]
            sumpaper=''
            sumlang=''
            if SummaryPaper.objects.filter(paper=paper,lang=lang).exists():
                sumpaper=SummaryPaper.objects.filter(paper=paper,lang=lang)[0]
            if SummaryPaper.objects.filter(paper=paper).exclude(lang=lang).exists():
                sumlang=SummaryPaper.objects.filter(paper=paper).exclude(lang=lang).values_list('lang',flat=True)
                print('sumlang',list(sumlang))
                sumlang=list(sumlang)

            chathistory=''
            if request.user.is_authenticated:
                print('authent')
                if AIassistant.objects.filter(arxiv_id=arxiv_id,active=True,user=request.user).exists():
                    print('exist AI assistant')
                    chathistory=AIassistant.objects.filter(arxiv_id=arxiv_id,active=True,user=request.user).order_by('created')

            alpaper=True
            print('paper',paper.abstract)
            total_votes=0

            if Vote.objects.filter(paper=paper,lang=lang).exists():
                nbvotes=Vote.objects.filter(paper=paper,lang=lang,active=True).aggregate(Sum('vote'))
                print('nbvotes',nbvotes)
                if nbvotes['vote__sum'] != None:
                    total_votes=nbvotes['vote__sum']
                print('totvotes',total_votes)

            #updated = timezone.make_aware(paper.updated)
            #if paper.updated >= (timezone.now() - timezone.timedelta(minutes=1)):
            if paper.updated >= (timezone.now() - timezone.timedelta(days=365)):
                toolong=False

                # code to run if the paper was updated in the last year
                print('1 days a')

            else:
                toolong=True
                # code to run if the paper was not updated in the last year
                print('1 days b')

            cc_format, public = license_display(paper.license)

            print('cc',cc_format) # Output: CC BY-NC-SA 4.0

            #paper.abstract = escape_latex(paper.abstract)
            notes=''
            kw=''
            keywords=''
            notes2=''
            if sumpaper:
                if (sumpaper.notes is not None) and (sumpaper.notes != "") and (sumpaper.notes != "['']") and (sumpaper.notes != 'Error: needs to be re-run'):
                    print('nnnnnnoooottees',sumpaper.notes)
                    try:
                        notes = ast.literal_eval(sumpaper.notes)
                        notes2=[]
                        for note in notes:
                            noteb=note.replace('•', '').strip()
                            if noteb.startswith("-"):
                                noteb = noteb[1:]
                            notes2.append(noteb)
                        #notes2 = [note.replace('•', '') for note in notes]

                    except ValueError:
                        # Handle the error by returning a response with an error message to the user
                        return HttpResponse("Invalid input: 'notes' attribute is not a valid Python literal.")
                else:
                    notes=['Error: needs to be re-run']
                    notes2=''

                if (sumpaper.keywords is not None) and (sumpaper.keywords != "") and (sumpaper.keywords != "['']"):
                    print('keywords',sumpaper.keywords)
                    try:
                        keywords_str = sumpaper.keywords.strip()  # Remove any leading or trailing whitespace
                        print('keystr',keywords_str)
                        keywords_list = [keyword.strip() for keyword in keywords_str.split(',')]  # Split the keywords string into a list
                        print('keylist',keywords_list)
                        keywords_repr = json.dumps(keywords_list)
                        keywords = json.loads(keywords_repr)
                        #keywords_repr = ", ".join([f"'{keyword}'" for keyword in keywords_list])  # Enclose each keyword in quotes and join the list with commas
                        #keywords_repr = ", ".join([fr"r'{keyword}'" for keyword in keywords_list])
                        print('keyrepr',keywords_repr)
                        #keywords = ast.literal_eval('[' + keywords_repr + ']')  # Evaluate the resulting string as a Python list
                        #keywords = ast.literal_eval(keywords_list)  # Evaluate the resulting string as a Python list

                        #keywords_list = [keyword.replace('"', '\\"').strip() for keyword in keywords_str.split(',')]  # Replace double quotes with escaped double quotes
                        #keywords_repr = ", ".join([f'"{keyword}"' for keyword in keywords_list])  # Enclose each keyword in double quotes and join the list with commas
                        #keywords = ast.literal_eval('[' + keywords_repr + ']')  # Evaluate the resulting string as a Python list


                        #keywords = ast.literal_eval('['+sumpaper.keywords+']')
                        print('kegggg',keywords)
                    except ValueError:
                        # Handle the error by returning a response with an error message to the user
                        return HttpResponse("Invalid input: 'keywords' attribute is not a valid Python literal.")
                else:
                    keywords=''

            #closest_papers=asyncio.run(utils.findclosestpapers(arxiv_id,lang,settings.OPENAI_KEY))
            #print('closest_papers',closest_papers)
            relpapers=''
            scores=[]
            max_score=0
            MAX_REL_PAPERS=7
            if PaperScore.objects.filter(from_paper=paper,active=True).exists():
                relpapers = PaperScore.objects.filter(from_paper=paper,active=True).order_by('score')[:MAX_REL_PAPERS]
                max_score = min(rel.score for rel in relpapers)
                #for rel in relpapers:
                #    rel.left_pos = 100 * (relpapers.index(rel) % 10) / 10
                for pap in relpapers:
                    scores.append(similar.percent(pap.score))

            stuff_for_frontend.update({
                'paper':paper,
                'structured_data':paper_structured_data(paper, sumpaper, lang),
                'relpapers':relpapers,
                'max_score':max_score,
                'scores':scores,
                'keywords':keywords,
                'sumpaper':sumpaper,
                'sumlang':sumlang,
                'notes':notes,
                'notes2':notes2,
                'cc_format':cc_format,
                'toolong':toolong,
                'public':public,
                'total_votes':total_votes,
                'chathistory':chathistory,
                'my_collections':library.paper_collections(request.user, arxiv_id),
                'following':citations.is_following(request.user, arxiv_id),
                'reading_levels':reading.LEVELS,
                'reading_langs':reading.OUTPUT_LANGUAGES,
                'reading_left':reading.quota_left(request),
                'citation_meta':citation_meta(paper),
            })
        else:
            print('nope')
            alpaper=False
            if external:
                to_arxiv = arxiv_version_redirect(arxiv_id)
                if to_arxiv:
                    return to_arxiv
            arxiv_detailsf=arxiv_metadata_or_none(arxiv_id)
            if arxiv_detailsf is None:
                return arxiv_unavailable(request, arxiv_id)
            exist=arxiv_detailsf[0]
            #exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments
            if arxiv_detailsf[0] == 0:
                stuff_for_frontend.update({
                    'exist':exist,
                })
            else:


                arxiv_details=arxiv_detailsf[1:]

                #[authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published_datetime, journal_ref, comments]
                keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments', 'license']

                arxiv_dict = dict(zip(keys, arxiv_details))
                print('jfgh',arxiv_detailsf)
                published_datetime = datetime.strptime(arxiv_dict['published_arxiv'], '%Y-%m-%dT%H:%M:%SZ')

                arxiv_dict['published_arxiv']=published_datetime

                cc_format, public = license_display(arxiv_dict['license'])

                print(cc_format) # Output: CC BY-NC-SA 4.0

                stuff_for_frontend.update({
                    'exist':exist,
                    'public':public,
                    'details':arxiv_dict,
                    'cc_format':cc_format
                })

        stuff_for_frontend.update({
            'alpaper':alpaper,
        })

        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)



def vote(request, paper_id):
    lang = get_language()
    # The paper page posts with fetch and updates the score in place; without
    # JavaScript the form posts normally and we redirect back to the page.
    wants_json = request.headers.get('x-requested-with') == 'fetch'

    if request.method == 'POST':
        direction = request.POST.get('direction')
        paper = get_object_or_404(ArxivPaper, arxiv_id=paper_id)
        hashed_ip_address = access.hash_ip(utils.get_client_ip(request))

        already = Vote.objects.filter(paper=paper, lang=lang, ip_address=hashed_ip_address).exists()
        valuevote = {'up': 1, 'down': -1}.get(direction, 0)
        if not already and valuevote != 0:
            userinst = request.user if request.user.is_authenticated else None
            Vote.objects.create(paper=paper, lang=lang, ip_address=hashed_ip_address, vote=valuevote, user=userinst)

        if wants_json:
            total = Vote.objects.filter(paper=paper, lang=lang, active=True).aggregate(Sum('vote'))['vote__sum'] or 0
            return JsonResponse({
                'total': total,
                'message': _('You have already voted') if already else _('Thank you for your vote'),
                'already': already,
            })
        if already:
            return redirect('/' + lang + '/arxiv-id/' + paper_id + '/vote#totvote')

    return redirect('arxividpage', arxiv_id=paper_id)

def history(request):
    stuff_for_frontend={}

    paper_history=[]
    if request.user.is_authenticated:
        #access user table
        auth=True
        history=PaperHistory.objects.filter(user=request.user).order_by('-created')
        print('history',history)
        for h in history:
            arxiv_paper = ArxivPaper.objects.filter(arxiv_id=h.arxiv_id).first()
            paperdate = h.created#.strftime('%d/%m/%Y')

            if arxiv_paper:
                paper_history.append({'arxiv_id': h.arxiv_id, 'title': arxiv_paper.title, 'date': paperdate})
            else:
                paper_history.append({'arxiv_id': h.arxiv_id, 'date': paperdate})
    else:
        auth=False

    stuff_for_frontend.update({
        'auth':auth,
        'paper_history':paper_history
    })

    return render(request, "summarizer/history.html", stuff_for_frontend)


def update_cache(request):
    #clears the "summary already running" lock of a paper, staff only
    if not request.user.is_staff:
        raise PermissionDenied
    arxiv_id = request.GET.get('arxiv_id')
    for language in ('en', 'fr'):
        cache.set(utils.run_group(arxiv_id, language), False)
    return HttpResponse("Cache updated")


def bibtex(request, arxiv_id):
    """BibTeX entry of a paper in the format arXiv itself suggests."""
    get_object_or_404(ArxivPaper, arxiv_id=arxiv_id)
    meta = exports.paper_meta(arxiv_id)
    response = HttpResponse(exports.bibtex_entry(meta), content_type='text/plain; charset=utf-8')
    response['Content-Disposition'] = 'inline; filename="%s.bib"' % exports.file_name(arxiv_id)
    return response
