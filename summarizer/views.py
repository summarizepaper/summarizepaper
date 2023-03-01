# Create your views here.
from django.shortcuts import render, redirect, HttpResponseRedirect, get_object_or_404
from django.urls import reverse
from django.http import HttpResponse
import requests
import time
import re
import hashlib
from .models import ArxivPaper, Vote, PaperHistory, SummaryPaper, AIassistant
from .forms import RegistrationForm
from django.conf import settings
from django.core.mail import send_mail, EmailMessage
import summarizer.utils as utils
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
from django.contrib.auth.tokens import PasswordResetTokenGenerator
from django.contrib.auth import authenticate
from django.core.exceptions import PermissionDenied
from django.contrib.auth import views as auth_views
from django.urls import reverse_lazy
from django.core.exceptions import ValidationError
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth import logout
import os
from django.utils.translation import get_language,get_language_info
from django.db.models import Sum
from django.core.paginator import Paginator
from urllib.parse import urlencode
import urllib, urllib.request
from xml.etree import ElementTree

class CustomAuthenticationForm(AuthenticationForm):
    def clean_username(self):
        print('clean')
        username = self.cleaned_data.get('username')
        password = self.cleaned_data.get('password')
        user = authenticate(username=username, password=password)

        print('user',user)
        if user is not None and not user.is_active:
            print('activate brah')
            raise ValidationError("Your account is not activated yet.")
        return username

class CustomLoginView(auth_views.LoginView):
    authentication_form = CustomAuthenticationForm

    def dispatch(self, request, *args, **kwargs):
        print('dis')
        try:
            print('ok')
            return super().dispatch(request, *args, **kwargs)
        except PermissionDenied as e:
            print('pasok')
            messages.error(self.request, str(e))
            return self.form_invalid(self.get_form())

    def form_valid(self, form):
        print('not perm a')

        username = form.cleaned_data['username']
        password = form.cleaned_data['password']
        try:
            user = authenticate(username=username, password=password)
        except Exception as e:
            print(e)
            return self.form_invalid(form)
        #user = authenticate(username=username, password=password)
        print('not perm')

        if user is not None and not user.is_active:
            print('perm')
            raise PermissionDenied("Your account is not activated yet.")
        try:
            return super().form_valid(form)
        except PermissionDenied as e:
            print('ret')
            return self.render_to_response(
                self.get_context_data(form=form, permission_denied=str(e))
            )
        #return super().form_valid(form)

    def get_success_url(self):
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
            return redirect('summarize/?activated=True')
        return render(request, 'account_activation_invalid.html')


class RegisterView(View):
    def get(self, request):
        form = RegistrationForm()
        return render(request, 'register.html', {'form': form})

    def post(self, request):
        form = RegistrationForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            if User.objects.filter(username=username).exists():
                form.add_error('username', 'This username is already taken')
                return render(request, 'register.html', {'form': form})

            email = form.cleaned_data.get('email')
            password1 = form.cleaned_data.get('password1')
            password2 = form.cleaned_data.get('password2')

            if password1 and password2 and password1 != password2:
                form.add_error(None, "Passwords don't match")
                return render(request, 'register.html', {'form': form})

            user = User.objects.create_user(username=username, email=email, password=password1)
            user.is_active = False
            user.save()

            current_site = get_current_site(request)
            mail_subject = 'Activate your account.'
            print('user.pk',user.pk)
            uid=urlsafe_base64_encode(force_bytes(user.pk))
            token=generate_token.make_token(user)
            print('uid1',uid)
            print('token1',token)
            print('current_site.domain',current_site.domain)
            message = render_to_string('acc_active_email.html', {
                'user': user,
                'domain': current_site.domain,
                'uid': uid,#urlsafe_base64_encode(force_bytes(user.pk)).decode(),
                'token': token,
            })
            to_email = email
            email = EmailMessage(mail_subject, message, to=[to_email])
            email.content_subtype = "html"

            email.send()

            return redirect('account_activation_sent')

        return render(request, 'register.html', {'form': form})


def logout_view(request):
    logout(request)
    # Redirect the user to the login page or any other page of your choice
    return redirect('login')

def search_results(request):
    query = request.GET.get('q')
    page_num = request.GET.get('page', 1)

    # perform search and pagination logic here
    search_results = []
    #total_results = 0
    items_per_page = 10
    #start = (int(page_num) - 1) * items_per_page
    max_results=25
    print('query',query)

    query1 = urllib.parse.quote(query)
    print('query1',query1)
    # Define the API endpoint URL
    #url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=25"

    #response = urllib.request.urlopen(url)
    #data = response.read()
    #print('data',data.decode('utf-8'))

    # find and modify the value of an element
    ns = {'ns0': 'http://www.w3.org/2005/Atom','ns1':'http://a9.com/-/spec/opensearch/1.1/','ns2':'http://arxiv.org/schemas/atom'} # add more as needed
    #tit=root.find('ns0:title', ns).text

    #url = 'http://export.arxiv.org/api/query?search_query=all:'+query1+'&start='+str(start)+'&max_results='+str(items_per_page)
    url = 'http://export.arxiv.org/api/query?search_query=all:"'+query1+'"&start=0&max_results='+str(max_results)+'&sortBy=submittedDate&sortOrder=descending'

    print('url',url)

    data = urllib.request.urlopen(url).read().decode('utf-8')
    #print('data',data)

    root = ElementTree.fromstring(data)

    for entry in root.findall("ns0:entry",ns):
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

    context = {'query': query, 'search': search_results,'page_obj': page_obj}

    return render(request, 'summarizer/search_results.html', context)

def summarize(request):
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

        # Match the pattern against the input string
        if not pattern1.match(arxiv_id) and not pattern2.match(arxiv_id):
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
            url = f'/search_results/?{urlencode(query_params)}'
            return redirect(url)
            #return redirect('search_results?q={}&page={}'.format(query, page))


            #return render(request, 'summarizer/search_results.html', context)
            #return render(request, 'summarizer/home.html', {'error': 'Invalid arXiv ID format. It should be a four-digit number, a dot, a four or five-digit number, and an optional version number consisting of the letter "v" and one or more digits. For example, 2101.1234, 2101.12345, and 2101.12345v2 are valid identifiers.'})
        else:
            if pattern1.match(arxiv_id):
                print('arxiv_id1',arxiv_id)
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

    return render(request, 'summarizer/home.html', {'activated':activated})

def legal(request):
    stuff_for_frontend = {}

    return render(request, "summarizer/legal-notice.html", stuff_for_frontend)

def about(request):
    stuff_for_frontend = {}

    return render(request, "summarizer/about.html", stuff_for_frontend)

def faq(request):
    stuff_for_frontend = {}

    return render(request, "summarizer/faq.html", stuff_for_frontend)


def contact(request):
    stuff_for_frontend = {}

    if request.method == 'POST':
        name = request.POST['name']
        email = request.POST['email']
        message = request.POST['message']

        # Send an email to the specified email address
        subject = 'Paper Summarization Contact Form: ' + name
        emailto = ['carbonfreeconf@gmail.com']
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

def arxividpage(request, arxiv_id, error_message=None, cat=None):
    arxiv_id = arxiv_id.strip()

    print('cat',cat)
    print('id',arxiv_id)
    print('err',error_message)
    if cat is not None:
        arxiv_id=cat+'--'+arxiv_id
        print('comp',arxiv_id)

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


    stuff_for_frontend = {"arxiv_id": arxiv_id,"onhero":onhero,"language":lang}

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
    if not pattern1.match(arxiv_id) and not pattern2.match(arxiv_id):

        print('not')
        errormess='Wrong url. '+arxiv_id+' is an invalid arXiv ID format. It should be a four-digit number, a dot, a four or five-digit number, and an optional version number consisting of the letter "v" and one or more digits. For example, 2101.1234, 2101.12345, and 2101.12345v2 are valid identifiers.'
        # Return an error message if the format is incorrect
        #return render(request, 'summarizer/home.html', {'error': 'Invalid arXiv ID format. It should be a four-digit number, a dot, a four or five-digit number, and an optional version number consisting of the letter "v" and one or more digits. For example, 2101.1234, 2101.12345, and 2101.12345v2 are valid identifiers.'})
        stuff_for_frontend.update({
            'errormess':errormess,
        })
        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)

    page_running = cache.get("ar_"+arxiv_id)
    print('page_running',page_running)
    if page_running:
        print('in page_running')
        # Page is already running, do not start it again
        stuff_for_frontend.update({
            'run2':True,
        })
        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)


    #lang = get_language()
    #print('lang',lang)
    #if lang =='fr':
    if error_message:
        print('rrr',error_message)
        if error_message=="vote":
            if lang =='fr':
                error_message="Vous avez déjà voté"
            else:
                error_message="You have already voted"

        else:
            error_message=""

        stuff_for_frontend.update({
            'error_message':error_message,
        })

    if request.method == 'POST':
        print('in here',request.POST)
        if 'download_pdf' in request.POST:
            print('download')
            resp=utils.summary_pdf(arxiv_id,lang)
            response = HttpResponse(content_type="application/pdf")
            filename="SummarizePaper-"+str(arxiv_id)+".pdf"
            response['Content-Disposition'] = 'attachment; filename=%s' % filename  # force browser to download file
            response.write(resp)
            #print('sumpdf',sumpdf)
            return response

        if 'run_button' in request.POST:
            print('ok run')
            if request.user.is_authenticated:
                hist, created = PaperHistory.objects.update_or_create(
                    arxiv_id=arxiv_id,
                    user=request.user,
                    defaults={}
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

            stuff_for_frontend.update({
                'run':True,
            })
        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)
    else:
        if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
            print('deja')
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

            url = paper.license
            cc_format=''
            license = ''
            if url != '' and url != None:
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
                print('pub')

            print('cc',cc_format) # Output: CC BY-NC-SA 4.0

            #paper.abstract = escape_latex(paper.abstract)
            notes=''
            kw=''
            if sumpaper:
                if (sumpaper.notes is not None) and (sumpaper.notes != "") and (sumpaper.notes != "['']") and (sumpaper.notes != 'Error: needs to be re-run'):
                    print('nnnnnnoooottees',sumpaper.notes)
                    try:
                        notes = ast.literal_eval(sumpaper.notes)
                        notes2 = [note.replace('•', '') for note in notes]

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
                        keywords_list = [keyword.replace("'","\\'").strip() for keyword in keywords_str.split(',')]  # Split the keywords string into a list
                        keywords_repr = ", ".join([f"'{keyword}'" for keyword in keywords_list])  # Enclose each keyword in quotes and join the list with commas
                        keywords = ast.literal_eval('[' + keywords_repr + ']')  # Evaluate the resulting string as a Python list
                        #keywords = ast.literal_eval('['+sumpaper.keywords+']')
                    except ValueError:
                        # Handle the error by returning a response with an error message to the user
                        return HttpResponse("Invalid input: 'keywords' attribute is not a valid Python literal.")
                else:
                    keywords=''

            stuff_for_frontend.update({
                'paper':paper,
                'keywords':keywords,
                'sumpaper':sumpaper,
                'sumlang':sumlang,
                'notes':notes,
                'notes2':notes2,
                'cc_format':cc_format,
                'toolong':toolong,
                'public':public,
                'total_votes':total_votes,
                'chathistory':chathistory
            })
        else:
            print('nope')
            alpaper=False
            arxiv_detailsf=utils.get_arxiv_metadata(arxiv_id)
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
                if url != '':
                    parts = url.split('/')
                    print('parts',parts)
                    license = parts[-3]
                    version = parts[-2]
                    if license.upper() != "NONEXCLUSIVE-DISTRIB":
                        cc_format = 'CC ' + license.upper() + ' ' + version
                    else:
                        cc_format = license.upper() + ' ' + version

                print(cc_format) # Output: CC BY-NC-SA 4.0

                stuff_for_frontend.update({
                    'exist':exist,
                    'details':arxiv_dict,
                    'cc_format':cc_format
                })

        stuff_for_frontend.update({
            'alpaper':alpaper,
        })

        return render(request, "summarizer/arxividpage.html", stuff_for_frontend)


def vote(request, paper_id, direction):
    lang = get_language()

    print('in there',lang)
    paper = get_object_or_404(ArxivPaper, arxiv_id=paper_id)
    client_ip = request.META['REMOTE_ADDR']
    print('clientip',client_ip)
    # Check if this IP address has already voted on this post
    hashed_ip_address = hashlib.sha256(client_ip.encode('utf-8')).hexdigest()

    previous_votes = Vote.objects.filter(paper=paper, lang=lang, ip_address=hashed_ip_address)
    if previous_votes.exists() and not client_ip=='127.0.0.1':
        print('exist vote')

        error_message = 'vote#totvote'
        #error_message = urllib.parse.quote(error_message)
        #print('tturleee',error_message)
        url = '/arxiv-id/' + paper_id + '/' + error_message
        #print('tturl',url)
        return redirect(url)

        #return redirect('arxividpage', arxiv_id=paper_id, error_message=error_message)#%2523=#
        #return redirect('post_detail', post_id=post.pk)

    # Create a new vote
    if direction=="up":
        valuevote=1
    elif direction=="down":
        valuevote=-1
    else:
        valuevote=0

    if valuevote != 0:

        vote = Vote(paper=paper, lang=lang, ip_address=hashed_ip_address, vote=valuevote)
        vote.save()
        #if paper.total_votes+valuevote>=0:
        #paper.total_votes += valuevote
        #paper.save()

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
    arxiv_id = request.GET.get('arxiv_id')
    arxiv_group_name = "ar_%s" % arxiv_id
    cache.set(arxiv_group_name, False)
    return HttpResponse("Cache updated")
