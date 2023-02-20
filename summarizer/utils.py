import time
import requests
from .models import ArxivPaper
import asyncio
from asgiref.sync import sync_to_async,async_to_sync
from channels.db import database_sync_to_async
import pdfminer
from io import StringIO
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from django.shortcuts import redirect
from channels.layers import get_channel_layer
import re
from django.http import HttpResponse
import os
from django.conf import settings
import ast

channel_layer = get_channel_layer()
model="text-davinci-003"#"text-davinci-002"
temp=0.3

from io import StringIO
from html.parser import HTMLParser

class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs= True
        self.text = StringIO()
    def handle_data(self, d):
        self.text.write(d)
    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()


def summary_pdf(arxiv_id):
    # Get the summary object from the database
    if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

        print('paper',paper.title)
        # Generate the PDF file using reportlab
        #response = HttpResponse(content_type='application/pdf')
        #response = FileResponse(content_type='application/pdf')
        #response['Content-Disposition'] = f'attachment; filename="SummarizePaper-{str(arxiv_id)}.pdf"'
        filename="SummarizePaper-"+str(arxiv_id)+".pdf"
        response = HttpResponse(content_type="application/pdf")
        response['Content-Disposition'] = 'attachment; filename=%s' % filename
        # Create the PDF canvas
        from fpdf import FPDF, HTMLMixin

        class MyPDF(FPDF, HTMLMixin):
            def __init__(self):
                super().__init__(orientation='P', unit='mm', format='A4')
                self.add_font('DejaVu', '', os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed.ttf'), uni=True)
                self.add_font('DejaVu', 'B', os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed-Bold.ttf'), uni=True)
                self.add_font('DejaVu', 'I', os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed-Oblique.ttf'), uni=True)

                self.add_page()
                self.set_font("Arial", size=12)

            def header(self):
                self.set_font("DejaVu", "B", size=14)
                self.cell(0, 10, "Made from SummarizePaper.com for arXiv ID: "+str(arxiv_id), 1, 0, "C")
                self.ln(20)

            def section(self, title, text):
                self.set_font("DejaVu", "B", size=12)
                self.cell(0, 10, title, 0, 1)
                self.set_font("Arial", size=12)
                self.multi_cell(0, 10, text)
                self.ln(10)

        # Create a new PDF document with the MyPDF class
        pdf = MyPDF()

        # Add the first summary section to the document
        if paper.summary:
            pdf.section("Comprehensive Summary", paper.summary.lstrip().rstrip())

        # Add the second summary section to the document
        if paper.notes:
            notes = paper.notes.replace('•','')
            print('rrrr',notes)

        try:
            notesarr = ast.literal_eval(notes)
        except ValueError:
            # Handle the error by returning a response with an error message to the user
            return HttpResponse("Invalid input: 'notes' attribute is not a valid Python literal.")

        notestr=''
        if notesarr:
            for note in notesarr:
                notestr+='-'+note+'\n'
                print('n',note)

        if paper.notes:
            pdf.section("Key Points", notestr)

        if paper.longer_summary:
            pdf.section("10-yrs old summary", paper.longer_summary.lstrip().rstrip())

        if paper.blog:
            pdf.section("Blog Article", strip_tags(paper.blog.lstrip().rstrip()))

        #pdf.section("Blog Article", notestr)

        #pdf.section("Key Points", summary_2.encode('latin-1', 'replace').decode('latin-1'))

        # Save the PDF file
        out=pdf.output("summary.pdf",dest='S')
        print('resp')

        return out

    else:
        print('no paper')

        return HttpResponseRedirect(reverse('arxividpage', args=(arxiv_id,)))


def update_arxiv_paper(arxiv_id,summary):

    paper, created = ArxivPaper.objects.update_or_create(
        arxiv_id=arxiv_id,
        defaults={'summary': summary}
    )
    return paper, created



def summarizer(arxiv_id):
    total_steps = 10
    for i in range(total_steps):
        # Perform some action to update the loading message and progress

        progress = 100 * (i + 1) / total_steps
        print('jk',progress)
        #time.sleep(0.1)

        yield progress

async def extract_text_from_pdf(pdf_filename):
    # Open the PDF file
    with open(pdf_filename, 'rb') as file:
        # Create a PDF resource manager object that stores shared resources
        resource_manager = PDFResourceManager()

        # Create a string buffer object for text extraction
        text_io = StringIO()

        # Create a text converter object
        text_converter = TextConverter(resource_manager, text_io, laparams=LAParams())

        # Create a PDF page interpreter object
        page_interpreter = PDFPageInterpreter(resource_manager, text_converter)

        # Process each page in the PDF file
        for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
            page_interpreter.process_page(page)
            text = text_io.getvalue()

        # Get the extracted text
        #text = text_io.getvalue()

        '''
        keywords=["ABSTRACT", "Key words."]
        start_keyword, end_keyword = keywords
        start = text.find(start_keyword) + len(start_keyword)
        end = text.find(end_keyword)
        '''
        end = text.find("References")
        end2 = text.find("REFERENCES")
        end3 = text.find("Acknowledgements")
        end4 = text.find("ACKNOWLEDGEMENTS")

        print('end,end2,end3,end4',end,end2,end3,end4)
        numbers=[end,end2,end3,end4]
        min_positive = float('inf')
        for number in numbers:
            if number > 0 and number < min_positive:
                min_positive = number

        if min_positive != float('inf'):
            print("The smallest positive number is", min_positive)
            endf=min_positive
        else:
            print("There are no positive numbers in the list")
            endf=-1

        print('abs:',text[0:endf].strip())
        textlim=text[0:endf].strip()
        # Close the text buffer and the text converter
        text_io.close()
        text_converter.close()


        # Return the extracted text
        return textlim


async def send_message_now(arxiv_group_name,message):
    print('in sendmesnow')

    await channel_layer.group_send(
        arxiv_group_name, {"type": "progress_text_update", "message": message}
    )

async def summarize_book(arxiv_id, book_text, api_key):
    endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"

    message={}
    arxiv_group_name="ar_%s" % arxiv_id
    # Split the book into chunks of at most 4096 tokens
    print("len(book_text)",len(book_text))
    chunk_size = 4096
    chunks = [book_text[i:i+chunk_size] for i in range(0, len(book_text), chunk_size)]

    # Send each chunk to the API for summarization
    summarized_chunks = []
    for chunk in chunks:
        prompt = f"summarize the following text in 100 words: {chunk}"
        print("prompt:",prompt)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        response = requests.post(endpoint, headers=headers, json={"prompt": prompt, "max_tokens": 400, "temperature": temp, "n":1, "stop":None})

        try:
            print('in try1')
            if response.status_code != 200:
                print("in1 ! 200")
                raise Exception(f"Failed to summarize text: {response.text}")
        except Exception as e:
            print('in redirect1',str(e))
            # Redirect to the arxividpage and pass the error message
            return {
                "error_message": str(e),
            }
            #return redirect('arxividpage', arxiv_id=arxiv_id, error_message="e1")#str(e))
            #return render(request, "summarizer/arxividpage.html", stuff_for_frontend)


        #if response.status_code != 200:
        #    raise Exception(f"Failed to summarize chunk: {response.text}")

        summarized_chunks.append(response.json()["choices"][0]["text"])
        print('yo:\n',response.json()["choices"][0]["text"])

    # Concatenate the summarized chunks and send the result to the API for further summarization

    print("beeeefffffooooooorrreee1")
    message["progress"] = 35
    message["loading_message"] = "Summarizing in progress..."
    c=asyncio.create_task(send_message_now(arxiv_group_name,message))

    #c=asyncio.create_task(channel_layer.group_send(arxiv_group_name, {"type": "progress_text_update", "message": message}))
    await c
    #time.sleep(10.)
    print("afffffteeeeeeeerrrrrdsdrrreee1")

    summarized_text = " ".join(summarized_chunks)
    print("len(summarized_text)",len(summarized_text))
    print('summarized_text',summarized_text)

    cont=1
    final_summarized_text=summarized_text
    i=0
    while cont==1:
        print('iiiiiiiiiiiiiiiii:\n',i)
        chunks2 = [final_summarized_text[i:i+chunk_size] for i in range(0, len(final_summarized_text), chunk_size)]

        summarized_chunks2 = []
        jj=0
        for chunk2 in chunks2:
            print('jjjjjjjjjjjjjj:\n',jj)

            prompt2 = f"Summarize the following text from a research article in 300 words: {chunk2}"
            headers2 = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}"
            }
            response2 = requests.post(endpoint, headers=headers2, json={"prompt": prompt2, "max_tokens": 500, "temperature": temp, "n":1, "stop":None})

            #if response2.status_code != 200:
            #    raise Exception(f"Failed to summarize text: {response2.text}")
                #it happens sometimes so to be treated...
            try:
                print('in try')
                if response2.status_code != 200:
                    print("in ! 200")
                    raise Exception(f"Failed to summarize text: {response2.text}")
            except Exception as e:
                print('in redirect')
                # Redirect to the arxividpage and pass the error message
                return {
                    "error_message": str(e),
                }                #return render(request, "summarizer/arxividpage.html", stuff_for_frontend)

            summarized_chunks2.append(response2.json()["choices"][0]["text"])
            jj+=1

        print('len summarized_chunks2',len(summarized_chunks2))

        print("beeeefffffooooooorrreee2")
        message["progress"] = 50
        message["loading_message"] = "Extracting key points..."
        c=asyncio.create_task(send_message_now(arxiv_group_name,message))

        #c=asyncio.create_task(channel_layer.group_send(arxiv_group_name, {"type": "progress_text_update", "message": message}))
        await c
        #time.sleep(10.)
        print("afffffteeeeeeeerrrrrdsdrrreee2")


        summarized_text2 = " ".join(summarized_chunks2)
        print('\nsummmmmmmmmmmmmmm\n',summarized_text2)
        if len(summarized_chunks2)==1:
        #if len(summarized_text2)<chunk_size:
            cont=0
            print('yes:\n',len(summarized_chunks2))
        else:
            print('no:\n',len(summarized_chunks2))
            final_summarized_text=summarized_text2#summarized_chunks2
        i+=1

    final_summarized_text = summarized_text2#response2.json()["choices"][0]["text"]
    print('yoyo:\n',final_summarized_text)

    return final_summarized_text

async def extract_key_points(arxiv_id, summary, api_key):
    endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"
    #Extract the most important key points from the following text
    prompt3 = f"Extract the most important key points from the following text and use bullet points for each of them: {summary}"
    print('key sum',prompt3)

    headers3 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response3 = requests.post(endpoint, headers=headers3, json={"prompt": prompt3, "max_tokens": 500, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try2')
        if response3.status_code != 200:
            print("in2 ! 200")
            raise Exception(f"Failed to summarize text2: {response3.text}")
    except Exception as e:
        print('in redirect2')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }        #return render(request, "summarizer/arxividpage.html", stuff_for_frontend)

    #if response3.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response3.text}")

    print('icccciiiiii',response3.json()["choices"][0]["text"])

    key_points = response3.json()["choices"][0]["text"].strip().split("\n")
    print('key_points',key_points)

    return key_points

async def extract_simple_summary(arxiv_id, keyp, api_key):
    endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"

    prompt4 = f"Summarize the following key points in 3 sentences for a 10 yr old: {keyp}"
    headers4 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response4 = requests.post(endpoint, headers=headers4, json={"prompt": prompt4, "max_tokens": 300, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try3')
        if response4.status_code != 200:
            print("in3 ! 200")
            raise Exception(f"Failed to summarize text: {response4.text}")
    except Exception as e:
        print('in redirect3')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }
        #return redirect('arxividpage', arxiv_id=arxiv_id, error_message="e1")
        #return render(request, "summarizer/arxividpage.html", stuff_for_frontend)

    #if response4.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response4.text}")

    simple_sum = response4.json()["choices"][0]["text"]#.strip().split("\n")
    print('simple_sum',simple_sum)

    return simple_sum

async def extract_blog_article(arxiv_id, summary, api_key):
    endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"

    prompt5 = f"Create a blog article in html about this research paper: {summary}"
    headers5 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    response5 = requests.post(endpoint, headers=headers5, json={"prompt": prompt5, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try5')
        if response5.status_code != 200:
            print("in5 ! 200")
            raise Exception(f"Failed to summarize text: {response5.text}")
    except Exception as e:
        print('in redirect5')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }
        #return redirect('arxividpage', arxiv_id=arxiv_id, error_message="e1")
        #return render(request, "summarizer/arxividpage.html", stuff_for_frontend)

    #if response4.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response4.text}")

    blog_article = response5.json()["choices"][0]["text"]#.strip().split("\n")
    print('blog article',blog_article)

    return blog_article

def get_arxiv_metadata(arxiv_id):
    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
    response = requests.get(url)
    #if response.status_code != 200:
    #    raise Exception(f"Failed to retrieve data: {response.text}")

    arxiv_id2= re.sub(r'v\d+$', '', arxiv_id)
    print('2',arxiv_id2)  # need to remove v1 or v2 as license is not found otherwise
    url2=f"http://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:{arxiv_id2}&metadataPrefix=arXiv"
    response2 = requests.get(url2)

    try:
        print('in try arxiv')
        if response.status_code != 200:
            print("in ! 200 arxiv")
            raise Exception(f"Failed to retrieve data: {response.text}")
    except Exception as e:
        print('in redirect arxiv')

    try:
        print('in try arxiv2')
        if response2.status_code != 200:
            print("in ! 200 arxiv2")
            raise Exception(f"Failed to retrieve data2: {response2.text}")
    except Exception as e:
        print('in redirect arxiv2')
        # Redirect to the arxividpage and pass the error message
        #return {
        #    "error_message": str(e),
        #}
        #return redirect('arxividpage', arxiv_id=arxiv_id, error_message="e0")
        #return render(request, "summarizer/arxividpage.html", stuff_for_frontend)


    data = response.text
    print('data',data)

    data2 = response2.text
    print('data2',data2)

    # Parse the XML response
    from xml.etree import ElementTree
    root = ElementTree.fromstring(data)

    root2 = ElementTree.fromstring(data2)

    # find and modify the value of an element

    #arxiv = root2.find(".//{http://arxiv.org/OAI/arXiv/}arXiv")
    ns = {'arxiv': 'http://arxiv.org/OAI/arXiv/'}
    arxiv = root2.find('.//arxiv:arXiv', ns)
    license_value=''

    if arxiv is not None:
        license_tag = arxiv.find('arxiv:license', ns)
        #license_tag = arxiv.find('.//license')

        if license_tag is not None:
            license_value = license_tag.text
            print(f"The license tag value is: {license_value}")
        else:
            #license_value=''
            print("The license tag was not found.")

    #for arxiv in root2.findall('.//{http://arxiv.org/OAI/arXiv/}arXiv'):
    #    print('a',arxiv)
    #    license_tag = arxiv.find('license').text
    #    print(f"The value of the 'license' tag is: {license_tag}")



    ns = {'ns0': 'http://www.w3.org/2005/Atom','ns1':'http://a9.com/-/spec/opensearch/1.1/','ns2':'http://arxiv.org/schemas/atom'} # add more as needed
    tit=root.find('ns0:title', ns).text

    # Extract the authors, title, and abstract

    #check if exists
    exist=0
    authors=""
    affiliation=""
    link_hp=""
    title=""
    link_doi=""
    abstract=""
    cat=""
    updated=""
    published=""
    journal_ref=""
    comments=""

    for entry in root.findall("ns0:entry",ns):
        if entry.find("ns0:title",ns) is not None:
            exist=1
            print('exist',exist)


    if exist == 1:
        authors = []
        affiliation = []
        title = ""
        abstract = ""
        for entry in root.findall("ns0:entry",ns):
            for author in entry.findall("ns0:author",ns):
                authors.append(author.find("ns0:name",ns).text)
                print('test')
                if author.find("ns2:affiliation",ns) is not None:
                    print('aff',author.find("ns2:affiliation",ns).text)
                    affiliation.append(author.find("ns2:affiliation",ns).text)
                else:
                    affiliation.append('')

            link_hp = entry.find("ns0:id",ns).text
            title = entry.find("ns0:title",ns).text
            link_doi = entry.find("ns0:link",ns).attrib['href']
            abstract = entry.find("ns0:summary",ns).text
            if entry.find("ns2:primary_category",ns) is not None:
                cat = entry.find("ns2:primary_category",ns).attrib['term']
            updated = entry.find("ns0:updated",ns).text
            if entry.find("ns0:published",ns) is not None:
                published = entry.find("ns0:published",ns).text
            #print('kllll',entry.find("ns2:journal_ref",ns))
            if entry.find("ns2:journal_ref",ns) is not None:
                journal_ref = entry.find("ns2:journal_ref",ns).text
            else:
                journal_ref = ''
            if entry.find("ns2:comment",ns) is not None:
                comments = entry.find("ns2:comment",ns).text
            else:
                comments = ''


    return [exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments,license_value]
