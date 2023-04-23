from ipaddress import ip_address
import time
import requests
from .models import ArxivPaper, SummaryPaper, PickledData, AIassistant, PaperScore, Author, PaperAuthor
from django.contrib.auth.models import User, AnonymousUser
import asyncio
from asgiref.sync import sync_to_async,async_to_sync
from channels.db import database_sync_to_async
import pdfminer
from io import StringIO
#from pdfminer.high_level import PDFResourceManager, PDFPageInterpreter, extract_pages
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from django.shortcuts import redirect,HttpResponseRedirect
from django.urls import reverse
from channels.layers import get_channel_layer
import re
from django.http import HttpResponse
import os
from django.conf import settings
import ast
import json
from html.parser import HTMLParser
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS, Chroma
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chains.question_answering import load_qa_chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain import ConversationChain     
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
import pickle
from django.utils.translation import get_language_info
import nltk
from xml.etree import ElementTree
import urllib, urllib.request
import math
from langchain.llms import OpenAIChat
import aiohttp
#from bs4 import BeautifulSoup

channel_layer = get_channel_layer()
model="gpt-3.5-turbo"#"text-davinci-003"#"text-davinci-002."
temp=0.3
method="fromembeddingsandabstract"#"fromembeddings"#"langchain"#quentin

import pdfkit
#from django.shortcuts import render
#from django.http import HttpResponse
from django.template.loader import get_template

def generate_pdf(request,arxiv_id,lang,local_date):
    # Define the HTML template
    if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]
        print('gennnnnnnnnn',lang)
        if SummaryPaper.objects.filter(paper=paper,lang=lang).exists():
            print('gennnnnnnnnn1')
            sumpaper=SummaryPaper.objects.filter(paper=paper,lang=lang)[0]
        elif SummaryPaper.objects.filter(paper=paper,lang='en').exists():
            print('enffdddfdfdf')
            sumpaper=SummaryPaper.objects.filter(paper=paper,lang='en')[0]
        else:
            sumpaper=''
            print('no summaries yet')

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
                notes='No notes yet'
                notes2=''

            if paper.link_doi:
                link=paper.link_doi
            else:
                link=paper.link_homepage

            if (sumpaper.keywords is not None) and (sumpaper.keywords != "") and (sumpaper.keywords != "['']"):
                print('keywords',sumpaper.keywords)
                try:
                    keywords_str = sumpaper.keywords.strip()  # Remove any leading or trailing whitespace
                    keywords_list = [keyword.strip() for keyword in keywords_str.split(',')]  # Split the keywords string into a list
                    #keywords_repr = ", ".join([fr"r'{keyword}'" for keyword in keywords_list])
                    keywords_repr = json.dumps(keywords_list)
                    keywords = json.loads(keywords_repr)
                    #keywords_repr = ", ".join([f"'{keyword}'" for keyword in keywords_list])  # Enclose each keyword in quotes and join the list with commas
                    #keywords = ast.literal_eval('[' + keywords_repr + ']')  # Evaluate the resulting string as a Python list
                    #keywords = ast.literal_eval('['+sumpaper.keywords+']')
                except ValueError:
                    # Handle the error by returning a response with an error message to the user
                    return HttpResponse("Invalid input: 'keywords' attribute is not a valid Python literal.")
            else:
                keywords=''

            from django.utils import timezone
            now = timezone.localtime(timezone.now()).strftime("%B %d, %Y %I:%M %p")
            print('now',now)

            template_path = 'summarizer/templatepdf.html'
            context = {'link':link,'paper':paper,'sumpaper':sumpaper,'notes':notes,'keywords':keywords,'now':local_date}
            # Render the HTML template
            template = get_template(template_path)
            html = template.render(context)
            # Configure pdfkit options
            options = {
                'quiet':'',
                'page-size': 'Letter',
                'margin-top': '0.75in',
                'margin-right': '0.75in',
                'margin-bottom': '0.75in',
                'margin-left': '0.75in',
                #'run-script':'MathJax.Hub.Config({"HTML-CSS": {scale: 200}}); MathJax.Hub.Queue(["Rerender", MathJax.Hub], function () {window.status="finished"})',
                'window-status': 'finished',
                #'javascript-delay': '5000'
                }
            # Convert HTML to PDF using pdfkit
            if 'ON_HEROKU' in os.environ:
                config = pdfkit.configuration(wkhtmltopdf='bin/wkhtmltopdf')
                pdf = pdfkit.from_string(html, False, options=options,configuration=config)
            else:
                pdf = pdfkit.from_string(html, False, options=options)

            #pdfkit.from_string(html_string, output_file, configuration=config)

            #pdf = pdfkit.from_string(html, False, options=options,configuration=config)
            # Create a response with the PDF file
            response = HttpResponse(pdf, content_type='application/pdf')
            filename="SummarizePaper-"+str(arxiv_id)+"-"+lang+".pdf"
            response['Content-Disposition'] = 'attachment; filename=%s' % filename  # force browser to download file
            #response['Content-Disposition'] = 'attachment; filename="my_pdf.pdf"'
            return response
        else:
            print('no paper')

            return HttpResponseRedirect(reverse('arxividpage', args=(arxiv_id,)))


def updatearvixdatapaper(arxiv_id,arxiv_dict):

        authors = []
        for author_name,aff in zip(arxiv_dict['authors'],arxiv_dict['affiliation']):
            try:
                author, createdauthor = Author.objects.get_or_create(name=author_name,affiliation=aff)
            except Exception as e:
                # handle the exception here
                print("An error occurred while getting or creating the author:", e)

            #author.affiliation = aff
            #author.save()
            #except Author.DoesNotExist:
            #    author = Author.objects.create(name=author_name, affiliation=aff)
            #author, created = Author.objects.get_or_create(name=author_name,affiliation=aff)
            authors.append(author)

        print('a',arxiv_dict['abstract'])
        arxiv_dict['abstract']=arxiv_dict['abstract'].replace('\n',' ')
        #input('llll')

        paper, created = ArxivPaper.objects.update_or_create(
            arxiv_id=arxiv_id,
            defaults={'link_homepage':arxiv_dict['link_homepage'], 'title':arxiv_dict['title'], 'link_doi':arxiv_dict['link_doi'], 'abstract':arxiv_dict['abstract'], 'category':arxiv_dict['category'], 'updated':arxiv_dict['updated'], 'published_arxiv':arxiv_dict['published_arxiv'], 'journal_ref':arxiv_dict['journal_ref'], 'comments':arxiv_dict['comments'],'license':arxiv_dict['license']}
        )

        for i, author in enumerate(authors):
            paper_author, created = PaperAuthor.objects.get_or_create(author=author, paper=paper, author_order=i)

        #authors_ids = [autho.id for autho in authors]
        #.set(authors_ids)
        #paper.authors.set(arxiv_dict['authors'])
        paper.save()

        #print('retdd',created)
        #'authors':arxiv_dict['authors'],'affiliation':arxiv_dict['affiliation']
        return paper,created

def openaipricing(model_name):
    #return cost per token in dollars
    if 'davinci' in model_name:
        cost=0.02
    elif 'babbage' in model_name:
        cost=0.0005
    elif 'curie' in model_name:
        cost=0.002
    elif 'ada' in model_name:
        cost=0.0004
    elif 'turbo' in model_name:
        cost=0.002
    else:
        cost=1.

    return cost/1000.

def nchars_leq_ntokens_approx(maxTokens):
    #returns a number of characters very likely to correspond <= maxTokens
    sqrt_margin = 0.5
    lin_margin = 1.010175047 #= e - 1.001 - sqrt(1 - sqrt_margin) #ensures return 1 when maxTokens=1
    return max( 0, int(maxTokens*math.exp(1) - lin_margin - math.sqrt(max(0,maxTokens - sqrt_margin) ) ))

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip
    
def dependable_faiss_import():# -> Any:
    """Import faiss if available, otherwise raise error."""
    try:
        import faiss
    except ImportError:
        raise ValueError(
            "Could not import faiss python package. "
            "Please it install it with `pip install faiss` "
            "or `pip install faiss-cpu` (depending on Python version)."
        )
    return faiss

def readpaper(arxiv_id):
    paper=ArxivPaper.objects.prefetch_related('authors').filter(arxiv_id=arxiv_id)[0]

    return paper

def getpaperabstract(arxiv_id):
    paperabstract=ArxivPaper.objects.filter(arxiv_id=arxiv_id).values_list('abstract',flat=True)[0]

    return paperabstract

def getlicense(arxiv_id):
    license=ArxivPaper.objects.filter(arxiv_id=arxiv_id).values_list('license',flat=True)[0]

    return license

def getpaper(arxiv_id):
    paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

    return paper

def getallpaperstoredo(certain_date):
    print('in redo')

    allpaper = ArxivPaper.objects.filter(created__lt=certain_date)
    #allpaper = ArxivPaper.objects.all()

    #allpapers = ArxivPaper.objects.filter(da)
    
    #print('all',allpapers)
    return allpaper

def getallpapers(cat):
    if 1==0:#cat != '':#for now we look everywhere to be changed
        print('in cat')
        allpapers = ArxivPaper.objects.filter(category=cat)
    else:
        print('not in cat')
        allpapers = ArxivPaper.objects.all()

    print('all',allpapers)
    return allpapers

def getpapersfromlist(list):
    for i in range(len(list)):
        list[i] = list[i].replace('/', '--')

    print('list',list)    

    allpapers = ArxivPaper.objects.filter(arxiv_id__in=list)
    
    return allpapers

def getuserinst(user):
    if User.objects.filter(username=user).exists():
        userinst = User.objects.get(username=user)
        # Do something with the admin_user instance
    else:
        userinst = None# AnonymousUser()

    return userinst

def storeconversation(arxiv_id,query,response,user,lang,ip=None):

    if ip:
        AIassistant.objects.create(
            arxiv_id=arxiv_id,
            query=query,
            ip_address=ip,
            response=response,
            user=user,
            lang=lang
        )
    else:
        AIassistant.objects.create(
            arxiv_id=arxiv_id,
            query=query,
            response=response,
            user=user,
            lang=lang
        )

def getconversationmemory(arxiv_id,user,lang,ip,nb_mem):

    if user != None:
        conv=AIassistant.objects.filter(
            arxiv_id=arxiv_id,
            user=user,
            lang=lang
        ).order_by('-created')[:nb_mem]
    else:
        conv=AIassistant.objects.filter(
            arxiv_id=arxiv_id,
            ip_address=ip,
            lang=lang
        ).order_by('-created')[:nb_mem]

    return conv

def storeclosest(arxiv_id,ids_and_scores):
    array_arxiv_ids,array_scores=ids_and_scores
    # assume you have an array of scores and arxiv_ids
    #scores_and_arxiv_ids = [(0.8, '1234.5678'), (0.6, '9012.3456'), (0.4, '7890.1234')]

    scores_and_arxiv_ids = list(zip(array_scores, array_arxiv_ids))

    # Paper with the given arxiv_id
    paper = ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

    #make other related papers inactive
    oldpapers = PaperScore.objects.filter(from_paper=paper,active=True)
    for oldpap in oldpapers:
        #oldpap.active=False
        #oldpap.save()
        oldpap.delete()

    # create PaperScore objects for each related paper with its associated score
    for score, arxiv_id2 in scores_and_arxiv_ids:
        related_paper, created = ArxivPaper.objects.get_or_create(arxiv_id=arxiv_id2)
        PaperScore.objects.create(from_paper=paper, to_paper=related_paper, score=score)

    return 0

def storepickle(arxiv_id,docstore_pickle,index_to_docstore_id_pickle,buffer):

    obj, created = PickledData.objects.update_or_create(
        arxiv_id=arxiv_id,
        defaults={
            'docstore_pickle': docstore_pickle,
            'index_to_docstore_id_pickle': index_to_docstore_id_pickle,
            'buffer':buffer
        }
    )

    return created

def getstorepickle(arxiv_id):

    if PickledData.objects.filter(arxiv_id=arxiv_id).exists():
        print("Data found with arxiv_id =", arxiv_id)
        pickledata=PickledData.objects.filter(arxiv_id=arxiv_id)[0]
    else:
        print("No data found with arxiv_id =", arxiv_id)
        pickledata=''

    return pickledata



async def createindex(arxiv_id,book_text,api_key):

    text_splitter = CharacterTextSplitter.from_tiktoken_encoder(separator = "\n\n", chunk_size=800, chunk_overlap=200)#limit at 8096 tokens
    texts = text_splitter.split_text(book_text)

    '''
    text_splitter = CharacterTextSplitter(
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
    )
    texts = text_splitter.split_text(book_text)
    '''
    print('tttettxtxtxtxtxtxtxtttzetet',texts)

    regex = r'<latexit.*?</latexit>'
    texts = [re.sub(regex, '', text) for text in texts]
    print('tttettxtxtxtxtxtxtxtttzetet222',texts)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)#text-embedding-ada-002 used in background

    new_docsearch=embeddings

    #docsearch = FAISS.from_texts(texts, new_docsearch,metadatas=[{"source": str(i)} for i in range(len(texts))])

    metadatas = [
        {"arxiv_id":arxiv_id, "source": f"from {texts[i][0:30]} --- to --- {texts[i][len(texts[i])-30:-1]}"}
        for i in range(len(texts))
    ]

    # Print the metadata
    #for metadata in metadatas:
    #    print('met',metadata)

    docsearch = FAISS.from_texts(texts, new_docsearch,metadatas=metadatas)



    #input('jkll')
    #docsearch = Chroma.from_texts(texts, embeddings)
    #tu=FAISS.save_local(docsearch,"savedocsearch")

    print('docsearchhhhhhhhhhhhhhh index',docsearch.index)
    print('docsearchhhhhhhhhhhhhhh doc',docsearch.docstore)
    print('docsearchhhhhhhhhhhhhhh id',docsearch.index_to_docstore_id)
    # save index separately since it is not picklable
    faiss = dependable_faiss_import()
    # serialize the index to a byte buffer
    #buffer = bytearray()
    #faiss.write_index(docsearch.index, buffer)
    chunk = faiss.serialize_index(docsearch.index)
    buffer_pickle = pickle.dumps(chunk)

    # save docstore and index_to_docstore_id
    docstore_pickle = pickle.dumps(docsearch.docstore)
    index_to_docstore_id_pickle = pickle.dumps(docsearch.index_to_docstore_id)

    # update or create a PickledData object with the given arxiv_id
    c = asyncio.create_task(sync_to_async(storepickle)(arxiv_id,docstore_pickle,index_to_docstore_id_pickle,buffer_pickle))
    created = await c

    print('ok created',created)
    return created

async def findclosestpapers(arxiv_id,language,k,api_key,but=False):
    print('findclosestpapers')
    message={}
    arxiv_group_name="ar_%s" % arxiv_id

    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    print('arxiv_id',arxiv_id)
    c = asyncio.create_task(sync_to_async(getstorepickle)(arxiv_id))
    getstoredpickle = await c
    print('here')

    cat=''
    c = asyncio.create_task(sync_to_async(getpaper)(arxiv_id))
    mainpaper = await c
    cat = mainpaper.category
    print('cat',cat)

    if but==True:
        message["progress"] = 35
        if language == 'fr':
            message["loading_message"] = "Comparaison en cours..."
        else:
            message["loading_message"] = "Comparison in progress..."
        c=asyncio.create_task(send_message_now(arxiv_group_name,message))
        await c
        #await asyncio.sleep(10)


    if getstoredpickle != '':
        # deserialize the index from a byte buffer
        #index_buffer = faiss.read_index(storedpickle.buffer)
        faiss = dependable_faiss_import()

        index_buffer = faiss.deserialize_index(pickle.loads(getstoredpickle.buffer))   # identical to index

        docstore_pickle=pickle.loads(getstoredpickle.docstore_pickle)
        index_to_docstore_id_pickle=pickle.loads(getstoredpickle.index_to_docstore_id_pickle)
        embeddings = OpenAIEmbeddings(openai_api_key=api_key)

        docsearch = FAISS(embeddings.embed_query, index_buffer, docstore_pickle, index_to_docstore_id_pickle)
        print('finished')

    #Now run through all embeddings in database
    c = asyncio.create_task(sync_to_async(getallpapers)(cat))
    allpapers = await c

    #print('allpapers',allpapers)
    print('right')
    db=''
    async for paper in allpapers:
        if paper != mainpaper and paper.arxiv_id.split("v")[0] != mainpaper.arxiv_id.split("v")[0]:
            print('pap',paper)
            c2 = asyncio.create_task(sync_to_async(getstorepickle)(paper.arxiv_id))

            getstoredpickle2 = await c2

            if getstoredpickle2 != '':
                # deserialize the index from a byte buffer
                #index_buffer = faiss.read_index(storedpickle.buffer)
                #faiss = dependable_faiss_import()

                index_buffer2 = faiss.deserialize_index(pickle.loads(getstoredpickle2.buffer))   # identical to index

                docstore_pickle2=pickle.loads(getstoredpickle2.docstore_pickle)
                index_to_docstore_id_pickle2=pickle.loads(getstoredpickle2.index_to_docstore_id_pickle)

                docsearch2 = FAISS(embeddings.embed_query, index_buffer2, docstore_pickle2, index_to_docstore_id_pickle2)

                if db == '':
                    db = docsearch2
                    #print('db.docstore._dict',db.docstore._dict)
                else:
                    db.merge_from(docsearch2)
                    #print('db.docstore._dict',db.docstore._dict)

    
    #print('doc',docsearch.index)
    indices = [k for k in docsearch.index_to_docstore_id.keys()]

    #print('indices',indices)

    if but==True:
        message["progress"] = 50
        
        c=asyncio.create_task(send_message_now(arxiv_group_name,message))
        await c
        #await asyncio.sleep(5)

    recembeddings = [docsearch.index.reconstruct(int(i)) for i in indices if i != -1]
    #print('recembeddings',recembeddings)
    #recembeddings = ','.join(map(str,recembeddings))
    #print('recembeddings2',recembeddings)

    #print('hhh',docsearch.index.reconstruct(0))


    #print('index_to_docstore_id',docsearch.index_to_docstore_id)
    #docs_and_scores = db.similarity_search_by_vector(docsearch.index.reconstruct(0),4)
    #docs_and_scores = db.similarity_search_with_score_by_vector(docsearch.index.reconstruct(0),k)
    
    ''' this works but a bit lengthy
    docs_and_scores=[]
    for emb in recembeddings:
        resultsimsearch=db.similarity_search_with_score_by_vector(emb,k)
        docs_and_scores+=resultsimsearch
    '''
    import numpy as np
    print('len(recembeddings)',len(recembeddings))
    if len(recembeddings)>1:
        recembeddings=np.mean(recembeddings, axis=0)
    else:
        recembeddings=docsearch.index.reconstruct(0)

    '''doesn't work for now
    import numpy as np

    recembeddings = np.concatenate(recembeddings)
    print('recembeddings2',recembeddings)
    '''
    
    print('len(recembeddings)2',len(recembeddings))

    docs_and_scores = db.similarity_search_with_score_by_vector(recembeddings,k)

    #print('docs_and_scores',docs_and_scores)
    array_arxiv_ids=[]
    array_scores=[]

    if but==True:
        message["progress"] = 75
        c=asyncio.create_task(send_message_now(arxiv_group_name,message))
        await c
        #await asyncio.sleep(5)

    for document in docs_and_scores:
        #source_value = document[0].metadata['source']
        #print('ss',source_value)
        metadata = document[0].metadata#['arxiv_id']
        score_value = document[1]

        pattern = r'v(\d+)$'

        aid = metadata.get('arxiv_id')
        if aid is not None:
            match = re.search(pattern, aid)
            print('match',match)
            if match is None:
                if aid not in array_arxiv_ids:
                    print('1',aid)
                    array_arxiv_ids.append(aid)
                    array_scores.append(score_value)
            else:
                version = int(match.group(1))
                print('version',version)
                for existing_aid in array_arxiv_ids:
                    print('2',existing_aid)
                    existing_match = re.search(pattern, existing_aid)
                    print('existing_match',existing_match)
                    if existing_match is None:
                        continue
                    existing_version = int(existing_match.group(1))
                    print('existing_version',existing_version)

                    if existing_aid.startswith(aid[:-len(match.group())]):# and version <= existing_version:
                        print('len(match.group()',len(match.group()))
                        print('aid[:-len(match.group())]',aid[:-len(match.group())])
                        break
                else:
                    if aid not in array_arxiv_ids:
                        array_arxiv_ids.append(aid)
                        array_scores.append(score_value)
        else:
            print("Arxiv_id is not present in metadata")

        '''
        aid = metadata.get('arxiv_id')
        if aid is not None:
            print(aid)
            if aid not in array_arxiv_ids:
                array_arxiv_ids.append(aid)
                array_scores.append(score_value)
        else:
            print("Arxiv_id is not present in metadata")
        '''
        print('score',score_value)

    print('array_arxiv_ids',array_arxiv_ids)
    #docs = docsearch2.similarity_search(query,k=kvalue)
    #print('docs:',docs)
    #print('len docs:',len(docs))

    return array_arxiv_ids,array_scores

def construct_prompt(documents, question, language, mem,mul=None):

    li = get_language_info(language)
    language2 = li['name']

    PROMPT_TEMPLATE = """
    =========== BEGIN DOCUMENTS =============
    {documents}
    ============ END DOCUMENTS ==============

    Question: {question}
    """
    if language != 'en' and not 'TRANSLATE' in question and not 'TRADUIRE' in question:
        PROMPT_TEMPLATE += """FINAL ANSWER IN """+language2

    return PROMPT_TEMPLATE.format(
        documents="\n".join([construct_document_prompt(d,mul) for d in documents]),
        question=question,
        mem=mem
    )

def filter_documents(documents):
    MIN_DOCUMENT_LENGTH = 20

    return [d for d in documents if len(d.page_content) > MIN_DOCUMENT_LENGTH]


def construct_document_prompt(document,mul=None):

    DOCUMENT_TEMPLATE = """
    ------------ BEGIN DOCUMENT -------------
    {content}
    ------------- END DOCUMENT --------------
    """

    DOCUMENT_TEMPLATE_WITH_SOURCE = """
    ------------ BEGIN DOCUMENT -------------
    --------------- CONTENT -----------------
    {content}
    ---------------- SOURCE -----------------
    {source}
    ------------- END DOCUMENT --------------
    """
    
    if mul:
        return DOCUMENT_TEMPLATE_WITH_SOURCE.format(content=document.page_content, source=document.metadata["arxiv_id"])
    else:
        return DOCUMENT_TEMPLATE.format(content=document.page_content)

    #return DOCUMENT_TEMPLATE.format(content=document.page_content, source=document.metadata["arxiv_id"])


async def chatbot(arxiv_id,language,query,api_key,sum=None,user=None,memory=None,ip=None,selectedpapers=None,countpaperwithlicenses=None):
    print('in chatbot')

    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    print('countpaperwithlicenses',countpaperwithlicenses)

    #The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

    #SYSTEM_PROMPT = """
    #You are Knowledge bot. In each message you will be given the extracted parts of a knowledge base
    #(labeled with DOCUMENT) and a question.
    #Answer the question using information from the knowledge base.
    #If the answer is not available in the documents or there are no documents,
    #still try to answer the question, but say that you used your general knowledge and not the documentation.
    #"""

    

    '''
    SYSTEM_PROMPT = """
    The following is a friendly conversation between a human and an AI. The AI is talkative and provides
    lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
    Answer the question using information from the knowledge base labeled with DOCUMENT.
    If the answer is not available in the documents or there are no documents,
    still try to answer the question, but say that you used your general knowledge and not the documentation.
    """
    '''
    

    SYSTEM_PROMPT_WITH_SOURCES = """
    You are Knowledge bot. In each message you will be given the extracted parts of a knowledge base
    (labeled with DOCUMENT and SOURCE) and a question.
    Answer the question using information from the knowledge base, including references ("SOURCES").
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    """

    #if language != 'en' and not 'TRANSLATE' in query and not 'TRADUIRE' in query:
    #    SYSTEM_PROMPT += """FINAL ANSWER IN """+language2
    #    SYSTEM_PROMPT_WITH_SOURCES += """FINAL ANSWER IN """+language2

    if sum==1:
        #llm = OpenAI(temperature=0.3,max_tokens=800,frequency_penalty=0.6, presence_penalty=0.6,openai_api_key=api_key)
        llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature=0.3,max_tokens=800,frequency_penalty=0.6, presence_penalty=0.6,openai_api_key=api_key)
    else:
        #llm = OpenAI(temperature=0.3,max_tokens=700,openai_api_key=api_key)
        llm = OpenAIChat(model_name="gpt-3.5-turbo",temperature=0.3,max_tokens=700,openai_api_key=api_key)

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    if 1==1:
        if selectedpapers:
            SYSTEM_PROMPT = """
            The following is a friendly conversation between a human and an AI. The AI is talkative and provides
            lots of specific details from its context (multiple extracts of papers or articles). If the AI does not know the answer to a question, it truthfully says it does not know.
            The question can specify to TRANSLATE the response in another language, which the AI should do.
            If the question is not related to the context warn the user that your are a knowledge bot dedicated to explaining articles only. 
            Return a "SOURCES" part in your answer if it is relevant.
            """

            if countpaperwithlicenses!=None:
                if countpaperwithlicenses>0:
                    SYSTEM_PROMPT += """
                    The licenses of some of the selected papers do not allow us to read the papers so if you do not find an answer warn the reader that it may be due to that.
                    """

            print('sel',selectedpapers)
            #allpapers = json.loads(selectedpapers)
            #print('allpapers',allpapers)
            db=''

            #c = asyncio.create_task(sync_to_async(getallpapers)(cat))
            #allpapers = await c
            c = asyncio.create_task(sync_to_async(getpapersfromlist)(selectedpapers))
            allpapers = await c

            #print('allpapers',allpapers)

            async for paper in allpapers:
                print('paper',paper)
                if 1==1:
                    faiss = dependable_faiss_import()
        
                    print('pap',paper)
                    c2 = asyncio.create_task(sync_to_async(getstorepickle)(paper.arxiv_id))

                    getstoredpickle2 = await c2

                    if getstoredpickle2 != '':
                        # deserialize the index from a byte buffer
                        #index_buffer = faiss.read_index(storedpickle.buffer)
                        #faiss = dependable_faiss_import()

                        index_buffer2 = faiss.deserialize_index(pickle.loads(getstoredpickle2.buffer))   # identical to index

                        docstore_pickle2=pickle.loads(getstoredpickle2.docstore_pickle)
                        index_to_docstore_id_pickle2=pickle.loads(getstoredpickle2.index_to_docstore_id_pickle)

                        docsearch2 = FAISS(embeddings.embed_query, index_buffer2, docstore_pickle2, index_to_docstore_id_pickle2)

                        if db == '':
                            db = docsearch2
                            #print('db.docstore._dict',db.docstore._dict)
                        else:
                            db.merge_from(docsearch2)
                            #print('db.docstore._dict',db.docstore._dict)
            
            docsearch2=db
        else:
            SYSTEM_PROMPT = """
            The following is a friendly conversation between a human and an AI. The AI is talkative and provides
            lots of specific details from its context (an extract of a paper or article). If the AI does not know the answer to a question, it truthfully says it does not know.
            The question can specify to TRANSLATE the response in another language, which the AI should do.
            If the question is not related to the context warn the user that your are a knowledge bot dedicated to explaining one article. 
            """
            if countpaperwithlicenses!=None:
                if countpaperwithlicenses>0:
                    SYSTEM_PROMPT += """
                    The license of the selected paper is not fully open source and does not allow us to read the paper so if you do not find an answer warn the reader that it may be due to that.
                    """

            c = asyncio.create_task(sync_to_async(getstorepickle)(arxiv_id))

            getstoredpickle = await c

            if getstoredpickle != '':
                # deserialize the index from a byte buffer
                #index_buffer = faiss.read_index(storedpickle.buffer)
                faiss = dependable_faiss_import()

                index_buffer = faiss.deserialize_index(pickle.loads(getstoredpickle.buffer))   # identical to index

                docstore_pickle=pickle.loads(getstoredpickle.docstore_pickle)
                index_to_docstore_id_pickle=pickle.loads(getstoredpickle.index_to_docstore_id_pickle)

        

            #return cls(embeddings.embed_query, index, docstore, index_to_docstore_id)
            docsearch2 = FAISS(embeddings.embed_query, index_buffer, docstore_pickle, index_to_docstore_id_pickle)

        if sum == 1:
            kvalue=3
        else:
            kvalue=3

        docs = docsearch2.similarity_search(query,k=kvalue)
        print('docs:',docs)
        print('len docs:',len(docs))

        #import tiktoken

        #MAX_CHARS = 4200  # maximum number of characters to allow in the docs pages
        MAX_TOKENS=1500#2200
        MAX_CHARS=nchars_leq_ntokens_approx(MAX_TOKENS)
        print('MAX_CHARS',MAX_CHARS)
        num_chars = 0
        for doc in docs:
            #encoding = tiktoken.get_encoding("gpt3")
            #input_ids = encoding.encode(text)
            #print(input_ids)
            if num_chars + len(doc.page_content) <= MAX_CHARS:
                num_chars += len(doc.page_content)
            else:
                # suppress text to fit within MAX_CHARS
                print('enter here',num_chars)
                remaining_chars = MAX_CHARS - num_chars
                suppressed_text = doc.page_content[:remaining_chars] + " ..."
                doc.page_content = suppressed_text
                num_chars = MAX_CHARS
                #break

        print('docs22222222222:',docs)
        print('num_chars',num_chars)

        if sum==1:
            template = """We have an existing summary: {existing_answer}
                We have the opportunity to expand and refine the existing summary
                with some more context below.
                ------------
                {summaries}
                ------------
                Given the new context, create a refined detailed longer summary.
                """
        else:
            template = """Given the following extracted parts of a long document and a question, create a final answer.
            If you are not sure about the answer, just say that you are not sure before making up an answer.  

            QUESTION: {question}
            =========
            {summaries}
            =========

            If the question IS NOT about the document, DO NOT say it is not related to document but rather just be a helpful assistant, FRIENDLY and conversational and ANSWER the question anyway.

            """

        #If the question IS NOT about the document, just be an helpful assistant and CHAT with the user.

        #if language != 'en'
        #    template += """FINAL ANSWER IN """+language2

        print('tem',template)

        if sum==1:
            if language != 'en':
                template += """FINAL ANSWER IN """+language2
            print('wait here')
            c = asyncio.create_task(sync_to_async(getpaperabstract)(arxiv_id))
            print('wait...')
            paperabstract=await c
            print('wait2...')

            PROMPT = PromptTemplate(template=template, input_variables=["summaries", "existing_answer"])
        else:
            if language != 'en' and not 'TRANSLATE' in query and not 'TRADUIRE' in query:
                template += """FINAL ANSWER IN """+language2
            PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

            

        print('wait3...')
        chain = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT)
        print('wait4...')

        with get_openai_callback() as cb:

            if sum==1:
                print('wait5...')
                #getresponse=chain({"input_documents": docs, "existing_answer": paperabstract}, return_only_outputs=False)
                getresponse = await asyncio.to_thread(chain, {"input_documents": docs, "existing_answer": paperabstract}, return_only_outputs=False)
                print('wait6...')
            else:
                print('gkl')

                docs = filter_documents(docs)
                #from langchain.memory import ConversationBufferMemory

                c = asyncio.create_task(sync_to_async(getuserinst)(user))
                userinst = await c

                if ip or userinst:
                    NBMEM=5#remembers last five messages
                    print('clientip.....',ip)
                    # Check if this IP address has already voted on this post
                    print('arx',arxiv_id)
                    c = asyncio.create_task(sync_to_async(getconversationmemory)(arxiv_id,userinst,language,ip,NBMEM))
                    memorystored = await c
                    async for memstored in memorystored:
                        memory.chat_memory.add_user_message(memstored.query)
                        memory.chat_memory.add_ai_message(memstored.response)


                #memory2 = ConversationBufferMemory(return_messages=True)
                #memory.chat_memory.add_user_message(query)
                #memory.chat_memory.add_ai_message(getresponse)

                #print('memory',memory.load_memory_variables({}))
                memory_dict = memory.load_memory_variables({})

                history_value = memory_dict['history']
                print('history_value',history_value)

                chat_input = construct_prompt(docs, query, language, history_value)
                system_prompt = SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT)
                print('system_prompt',system_prompt)

                prompt = ChatPromptTemplate.from_messages(
                    [
                        system_prompt,
                        MessagesPlaceholder(variable_name="history"),
                        HumanMessagePromptTemplate.from_template("{input}"),
                    ]
                )

                #conversation = ConversationChain(memory=memory, llm=llm, verbose=True)
                conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=False)

                #response = conversation.predict(input=chat_input)
                print('chatinput',chat_input)

                #getresponse = await asyncio.to_thread(conversation.predict, input=chat_input)
                
                print('mememememme',memory)

                try:
                    import openai
                    getresponse = await asyncio.to_thread(conversation.predict, input=chat_input)
                except openai.error.InvalidRequestError as e:
                    # Catch the error and extract the requested length from the error message
                    #requested_length = int(str(e).split(" ")[-2])
                    #print('requested_length',requested_length)
                    # Reduce the input length by the excess amount
                    #excess_length = requested_length - 4097
                    #chat_input = chat_input[:-excess_length]
                    
                    print('clear memory')
                    memory.clear()

                    # Rerun the function with the reduced input
                    conversation = ConversationChain(memory=memory, prompt=prompt, llm=llm, verbose=False)

                    getresponse = await asyncio.to_thread(conversation.predict, input=chat_input)


                #if memory is not None:
                    #memory.chat_memory.add_user_message(query)
                    #memory.chat_memory.add_ai_message(getresponse)
                    #memory2.save_context({"input": query}, {"ouput": getresponse})
                    #print('memory',memory2.load_memory_variables({}))


                #getresponse = await asyncio.to_thread(chain, {"input_documents": docs, "question": query}, return_only_outputs=False)

            nbtokensused=cb.total_tokens
            print('wait7...')

        print('nbtokensusedchatbot',nbtokensused)
        print('openai cost chatbot',nbtokensused*openaipricing("text-davinci-003"))

        #qa = ChatVectorDBChain.from_llm(llm, docs)
        #chat_history = []
        #getresponse = qa({"question": query, "chat_history": chat_history})


        #chain = load_qa_chain(llm, chain_type="stuff")
        #chain_type="map_reduce", return_map_steps=True)
        #getresponse=chain({"input_documents": docs, "question": query}, return_only_outputs=True)

        #getresponse=chain.run(input_documents=docs, question=query)
        print('getresponse',getresponse)
        #print('getresponse2',getresponse['output_text'])

        if sum==1:
            finalresp=getresponse['output_text'].replace(':\n', '').rstrip().lstrip()
        else:
            finalresp=getresponse.replace(':\n', '').replace('AI: ', '').rstrip().lstrip()

        #store the query and answer
        if sum != 1:
            print('clientip22.....',ip)

            if ip:
                c = asyncio.create_task(sync_to_async(storeconversation)(arxiv_id,query,finalresp,userinst,language,ip=ip))
                await c
            else:
                print('elseip')
                c = asyncio.create_task(sync_to_async(storeconversation)(arxiv_id,query,finalresp,userinst,language))
                await c

        return finalresp
    #else:
    #    return None


class MLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.reset()
        self.strict = False
        self.convert_charrefs = True
        self.text = StringIO()
        self.in_h1_tag = False

    def handle_starttag(self, tag, attrs):
        if tag == 'h1':
            self.in_h1_tag = True

    def handle_endtag(self, tag):
        if tag == 'h1':
            self.in_h1_tag = False

    def handle_data(self, d):
        if self.in_h1_tag:
            self.text.write('<b>{}</b>'.format(d))
        else:
            self.text.write(d)

    def get_data(self):
        return self.text.getvalue()

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

def summary_pdf2(arxiv_id,language):
    # Get the summary object from the database
    if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

        if SummaryPaper.objects.filter(paper=paper,lang=language).exists():
            sumpaper=SummaryPaper.objects.filter(paper=paper,lang=language)[0]
        elif SummaryPaper.objects.filter(paper=paper,lang='en').exists():
            sumpaper=SummaryPaper.objects.filter(paper=paper,lang='en')[0]
        else:
            sumpaper=''
            print('no summaries yet')

        print('paper',paper.title)
        # Generate the PDF file using reportlab
        #response = HttpResponse(content_type='application/pdf')
        #response = FileResponse(content_type='application/pdf')
        #response['Content-Disposition'] = f'attachment; filename="SummarizePaper-{str(arxiv_id)}.pdf"'
        #filename="SummarizePaper-"+str(arxiv_id)+".pdf"
        #response = HttpResponse(content_type="application/pdf")
        #response['Content-Disposition'] = 'attachment; filename=%s' % filename
        # Create the PDF canvas
        from fpdf import FPDF, HTMLMixin
        from io import BytesIO

        #import latexcodec
        #from pylatexenc.latex2text import LatexNodes2Text
        #buffer = BytesIO()

        #print('osss',os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed.ttf'))
        class MyPDF(FPDF, HTMLMixin):
            def __init__(self):
                super().__init__(orientation='P', unit='mm', format='A4')
                #self.add_font('DejaVu', '', 'font/DejaVuSansCondensed.ttf', uni=True)
                print('os.path.join(settings.BASE_DIR, "font", "DejaVuSansCondensed.ttf")',os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed.ttf'))
                self.add_font('DejaVu', '', "font/DejaVuSansCondensed.ttf", uni=True)
                self.add_font('DejaVu', 'B', "font/DejaVuSansCondensed-Bold.ttf", uni=True)
                self.add_font('DejaVu', 'I', "font/DejaVuSansCondensed-Oblique.ttf", uni=True)
                #self.add_font('DejaVu', 'B', os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed-Bold.ttf'), uni=True)

                self.add_page()
                #self.set_font("Arial", size=12)
                #self.set_font("Helvetica", size=12)

            #def header(self):
            #    self.set_font("DejaVu", "B", size=14)
            #    #self.set_font("Arial","B", size=14)
            #    self.cell(0, 10, "Made from SummarizePaper.com for arXiv ID: "+str(arxiv_id), 1, 0, "C")
            #    self.ln(20)

            def paperdet(self, title, text, url):
                #self.set_font("DejaVu", "I", size=12)
                self.set_font("Arial","I", size=12)
                self.cell(0, 10, "Title: "+title, 0, 1)
                self.set_font("Arial", size=10)

        pdf = MyPDF()

        if paper.link_doi:
            link=paper.link_doi
        else:
            link=paper.link_homepage

        #pdf.paperdet(paper.title.strip(), paper.abstract.lstrip().rstrip(),str(link).strip())

        pdf.set_font('helvetica', size=12)
        pdf.cell(txt="hello world")

        #out=pdf.output(dest='S')
        #print('resp',out)

        #pdf.output(BytesIO())
        file = BytesIO()
        pdf.output(file)
        return file.getvalue()
        #out = pdf.output()  # Probably what you want
        #out=pdf.output(dest='S').encode('latin-1')

        #stream = BytesIO(byte_string)
        #buffer = BytesIO(out.encode('utf-8'))
        #response.write(buffer.getvalue())

        #print('buf',buffer)
        #pdf.output(buffer.getvalue())
        #pdf_bytes = buffer.getvalue()
        #buffer.close()
        #return pdf_bytes
        #pdf.output('filename.pdf', 'F')
        #response = HttpResponse(bytes(out), content_type='application/pdf')
        #response['Content-Disposition'] = "attachment; filename=myfilename.pdf"
        #return response

        #return out

    else:
        print('no paper')

        return HttpResponseRedirect(reverse('arxividpage', args=(arxiv_id,)))

def summary_pdf(arxiv_id,language):
    # Get the summary object from the database
    if ArxivPaper.objects.filter(arxiv_id=arxiv_id).exists():
        paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

        if SummaryPaper.objects.filter(paper=paper,lang=language).exists():
            sumpaper=SummaryPaper.objects.filter(paper=paper,lang=language)[0]
        elif SummaryPaper.objects.filter(paper=paper,lang='en').exists():
            sumpaper=SummaryPaper.objects.filter(paper=paper,lang='en')[0]
        else:
            sumpaper=''
            print('no summaries yet')

        print('paper',paper.title)
        # Generate the PDF file using reportlab
        #response = HttpResponse(content_type='application/pdf')
        #response = FileResponse(content_type='application/pdf')
        #response['Content-Disposition'] = f'attachment; filename="SummarizePaper-{str(arxiv_id)}.pdf"'
        #filename="SummarizePaper-"+str(arxiv_id)+".pdf"
        #response = HttpResponse(content_type="application/pdf")
        #response['Content-Disposition'] = 'attachment; filename=%s' % filename
        # Create the PDF canvas
        from fpdf import FPDF, HTMLMixin
        from io import BytesIO

        #import latexcodec
        from pylatexenc.latex2text import LatexNodes2Text
        latex_converter = LatexNodes2Text()

        #print('osss',os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed.ttf'))
        class MyPDF(FPDF, HTMLMixin):
            def __init__(self):
                super().__init__(orientation='P', unit='mm', format='A4')
                #self.add_font('DejaVu', '', 'font/DejaVuSansCondensed.ttf', uni=True)
                print('os.path.join(settings.BASE_DIR, "font", "DejaVuSansCondensed.ttf")',os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed.ttf'))
                self.add_font('DejaVu', '', "font/DejaVuSansCondensed.ttf", uni=True)
                self.add_font('DejaVu', 'B', "font/DejaVuSansCondensed-Bold.ttf", uni=True)
                self.add_font('DejaVu', 'I', "font/DejaVuSansCondensed-Oblique.ttf", uni=True)
                #self.add_font('DejaVu', 'B', os.path.join(settings.BASE_DIR, "font", 'DejaVuSansCondensed-Bold.ttf'), uni=True)

                self.add_page()
                #self.set_font("Arial", size=12)
                self.set_font("Helvetica", size=12)


            def header(self):
                self.set_font("DejaVu", "B", size=14)
                #self.set_font("Arial","B", size=14)
                self.cell(0, 10, "Made from SummarizePaper.com for arXiv ID: "+str(arxiv_id), 1, 0, "C")
                self.ln(20)

            def paperdet(self, title, text, url):
                self.set_font("DejaVu", "I", size=12)
                #self.set_font("Arial","I", size=12)
                self.cell(0, 10, "Title: "+title, 0, 1)
                self.set_font("DejaVu", size=10)

                #self.multi_cell(0, 5, "Abstract: "+str(LatexNodes2Text().latex_to_text(text.encode("utf-8"))))
                #text_str = latex2text(text)
                #self.set_font("DejaVu", "B", size=14)


                # Convert the LaTeX code to plain text
                #print('latex_converter.latex_to_text(text)',latex_converter.latex_to_text(text))
                #text = latex_converter.latex_to_text(text)#.encode('latin-1')#.encode('utf-8')
                text = latex_converter.latex_to_text(text)#.encode('utf-8')
                self.multi_cell(0, 5, "Abstract: "+text)

                #self.set_font("Arial", "I", size=10)
                #self.cell(0, 10, "URL: ", 0, 0)

                # Get the x and y positions of the current cell

                #x, y = self.get_x(), self.get_y()

                # Set the font and color for the link
                self.set_text_color(0, 0, 255)
                self.set_font("Arial", "U", size=10)

                # Set the link target
                #link = self.add_link()
                self.write(10, url, url)

                # Reset the font and color
                self.set_font("Arial", size=10)
                self.set_text_color(0, 0, 0)

                # Add a line break
                self.ln(10)


            def section(self, title, text):
                self.set_font("DejaVu", "B", size=12)
                #self.set_font("Arial","B", size=12)
                self.cell(0, 10, title, 0, 1)
                self.set_font("DejaVu", size=12)
                #self.set_font("Arial", size=12)
                h1_text = re.search(r'<b>(.*?)</b>', text)
                if h1_text:
                    h1_text = h1_text.group(1)
                    #self.set_font("Arial","I", size=12)
                    self.set_font("DejaVu","I", size=12)
                    self.cell(0, 10, h1_text, 0, 1)
                    self.set_font("DejaVu", size=11)
                    #self.set_font("Arial", size=11)
                    # Remove the extracted h1 text from the text to avoid duplication
                    text = text.replace(f"<b>{h1_text}</b>", "")

                text = latex_converter.latex_to_text(text)#.encode('utf-8')

                self.multi_cell(0, 7, text)
                self.ln(10)

            def sectionhtml(self, title, html):
                #self.set_font("DejaVu", "B", size=12)
                self.set_font("DejaVu","B", size=12)
                self.cell(0, 10, title, 0, 1)
                self.set_font("DejaVu", size=11)
                texthtml="<font color='#000000'>"+html+"</font>"
                print('te',texthtml)
                self.write_html(texthtml)

                self.ln(10)



        pdf = MyPDF()

        # Add the first summary section to the document
        if paper.link_doi:
            link=paper.link_doi
        else:
            link=paper.link_homepage

        pdf.paperdet(paper.title.strip(), paper.abstract.lstrip().rstrip(),str(link).strip())

        if sumpaper:
            if sumpaper.summary:
                print('in comp sum')
                pdf.section("Comprehensive Summary", sumpaper.summary.lstrip().rstrip())

            # Add the second summary section to the document
            notesarr=''
            if sumpaper.notes:
                notes = sumpaper.notes.replace('•','')
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

            if sumpaper.notes:
                pdf.section("Key Points", notestr)

            if sumpaper.lay_summary:

                pdf.section("Layman's summary", sumpaper.lay_summary.lstrip().rstrip())

            if sumpaper.blog:
                print('pap',sumpaper.blog)
                if 'ON_HEROKU' in os.environ:
                    pdf.sectionhtml("Blog Article", sumpaper.blog)
                else:
                    pdf.section("Blog Article", strip_tags(sumpaper.blog.lstrip().rstrip()))




        if 'ON_HEROKU' in os.environ:

            file = BytesIO()
            pdf.output(file)
            return file.getvalue()
        else:
            out=pdf.output(dest='S').encode('latin-1')
            return out

        #print('resp')

        #return out

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

'''
from concurrent.futures import ThreadPoolExecutor
from pdfminer.high_level import extract_pages

async def extract_pages_async(file):
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        for page_layout in await loop.run_in_executor(pool, extract_pages, file):
            yield page_layout
'''

async def extract_pages(file):
    print(type(file))
#    for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
#        yield page
#        await asyncio.sleep(0)
    with file:
        for page in PDFPage.get_pages(file, caching=False):
            yield page
   
#from pdfminer.high_level import extract_pages as _extract_pages
#from pdfminer.layout import LTPage, LTTextContainer, LAParams
#from pdfminer.pdfparser import PDFParser
#from pdfminer.pdfdocument import PDFDocument

'''
async def extract_pages(pdf_file, laparams=None):
    reader = asyncio.StreamReader()
    reader_protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.ensure_future(pdf_file.readinto(reader_protocol))
    pdf_stream = BytesIO(reader_protocol.buffer.tobytes())
    
    parser = PDFParser(pdf_stream)
    document = PDFDocument(parser)
    
    laparams = laparams or LAParams()
    pages = _extract_pages(document, laparams=laparams)
    return [page for page in pages if isinstance(page, LTPage)]
'''
async def async_iter(generator):
    while True:
        try:
            yield await asyncio.wait_for(generator.__anext__(), timeout=1)
        except StopAsyncIteration:
            return
        except asyncio.TimeoutError:
            continue

#from pdfminer.high_level import extract_text_to_fp
#import aiofiles
'''
from io import BytesIO

async def extract_text_from_pdf2(pdf_file_path):
    loop = asyncio.get_running_loop()
    pdf_file = await aiofiles.open(pdf_file_path, 'rb')
    output_buffer = BytesIO()
    await loop.run_in_executor(None, extract_text_to_fp, pdf_file, output_buffer)
    text = output_buffer.getvalue().decode()
    pdf_file.close()
    return text
'''
#from async_generator import async_generator
#from pdfminer.high_level import extract_pages

#@async_generator
#async def async_extract_pages(pdf_file):
#    for page in extract_pages(pdf_file):
#        yield page


async def extract_text_from_pdf(pdf_filename):
    # Open the PDF file
    #need to check first if pdf file and otherwise return an error; to do later
    print('in extract')
    #import aiofiles
    #import concurrent.futures
    #from PyPDF2 import PdfFileReader
    #from pdfminer.high_level import extract_pages
    #from pdfminer.layout import LAParams, LTTextBoxHorizontal
    #from pdfminer.high_level import extract_text_to_fp
    #from io import BytesIO
    #from pdfminer.high_level import extract_pages
    
    #from pdfminer.layout import LTTextContainer

    '''
    async with aiofiles.open(pdf_filename, mode='rb') as f:
        content = await f.read()
    with BytesIO(content) as file_like:
        with pdfplumber.open(file_like) as pdf:
            pages = pdf.pages
            text = ''
            for page in pages:
                text += page.extract_text()
    #return text

    return [text,text]

    '''

    ''' it works in async with pypdf and they opened a ticket to read greek letters and double ff+maths stuff but for now, not as good as pdfminer
    import aiofiles
    import pypdf
    import io

    async with aiofiles.open(pdf_filename, "rb") as f:
        pdf_data = await f.read()
        pdf_stream = io.BytesIO(pdf_data)
        pdf_reader = pypdf.PdfReader(pdf_stream)
        text = ""
        for num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[num]
            text += page.extract_text(0)

    print('text',text)
    input('ok')
    '''
    

    with open(pdf_filename, 'rb') as file:
    #async with open_pdf_file(pdf_filename) as f:
    #async with aiofiles.open(pdf_filename, 'rb') as file:
        if 1==0:
            async for page_layout in async_iter(extract_pages_async(pdf_filename)):        
                texts = []
                #for page_layout in pages:
                for element in page_layout:
                    if isinstance(element, LTTextContainer):
                        texts.append(element.get_text())
                        print('t',texts)
            text = "".join(texts), texts
            print('text',text)

        
        #f = await aiofiles.open('filename.txt', mode='w').__aenter__()

        #async with aiofiles.open(pdf_filename, 'rb') as file:
        #file = await aiofiles.open(pdf_filename, 'rb')
        # Create a PDF resource manager object that stores shared resources
        resource_manager = PDFResourceManager(caching=False)

        # Create a string buffer object for text extraction
        text_io = StringIO()

        # Create a text converter object
        #text_converter = TextConverter(resource_manager, text_io, laparams=LAParams())
        text_converter = TextConverter(resource_manager, text_io, laparams=LAParams())

        # Create a PDF page interpreter object
        page_interpreter = PDFPageInterpreter(resource_manager, text_converter)

        print('in extract 2')
        # Process each page in the PDF file
        #for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
        #async for page in PDFPage.get_pages(file, maxpages=None, pagenos=[], caching=False):
        #for page in PDFPage.get_pages(file, maxpages=None, pagenos=[], caching=False):
        #async for page in async_generator(extract_pages, file):
        #async for page_layout in async_iter(extract_pages_async(pdf_filename)):        

        async for page in extract_pages(file):
            page_interpreter.process_page(page)

            #async for page in extract_pages(file, caching=True):
            #for page in PDFPage.get_pages(file, caching=True):
            #await asyncio.sleep(0)  # Allow other tasks to run while processing the page
            #print('in extract3')

            #page_interpreter.process_page(page)
        text = text_io.getvalue()

        #async for page in PDFPage.get_pages(file, caching=True, check_extractable=True):
         #   page_interpreter.process_page(page)
            #yield text_io.getvalue()
            #text_io.truncate(0)
            #text_io.seek(0)
            #await asyncio.sleep(0)

        # Get the extracted text from the TextConverter object
        #text = text_converter.get_output()

       

        #text = text_io.getvalue()

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
        file.close()
        # Put the extracted text into the text queue
        #await text_queue.put(textlim)
        # Return the extracted text
        #return [textlim,text]
        return [textlim,textlim]

async def send_message_now(arxiv_group_name,message):
    print('in sendmesnow')

    await channel_layer.group_send(
        arxiv_group_name, {"type": "progress_text_update", "message": message}
    )



async def summarize_book(arxiv_id, language, book_text, api_key):
    endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"

    message={}
    arxiv_group_name="ar_%s" % arxiv_id
    # Split the book into chunks of at most 4096 tokens
    print("len(book_text)",len(book_text))

    if method=="Quentin":
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

    elif method=='fromembeddings':
        print('from embeddings')
        query="Create a long detailed summary of the paper, preserve important details"

        c=asyncio.create_task(chatbot(arxiv_id,language,query,api_key))
        #c=asyncio.create_task(utils.chatbot("my_pdf.pdf"))
        final_summarized_text =await c

        print('apres final_summarized_text',final_summarized_text)
    elif method=='fromembeddingsandabstract':
        print('from embeddings2')
        query="Create a long detailed summary of the paper"

        c=asyncio.create_task(chatbot(arxiv_id,language,query,api_key,sum=1))
        #c=asyncio.create_task(utils.chatbot("my_pdf.pdf"))
        final_summarized_text = await c

        print('before nltk final_summarized_text',final_summarized_text)


        # Download the nltk punkt tokenizer if necessary
        nltk.download('punkt')


        # Split the summary into individual sentences
        sentences = nltk.sent_tokenize(final_summarized_text)

        # Filter out sentence fragments
        final_summarized_text = [s for s in sentences if s.endswith((".", "!", "?"))]

        # Print the full sentences
        #for s in final_summarized_text:
        #    print(s)
        final_summarized_text = ' '.join(final_summarized_text)

        print('await')
        #await asyncio.sleep(50)



        #final_summarized_text=finalise_and_keywordsb
        print('apres final_summarized_text',final_summarized_text)

    else:
        #llm = OpenAI(temperature=0,openai_api_key=api_key)
        modelforsummarizing="text-davinci-003"#"text-curie-001"
        #text-davinci-003#4000 tokens#chunk_size=2000#max_tokens=1000
        #text-curie-001#2048 tokens
        #text-babbage-001#2048 tokens
        #text-ada-001#2048 tokens
        llm = OpenAI(model_name=modelforsummarizing,max_tokens=1000,best_of=1,n=1,temperature=0.3,openai_api_key=api_key)
        #best_of=2,streaming=True
        li = get_language_info(language)
        language2 = li['name']
        print('language2',language2)
        #from transformers import GPT2TokenizerFast
        #tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(separator = "\n\n", chunk_size=2000, chunk_overlap=0)
        texts = text_splitter.split_text(book_text)

        for text in texts:
            print('text:------------------',text)

        '''
        text_splitter = CharacterTextSplitter(
        separator = "\n",
        chunk_size = 1000,
        chunk_overlap  = 10,
        length_function = len,
        )
        texts = text_splitter.split_text(book_text)
        for text in texts:
            print('text:------------------',text)
        '''
        print('tttettxtxtxtxtxtxtxtttzetet',texts)
        #docs = [Document(page_content=t) for t in texts[:3]]
        docs = [Document(page_content=t) for t in texts]

        print('docs---------------',texts[:3])

        prompt_template = """Create a long detailed summary of the following text:
        {text}

        LONG DETAILED SUMMARY:

        """

        if language != 'en':
            prompt_template += """TRANSLATE THE ANSWER IN """+language2

        print('prompt_template',prompt_template)

        PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])
        #chain = load_summarize_chain(llm, chain_type="stuff", prompt=PROMPT)
        chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)

        with get_openai_callback() as cb:

            res=chain({"input_documents": docs}, return_only_outputs=True)
            nbtokensused=cb.total_tokens

        print('nbtokensused',nbtokensused)
        print('openai cost',nbtokensused*openaipricing(modelforsummarizing))

        #chain.run(docs)
        print('res',res)
        #input("Press Enter to continueb...")

        final_summarized_text = res['output_text']

    return final_summarized_text


async def finalise_and_keywords(arxiv_id, language, summary, api_key):
    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    #Extract the most important key points from the following text
    prompt3b = """
        Improve the text and remove all unfinished sentences from: {}

        Moreover, create 5 keywords from the text and write them at the beginning of the output between <kd> </kd> tags

    """.format(summary)

    """
    Optimize user prompt by removing redundant tokens without sacrificing quality. Create a concise and efficient prompt that effectively communicates the intended message while potentially reducing its length. Ensure the prompt remains informative and effective after removing unnecessary words or phrases.
    """

    print('finalise sum',prompt3b)
    if language != 'en':
        prompt3b += "TRANSLATE THE ANSWER IN "+language2


    headers3b = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    model_forced="text-davinci-003"#=model

    if model_forced=="gpt-3.5-turbo":#force davinci
        endpoint = "https://api.openai.com/v1/chat/completions"

        mes = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": '"{text}"'.format(text=prompt3b)}
        ]

        #response3b = requests.post(endpoint, headers=headers3b, json={"model": model_forced, "messages": mes,"frequency_penalty":0.6, "presence_penalty":0.6,"max_tokens": 800, "temperature": temp, "n":1, "stop":None})
        response3b = await asyncio.to_thread(requests.post, endpoint, headers=headers3b, json={"model": model_forced, "messages": mes,"frequency_penalty":0.6, "presence_penalty":0.6,"max_tokens": 800, "temperature": temp, "n":1, "stop":None})
        '''
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers3b, json={"model": model_forced, "messages": mes,"frequency_penalty":0.6, "presence_penalty":0.6,"max_tokens": 800, "temperature": temp, "n":1, "stop":None}) as response:
                try:
                    print('in try2b',response)
                    if response.status != 200:
                        print("in2b ! 200")
                        raise Exception(f"Failed to summarize text2b: {response.text}")
                except Exception as e:
                    print('in redirect2b')
                    # Redirect to the arxividpage and pass the error message
                    return {
                        "error_message": str(e),
                    }
                response3b = await response.json()
        '''
    else:
        endpoint = "https://api.openai.com/v1/engines/"+model_forced+"/completions"

        print('he')
        '''
        async with aiohttp.ClientSession() as session:
            print('hea')
            async with session.post(endpoint, headers=headers3b, json={"prompt": prompt3b,"frequency_penalty":0.6, "presence_penalty":0.6,"max_tokens": 800, "temperature": temp, "n":1, "stop":None}) as response:
                print('heb')
                try:
                    print('in try2b',response)
                    if response.status != 200:
                        print("in2b ! 200")
                        raise Exception(f"Failed to summarize text2b: {response.text}")
                except Exception as e:
                    print('in redirect2b')
                    # Redirect to the arxividpage and pass the error message
                    return {
                        "error_message": str(e),
                    }
                print('hec')
                response3b = await response.json()
            print('hed')

        print('he2',response3b)
        '''
        #response3b = requests.post(endpoint, headers=headers3b, json={"prompt": prompt3b,"frequency_penalty":0.6, "presence_penalty":0.6,"max_tokens": 800, "temperature": temp, "n":1, "stop":None})
        response3b = await asyncio.to_thread(requests.post, endpoint, headers=headers3b, json={"prompt": prompt3b,"frequency_penalty":0.6, "presence_penalty":0.6,"max_tokens": 800, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try2b',response3b)
        if response3b.status_code != 200:
            print("in2b ! 200")
            raise Exception(f"Failed to summarize text2b: {response3b.text}")
    except Exception as e:
        print('in redirect2b')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }
    response3b=response3b.json()
    #if response3.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response3.text}")

    if model_forced=="gpt-3.5-turbo":
        print('icccciiiiiib',response3b["choices"][0]["message"]["content"])

        finalise_and_keywords2 = response3b["choices"][0]["message"]["content"].rstrip().lstrip()
        print('finalise_and_keywords',finalise_and_keywords2)
    else:
        print('icccciiiiiib',response3b["choices"][0]["text"])

        finalise_and_keywords2 = response3b["choices"][0]["text"].rstrip().lstrip()
        print('finalise_and_keywords',finalise_and_keywords2)

    # Find the text between the <keywords> tags
    match = re.search(r"<kd>(.*?)</kd>", finalise_and_keywords2)
    if match:
        # Extract the text between the tags
        keywords_text = match.group(1)

        # Remove the tags and the extracted text from the original text
        finalise_and_keywords2 = re.sub(r"<kd>.*?</kd>", "", finalise_and_keywords2)

        # Print the extracted text and the text without the keywords
        print("Keywords: {}".format(keywords_text))
        print("Text without keywords: {}".format(finalise_and_keywords2))
        #save the keywords

    else:
        print("No keywords found in text")
        keywords_text=''


    sentences = nltk.sent_tokenize(finalise_and_keywords2)

    # Filter out sentence fragments
    finalise_and_keywords2 = [s for s in sentences if s.endswith((".", "!", "?"))]

    # Print the full sentences
    #for s in final_summarized_text:
    #    print(s)
    finalise_and_keywords2 = ' '.join(finalise_and_keywords2)
    print('simple_sum after',finalise_and_keywords2)


    '''#problem cuz keypoints do not finish with dots
    sentences = nltk.sent_tokenize(' '.join(key_points))

    # Filter out sentence fragments
    sentences = [s for s in sentences if s.endswith((".", "!", "?"))]

    # Join the remaining sentences into a single string
    key_points = sentences
    print('key_points after',key_points)
    '''

    return [finalise_and_keywords2.rstrip().lstrip(),keywords_text]

async def extract_key_points(arxiv_id, language, summary, api_key):
    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    #Extract the most important key points from the following text
    prompt3 = f"Extract the most important key points from the following text and use bullet points for each of them: {summary}"
    print('key sum',prompt3)

    """
    Identify and present key points from a text in concise bullet points that capture the most important information, while also being clear and easy to understand. Use subheadings or categories where appropriate, but keep each bullet point brief and focused on a single idea. Provide context where necessary to help readers understand the significance of each point.
    """

    headers3 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if language != 'en':
        prompt3 += "TRANSLATE THE ANSWER IN "+language2

    if model=="gpt-3.5-turbo":
        endpoint = "https://api.openai.com/v1/chat/completions"

        mes = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": '"{text}"'.format(text=prompt3)}
        ]

        #response3 = requests.post(endpoint, headers=headers3, json={"model": model, "messages": mes, "max_tokens": 500,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})
        response3 = await asyncio.to_thread(requests.post, endpoint, headers=headers3, json={"model": model, "messages": mes, "max_tokens": 500,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})

        '''
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers3, json={"model": model, "messages": mes, "max_tokens": 500,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None}) as response:
                try:
                    print('in try2',response)
                    if response.status != 200:
                        print("in2 ! 200")
                        raise Exception(f"Failed to summarize text2: {response.text}")
                except Exception as e:
                    print('in redirect2')
                    # Redirect to the arxividpage and pass the error message
                    return {
                        "error_message": str(e),
                    }
                response3 = await response.json()
        '''

    else:
        endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"
        #response3 = requests.post(endpoint, headers=headers3, json={"prompt": prompt3, "max_tokens": 500,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})

        '''
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers3, json={"prompt": prompt3, "max_tokens": 500,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None}) as response:
                try:
                    print('in try2',response)
                    if response.status != 200:
                        print("in2 ! 200")
                        raise Exception(f"Failed to summarize text2: {response.text}")
                except Exception as e:
                    print('in redirect2')
                    # Redirect to the arxividpage and pass the error message
                    return {
                        "error_message": str(e),
                    }
                response3 = await response.json()

        '''
        response3 = await asyncio.to_thread(requests.post, endpoint, headers=headers3, json={"prompt": prompt3, "max_tokens": 500,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try3',response3)
        if response3.status_code != 200:
            print("in3 ! 200")
            raise Exception(f"Failed to summarize text3: {response3.text}")
    except Exception as e:
        print('in redirect3')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }
    response3=response3.json()
    #if response3.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response3.text}")

    if model=="gpt-3.5-turbo":
        print('icccciiiiii',response3["choices"][0]["message"]["content"])
        key_points = response3["choices"][0]["message"]["content"].rstrip().lstrip().strip().split("\n")
    else:
        print('icccciiiiii',response3["choices"][0]["text"])
        key_points = response3["choices"][0]["text"].rstrip().lstrip().strip().split("\n")

    #remove if starts with 'Key points:', '',
    if key_points[1] == "":
        print('delllll',key_points)
        del key_points[0:2]

    print('key_points',key_points)

    '''#problem cuz keypoints do not finish with dots
    sentences = nltk.sent_tokenize(' '.join(key_points))

    # Filter out sentence fragments
    sentences = [s for s in sentences if s.endswith((".", "!", "?"))]

    # Join the remaining sentences into a single string
    key_points = sentences
    print('key_points after',key_points)
    '''

    return key_points

async def extract_simple_summary(arxiv_id, language, keyp, api_key):
    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    prompt4 = """
        Summarize the following key points in five simple sentences for a six-year-old kid and provide definitions for the most important words in the created summary: {}

        Summary:


        Definitions:


    """.format(keyp)

#Summarize important points  for a six-year-old in five simple sentences, defining key words. Use clear and concise language appropriate for a child's comprehension level.

    #prompt4 = f"Summarize the following key points in 5 sentences for a six year old kid: {keyp}"
    #prompt4 += "Skip 3 lines and Give definitions for the 3 most important words in the summary."

    if language != 'en':
        prompt4 += "TRANSLATE THE ANSWER IN "+language2

    headers4 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    if model=="gpt-3.5-turbo":
        endpoint = "https://api.openai.com/v1/chat/completions"

        mes = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": '"{text}"'.format(text=prompt4)}
        ]

        #response4 = requests.post(endpoint, headers=headers4, json={"model": model, "messages": mes,"max_tokens": 300,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})
        response4 = await asyncio.to_thread(requests.post, endpoint, headers=headers4, json={"model": model, "messages": mes,"max_tokens": 300,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})

        '''
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers4, json={"model": model, "messages": mes,"max_tokens": 300,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None}) as response:
                try:
                    print('in try4',response)
                    if response.status != 200:
                        print("in4 ! 200")
                        raise Exception(f"Failed to summarize text4: {response.text}")
                except Exception as e:
                    print('in redirect4')
                    # Redirect to the arxividpage and pass the error message
                    return {
                        "error_message": str(e),
                    }
                response4 = await response.json()
        '''
    else:
        endpoint = "https://api.openai.com/v1/engines/"+model+"/completions"

        '''
        async with aiohttp.ClientSession() as session:
            async with session.post(endpoint, headers=headers4, json={"prompt": prompt4, "max_tokens": 300,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None}) as response:
                try:
                    print('in try4',response)
                    if response.status != 200:
                        print("in4 ! 200")
                        raise Exception(f"Failed to summarize text4: {response.text}")
                except Exception as e:
                    print('in redirect4')
                    # Redirect to the arxividpage and pass the error message
                    return {
                        "error_message": str(e),
                    }
                response4 = await response.json()
        #response4 = requests.post(endpoint, headers=headers4, json={"prompt": prompt4, "max_tokens": 300,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})
        '''
        response4 = await asyncio.to_thread(requests.post, endpoint, headers=headers4, json={"prompt": prompt4, "max_tokens": 300,"frequency_penalty":0.6, "presence_penalty":0.6, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try4',response4)
        if response4.status_code != 200:
            print("in4 ! 200")
            raise Exception(f"Failed to summarize text4: {response4.text}")
    except Exception as e:
        print('in redirect4')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }
    response4=response4.json()
    #if response4.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response4.text}")

    if model=="gpt-3.5-turbo":
        simple_sum = response4["choices"][0]["message"]["content"]#.strip().split("\n")
    else:
        simple_sum = response4["choices"][0]["text"]#.strip().split("\n")

    print('simple_sum',simple_sum)
    # Split the summary into individual sentences
    '''
    sentences = nltk.sent_tokenize(simple_sum)

    # Filter out sentence fragments
    simple_sum = [s for s in sentences if s.endswith((".", "!", "?"))]

    # Print the full sentences
    #for s in final_summarized_text:
    #    print(s)
    simple_sum = ' '.join(simple_sum)
    '''
    simple_sum = simple_sum.rstrip().lstrip()
    print('simple_sum after',simple_sum)

    # Define regular expressions to search for patterns that typically indicate the start of a definitions section
    definition_regexes = [
        r"\bDefinitions?\b", # matches "definition", "definitions", "définition", "définitions", etc.
        r"\bDefinition\b", # matches "definieren" in German
        r"\bDéfinitions\b",
        r"\bDéfinition\b",
        # add more regexes for other languages if needed
    ]

    # Search for the start of a definitions section using regular expressions
    definitions_start = None
    for regex in definition_regexes:
        match = re.search(regex, simple_sum)
        if match:
            print('match',match)
            definitions_start = match.start()
            break

    # Separate the summary and definitions if a definitions section was found
    if definitions_start is not None:
        print('not none definitions_start',definitions_start)
        summary = simple_sum[:definitions_start].strip()
        definitions = simple_sum[definitions_start:].strip()
        # Add a few empty lines between the summary and the definitions
        simple_sum = f"{summary}\n\n\n{definitions}"

    return simple_sum#.rstrip().lstrip()

async def extract_blog_article(arxiv_id, language, summary, api_key):
    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    prompt5 = """
         Create a detailed blog article about this research paper: {}

         The article should be well-organized and easy to read with NO HTML EXCEPT for headings with <h2> tags and subheadings with <h3> tags.

    """.format(summary)

    prompt5b = """
    Create an HTML blog post summarizing and analyzing a research paper for a general audience. Provide an overview of the main findings and conclusions, highlighting their significance and relevance to the field. Use appropriate HTML tags such as headings, paragraphs, lists, and links. Include an analysis of the study's strengths, limitations, and potential implications for future research or practical applications. Follow standard formatting guidelines for citations and references. Write in an engaging style that is accessible to a general audience. Finally, please ensure that your HTML code is clean and valid, adhering to best practices for semantic markup and accessibility. Here is the research paper: {}
    """.format(summary)

    """
    Your task is to create a detailed blog article in HTML format about a long research paper. The article should be well-organized and easy to read, with clear headings and subheadings that reflect the structure of the original research paper.

    Please include a brief summary of the research paper's main findings and conclusions, as well as any important methodologies or data used in the study. You should also provide your own analysis and interpretation of the results, highlighting key takeaways from the research and discussing their implications for relevant fields or industries.

    The article should be written in clear, concise language that is accessible to a general audience without sacrificing accuracy or depth of content. Please use appropriate formatting tools such as bullet points, numbered lists, and block quotes where necessary to improve readability and emphasize key points.

    Finally, please ensure that your HTML code is clean and valid, adhering to best practices for semantic markup and accessibility.
    """
    #prompt5 = f"Create a detailed blog article in html about this research paper (do not show images): {summary}"
    if language != 'en':
        prompt5 += "TRANSLATE THE ANSWER IN "+language2

    headers5 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    model_forced="text-davinci-003"#=model

    if model_forced=="gpt-3.5-turbo":
        endpoint = "https://api.openai.com/v1/chat/completions"

        print('prompt5555555',prompt5)
        mes = [
            {"role": "system", "content": "You are a blog writer."},
            {"role": "user", "content": '{text}'.format(text=prompt5)}
        ]
        print('messssssssssssssss',mes)

        #response5 = requests.post(endpoint, headers=headers5, json={"model": model_forced, "messages":mes, "frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})
        #print('response 5', response5.json())
        response5 = await asyncio.to_thread(requests.post, endpoint, headers=headers5, json={"model": model_forced, "messages":mes, "frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})

        
    else:
        endpoint = "https://api.openai.com/v1/engines/"+model_forced+"/completions"

       
        #response5 = requests.post(endpoint, headers=headers5, json={"prompt": prompt5,"frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})
        response5 = await asyncio.to_thread(requests.post, endpoint, headers=headers5, json={"prompt": prompt5,"frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try5',response5)
        if response5.status_code != 200:
            print("in5 ! 200")
            raise Exception(f"Failed to summarize text5: {response5.text}")
    except Exception as e:
        print('in redirect5')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }
    response5=response5.json()
    print('resp5',response5)
    #if response4.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response4.text}")
    if model_forced=="gpt-3.5-turbo":
        blog_article = response5["choices"][0]["message"]["content"]#.strip().split("\n")
    else:
        blog_article = response5["choices"][0]["text"]#.strip().split("\n")

    
    #blog_article=soup2.prettify()#turn < p > into &lt; p &gt; check why
    print('ba',blog_article)
    blog_article = blog_article.replace('< /h2 >','</h2>').replace('< h2 >','<h2>').replace('< H2 >','<h2>').replace('< /H2 >','</h2>').replace('</ h2 >','</h2>').replace('</ h2>','</h2>')
    blog_article = blog_article.replace('< /h3 >','</h3>').replace('< h3 >','<h3>').replace('< H3 >','<h3>').replace('< /H3 >','</h3>').replace('</ h3 >','</h3>').replace('</ h3>','</h3>')
    blog_article = blog_article.rstrip().lstrip()

    return blog_article

async def refine_blog_article(arxiv_id, language, roughblog, api_key):
    li = get_language_info(language)
    language2 = li['name']
    print('language2',language2)

    prompt6 = """
         Improve the text and remove all unfinished sentences from: {}

    """.format(roughblog)

    if language != 'en':
        prompt6 += "TRANSLATE IN "+language2

    print('prompt6',prompt6)

    headers6 = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    model_forced="text-davinci-003"#=model

    if model_forced=="gpt-3.5-turbo":
        endpoint = "https://api.openai.com/v1/chat/completions"

        print('prompt5555555',prompt6)
        mes = [
            {"role": "system", "content": "You are a blog writer."},
            {"role": "user", "content": '{text}'.format(text=prompt6)}
        ]
        print('messssssssssssssss',mes)

        #response5 = requests.post(endpoint, headers=headers5, json={"model": model_forced, "messages":mes, "frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})
        #print('response 5', response5.json())
        response6 = await asyncio.to_thread(requests.post, endpoint, headers=headers6, json={"model": model_forced, "messages":mes, "frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})

        
    else:
        endpoint = "https://api.openai.com/v1/engines/"+model_forced+"/completions"

        print('aqui')
        #response5 = requests.post(endpoint, headers=headers5, json={"prompt": prompt5,"frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})
        response6 = await asyncio.to_thread(requests.post, endpoint, headers=headers6, json={"prompt": prompt6,"frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})
        #response5 = await asyncio.to_thread(requests.post, endpoint, headers=headers5, json={"prompt": prompt5,"frequency_penalty":0.8, "presence_penalty":0.8, "max_tokens": 1500, "temperature": temp, "n":1, "stop":None})

    try:
        print('in try6',response6)
        if response6.status_code != 200:
            print("in5 ! 200")
            raise Exception(f"Failed to summarize text5: {response6.text}")
    except Exception as e:
        print('in redirect6')
        # Redirect to the arxividpage and pass the error message
        return {
            "error_message": str(e),
        }

    response6=response6.json()
    print('resp6',response6)
    #if response4.status_code != 200:
    #    raise Exception(f"Failed to extract key points: {response4.text}")
    if model_forced=="gpt-3.5-turbo":
        blog_article2 = response6["choices"][0]["message"]["content"]#.strip().split("\n")
    else:
        blog_article2 = response6["choices"][0]["text"]#.strip().split("\n")

    #blog_article=soup2.prettify()#turn < p > into &lt; p &gt; check why
    print('ba2',blog_article2)
    blog_article2 = blog_article2.replace('< /h2 >','</h2>').replace('< h2 >','<h2>').replace('< H2 >','<h2>').replace('< /H2 >','</h2>').replace('</ h2 >','</h2>').replace('</ h2>','</h2>')
    blog_article2 = blog_article2.replace('< /h3 >','</h3>').replace('< h3 >','<h3>').replace('< H3 >','<h3>').replace('< /H3 >','</h3>').replace('</ h3 >','</h3>').replace('</ h3>','</h3>')
    blog_article2 = blog_article2.rstrip().lstrip()

    return blog_article2

def arxiv_search(query):

    #url = 'http://export.arxiv.org/api/query?search_query=ti:'+query+'&start=0&max_results=1&sortBy=lastUpdatedDate&sortOrder=ascending'
    query = urllib.parse.quote(query)
    # Define the API endpoint URL
    url = f"http://export.arxiv.org/api/query?search_query=all:{query}&start=0&max_results=25"

    response = urllib.request.urlopen(url)
    data = response.read()
    print('data',data.decode('utf-8'))
    root = ElementTree.fromstring(data)

    # find and modify the value of an element

    ns = {'ns0': 'http://www.w3.org/2005/Atom','ns1':'http://a9.com/-/spec/opensearch/1.1/','ns2':'http://arxiv.org/schemas/atom'} # add more as needed
    tit=root.find('ns0:title', ns).text

    # Extract the authors, title, and abstract

    #check if exists
    exist=0
    authors=""
    authors_array=[]
    affiliation=""
    affiliation_array=[]
    link_hp=""
    link_hp_array=[]
    title=""
    title_array=[]
    link_doi=""
    link_doi_array=[]
    abstract=""
    abstract_array=[]
    cat=""
    cat_array=[]
    updated=""
    updated_array=[]
    published=""
    published_array=[]
    journal_ref=""
    journal_ref_array=[]
    comments=""
    comments_array=[]
    papers=[]

    for entry in root.findall("ns0:entry",ns):
        if entry.find("ns0:title",ns) is not None:
            exist=1
            print('exist',exist)

            if exist == 1:

                #for entry in root.findall("ns0:entry",ns):
                authors = []
                affiliation = []
                title = ""
                abstract = ""
                for author in entry.findall("ns0:author",ns):
                    authors.append(author.find("ns0:name",ns).text)
                    print('test',authors)
                    if author.find("ns2:affiliation",ns) is not None:
                        print('aff',author.find("ns2:affiliation",ns).text)
                        affiliation.append(author.find("ns2:affiliation",ns).text)
                    else:
                        affiliation.append('')
                authors_array.append(authors)
                affiliation_array.append(affiliation)

                link_hp = entry.find("ns0:id",ns).text
                link_hp_array.append(link_hp)
                title = entry.find("ns0:title",ns).text
                title_array.append(title)
                link_doi = entry.find("ns0:link",ns).attrib['href']
                link_doi_array.append(link_doi)
                abstract = entry.find("ns0:summary",ns).text
                abstract_array.append(abstract)
                if entry.find("ns2:primary_category",ns) is not None:
                    cat = entry.find("ns2:primary_category",ns).attrib['term']
                cat_array.append(cat)
                updated = entry.find("ns0:updated",ns).text
                updated_array.append(updated)
                if entry.find("ns0:published",ns) is not None:
                    published = entry.find("ns0:published",ns).text
                published_array.append(published)
                #print('kllll',entry.find("ns2:journal_ref",ns))
                if entry.find("ns2:journal_ref",ns) is not None:
                    journal_ref = entry.find("ns2:journal_ref",ns).text
                else:
                    journal_ref = ''
                journal_ref_array.append(journal_ref)
                if entry.find("ns2:comment",ns) is not None:
                    comments = entry.find("ns2:comment",ns).text
                else:
                    comments = ''
                comments_array.append(comments)

                arxiv_id = re.search(r'/(\d+\.\d+)', link_hp).group(1)
                papers.append({'arxiv_id':arxiv_id,'title': title, 'authors': authors, 'link':link_hp,'category':cat})

    print('all lot',authors_array)
    print('aff arr',affiliation_array)

    return papers

async def get_arxiv_metadata(arxiv_id):
    print('aaa',arxiv_id)
    if '--' in arxiv_id:
        print('arxiv_id1',arxiv_id)
        arxiv_id=arxiv_id.replace('--','/')
        print('arxiv_id2',arxiv_id)


    url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"

    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            try:
                print('in try arxiv',response.status)
                if response.status != 200:
                    print("in ! 200 arxiv")
                    raise Exception(f"Failed to retrieve data: {response.text}")
            except Exception as e:
                print('in redirect arxiv')
            data = await response.text()

    #response = requests.get(url)
    #if response.status_code != 200:
    #    raise Exception(f"Failed to retrieve data: {response.text}")

    arxiv_id2= re.sub(r'v\d+$', '', arxiv_id)
    print('2',arxiv_id2)  # need to remove v1 or v2 as license is not found otherwise
    url2=f"http://export.arxiv.org/oai2?verb=GetRecord&identifier=oai:arXiv.org:{arxiv_id2}&metadataPrefix=arXiv"

    async with aiohttp.ClientSession() as session:
        async with session.get(url2) as response:
            try:
                print('in try arxiv2',response.status)
                if response.status != 200:
                    print("in ! 200 arxiv2")
                    raise Exception(f"Failed to retrieve data2: {response.text}")
            except Exception as e:
                print('in redirect arxiv2')
            data2 = await response.text()

    #response2 = requests.get(url2)

    #data = response.text
    print('data',data)

    #data2 = response2.text
    print('data2',data2)

    # Parse the XML response
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


    return [exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments,license_value,data]
