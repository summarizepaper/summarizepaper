from channels.generic.websocket import AsyncWebsocketConsumer,WebsocketConsumer
import json
import asyncio
from asgiref.sync import async_to_sync, sync_to_async
import summarizer.utils as utils
from summarizer import access, assistant, keypoints, similar
import logging

logger = logging.getLogger(__name__)
from channels.db import database_sync_to_async
from .models import ArxivPaper, Author, PaperAuthor, SummaryPaper
from datetime import datetime
import urllib.request
import requests
from django.conf import settings
import time
from django.core.cache import cache
import aiohttp

class EmbeddingConsumer(AsyncWebsocketConsumer):

    async def connect(self):
        await self.accept()

    async def disconnect(self, close_code):
        pass

    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        paper_id = text_data_json['paper_id']
        c = self.create_embeddings(paper_id)
        licenseurl = await c

        url = licenseurl
        print('url',url)

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
        else:
            public=False
            licenseurl="http://arxiv.org/licenses/assumed-1991-2003/" 
            cc_format="ASSUMED 1991-2003"

        await self.send(text_data=json.dumps({
            'message': 'completed',
            'paper_id': paper_id,
            'licenseurl':licenseurl,
            'licensecc':cc_format,
            'publiclicense':public
        }))

    async def create_embeddings(self, paper_id):
        # Here you can put the code to create embeddings for the given paper ID
        # First check if embeddings already there

        if not await sync_to_async(utils.has_paper_text)(paper_id):

            print(f"Creating embeddings for paper {paper_id}...")
            #time.sleep(1)
            url = 'https://arxiv.org/pdf/'+paper_id+'.pdf'
            book_path = None

            c=asyncio.create_task(utils.get_arxiv_metadata(paper_id))
            arxivarrayf = await c

            #exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments
            if len(arxivarrayf)>1:
                keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments','license']
                arxiv_dict = dict(zip(keys, arxivarrayf[1:-1]))
                exist=arxivarrayf[0]
                data=arxivarrayf[-1]
                print('a',arxiv_dict)
                print("arxiv_dict['published_arxiv']",arxiv_dict['published_arxiv'])
                published_datetime = datetime.strptime(str(arxiv_dict['published_arxiv']), '%Y-%m-%dT%H:%M:%SZ')
                arxiv_dict['published_arxiv']=published_datetime

            #date = published_datetime.date()

            #print('published_datetime',published_datetime,date)

            if exist == 1:
                #arxivarray=[authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published_datetime, journal_ref, comments]

                #keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments']

                #arxiv_dict = dict(zip(keys, arxivarray))
                #print('arxiv_dict',arxiv_dict)
                print('hered',paper_id)
                #time.sleep(30.)
                paper_id=paper_id.replace('/','--')
                print('hered2',paper_id)

                c = asyncio.create_task(sync_to_async(utils.updatearvixdatapaper)(paper_id,arxiv_dict))
                arvixupdate = await c

            detpap=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract'],arxiv_dict['authors']]


            #details_paper=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract']]
            license,title,abstract,authors=detpap
            #look at license first to see what can be done
            

            print('lic',license)
            if license=='http://creativecommons.org/licenses/by/4.0/' or license=='http://creativecommons.org/licenses/by-sa/4.0/' or license=='http://creativecommons.org/licenses/by-nc-sa/4.0/' or license=='http://creativecommons.org/publicdomain/zero/1.0/':
                public=1
            else:
                public=0


            print('pubbbllllliiiiiiccccc',public)
            active=1
            if active==1:
                print('active....')
                if public==1:
                    #response = requests.get(url)
                    book_text, text_source = await utils.read_paper(paper_id, url)

                    ###book_text = utils.extract_text_from_pdf(my_raw_data)
                    print('raw data done')
                    #c=asyncio.create_task(utils.extract_text_from_pdf(book_path))
                    full_text=book_text
                    print('book:',book_text)
                    

                else:
                    print('else book text')
                    print('authors',authors)
                    book_text='Authors: '+str(authors)+'. Title: '+title+'. Abstract: '+abstract+'.'
                    text_source='abstract'



                #c = asyncio.create_task(sync_to_async(utils.getstorepickle)(paper_id))
                #pickledata = await c

                #print('dattaaaaaaaaaaaa',data)
                if 1==1:#pickledata=='':
                    if public==1:
                        book_text2=data+'    '+full_text
                    else:
                        book_text2=data+'    '+book_text

                    c = asyncio.create_task(utils.createindex(paper_id, book_text2, settings.OPENAI_KEY, source=text_source))
                    created=await c
                    print('crea',created)

                #input('stop here')

            print(f"Embeddings created for paper {paper_id}")
        else:
            c = asyncio.create_task(sync_to_async(utils.getlicense)(paper_id))
            license = await c

            print(f"Embeddings already exist for paper {paper_id}")

        return license


    

class LoadingConsumer(AsyncWebsocketConsumer):
    sendmessages_running = {}


    async def send_message_now(self,message):
        print('in sendmesnow',message)

        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_text_update", "message": message}
        )
        '''
        await self.send(text_data=json.dumps({
            'type': 'progress_text_update',
            'message': message
        }))
        '''
        print('in sendmesnow2',message)
        #await asyncio.sleep(20)

    async def send_message_arxiv(self,arxiv_dict):

        print('in sendmesarxiv',arxiv_dict)

        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_arxiv_update", "message": arxiv_dict}
        )

        print('in sendmesarxiv2',arxiv_dict)
        #await asyncio.sleep(20)


    async def send_message_sum(self,sum):
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_sum_update", "message": sum}
        )

    async def send_message_notes(self,notes):
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_notes_update", "message": notes}
        )

    async def send_message_laysum(self,laysum):
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_laysum_update", "message": laysum}
        )

    async def send_message_blog(self,blog):
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_blog_update", "message": blog}
        )

    async def createindexwithsources(self,arxiv_id,details_paper,data):
        url = 'https://arxiv.org/pdf/'+arxiv_id+'.pdf'
        book_path = None
        sum=""
        laysum=""
        notes=""

        #details_paper=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract']]
        license,title,abstract,authors=details_paper
        #look at license first to see what can be done
        

        print('lic',license)
        if license=='http://creativecommons.org/licenses/by/4.0/' or license=='http://creativecommons.org/licenses/by-sa/4.0/' or license=='http://creativecommons.org/licenses/by-nc-sa/4.0/' or license=='http://creativecommons.org/publicdomain/zero/1.0/':
            public=1
        else:
            public=0


        print('pubbbllllliiiiiiccccc',public)
        active=1
        if active==1:
            print('active....')
            if public==1:#tbb
                '''
                async with aiohttp.ClientSession() as session:
                    async with session.get(url) as response:
                        await response.read()

                response = await asyncio.to_thread(requests.get, url, timeout=120)
                '''
                

                book_text, text_source = await utils.read_paper(arxiv_id, url)

                        
                print('ici')
                print('ici2')

                ###book_text = utils.extract_text_from_pdf(my_raw_data)

                #c=asyncio.create_task(utils.extract_text_from_pdf(book_path))
                    
                full_text=book_text
                print('book:',book_text)
            else:
                print('else book text')
                print('authors',authors)
                book_text='Authors: '+str(authors)+'. Title: '+title+'. Abstract: '+abstract+'.'
                text_source='abstract'

            #c = asyncio.create_task(sync_to_async(utils.getstorepickle)(arxiv_id))
            pickledata = ''#await c

            #print('dattaaaaaaaaaaaa',data)
            if pickledata=='':
                if public==1:
                    book_text2=data+'    '+full_text
                else:
                    book_text2=data+'    '+book_text

                c = asyncio.create_task(utils.createindex(arxiv_id, book_text2, settings.OPENAI_KEY, source=text_source))
                created=await c
                print('crea',created)
                print('finished',arxiv_id)

        return 0

    async def computesummary(self,arxiv_id,language,details_paper,message,data,p):
        url = 'https://arxiv.org/pdf/'+arxiv_id+'.pdf'
        book_path = None
        sum=""
        laysum=""
        notes=""

        #details_paper=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract']]
        license,title,abstract,authors=details_paper
        #look at license first to see what can be done
        '''
        https://creativecommons.org/licenses/by/4.0/ CC BY: Creative Commons Attribution
        This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use.

        https://creativecommons.org/licenses/by-sa/4.0/ CC BY-SA: Creative Commons Attribution-ShareAlike
        This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. The license allows for commercial use. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.

        https://creativecommons.org/licenses/by-nc-sa/4.0/ CC BY-NC-SA: Creative Commons Attribution-Noncommercial-ShareAlike
        This license allows reusers to distribute, remix, adapt, and build upon the material in any medium or format for noncommercial purposes only, and only so long as attribution is given to the creator. If you remix, adapt, or build upon the material, you must license the modified material under identical terms.

        https://creativecommons.org/licenses/by-nc-nd/4.0/ CC BY-NC-ND: Creative Commons Attribution-Noncommercial-NoDerivatives
        This license allows reusers to copy and distribute the material in any medium or format in unadapted form only, for noncommercial purposes only, and only so long as attribution is given to the creator.

        https://arxiv.org/licenses/nonexclusive-distrib/1.0/
        arXiv.org perpetual, non-exclusive license
        This license gives limited rights to arXiv to distribute the article, and also limits re-use of any type from other entities or individuals.

        http://creativecommons.org/publicdomain/zero/1.0/ CC Zero is a public dedication tool, which allows creators to give up their copyright and put their works into the worldwide public domain. CC0 allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, with no conditions.

        So only perpetual and CC BY-NC-ND should be done with metadata only
        '''

        print('lic',license)
        if license=='http://creativecommons.org/licenses/by/4.0/' or license=='http://creativecommons.org/licenses/by-sa/4.0/' or license=='http://creativecommons.org/licenses/by-nc-sa/4.0/' or license=='http://creativecommons.org/publicdomain/zero/1.0/':
            public=1
        else:
            public=0


        print('pubbbllllliiiiiiccccc',public)
        active=1
        if active==1:
            print('active....')
            if public==1:
                #response = requests.get(url)
                book_text, text_source = await utils.read_paper(arxiv_id, url)


                
                message["progress"] = 20
                if language == 'fr':
                    message["loading_message"] = "Lecture du fichier pdf..."
                else:
                    message["loading_message"] = "Reading the pdf file..."
                #time.sleep(10.)

                c=asyncio.create_task(self.send_message_now(message))
                await c

                #time.sleep(3.)

                ###book_text = utils.extract_text_from_pdf(my_raw_data)
                #c=asyncio.create_task(utils.extract_text_from_pdf(book_path))
                full_text=book_text
                #print('book:',book_text)
                

            else:
              
                book_text='Authors: '+str(authors)+'. Title: '+title+'. Abstract: '+abstract+'.'
                text_source='abstract'

            message["progress"] = 30
            if language == 'fr':
                message["loading_message"] = "Indexation de l'article..."
            else:
                message["loading_message"] = "Indexing the paper..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c


            #print('dattaaaaaaaaaaaa',data)
            if not await sync_to_async(utils.has_paper_text)(arxiv_id):
                if public==1:
                    book_text2=data+'    '+full_text
                else:
                    book_text2=data+'    '+book_text

                c = asyncio.create_task(utils.createindex(arxiv_id, book_text2, settings.OPENAI_KEY, source=text_source))
                created=await c
                #print('crea',created)

            #input('stop here')

            message["progress"] = 40
            if language == 'fr':
                message["loading_message"] = "Création du résumé de l'article..."
            else:
                message["loading_message"] = "Summarizing the article..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c



            sum=''
            kw=''
            #await asyncio.sleep(30.)
            c = asyncio.create_task(utils.summarize_book(arxiv_id, language, book_text, p, settings.OPENAI_KEY))
            sum = await c
            # Print the summarized text
            #await sum

            message["progress"] = 50
            if language == 'fr':
                message["loading_message"] = "Création du résumé en cours..."
            else:
                message["loading_message"] = "Summarizing in progress..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c

            if sum != '':
                print('sum:',sum)
                #await task
                if 'error_message' in sum:
                    print("Received error message sum:", sum)
                    sum='Error: needs to be re-run'



                c=asyncio.create_task(utils.finalise_and_keywords(arxiv_id, language, sum, p, settings.OPENAI_KEY))
                res = await c
                if 'error_message' in res:
                    print("Received error message finalise:", res)
                    sum, kw = 'Error: needs to be re-run', ''
                else:
                    sum, kw = res



                c=asyncio.create_task(self.send_message_sum(sum))
                await c

                message["progress"] = 60
                if language == 'fr':
                    message["loading_message"] = "Extraction des points clefs de l'article..."
                else:
                    message["loading_message"] = "Extracting key points of the article..."
                #await self.send_message_now(message)


                c=asyncio.create_task(self.send_message_now(message))
                await c

                c = asyncio.create_task(utils.extract_key_points(arxiv_id, language, sum, p, settings.OPENAI_KEY))
                notes = await c
                if 'error_message' in notes:
                    print("Received error message notes:", notes)
                    notes='Error: needs to be re-run'


                # Print the key points
                for key_point in notes:
                    print('note',key_point)

                shown = notes if isinstance(notes, str) else [keypoints.point_html(n) for n in notes]
                c=asyncio.create_task(self.send_message_notes(shown))
                await c



                print('hfjggkg2')
                message["progress"] = 70
                #message["loading_message"] = "Extracting key points..."
                if language == 'fr':
                    message["loading_message"] = "Création d'un résumé vulgarisé"
                else:
                    message["loading_message"] = "Creating a simple laymans' summary"
                c=asyncio.create_task(self.send_message_now(message))
                await c

                c=asyncio.create_task(utils.extract_simple_summary(arxiv_id, language, notes, p, settings.OPENAI_KEY))
                laysum = await c
                print('laysum:',laysum)
                if 'error_message' in laysum:
                    print("Received error message laysum:", laysum)
                    laysum='Error: needs to be re-run'




                c=asyncio.create_task(self.send_message_laysum(laysum))
                await c

                message["progress"] = 80
                if language == 'fr':
                    message["loading_message"] = "Création d'un article type blog"
                else:
                    message["loading_message"] = "Creating a blog-like article"
                #message["loading_message"] = "Creating a simple summary for a 10 year old..."
                c=asyncio.create_task(self.send_message_now(message))
                await c

                c=asyncio.create_task(utils.extract_blog_article(arxiv_id, language, sum, p, settings.OPENAI_KEY))
                #roughblog = await c
                blog = await c
                print('roughblog:',blog)
                if 'error_message' in blog:
                    print("Received error message blog:", blog)
                    blog='Error: needs to be re-run'
                
                '''
                cb=asyncio.create_task(utils.refine_blog_article(arxiv_id, language, roughblog, settings.OPENAI_KEY))
                blog = await cb
                print('blog:',blog)
                if 'error_message' in blog:
                    print("Received error message blog:", blog)
                    blog='Error: needs to be re-run'
                '''


                c=asyncio.create_task(self.send_message_blog(blog))
                await c

            else:
                sum='Error: needs to be re-run'
                laysum=sum
                notes=sum
                blog=sum

            message["progress"] = 90
            if language == 'fr':
                message["loading_message"] = "Presque terminé..."
            else:
                message["loading_message"] = "Almost finished..."
            c=asyncio.create_task(self.send_message_now(message))
            await c

        suma=[sum.replace(':\n', ''),laysum.replace(':\n', ''),notes,blog,kw]
        return suma

    def updatesumpaper(self,arxiv_id,language,sumarray):
        paper=ArxivPaper.objects.filter(arxiv_id=arxiv_id)[0]

        sumpaper, created = SummaryPaper.objects.update_or_create(
            paper=paper,lang=language,
            defaults={'summary': sumarray['summary'],'notes': sumarray['notes'],'lay_summary': sumarray['lay_summary'],'blog': sumarray['blog'], 'keywords': sumarray['keywords']}
        )
        return sumpaper,created

    '''
    def updatearvixdatapaper(self,arxiv_id,arxiv_dict):

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
    '''
    async def sendclose(self,v,l,message):
        try:
            await self._sendclose(v,l,message)
        except Exception:
            logger.exception('Closest papers search failed for %s', v)
            cache.set(self.arxiv_group_name+'-close', False)
            await self.send_failure(l)

    async def _sendclose(self,v,l,message):
        print('in sendclose')

        page_running = cache.get(self.arxiv_group_name+'-close')
        print('pr',page_running)

        if page_running:
        # Page is already running, do not start it again
            return
        else:
            # Page is not running, start it now and update the status in the cache
            cache.set(self.arxiv_group_name+'-close', True)
            #print('get',cache.get(self.arxiv_group_name))
       
        message["progress"] = 5
        if l == 'fr':
            message["loading_message"] = "Cherche les articles les plus proches..."
        else:
            message["loading_message"] = "Look for closest papers..."

        c=asyncio.create_task(self.send_message_now(message))
        await c

        print('clossseeee')
        message["progress"] = 50
        await self.send_message_now(message)
        await asyncio.to_thread(similar.find_and_store, v)

        #await asyncio.sleep(10.)
        
        message["progress"] = 100
        if l == 'fr':
            message["loading_message"] = "La recherche est terminée"
        else:
            message["loading_message"] = "The search is over"
        c=asyncio.create_task(self.send_message_now(message))
        await c

        cache.set(self.arxiv_group_name+'-close', False)


    async def sendmessages(self,v,l,message,p):
        try:
            await self._sendmessages(v,l,message,p)
        except Exception:
            logger.exception('Summary failed for %s', v)
            #release the "running" lock so the paper can be run again
            cache.set(self.arxiv_group_name, False)
            await self.send_failure(l)

    async def send_failure(self, l):
        if l == 'fr':
            text = "Une erreur s'est produite pendant le traitement. Réessayez plus tard ou contactez-nous."
        else:
            text = "Something went wrong while processing the paper. Please try again later or contact us."
        await self.send_message_now({"loading_message": text, "progress": 0})

    async def _sendmessages(self,v,l,message,p):
        print('in sendmessages')

        page_running = cache.get(self.arxiv_group_name)
        print('pr',page_running)

        if page_running:
        # Page is already running, do not start it again
            return
        else:
            # Page is not running, start it now and update the status in the cache
            cache.set(self.arxiv_group_name, True)
            print('get',cache.get(self.arxiv_group_name))
        #if self.arxiv_group_name in self.sendmessages_running:
        #    return

        #self.sendmessages_running[self.arxiv_group_name] = True

        #print('in sendmessages2')

        message["progress"] = 5
        if l == 'fr':
            message["loading_message"] = "Chargement des informations du papier arXiv..."
        else:
            message["loading_message"] = "Loading the arXiv paper information..."

        c=asyncio.create_task(self.send_message_now(message))
        await c
        #await asyncio.sleep(10.)


        #time.sleep(3.)

        #arxivarrayf=utils.get_arxiv_metadata(v)
        c=asyncio.create_task(utils.get_arxiv_metadata(v))
        arxivarrayf = await c

        #exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments
        if len(arxivarrayf)>1:
            keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments','license']
            arxiv_dict = dict(zip(keys, arxivarrayf[1:-1]))
            exist=arxivarrayf[0]
            data=arxivarrayf[-1]
            published_datetime = datetime.strptime(str(arxiv_dict['published_arxiv']), '%Y-%m-%dT%H:%M:%SZ')
            arxiv_dict['published_arxiv']=published_datetime

        #date = published_datetime.date()

        #print('published_datetime',published_datetime,date)

        if exist == 1:
  
            c = asyncio.create_task(sync_to_async(utils.updatearvixdatapaper)(v,arxiv_dict))
            arvixupdate = await c


            message["progress"] = 10
            if l == 'fr':
                message["loading_message"] = "Informations d'arXiv chargée..."
            else:
                message["loading_message"] = "The arXiv info are loaded..."

            c=asyncio.create_task(self.send_message_now(message))
            await c
            #time.sleep(10.)


            published2 = published_datetime.strftime("%d %b. %Y")
            print('published2',published2)
            #arxiv_array2=[authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published2, journal_ref, comments]
            #arxiv_dict2 = dict(zip(keys, arxiv_array2))
            arxiv_dict2=arxiv_dict
            arxiv_dict2['published_arxiv']=published2
            c = asyncio.create_task(self.send_message_arxiv(arxiv_dict2))
            await c

        message["progress"] = 15
        if l == 'fr':
            message["loading_message"] = "Conversion du texte initial..."
        else:
            message["loading_message"] = "Converting the input text..."
        c=asyncio.create_task(self.send_message_now(message))
        await c
        #time.sleep(10.)

        #now compute Summary
        #print('avantcompute')
        #sumarray = await self.computesummary(v,message)
        detpap=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract'],arxiv_dict['authors']]

        
        if 1==0:#v=='2303.14917v1':

            certain_date = datetime(2023, 3, 30)  # replace with your desired date
            #print('certain_date',certain_date)
            #c = asyncio.create_task(sync_to_async(storepickle)(arxiv_id,docstore_pickle,index_to_docstore_id_pickle,buffer_pickle))

            #c = asyncio.create_task(sync_to_async(utils.getstorepickle)(arxiv_id))
            #pickledata = await c

            c = asyncio.create_task(sync_to_async(utils.getallpaperstoredo)(certain_date))
            #c = asyncio.create_task(utils.getallpaperstoredo())
            allpaperstoredo = await c
            #allpaperstoredo = async_to_sync(utils.getallpaperstoredo)(certain_date)
            #allpaperstoredo = await utils.getallpaperstoredo(certain_date)

            #print('allpaperstoredo',allpaperstoredo)
            avoidids=[
                "2303.10344v1",
                "2303.15418v1",
                "2303.08525v1",
                "2301.05624v1",
                "2302.12824v1",
                "2303.03444v1",
                "1905.03844v1",
                "1607.03000v1",
                "0709.1389v1",
                "2302.12218v1",
                "2303.14917v1",
                "astro-ph--0511657v1",
                "2210.09995v1",
                "2209.12759v1",
                "2303.01496v1",
                "2303.01138v1",
                "2302.13303v1",
                "1505.06861v1",
                "2003.03466v1",
                "2303.02121v1",
                "2205.14520v1",
                "1911.04865v2",
                "1506.08667v1",
                "2303.00802v1",
                "1908.02718v1",
                "1405.7516v1",
                "2210.10909v1",
                "2303.0001v1",
                "2205.15937v1",
                "2212.07286v1",
                "2303.10130v1",
                "2303.0002v1",
                "2303.0004v1",
                "2303.04703v1",
                "2303.0003v1",
                "2303.04157v1",
                "2303.04802v1",
                "2303.08774v1",
                "2302.10346v1",
                "1302.1410v1",
                "1903.02723v1",
                "2106.07499v1",
                "2303.11315v1",
                "2303.01903v1",
                "hep-ph--9411346v1",
                "2302.10071v1",
                "2303.01177v1",
                "2301.03063v1",
                "2101.1234v1",
                "1612.06582v1",
                "1110.1365v3",
                "2204.12956v1",
                "2210.15353v1",
                "2302.10328v1",
                "2210.11424v1",
                "2004.06572v3",
                "1706.03762v5",
                "2303.03378v1",
                "2303.09549v1",
                "2303.04081v1",
                "2212.08073v1",
                "2303.08830v1",
                "2206.05802v1",
                "2212.03007v1",
                "2303.09014v1",
                "2009.09941v3",
                "2303.01248v2",
                '2110.03183v5', '2303.08774v2', '2302.04761v1', '2303.08437v1', '2302.10497v1', '2302.00021v1', '1509.02971v1', '2210.12275v1', '2303.1236v1', '2212.01218v1', '2212.04478v1', '2003.05714v1', '2210.15486v2', '2206.00036v1', '2211.04191v1', '2303.02891v1', '2303.10130v2', '2303.00495v1', '1411.1173v1', '1707.08941v1', '2203.06016v1', '2303.11525v1', '2212.14024v1', '2303.08803v1', '2303.08698v1', '2206.07840v1', '2303.12712v1', '2004.01204v1', '1706.03762v1', '2303.12773v1','2303.04192v1','2108.08222v1', '2212.06659v1', '2303.05961v1', '2008.11720v1', '2303.08939v1', '2303.01129v2', '2303.12970v1', '2007.07068v1', '2302.11382v1', '2209.02057v3', '2303.13002v1', '2303.13517v1', '1611.03360v2', '2303.13584v1', '2302.14574v1', '2302.13971v1', '2303.13502v1', '2303.15352v1', '2201.13209v1', '2209.04874v2', '2302.06566v1', '2303.15712v1','2302.08466v1'
            ]

            storeidsa=[]
            async for all in allpaperstoredo:
                print('all',all)
                if all.arxiv_id not in avoidids:
                    storeidsa.append(all.arxiv_id)
                    c=asyncio.create_task(utils.get_arxiv_metadata(all.arxiv_id))
                    arxivarrayf = await c

                    #exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments
                    if len(arxivarrayf)>1:
                        keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments','license']
                        arxiv_dict = dict(zip(keys, arxivarrayf[1:-1]))
                        exist=arxivarrayf[0]
                        data=arxivarrayf[-1]
                        #published_datetime = datetime.strptime(str(arxiv_dict['published_arxiv']), '%Y-%m-%dT%H:%M:%SZ')
                        #arxiv_dict['published_arxiv']=published_datetime
                        detpap=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract'],arxiv_dict['authors']]

                        rewrite=asyncio.create_task(self.createindexwithsources(all.arxiv_id,detpap,data))
                        rewritey=await rewrite
                        print('rewriteb')
                        print('storeidsa',storeidsa)
                        #input('ok')

        sumarra=asyncio.create_task(self.computesummary(v,l,detpap,message,data,p))
        sumarray=await sumarra
        print('aprescompute')

        keysum = ['summary', 'lay_summary', 'notes','blog', 'keywords']
        sum_dict = dict(zip(keysum, sumarray))

        #print('suma',sum_dict)
        c = asyncio.create_task(sync_to_async(self.updatesumpaper)(v,l,sum_dict))
        updatethesum = await c
        #print('testpap',testpaper)

        message["progress"] = 95
        if l == 'fr':
            message["loading_message"] = "Recherche des papiers similaires"
        else:
            message["loading_message"] = "Finding similar papers"
        c=asyncio.create_task(self.send_message_now(message))
        await c

        await asyncio.to_thread(similar.find_and_store, v)

        message["progress"] = 100
        if l == 'fr':
            message["loading_message"] = "L'article est maintenant traité - Regardez les résultats ci-dessous"
        else:
            message["loading_message"] = "The paper is now summarized - Look at the results down below"
        c=asyncio.create_task(self.send_message_now(message))
        await c


        #del self.sendmessages_running[self.arxiv_group_name]
        cache.set(self.arxiv_group_name, False)

    async def connect(self):
        self.arxiv_id = self.scope["url_route"]["kwargs"]["arxiv_id"]
        self.language = self.scope["url_route"]["kwargs"]["language"]

        print('self.arxiv_id',self.arxiv_id)
        print('self.language',self.language)

        self.arxiv_group_name = "ar_%s" % self.arxiv_id
        print('self.arxiv_group_name',self.arxiv_group_name)

        #identity and paid status come from the session cookie, never from the client messages
        self.user = self.scope.get("user")
        self.hashed_ip = access.hash_ip(access.client_ip_from_scope(self.scope))
        self.paid = await database_sync_to_async(access.is_paid)(self.user)
        # Join room group
        await self.channel_layer.group_add(self.arxiv_group_name, self.channel_name)
        await self.accept()
        print('conn')

        #message = {
        #    "loading_message": "Loading...",
        #    "progress": 0
        #}

        #v=self.arxiv_id
        #l=self.language

        #ta=asyncio.ensure_future(self.sendmessages(v,message))#needed for channel_layer_group_send to work instantaneously
        #await self.sendmessages(v,message)
        #await ta
        #asyncio.create_task(self.sendmessages(v,l,message))

    # Receive message from WebSocket
    async def receive(self, text_data):
        print('rec',text_data)
        data = json.loads(text_data)

        paid = self.paid

        if 'command' in data:
            command = data['command']

            if command == 'start_background_task':
                #only start what the HTTP view allowed (quota checked there), otherwise just listen to the group
                if not await self.consume_run('run'):
                    return
                message = {
                    "loading_message": "Loading...",
                    "progress": 0
                }
                v=self.arxiv_id
                l=self.language

                #ta=asyncio.ensure_future(self.sendmessages(v,message))#needed for channel_layer_group_send to work instantaneously
                #await self.sendmessages(v,message)
                #await ta
                #if paid:
                #    print('paiddddddd')

                asyncio.create_task(self.sendmessages(v,l,message,paid))
            elif command == 'start_close_task':
                if not await self.consume_run('close'):
                    return
                message = {
                    "loading_message": "Loading...",
                    "progress": 0
                }
                v=self.arxiv_id
                l=self.language

                #ta=asyncio.ensure_future(self.sendmessages(v,message))#needed for channel_layer_group_send to work instantaneously
                #await self.sendmessages(v,message)
                #await ta
                asyncio.create_task(self.sendclose(v,l,message))

        else:
            message = data["message"]
            print('receive',message)
            selectedpapers=''

            if await database_sync_to_async(access.chat_quota_left)(self.user, self.hashed_ip) <= 0:
                if self.language == 'fr':
                    limit_text = "Vous avez atteint la limite quotidienne de questions. Revenez demain ou passez au compte Premium."
                else:
                    limit_text = "You have reached the daily question limit. Come back tomorrow or upgrade to Premium."
                await self.send(text_data=json.dumps({'message': limit_text}))
                return

            if "selectedpapers" in data:
                selectedpapers=data["selectedpapers"]
                print('selectedpapers',selectedpapers)

            #the multi-paper chat page sends the papers to discuss, a paper page discusses its own paper
            if selectedpapers:
                arxiv_ids = [str(p).replace('/', '--') for p in selectedpapers][:10]
            else:
                arxiv_ids = [self.arxiv_id]

            async def send_delta(delta):
                await self.send(text_data=json.dumps({'type': 'chat_delta', 'delta': delta}))

            try:
                chatbot_text = await assistant.answer(
                    arxiv_ids, self.language, message, paid, user=self.user, hashed_ip=self.hashed_ip,
                    on_delta=send_delta, conversation_id=self.arxiv_id)
            except Exception:
                logger.exception('Assistant failed for %s', self.arxiv_id)
                chatbot_text = None

            if not chatbot_text:
                if self.language == 'fr':
                    chatbot_text = "Une erreur s'est produite. Contactez-nous si le problème persiste."
                else:
                    chatbot_text = "Something went wrong... Contact the administrators if it keeps on happening."

            await self.send(text_data=json.dumps({
                'message': chatbot_text
            }))


    async def progress_text_update(self, event):
        print('progtext',event['message'])

        await self.send(text_data=json.dumps({
            'type': 'progress_text_update',
            'message': event['message']
        }))

    async def progress_arxiv_update(self, event):
        print('progtextarxiv')

        await self.send(text_data=json.dumps({
            'type': 'progress_arxiv_update',
            'message': event['message']
        }))

    async def progress_sum_update(self, event):
        print('progtextsum')

        await self.send(text_data=json.dumps({
            'type': 'progress_sum_update',
            'message': event['message']
        }))

    async def progress_notes_update(self, event):
        print('progtextnotes')

        await self.send(text_data=json.dumps({
            'type': 'progress_notes_update',
            'message': event['message']
        }))

    async def progress_laysum_update(self, event):
        print('progtextlaysum')

        await self.send(text_data=json.dumps({
            'type': 'progress_laysum_update',
            'message': event['message']
        }))

    async def progress_blog_update(self, event):
        print('progtextblog')

        await self.send(text_data=json.dumps({
            'type': 'progress_blog_update',
            'message': event['message']
        }))


    async def progress_update(self, event):
        print('prog')

        await self.send(text_data=json.dumps({
            'type': 'progress_update',
            'message': event['message']
        }))

    async def consume_run(self, kind):
        session = self.scope.get("session")
        if session is None:
            return False
        allowed = await database_sync_to_async(access.consume_run)(session, self.arxiv_id, kind)
        if allowed:
            await database_sync_to_async(session.save)()
        return allowed

    async def disconnect(self, close_code):
        print('closed')
        cache.set(self.arxiv_group_name, False)

        await self.send(text_data=json.dumps({
            'type': 'disconnect',
            'message': 'Disconnected'
        }))
        await self.channel_layer.group_discard(self.arxiv_group_name, self.channel_name)
