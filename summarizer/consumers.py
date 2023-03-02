from channels.generic.websocket import AsyncWebsocketConsumer,WebsocketConsumer
import json
import asyncio
from asgiref.sync import async_to_sync, sync_to_async
import summarizer.utils as utils
from channels.db import database_sync_to_async
from .models import ArxivPaper, Author, PaperAuthor, SummaryPaper
from datetime import datetime
import urllib.request
import requests
from django.conf import settings
import time
from django.core.cache import cache

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

    async def computesummary(self,arxiv_id,language,details_paper,message):
        url = 'https://arxiv.org/pdf/'+arxiv_id+'.pdf'
        book_path = "test1.pdf"
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
                response = requests.get(url)
                my_raw_data = response.content

                print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf')
                message["progress"] = 20
                if language == 'fr':
                    message["loading_message"] = "Lecture du fichier pdf..."
                else:
                    message["loading_message"] = "Reading the pdf file..."
                print('hhjfkfkkfkfkf')
                #time.sleep(10.)

                c=asyncio.create_task(self.send_message_now(message))
                await c

                #time.sleep(3.)

                #print('ookkkkkkkkkk')

                with open("my_pdf.pdf", 'wb') as my_data:
                    my_data.write(my_raw_data)

                ###book_text = utils.extract_text_from_pdf("my_pdf.pdf")

                #c=asyncio.create_task(utils.extract_text_from_pdf(book_path))
                c=asyncio.create_task(utils.extract_text_from_pdf("my_pdf.pdf"))
                book_text,full_text=await c
                print('book:',book_text)
            else:
                print('else book text')
                print('authors',authors)
                book_text='Authors: '+str(authors)+'. Title: '+title+'. Abstract: '+abstract+'.'



            message["progress"] = 30
            if language == 'fr':
                message["loading_message"] = "Indexation de l'article..."
            else:
                message["loading_message"] = "Indexing the paper..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c


            c = asyncio.create_task(sync_to_async(utils.getstorepickle)(arxiv_id))
            pickledata = await c

            if pickledata=='':
                if public==1:
                    book_text2=full_text
                else:
                    book_text2=book_text

                c = asyncio.create_task(utils.createindex(arxiv_id, book_text2, settings.OPENAI_KEY))
                created=await c
                print('crea',created)

            #input('stop here')

            message["progress"] = 40
            if language == 'fr':
                message["loading_message"] = "Création du résumé de l'article..."
            else:
                message["loading_message"] = "Summarizing the article..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c

            '''
            message["progress"] = 45
            if language == 'fr':
                message["loading_message"] = "Création du résumé en cours..."
            else:
                message["loading_message"] = "Summarizing in progress..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c
            '''


            c = asyncio.create_task(utils.summarize_book(arxiv_id, language, book_text, settings.OPENAI_KEY))
            sum = await c
            # Print the summarized text
            #await sum
            if sum != '':
                print('sum:',sum)
                #await task
                if 'error_message' in sum:
                    print("Received error message sum:", sum)
                    sum='Error: needs to be re-run'

                print('hfjggkg0a')



                c=asyncio.create_task(utils.finalise_and_keywords(arxiv_id, language, sum, settings.OPENAI_KEY))
                sum, kw = await c

                message["progress"] = 50
                if language == 'fr':
                    message["loading_message"] = "Extraction des points clefs de l'article..."
                else:
                    message["loading_message"] = "Extracting key points of the article..."
                #await self.send_message_now(message)

                print('hfjggkg0')

                c=asyncio.create_task(self.send_message_now(message))
                await c

                c=asyncio.create_task(self.send_message_sum(sum))
                await c



                message["progress"] = 60
                #message["loading_message"] = "Extracting key points..."
                if language == 'fr':
                    message["loading_message"] = "Création d'un résumé vulgarisé"
                else:
                    message["loading_message"] = "Creating a simple laymans' summary"
                c=asyncio.create_task(self.send_message_now(message))
                await c

                print('hfjggkg2')


                c = asyncio.create_task(utils.extract_key_points(arxiv_id, language, sum, settings.OPENAI_KEY))
                notes = await c
                if 'error_message' in notes:
                    print("Received error message notes:", notes)
                    notes='Error: needs to be re-run'



                c=asyncio.create_task(self.send_message_notes(notes))
                await c

                print('hfjggkg3')

                # Print the key points
                for key_point in notes:
                    print('note',key_point)



                c=asyncio.create_task(utils.extract_simple_summary(arxiv_id, language, notes, settings.OPENAI_KEY))
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

                c=asyncio.create_task(utils.extract_blog_article(arxiv_id, language, sum, settings.OPENAI_KEY))
                blog = await c
                print('blog:',blog)
                if 'error_message' in blog:
                    print("Received error message blog:", blog)
                    blog='Error: needs to be re-run'
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

    async def sendmessages(self,v,l,message):
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

        print('in sendmessages2')

        print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf1')

        message["progress"] = 5
        if l == 'fr':
            message["loading_message"] = "Chargement des informations du papier arXiv..."
        else:
            message["loading_message"] = "Loading the arXiv paper informartion..."

        c=asyncio.create_task(self.send_message_now(message))
        await c
        #await asyncio.sleep(10.)

        print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf1b')

        #time.sleep(3.)

        arxivarrayf=utils.get_arxiv_metadata(v)

        #exist, authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published, journal_ref, comments
        if len(arxivarrayf)>1:
            keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments','license']
            arxiv_dict = dict(zip(keys, arxivarrayf[1:]))
            exist=arxivarrayf[0]
            published_datetime = datetime.strptime(str(arxiv_dict['published_arxiv']), '%Y-%m-%dT%H:%M:%SZ')
            arxiv_dict['published_arxiv']=published_datetime

        #date = published_datetime.date()

        #print('published_datetime',published_datetime,date)

        if exist == 1:
            #arxivarray=[authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published_datetime, journal_ref, comments]

            #keys = ['authors', 'affiliation', 'link_homepage', 'title', 'link_doi', 'abstract', 'category', 'updated', 'published_arxiv', 'journal_ref', 'comments']

            #arxiv_dict = dict(zip(keys, arxivarray))
            #print('arxiv_dict',arxiv_dict)
            print('here')
            #time.sleep(30.)

            c = asyncio.create_task(sync_to_async(self.updatearvixdatapaper)(v,arxiv_dict))
            arvixupdate = await c
            #arvixupdate = await sync_to_async(self.updatearvixdatapaper)(v,arxiv_dict)

            print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf2')

            message["progress"] = 10
            if l == 'fr':
                message["loading_message"] = "Informations d'arXiv chargée..."
            else:
                message["loading_message"] = "The arXiv info are loaded..."

            c=asyncio.create_task(self.send_message_now(message))
            await c
            #time.sleep(10.)



            print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf2ba')


            published2 = published_datetime.strftime("%d %b. %Y")
            print('published2',published2)
            #arxiv_array2=[authors, affiliation, link_hp, title, link_doi, abstract, cat, updated, published2, journal_ref, comments]
            #arxiv_dict2 = dict(zip(keys, arxiv_array2))
            arxiv_dict2=arxiv_dict
            arxiv_dict2['published_arxiv']=published2
            c = asyncio.create_task(self.send_message_arxiv(arxiv_dict2))
            await c

        print('exist',exist)

        print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf3')

        message["progress"] = 15
        if l == 'fr':
            message["loading_message"] = "Conversion du texte initial..."
        else:
            message["loading_message"] = "Converting the input text..."
        c=asyncio.create_task(self.send_message_now(message))
        await c
        #time.sleep(10.)

        print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf3b')

        #now compute Summary
        print('avantcompute')
        #sumarray = await self.computesummary(v,message)
        detpap=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract'],arxiv_dict['authors']]
        sumarra=asyncio.create_task(self.computesummary(v,l,detpap,message))
        sumarray=await sumarra
        print('aprescompute')

        keysum = ['summary', 'lay_summary', 'notes','blog', 'keywords']
        sum_dict = dict(zip(keysum, sumarray))

        print('suma',sum_dict)
        c = asyncio.create_task(sync_to_async(self.updatesumpaper)(v,l,sum_dict))
        updatethesum = await c
        #print('testpap',testpaper)

        message["progress"] = 100
        if l == 'fr':
            message["loading_message"] = "L'article est maintenant traité - Regardez les résultats ci-dessous"
        else:
            message["loading_message"] = "The paper is now summarized - Look at the results down below"
        c=asyncio.create_task(self.send_message_now(message))
        await c


        '''
        for progress in utils.summarizer(v):
            message["progress"] = progress
            message["loading_message"] = "Loading data..."
            if message["progress"] == 100:
                message["loading_message"] = "Loading finished"
            await self.channel_layer.group_send(
                self.arxiv_group_name, {"type": "progress_update", "message": message}
            )
        '''

        #del self.sendmessages_running[self.arxiv_group_name]
        cache.set(self.arxiv_group_name, False)

    async def connect(self):
        self.arxiv_id = self.scope["url_route"]["kwargs"]["arxiv_id"]
        self.language = self.scope["url_route"]["kwargs"]["language"]

        print('self.arxiv_id',self.arxiv_id)
        print('self.language',self.language)

        self.arxiv_group_name = "ar_%s" % self.arxiv_id
        print('self.arxiv_group_name',self.arxiv_group_name)
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

        if 'command' in data:
            command = data['command']

            if command == 'start_background_task':
                message = {
                    "loading_message": "Loading...",
                    "progress": 0
                }
                v=self.arxiv_id
                l=self.language

                #ta=asyncio.ensure_future(self.sendmessages(v,message))#needed for channel_layer_group_send to work instantaneously
                #await self.sendmessages(v,message)
                #await ta
                asyncio.create_task(self.sendmessages(v,l,message))
        else:
            message = data["message"]
            print('receive',message)
            user = data["user"]
            print('receive',user)
            #response = "reponse brah"#process_message(message)
            print('avant chat bot')
            #query = "Create a summary and tell me who the authors are?"

            c=asyncio.create_task(utils.chatbot(self.arxiv_id,self.language,message,settings.OPENAI_KEY,user=user))
            #c=asyncio.create_task(utils.chatbot("my_pdf.pdf"))
            chatbot_text=await c
            print('apres chat bot',chatbot_text)

            #async for text_chunk in utils.chatbot(self.arxiv_id, self.language, message, settings.OPENAI_KEY):

            #    await self.send(text_data=json.dumps({
            #        'message': text_chunk
            #    }))

            if chatbot_text==None:
                chatbot_text="Something went wrong... Contact the administrators if it keeps on happening."
            # Send the response back to the client over the WebSocket connection

            await self.send(text_data=json.dumps({
                'message': chatbot_text.lstrip(": ")
            }))
            print('send 2',chatbot_text)


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

    async def disconnect(self, close_code):
        print('closed')
        cache.set(self.arxiv_group_name, False)

        await self.send(text_data=json.dumps({
            'type': 'disconnect',
            'message': 'Disconnected'
        }))
        await self.channel_layer.group_discard(self.arxiv_group_name, self.channel_name)
