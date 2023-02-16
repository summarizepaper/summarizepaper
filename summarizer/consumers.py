from channels.generic.websocket import AsyncWebsocketConsumer,WebsocketConsumer
import json
import asyncio
from asgiref.sync import async_to_sync, sync_to_async
import summarizer.utils as utils
from channels.db import database_sync_to_async
from .models import ArxivPaper, Author
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

    async def send_message_longsum(self,longsum):
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_longsum_update", "message": longsum}
        )

    async def send_message_blog(self,blog):
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_blog_update", "message": blog}
        )

    async def computesummary(self,arxiv_id,details_paper,message):
        url = 'https://arxiv.org/pdf/'+arxiv_id+'.pdf'
        book_path = "test1.pdf"
        sum=""
        longsum=""
        notes=""

        #details_paper=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract']]
        license,title,abstract=details_paper
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
                book_text=await c
                print('book:',book_text)
            else:
                print('else book text')
                book_text='Title: '+title+'. Abstract: '+abstract+'.'

            message["progress"] = 25
            message["loading_message"] = "Summarizing the article..."
            #await self.send_message_now(message)

            c=asyncio.create_task(self.send_message_now(message))
            await c

            c = asyncio.create_task(utils.summarize_book(arxiv_id, book_text, settings.OPENAI_KEY))
            sum = await c
            # Print the summarized text
            #await sum

            print('sum:',sum)
            #await task
            if 'error_message' in sum:
                # longsum contains the error_message
                #error_message = sum['error_message']
                # Handle error message
                #error_message = sum.split("error_message")[1]
                print("Received error message sum:", sum)
                sum='Error: needs to be re-run'
            c=asyncio.create_task(self.send_message_sum(sum))
            await c

            message["progress"] = 60
            #message["loading_message"] = "Extracting key points..."
            message["loading_message"] = "Creating a simple summary for a 10 year old..."
            c=asyncio.create_task(self.send_message_now(message))
            await c

            c = asyncio.create_task(utils.extract_key_points(arxiv_id, sum, settings.OPENAI_KEY))
            notes = await c
            if 'error_message' in notes:
                # longsum contains the error_message
                #error_message = notes['error_message']
                print("Received error message notes:", notes)
                notes='Error: needs to be re-run'
            c=asyncio.create_task(self.send_message_notes(notes))
            await c

            # Print the key points
            for key_point in notes:
                print('note',key_point)

            message["progress"] = 80
            message["loading_message"] = "Creating a blog-like article"
            #message["loading_message"] = "Creating a simple summary for a 10 year old..."
            c=asyncio.create_task(self.send_message_now(message))
            await c

            c=asyncio.create_task(utils.extract_simple_summary(arxiv_id, notes, settings.OPENAI_KEY))
            longsum = await c
            print('longsum:',longsum)
            if 'error_message' in longsum:
                # longsum contains the error_message
                #error_message = longsum['error_message']
                print("Received error message longsum:", longsum)
                longsum='Error: needs to be re-run'
            c=asyncio.create_task(self.send_message_longsum(longsum))
            await c

            message["progress"] = 90
            message["loading_message"] = "Wrapping this up..."
            c=asyncio.create_task(self.send_message_now(message))
            await c

            c=asyncio.create_task(utils.extract_blog_article(arxiv_id, sum, settings.OPENAI_KEY))
            blog = await c
            print('blog:',blog)
            if 'error_message' in blog:
                # longsum contains the error_message
                #error_message = blog['error_message']
                print("Received error message blog:", blog)
                blog='Error: needs to be re-run'
            c=asyncio.create_task(self.send_message_blog(blog))
            await c

            message["progress"] = 95
            message["loading_message"] = "Almost finished..."
            c=asyncio.create_task(self.send_message_now(message))
            await c

        suma=[sum,longsum,notes,blog]
        return suma#,longsum,notes

    def updatepaper(self,arxiv_id,sumarray):
        paper, created = ArxivPaper.objects.update_or_create(
            arxiv_id=arxiv_id,
            defaults={'summary': sumarray['summary'],'notes': sumarray['notes'],'longer_summary': sumarray['longer_summary'],'blog': sumarray['blog']}
        )
        return paper,created

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

        paper, created = ArxivPaper.objects.update_or_create(
            arxiv_id=arxiv_id,
            defaults={'link_homepage':arxiv_dict['link_homepage'], 'title':arxiv_dict['title'], 'link_doi':arxiv_dict['link_doi'], 'abstract':arxiv_dict['abstract'], 'category':arxiv_dict['category'], 'updated':arxiv_dict['updated'], 'published_arxiv':arxiv_dict['published_arxiv'], 'journal_ref':arxiv_dict['journal_ref'], 'comments':arxiv_dict['comments'],'license':arxiv_dict['license']}
        )
        authors_ids = [autho.id for autho in authors]
        paper.authors.set(authors_ids)
        #paper.authors.set(arxiv_dict['authors'])
        paper.save()

        #print('retdd',created)
        #'authors':arxiv_dict['authors'],'affiliation':arxiv_dict['affiliation']
        return paper,created

    async def sendmessages(self,v,message):
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
        message["loading_message"] = "Loading the arXiv paper informartion."
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
            message["loading_message"] = "arXiv info loaded..."

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
        message["loading_message"] = "Converting the input text..."
        c=asyncio.create_task(self.send_message_now(message))
        await c
        #time.sleep(10.)

        print('hhjfkfkkfkfkfcfjkndkjvhfdkjvnkgjfdnvkjf3b')

        #now compute Summary
        print('avantcompute')
        #sumarray = await self.computesummary(v,message)
        detpap=[arxiv_dict['license'],arxiv_dict['title'],arxiv_dict['abstract']]
        sumarra=asyncio.create_task(self.computesummary(v,detpap,message))
        sumarray=await sumarra
        #sum,longsum,notes=comsum
        #sumarray=[sum,longsum,notes]
        print('aprescompute')

        keysum = ['summary', 'longer_summary', 'notes','blog']
        sum_dict = dict(zip(keysum, sumarray))

        print('suma',sum_dict)
        c = asyncio.create_task(sync_to_async(self.updatepaper)(v,sum_dict))
        updatethesum = await c
        #print('testpap',testpaper)

        message["progress"] = 100
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
        print('self.arxiv_id',self.arxiv_id)
        self.arxiv_group_name = "ar_%s" % self.arxiv_id
        print('self.arxiv_group_name',self.arxiv_group_name)
        # Join room group
        await self.channel_layer.group_add(self.arxiv_group_name, self.channel_name)
        await self.accept()
        print('conn')

        message = {
            "loading_message": "Loading...",
            "progress": 0
        }

        v=self.arxiv_id

        #ta=asyncio.ensure_future(self.sendmessages(v,message))#needed for channel_layer_group_send to work instantaneously
        #await self.sendmessages(v,message)
        #await ta
        asyncio.create_task(self.sendmessages(v,message))

    # Receive message from WebSocket
    async def receive(self, text_data):
        text_data_json = json.loads(text_data)
        message = text_data_json["message"]
        print('receive')
        # Send message to room group
        await self.channel_layer.group_send(
            self.arxiv_group_name, {"type": "progress_update","message": message}
        )

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

    async def progress_longsum_update(self, event):
        print('progtextlongsum')

        await self.send(text_data=json.dumps({
            'type': 'progress_longsum_update',
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
