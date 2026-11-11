"""Tests of the exports (BibTeX, RIS, Obsidian, Notion) and of Zotero: sending without duplicates, collections kept
in sync both ways, a bibliography check and a literature review written back, the pickers and the connection."""
import datetime
import io
import json
import re
import zipfile
from unittest import mock
from urllib.parse import parse_qs, urlparse

from django.test import Client, override_settings
from django.utils import timezone

from . import exports, publication_status, views, zotero
from .models import ArxivPaper, Collection, CollectionPaper, Generated, ZoteroAccount
from .test_tools import ToolsTestCase


class FakeZotero:
    """The Zotero web API in memory: one library of items and collections, the keys endpoint, the groups."""

    def __init__(self, user='123', write=True, groups=()):
        self.user, self.write, self.groups = user, write, list(groups)
        self.items, self.collections, self.calls, self.count = {}, {}, [], 0
        self.version, self.deleted, self.fulltext = 1, {}, {}  # the library's version; deleted keys; attachment texts
        self.patches = [mock.patch('summarizer.zotero.requests.request', side_effect=self.request),
                        mock.patch('summarizer.zotero.requests.get', side_effect=self.get),
                        mock.patch('summarizer.zotero.time.sleep')]

    def __enter__(self):
        for p in self.patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()

    def new_key(self):
        self.count += 1
        return 'K%07d' % self.count

    def add(self, data):
        key = self.new_key()
        data = dict(data, key=key, collections=list(data.get('collections', [])), tags=list(data.get('tags', [])))
        self.version += 1
        self.items[key] = {'key': key, 'version': self.version, 'data': data}
        return key

    def change(self, key, **fields):
        """The researcher changes an item in Zotero."""
        self.version += 1
        self.items[key]['data'].update(fields)
        self.items[key]['version'] = self.version

    def delete(self, key):
        self.version += 1
        self.deleted[self.items.pop(key)['key']] = self.version

    def collection(self, name, parent=''):
        key = self.new_key()
        self.collections[key] = {'name': name, 'parent': parent}
        return key

    def only_item(self):
        found = [i['data'] for i in self.items.values() if i['data']['itemType'] != 'note']
        assert len(found) == 1, found
        return found[0]

    def notes_of(self, key):
        return [i['data']['note'] for i in self.items.values() if i['data'].get('parentItem') == key]

    def top(self, collection=None, trashed=False):
        return [i for i in self.items.values() if i['data']['itemType'] not in ('note', 'attachment')
                and not i['data'].get('parentItem') and (trashed or not i['data'].get('deleted'))
                and (collection is None or collection in i['data']['collections'])]

    def answer(self, body=None, status=200):
        return mock.Mock(status_code=status, headers={'Last-Modified-Version': str(self.version)}, json=lambda: body,
                         content=body if isinstance(body, bytes) else b'',
                         text=body if isinstance(body, str) else json.dumps(body) if not isinstance(body, bytes) else '')

    def with_bibtex(self, items, params):
        if 'bibtex' not in params.get('include', ''):
            return items
        return [dict(i, bibtex='@article{%s_%s,\n  title={%s}\n}' % (
            i['data'].get('creators', [{}])[0].get('lastName', 'x').lower(), i['key'].lower(), i['data'].get('title', '')))
            for i in items]

    def get(self, url, **kwargs):
        if '/keys/' in url:
            return self.answer({'userID': int(self.user), 'username': 'ann_z',
                                'access': {'user': {'library': True, 'write': self.write},
                                           'groups': {'all': {'library': True, 'write': True}} if self.groups else {}}})
        if url.endswith('/groups'):
            return self.answer([{'id': gid, 'data': {'name': name}} for gid, name in self.groups])
        return self.answer(None, 404)

    def request(self, method, url, params=None, json=None, headers=None, timeout=None):
        path = url.split('api.zotero.org/', 1)[1].split('/')[2:]
        params = params or {}
        self.calls.append((method, '/'.join(path)))
        start, limit = int(params.get('start', 0)), int(params.get('limit', 100))
        if method == 'GET':
            if path == ['collections']:
                rows = [{'key': k, 'data': {'name': c['name'], 'parentCollection': c['parent'] or False},
                         'meta': {'numItems': len(self.top(k))}} for k, c in self.collections.items()]
                return self.answer(rows[start:start + limit])
            if path[0] == 'collections' and path[2:] == ['items', 'top']:
                return self.answer(self.with_bibtex(self.top(path[1]), params)[start:start + limit])
            if path == ['items', 'top'] and 'since' in params:
                found = [i for i in self.top(trashed=bool(params.get('includeTrashed'))) if i['version'] > int(params['since'])]
                return self.answer(found[start:start + limit])
            if path == ['deleted']:
                return self.answer({'items': [k for k, v in self.deleted.items() if v > int(params['since'])]})
            if path[0] == 'items' and path[2:] == ['fulltext']:
                return self.answer({'content': self.fulltext[path[1]]}) if path[1] in self.fulltext else self.answer({}, 404)
            if path == ['items', 'top']:  # quick search: titles, creators and years only, as the real API
                query = params.get('q', '').lower()
                fields = lambda d: [d.get('title', ''), str(d.get('date', ''))] + \
                    [c.get('lastName') or c.get('name', '') for c in d.get('creators', [])]
                return self.answer([i for i in self.top() if any(query in f.lower() for f in fields(i['data']))])
            if path[0] == 'items' and len(path) == 2:
                return self.answer(self.items[path[1]]) if path[1] in self.items else self.answer({}, 404)
            if path[0] == 'items' and path[2:] == ['children']:
                return self.answer([i for i in self.items.values() if i['data'].get('parentItem') == path[1]])
        if method == 'POST' and path == ['items']:
            return self.answer({'successful': {str(n): {'key': self.add(obj)} for n, obj in enumerate(json)}})
        if method == 'POST' and path == ['collections']:
            return self.answer({'successful': {str(n): {'key': self.collection(c['name'], c.get('parentCollection', ''))}
                                               for n, c in enumerate(json)}})
        if method == 'PATCH' and path[0] == 'items':
            self.change(path[1], **json)
            return self.answer(None, 204)
        return self.answer({}, 404)


class ExportTests(ToolsTestCase):
    def test_bibtex_keys_are_unique_and_titles_protected(self):
        self.paper('2101.00001v1', title='Clouds: GPT & JWST see 50% of clouds on $T_{eq}$ planets')
        self.paper('2101.00002v1', title='Clouds again', summary='')
        user = self.user(paid=True)
        collection = Collection.objects.create(user=user, name='Hot planets')
        for aid in ('2101.00001v1', '2101.00002v1'):
            CollectionPaper.objects.create(collection=collection, arxiv_id=aid, title='t', authors='Ann Lee')
        bib = self.client_for(user).get('/en/collections/%d/export/bib/' % collection.pk).content.decode()
        # the first paper got its key before the second shared it: keys never change once given
        self.assertEqual(re.findall(r'@misc\{([^,]+),', bib), ['lee2021clouds', 'lee2021cloudsb'])
        self.assertIn('title={Clouds: {GPT} \\& {JWST} see 50\\% of clouds on $T_{eq}$ planets}', bib)

    def test_a_published_preprint_is_cited_as_published(self):
        paper = self.paper()
        publication_status.Generated.objects.create(kind=publication_status.KIND, key=paper.arxiv_id, lang='en', status='done',
                                                    content=json.dumps({'status': 'published', 'doi': '10.3847/1538-4357/ab1234',
                                                                        'journal': 'The Astrophysical Journal', 'date': '2022-03-01'}))
        meta = exports.paper_meta(paper.arxiv_id)
        bib = exports.bibtex_entry(meta)
        self.assertTrue(bib.startswith('@article{lee2022clouds,'))  # the year of its year field
        for field in ('journal={The Astrophysical Journal}', 'year={2022}', 'eprint={2101.00001}', 'doi={10.3847/1538-4357/ab1234}'):
            self.assertIn(field, bib)
        ris = exports.ris_entry(meta)
        self.assertIn('TY  - JOUR', ris)
        self.assertIn('T2  - The Astrophysical Journal', ris)
        self.assertNotIn('PB  - arXiv', ris)
        item = zotero.item_for(meta, 'lee2021clouds')
        self.assertEqual((item['itemType'], item['publicationTitle'], item['DOI']), ('journalArticle', 'The Astrophysical Journal',
                                                                                    '10.3847/1538-4357/ab1234'))
        self.assertIn('arXiv: 2101.00001', item['extra'])
        tags = dict(views.citation_meta(paper))
        self.assertEqual((tags['citation_journal_title'], tags['citation_doi']), ('The Astrophysical Journal', '10.3847/1538-4357/ab1234'))
        self.assertNotIn('citation_publisher', tags)

    def test_markdown_for_obsidian_and_notion(self):
        self.paper(authors=['Ann Lee'] + ['Coauthor Number%d' % i for i in range(24)])
        md = Client().get('/en/export/2101.00001v1/md/')
        self.assertIn('lee2021clouds.md', md['Content-Disposition'])
        text = md.content.decode()
        self.assertTrue(text.startswith('---\ntitle: "Clouds on a hot planet"\nauthors:\n  - "Ann Lee"'))
        self.assertIn('citekey: "lee2021clouds"', text)
        self.assertIn('arxiv: "2101.00001"', text)
        self.assertIn('Coauthor Number18 et al.', text)
        self.assertIn('Summary written by an AI model (SummarizePaper) from arXiv:2101.00001v1 on', text)
        notion = Client().get('/en/export/2101.00001v1/notion/')
        self.assertIn('filename="Clouds on a hot planet.md"', notion['Content-Disposition'])
        self.assertTrue(notion.content.decode().startswith('**Authors:** Ann Lee'))
        self.assertIn('**Auteurs :** Ann Lee', Client().get('/fr/export/2101.00001v1/notion/').content.decode())

    def test_a_collection_downloads_as_an_obsidian_folder(self):
        self.paper()
        user = self.user()
        collection = Collection.objects.create(user=user, name='Hot planets')
        CollectionPaper.objects.create(collection=collection, arxiv_id='2101.00001v1', title='t', authors='Ann Lee',
                                       note='Read twice', tags='clouds, hot jupiters', reading='key')
        response = self.client_for(user).get('/en/collections/%d/export/obsidian/' % collection.pk)
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        self.assertEqual(sorted(archive.namelist()), ['Hot-planets/Hot-planets.md', 'Hot-planets/lee2021clouds.md'])
        note = archive.read('Hot-planets/lee2021clouds.md').decode()
        for part in ('tags:\n  - "clouds"\n  - "hot-jupiters"', 'status: "Key paper"', '## My note\n\nRead twice',
                     'Collection: [[Hot-planets]]'):
            self.assertIn(part, note)
        self.assertIn('- [[lee2021clouds]] Clouds on a hot planet (2021)', archive.read('Hot-planets/Hot-planets.md').decode())
        collection.name = 'Gaz dans les disques de débris'
        collection.save()
        response = self.client_for(user).get('/fr/collections/%d/export/obsidian/' % collection.pk)
        self.assertIn('Gaz-dans-les-disques-de-debris/Gaz-dans-les-disques-de-debris.md',
                      zipfile.ZipFile(io.BytesIO(response.content)).namelist())

    def test_names_keep_their_particles(self):
        self.assertEqual(exports.split_name('Jean-Pierre van der Waals'), ('Jean-Pierre', 'van der Waals'))
        self.assertEqual(exports.split_name('Lee, Ann'), ('Ann', 'Lee'))
        self.assertEqual(exports.split_name('Plato'), ('', 'Plato'))
        meta = {'authors': ['Daniel Beránek'], 'year': 2026, 'title': 'A DIO/ODI-Informed Guardrail'}
        self.assertEqual(exports.cite_key(meta), 'beranek2026informed')
        self.assertEqual(exports.bib_text(meta['title']), 'A {DIO}/{ODI}-Informed Guardrail')
        self.assertEqual(zotero.creator('Ludwig van Beethoven')['lastName'], 'van Beethoven')


class ZoteroTests(ToolsTestCase):
    def account(self, user, **kw):
        return ZoteroAccount.objects.create(user=user, library_id='123', api_key='k', username='ann_z', **kw)

    def test_a_paper_whose_title_changed_is_found_by_its_first_author(self):
        meta = {'arxiv_id': '2101.00001v1', 'title': 'Clouds on a hot planet', 'doi': '', 'authors': ['Ann Lee']}
        with FakeZotero() as fake:
            key = fake.add({'itemType': 'journalArticle', 'title': 'Thick hazes of a warm world', 'extra': 'arXiv: 2101.00001',
                            'creators': [{'creatorType': 'author', 'firstName': 'Ann', 'lastName': 'Lee'}]})
            account = ZoteroAccount(library_id='123', api_key='k')
            self.assertIsNone(zotero.find(account, zotero.meta_identifiers(meta), meta['title']))
            self.assertEqual(zotero.find(account, zotero.meta_identifiers(meta), meta['title'], author='Ann Lee')['key'], key)

    def test_a_paper_already_in_the_library_is_not_duplicated(self):
        self.paper()
        user = self.user(paid=True)
        account = self.account(user)
        client = self.client_for(user)
        with FakeZotero() as fake:
            fake.add({'itemType': 'preprint', 'title': 'Clouds on a hot planet', 'archiveID': 'arXiv:2101.00001v2'})
            inbox = fake.collection('To read')
            account.inbox = inbox
            account.save()
            for _ in range(2):
                data = Generated.objects.get(pk=client.post('/en/zotero/send/', {'arxiv_id': '2101.00001v1'}).json()['id']).data()
                self.assertEqual((data['saved'], data['existing']), (0, 1))
        item = fake.only_item()
        self.assertEqual(item['collections'], [inbox])
        self.assertEqual(len(fake.notes_of(item['key'])), 1)  # the summary, once

    def test_a_collection_stays_in_sync_both_ways(self):
        self.paper('2101.00001v1')
        self.paper('2101.00002v1', title='Hazes on a warm planet')
        user = self.user(paid=True)
        self.account(user)
        client = self.client_for(user)
        collection = Collection.objects.create(user=user, name='Thesis')
        CollectionPaper.objects.create(collection=collection, arxiv_id='2101.00001v1', title='Clouds on a hot planet',
                                       authors='Ann Lee', note='Method to reuse', tags='clouds', reading='to_read')
        with FakeZotero() as fake, mock.patch('summarizer.library.lookup_many',
                                              return_value={'2101.00002': ('Hazes on a warm planet', 'Ann Lee')}):
            key = fake.collection('Thesis')
            fake.add({'itemType': 'preprint', 'title': 'Hazes on a warm planet', 'archiveID': 'arXiv:2101.00002',
                      'collections': [key]})
            first = Generated.objects.get(pk=client.post('/en/zotero/send/', {'collection': collection.pk}).json()['id']).data()
            self.assertEqual((first['added'], first['saved'], first['existing']), (1, 1, 1))
            collection.refresh_from_db()
            self.assertEqual(collection.zotero_key, key)  # the Zotero collection of the same name
            sent = next(i['data'] for i in fake.top(key) if i['data']['title'] == 'Clouds on a hot planet')
            self.assertEqual({t['tag'] for t in sent['tags']}, {'SummarizePaper', 'clouds', 'To read'})
            notes = fake.notes_of(sent['key'])
            self.assertTrue(any('Method to reuse' in n for n in notes))
            self.assertTrue(any('Long summary of the clouds.' in n for n in notes))
            self.assertEqual(sorted(collection.papers.values_list('arxiv_id', flat=True)), ['2101.00001v1', '2101.00002'])
            # removed here: out of the Zotero collection (still in the library); removed in Zotero: out of here
            client.post('/en/collections/%d/' % collection.pk, {'action': 'remove', 'arxiv_id': '2101.00001v1'})
            hazes = next(i for i in fake.top(key) if i['data']['title'] == 'Hazes on a warm planet')
            hazes['data']['collections'] = []
            second = Generated.objects.get(pk=client.post('/en/zotero/send/', {'collection': collection.pk}).json()['id']).data()
        self.assertEqual(second['removed'], 1)
        self.assertEqual(fake.top(key), [])
        self.assertIn(sent['key'], fake.items)
        self.assertFalse(collection.papers.exists())
        # the new item, then its notes: nothing written twice
        self.assertEqual(len([c for c in fake.calls if c[0] == 'POST' and c[1] == 'items']), 2)

    def test_a_bibliography_check_is_written_back(self):
        user = self.user(paid=True)
        self.account(user)
        result = {'zotero_collection': '', 'problems': [
            {'key': 'old', 'title': 'Clouds on a hot planet', 'arxiv': '2101.00001', 'doi': '', 'issues': [
                {'kind': 'published', 'level': 'warning', 'doi': '10.3847/abc', 'journal': 'ApJ',
                 'bibtex': '@article{old,\n  title={Clouds on a hot planet},\n  author={Lee, Ann and Ray, Bob},\n'
                           '  journal={ApJ},\n  volume={900},\n  pages={12},\n  year={2022},\n  doi={10.3847/abc}\n}'}]},
            {'key': 'bad', 'title': 'A retracted study of galaxies', 'arxiv': '', 'doi': '10.1/xyz', 'issues': [
                {'kind': 'retracted', 'level': 'error', 'doi': '10.1/xyz', 'notice': '10.1/notice'}]},
            {'key': 'meant', 'title': 'A corrected study of planets', 'arxiv': '', 'doi': '10.1/c', 'issues': [
                {'kind': 'corrected', 'level': 'warning', 'notice': '10.1/n2', 'mark': 'intended'}]}]}
        Generated.objects.create(kind='refcheck', key='c1', user=user, status='done', lang='en', content=json.dumps(result))
        with FakeZotero() as fake:
            preprint = fake.add({'itemType': 'preprint', 'title': 'Clouds on a hot planet', 'archiveID': 'arXiv:2101.00001',
                                 'collections': ['COLLAAAA']})
            retracted = fake.add({'itemType': 'journalArticle', 'title': 'A retracted study of galaxies', 'DOI': '10.1/xyz'})
            corrected = fake.add({'itemType': 'journalArticle', 'title': 'A corrected study of planets', 'DOI': '10.1/c'})
            data = Generated.objects.get(pk=self.client_for(user).post('/en/refcheck/c1/zotero/').json()['id']).data()
        self.assertEqual((data['tagged'], data['added'], data['missing']), (2, 1, []))
        self.assertEqual([t['tag'] for t in fake.items[retracted]['data']['tags']], ['Retracted'])
        self.assertIn('/en/refcheck/c1/', fake.notes_of(retracted)[0])
        self.assertEqual(fake.items[corrected]['data']['tags'], [])  # marked intended
        published = next(i['data'] for i in fake.items.values() if i['data'].get('DOI') == '10.3847/abc')
        self.assertEqual((published['itemType'], published['publicationTitle'], published['volume'], published['collections']),
                         ('journalArticle', 'ApJ', '900', ['COLLAAAA']))
        self.assertEqual(published['relations']['dc:relation'], ['http://zotero.org/users/123/items/%s' % preprint])
        self.assertEqual(fake.items[preprint]['data']['relations']['dc:relation'],
                         ['http://zotero.org/users/123/items/%s' % published['key']])

    def test_a_literature_review_becomes_collections_by_approach(self):
        user = self.user(paid=True)
        self.account(user)
        result = {'title': 'Microplastics', 'searched': '2026-11-01', 'papers': [
            {'n': 1, 'key': 'doi_10.1000--w1', 'title': 'Sinks of plastic', 'authors': 'Ann Lee', 'year': 2012, 'doi': '10.1000/w1',
             'venue': 'Marine Journal'}],
            'themes': [{'name': 'Field surveys', 'papers': [1], 'findings': [
                {'text': 'Plastic stays down.', 'papers': [1], 'support': [{'n': 1, 'quote': 'Plastic sinks & stays.'}]}]}],
            'consensus': [], 'disagreements': [], 'gaps': []}
        Generated.objects.create(kind='literature', key='r1', user=user, status='done', lang='en', content=json.dumps(result))
        with FakeZotero() as fake:
            Generated.objects.get(pk=self.client_for(user).post('/en/zotero/send/', {'literature': 'r1'}).json()['id'])
        names = {c['name']: c['parent'] for c in fake.collections.values()}
        parent = next(k for k, c in fake.collections.items() if c['name'] == 'Microplastics')
        self.assertEqual(names, {'Microplastics': '', 'Field surveys': parent})
        item = fake.only_item()
        self.assertEqual(len(item['collections']), 2)
        note = fake.notes_of(item['key'])[0]
        self.assertIn('Plastic stays down.', note)
        self.assertIn('<blockquote>Plastic sinks &amp; stays.</blockquote>', note)

    def test_formulas_are_marked_for_zotero_notes(self):
        self.assertEqual(zotero.with_math(r'up to $10^{-1}\,M_\oplus$ & $$E=mc^2$$ for \$5'),
                         r'up to <span class="math">$10^{-1}\,M_\oplus$</span> &amp; '
                         r'<pre class="math">$$E=mc^2$$</pre> for \$5')

    def test_reference_papers_come_from_a_zotero_collection(self):
        user = self.user(paid=True)
        self.account(user)
        with FakeZotero() as fake:
            key = fake.collection('Mine')
            fake.add({'itemType': 'preprint', 'title': 'A', 'archiveID': 'arXiv:2101.00001', 'collections': [key]})
            fake.add({'itemType': 'journalArticle', 'title': 'B', 'DOI': '10.1093/mnras/stad1234', 'collections': [key]})
            fake.add({'itemType': 'book', 'title': 'C', 'collections': [key]})
            data = self.client_for(user).get('/en/zotero/references/', {'key': key, 'n': 5}).json()
        self.assertEqual((data['refs'], data['without']), (['2101.00001', '10.1093/mnras/stad1234'], 1))
        self.assertContains(self.client_for(user).get('/en/digest/'), 'Take them from a Zotero collection')

    def test_group_library_and_page(self):
        user = self.user(paid=True)
        account = self.account(user)
        client = self.client_for(user)
        with FakeZotero(groups=[(77, 'Lab')]) as fake:
            fake.collection('Thesis')
            page = client.get('/en/integrations/')
            self.assertContains(page, 'Group: Lab')
            self.assertContains(page, 'id="zotero-hub"')
            client.post('/en/integrations/', {'action': 'library', 'library': 'groups/77'})
            client.post('/en/integrations/', {'action': 'library', 'library': 'groups/99'})  # not one of theirs
        account.refresh_from_db()
        self.assertEqual((account.prefix(), account.library_name), ('groups/77', 'Lab'))
        self.assertEqual(zotero.open_link(account, 'ABCD1234'), 'zotero://select/groups/77/items/ABCD1234')

    @override_settings(ZOTERO_CLIENT_KEY='ck', ZOTERO_CLIENT_SECRET='cs')
    def test_oauth_connects_without_a_key_to_copy(self):
        user = self.user(paid=True)
        client = self.client_for(user)
        self.assertContains(client.get('/en/integrations/'), '/en/integrations/zotero/connect/')
        with mock.patch('summarizer.zotero.oauth_post', return_value={'oauth_token': 'rt', 'oauth_token_secret': 'rs'}):
            r = client.get('/en/integrations/zotero/connect/')
        self.assertEqual(parse_qs(urlparse(r['Location']).query)['oauth_token'], ['rt'])
        with mock.patch('summarizer.zotero.oauth_post', return_value={'oauth_token': 'at', 'oauth_token_secret': 'thekey',
                                                                     'userID': '123', 'username': 'ann_z'}) as post, FakeZotero():
            r = client.get('/en/integrations/zotero/callback/', {'oauth_token': 'rt', 'oauth_verifier': 'v'})
        self.assertEqual(post.call_args[0][1], 'rs')
        self.assertIn('saved=1', r['Location'])
        self.assertEqual(ZoteroAccount.objects.get(user=user).api_key, 'thekey')
        self.assertIn('error=oauth', client.get('/en/integrations/zotero/callback/', {'oauth_token': 'rt'})['Location'])

    def test_oauth_signature(self):
        # the example of RFC 5849 section 1.2, with HMAC-SHA1
        with override_settings(ZOTERO_CLIENT_KEY='dpf43f3p2l4k3l03', ZOTERO_CLIENT_SECRET='kd94hf93k423kf44'), \
             mock.patch('summarizer.zotero.secrets.token_hex', return_value='wIjqoS'), \
             mock.patch('summarizer.zotero.time.time', return_value=137131200), \
             mock.patch('summarizer.zotero.requests.post', return_value=mock.Mock(text='a=1', raise_for_status=lambda: None)) as post:
            zotero.oauth_post('https://photos.example.net/initiate', oauth_callback='http://printer.example.com/ready')
        self.assertIn('oauth_signature="74KNZJeDHnMBp0EMJ9ZHt%2FXKycU%3D"', post.call_args[1]['headers']['Authorization'])

    def test_retractions_found_in_a_synced_collection_become_tags(self):
        paper = self.paper()
        user = self.user(paid=True)
        self.account(user)
        publication_status.Generated.objects.create(kind=publication_status.KIND, key=paper.arxiv_id, lang='en', status='done',
                                                    content=json.dumps({'status': 'published', 'doi': '10.1/a', 'journal': 'J',
                                                                        'date': '2022-01-01', 'updates': [{'type': 'retraction', 'doi': '10.1/r'}]}))
        collection = Collection.objects.create(user=user, name='T', zotero_key='COLLAAAA', zotero_library='users/123')
        with FakeZotero() as fake:
            key = fake.add({'itemType': 'journalArticle', 'title': 'Clouds on a hot planet', 'collections': ['COLLAAAA']})
            CollectionPaper.objects.create(collection=collection, arxiv_id=paper.arxiv_id, title='t', zotero_key=key)
            zotero.write_notices(collection)
        self.assertEqual([t['tag'] for t in fake.items[key]['data']['tags']], ['Retracted', 'Published version available'])

    def test_keys_are_encrypted_at_rest(self):
        account = self.account(self.user(paid=True))
        self.assertNotEqual(account.secret, 'k')
        self.assertEqual(ZoteroAccount.objects.get(pk=account.pk).api_key, 'k')
        with override_settings(SECRET_KEY='another'):
            self.assertEqual(ZoteroAccount.objects.get(pk=account.pk).api_key, '')
            self.assertEqual(self.client_for(account.user).post('/en/zotero/send/', {'arxiv_id': '2101.00001v1'}).json()['status'],
                             'setup')


class ZoteroCheckAndDailySyncTests(ToolsTestCase):
    def account(self, user):
        return ZoteroAccount.objects.create(user=user, library_id='123', api_key='k', username='ann_z')

    def test_the_keys_a_manuscript_may_cite_an_item_with(self):
        data = {'title': 'Clouds on a hot planet', 'date': '2021-01-04', 'extra': '',
                'creators': [{'creatorType': 'author', 'firstName': 'Ann', 'lastName': 'Lee'}]}
        self.assertEqual(zotero.citation_keys(data), ['leeCloudsHotPlanet2021', 'lee2021clouds'])
        self.assertEqual(zotero.citation_keys(dict(data, extra='tex.x: 1\nCitation Key: lee_clouds'))[0], 'lee_clouds')
        accented = dict(data, creators=[{'creatorType': 'author', 'firstName': 'Daniel', 'lastName': 'Beránek'}])
        self.assertEqual(zotero.citation_keys(accented)[0], 'beranekCloudsHotPlanet2021')

    def test_a_zotero_collection_is_checked_with_the_manuscript_keys_and_the_researchers_pdfs(self):
        from . import refcheck
        account = self.account(self.user(paid=True))
        text = 'We measured the cloud top pressure of the planet at 3.2 mbar with transmission spectra. ' * 60
        with FakeZotero() as fake:
            key = fake.collection('Draft')
            lee = fake.add({'itemType': 'journalArticle', 'title': 'Clouds on a hot planet', 'date': '2021', 'collections': [key],
                            'creators': [{'creatorType': 'author', 'firstName': 'Ann', 'lastName': 'Lee'}]})
            pdf = fake.add({'itemType': 'attachment', 'parentItem': lee, 'contentType': 'application/pdf', 'linkMode': 'imported_file'})
            fake.fulltext[pdf] = text
            fake.add({'itemType': 'journalArticle', 'title': 'Hazes', 'date': '2020', 'collections': [key],
                      'extra': 'Citation Key: smith_hazes', 'creators': [{'creatorType': 'author', 'lastName': 'Smith'}]})
            fake.add({'itemType': 'journalArticle', 'title': 'Never cited', 'date': '2019', 'collections': [key],
                      'creators': [{'creatorType': 'author', 'lastName': 'Ray'}]})
            tex = 'Clouds sit at 3.2 mbar \\citep{leeCloudsHotPlanet2021}. Hazes too \\citep{smith_hazes}.'
            bib, texts, origin, _pdf, own = refcheck.loader('', tex, 'main.tex', None, None, '', (account, key))()
            self.assertIn('@article{leeCloudsHotPlanet2021,', bib)  # the key the manuscript cites
            self.assertIn('@article{smith_hazes,', bib)  # pinned in Zotero
            self.assertIn('@article{ray_', bib)  # neither cited nor pinned: Zotero's own key
            read = own('leeCloudsHotPlanet2021')
        self.assertEqual((read['source'], read['text']), ('zotero', text))
        self.assertIsNone(own('ray_x'))

    def test_the_daily_sync_reads_only_what_changed(self):
        self.paper('2101.00001v1')
        self.paper('2101.00002v1', title='Hazes on a warm planet')
        user = self.user(paid=True)
        self.account(user)
        collection = Collection.objects.create(user=user, name='Thesis')
        clouds = CollectionPaper.objects.create(collection=collection, arxiv_id='2101.00001v1', title='Clouds on a hot planet',
                                                authors='Ann Lee')
        free = self.user('free')
        ZoteroAccount.objects.create(user=free, library_id='9', api_key='k')
        Collection.objects.create(user=free, name='Lapsed', zotero_key='COLLBBBB', zotero_library='users/9')
        with FakeZotero() as fake, mock.patch('summarizer.library.lookup_many',
                                              return_value={'2101.00002': ('Hazes on a warm planet', 'Ann Lee')}):
            key = fake.collection('Thesis')
            before = fake.version
            self.client_for(user).post('/en/zotero/send/', {'collection': collection.pk})
            collection.refresh_from_db()
            # the version read before writing: the next sync sees what changed since, its own writes included
            self.assertEqual((collection.zotero_version, collection.zotero_lang), (before, 'en'))
            item = CollectionPaper.objects.get(pk=clouds.pk).zotero_key
            # in Zotero: a paper added to the collection, the other put in the trash; here: nothing changed
            fake.add({'itemType': 'preprint', 'title': 'Hazes on a warm planet', 'archiveID': 'arXiv:2101.00002',
                      'collections': [key]})
            fake.change(item, deleted=1)
            fake.calls.clear()
            self.assertEqual(zotero.run(timezone.now() + datetime.timedelta(days=1)), 1)  # not the lapsed Premium
        self.assertEqual(list(collection.papers.values_list('arxiv_id', flat=True)), ['2101.00002'])
        reads = [path for method, path in fake.calls if method == 'GET']
        self.assertNotIn('collections/%s/items/top' % key, reads)  # what changed only
        self.assertIn('items/top', reads)
        self.assertFalse([c for c in fake.calls if c[0] == 'PATCH' and c[1] == 'items/%s' % item])  # nothing written back to it
