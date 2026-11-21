import datetime
import json
import os
import shutil
import tempfile
from unittest import mock

from bs4 import BeautifulSoup
from django.core.cache import cache
from django.test import TestCase

from . import arxiv_metadata, data_guides, guides, trends
from .models import Guide, TermWeek

TODAY = datetime.date(2026, 11, 20)


def keep_topics(scope, items):
    """The model's check without the model: drops the generic phrase, labels the rest."""
    return {i['term']: {'label': i['term'].title(), 'gloss': {'en': 'about ' + i['term'], 'fr': 'sur ' + i['term']}}
            for i in items if i['term'] != 'higher mean'}


class TopicsGuideTests(TestCase):
    def setUp(self):
        cache.clear()

    def fill(self, weeks=30):
        last = trends.closed_week(TODAY)
        for i in range(weeks):
            week = last - datetime.timedelta(weeks=i)
            recent = i < data_guides.WINDOW
            counts = {'stable topic': 20, 'graph neural operator': 12 if recent else 0, 'higher mean': 15 if recent else 0,
                      'old topic': 2 if recent else 30, 'neutron-star': 12 if recent else 0, 'neutron star': 40}
            counts = {k: v for k, v in counts.items() if v >= 2}
            for scope in ('*', 'cs'):
                TermWeek.objects.create(scope=scope, week=week, papers=300, counts=json.dumps(counts))

    def test_writes_both_languages_from_the_counts(self):
        self.fill()
        self.assertTrue(data_guides.write_topics(TODAY, check=keep_topics))
        en = Guide.objects.get(lang='en', slug=data_guides.TOPICS_SLUG)
        fr = Guide.objects.get(lang='fr', slug=data_guides.TOPICS_SLUG)
        self.assertFalse(en.draft)
        self.assertIn('Graph Neural Operator', en.body)
        self.assertNotIn('Higher Mean', en.body)  # dropped by the check
        self.assertNotIn('Neutron-Star', en.body)  # a new spelling of a common term, not a new topic
        self.assertNotIn('Stable Topic', en.body.split('losing ground')[0])  # did not grow
        self.assertIn('Old Topic', en.body.split('losing ground')[1])
        self.assertIn('computer science and AI', en.body)
        self.assertTrue(fr.title.startswith('Les sujets de recherche'))
        self.assertIn('novembre 2026', fr.title)
        meta = en.extra_meta()
        self.assertIn('faq', meta)
        self.assertNotIn('&id', en.extra)
        response = self.client.get('/en/guides/%s/' % data_guides.TOPICS_SLUG)
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'FAQPage')
        self.assertContains(response, '/en/trends/explore/?t=graph+neural+operator')
        self.assertEqual(self.client.get('/fr/guides/%s/' % data_guides.TOPICS_SLUG).status_code, 200)

    def test_keeps_the_first_date_and_refreshes_monthly(self):
        self.fill()
        with mock.patch.object(data_guides, 'check_terms', keep_topics):
            self.assertTrue(data_guides.refresh_due(datetime.datetime(2026, 11, 20, 12, tzinfo=datetime.timezone.utc)))
            self.assertFalse(data_guides.refresh_due(datetime.datetime(2026, 12, 1, 12, tzinfo=datetime.timezone.utc)))
            cache.clear()
            self.assertTrue(data_guides.refresh_due(datetime.datetime(2026, 12, 19, 12, tzinfo=datetime.timezone.utc)))
        row = Guide.objects.get(lang='en', slug=data_guides.TOPICS_SLUG)
        self.assertEqual(row.date, datetime.date(2026, 11, 20))
        self.assertEqual(row.updated, datetime.date(2026, 12, 19))

    def test_needs_enough_weeks_and_tries_once_a_day(self):
        self.fill(weeks=10)
        with mock.patch.object(data_guides, 'check_terms', keep_topics) as check:
            self.assertFalse(data_guides.refresh_due(datetime.datetime(2026, 11, 20, 12, tzinfo=datetime.timezone.utc)))
            self.assertFalse(data_guides.refresh_due(datetime.datetime(2026, 11, 20, 13, tzinfo=datetime.timezone.utc)))
        self.assertFalse(Guide.objects.filter(slug=data_guides.TOPICS_SLUG).exists())


def record(arxiv_id, cats, authors, jref='', doi=''):
    return {'id': arxiv_id, 'cats': cats.split(), 'authors': authors, 'jref': jref, 'doi': doi}


class MetadataGuidesTests(TestCase):
    def records(self):
        found = {}
        n = 0
        for year in (22, 23, 24, 25):
            for field, cats, authors, jref in (('astro', 'astro-ph.EP', 6, 'Astrophys.J. 930 (2022) 1'),
                                               ('math', 'math.AP', 2, ''), ('hep', 'hep-ex', 1, 'Phys. Rev. D 1, 2')):
                for i in range(400):
                    n += 1
                    arxiv_id = '%d%02d.%05d' % (year, 1 + i % 12, n)
                    published = jref and (i % 4 != 0 or year < 25)
                    found[arxiv_id] = record(arxiv_id, cats + ' cs.LG', authors + i % 3, jref if published else '')
        return found

    def test_pages_render_in_both_languages(self):
        records = self.records()
        authors = data_guides.authors_data(records, 2025, 2022)
        journals = data_guides.journal_data(records, [2022, 2023, 2024, 2025])
        self.assertEqual(authors['all']['papers'], 1200)
        self.assertEqual([f['code'] for f in authors['fields']], ['math', 'astro-ph', 'hep-ex'])
        for lang in data_guides.LANGS:
            meta, body = data_guides.authors_page(authors, lang, TODAY)
            self.assertIn(t := ('Astrophysics (astro-ph)' if lang == 'en' else 'Astrophysique (astro-ph)'), body)
            # high-energy experiment signs as one collaboration: never the "most single-author" field
            solo = body.split('**')[9]
            self.assertNotIn('hep-ex', solo)
            meta, body = data_guides.journals_page(journals, lang, TODAY)
            self.assertIn('Astrophys. J. (400)', body.replace(' ', ' '))
            self.assertIn('Phys. Rev. D', body)
            self.assertTrue(meta['faq'])

    def test_charts_links_and_masked_collaborations(self):
        records = self.records()
        authors = data_guides.authors_data(records, 2025, 2022)
        self.assertFalse(authors['groups_known'])
        meta, body = data_guides.authors_page(authors, 'en', TODAY)
        self.assertEqual(body.count('<figure class="viz">'), 1)
        self.assertIn('/en/guides/arxiv-papers-published-in-journals/', body)
        self.assertNotIn('/en/guides/authors-per-paper/', body)  # no link to itself
        hep = [line for line in body.splitlines() if line.startswith('| High-energy physics, experiment')]
        self.assertTrue(hep and all('—' in line for line in hep))
        meta, body = data_guides.journals_page(data_guides.journal_data(records, [2022, 2023, 2024, 2025]), 'fr', TODAY)
        self.assertEqual(body.count('<figure class="viz">'), 1)
        self.assertNotIn('Et les articles de conférence', body)  # no comments harvested

    def test_groups_and_accepted_comments(self):
        records = self.records()
        for r in records.values():
            r['groups'] = 1 if r['cats'][0] == 'hep-ex' else 0
            r['comments'] = 'Accepted at NeurIPS 2023' if r['cats'][0] == 'math.AP' else 'Submitted to ICML'
        authors = data_guides.authors_data(records, 2025, 2022)
        self.assertTrue(authors['groups_known'])
        hep = next(f for f in authors['fields'] if f['code'] == 'hep-ex')
        self.assertEqual(hep['now']['solo'], 0)  # signed by a collaboration
        journals = data_guides.journal_data(records, [2022, 2023, 2024, 2025])
        math = next(f for f in journals['fields'] if f['code'] == 'math')
        self.assertEqual(math['shares'][2022][1:], (0, 400))
        self.assertEqual(math['journals'], [('NeurIPS', 400)])
        meta, body = data_guides.journals_page(journals, 'en', TODAY)
        self.assertIn('What about conference papers?', body)
        self.assertFalse(data_guides.accepted('Submitted to ICML, under review'))
        self.assertTrue(data_guides.accepted('To appear in JHEP'))
        self.assertTrue(data_guides.accepted('In Proceedings of the 2021 IEEE/RSJ IROS'))
        self.assertFalse(data_guides.accepted('11 pages, extension of our proceedings of last year, submitted to PRD'))
        self.assertEqual(data_guides.venue_of({'jref': '', 'doi': '', 'comments': 'Accepted to NeurIPS2022'}), 'NeurIPS')
        self.assertEqual(data_guides.journal_of('Chin. Phys. C 46 (2022) 1'), 'Chin. Phys. C')

    def test_write_file_keeps_the_publication_date(self):
        folder = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, folder)
        os.makedirs(os.path.join(folder, 'en'))
        with mock.patch.object(guides, 'CONTENT_DIR', folder):
            meta = {'title': 'T', 'description': 'D', 'date': datetime.date(2026, 1, 1), 'updated': datetime.date(2026, 1, 1)}
            data_guides.write_file('en', 'x', meta, 'Body')
            data_guides.write_file('en', 'x', dict(meta, date=TODAY, updated=TODAY), 'Body 2')
            guide = guides.load('en', 'x')
        self.assertEqual(guide['date'], datetime.date(2026, 1, 1))
        self.assertEqual(guide['updated'], TODAY)

    def test_journal_names(self):
        for jref, name in (('Physical Review B 105, 1234 (2022)', 'Phys. Rev. B'), ('Phys.Rev.Lett. 128 (2022) 1', 'Phys. Rev. Lett.'),
                           ('Monthly Notices of the Royal Astronomical Society, 512, 1', 'MNRAS'), ('MNRAS 512, 2', 'MNRAS'),
                           ('The Astrophysical Journal, 930:1', 'Astrophys. J.'), ('A&A 660, A1 (2022)', 'Astron. Astrophys.'),
                           ('Proceedings of the 36th AAAI Conference', 'AAAI'), ('Proceedings of the 2022 Conference', ''),
                           ('Universe 8 (2022) 1', 'Universe'), ('', '')):
            self.assertEqual(data_guides.journal_of(jref), name, jref)
        for doi, name in (('10.1093/mnras/stac123', 'MNRAS'), ('10.1103/PhysRevD.105.1', 'Phys. Rev. D'),
                          ('10.1103/PhysRevLett.128.1', 'Phys. Rev. Lett.'), ('10.3847/1538-4357/ac1', 'Astrophys. J.'),
                          ('10.1038/s41467-022-1', 'Nature Commun.'), ('10.1234/unknown', '')):
            self.assertEqual(data_guides.journal_from_doi(doi), name, doi)
        self.assertEqual(data_guides.journal_of('Insurance: Mathematics and Economics 105 (2022) 1'), 'Insurance: Mathematics and Economics')
        self.assertEqual(data_guides.journal_of('Pacific J. Math. 318 (2022)'), 'Pacific J. Math.')

    def test_field_and_year(self):
        self.assertEqual(data_guides.field_of(['nucl-th', 'hep-ph']), 'nucl')
        self.assertEqual(data_guides.field_of(['physics.optics']), 'physics')
        self.assertIsNone(data_guides.field_of([]))
        self.assertEqual(data_guides.year_of('2501.01234'), 2025)
        self.assertIsNone(data_guides.year_of('hep-th/9901001'))


class HarvestParseTests(TestCase):
    PAGE = b'''<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><ListRecords>
<record><header><identifier>oai:arXiv.org:2501.00001</identifier><datestamp>2025-01-03</datestamp></header>
<metadata><arXiv xmlns="http://arxiv.org/OAI/arXiv/"><id>2501.00001</id><authors><author><keyname>A</keyname></author>
<author><keyname>ATLAS Collaboration</keyname></author></authors><categories>astro-ph.EP cs.LG</categories><comments>Accepted at ICML 2025</comments>
<journal-ref>MNRAS 1, 2</journal-ref><doi>10.1093/x 10.1093/y</doi></arXiv></metadata></record>
<record><header status="deleted"><identifier>oai:arXiv.org:2501.00002</identifier><datestamp>2025-01-04</datestamp></header></record>
<resumptionToken cursor="0">abc</resumptionToken></ListRecords></OAI-PMH>'''

    def test_parse(self):
        records, token, latest = arxiv_metadata.parse(self.PAGE)
        self.assertEqual(records, [{'id': '2501.00001', 'cats': ['astro-ph.EP', 'cs.LG'], 'authors': 2, 'groups': 1,
                                    'jref': 'MNRAS 1, 2', 'doi': '10.1093/x', 'comments': 'Accepted at ICML 2025'}])
        self.assertEqual((token, latest), ('abc', '2025-01-04'))

    def test_expired_token_is_an_error(self):
        with self.assertRaises(arxiv_metadata.OAIError) as error:
            arxiv_metadata.parse(b'<OAI-PMH xmlns="http://www.openarchives.org/OAI/2.0/"><error code="badResumptionToken">x</error></OAI-PMH>')
        self.assertEqual(error.exception.code, 'badResumptionToken')


class HomeGuidesTests(TestCase):
    def setUp(self):
        cache.clear()

    def test_home_links_the_published_guides(self):
        for lang in ('en', 'fr'):
            soup = BeautifulSoup(self.client.get('/%s/' % lang).content, 'html.parser')
            links = [a['href'] for a in soup.select('.home-guide-list a')]
            self.assertIn('/%s/guides/arxiv-statistics/' % lang, links)
            self.assertNotIn('/%s/guides/authors-per-paper/' % lang, links) if not guides.load(lang, 'authors-per-paper') else None
