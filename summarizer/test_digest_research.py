"""The digest for researchers: keywords, multi-subject profiles, reasons, structured summaries, the web page
of each email, counted links, saving to a collection, the live preview, imports and bundled alerts."""
import datetime
import json
import re
from io import StringIO
from unittest import mock

import numpy as np
from django.contrib.auth.models import User
from django.core import mail
from django.core.cache import cache
from django.core.management import call_command
from django.test import Client, TestCase

from summarizer import digest, digest_bundle, digest_web
from summarizer.models import (AnnouncedPaper, ArxivBatch, CategorySynthesis, Collection, CollectionPaper, DigestClick,
                               DigestEdition, DigestFeedback, DigestItem, DigestSubscription, DigestVector, ScoopMatch,
                               UserPlan, WatchedProject)
from summarizer.tests import MONDAY, announced, blurbs, fake_embed, fake_paper


def paper(aid, title, abstract='', category='astro-ph.EP'):
    return dict(fake_paper(aid, category, title), abstract=abstract or title + ' abstract.')


def entry_ids(entries):
    return [e['paper']['arxiv_id'] for e in entries]


class ResearchDigestCase(TestCase):
    def setUp(self):
        cache.clear()

    def user(self, name, paid=False):
        u = User.objects.create_user(name, email='%s@example.com' % name, password='x')
        if paid:
            UserPlan.objects.create(user=u, paid=True)
        return u

    def run_digest(self, papers, now=MONDAY, chat=None):
        with announced(papers), \
             mock.patch('summarizer.digest.write_blurbs', side_effect=blurbs), \
             mock.patch('summarizer.similar.embed', side_effect=fake_embed), \
             mock.patch('summarizer.digest.fetch_ids', return_value=[]), \
             mock.patch('summarizer.consumers.summarize_now'), \
             mock.patch('summarizer.audio.ensure', return_value=[]), \
             mock.patch('summarizer.digest._chat', return_value=chat or json.dumps({'trends': []})):
            return digest.run(now=now)


class KeywordTests(ResearchDigestCase):
    def test_whole_words_plurals_phrases_and_exclusions(self):
        keywords = digest.Keywords('comets, "debris disk", ai, -review')
        self.assertEqual([t for t, _ in keywords.include], ['comets', 'debris disk', 'ai'])
        self.assertEqual([t for t, _ in keywords.exclude], ['review'])
        self.assertEqual(keywords.matches(paper('1', 'An active comet'))[1], ['comets'])  # plural typed, singular found
        self.assertEqual(keywords.matches(paper('2', 'Cometary dust'))[1], [])  # not a part of a word
        self.assertEqual(keywords.matches(paper('3', 'Debris-disk structure'))[1], ['debris disk'])
        self.assertEqual(keywords.matches(paper('4', 'Gain and chain'))[1], [])  # 'ai' is not in 'gain'
        self.assertEqual(keywords.matches(paper('5', 'AI for telescopes'))[1], ['ai'])
        self.assertTrue(keywords.excluded(paper('6', 'A review of comets')))
        self.assertFalse(keywords.excluded(paper('7', 'Comets reviewed')))

    def test_keywords_filter_without_profile_and_raise_with_one(self):
        papers = {p['arxiv_id']: p for p in [paper('2610.00001v1', 'Galaxy mergers'),
                                             paper('2610.00002v1', 'Planet clouds with JWST')]}
        sub = DigestSubscription(categories='astro-ph.EP', keywords='JWST')
        self.assertEqual(entry_ids(digest.rank_papers(sub, papers, set())), ['2610.00002v1'])
        vectors = {aid: fake_embed([digest.paper_text(p)])[0] for aid, p in papers.items()}
        ranking = digest.Ranking(rows=fake_embed(['galaxy']), labels=[{'kind': 'profile', 'text': 'galaxies'}])
        entries = digest.rank_papers(sub, papers, set(), ranking=ranking, vectors=vectors)
        self.assertEqual(entry_ids(entries), ['2610.00001v1', '2610.00002v1'])  # the profile wins, nothing left out
        self.assertEqual(entries[1]['keywords'], ['JWST'])


class RankingTests(ResearchDigestCase):
    def test_a_researcher_with_two_subjects_keeps_both(self):
        u = self.user('two', paid=True)
        sub = DigestSubscription.objects.create(user=u, categories='astro-ph.EP',
                                                profile='I study the clouds of hot planets.\nI also study galaxy mergers in clusters.')
        with mock.patch('summarizer.similar.embed', side_effect=fake_embed) as embed:
            rows, labels = digest.profile_matrix(sub)
            self.assertEqual(rows.shape[0], 2)
            digest.profile_matrix(sub)
        self.assertEqual(embed.call_count, 1)  # stored with the subscription
        ranking = digest.ranking_for(sub, paid=True)
        planet, galaxy = fake_embed(['planet', 'galaxy'])
        self.assertGreater(ranking.score(planet)[0], 0.9)
        self.assertGreater(ranking.score(galaxy)[0], 0.9)  # an average of both would give ~0.7 to each
        self.assertIn('galaxy', ranking.score(galaxy)[1]['text'])

    def test_disliked_papers_go_down_and_liked_ones_say_so(self):
        sub = DigestSubscription.objects.create(user=self.user('fb'), categories='astro-ph.EP')
        planet, galaxy = fake_embed(['planet', 'galaxy'])
        DigestItem.objects.create(arxiv_id='l1', lang='en', title='Liked galaxy paper', blurb='b')
        DigestFeedback.objects.create(subscription=sub, arxiv_id='l1', value=1, vector=galaxy.tobytes())
        DigestFeedback.objects.create(subscription=sub, arxiv_id='d1', value=-1, vector=planet.tobytes())
        ranking = digest.ranking_for(sub, paid=False)  # votes rank free digests too
        self.assertGreater(ranking.score(galaxy)[0], ranking.score(planet)[0])
        self.assertEqual(ranking.score(galaxy)[1], {'kind': 'liked', 'id': 'l1', 'title': 'Liked galaxy paper'})

    def test_ten_papers_do_not_tell_the_same_story(self):
        papers = {}
        for i in range(4):
            papers['g%d' % i] = paper('g%d' % i, 'Galaxy %d' % i)
        papers['p0'] = paper('p0', 'Planet rings')
        vectors = {aid: fake_embed([p['title']])[0] for aid, p in papers.items()}
        sub = DigestSubscription(categories='astro-ph.EP')
        top = entry_ids(digest.rank_papers(sub, papers, set(), vectors=vectors))[:2]
        self.assertIn('p0', top)  # the second pick differs from the first

    def test_threshold_digests_send_every_close_paper(self):
        u = self.user('t', paid=True)
        DigestSubscription.objects.create(user=u, categories='astro-ph.EP', timezone='UTC', send_hour=7,
                                          profile='Hot planet atmospheres and their clouds.', volume='high')
        papers = {p['arxiv_id']: p for p in [fake_paper('2610.0000%dv1' % i, 'astro-ph.EP', 'Planet %d' % i) for i in range(4)] +
                  [fake_paper('2610.00009v1', 'astro-ph.EP', 'Galaxy far away')]}
        self.assertEqual(self.run_digest(papers), 1)
        body = mail.outbox[0].body
        self.assertEqual(body.count('Planet '), 4)
        self.assertNotIn('Galaxy far away', body)


class EmailTests(ResearchDigestCase):
    def test_researcher_summary_reasons_identifiers_and_subject(self):
        u = self.user('r', paid=True)
        DigestSubscription.objects.create(user=u, categories='astro-ph.EP', timezone='UTC', send_hour=7,
                                          reference_papers='2501.00001')
        papers = {p['arxiv_id']: p for p in [fake_paper('2610.00001v1', 'astro-ph.EP', 'Clouds on a hot planet'),
                                             fake_paper('2610.00002v1', 'astro-ph.GA', 'Galaxy merger')]}
        papers['2610.00002v1']['categories'] = ['astro-ph.GA', 'astro-ph.EP']

        def structured(batch, lang):
            return {p['arxiv_id']: {'summary': 'Plain words.', 'points': {
                'contribution': 'First cloud map', 'method': 'JWST spectra', 'result': 'Clouds at 2 bar'}} for p in batch}
        reference = {'arxiv_id': '2501.00001v1', 'title': 'Hot planet clouds', 'abstract': 'Planet clouds.'}
        with announced(papers), mock.patch('summarizer.digest.write_blurbs', side_effect=structured), \
             mock.patch('summarizer.similar.embed', side_effect=fake_embed), \
             mock.patch('summarizer.digest.fetch_ids', return_value=[reference]), \
             mock.patch('summarizer.consumers.summarize_now'), mock.patch('summarizer.audio.ensure', return_value=[]):
            digest.run(now=MONDAY)
        message = mail.outbox[0]
        self.assertEqual(message.subject, 'Clouds on a hot planet and 1 more new papers')
        self.assertIn('Contribution: First cloud map', message.body)
        self.assertNotIn('Plain words.', message.body)
        self.assertIn('Why: Close to your reference paper “Hot planet clouds”', message.body)
        self.assertIn('Cross-listed in astro-ph.EP from astro-ph.GA', message.body)
        self.assertIn('arXiv:2610.00001v1 - astro-ph.EP', message.body)
        self.assertIn('Earth and Planetary Astrophysics - 2 of 2 papers shown', message.body)
        self.assertIn('https://arxiv.org/pdf/2610.00001v1', Client().get(
            re.search(r'https://www.summarizepaper.com(/en/digest/open/[^/]+/)', message.alternatives[0][0].split('>PDF<')[0]
                      .rsplit('href="', 1)[1]).group(1))['Location'])
        edition = DigestEdition.objects.get()
        self.assertEqual(json.loads(edition.papers), ['2610.00001v1', '2610.00002v1'])

    def test_plain_language_level_and_points_added_once_to_older_items(self):
        DigestItem.objects.create(arxiv_id='2610.00001v1', lang='en', title='Old', blurb='Older words.')
        DigestSubscription.objects.create(user=self.user('g'), categories='astro-ph.EP', timezone='UTC', send_hour=7,
                                          level='general')
        self.run_digest({'2610.00001v1': fake_paper('2610.00001v1', 'astro-ph.EP', 'Old')})
        self.assertIn('Older words.', mail.outbox[0].body)
        self.assertEqual(DigestItem.objects.get().points, '{}')  # asked once; the model gave no points here


class EditionPageTests(ResearchDigestCase):
    def test_every_paper_ranked_with_why_and_which_were_in_the_email(self):
        u = self.user('e')
        DigestSubscription.objects.create(user=u, categories='astro-ph.EP', timezone='UTC', send_hour=7, keywords='planet')
        papers = [fake_paper('2610.000%02dv1' % i, 'astro-ph.EP', 'Planet %d' % i) for i in range(12)]
        papers.append(fake_paper('2610.00099v1', 'astro-ph.EP', 'Galaxy only'))
        for p in papers:
            AnnouncedPaper.objects.create(arxiv_id=p['arxiv_id'], announced=p['announced'], title=p['title'],
                                          abstract=p['abstract'], authors='["Ann Lee"]', categories=','.join(p['categories']))
        self.run_digest({p['arxiv_id']: p for p in papers})
        self.assertIn('See the 2 other papers of your categories', mail.outbox[0].body)
        url = re.search(r'https://www.summarizepaper.com(/en/digest/e/[^/?]+/)', mail.outbox[0].body).group(1)
        page = Client().get(url)
        self.assertContains(page, 'in your email', count=10)
        self.assertContains(page, 'Keywords: planet', count=12)
        self.assertNotContains(page, 'Galaxy only')  # left out by the keywords, as in the email
        self.assertContains(page, 'noindex')
        AnnouncedPaper.objects.all().delete()
        self.assertContains(Client().get(url), 'no longer kept')
        self.assertEqual(Client().get('/en/digest/e/forged/').status_code, 404)
        self.assertContains(self._settings_page(u), '/en/digest/e/')  # the latest emails are listed with the settings

    def _settings_page(self, user):
        c = Client()
        c.force_login(user)
        return c.get('/en/digest/')


class LinkTests(ResearchDigestCase):
    def setUp(self):
        super().setUp()
        self.sub = DigestSubscription.objects.create(user=self.user('l'), categories='astro-ph.EP')
        self.edition = DigestEdition.objects.create(subscription=self.sub, sent=MONDAY, days='2026-10-19', papers='[]')
        DigestVector.objects.create(arxiv_id='2610.00001v1', vector=fake_embed(['planet'])[0].tobytes())

    def url(self, aid='2610.00001v1', rank=1, target='summary'):
        return '/en/digest/open/%s/' % digest.open_token(self.sub, self.edition, aid, rank, target)

    def test_a_click_is_counted_and_ranks_the_next_digests(self):
        response = Client().get(self.url())
        self.assertIn('/en/paper/2610.00001v1/?autorun=1', response['Location'])
        self.assertIn('utm_content=r1', response['Location'])
        click = DigestClick.objects.get()
        self.assertEqual((click.rank, click.edition, click.burst), (1, self.edition, False))
        self.assertEqual(DigestFeedback.objects.get().source, 'click')
        self.assertEqual(Client().get(self.url(target='abs'))['Location'], 'https://arxiv.org/abs/2610.00001v1')

    def test_mail_scanners_neither_count_nor_teach(self):
        Client(HTTP_USER_AGENT='Mozilla/5.0 (compatible; Barracuda Sentinel)').get(self.url())
        self.assertFalse(DigestClick.objects.exists())
        for i in range(digest_web.BURST_CLICKS):
            Client().get(self.url('2610.0000%dv1' % (i + 1), rank=i + 1))
        self.assertTrue(all(DigestClick.objects.values_list('burst', flat=True)))
        self.assertFalse(DigestFeedback.objects.exists())
        self.assertEqual(Client().get('/en/digest/open/forged/').status_code, 404)

    def test_save_to_a_new_or_an_existing_collection(self):
        token = digest.save_token(self.sub, '2610.00001v1')
        url = '/en/digest/save/%s/' % token
        AnnouncedPaper.objects.create(arxiv_id='2610.00001v1', announced=MONDAY.date(), title='Planet', authors='["A B"]')
        self.assertContains(Client().get(url), 'Or a new collection')
        self.assertFalse(Collection.objects.exists())  # a link checker following the link saves nothing
        self.assertContains(Client().post(url, {'collection': 'new', 'name': 'To read'}), 'Saved in “To read”')
        collection = Collection.objects.get()
        self.assertEqual(list(CollectionPaper.objects.values_list('arxiv_id', flat=True)), ['2610.00001v1'])
        self.assertContains(Client().post(url, {'collection': str(collection.pk)}), 'already in')
        other = Collection.objects.create(user=self.user('other'), name='Not mine')
        Client().post(url, {'collection': str(other.pk)})
        self.assertFalse(other.papers.exists())


class SettingsTests(ResearchDigestCase):
    def setUp(self):
        super().setUp()
        self.u = self.user('s', paid=True)
        self.c = Client()
        self.c.force_login(self.u)

    def test_new_settings_are_saved(self):
        self.c.post('/en/digest/', {'categories': ['math.AG'], 'level': 'general', 'volume': 'medium', 'bundle_alerts': '1'})
        sub = DigestSubscription.objects.get()
        self.assertEqual((sub.categories, sub.level, sub.volume, sub.bundle_alerts), ('math.AG', 'general', 'medium', True))
        page = self.c.get('/en/digest/')
        self.assertContains(page, 'Algebraic Geometry')  # every arXiv category is offered
        self.assertContains(page, 'data-label="Algebraic Geometry" value="math.AG" checked')

    def test_pages_in_french(self):
        page = self.c.get('/fr/digest/')
        self.assertContains(page, 'Partir de vos publications')
        self.assertContains(page, 'Ce que vous auriez reçu')
        ArxivBatch.objects.create(day=MONDAY.date(), papers=1)
        AnnouncedPaper.objects.create(arxiv_id='2610.00001v1', announced=MONDAY.date(), title='Planet clouds',
                                      abstract='x', authors='["A B"]', categories='astro-ph.EP')
        self.assertContains(self.c.post('/fr/digest/try/', {'categories': ['astro-ph.EP'], 'lang': 'fr'}),
                            '1 article de vos catégories et mots-clés')

    def test_live_preview_shows_what_would_have_been_sent(self):
        ArxivBatch.objects.create(day=MONDAY.date(), papers=2)
        for aid, title in (('2610.00001v1', 'Planet clouds'), ('2610.00002v1', 'Galaxy merger')):
            AnnouncedPaper.objects.create(arxiv_id=aid, announced=MONDAY.date(), title=title, abstract=title,
                                          authors='["A B"]', categories='astro-ph.EP')
        page = self.c.post('/en/digest/try/', {'categories': ['astro-ph.EP'], 'keywords': '-galaxy'})
        self.assertContains(page, 'Planet clouds')
        self.assertNotContains(page, 'Galaxy merger')
        self.assertContains(page, '1 paper of your categories and keywords')
        self.assertFalse(DigestSubscription.objects.exists())  # nothing saved
        self.assertEqual(Client().post('/en/digest/try/', {'categories': ['astro-ph.EP']}).status_code, 403)

    def test_import_from_arxiv_author_page_fills_the_form(self):
        feed = digest_web_feed([('2601.00001v1', 'astro-ph.EP', ['Jane Roe', 'Al Co', 'Bo Other']),
                                ('2602.00002v1', 'astro-ph.EP', ['J. Roe', 'Al Co']),
                                ('2603.00003v1', 'astro-ph.SR', ['Jane Roe'])])
        response = mock.Mock(status_code=200, text=feed)
        with mock.patch('summarizer.digest_web.requests.get', return_value=response) as get:
            page = self.c.post('/en/digest/', {'action': 'import', 'identity': 'roe_j_1', 'categories': ['cs.LG']})
        self.assertEqual(get.call_args.args[0], 'https://arxiv.org/a/roe_j_1.atom')
        self.assertContains(page, '3 of your papers read')
        self.assertContains(page, 'value="astro-ph.EP" checked')
        self.assertContains(page, 'value="cs.LG" checked')  # what was already chosen stays
        self.assertContains(page, '2601.00001v1, 2602.00002v1, 2603.00003v1')
        self.assertContains(page, 'value="Al Co"')  # a frequent co-author, not the researcher themself
        self.assertFalse(DigestSubscription.objects.exists())  # to check, then save
        self.assertContains(self.c.post('/en/digest/', {'action': 'import', 'identity': 'not an id'}), 'Give an ORCID iD')

    def test_orcid_works(self):
        works = {'group': [
            {'external-ids': {'external-id': [{'external-id-type': 'arxiv', 'external-id-value': 'arXiv:2101.00001'}]},
             'work-summary': [{'publication-date': {'year': {'value': '2021'}}}]},
            {'external-ids': {'external-id': [{'external-id-type': 'doi', 'external-id-value': '10.1093/mnras/x1'}]},
             'work-summary': [{'publication-date': {'year': {'value': '2024'}}}]}]}
        with mock.patch('summarizer.digest_web.requests.get', return_value=mock.Mock(status_code=200, json=lambda: works)):
            self.assertEqual(digest_web.orcid_works('0000-0002-1825-0097'), (['2101.00001'], ['10.1093/mnras/x1']))


def digest_web_feed(entries):
    rows = ''.join(
        '<entry><id>http://arxiv.org/abs/%s</id><title>T %s</title><summary>S</summary>%s'
        '<category term="%s"/></entry>' % (aid, aid, ''.join('<author><name>%s</name></author>' % a for a in authors), cat)
        for aid, cat, authors in entries)
    return '<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">%s</feed>' % rows


class BundleTests(ResearchDigestCase):
    def test_alerts_come_in_the_digest_instead_of_their_own_email(self):
        from summarizer import scoops
        u = self.user('b', paid=True)
        DigestSubscription.objects.create(user=u, categories='astro-ph.EP', timezone='UTC', send_hour=7, bundle_alerts=True)
        project = WatchedProject.objects.create(user=u, title='My disk project')
        ScoopMatch.objects.create(project=project, arxiv_id='2610.00005v1', title='A close disk paper', announced=MONDAY.date(),
                                  similarity=0.8, overlap='high', shared='Same disks.')
        self.assertEqual(digest_bundle.bundled_users(), {u.pk})
        self.assertEqual(scoops.send_alerts(MONDAY, 10), 0)  # left for the digest
        self.run_digest({'2610.00001v1': fake_paper('2610.00001v1', 'astro-ph.EP', 'Planet')})
        body = mail.outbox[0].body
        self.assertIn('== Close to your projects ==', body)
        self.assertIn('A close disk paper (Strong overlap)', body)
        self.assertTrue(ScoopMatch.objects.get().emailed)

    def test_alerts_alone_still_make_an_email(self):
        u = self.user('b2', paid=True)
        DigestSubscription.objects.create(user=u, categories='astro-ph.EP', timezone='UTC', send_hour=7, bundle_alerts=True)
        project = WatchedProject.objects.create(user=u, title='P')
        ScoopMatch.objects.create(project=project, arxiv_id='x', title='Close', announced=MONDAY.date(), similarity=0.8,
                                  overlap='partial')
        self.run_digest({'2610.00001v1': fake_paper('2610.00001v1', 'cs.LG', 'Elsewhere')})
        self.assertEqual(mail.outbox[0].subject, 'Your research alerts')


class SynthesisTests(ResearchDigestCase):
    def test_trends_cite_their_papers_and_notable_papers_say_why(self):
        papers = {p['arxiv_id']: p for p in [fake_paper('2610.0000%dv1' % i, 'astro-ph.EP', 'Planet %d' % i) for i in range(6)]}
        answer = {'trends': [{'title': 'Clouds', 'text': 'Many.', 'papers': ['2610.00001v1', 'bogus', '2610.00002v1']}],
                  'notable': [{'id': '2610.00003v1', 'why': 'First direct map.'}, 'bogus']}
        with mock.patch('summarizer.digest._chat', return_value=json.dumps(answer)) as chat:
            content = digest.category_synthesis('astro-ph.EP', 'en', MONDAY.date(), papers)
        self.assertEqual([p['arxiv_id'] for p in content['trends'][0]['papers']], ['2610.00001v1', '2610.00002v1'])
        self.assertEqual(content['notable'], [{'arxiv_id': '2610.00003v1', 'title': 'Planet 3', 'why': 'First direct map.'}])
        self.assertIn('Shortlist:', chat.call_args.args[0])
        self.assertTrue(CategorySynthesis.objects.exists())


class StatsTests(ResearchDigestCase):
    def test_clicks_by_rank(self):
        sub = DigestSubscription.objects.create(user=self.user('st'), categories='cs.LG')
        edition = DigestEdition.objects.create(subscription=sub, sent=datetime.datetime.now(datetime.timezone.utc),
                                               days='2026-10-19', papers='["a", "b"]')
        DigestClick.objects.create(subscription=sub, edition=edition, arxiv_id='a', rank=1)
        DigestClick.objects.create(subscription=sub, edition=edition, arxiv_id='a', rank=1, target='pdf')
        out = StringIO()
        call_command('digest_stats', stdout=out)
        self.assertRegex(out.getvalue(), r'\n\s+1\s+1\s+1\s+100\.0%')
        self.assertRegex(out.getvalue(), r'\n\s+2\s+1\s+0\s+0\.0%')
