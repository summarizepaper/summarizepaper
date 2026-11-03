import json
from datetime import timedelta

from django.utils import timezone

from summarizer import removal
from summarizer.models import (AIassistant, ChatConversation, ChatShare, Collection, CollectionPaper, ConferenceMark,
                               ConferenceMember, ConferenceProgram, FollowedPaper, Generated, PaperVisit,
                               ReferenceWatch, SavedSearch, StudentShortlist, TopicMapWatch, ToolUse, Trash,
                               WatchedProject)
from summarizer.test_tools import ToolsTestCase


class RemovalTests(ToolsTestCase):
    def setUp(self):
        super().setUp()
        self.reader = self.user(paid=True)
        self.other = self.user('other', paid=True)
        self.client = self.client_for(self.reader)

    def result(self, kind, key, user=None, data=None, status='done'):
        return Generated.objects.create(kind=kind, key=key, user=user or self.reader, status=status,
                                        content=json.dumps(data or {'title': key}))

    def test_one_result_goes_with_what_hangs_on_it(self):
        missed = self.result('missed', 'm1')
        draft = self.result('missed_draft', 'm1')
        review = self.result('literature', 'l1', data={'question': 'Clouds?'})
        pool = Generated.objects.create(kind='literature_pool', key='l1', content='{}')
        check = self.result('refcheck', 'r1', data={'entries': 3})
        ReferenceWatch.objects.create(user=self.reader, check_key='r1', until=timezone.now() + timedelta(days=30))
        topic = self.result('teams', 't1')
        TopicMapWatch.objects.create(user=self.reader, row=topic)
        kept = self.result('missed', 'm2')
        for row in (missed, review, check, topic):
            response = self.client.post('/en/delete/results/', {'one': row.pk, 'next': '/en/missed/'})
            self.assertTrue(response['Location'].startswith('/en/missed/?done=deleted&deleted=1&trash='))
        self.assertEqual(list(Generated.objects.values_list('pk', flat=True)), [kept.pk])
        self.assertFalse(ReferenceWatch.objects.exists())
        self.assertFalse(TopicMapWatch.objects.exists())
        self.assertFalse(Generated.objects.filter(pk__in=[draft.pk, pool.pk]).exists())

    def test_ticked_results_only_the_users_own_and_shared_ones_leave_the_list(self):
        mine = [self.result('teams', 't%d' % i) for i in range(3)]
        theirs = self.result('teams', 'theirs', user=self.other)
        kit = Generated.objects.create(kind='presskit', key='2101.00001', status='done', content='{}')
        ToolUse.objects.create(user=self.reader, row=kit, url='/en/press-kit/2101.00001/', title='Clouds')
        ToolUse.objects.create(user=self.other, row=kit, url='/en/press-kit/2101.00001/', title='Clouds')
        response = self.client.post('/en/delete/results/', {
            'r': [mine[0].pk, mine[1].pk, theirs.pk, kit.pk], 'next': '/en/history/?tab=results'})
        self.assertTrue(response['Location'].startswith('/en/history/?tab=results&done=deleted&deleted=3&trash='))
        self.assertEqual(set(Generated.objects.values_list('pk', flat=True)), {mine[2].pk, theirs.pk, kit.pk})
        self.assertEqual(list(ToolUse.objects.values_list('user', flat=True)), [self.other.pk])
        response = self.client.get('/en/history/', {'tab': 'results', 'done': 'deleted', 'deleted': 3})
        self.assertContains(response, '3 items deleted.')
        self.assertContains(response, 'name="r" value="%d"' % mine[2].pk)

    def test_every_result_of_a_filter(self):
        for i in range(3):
            self.result('missed', 'm%d' % i)
        kept = self.result('teams', 't1')
        self.client.post('/en/delete/results/', {'all': '1', 'kind': 'missed'})
        self.assertEqual(list(Generated.objects.values_list('pk', flat=True)), [kept.pk])

    def test_next_stays_on_this_site(self):
        row = self.result('teams', 't1')
        response = self.client.post('/en/delete/results/', {'one': row.pk, 'next': '//evil.example/x'})
        self.assertTrue(response['Location'].startswith('/en/history/?tab=results'))

    def test_conference_program_goes_with_its_last_ranking(self):
        program = ConferenceProgram.objects.create(user=self.reader, name='ACAT')
        ConferenceMember.objects.create(program=program, user=self.other)
        ConferenceMark.objects.create(program=program, user=self.other, talk='t1', state='going')
        first = self.result('conference', 'c1', data={'program': program.pk, 'talks': []})
        second = self.result('conference', 'c2', data={'program': program.pk, 'talks': []})
        self.client.post('/en/delete/results/', {'one': first.pk})
        self.assertTrue(ConferenceProgram.objects.filter(pk=program.pk).exists())
        self.client.post('/en/delete/results/', {'one': second.pk})
        self.assertFalse(ConferenceProgram.objects.filter(pk=program.pk).exists())
        self.assertFalse(ConferenceMark.objects.exists())

    def test_a_members_ranking_keeps_the_program(self):
        program = ConferenceProgram.objects.create(user=self.reader, name='ACAT')
        self.result('conference', 'theirs', user=self.other, data={'program': program.pk, 'talks': []})
        mine = self.result('conference', 'c1', data={'program': program.pk, 'talks': []})
        self.client.post('/en/delete/results/', {'one': mine.pk})
        self.assertTrue(ConferenceProgram.objects.filter(pk=program.pk).exists())

    def test_tool_pages_list_results_with_their_delete_buttons(self):
        rows = {kind: self.result(kind, kind + '1', data={'title': kind, 'entries': 1}) for kind in
                ('literature', 'missed', 'refcheck', 'teams', 'students', 'conference')}
        for kind, path in (('literature', 'literature-review'), ('missed', 'missed'), ('refcheck', 'refcheck'),
                           ('teams', 'teams'), ('students', 'students'), ('conference', 'conference')):
            response = self.client.get('/en/%s/' % path)
            self.assertContains(response, 'name="one" value="%d"' % rows[kind].pk, msg_prefix=kind)
            self.assertContains(response, 'Delete the ticked ones', msg_prefix=kind)

    def test_collections_deleted_together_with_their_analyses(self):
        a = Collection.objects.create(user=self.reader, name='A')
        b = Collection.objects.create(user=self.reader, name='B')
        theirs = Collection.objects.create(user=self.other, name='C')
        Generated.objects.create(kind='compare', key='%d:general' % a.pk, content='{}')
        other_analysis = Generated.objects.create(kind='compare', key='%d:general' % theirs.pk, content='{}')
        response = self.client.get('/en/collections/')
        self.assertContains(response, 'name="c" value="%d"' % a.pk)
        self.client.post('/en/delete/collections/', {'c': [a.pk, b.pk, theirs.pk], 'next': '/en/collections/'})
        self.assertEqual(list(Collection.objects.values_list('pk', flat=True)), [theirs.pk])
        self.assertEqual(list(Generated.objects.values_list('pk', flat=True)), [other_analysis.pk])

    def test_conversations_deleted_together(self):
        for room in ('roomone1234', 'roomtwo1234'):
            AIassistant.objects.create(user=self.reader, arxiv_id=room, query='Q', response='A')
            ChatShare.objects.create(user=self.reader, room=room, token=room, upto=timezone.now())
        ChatConversation.objects.create(user=self.reader, room='roomone1234', title='Named')
        AIassistant.objects.create(user=self.other, arxiv_id='roomthree1234', query='Q', response='A')
        response = self.client.get('/en/history/', {'tab': 'conversations'})
        self.assertContains(response, 'name="room" value="roomone1234"')
        response = self.client.post('/en/delete/conversations/', {
            'room': ['roomone1234', 'roomtwo1234', 'roomthree1234'], 'next': '/en/history/?tab=conversations'})
        self.assertTrue(response['Location'].startswith('/en/history/?tab=conversations&done=deleted&deleted=2&trash='))
        self.assertFalse(AIassistant.objects.filter(user=self.reader, active=True).exists())
        self.assertTrue(AIassistant.objects.filter(user=self.other, active=True).exists())
        self.assertFalse(ChatShare.objects.filter(active=True).exists())
        self.assertFalse(ChatConversation.objects.exists())

    def test_french_page(self):
        self.result('teams', 't1')
        response = self.client.get('/fr/history/', {'tab': 'results', 'done': 'deleted', 'deleted': 2})
        self.assertContains(response, 'Supprimer la sélection')
        self.assertContains(response, '2 éléments supprimés.')
        self.assertContains(response, 'Supprimer les {n} résultats cochés')

    def test_a_running_result_is_kept(self):
        running = self.result('missed', 'm1', status='pending')
        self.assertContains(self.client.get('/en/missed/'), 'In progress')
        self.assertNotContains(self.client.get('/en/missed/'), 'name="one" value="%d"' % running.pk)
        self.client.post('/en/delete/results/', {'one': running.pk})
        self.assertTrue(Generated.objects.filter(pk=running.pk).exists())
        Generated.objects.filter(pk=running.pk).update(updated=timezone.now() - timedelta(hours=1))  # lost
        self.client.post('/en/delete/results/', {'one': running.pk})
        self.assertFalse(Generated.objects.filter(pk=running.pk).exists())

    def test_undo_brings_back_everything_that_went(self):
        review = self.result('literature', 'l1', data={'question': 'Clouds?'})
        review.share_token = 'tok'
        review.save()
        Generated.objects.create(kind='literature_pool', key='l1', content='{"pool": 1}')
        collection = Collection.objects.create(user=self.reader, name='Thesis')
        CollectionPaper.objects.create(collection=collection, arxiv_id='2101.00001', note='Key')
        Generated.objects.create(kind='compare', key='%d:general' % collection.pk, content='{}')
        response = self.client.post('/en/delete/results/', {'one': review.pk, 'next': '/en/literature-review/'})
        page = self.client.get(response['Location'])
        self.assertContains(page, '1 item deleted.')
        self.assertContains(page, 'Undo')
        trash = Trash.objects.get()
        self.assertIn('Clouds?', trash.label)
        self.client.post('/en/delete/collections/', {'one': collection.pk})
        self.assertEqual(Generated.objects.count(), 0)
        response = self.client.post('/en/trash/%d/restore/' % trash.pk, {'next': '/en/literature-review/'})
        self.assertEqual(response['Location'], '/en/literature-review/?done=restored')
        restored = Generated.objects.get(kind='literature')
        self.assertEqual((restored.pk, restored.share_token, restored.data()), (review.pk, 'tok', {'question': 'Clouds?'}))
        self.assertTrue(Generated.objects.filter(kind='literature_pool', key='l1').exists())
        history = self.client.get('/en/history/')
        self.assertContains(history, 'Recently deleted')
        self.assertContains(history, 'Collections: Thesis')
        self.client.post('/en/trash/%d/restore/' % Trash.objects.get().pk)
        self.assertEqual(CollectionPaper.objects.get().note, 'Key')
        self.assertTrue(Generated.objects.filter(kind='compare').exists())
        self.assertFalse(Trash.objects.exists())

    def test_someone_elses_trash_and_a_clash_are_not_restored(self):
        followed = FollowedPaper.objects.create(user=self.reader, arxiv_id='2101.00001', title='Clouds')
        self.client.post('/en/delete/followed/', {'pk': [followed.pk]})
        trash = Trash.objects.get()
        self.client_for(self.other).post('/en/trash/%d/restore/' % trash.pk)
        self.assertFalse(FollowedPaper.objects.exists())
        FollowedPaper.objects.create(pk=followed.pk, user=self.reader, arxiv_id='2101.00009')  # its id is taken again
        response = self.client.post('/en/trash/%d/restore/' % trash.pk, {'next': '/en/citations/'})
        self.assertIn('done=unrestorable', response['Location'])
        self.assertTrue(Trash.objects.exists())

    def test_conversations_come_back_with_their_name_and_link(self):
        AIassistant.objects.create(user=self.reader, arxiv_id='roomone1234', query='Q', response='A')
        ChatShare.objects.create(user=self.reader, room='roomone1234', token='t1', upto=timezone.now())
        ChatConversation.objects.create(user=self.reader, room='roomone1234', title='Named')
        self.client.post('/en/delete/conversations/', {'one': 'roomone1234'})
        self.assertEqual(Trash.objects.get().label, 'Conversations: Named')
        self.client.post('/en/trash/%d/restore/' % Trash.objects.get().pk)
        self.assertTrue(AIassistant.objects.get().active)
        self.assertTrue(ChatShare.objects.get().active)
        self.assertEqual(ChatConversation.objects.get().title, 'Named')

    def test_trash_emptied_and_purged(self):
        for i in range(3):
            SavedSearch.objects.create(user=self.reader, name='S%d' % i, fingerprint='f%d' % i)
        self.client.post('/en/delete/saved-searches/', {'pk': list(SavedSearch.objects.values_list('pk', flat=True))[:2]})
        self.client.post('/en/delete/saved-searches/', {'one': SavedSearch.objects.get().pk})
        self.assertEqual(Trash.objects.count(), 2)
        self.client.post('/en/trash/empty/', {'one': Trash.objects.first().pk})
        self.assertEqual(Trash.objects.count(), 1)
        Trash.objects.update(created=timezone.now() - timedelta(days=removal.TRASH_DAYS + 1))
        self.assertEqual(removal.purge(), 1)
        Trash.objects.create(user=self.reader, label='x')
        self.client.post('/en/trash/empty/')
        self.assertFalse(Trash.objects.exists())

    def test_reading_history_deleted_for_real(self):
        article = self.paper()
        PaperVisit.objects.create(user=self.reader, arxiv_id=article.arxiv_id, paper=article, note='Mine', reading='key')
        PaperVisit.objects.create(user=self.reader, arxiv_id='2101.00009', hidden=True)
        response = self.client.post('/en/history/clear/', {'next': '/en/history/?tab=readings'})
        self.assertTrue(response['Location'].startswith('/en/history/?tab=readings&done=deleted&deleted=2&trash='))
        self.assertFalse(PaperVisit.objects.exists())
        self.client.post('/en/trash/%d/restore/' % Trash.objects.get().pk)
        self.assertEqual(PaperVisit.objects.get(paper=article).note, 'Mine')
        visit = PaperVisit.objects.get(paper=article)
        self.client.post('/en/history/selection/', {'action': 'delete', 'v': [visit.pk], 'next': '/en/history/?tab=readings'})
        self.assertEqual(list(PaperVisit.objects.values_list('arxiv_id', flat=True)), ['2101.00009'])

    def test_the_other_lists_delete_together(self):
        projects = [WatchedProject.objects.create(user=self.reader, title='P%d' % i, description='d') for i in range(2)]
        leads = [StudentShortlist.objects.create(user=self.reader, candidate_key='k%d' % i, snapshot={'name': 'N%d' % i})
                 for i in range(2)]
        theirs = StudentShortlist.objects.create(user=self.other, candidate_key='k', snapshot={'name': 'X'})
        self.assertContains(self.client.get('/en/scoops/'), 'form="delete-projects"')
        self.assertContains(self.client.get('/en/students/leads/'), 'form="delete-leads"')
        self.client.post('/en/delete/scoops/', {'pk': [p.pk for p in projects]})
        self.client.post('/en/delete/leads/', {'pk': [l.pk for l in leads] + [theirs.pk]})
        self.assertFalse(WatchedProject.objects.exists())
        self.assertEqual(list(StudentShortlist.objects.all()), [theirs])
        self.assertEqual(Trash.objects.first().label, 'Placement leads: N0, N1')
        followed = FollowedPaper.objects.create(user=self.reader, arxiv_id='2101.00001', title='Clouds')
        self.assertContains(self.client.get('/en/citations/'), 'form="delete-followed"')
        self.client.post('/en/citations/', {'action': 'unfollow', 'pk': followed.pk})  # its own button: to the trash too
        self.assertEqual(Trash.objects.first().label, 'Citation alerts: Clouds')

    def test_result_page_has_its_delete_button(self):
        row = self.result('teams', 't1', data={'title': 'Clouds', 'people': [], 'institutions': [], 'papers': []})
        response = self.client.get('/en/teams/t1/')
        self.assertContains(response, 'Delete this result')
        self.assertContains(response, 'name="one" value="%d"' % row.pk)
