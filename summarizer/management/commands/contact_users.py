from django.core.management.base import BaseCommand, CommandError
from summarizer import user_outreach


class Command(BaseCommand):
    help = 'Relance de retour utilisateur : aperçu par défaut, 6 messages maximum, 48 h entre deux lots.'

    def add_arguments(self, parser):
        parser.add_argument('--send', action='store_true')
        parser.add_argument('--limit', type=int, default=6)
        parser.add_argument('--segment', choices=['recent','older'], default='recent')

    def handle(self, **options):
        if not options['send']:
            users = user_outreach.candidates(options['segment'])
            self.stdout.write('%d candidats, segment %s' % (len(users),options['segment']))
            for user in users[:max(0,min(options['limit'],6))]:
                self.stdout.write('Utilisateur %s · %s · %s' % (user.pk,user_outreach.language(user),user.outreach_activity.date()))
            self.stdout.write(user_outreach.message('fr','https://www.summarizepaper.com/desinscription/APERÇU')[1])
            return
        try: sent = user_outreach.send_batch(options['limit'],options['segment'])
        except user_outreach.BatchStopped as exc: raise CommandError(str(exc))
        self.stdout.write('%d message(s) accepté(s) par SMTP ; livraison et réponses à vérifier.' % len(sent))
