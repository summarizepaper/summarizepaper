from django.core.management.base import BaseCommand, CommandError
from summarizer import user_outreach


class Command(BaseCommand):
    help = 'Relance de retour utilisateur : aperçu par défaut, 6 messages maximum par jour (heure de Paris).'

    def add_arguments(self, parser):
        parser.add_argument('--send', action='store_true')
        parser.add_argument('--limit', type=int, default=6)
        parser.add_argument('--segment', choices=['auto','recent','older'], default='auto')

    def handle(self, **options):
        if not options['send']:
            segments = ['recent','older'] if options['segment'] == 'auto' else [options['segment']]
            users = [user for segment in segments for user in user_outreach.candidates(segment)]
            self.stdout.write('%d candidats, segment %s' % (len(users),options['segment']))
            for user in users[:max(0,min(options['limit'],6))]:
                self.stdout.write('Utilisateur %s · %s · %s' % (user.pk,user_outreach.language(user),user.outreach_activity.date()))
            self.stdout.write(user_outreach.message('fr','https://www.summarizepaper.com/desinscription/APERÇU')[1])
            return
        try: sent = user_outreach.send_batch(options['limit'],options['segment'])
        except user_outreach.BatchStopped as exc: raise CommandError(str(exc))
        self.stdout.write('%d message(s) accepté(s) par SMTP ; livraison et réponses à vérifier.' % len(sent))
        if not sent: self.stdout.write('Plus aucun compte à contacter.')
