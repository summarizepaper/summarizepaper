from django.core.management.base import BaseCommand

from summarizer import conversion_emails, digest


class Command(BaseCommand):
    help = ('Email the members who signed up before the free tries existed about the tries they may still run, once '
            'each (conversion_emails.send_announce). Without --send, only lists who would get it. --limit keeps each '
            'run small: run it again, a batch a day, until done.')

    def add_arguments(self, parser):
        parser.add_argument('--send', action='store_true', help='Really send the emails.')
        parser.add_argument('--limit', type=int, default=20)
        parser.add_argument('--base-url', default=digest.BASE_URL)

    def handle(self, *args, **options):
        waiting = conversion_emails.announce_candidates()
        self.stdout.write('%d member(s) have not had it yet' % len(waiting))
        if not options['send']:
            for user in waiting[:options['limit']]:
                self.stdout.write('  %s  %s  last login %s' % (user.pk, user.email.split('@')[-1],
                                                               user.last_login.date() if user.last_login else '-'))
            return
        sent = sum(conversion_emails.send_announce(user, options['base_url']) for user in waiting[:options['limit']])
        self.stdout.write(self.style.SUCCESS('%d sent' % sent))
