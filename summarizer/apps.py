from django.apps import AppConfig


class SummarizerConfig(AppConfig):
    name = 'summarizer'

    def ready(self):
        # Message-IDs name the sending machine by default (a Heroku dyno): name the domain of the From address,
        # one spam signal fewer for the activation and digest emails
        from django.core.mail.utils import DNS_NAME
        DNS_NAME._fqdn = 'summarizepaper.com'
