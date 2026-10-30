"""Public contact form: validated input, shared rate limits and duplicate protection."""
import hashlib
import logging
import time

from django import forms
from django.core.cache import cache
from django.core.mail import EmailMessage
from django.shortcuts import redirect, render
from django.utils.translation import gettext as _

from . import access

logger = logging.getLogger(__name__)


class ContactForm(forms.Form):
    name = forms.CharField(max_length=120)
    email = forms.EmailField(max_length=254)
    message = forms.CharField(max_length=10000, widget=forms.Textarea)
    website = forms.CharField(required=False, widget=forms.HiddenInput)

    def clean_name(self):
        name = self.cleaned_data['name']
        if '\n' in name or '\r' in name:
            raise forms.ValidationError(_('Enter a valid name.'))
        return name


def _over_limit(scope, identity, maximum):
    key = 'contact:limit:%s:%s:%s' % (scope, identity, int(time.time()) // 3600)
    cache.add(key, 0, 7200)
    return cache.incr(key) > maximum


def page(request):
    form = ContactForm(request.POST if request.method == 'POST' else None)
    context = {'form': form, 'sent': 1 if request.session.pop('contact_sent', False) else None}
    if request.method != 'POST':
        return render(request, 'summarizer/contact.html', context)
    if not form.is_valid():
        return render(request, 'summarizer/contact.html', context, status=400)
    if form.cleaned_data['website']:
        # Pretend success for bots filling the hidden field, without sending mail.
        request.session['contact_sent'] = True
        return redirect('contact')
    data = form.cleaned_data
    email = data['email'].lower()
    digest = hashlib.sha256((email + '\n' + data['name'] + '\n' + data['message']).encode()).hexdigest()
    duplicate_key = 'contact:message:' + digest
    try:
        if cache.get(duplicate_key):
            request.session['contact_sent'] = True
            return redirect('contact')
        ip = access.hash_ip(access.client_ip(request))
        email_hash = hashlib.sha256(email.encode()).hexdigest()
        if (_over_limit('ip', ip, 3) or _over_limit('email', email_hash, 3)
                or _over_limit('global', 'all', 30)):
            form.add_error(None, _('Too many messages. Please try again in one hour.'))
            return render(request, 'summarizer/contact.html', context, status=429)
        # Reserve atomically before SMTP so simultaneous submissions send only once.
        if not cache.add(duplicate_key, True, 86400):
            request.session['contact_sent'] = True
            return redirect('contact')
    except Exception:
        logger.exception('Contact form protection unavailable; email not sent')
        form.add_error(None, _('The email could not be sent. If it happens again, please contact us.'))
        return render(request, 'summarizer/contact.html', context, status=503)
    try:
        sent = EmailMessage(
            'Paper Summarization Contact Form: ' + data['name'],
            data['message'] + '\n\n\nFrom: ' + data['name'] + ' (' + email + ')',
            'SummarizePaper <communication@summarizepaper.com>',
            ['contact@summarizepaper.com'], reply_to=[email],
            headers={'Message-From': 'www.summarizepaper.com'},
        ).send(fail_silently=False)
        if sent != 1:
            raise RuntimeError('Contact email not accepted')
    except Exception:
        cache.delete(duplicate_key)
        logger.exception('Contact form email failed')
        context['sent'] = 0
        return render(request, 'summarizer/contact.html', context, status=503)
    request.session['contact_sent'] = True
    return redirect('contact')
