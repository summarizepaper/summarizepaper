"""CloudMailin delivery events webhook: what happened to an email after the SMTP server accepted it.

CloudMailin posts {"events": [{"kind", "recipient", "message_id", "details"}, ...]} to
/cloudmailin/events/<CLOUDMAILIN_EVENTS_SECRET>/ (set in its dashboard). Every event is kept as an
EmailEvent; feedback invitations (OutreachContact) follow them, matched by recipient since Django's
SMTP backend does not return CloudMailin's queue id.
"""
import datetime
import hmac
import json

from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseBadRequest
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

from . import mailcheck
from .models import EmailEvent, OutreachContact

MAX_EVENTS = 500
OUTREACH_WINDOW = datetime.timedelta(days=30)
# event kind -> (outreach status it sets, statuses it may replace); a later event never undoes a reply or opt-out
OUTREACH_STATUS = {
    'dispatch': ('dispatched', ('sending', 'sent')),
    'delivery': ('delivered', ('sending', 'sent', 'dispatched')),
    'suppressed': ('suppressed', ('sending', 'sent', 'dispatched')),
    'retries_exhausted': ('failed', ('sending', 'sent', 'dispatched')),
    'bounce': ('bounced', ('sending', 'sent', 'dispatched', 'delivered')),
}


def record(event):
    recipient = str(event.get('recipient') or '').strip().lower()[:254]
    kind = str(event.get('kind') or '')[:40]
    if not recipient or not kind:
        return
    details = event.get('details')
    EmailEvent.objects.create(recipient=recipient, kind=kind, message_id=str(event.get('message_id') or '')[:200],
                              details=details if isinstance(details, dict) else {'value': details})
    if kind == 'bounce':
        # A hard bounce: writing again hurts the sender score of every email we send.
        mailcheck.suppress(recipient, 'CloudMailin: bounce')
    if kind in OUTREACH_STATUS:
        status, replaces = OUTREACH_STATUS[kind]
        OutreachContact.objects.filter(email=recipient, status__in=replaces,
                                       created__gte=timezone.now() - OUTREACH_WINDOW).update(status=status)


@csrf_exempt
@require_POST
def webhook(request, secret):
    if not settings.CLOUDMAILIN_EVENTS_SECRET or not hmac.compare_digest(secret, settings.CLOUDMAILIN_EVENTS_SECRET):
        raise Http404
    try:
        payload = json.loads(request.body)
    except ValueError:
        return HttpResponseBadRequest('invalid json')
    events = payload.get('events') if isinstance(payload, dict) else None
    if not isinstance(events, list):
        return HttpResponseBadRequest('no events')
    for event in events[:MAX_EVENTS]:
        if isinstance(event, dict):
            record(event)
    return HttpResponse('ok')
