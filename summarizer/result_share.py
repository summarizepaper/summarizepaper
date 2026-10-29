"""Read-only links to a private result (a literature review, a bibliography check), for a supervisor, co-authors
or a team: anyone with the link reads the result as it was made, never who made it. The owner can extend the
link or stop it, as for a shared collection (see library.share)."""
import datetime
import secrets

from django.http import Http404
from django.utils import formats, timezone
from django.utils.translation import get_language, gettext as _, ngettext

from .models import Generated

SHARE_DAYS = 90  # a shared link works this long after its last sharing


def act(row, action):
    """Share a finished result (again: 90 more days, the same link) or stop sharing it. False for another action."""
    if action == 'share' and row.status == 'done':
        if not row.share_token:
            row.share_token = secrets.token_urlsafe(16)
        row.share_expires = timezone.now() + datetime.timedelta(days=SHARE_DAYS)
    elif action == 'unshare':
        row.share_token, row.share_expires = '', None
    else:
        return False
    row.save(update_fields=['share_token', 'share_expires'])
    return True


def shared_url(request, row, path):
    """The public link of a result its owner shares ('' when not shared); path: the tool's, e.g. 'refcheck'."""
    if not row.share_token or (row.share_expires and row.share_expires < timezone.now()):
        return ''
    return request.build_absolute_uri('/%s/%s/shared/%s/' % (get_language(), path, row.share_token))


def shared_row(kind, token):
    """The finished result shared under this token (404 when stopped or expired)."""
    row = Generated.objects.filter(kind=kind, share_token=token, status='done').first() if token else None
    if row is None or (row.share_expires and row.share_expires < timezone.now()):
        raise Http404
    return row


def timing(expires, days=SHARE_DAYS, now=None):
    """Last activation/renewal and remaining validity, including existing links.

    All expiring links are set to now + their fixed lifetime when shared or
    renewed. Subtracting that lifetime recovers that event, not initial creation.
    """
    if expires is None:
        return _('No scheduled expiration.')
    now = now or timezone.now()
    def date(value):
        local = timezone.localtime(value)
        return '%s (%s)' % (formats.date_format(local, 'DATETIME_FORMAT'), local.tzname())
    activated = expires - datetime.timedelta(days=days)
    dates = _('Activated or renewed on %(activated)s. Expires on %(expires)s.') % {
        'activated': date(activated), 'expires': date(expires)}
    seconds = (expires - now).total_seconds()
    if seconds <= 0:
        return dates + ' ' + _('Expired link.')
    minutes = int(seconds // 60)
    if minutes < 1:
        remaining = _('Less than a minute')
    else:
        d, rest = divmod(minutes, 1440)
        h, m = divmod(rest, 60)
        parts = []
        if d:
            parts.append(ngettext('%(n)s day', '%(n)s days', d) % {'n': d})
        if h:
            parts.append(ngettext('%(n)s hour', '%(n)s hours', h) % {'n': h})
        if not d and m:
            parts.append(ngettext('%(n)s minute', '%(n)s minutes', m) % {'n': m})
        remaining = ', '.join(parts)
    return dates + ' ' + _('Time remaining: %(remaining)s.') % {'remaining': remaining}
