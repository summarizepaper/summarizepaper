"""Premium alerts in the digest: one email in the morning instead of five.

A Premium user may ask their digest to carry their scoop, citation, new version and publication alerts
(DigestSubscription.bundle_alerts). Each module then leaves that user's queued alerts for the digest,
which takes them as sections when it is sent and marks them sent only once the email went out.
"""
import logging

from django.utils import translation

from . import access
from .models import DigestSubscription

logger = logging.getLogger(__name__)


def bundled_users():
    """Ids of the users whose alerts wait for their digest: an active digest asks for them, with Premium."""
    users = set(DigestSubscription.objects.filter(active=True, bundle_alerts=True).values_list('user_id', flat=True))
    from django.contrib.auth.models import User
    return {user.pk for user in User.objects.filter(pk__in=users) if access.has_tool(user, 'digest')}


def collect(user, lang, base_url, now):
    """The sections of alerts waiting for this user's digest: [{'title', 'items', 'more', 'manage_url', 'commit'}].
    commit() marks a section's alerts sent; the digest calls it once the email went out."""
    from . import changes, citations, publication_alerts, scoops
    sections = []
    with translation.override(lang):
        for build in (lambda: scoops.bundle_section(user, base_url, now),
                      lambda: citations.bundle_section(user, base_url, lang, now),
                      lambda: changes.bundle_section(user, base_url, lang),
                      lambda: publication_alerts.bundle_section(user, base_url, lang)):
            try:
                section = build()
            except Exception:
                logger.exception('An alert section of the digest of %s failed; it waits for the next one', user.username)
                continue
            if section and section['items']:
                sections.append(section)
    return sections
