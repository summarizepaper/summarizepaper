"""Stable scopes shared by administrative grants and server-side tool gates."""
from django.utils.translation import gettext_lazy as _

TOOL_CHOICES = (
    ('all', _('Full Premium')),
    ('presskit', _('Press kit')),
    ('journal_club', _('Journal club')),
    ('refcheck', _('Bibliography check')),
    ('missed', _('What did I miss?')),
    ('scoops', _('Scoop watch')),
    ('conference', _('Conference companion')),
    ('teams', _('Who works on it?')),
    ('collections', _('Collections and comparison tables')),
    ('zotero', _('Zotero integration')),
    ('changes', _('Version alerts')),
    ('citations', _('Citation alerts')),
    ('digest', _('Daily Digest')),
    ('letters', _('Weekly letters')),
    ('trends', _('Rising topics')),
    ('saved_searches', _('Saved search watches')),
    ('reading', _('Reading levels')),
)
TOOL_KEYS = frozenset(k for k, _ in TOOL_CHOICES)
GENERATED_TOOLS = {
    'presskit': 'presskit', 'presskit_status': 'presskit',
    'slides': 'journal_club', 'slides_v2': 'journal_club',
    'slides_edit': 'journal_club', 'slides_edit_v2': 'journal_club',
    'compare': 'collections', 'questions': 'collections',
    **{k: k for k in ('missed', 'zotero', 'refcheck', 'changes', 'teams', 'conference')},
}
