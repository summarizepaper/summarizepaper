"""Validation and journal lookup for the optional search refinements."""
import hashlib
import re

from django import forms
from django.core.cache import cache
from django.http import JsonResponse
from django.utils import timezone
from django.utils.translation import gettext_lazy as _
from django.views.decorators.http import require_GET

from . import sources


class SourceIds(forms.Field):
    widget = forms.SelectMultiple

    def clean(self, value):
        values = list(dict.fromkeys(value or []))
        if len(values) > 5 or any(not re.fullmatch(r'S\d{1,15}', v) for v in values):
            raise forms.ValidationError(_('Choose up to five journals from the suggestions.'))
        return sorted(values)


def clean_orcid(value):
    value = re.sub(r'^https?://orcid\.org/', '', value.strip().rstrip('/'), flags=re.I).upper()
    if not value:
        return ''
    if not re.fullmatch(r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]', value):
        raise forms.ValidationError(_('Enter a valid ORCID, for example 0000-0002-9322-3515.'))
    total = 0
    for digit in value.replace('-', '')[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    if value[-1] != ('X' if check == 10 else str(check)):
        raise forms.ValidationError(_('The ORCID checksum is invalid.'))
    return value


class SearchFilters(forms.Form):
    year_from = forms.IntegerField(required=False, min_value=1000, label=_('From year'))
    year_to = forms.IntegerField(required=False, min_value=1000, label=_('To year'))
    sort = forms.ChoiceField(required=False, choices=[('relevance', _('Relevance')),
        ('newest', _('Newest first')), ('oldest', _('Oldest first'))], label=_('Sort by'))
    orcid = forms.CharField(required=False, max_length=100, label=_('Author ORCID'),
                            widget=forms.TextInput(attrs={'placeholder': '0000-0002-9322-3515'}))
    journals = SourceIds(required=False, label=_('Include journals'))
    exclude_journals = SourceIds(required=False, label=_('Exclude journals'))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for key in ('year_from', 'year_to'):
            self.fields[key].max_value = timezone.now().year + 1
            self.fields[key].widget.attrs.update({'min': 1000, 'max': timezone.now().year + 1})

    def clean_orcid(self):
        return clean_orcid(self.cleaned_data['orcid'])

    def clean(self):
        data = super().clean()
        start, end = data.get('year_from'), data.get('year_to')
        # Dynamic max_value must be validated as well as shown in the input.
        for key in ('year_from', 'year_to'):
            if data.get(key) and data[key] > timezone.now().year + 1:
                self.add_error(key, _('Choose a year no later than next year.'))
        if start and end and start > end:
            self.add_error('year_to', _('The end year must be on or after the start year.'))
        if set(data.get('journals', [])) & set(data.get('exclude_journals', [])):
            self.add_error('exclude_journals', _('A journal cannot be included and excluded at the same time.'))
        return data

    def options(self):
        return {k: v for k, v in self.cleaned_data.items() if v and not (k == 'sort' and v == 'relevance')}


def remember_journal(row):
    source_id = row.get('id', '').rsplit('/', 1)[-1]
    if not re.fullmatch(r'S\d{1,15}', source_id):
        return None
    item = {'id': source_id, 'name': row.get('display_name') or source_id,
            'issn': row.get('issn_l') or '', 'publisher': row.get('host_organization_name') or ''}
    cache.set('search-journal:' + source_id, item, 86400)
    return item


def selected_journals(ids):
    missing = [v for v in ids if cache.get('search-journal:' + v) is None]
    if missing and sources.search_allowed():
        try:
            rows = sources._json(sources.OPENALEX + '/sources', sources._openalex_params({
                'filter': 'type:journal,ids.openalex:' + '|'.join(missing), 'per_page': 10,
                'select': 'id,display_name,issn_l,host_organization_name'}))
            for row in (rows or {}).get('results', []):
                remember_journal(row)
            for source_id in missing:
                if cache.get('search-journal:' + source_id) is None:
                    cache.set('search-journal:' + source_id, {'id': source_id, 'name': source_id}, 60)
        except sources.SourceUnavailable:
            pass  # Keep the selected IDs through an index outage.
    return [cache.get('search-journal:' + v) or {'id': v, 'name': v} for v in ids]


@require_GET
def journal_suggestions(request):
    with sources.searching_as(request):
        return _journal_suggestions(request)


def _journal_suggestions(request):
    query = (request.GET.get('q') or '').strip()[:100]
    if len(query) < 2:
        return JsonResponse({'results': []})
    key = 'journal-suggestions:' + hashlib.sha256(query.casefold().encode()).hexdigest()
    rows = cache.get(key)
    if rows is None:
        if not sources.search_allowed():
            return JsonResponse({'results': [], 'unavailable': True}, status=503)
        try:
            response = sources._json(sources.OPENALEX + '/sources', sources._openalex_params({
                'search': query, 'filter': 'type:journal', 'per_page': 6,
                'select': 'id,display_name,issn_l,host_organization_name'}))
            rows = [remember_journal(r) for r in (response or {}).get('results', [])]
            rows = [r for r in rows if r]
            cache.set(key, rows, 900)
        except sources.SourceUnavailable:
            return JsonResponse({'results': [], 'unavailable': True}, status=503)
    return JsonResponse({'results': rows})


def filter_chips(state, selected):
    """Each removal keeps the other validated filters and starts a fresh result window."""
    from django.utils.translation import gettext as _
    chips = []
    labels = {'year_from': _('From year'), 'year_to': _('To year'), 'orcid': _('Author ORCID'),
              'sort': _('Sort by'), 'type': _('Publication type'), 'src': _('Sources'), 'field': _('Search by')}
    values = {'newest': _('Newest first'), 'oldest': _('Oldest first'), 'journals': _('Articles'),
              'preprints': _('Preprints'), 'reviews': _('Literature reviews'), 'author': _('Author'),
              'title': _('Title'), 'abstract': _('Abstract'), 'open': _('Journals and preprints (open access)'),
              'arxiv': 'arXiv'}
    for key in ('field', 'year_from', 'year_to', 'sort', 'type', 'src', 'orcid'):
        value = state.get(key)
        if not value or value in ('all', 'relevance'):
            continue
        copy = state.copy()
        copy.pop(key, None)
        chips.append({'label': _('%(label)s: %(value)s') % {'label': labels[key], 'value': values.get(value, value)},
                      'url': '?' + copy.urlencode()})
    for key, label in (('journals', _('Include')), ('exclude_journals', _('Exclude'))):
        for item in selected[key]:
            copy = state.copy()
            copy.setlist(key, [v for v in state.getlist(key) if v != item['id']])
            chips.append({'label': _('%(label)s: %(value)s') % {'label': label, 'value': item['name']}, 'url': '?' + copy.urlencode()})
    return chips
