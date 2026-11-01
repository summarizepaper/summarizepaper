import hashlib

from django.db import migrations

DECK_KINDS = ('slides_v3', 'slides_v2', 'slides', 'slides_coll')


def backfill(apps, schema_editor):
    """The decks users edited (their private copies, keyed by a hash of the deck: see slides.edit_location) and
    the press kits they reported a mistake in become their uses."""
    Generated = apps.get_model('summarizer', 'Generated')
    ToolUse = apps.get_model('summarizer', 'ToolUse')
    ArxivPaper = apps.get_model('summarizer', 'ArxivPaper')
    PressKitReport = apps.get_model('summarizer', 'PressKitReport')
    Collection = apps.get_model('summarizer', 'Collection')
    edits = {}
    for user_id, key, lang in Generated.objects.filter(kind__in=('slides_edit_v3', 'slides_edit_v2', 'slides_edit'),
                                                       user__isnull=False).values_list('user_id', 'key', 'lang'):
        edits.setdefault(key.split(':', 1)[1], []).append((user_id, lang))
    if edits:
        for row in Generated.objects.filter(kind__in=DECK_KINDS).only('pk', 'kind', 'key', 'lang'):
            text = row.key if row.kind in ('slides_v2', 'slides') else '%s:%s' % (row.kind, row.key)  # legacy copies hash the paper id
            for user_id, lang in edits.get(hashlib.sha256(text.encode()).hexdigest(), []):
                if lang != row.lang:
                    continue
                if row.kind == 'slides_coll':
                    pk = row.key.split(':')[0]
                    title = Collection.objects.filter(pk=pk).values_list('name', flat=True).first() or ''
                    url = '/%s/journal-club/collection/%s/?deck_lang=%s' % (row.lang, pk, row.lang)
                else:
                    arxiv_id = row.key.split('|')[0]
                    title = ArxivPaper.objects.filter(arxiv_id=arxiv_id).values_list('title', flat=True).first() or arxiv_id
                    url = '/%s/journal-club/%s/?deck_lang=%s' % (row.lang, arxiv_id, row.lang)
                    if '|' in row.key:
                        url += '&audience=%s&duration=%s' % tuple(row.key.split('|')[1:3])
                ToolUse.objects.get_or_create(user_id=user_id, row_id=row.pk, defaults={'url': url, 'title': title[:500]})
    for user_id, arxiv_id, lang in PressKitReport.objects.filter(user__isnull=False).values_list('user_id', 'arxiv_id', 'lang').distinct():
        row = Generated.objects.filter(kind='presskit', key=arxiv_id, lang=lang).first()
        if row is not None:
            title = ArxivPaper.objects.filter(arxiv_id=arxiv_id).values_list('title', flat=True).first() or arxiv_id
            ToolUse.objects.get_or_create(user_id=user_id, row_id=row.pk,
                                          defaults={'url': '/%s/press-kit/%s/' % (lang, arxiv_id), 'title': title[:500]})


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0101_tool_use'),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
