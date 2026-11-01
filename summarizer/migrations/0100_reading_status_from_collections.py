import re

from django.db import migrations

ARXIV_VERSION = re.compile(r'^(\d{4}\.\d{4,5}|[a-z-]+--\d{7})v\d+$')


def base(arxiv_id):
    match = ARXIV_VERSION.match(arxiv_id)
    return match.group(1) if match else arxiv_id


def backfill(apps, schema_editor):
    """The reading status a user gave a paper in their own collections becomes their status of the paper."""
    PaperVisit = apps.get_model('summarizer', 'PaperVisit')
    CollectionPaper = apps.get_model('summarizer', 'CollectionPaper')
    ArxivPaper = apps.get_model('summarizer', 'ArxivPaper')
    visits = {(u, base(a)): pk for pk, u, a in PaperVisit.objects.values_list('pk', 'user_id', 'arxiv_id')}
    new = {}
    rows = (CollectionPaper.objects.exclude(reading='').order_by('changed', 'added')
            .values_list('collection__user_id', 'arxiv_id', 'title', 'reading', 'changed', 'added'))
    for user_id, arxiv_id, title, reading, changed, added in rows:  # the latest change wins
        found = visits.get((user_id, base(arxiv_id)))
        if found:
            PaperVisit.objects.filter(pk=found).update(reading=reading)
        else:
            new[(user_id, base(arxiv_id))] = (arxiv_id, title, reading, changed or added)
    papers = dict(ArxivPaper.objects.filter(arxiv_id__in={v[0] for v in new.values()}).values_list('arxiv_id', 'pk'))
    PaperVisit.objects.bulk_create([
        PaperVisit(user_id=u, arxiv_id=a, paper_id=papers.get(a), title=(t or '')[:500], reading=r, marked=d, visited=None)
        for (u, _b), (a, t, r, d) in new.items()], batch_size=500)


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0099_reading_notes'),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
