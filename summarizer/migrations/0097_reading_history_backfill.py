from django.db import migrations
from django.db.models import Max


def backfill(apps, schema_editor):
    """Visits get the id of their paper; the summaries a user requested become visits too, so that the history
    reads one table."""
    PaperVisit = apps.get_model('summarizer', 'PaperVisit')
    PaperHistory = apps.get_model('summarizer', 'PaperHistory')
    ArxivPaper = apps.get_model('summarizer', 'ArxivPaper')
    for visit in PaperVisit.objects.select_related('paper').iterator():
        PaperVisit.objects.filter(pk=visit.pk).update(arxiv_id=visit.paper.arxiv_id)
    known = {(u, a): pk for pk, u, a in PaperVisit.objects.values_list('pk', 'user_id', 'arxiv_id')}
    requests = PaperHistory.objects.filter(user__isnull=False).values('user_id', 'arxiv_id').annotate(last=Max('created'))
    new = []
    for row in requests.iterator():
        pk = known.get((row['user_id'], row['arxiv_id']))
        if pk:
            PaperVisit.objects.filter(pk=pk).update(requested=row['last'])
        else:
            new.append(row)
    papers = dict(ArxivPaper.objects.filter(arxiv_id__in={r['arxiv_id'] for r in new}).values_list('arxiv_id', 'pk'))
    PaperVisit.objects.bulk_create([PaperVisit(user_id=r['user_id'], arxiv_id=r['arxiv_id'], paper_id=papers.get(r['arxiv_id']),
                                               visited=None, requested=r['last']) for r in new], batch_size=500)


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0096_reading_history'),
    ]

    operations = [
        migrations.RunPython(backfill, migrations.RunPython.noop),
    ]
