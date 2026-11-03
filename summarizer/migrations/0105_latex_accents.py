import json

from django.db import migrations


def convert(apps, schema_editor):
    """R\\'enyi -> Rényi in the titles and abstracts stored before they were converted on arrival."""
    from summarizer.sources import latex_accents
    for model, fields in (('ArxivPaper', ('title', 'abstract')), ('AnnouncedPaper', ('title', 'abstract')),
                          ('DigestItem', ('title',))):
        Model = apps.get_model('summarizer', model)
        for field in fields:
            for row in Model.objects.filter(**{field + '__contains': '\\'}).only('pk', field).iterator():
                value = latex_accents(getattr(row, field))
                if value != getattr(row, field):
                    Model.objects.filter(pk=row.pk).update(**{field: value})
    Letter = apps.get_model('summarizer', 'Letter')
    for letter in Letter.objects.filter(content__contains='\\\\').only('pk', 'content').iterator():
        content = json.loads(letter.content)
        content['papers'] = {k: latex_accents(v) for k, v in (content.get('papers') or {}).items()}
        for papers in (content.get('earlier') or {}).values():
            for paper in papers:
                if isinstance(paper, dict) and paper.get('title'):
                    paper['title'] = latex_accents(paper['title'])
        Letter.objects.filter(pk=letter.pk).update(content=json.dumps(content))


class Migration(migrations.Migration):
    dependencies = [('summarizer', '0104_arxiv_fetch')]
    operations = [migrations.RunPython(convert, migrations.RunPython.noop)]
