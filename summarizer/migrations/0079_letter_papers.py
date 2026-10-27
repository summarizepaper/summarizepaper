import json
import re

from django.db import migrations, models


def fill_papers(apps, schema_editor):
    Letter = apps.get_model('summarizer', 'Letter')
    for letter in Letter.objects.all():
        ids = list(json.loads(letter.content).get('papers', {}))
        letter.papers = ',%s,' % ','.join(re.sub(r'v\d+$', '', a) for a in ids) if ids else ''
        letter.save(update_fields=['papers'])


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0078_trends_weeks'),
    ]

    operations = [
        migrations.AddField(model_name='letter', name='papers', field=models.TextField(blank=True, default='')),
        migrations.AddField(model_name='letter', name='vectors', field=models.BinaryField(blank=True, null=True)),
        migrations.RunPython(fill_papers, migrations.RunPython.noop),
    ]
