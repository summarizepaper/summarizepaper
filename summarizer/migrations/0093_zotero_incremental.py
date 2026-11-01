from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0092_zotero_sync'),
    ]

    operations = [
        migrations.AddField(model_name='collection', name='zotero_version', field=models.PositiveBigIntegerField(default=0)),
        migrations.AddField(model_name='collection', name='zotero_lang', field=models.CharField(blank=True, default='', max_length=10)),
        migrations.AddField(model_name='collectionpaper', name='changed', field=models.DateTimeField(blank=True, null=True)),
    ]
