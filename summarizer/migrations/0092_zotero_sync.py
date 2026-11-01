from django.db import migrations, models


def encrypt_keys(apps, schema_editor):
    from summarizer.models import zotero_cipher
    cipher = zotero_cipher()
    for account in apps.get_model('summarizer', 'ZoteroAccount').objects.all():
        account.secret = cipher.encrypt(account.api_key.encode()).decode()
        account.save(update_fields=['secret'])


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0091_reference_watch'),
    ]

    operations = [
        migrations.AddField(model_name='zoteroaccount', name='secret', field=models.TextField(blank=True, default='')),
        migrations.AddField(model_name='zoteroaccount', name='username', field=models.CharField(blank=True, default='', max_length=100)),
        migrations.AddField(model_name='zoteroaccount', name='library', field=models.CharField(blank=True, default='', max_length=40)),
        migrations.AddField(model_name='zoteroaccount', name='library_name', field=models.CharField(blank=True, default='', max_length=200)),
        migrations.AddField(model_name='zoteroaccount', name='inbox', field=models.CharField(blank=True, default='', max_length=20)),
        migrations.AddField(model_name='zoteroaccount', name='inbox_name', field=models.CharField(blank=True, default='', max_length=200)),
        migrations.RunPython(encrypt_keys, migrations.RunPython.noop),
        migrations.RemoveField(model_name='zoteroaccount', name='api_key'),
        migrations.AddField(model_name='collection', name='zotero_key', field=models.CharField(blank=True, default='', max_length=20)),
        migrations.AddField(model_name='collection', name='zotero_library', field=models.CharField(blank=True, default='', max_length=40)),
        migrations.AddField(model_name='collection', name='zotero_synced', field=models.DateTimeField(blank=True, null=True)),
        migrations.AddField(model_name='collection', name='zotero_removed', field=models.TextField(blank=True, default='')),
        migrations.AddField(model_name='collectionpaper', name='zotero_key', field=models.CharField(blank=True, default='', max_length=20)),
    ]
