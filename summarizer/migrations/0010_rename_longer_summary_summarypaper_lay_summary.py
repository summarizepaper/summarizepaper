# Generated by Django 4.1.6 on 2023-02-23 12:55

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0009_alter_pickleddata_buffer_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='summarypaper',
            old_name='longer_summary',
            new_name='lay_summary',
        ),
    ]
