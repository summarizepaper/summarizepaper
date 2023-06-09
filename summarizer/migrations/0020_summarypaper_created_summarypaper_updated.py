# Generated by Django 4.1.6 on 2023-03-20 19:42

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0019_remove_paperhistory_updated'),
    ]

    operations = [
        migrations.AddField(
            model_name='summarypaper',
            name='created',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='summarypaper',
            name='updated',
            field=models.DateTimeField(auto_now=True),
        ),
    ]
