# Generated by Django 4.1.6 on 2023-02-25 21:40

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('summarizer', '0013_aiassistant_active_aiassistant_lang'),
    ]

    operations = [
        migrations.AlterField(
            model_name='aiassistant',
            name='user',
            field=models.ForeignKey(blank=True, null=True, on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL),
        ),
    ]
