# Generated by Django 4.1.6 on 2023-02-25 11:28

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('summarizer', '0011_alter_author_id_alter_paperauthor_id_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AIassistant',
            fields=[
                ('id', models.AutoField(primary_key=True, serialize=False)),
                ('arxiv_id', models.CharField(max_length=20)),
                ('query', models.TextField(blank=True, null=True)),
                ('response', models.TextField(blank=True, null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
