from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
        ('summarizer', '0046_other_sources'),
    ]

    operations = [
        migrations.CreateModel(
            name='WatchedProject',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(max_length=300)),
                ('description', models.TextField(blank=True, default='')),
                ('author_name', models.CharField(blank=True, default='', max_length=200)),
                ('lang', models.CharField(default='en', max_length=10)),
                ('vector', models.BinaryField(blank=True, null=True)),
                ('vector_key', models.CharField(blank=True, default='', max_length=64)),
                ('last_day', models.DateField(blank=True, null=True)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='watched_projects', to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ('-created',),
            },
        ),
        migrations.CreateModel(
            name='ScoopMatch',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('arxiv_id', models.CharField(max_length=100)),
                ('title', models.TextField()),
                ('authors', models.TextField(blank=True, default='')),
                ('announced', models.DateField()),
                ('similarity', models.FloatField()),
                ('overlap', models.CharField(max_length=10)),
                ('shared', models.TextField(blank=True, default='')),
                ('original', models.TextField(blank=True, default='')),
                ('emailed', models.BooleanField(default=False)),
                ('created', models.DateTimeField(auto_now_add=True)),
                ('project', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='matches', to='summarizer.watchedproject')),
            ],
            options={
                'ordering': ('-announced', '-similarity'),
                'unique_together': {('project', 'arxiv_id')},
            },
        ),
    ]
