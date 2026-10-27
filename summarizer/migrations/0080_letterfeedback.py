import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0079_letter_papers'),
    ]

    operations = [
        migrations.CreateModel(
            name='LetterFeedback',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('arxiv_id', models.CharField(blank=True, default='', max_length=100)),
                ('value', models.SmallIntegerField()),
                ('visitor', models.CharField(max_length=80)),
                ('created', models.DateTimeField(auto_now=True)),
                ('letter', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='feedback', to='summarizer.letter')),
            ],
            options={'unique_together': {('letter', 'arxiv_id', 'visitor')}},
        ),
    ]
