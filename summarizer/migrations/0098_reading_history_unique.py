from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('summarizer', '0097_reading_history_backfill'),
    ]

    operations = [
        migrations.AddConstraint(
            model_name='papervisit',
            constraint=models.UniqueConstraint(fields=('user', 'arxiv_id'), name='unique_user_arxiv_visit'),
        ),
    ]
