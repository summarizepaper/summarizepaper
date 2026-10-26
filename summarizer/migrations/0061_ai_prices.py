"""Prices of the models the site uses, from https://developers.openai.com/api/docs/pricing (standard tier,
October 2026). Rows already set in the admin are left as they are."""
from django.db import migrations

PRICES = [
    # model, $ per million input tokens, $ per million output tokens
    ('gpt-6-luna', '0.10', '0.50'),
    ('gpt-6.1-sol', '2.00', '10.00'),
    ('text-embedding-3-small', '0.02', '0'),
    # counted in characters of text (audio.py): about $0.015 a minute of speech, ~1,000 characters a minute
    ('gpt-4o-mini-tts', '15.00', '0'),
]


def add_prices(apps, schema_editor):
    AIPrice = apps.get_model('summarizer', 'AIPrice')
    for model, given, returned in PRICES:
        AIPrice.objects.get_or_create(model=model, defaults={'input_per_million': given, 'output_per_million': returned})


class Migration(migrations.Migration):
    dependencies = [('summarizer', '0060_userplan_tier_premium_plus')]
    operations = [migrations.RunPython(add_prices, migrations.RunPython.noop)]
