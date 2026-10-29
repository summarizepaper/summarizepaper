"""Validity information for all expiring share links."""
from django import template
from summarizer.result_share import timing

register = template.Library()
register.simple_tag(timing, name='share_timing')
