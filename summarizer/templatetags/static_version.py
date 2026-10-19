"""{% static_v 'path' %}: the static URL with a hash of the file, so browsers and Cloudflare fetch a new
version after each deploy instead of serving the cached one (the static files keep their names)."""
import hashlib
from functools import lru_cache

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@lru_cache(maxsize=None)
def _digest(path):
    found = finders.find(path)
    if not found:
        return ''
    with open(found, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()[:10]


@register.simple_tag
def static_v(path):
    digest = _digest(path)
    return static(path) + ('?v=' + digest if digest else '')
