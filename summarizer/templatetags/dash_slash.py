from django import template
register = template.Library()

@register.filter
def dash_slash(value):
    return value.replace("--","/")
