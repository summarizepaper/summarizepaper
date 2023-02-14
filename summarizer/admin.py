from django.contrib import admin
from .models import ArxivPaper, Author, Vote

admin.site.register(ArxivPaper)
admin.site.register(Author)
admin.site.register(Vote)
