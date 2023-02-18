from django.contrib import admin
from .models import ArxivPaper, Author, Vote, PaperHistory

admin.site.register(ArxivPaper)
admin.site.register(Author)
admin.site.register(Vote)

class MyModelAdmin(admin.ModelAdmin):
    list_display = ('arxiv_id', 'user', 'created', 'updated')

admin.site.register(PaperHistory, MyModelAdmin)
