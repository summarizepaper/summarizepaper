from django.contrib import admin
from .models import ArxivPaper, Author, Vote, SummaryPaper, PaperHistory, PDFHistory, PaperScore, PaperAuthor, PickledData, AIassistant, Search
from django import forms
from django.db import models

admin.site.register(ArxivPaper)
admin.site.register(Author)
admin.site.register(Vote)
admin.site.register(PaperAuthor)
admin.site.register(SummaryPaper)
admin.site.register(AIassistant)
admin.site.register(Search)
admin.site.register(PaperScore)

class MyModelAdmin(admin.ModelAdmin):
    list_display = ('arxiv_id', 'user', 'created')

admin.site.register(PaperHistory, MyModelAdmin)
admin.site.register(PDFHistory, MyModelAdmin)

from django.utils.translation import gettext_lazy as _
from django.forms.widgets import ClearableFileInput


class MyModelAdmin2(admin.ModelAdmin):
    formfield_overrides = {
    models.BinaryField: { 'widget':forms.Textarea(attrs=dict(readonly=True))},
    }


admin.site.register(PickledData, MyModelAdmin2)
#admin.site.register(PickledData)
