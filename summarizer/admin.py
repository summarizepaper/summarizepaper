from django.contrib import admin
from .models import ArxivPaper, Author, Vote, SummaryPaper, PaperHistory, PaperAuthor, PickledData
from django import forms
from django.db import models

admin.site.register(ArxivPaper)
admin.site.register(Author)
admin.site.register(Vote)
admin.site.register(PaperAuthor)
admin.site.register(SummaryPaper)

class MyModelAdmin(admin.ModelAdmin):
    list_display = ('arxiv_id', 'user', 'created', 'updated')

admin.site.register(PaperHistory, MyModelAdmin)

from django.utils.translation import gettext_lazy as _
from django.forms.widgets import ClearableFileInput


class MyModelAdmin2(admin.ModelAdmin):
    formfield_overrides = {
    models.BinaryField: { 'widget':forms.Textarea(attrs=dict(readonly=True))},
    }


admin.site.register(PickledData, MyModelAdmin2)
#admin.site.register(PickledData)
