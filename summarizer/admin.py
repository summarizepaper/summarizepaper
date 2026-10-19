import requests
from django import forms
from django.conf import settings
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from django.contrib.auth.models import User
from django.db import models
from django.db.models import Count, IntegerField, OuterRef, Q, Subquery
from django.db.models.functions import Coalesce
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.html import format_html

from .models import (AIassistant, ArxivPaper, Author, DigestSubscription, Guide, PaperAuthor, PaperHistory,
                     PaperScore, PaymentPlan, PDFHistory, PickledData, Search, SiteSettings, SummaryPaper,
                     SuppressedEmail, UserPlan, Vote)

admin.site.site_header = 'SummarizePaper admin'
admin.site.site_title = 'SummarizePaper admin'
admin.site.index_title = 'Administration'


# --- site settings: quotas and AI models ---------------------------------

def _check_model(name):
    """Error message when OpenAI does not know this model, None when it does or cannot be asked."""
    key = getattr(settings, 'OPENAI_KEY', None)
    if not key:
        return None
    try:
        response = requests.get('https://api.openai.com/v1/models/' + name,
                                headers={'Authorization': 'Bearer %s' % key}, timeout=10)
    except requests.RequestException:
        return None
    if response.status_code == 404:
        return 'OpenAI ne connaît pas le modèle « %s » avec notre clé API.' % name
    return None


class SiteSettingsForm(forms.ModelForm):
    class Meta:
        model = SiteSettings
        fields = '__all__'

    def clean(self):
        data = super().clean()
        for field in ('model_free', 'model_paid'):
            name = (data.get(field) or '').strip()
            data[field] = name
            if name and field in self.changed_data:
                error = _check_model(name)
                if error:
                    self.add_error(field, error)
        return data


@admin.register(SiteSettings)
class SiteSettingsAdmin(admin.ModelAdmin):
    form = SiteSettingsForm
    readonly_fields = ('updated',)
    fieldsets = (
        ('Résumés', {'fields': ('free_runs_per_day',)}),
        ('Questions à l’assistant', {'fields': ('chat_anonymous_per_day', 'chat_user_per_day', 'chat_paid_per_day')}),
        ('Modèles IA', {'fields': ('model_free', 'model_paid'),
                        'description': 'Les changements s’appliquent en moins de 30 secondes sur tous les serveurs.'}),
        (None, {'fields': ('updated',)}),
    )

    def has_add_permission(self, request):
        return False

    def has_delete_permission(self, request, obj=None):
        return False

    def changelist_view(self, request, extra_context=None):
        # a single row: go straight to its form
        return redirect(reverse('admin:summarizer_sitesettings_change', args=[SiteSettings.current().pk]))


# --- users ----------------------------------------------------------------

def _count_per_user(model):
    rows = model.objects.filter(user=OuterRef('pk')).order_by().values('user').annotate(n=Count('pk')).values('n')
    return Coalesce(Subquery(rows, output_field=IntegerField()), 0)


def _premium_q():
    return Q(plan_paid=True) & (Q(plan_expires__isnull=True) | Q(plan_expires__gte=timezone.now()))


class PremiumFilter(admin.SimpleListFilter):
    title = 'Premium'
    parameter_name = 'premium'

    def lookups(self, request, model_admin):
        return (('yes', 'Oui'), ('no', 'Non'))

    def queryset(self, request, queryset):
        if self.value() == 'yes':
            return queryset.filter(_premium_q())
        if self.value() == 'no':
            return queryset.exclude(_premium_q())
        return queryset


class UserPlanInline(admin.StackedInline):
    model = UserPlan
    extra = 0
    fields = ('payment_plan', 'paid', 'expires', 'canceling', 'stripe_customer_id', 'stripe_subscription_id', 'created')
    readonly_fields = ('created',)
    verbose_name = 'abonnement'
    verbose_name_plural = 'abonnements (le plus récent compte ; sans date de fin = Premium offert à vie)'


class DigestInline(admin.StackedInline):
    model = DigestSubscription
    extra = 0
    fields = ('categories', 'keywords', 'lang', 'send_hour', 'timezone', 'active', 'last_sent')
    readonly_fields = ('last_sent',)


admin.site.unregister(User)


@admin.register(User)
class SiteUserAdmin(UserAdmin):
    list_display = ('username', 'email', 'date_joined', 'last_login', 'is_active', 'premium', 'summaries', 'questions')
    list_filter = (PremiumFilter, 'is_active', 'is_staff', 'date_joined')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    ordering = ('-date_joined',)
    inlines = (UserPlanInline, DigestInline)

    def get_queryset(self, request):
        latest_plan = UserPlan.objects.filter(user=OuterRef('pk')).order_by('-created')
        return super().get_queryset(request).annotate(
            plan_paid=Subquery(latest_plan.values('paid')[:1]),
            plan_expires=Subquery(latest_plan.values('expires')[:1]),
            n_summaries=_count_per_user(PaperHistory),
            n_questions=_count_per_user(AIassistant),
        )

    @admin.display(boolean=True, description='Premium')
    def premium(self, user):
        return bool(user.plan_paid) and (user.plan_expires is None or user.plan_expires >= timezone.now())

    @admin.display(description='résumés', ordering='n_summaries')
    def summaries(self, user):
        return user.n_summaries

    @admin.display(description='questions', ordering='n_questions')
    def questions(self, user):
        return user.n_questions


# --- blog articles --------------------------------------------------------

class GuideForm(forms.ModelForm):
    class Meta:
        model = Guide
        fields = '__all__'
        widgets = {
            'body': forms.Textarea(attrs={'rows': 40, 'style': 'width: 100%; font-family: monospace'}),
            'extra': forms.Textarea(attrs={'rows': 10, 'style': 'width: 100%; font-family: monospace'}),
            'description': forms.TextInput(attrs={'size': 120, 'maxlength': 300}),
            'title': forms.TextInput(attrs={'size': 120}),
        }


@admin.register(Guide)
class GuideAdmin(admin.ModelAdmin):
    form = GuideForm
    list_display = ('title', 'lang', 'slug', 'date', 'updated', 'draft', 'view_link')
    list_filter = ('lang', 'draft')
    search_fields = ('title', 'slug', 'description', 'body')
    prepopulated_fields = {'slug': ('title',)}
    actions = ('publish', 'unpublish')
    fieldsets = (
        (None, {'fields': ('lang', 'title', 'slug', 'description', 'draft', 'date', 'updated')}),
        ('Contenu', {'fields': ('body',)}),
        ('Avancé', {'classes': ('collapse',), 'fields': ('product', 'extra')}),
    )

    @admin.display(description='lien')
    def view_link(self, guide):
        return format_html('<a href="/{}/guides/{}/" target="_blank">voir</a>', guide.lang, guide.slug)

    @admin.action(description='Publier les articles sélectionnés')
    def publish(self, request, queryset):
        queryset.update(draft=False)

    @admin.action(description='Repasser en brouillon')
    def unpublish(self, request, queryset):
        queryset.update(draft=True)


# --- everything else ------------------------------------------------------

@admin.register(UserPlan)
class UserPlanAdmin(admin.ModelAdmin):
    list_display = ('user', 'payment_plan', 'paid', 'expires', 'canceling', 'created')
    list_filter = ('paid', 'canceling')
    search_fields = ('user__username', 'user__email', 'stripe_customer_id', 'stripe_subscription_id')
    raw_id_fields = ('user',)


class HistoryAdmin(admin.ModelAdmin):
    list_display = ('arxiv_id', 'user', 'lang', 'created')
    list_filter = ('lang',)
    search_fields = ('arxiv_id', 'user__username', 'user__email')
    raw_id_fields = ('user',)


admin.site.register(PaperHistory, HistoryAdmin)
admin.site.register(PDFHistory, HistoryAdmin)


@admin.register(AIassistant)
class AIassistantAdmin(admin.ModelAdmin):
    list_display = ('arxiv_id', 'user', 'query', 'lang', 'created')
    list_filter = ('lang',)
    search_fields = ('arxiv_id', 'query', 'user__username', 'user__email')
    raw_id_fields = ('user',)


@admin.register(SummaryPaper)
class SummaryPaperAdmin(admin.ModelAdmin):
    list_display = ('paper', 'lang', 'created', 'updated')
    list_filter = ('lang',)
    search_fields = ('paper__arxiv_id', 'paper__title')
    raw_id_fields = ('paper',)


@admin.register(ArxivPaper)
class ArxivPaperAdmin(admin.ModelAdmin):
    list_display = ('arxiv_id', 'title', 'category', 'published_arxiv', 'created')
    search_fields = ('arxiv_id', 'title')


@admin.register(DigestSubscription)
class DigestSubscriptionAdmin(admin.ModelAdmin):
    list_display = ('user', 'categories', 'lang', 'active', 'last_sent')
    list_filter = ('active', 'lang')
    search_fields = ('user__username', 'user__email', 'categories')
    raw_id_fields = ('user',)


@admin.register(SuppressedEmail)
class SuppressedEmailAdmin(admin.ModelAdmin):
    list_display = ('email', 'reason', 'created')
    search_fields = ('email',)


admin.site.register(Author)
admin.site.register(Vote)
admin.site.register(PaperAuthor)
admin.site.register(Search)
admin.site.register(PaperScore)
admin.site.register(PaymentPlan)


class PickledDataAdmin(admin.ModelAdmin):
    formfield_overrides = {
        models.BinaryField: {'widget': forms.Textarea(attrs=dict(readonly=True))},
    }


admin.site.register(PickledData, PickledDataAdmin)
