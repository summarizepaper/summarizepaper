import requests
from django import forms
from django.conf import settings
from django.contrib import admin, messages
from django.contrib.auth.admin import GroupAdmin as BaseGroupAdmin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from django.contrib.auth.models import Group, User
from django.db import models
from django.db.models import Count, Exists, IntegerField, OuterRef, Prefetch, Q, Subquery
from django.db.models.functions import Coalesce
from django.shortcuts import redirect
from django.urls import path, reverse
from django.utils import timezone
from django.utils.html import format_html
from django.views.decorators.http import require_POST
from unfold.admin import ModelAdmin, StackedInline
from unfold.forms import AdminPasswordChangeForm, UserChangeForm, UserCreationForm
from unfold.widgets import UnfoldAdminTextareaWidget

from . import guides, usage
from .models import (AIPrice, AIUsage, AIassistant, AnnouncedPaper, ArxivBatch, ArxivPaper, Author, CategorySynthesis, DigestFeedback,
                     DigestItem, DigestSubscription, Guide, PaperAuthor, PaperHistory,
                     PaperScore, PaymentPlan, PDFHistory, PickledData, Search, SiteSettings, SummaryPaper,
                     SuppressedEmail, UserPlan, PremiumGrant, Vote)

admin.site.index_title = 'Tableau de bord'


def dashboard_callback(request, context):
    """Numbers shown on the admin home page (UNFOLD["DASHBOARD_CALLBACK"])."""
    now = timezone.now()
    day, week = now - timezone.timedelta(days=1), now - timezone.timedelta(days=7)
    premium = User.objects.annotate(**_premium_annotations()).filter(_premium_q()).count()
    cfg = SiteSettings.current()
    context['kpis'] = [
        ('Utilisateurs', User.objects.count(), '+%d cette semaine' % User.objects.filter(date_joined__gte=week).count(),
         reverse('admin:auth_user_changelist')),
        ('Premium', premium, 'abonnements ou accès complets offerts', reverse('admin:auth_user_changelist') + '?premium=yes'),
        ('Résumés (24 h)', PaperHistory.objects.filter(created__gte=day).count(),
         '%d sur 7 jours' % PaperHistory.objects.filter(created__gte=week).count(),
         reverse('admin:summarizer_paperhistory_changelist')),
        ('Questions (24 h)', AIassistant.objects.filter(created__gte=day).count(),
         '%d sur 7 jours' % AIassistant.objects.filter(created__gte=week).count(),
         reverse('admin:summarizer_aiassistant_changelist')),
    ]
    week_cost = usage.summary(7)
    context['kpis'].append(('Coût IA (7 j)', '$%.2f' % week_cost['total'],
                            'modèles sans prix : %s' % ', '.join(week_cost['unpriced']) if week_cost['unpriced']
                            else 'résumés, assistant, outils, digests', reverse('admin:summarizer_aiusage_changelist')))
    context['site_settings'] = cfg
    context['recent_users'] = User.objects.order_by('-date_joined')[:8]
    return context


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
        for field in ('model_free', 'model_paid', 'model_tts'):
            name = (data.get(field) or '').strip()
            data[field] = name
            if name and field in self.changed_data:
                error = _check_model(name)
                if error:
                    self.add_error(field, error)
        return data


@admin.register(SiteSettings)
class SiteSettingsAdmin(ModelAdmin):
    form = SiteSettingsForm
    readonly_fields = ('last_change',)
    fieldsets = (
        ('Résumés', {'fields': ('free_runs_per_day', 'free_readings_per_day')}),
        ('Questions à l’assistant', {'fields': ('chat_anonymous_per_day', 'chat_user_per_day', 'chat_paid_per_day')}),
        ('Modèles IA', {'fields': ('model_free', 'model_paid', 'model_tts', 'tts_voice'),
                        'description': 'Les changements s’appliquent en moins de 30 secondes sur tous les serveurs.'}),
        (None, {'fields': ('last_change',)}),
    )

    @admin.display(description='dernière modification')
    def last_change(self, cfg):
        return cfg.updated

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


def _current_grants():
    return PremiumGrant.objects.filter(active=True, user__is_active=True).filter(Q(expires__isnull=True) | Q(expires__gte=timezone.now()))


def _premium_annotations():
    plan = UserPlan.objects.filter(user=OuterRef('pk')).order_by('-created', '-pk')
    return {'plan_paid': Subquery(plan.values('paid')[:1]),
            'plan_expires': Subquery(plan.values('expires')[:1]),
            'gift_premium': Exists(_current_grants().filter(user=OuterRef('pk'), tool__in=('all', 'plus')))}


def _premium_q():
    return Q(is_active=True) & (Q(gift_premium=True) | (Q(plan_paid=True) & (Q(plan_expires__isnull=True) | Q(plan_expires__gte=timezone.now()))))


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


class UserPlanInline(StackedInline):
    model = UserPlan
    extra = 0
    fields = ('payment_plan', 'paid', 'tier', 'expires', 'canceling', 'stripe_customer_id', 'stripe_subscription_id', 'created')
    readonly_fields = ('created',)
    verbose_name = 'abonnement'
    verbose_name_plural = 'abonnements (le plus récent compte ; sans date de fin = Premium offert à vie)'


class PremiumGrantForm(forms.ModelForm):
    class Meta:
        model = PremiumGrant
        fields = '__all__'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from .premium_tools import TOOL_CHOICES
        # A newly added inline must not silently default to full Premium.
        self.fields['tool'].choices = [('', 'Choisissez un accès')] + list(TOOL_CHOICES)


class PremiumGrantInline(StackedInline):
    model = PremiumGrant
    form = PremiumGrantForm
    fk_name = 'user'
    extra = 0
    fields = ('tool', 'active', 'expires', 'note', 'granted_by', 'created')
    readonly_fields = ('granted_by', 'created')
    verbose_name = 'accès Premium offert'
    verbose_name_plural = 'accès offerts : Premium complet ou outil au choix (indépendants de Stripe)'


class GrantedToolFilter(admin.SimpleListFilter):
    title = 'Accès offert'
    parameter_name = 'granted_tool'

    def lookups(self, request, model_admin):
        from .premium_tools import TOOL_CHOICES
        return TOOL_CHOICES

    def queryset(self, request, queryset):
        from .premium_tools import TOOL_KEYS
        if self.value() in TOOL_KEYS:
            return queryset.filter(pk__in=_current_grants().filter(tool=self.value()).values('user_id'))
        return queryset


class DigestInline(StackedInline):
    model = DigestSubscription
    extra = 0
    fields = ('name', 'categories', 'keywords', 'profile', 'reference_papers', 'authors', 'lang', 'frequency', 'weekday',
              'send_hour', 'timezone', 'active', 'last_sent')
    readonly_fields = ('last_sent',)


admin.site.unregister(User)
admin.site.unregister(Group)


@admin.register(Group)
class GroupAdmin(BaseGroupAdmin, ModelAdmin):
    pass


@admin.register(User)
class SiteUserAdmin(BaseUserAdmin, ModelAdmin):
    form = UserChangeForm
    add_form = UserCreationForm
    change_password_form = AdminPasswordChangeForm
    list_display = ('username', 'email', 'date_joined', 'last_login', 'is_active', 'premium', 'offered_tools', 'summaries', 'questions')
    list_filter = (PremiumFilter, GrantedToolFilter, 'is_active', 'is_staff', 'date_joined')
    search_fields = ('username', 'email', 'first_name', 'last_name')
    ordering = ('-date_joined',)
    inlines = (PremiumGrantInline, UserPlanInline, DigestInline)

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            **_premium_annotations(), n_summaries=_count_per_user(PaperHistory),
            n_questions=_count_per_user(AIassistant),
        ).prefetch_related(Prefetch('premium_grants', queryset=_current_grants(), to_attr='current_grants'))

    @admin.display(boolean=True, description='Premium complet')
    def premium(self, user):
        return user.is_active and (bool(user.gift_premium) or (bool(user.plan_paid) and
                (user.plan_expires is None or user.plan_expires >= timezone.now())))

    @admin.display(description='Accès offerts')
    def offered_tools(self, user):
        return ', '.join(str(g.get_tool_display()) for g in user.current_grants) or '—'

    def save_formset(self, request, form, formset, change):
        if formset.model is PremiumGrant:
            instances = formset.save(commit=False)
            for obj in formset.deleted_objects:
                obj.delete()
            for obj in instances:
                if not obj.pk:
                    obj.granted_by = request.user
                obj.save()
            formset.save_m2m()
        else:
            super().save_formset(request, form, formset, change)

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
            'body': UnfoldAdminTextareaWidget(attrs={'rows': 40, 'style': 'font-family: ui-monospace, monospace'}),
            'extra': UnfoldAdminTextareaWidget(attrs={'rows': 10, 'style': 'font-family: ui-monospace, monospace'}),
        }


@admin.register(Guide)
class GuideAdmin(ModelAdmin):
    form = GuideForm
    list_after_template = 'admin/summarizer/guide/repository_guides.html'
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

    def get_urls(self):
        view = self.admin_site.admin_view(require_POST(self.import_view))
        return [path('import/<str:lang>/<slug:slug>/', view, name='summarizer_guide_import')] + super().get_urls()

    def import_view(self, request, lang, slug):
        if (lang, slug) not in guides.repository_guides():
            messages.error(request, 'Aucun fichier content/guides/%s/%s.md.' % (lang, slug))
            return redirect('admin:summarizer_guide_changelist')
        guide = guides.import_file(lang, slug)
        messages.success(request, '« %s » est maintenant géré ici : le fichier du dépôt est ignoré pour cet article.'
                         % guide.title)
        return redirect('admin:summarizer_guide_change', guide.pk)

    def changelist_view(self, request, extra_context=None):
        in_db = set(Guide.objects.values_list('lang', 'slug'))
        repo = []
        for lang, slug in guides.repository_guides():
            if (lang, slug) not in in_db:
                guide = guides.load(lang, slug)
                if guide:
                    repo.append(guide)
        extra_context = dict(extra_context or {}, repository_guides=repo)
        return super().changelist_view(request, extra_context)

    @admin.action(description='Publier les articles sélectionnés')
    def publish(self, request, queryset):
        queryset.update(draft=False)

    @admin.action(description='Repasser en brouillon')
    def unpublish(self, request, queryset):
        queryset.update(draft=True)


# --- everything else ------------------------------------------------------

@admin.register(UserPlan)
class UserPlanAdmin(ModelAdmin):
    list_display = ('user', 'payment_plan', 'paid', 'tier', 'expires', 'canceling', 'created')
    list_filter = ('paid', 'tier', 'canceling')
    search_fields = ('user__username', 'user__email', 'stripe_customer_id', 'stripe_subscription_id')
    raw_id_fields = ('user',)


@admin.register(PremiumGrant)
class PremiumGrantAdmin(ModelAdmin):
    form = PremiumGrantForm
    list_display = ('user', 'tool', 'active', 'current', 'expires', 'granted_by', 'created')
    list_filter = ('tool', 'active', 'expires')
    search_fields = ('user__username', 'user__email', 'note')
    autocomplete_fields = ('user',)
    list_select_related = ('user', 'granted_by')
    readonly_fields = ('granted_by', 'created', 'updated')
    fieldsets = (
        ('Accès offert', {'fields': ('user', 'tool', 'active', 'expires'),
                         'description': 'Accordez tous les avantages Premium ou seulement l’outil choisi. '
                                        'Ces accès ne créent ni paiement ni abonnement Stripe. '
                                        'Révoquer un accès offert ne résilie pas un abonnement payant.'}),
        ('Suivi interne', {'fields': ('note', 'granted_by', 'created', 'updated')}),
    )

    @admin.display(boolean=True, description='Valable maintenant')
    def current(self, obj):
        return obj.active and obj.user.is_active and (obj.expires is None or obj.expires >= timezone.now())

    def save_model(self, request, obj, form, change):
        if not change:
            obj.granted_by = request.user
        super().save_model(request, obj, form, change)


class HistoryAdmin(ModelAdmin):
    list_display = ('arxiv_id', 'user', 'lang', 'created')
    list_filter = ('lang',)
    search_fields = ('arxiv_id', 'user__username', 'user__email')
    raw_id_fields = ('user',)


admin.site.register(PaperHistory, HistoryAdmin)
admin.site.register(PDFHistory, HistoryAdmin)


@admin.register(AIassistant)
class AIassistantAdmin(ModelAdmin):
    list_display = ('arxiv_id', 'user', 'query', 'lang', 'created')
    list_filter = ('lang',)
    search_fields = ('arxiv_id', 'query', 'user__username', 'user__email')
    raw_id_fields = ('user',)


@admin.register(SummaryPaper)
class SummaryPaperAdmin(ModelAdmin):
    list_display = ('paper', 'lang', 'created', 'updated')
    list_filter = ('lang',)
    search_fields = ('paper__arxiv_id', 'paper__title')
    raw_id_fields = ('paper',)


@admin.register(ArxivPaper)
class ArxivPaperAdmin(ModelAdmin):
    list_display = ('arxiv_id', 'title', 'category', 'published_arxiv', 'created')
    search_fields = ('arxiv_id', 'title')


@admin.register(DigestSubscription)
class DigestSubscriptionAdmin(ModelAdmin):
    list_display = ('user', 'name', 'categories', 'frequency', 'lang', 'active', 'last_sent')
    list_filter = ('active', 'frequency', 'lang')
    search_fields = ('user__username', 'user__email', 'categories', 'name')
    raw_id_fields = ('user',)
    exclude = ('profile_vector', 'profile_key')


@admin.register(ArxivBatch)
class ArxivBatchAdmin(ModelAdmin):
    list_display = ('day', 'papers', 'missing', 'fetched')
    ordering = ('-day',)


@admin.register(AnnouncedPaper)
class AnnouncedPaperAdmin(ModelAdmin):
    list_display = ('arxiv_id', 'title', 'announced', 'categories')
    list_filter = ('announced',)
    search_fields = ('arxiv_id', 'title', 'authors')


@admin.register(DigestFeedback)
class DigestFeedbackAdmin(ModelAdmin):
    list_display = ('subscription', 'arxiv_id', 'value', 'created')
    list_filter = ('value',)
    search_fields = ('arxiv_id', 'subscription__user__username')
    exclude = ('vector',)


@admin.register(DigestItem)
class DigestItemAdmin(ModelAdmin):
    list_display = ('arxiv_id', 'lang', 'title', 'created')
    list_filter = ('lang',)
    search_fields = ('arxiv_id', 'title')


@admin.register(CategorySynthesis)
class CategorySynthesisAdmin(ModelAdmin):
    list_display = ('category', 'lang', 'day', 'papers', 'created')
    list_filter = ('lang', 'day')
    search_fields = ('category',)


@admin.register(SuppressedEmail)
class SuppressedEmailAdmin(ModelAdmin):
    list_display = ('email', 'reason', 'created')
    search_fields = ('email',)


admin.site.register(Author, ModelAdmin)
admin.site.register(Vote, ModelAdmin)
admin.site.register(PaperAuthor, ModelAdmin)
admin.site.register(Search, ModelAdmin)
admin.site.register(PaperScore, ModelAdmin)
admin.site.register(PaymentPlan, ModelAdmin)


class PickledDataAdmin(ModelAdmin):
    formfield_overrides = {
        models.BinaryField: {'widget': forms.Textarea(attrs=dict(readonly=True))},
    }


admin.site.register(PickledData, PickledDataAdmin)


# --- AI usage and prices (usage.py) ----------------------------------------

@admin.register(AIPrice)
class AIPriceAdmin(ModelAdmin):
    list_display = ('model', 'input_per_million', 'output_per_million')
    list_editable = ('input_per_million', 'output_per_million')


@admin.register(AIUsage)
class AIUsageAdmin(ModelAdmin):
    list_display = ('day', 'task', 'model', 'user', 'calls', 'prompt_tokens', 'completion_tokens', 'dollars')
    list_filter = ('task', 'model')
    date_hierarchy = 'day'
    search_fields = ('user__username', 'user__email')
    list_before_template = 'admin/summarizer/aiusage_summary.html'
    readonly_fields = [f.name for f in AIUsage._meta.fields]

    def has_add_permission(self, request):
        return False

    @admin.display(description='coût ($)')
    def dollars(self, obj):
        cost = obj.cost()
        return '—' if cost is None else '%.4f' % cost

    def changelist_view(self, request, extra_context=None):
        extra_context = dict(extra_context or {}, usage=usage.summary(30))
        return super().changelist_view(request, extra_context)
