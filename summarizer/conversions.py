"""The staff page /admin/conversions/: from a visit to a paid subscription, over the last days, from what the
site keeps (no analytics service): the pages people opened (visits.py, robots and the team left out as on the
activity page), the accounts, the free tries of the tools (Generated.trial), the emails that follow them
(ConversionEmail) and the subscriptions (UserPlan, its Stripe trial in trial_end).

Each sign-up is followed back through the pages its browser opened the same day before the account existed
(PageView.visitor changes every day): which tool pages and examples it saw, where it came from. The clicks on
the examples themselves (teaser_* events) are counted in Umami.
"""
import datetime

from django.contrib.auth.models import User
from django.db.models import Count
from django.utils import timezone

from . import activity, teasers
from .models import ConversionEmail, Generated, PageView, UserPlan

PERIODS = (7, 30, 60)  # PageView rows are kept visits.VISIT_DAYS (60) days
# The page of each tool with an example, by url name (summarizer/urls.py), in the order of teasers.SHOWCASE_TOOLS
TOOL_VIEWS = (('teams', 'teams', 'Qui travaille dessus ?'), ('missed', 'missed', 'Qu’ai-je manqué ?'),
              ('refcheck', 'refcheck', 'Vérification de bibliographie'), ('literature', 'literature', 'Revue de littérature'),
              ('journal_club', 'slides_home', 'Journal club'), ('collections', 'collections', 'Collections et tableaux'),
              ('scoops', 'scoops', 'Veille anti-scoop'), ('citations', 'citations', 'Alertes de citations'))
EXAMPLE_VIEWS = ('examples', 'example')
KIND_LABELS = {teasers.KIND_OF.get(tool, tool): label for tool, _view, label in TOOL_VIEWS}


def _visitors(views, **filters):
    return views.filter(**filters).values('visitor').distinct().count()


def _paid_plans(start, end):
    """Stripe subscriptions started in the period, the team's left out."""
    return (UserPlan.objects.filter(created__gte=start, created__lt=end, user__is_staff=False)
            .exclude(stripe_subscription_id='').select_related('user'))


def _state(plan, now):
    if plan.trial_end and plan.trial_end > now:
        return 'essai en cours' + (' (annulé)' if plan.canceling else '')
    if plan.paid:
        return 'payant' + (' (annulé, fin de période)' if plan.canceling else '')
    return 'terminé' if plan.trial_end else 'arrêté'


def journeys(start, end, now):
    """The sign-ups of the period, newest first, each with the pages seen before it and what followed."""
    users = list(User.objects.filter(date_joined__gte=start, date_joined__lt=end, is_staff=False).order_by('-date_joined'))
    names = {view: label for _tool, view, label in TOOL_VIEWS}
    names.update(premium='Premium', examples='Exemples', example='Exemple', register='Inscription')
    tries = {}
    for row in Generated.objects.filter(user__in=users, trial=True).order_by('created'):
        tries.setdefault(row.user_id, []).append(KIND_LABELS.get(row.kind, row.kind))
    plans = {p.user_id: p for p in UserPlan.objects.filter(user__in=users).exclude(stripe_subscription_id='')}
    found = []
    for user in users:
        joined = user.date_joined
        day = [joined.date(), (joined - datetime.timedelta(hours=1)).date()]
        browsers = set(PageView.objects.filter(user=user, created__date__in=day).values_list('visitor', flat=True))
        before = list(PageView.objects.filter(visitor__in=browsers, created__lte=joined + datetime.timedelta(minutes=2),
                                              created__gte=joined - datetime.timedelta(hours=12)).order_by('created'))
        first = before[0] if before else None
        seen = list(dict.fromkeys(names[v.view] for v in before if v.view in names and v.view != 'register'))
        plan = plans.get(user.pk)
        found.append({'user': user, 'active': user.is_active, 'first': first.path if first else '',
                      'origin': (first.source or first.referrer) if first else '', 'seen': seen,
                      'papers': sum(1 for v in before if v.view == 'arxividpage'), 'tries': tries.get(user.pk, []),
                      'plan': _state(plan, now) if plan else ''})
    return found


def report(days, now=None):
    now = now or timezone.now()
    start = now - datetime.timedelta(days=days)
    views = activity._visits(start, now)
    anonymous = views.filter(user__isnull=True)
    signups = User.objects.filter(date_joined__gte=start, date_joined__lt=now, is_staff=False)
    active = signups.filter(is_active=True)
    tries = Generated.objects.filter(trial=True, created__gte=start, created__lt=now, user__is_staff=False)
    plans = list(_paid_plans(start, now))
    trials = [p for p in plans if p.trial_end]
    ended = [p for p in trials if p.trial_end <= now]
    converted = [p for p in ended if p.paid]
    followups = ConversionEmail.objects.filter(sent__gte=start, sent__lt=now)
    followed = set(followups.filter(kind='trial_result').values_list('user_id', flat=True))
    funnel = [
        ('Visiteurs (personnes, hors équipe)', _visitors(views), ''),
        ('… sans compte, sur une page d’outil Premium', _visitors(anonymous, view__in=[v for _t, v, _l in TOOL_VIEWS]), 'voient l’exemple de l’outil'),
        ('… sur une page d’exemple publique', _visitors(views, view__in=EXAMPLE_VIEWS), '/examples/'),
        ('… sur la page Premium', _visitors(views, view='premium'), ''),
        ('… sur la page d’inscription', _visitors(anonymous, view='register'), ''),
        ('Inscriptions', signups.count(), '%d activées' % active.count()),
        ('E-mails de bienvenue (essais gratuits)', followups.filter(kind='welcome').count(), 'le lendemain de l’inscription'),
        ('Annonce des essais aux anciens membres', followups.filter(kind='announce').count(), 'commande announce_trials'),
        ('Essais gratuits d’un outil', tries.count(), '%d comptes' % tries.values('user').distinct().count()),
        ('E-mails après l’essai gratuit', followups.filter(kind='trial_result').count(), ''),
        ('Abonnements Stripe commencés', len(plans), '%d avec essai de 7 jours' % len(trials)),
        ('E-mails avant la fin de l’essai', followups.filter(kind='trial_ending').count(), ''),
        ('Essais terminés → payants', len(converted), 'sur %d essais terminés' % len(ended)),
        ('Abonnés après un e-mail d’essai gratuit', UserPlan.objects.filter(user_id__in=followed, paid=True).exclude(stripe_subscription_id='').count(), ''),
    ]
    tools = []
    for tool, view, label in TOOL_VIEWS:
        kind = teasers.KIND_OF.get(tool, tool)
        tried = tries.filter(kind=kind)
        triers = set(tried.values_list('user_id', flat=True))
        tools.append({'label': label, 'visitors': _visitors(anonymous, view=view), 'members': _visitors(views.filter(user__isnull=False), view=view),
                      'examples': _visitors(views, view='example', path__contains='/examples/%s' % _prefix(kind)),
                      'tries': tried.count(), 'paid': UserPlan.objects.filter(user_id__in=triers, paid=True).exclude(stripe_subscription_id='').count()})
    found = journeys(start, now, now)
    return {'days': days, 'start': start, 'funnel': funnel, 'tools': tools, 'journeys': found, 'sources': by_source(views, found),
            'trials': [{'user': p.user, 'created': p.created, 'end': p.trial_end, 'tier': p.get_tier_display(), 'state': _state(p, now)}
                       for p in sorted(plans, key=lambda p: p.created, reverse=True)],
            'optouts': ConversionEmail.objects.filter(kind='optout').count()}


def by_source(views, found):
    """Visitors, sign-ups and subscriptions by utm_source (bluesky-bot, outreach-author, the share buttons...)."""
    rows = {r['source']: {'source': r['source'], 'visitors': r['n'], 'signups': 0, 'paid': 0}
            for r in views.exclude(source='').values('source').annotate(n=Count('visitor', distinct=True))}
    for j in found:
        first = j['origin'] if j['origin'] in rows else None
        if first:
            rows[first]['signups'] += 1
            rows[first]['paid'] += 1 if j['plan'].startswith(('payant', 'essai')) else 0
    return sorted(rows.values(), key=lambda r: (-r['signups'], -r['visitors']))


def _prefix(kind):
    from .example_pages import PREFIXES
    return PREFIXES.get(kind, kind)
