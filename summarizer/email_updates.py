"""A short, explicit opt-in to the existing digest and weekly letter subscriptions."""
import time
from urllib.parse import urlencode

from django.contrib.auth.models import User
from django.db import transaction
from django.shortcuts import redirect, render
from django.urls import reverse
from django.utils.translation import get_language, gettext as _
from django.views.decorators.http import require_http_methods

from . import access, digest, letters
from .models import DigestSubscription, LetterSubscription

# At most five starter categories: compatible with a free digest, and editable afterwards.
STARTERS = {
    'ai': 'cs.AI,cs.LG,cs.CL,cs.CV,stat.ML',
    'space': 'astro-ph.GA,astro-ph.CO,astro-ph.EP,astro-ph.HE,astro-ph.SR',
    'quantum': 'quant-ph',
    'matter': 'cond-mat.mtrl-sci,cond-mat.mes-hall,cond-mat.soft,physics.optics,physics.app-ph',
    'particles': 'hep-ph,hep-th,hep-ex,nucl-th,gr-qc',
    'brain': 'q-bio.NC,biorxiv.neuroscience,medrxiv.neurology',
    'life': 'q-bio.BM,q-bio.CB,q-bio.GN,biorxiv.cell-biology,medrxiv.infectious-diseases',
    'climate': 'physics.ao-ph,physics.geo-ph,astro-ph.EP,biorxiv.ecology',
    'math': 'math.PR,math.ST,math.OC,stat.ME,stat.TH',
    'computing': 'cs.DS,cs.SE,cs.CR,cs.DC,eess.SP',
    'economy': 'econ.EM,econ.GN,q-fin.EC,q-fin.ST,q-fin.PM',
}
PENDING = 'email_updates_pending'
PENDING_MAX_AGE = 24 * 3600


def form_context(topic='', products=None):
    return {'update_topics': [(slug, letters.TOPICS[slug]['name']) for slug in STARTERS],
            'update_topic': topic, 'update_products': ['digest', 'letters'] if products is None else products}


@require_http_methods(['GET', 'POST'])
def signup(request):
    data = request.POST if request.method == 'POST' else request.GET
    topic = data.get('topic', '')
    products = [p for p in ('digest', 'letters') if p in data.getlist('products')]
    context = form_context(topic, products if request.method == 'POST' or 'products' in data else None)
    if request.method == 'GET':
        pending = request.session.pop(PENDING, None) if request.user.is_authenticated else None
        if (pending and time.time() - pending['at'] < PENDING_MAX_AGE
                and pending['topic'] in STARTERS):
            error = subscribe(request, pending['topic'], [p for p in ('digest', 'letters') if p in pending['products']])
            if error:
                context['update_error'] = error
        if request.user.is_authenticated:
            context['updates_done'] = request.session.pop('email_updates_done', None)
        return render(request, 'summarizer/email_updates.html', context)
    if topic not in STARTERS or not products:
        context['update_error'] = _('Choose a field and at least one email format.')
        return render(request, 'summarizer/email_updates.html', context, status=400)
    if not request.user.is_authenticated:
        # Only this browser's session, filled by a CSRF-checked POST, applies choices after sign-in:
        # a link alone never subscribes anyone.
        request.session[PENDING] = {'topic': topic, 'products': products, 'at': time.time()}
        next_url = reverse('email_updates') + '?' + urlencode({'topic': topic, 'products': products}, doseq=True)
        return redirect(reverse('register') + '?' + urlencode({'next': next_url}))
    error = subscribe(request, topic, products)
    if error:
        context['update_error'] = error
        return render(request, 'summarizer/email_updates.html', context, status=409)
    return redirect(reverse('email_updates'))


def subscribe(request, topic, products):
    """Opt the signed-in user in to the chosen formats; an error message, or None once done."""
    lang = get_language()
    with transaction.atomic():
        # Serialize duplicate clicks, including across web workers.
        User.objects.select_for_update().get(pk=request.user.pk)
        categories = STARTERS[topic].split(',')
        subscriptions = list(DigestSubscription.objects.filter(user=request.user))
        existing_digest = next((s for s in subscriptions if s.active and set(categories) <= set(s.category_list())), None)
        if existing_digest is None:
            existing_digest = next((s for s in subscriptions if not s.active and set(categories) == set(s.category_list())), None)
        existing_letter = LetterSubscription.objects.filter(user=request.user, topic=topic).first()
        error = None
        if ('digest' in products and existing_digest is None and
                len(subscriptions) >= digest.limits(access.has_tool(request.user, 'digest'))['digests']):
            error = _('You already have a digest. Personalize it below to keep your existing preferences.')
        if ('letters' in products and not (existing_letter and existing_letter.active) and
                not access.has_tool(request.user, 'letters') and
                LetterSubscription.objects.filter(user=request.user, active=True).count() >= letters.FREE_SUBSCRIPTIONS):
            error = _('You already receive a weekly letter. Change your choice below, or keep your current subscription.')
        if error:
            return error
        if 'digest' in products:
            if existing_digest is None:
                existing_digest = DigestSubscription.objects.create(user=request.user, categories=STARTERS[topic],
                    lang=lang, frequency='daily', level='general', send_hour=7, timezone='Europe/Paris')
                request.session['digest_configured'] = existing_digest.pk
            elif not existing_digest.active:
                existing_digest.active = True
                existing_digest.save(update_fields=['active', 'updated'])
        if 'letters' in products and not (existing_letter and existing_letter.active):
            LetterSubscription.objects.update_or_create(user=request.user, topic=topic,
                defaults={'active': True, 'lang': lang})
        request.session['email_updates_done'] = {'topic': topic, 'products': products,
                                                'digest_id': existing_digest.pk if 'digest' in products else None}


