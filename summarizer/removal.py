"""Deleting what a user made with the tools, one at a time or ticked together, with a trash to undo it.

Private results (the Generated rows of personal_activity.RESULT_KINDS) are deleted with what hangs on them: the
draft of a "What did I miss?", the paper pool of a literature review, the reference watch of a bibliography check,
the program of a conference ranking no other ranking reads. A shared result (a press kit, a journal club deck,
read by every user) is only taken out of the user's list. A result whose work is still running is kept: deleting
it would waste that work. Daily quotas are counted apart: deleting gives none back.

Everything deleted at once, with what the database deleted with it, is kept serialized in a Trash row for
TRASH_DAYS: the user restores it from the notice that follows, or from My activity, or empties it at once.
"""
from datetime import timedelta

from django.contrib.auth.decorators import login_required
from django.core import serializers
from django.db import IntegrityError, transaction
from django.db.models import Q
from django.db.models.deletion import Collector
from django.http import Http404
from django.shortcuts import redirect
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme, urlencode
from django.utils.translation import gettext as _, gettext_lazy, ngettext
from django.views.decorators.http import require_POST

from .models import (AIassistant, ChatConversation, ChatShare, Collection, ConferenceProgram, FollowedPaper,
                     Generated, PaperVisit, ReferenceWatch, SavedSearch, StudentShortlist, ToolUse, Trash,
                     WatchedProject)

MAX = 200  # things one request deletes at most
TRASH_DAYS = 30
LABEL_NAMES = 3  # names a trash line shows


def ids(pks):
    return [int(p) for p in pks if str(p).isdigit()][:MAX]


def discard(user, label, querysets, count, revive=None):
    """Delete these querysets with what cascades from them, keeping all of it in the user's trash; the Trash."""
    collector = Collector(using='default', origin=None)
    for queryset in querysets:
        collector.collect(queryset)
    objects = [obj for instances in collector.data.values() for obj in instances]
    for queryset in collector.fast_deletes:
        objects.extend(queryset)
    if not objects and not revive:
        return None
    with transaction.atomic():
        trash = Trash.objects.create(user=user, label=label[:300], count=count, revive=revive or {},
                                     rows=serializers.serialize('json', objects))
        collector.delete()
    return trash


def named(tool, names, count):
    """'Tool: a, b, c and 2 more', for the trash list."""
    names = [n for n in names if n]
    text = ', '.join(names[:LABEL_NAMES])
    if count > len(names[:LABEL_NAMES]):
        more = count - len(names[:LABEL_NAMES])
        text = (text + ' ' if text else '') + ngettext('and %(n)d more', 'and %(n)d more', more) % {'n': more}
    return '%s: %s' % (tool, text) if text else str(tool)


def remove_results(user, pks):
    """Delete the user's results among these Generated pks; the Trash, or None when nothing was deleted."""
    from . import literature, missed
    from .personal_activity import RESULT_KINDS, SHARED_KINDS
    pks = ids(pks)
    own = [r for r in Generated.objects.filter(pk__in=pks, user=user, kind__in=list(RESULT_KINDS)) if not r.running()]
    uses = list(ToolUse.objects.filter(user=user, row_id__in=pks, row__kind__in=list(SHARED_KINDS)).select_related('row'))
    if not own and not uses:
        return None
    keys = {kind: [r.key for r in own if r.kind == kind] for kind in RESULT_KINDS}
    programs = {r.data().get('program') for r in own if r.kind == 'conference' and r.status == 'done' and r.content}
    unread = []
    for pk in programs - {None}:
        others = Generated.objects.filter(kind='conference', status='done', content__contains='"program": %d' % pk)
        if not any(r.data().get('program') == pk for r in others.exclude(pk__in=[r.pk for r in own])):
            unread.append(pk)  # its team, marks and notes go with it
    names = []
    for row in own:
        data = row.data() if row.status == 'done' and row.content else {}
        names.append(str(RESULT_KINDS[row.kind][2](data) or RESULT_KINDS[row.kind][0]) if data else str(RESULT_KINDS[row.kind][0]))
    names += [u.title or str(SHARED_KINDS[u.row.kind]) for u in uses]
    count = len(own) + len(uses)
    return discard(user, named(_('Results'), names, count), [
        Generated.objects.filter(kind=missed.DRAFT_KIND, user=user, key__in=keys['missed']),
        Generated.objects.filter(kind=literature.POOL_KIND, key__in=keys['literature']),
        ReferenceWatch.objects.filter(user=user, check_key__in=keys['refcheck']),
        ConferenceProgram.objects.filter(pk__in=unread, user=user),
        Generated.objects.filter(pk__in=[r.pk for r in own]),
        ToolUse.objects.filter(pk__in=[u.pk for u in uses]),
    ], count)


def delete_collections(user, pks):
    """Delete the user's own collections among these pks, with the analyses written for them."""
    from .generated import COLLECTION_KINDS
    collections = list(Collection.objects.filter(user=user, pk__in=ids(pks)))
    if not collections:
        return None
    keys = Q()
    for c in collections:
        keys |= Q(key__startswith='%d:' % c.pk)
    return discard(user, named(_('Collections'), [c.name for c in collections], len(collections)), [
        Generated.objects.filter(keys, kind__in=COLLECTION_KINDS),
        Collection.objects.filter(pk__in=[c.pk for c in collections])], len(collections))


def remove_conversations(user, rooms):
    """Take these conversations out of the user's history, as the chat page's delete does: its exchanges are
    deactivated (the trash remembers which, to bring them back), its name deleted, its shared link stopped."""
    rooms = [r for r in rooms if r][:MAX]
    exchanges = AIassistant.objects.filter(user=user, arxiv_id__in=rooms, active=True)
    found = sorted(set(exchanges.values_list('arxiv_id', flat=True)))
    if not found:
        return None
    shares = ChatShare.objects.filter(user=user, room__in=found, active=True)
    titles = dict(ChatConversation.objects.filter(user=user, room__in=found).values_list('room', 'title'))
    firsts = {}
    for room, query in exchanges.order_by('pk').values_list('arxiv_id', 'query'):
        firsts.setdefault(room, (query or '')[:80])
    revive = {'assistant': list(exchanges.values_list('pk', flat=True)), 'share': list(shares.values_list('pk', flat=True))}
    with transaction.atomic():
        exchanges.update(active=False)
        shares.update(active=False)
        trash = discard(user, named(_('Conversations'), [titles.get(r) or firsts.get(r) for r in found], len(found)),
                        [ChatConversation.objects.filter(user=user, room__in=found)], len(found), revive)
    return trash


def delete_readings(user, pks=None):
    """Delete papers of the reading history with their notes and reading status (every paper when pks is None)."""
    visits = PaperVisit.objects.filter(user=user)
    if pks is not None:
        visits = visits.filter(pk__in=ids(pks))
    rows = list(visits.values_list('pk', 'title', 'arxiv_id')[:5000 if pks is None else MAX])
    if not rows:
        return None
    return discard(user, named(_('Readings'), [title or aid for _pk, title, aid in rows], len(rows)),
                   [PaperVisit.objects.filter(pk__in=[r[0] for r in rows])], len(rows))


def simple(model, tool, name):
    """The deletion of the user's own rows of a model that cascades by itself."""
    def remove(user, pks):
        rows = list(model.objects.filter(user=user, pk__in=ids(pks)).order_by('pk'))
        if not rows:
            return None
        return discard(user, named(tool, [name(r) for r in rows], len(rows)),
                       [model.objects.filter(pk__in=[r.pk for r in rows])], len(rows))
    return remove


# what the delete view takes: the name in its URL, the field of the ticked lines
REMOVERS = {
    'results': (remove_results, 'r'),
    'conversations': (remove_conversations, 'room'),
    'collections': (delete_collections, 'c'),
    'readings': (delete_readings, 'v'),
    'saved-searches': (simple(SavedSearch, gettext_lazy('Saved searches'), lambda r: r.name), 'pk'),
    'scoops': (simple(WatchedProject, gettext_lazy('Scoop watch'), lambda r: r.title), 'pk'),
    'followed': (simple(FollowedPaper, gettext_lazy('Citation alerts'), lambda r: r.title or r.arxiv_id), 'pk'),
    'leads': (simple(StudentShortlist, gettext_lazy('Placement leads'), lambda r: (r.snapshot or {}).get('name', '')), 'pk'),
}


def restore(user, pk):
    """Put back what a trash line holds; False when it cannot be (something it needs is gone, or came back)."""
    trash = Trash.objects.filter(user=user, pk=pk).first()
    if trash is None:
        return False
    try:
        with transaction.atomic():
            for obj in serializers.deserialize('json', trash.rows):
                obj.save(force_insert=True)  # never over a row made since with the same id
            AIassistant.objects.filter(user=user, pk__in=trash.revive.get('assistant', [])).update(active=True)
            ChatShare.objects.filter(user=user, pk__in=trash.revive.get('share', [])).update(active=True)
            trash.delete()
    except IntegrityError:
        return False
    return True


def purge(now=None):
    """Forget the trash older than TRASH_DAYS (hourly, send_digest)."""
    return Trash.objects.filter(created__lt=(now or timezone.now()) - timedelta(days=TRASH_DAYS)).delete()[0]


def back_to(request, **params):
    """To the page of this site the form was sent from, saying what was done."""
    here = request.POST.get('next', '')
    if not here.startswith('/') or not url_has_allowed_host_and_scheme(here, allowed_hosts={request.get_host()}):
        here = reverse('history') + '?tab=results'
    base, _sep, query = here.split('#')[0].partition('?')
    kept = [p for p in query.split('&') if p and not p.startswith(('done=', 'deleted=', 'trash='))]
    kept.append(urlencode(params))
    return redirect(base + '?' + '&'.join(kept))


@login_required(login_url='login')
@require_POST
def delete_view(request, what):
    """Delete one line (the button of the line, one=pk), the ticked ones, or, for results, every result of the
    page's tool filter (all=1 with kind)."""
    if what not in REMOVERS:
        raise Http404
    remove, field = REMOVERS[what]
    if request.POST.get('one'):
        pks = [request.POST['one']]
    elif what == 'results' and request.POST.get('all') == '1':
        from .personal_activity import results
        pks = list(results(request.user, request.POST.get('kind', '')).values_list('pk', flat=True)[:MAX])
    elif what == 'readings' and request.POST.get('all') == '1':
        pks = None  # the whole reading history
    else:
        pks = request.POST.getlist(field)
    trash = remove(request.user, pks)
    if trash is None:
        return back_to(request, done='deleted', deleted=0)
    return back_to(request, done='deleted', deleted=trash.count, trash=trash.pk)


@login_required(login_url='login')
@require_POST
def restore_view(request, pk):
    return back_to(request, done='restored' if restore(request.user, pk) else 'unrestorable')


@login_required(login_url='login')
@require_POST
def empty_view(request):
    """Empty the trash at once (one=pk: only that line): what it held cannot come back."""
    trash = Trash.objects.filter(user=request.user)
    if request.POST.get('one'):
        trash = trash.filter(pk=request.POST['one'] if request.POST['one'].isdigit() else 0)
    trash.delete()
    return back_to(request, done='emptied')
