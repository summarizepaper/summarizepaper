"""Server-side identity, paid status and quotas.

Everything that decides who a visitor is, whether they get the paid model, and
how much they may spend lives here, so the HTTP views and the websocket
consumers apply the same rules. Nothing in here trusts values sent by the
browser.
"""
import hashlib
import ipaddress

from django.utils import timezone
from django.db.models import Q, Sum

from .models import AIassistant, PaperHistory, SiteSettings, UserPlan, PremiumGrant
from .premium_tools import PLUS_TOOLS, TOOL_KEYS

# https://www.cloudflare.com/ips/ - requests coming from these addresses are
# Cloudflare edges, so their CF-Connecting-IP header is trustworthy.
CLOUDFLARE_NETWORKS = [ipaddress.ip_network(n) for n in (
    "173.245.48.0/20", "103.21.244.0/22", "103.22.200.0/22", "103.31.4.0/22",
    "141.101.64.0/18", "108.162.192.0/18", "190.93.240.0/20", "188.114.96.0/20",
    "197.234.240.0/22", "198.41.128.0/17", "162.158.0.0/15", "104.16.0.0/13",
    "104.24.0.0/14", "172.64.0.0/13", "131.0.72.0/22",
    "2400:cb00::/32", "2606:4700::/32", "2803:f800::/32", "2405:b500::/32",
    "2405:8100::/32", "2a06:98c0::/29", "2c0f:f248::/32",
)]

RUN_MIN_INTERVAL_SECONDS = 10
RUN_TOKENS_SESSION_KEY = "allowed_runs"


def _is_cloudflare(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in CLOUDFLARE_NETWORKS)


def client_ip_from_headers(get_header, remote_addr):
    """Return the visitor IP.

    The Heroku router appends the address that connected to it as the last
    X-Forwarded-For entry; earlier entries are whatever the client sent and can
    be forged. When that last hop is a Cloudflare edge, Cloudflare's own
    CF-Connecting-IP header is the visitor.
    """
    xff = get_header("x-forwarded-for")
    if not xff:
        return remote_addr or ""
    last_hop = xff.split(",")[-1].strip()
    if _is_cloudflare(last_hop):
        cf_ip = get_header("cf-connecting-ip")
        if cf_ip:
            return cf_ip.strip()
    return last_hop


def client_ip(request):
    return client_ip_from_headers(
        lambda name: request.META.get("HTTP_" + name.upper().replace("-", "_")),
        request.META.get("REMOTE_ADDR"),
    )


def client_ip_from_scope(scope):
    headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
    client = scope.get("client") or (None, None)
    return client_ip_from_headers(headers.get, client[0])


def hash_ip(ip):
    return hashlib.sha256((ip or "").encode("utf-8")).hexdigest()


def runs_per_day():
    """New summaries per day without Premium (set in the admin)."""
    return SiteSettings.current().free_runs_per_day


def chat_messages_per_day():
    """Questions per day to the assistant, by tier (set in the admin)."""
    cfg = SiteSettings.current()
    return {"anonymous": cfg.chat_anonymous_per_day, "user": cfg.chat_user_per_day, "paid": cfg.chat_paid_per_day}


def active_grants(user):
    """Read current grants afresh: revocations apply to the next request."""
    if user is None or not user.is_authenticated or not getattr(user, 'is_active', True):
        return PremiumGrant.objects.none()
    return PremiumGrant.objects.filter(user=user, active=True).filter(Q(expires__isnull=True) | Q(expires__gte=timezone.now()))


def current_plan(user):
    """The user's subscription when it is paid and running, else None."""
    if user is None or not user.is_authenticated or not getattr(user, 'is_active', True):
        return None
    plan = UserPlan.objects.filter(user=user).order_by('created', 'pk').last()
    if plan and plan.paid and (plan.expires is None or plan.expires >= timezone.now()):
        return plan
    return None


def is_paid(user):
    """Full Premium or Premium+ only. A tool grant must never lift global AI quotas or block checkout."""
    return current_plan(user) is not None or active_grants(user).filter(tool__in=('all', 'plus')).exists()


def is_plus(user):
    """Premium+: a Premium+ subscription or a full Premium+ grant."""
    plan = current_plan(user)
    return (plan is not None and plan.tier == 'plus') or active_grants(user).filter(tool='plus').exists()


def has_tool(user, tool):
    """Authorization shared by pages, generation, status APIs, downloads and jobs."""
    if tool not in TOOL_KEYS:
        return False  # typos and unregistered scopes cannot grant access
    if active_grants(user).filter(tool=tool).exists():
        return True
    return is_plus(user) if tool in PLUS_TOOLS else is_paid(user)


def run_quota(user, hashed_ip):
    """Return (allowed, reason) for starting a new paper summary."""
    if user is not None and user.is_superuser:
        return True, None
    now = timezone.now()
    recent = PaperHistory.objects.filter(
        ip_address=hashed_ip,
        created__gte=now - timezone.timedelta(seconds=RUN_MIN_INTERVAL_SECONDS),
    ).exists()
    if recent:
        return False, "robot"
    runs_today = PaperHistory.objects.filter(
        ip_address=hashed_ip, created__gte=now - timezone.timedelta(days=1)
    ).count()
    if runs_today >= runs_per_day() and not is_paid(user):
        return False, "toomany"
    return True, None


def throttled(name, request, per_minute):
    """True when this visitor (account, else IP) already made per_minute requests of this kind this minute."""
    from django.conf import settings
    from django.core.cache import cache
    if not settings.THROTTLE:
        return False
    who = 'u%s' % request.user.pk if request.user.is_authenticated else hash_ip(client_ip(request))[:16]
    key = 'throttle:%s:%s:%d' % (name, who, int(timezone.now().timestamp()) // 60)
    cache.add(key, 0, 120)
    try:
        return cache.incr(key) > per_minute
    except ValueError:
        return False


ANONYMOUS_SEARCHES_PER_HOUR = 30  # result pages reading the journals and preprints, per address without an account


def anonymous_searches_over(request):
    """True when this visitor, without an account, already made ANONYMOUS_SEARCHES_PER_HOUR searches this hour:
    their next ones read arXiv only (see sources.search_allowed)."""
    from django.conf import settings
    from django.core.cache import cache
    if request.user.is_authenticated or not settings.THROTTLE:
        return False
    key = 'anonymous-searches:%s:%d' % (hash_ip(client_ip(request))[:16], int(timezone.now().timestamp()) // 3600)
    cache.add(key, 0, 7200)
    try:
        return cache.incr(key) > ANONYMOUS_SEARCHES_PER_HOUR
    except ValueError:
        return False


def chat_quota_left(user, hashed_ip):
    if user is not None and user.is_superuser:
        return 1
    since = timezone.now() - timezone.timedelta(days=1)
    if user is not None and user.is_authenticated:
        tier = "paid" if is_paid(user) else "user"
        rows = AIassistant.objects.filter(user=user, created__gte=since)
    else:
        tier = "anonymous"
        rows = AIassistant.objects.filter(user=None, ip_address=hashed_ip, created__gte=since)
    return chat_messages_per_day()[tier] - (rows.aggregate(used=Sum("cost"))["used"] or 0)


def grant_run(session, arxiv_id, kind):
    """Allow the websocket in this session to start one task for this paper."""
    tokens = session.get(RUN_TOKENS_SESSION_KEY, [])
    token = "%s:%s" % (kind, arxiv_id)
    if token not in tokens:
        tokens.append(token)
    session[RUN_TOKENS_SESSION_KEY] = tokens[-20:]


def consume_run(session, arxiv_id, kind):
    """Return True and spend the token if the HTTP view granted this task."""
    tokens = session.get(RUN_TOKENS_SESSION_KEY, [])
    token = "%s:%s" % (kind, arxiv_id)
    if token not in tokens:
        return False
    tokens.remove(token)
    session[RUN_TOKENS_SESSION_KEY] = tokens
    return True
