"""Server-side identity, paid status and quotas.

Everything that decides who a visitor is, whether they get the paid model, and
how much they may spend lives here, so the HTTP views and the websocket
consumers apply the same rules. Nothing in here trusts values sent by the
browser.
"""
import hashlib
import ipaddress

from django.utils import timezone

from .models import AIassistant, PaperHistory, UserPlan

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

RUNS_PER_DAY = 10
RUN_MIN_INTERVAL_SECONDS = 10
CHAT_MESSAGES_PER_DAY = {"anonymous": 30, "user": 100, "paid": 500}
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


def is_paid(user):
    if user is None or not user.is_authenticated:
        return False
    plan = UserPlan.objects.filter(user=user).order_by("created").last()
    if plan is None or not plan.paid:
        return False
    if plan.expires is not None and plan.expires < timezone.now():
        return False
    return True


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
    if runs_today >= RUNS_PER_DAY and not is_paid(user):
        return False, "toomany"
    return True, None


def chat_quota_left(user, hashed_ip):
    if user is not None and user.is_superuser:
        return 1
    since = timezone.now() - timezone.timedelta(days=1)
    if user is not None and user.is_authenticated:
        tier = "paid" if is_paid(user) else "user"
        used = AIassistant.objects.filter(user=user, created__gte=since).count()
    else:
        tier = "anonymous"
        used = AIassistant.objects.filter(user=None, ip_address=hashed_ip, created__gte=since).count()
    return CHAT_MESSAGES_PER_DAY[tier] - used


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
