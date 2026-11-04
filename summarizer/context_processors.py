import ipaddress

from django.conf import settings

from . import access
from .tool_list import TOOL_GROUPS

# Crawlers that run JavaScript with a regular browser user agent, so the
# client-side filter in base.html lets them through. Huawei Cloud Singapore
# (PetalBot range) walks /tree/ pages from a new IP each request and showed up
# as ~1,200 "visitors" a day. Tencent Cloud and 202.46.62.x scroll the paper pages and asked for the
# sentences of the papers around them (neighbours.py) from dozens of addresses, 2026-10-31.
UNTRACKED_NETWORKS = [ipaddress.ip_network(n) for n in (
    "114.119.128.0/18",
    "43.172.0.0/15",
    "202.46.62.0/24",
)]


def untracked(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in UNTRACKED_NETWORKS)


def google_login(request):
    """Whether the "Continue with Google" button can be shown (accounts.google_start)."""
    return {'google_login': bool(settings.GOOGLE_CLIENT_ID and settings.GOOGLE_CLIENT_SECRET)}


def analytics(request):
    if untracked(access.client_ip(request)):
        return {'UMAMI_WEBSITE_ID': ''}
    return {'UMAMI_WEBSITE_ID': settings.UMAMI_WEBSITE_ID}


def tools_menu(request):
    return {'TOOL_GROUPS': TOOL_GROUPS}
