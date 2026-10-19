import ipaddress

from django.conf import settings

from . import access

# Crawlers that run JavaScript with a regular browser user agent, so the
# client-side filter in base.html lets them through. Huawei Cloud Singapore
# (PetalBot range) walks /tree/ pages from a new IP each request and showed up
# as ~1,200 "visitors" a day.
UNTRACKED_NETWORKS = [ipaddress.ip_network(n) for n in (
    "114.119.128.0/18",
)]


def _untracked(ip):
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return False
    return any(addr in net for net in UNTRACKED_NETWORKS)


def analytics(request):
    if _untracked(access.client_ip(request)):
        return {'UMAMI_WEBSITE_ID': ''}
    return {'UMAMI_WEBSITE_ID': settings.UMAMI_WEBSITE_ID}
