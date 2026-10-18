from django.conf import settings


def analytics(request):
    return {'UMAMI_WEBSITE_ID': settings.UMAMI_WEBSITE_ID}
