import os

from channels.auth import AuthMiddlewareStack
from channels.routing import ProtocolTypeRouter, URLRouter
from channels.security.websocket import AllowedHostsOriginValidator
#from django.core.asgi import get_asgi_application
from django.core.asgi import get_asgi_application
from channels.routing import get_default_application
import django

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'arxivsummarizer.settings')
#os.environ['ASGI_THREADS']="4"#to try later if too many connections arrives or use  pgbouncer https://stackoverflow.com/questions/60339917/fatal-too-many-connections-for-role-heroku-django-only-while-using-asgi

django.setup()

from summarizer.routing import websocket_urlpatterns

#django_asgi_app = get_default_application()

# Initialize Django ASGI application early to ensure the AppRegistry
# is populated before importing code that may import ORM models.
#django_asgi_app = get_default_application()
django_asgi_app = get_asgi_application()
import summarizer.routing

'''
application = ProtocolTypeRouter({
    "http": django_asgi_app,
    # Just HTTP for now. (We can add other protocols later.)
})'''

print('joe')

application = ProtocolTypeRouter(
    {
        "http": django_asgi_app,
        "websocket": AllowedHostsOriginValidator(
            AuthMiddlewareStack(URLRouter(websocket_urlpatterns))
        ),
    }
)
