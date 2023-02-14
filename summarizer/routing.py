from django.urls import path, re_path
from . import consumers


websocket_urlpatterns = [
    re_path(r"ws/arxivid/(?P<arxiv_id>\w.+)/$", consumers.LoadingConsumer.as_asgi()),
]
