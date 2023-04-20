from django.urls import path, re_path
from . import consumers


websocket_urlpatterns = [
    #re_path(r"ws/arxivid/(?P<arxiv_id>\w.+)/$", consumers.LoadingConsumer.as_asgi()),
    re_path(r"ws/arxivid/(?P<arxiv_id>\w.+)/(?P<language>\w.+)/$", consumers.LoadingConsumer.as_asgi()),
    re_path(r'ws/emb/$', consumers.EmbeddingConsumer.as_asgi()),

]
