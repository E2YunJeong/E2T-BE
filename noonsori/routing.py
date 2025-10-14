from django.urls import re_path

from .consumers import BlinkConsumer

websocket_urlpatterns = [
    re_path(r"^ws/blink/(?P<session_id>[A-Za-z0-9_\-]+)/$", BlinkConsumer.as_asgi()),
]
