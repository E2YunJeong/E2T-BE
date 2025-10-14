from django.urls import path

from . import views

urlpatterns = [
    path('new_session', views.NewSession.as_view(), name='new_session'),
    path('frame', views.FrameAPI.as_view(), name='frame'),
    path('reset', views.ResetSession.as_view(), name='reset'),
]
