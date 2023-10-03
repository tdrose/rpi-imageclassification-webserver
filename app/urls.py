from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('getprediction/', views.get_prediction, name='getprediction')
]