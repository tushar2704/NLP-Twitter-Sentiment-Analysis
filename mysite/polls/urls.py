from django.urls import path
from django.conf.urls.static import static
from .views import sentiment

urlpatterns = [
    path('', sentiment, name='homepage'),
    ]