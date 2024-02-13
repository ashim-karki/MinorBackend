from django.urls import path
from .views import UseMlModel

urlpatterns = [path("", UseMlModel, name="mlmodel")]
