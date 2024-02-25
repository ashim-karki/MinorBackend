from django.urls import path
from .views import UseMlModel,UseMlModelTemp,DetectEmotion

urlpatterns = [path("mlmodel", UseMlModel, name="mlmodel"),
               path("detect",DetectEmotion,name= 'detect'),
               ]
