from django.urls import path
from .views import UseMlModel,UseMlModelTemp,DetectEmotion

urlpatterns = [path("mlmodel", UseMlModelTemp, name="mlmodel"),
               path("detect",DetectEmotion,name= 'detect'),
               ]
