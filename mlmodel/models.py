import os
from django.db import models


# Create your models here.
class ImageModel(models.Model):
    tempid = models.IntegerField(default=1)
    image = models.ImageField(upload_to="mlmodel/images/")

    def filename(self):
        return os.path.basename(self.image.name)
