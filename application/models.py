from django.core.validators import FileExtensionValidator
from django.db import models


class PredictionModel(models.Model):
    name = models.CharField('Назва файлу', max_length=100)
    image = models.ImageField('Зображення', validators=[FileExtensionValidator(['jpg', 'jpeg', 'png', 'svg'])])
    classification = models.CharField('Клас', max_length=100)
    prob = models.CharField('Впевненість', max_length=20)

    def __str__(self):
        return self.name
