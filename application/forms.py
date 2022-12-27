from django import forms

from .models import PredictionModel


class PredictionModelForm(forms.ModelForm):
    class Meta:
        model = PredictionModel
        fields = ('image',)
