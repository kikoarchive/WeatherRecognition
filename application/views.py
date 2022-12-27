import os
import pickle
from PIL import Image
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import cv2
from keras.utils import img_to_array
from tensorflow.python.keras.models import load_model

from application.forms import PredictionModelForm
from application.models import PredictionModel

model = load_model("model/prediction-model2.h5", compile=False)

names = {
    0: "хмарно",
    1: "туманність",
    2: "дощ",
    3: "сонячно",
    4: "схід сонця"
}


def toFixed(Obj, digits=0):
    return f"{Obj:.{digits}f}"


def index(request):
    if request.method == 'POST':
        file = request.FILES["filePath"]
        fs = FileSystemStorage()
        file_path = fs.save(file.name, file)
        file_path = "."+fs.url(file_path)
        try:
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)
            result = model.predict(image)[0]
            ind = np.argmax(result)
            res = toFixed(result[ind] * 100)
            label = names[ind]
            pmodel = PredictionModel()
            pmodel.name = file.name
            pmodel.image = file
            pmodel.classification = label;
            pmodel.prob = res
            pmodel.save()
            context = {
                'filePathName': file_path,
                'probability': res,
                'label': label,
            }
        except:
            context = {
                'error': "Зображення пошкоджено!"
            }
        return render(request, 'index.html', context)
    else:
        return render(request, 'index.html')