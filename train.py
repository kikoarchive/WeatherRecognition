import os
import pandas as pd
import tensorflow as tf
import cv2
import numpy as np
from imutils import paths
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
from keras.utils import img_to_array
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

imgPath = list(paths.list_images('dataset'))
images = []
labels = []
errors = []
category = 0
names = {
    0: "хмарна",
    1: "туманна",
    2: "дощить",
    3: "сонячна",
    4: "схід сонця"
}
for i in imgPath:
    try:
        img = cv2.imread(i)
        img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
        img = img_to_array(img)
        images.append(np.array(img))
        label = i.split(os.path.sep)[-2]
        labels.append(label)
    except:
        errors.append(i)

data_train, labels_train = np.array(images), labels
labels_train = pd.factorize(np.array(labels_train))[0]

np.unique(labels_train)
x_train, x_test, y_train, y_test = train_test_split(data_train, labels_train, test_size=0.25, random_state=True)

model = Sequential()
model.add(Conv2D(32, activation='relu', kernel_size=3, input_shape=x_train.shape[1:]))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(64, activation='relu', kernel_size=3))
model.add(MaxPool2D(2, 2))
model.add(Conv2D(128, activation='relu', kernel_size=3))
model.add(MaxPool2D(2, 2))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dense(5, activation="softmax"))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

model.fit(x_train, y_train, epochs=40)

loss, accuracy = model.evaluate(x_test, y_test, batch_size=256)

model.summary()
model.save('model/prediction-model3.h5')

result = model.predict(x_test[200:201])
index = np.argmax(result)
# plt.imshow(x_test[200:201])
print(names[index])
print(result)
