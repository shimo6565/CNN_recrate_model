from tensorflow.keras import layers
from tensorflow.keras import models
import numpy as np

def build_model():
    model = models.Sequential() ##added
    model.add(layers.Conv2D(filters = 16, kernel_size = (2, 1), strides = (1,1),padding = 'same', activation = 'relu', input_shape = (40, 14, 3)))
    model.add(layers.Conv2D(filters = 16, kernel_size = (2, 1), strides = (1,1),padding = 'same', activation = 'relu'))
    model.add(layers.Dropout(rate = 0.2))
    model.add(layers.Conv2D(filters = 16, kernel_size = (2, 1), strides = (1,1),padding = 'same', activation = 'relu'))
    model.add(layers.MaxPooling2D(pool_size = (2,1), strides = (2,1),padding = 'same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(units = 1024, activation = 'relu'))
    model.add(layers.Dense(units = 9, activation = 'softmax'))
    model.compile(optimizer = 'rmsprop', loss = 'categorical_crossentropy',metrics = ['acc'])
    return model

X = np.load('X.npy')
y = np.load('y.npy')

model = build_model()
model.fit(X, y, epochs = 100, batch_size = 1, verbose = 0)