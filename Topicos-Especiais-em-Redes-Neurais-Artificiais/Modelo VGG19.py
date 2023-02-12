import os
import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping

model = Sequential()
model.add(Conv2D(input_shape = (224,224,3), filters = 64,
                 kernel_size = (3,3), padding = "same",
                 activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 64, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(filters = 128, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 128, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(filters = 256, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 256, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(Conv2D(filters = 512, kernel_size = (3,3),
                 padding = "same", activation = "relu"))
model.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
model.add(Flatten())
model.add(Dense(units = 4096, activation = "relu"))
model.add(Dense(units = 4096, activation = "relu"))
model.add(Dense(units = 2, activation = "softmax"))
model.compile(optimizer = Adam(learning_rate = 0.001),
              loss = keras.losses.categorical_crossentropy,
              metrics = ['accuracy'])
model.summary()
