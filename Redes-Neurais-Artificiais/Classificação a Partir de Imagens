from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

base = keras.datasets.fashion_mnist

(etreino,streino),(eteste,steste) = base.load_data()

rotulos = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt',
           'Sneaker','Bag','Ankle boot']

etreino = etreino / 255
streino = streino / 255

plt.figure(figsize = (10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(etreino[i],
               cmap = plt.cm.binary)
    plt.xlabel(rotulos,[steste[i]])
plt.show()

classificador = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

classificador.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
classificador.fit(etreino,
                  streino,
                  epochs = 5)
