import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

# carregando a base de dados

nomes = ['airplane',
         'automobile',
         'bird',
         'cat',
         'deer',
         'dog',
         'frog',
         'horse',
         'ship',
         'truck']

(X_treino, y_treino), (X_teste, y_teste) = cifar10.load_data()

# Normalização das imagens

X_treino = X_treino / 255.0
X_teste = X_teste / 255.0

plt.imshow(X_teste[55])


# Construção do modelo

modelo = tf.keras.models.Sequential()

modelo.add(tf.keras.layers.Conv2D(filters = 32,
                                  kernel_size = 3,
                                  padding = 'same',
                                  activation = 'relu',
                                  input_shape = [32, 32, 3]))
modelo.add(tf.keras.layers.Conv2D(filters = 32,
                                  kernel_size = 3,
                                  padding = 'same',
                                  activation = 'relu'))
modelo.add(tf.keras.layers.MaxPool2D(pool_size = 2,
                                     strides = 2,
                                     padding = 'valid'))
modelo.add(tf.keras.layers.Conv2D(filters = 64,
                                  kernel_size = 3,
                                  padding = 'same',
                                  activation = 'relu'))
modelo.add(tf.keras.layers.Conv2D(filters = 64,
                                  kernel_size = 3,
                                  padding = 'same',
                                  activation = 'relu'))
modelo.add(tf.keras.layers.MaxPool2D(pool_size = 2,
                                     strides = 2,
                                     padding = 'valid'))
modelo.add(tf.keras.layers.Flatten())
modelo.add(tf.keras.layers.Dense(units = 128,
                                 activation = 'relu'))
modelo.add(tf.keras.layers.Dense(units = 10,
                                 activation = 'softmax'))
modelo.compile(loss = 'sparse_categorical_crossentropy',
               optimizer = 'Adam',
               metrics = ['sparse_categorical_accuracy'])
modelo.fit(X_treino, y_treino, epochs = 10)


print(modelo.summary())

# Avaliando o modelo

margem_erro, margem_acerto = modelo.evaluate(X_teste, y_teste)

print(f'Precisão: {margem_acerto}')
