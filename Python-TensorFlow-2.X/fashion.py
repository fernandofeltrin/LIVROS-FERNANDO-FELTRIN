import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Carregando a base de dados

(X_treino, y_treino), (X_teste, y_teste) = fashion_mnist.load_data()

print(X_treino)
print(X_treino[0]) # visualizando o primeiro elemento
print(y_treino)
print(y_treino[0])


# Normalizando as imagens

X_treino = X_treino / 255.0 # convertendo da escala 0-255 para escala 0-1
X_teste = X_teste / 255.0

print(X_treino[0])

# Remodelando a base de dados

print(X_treino.shape)

X_treino = X_treino.reshape(-1, 28*28) # -1 significa todos elementos, altura*largura

print(X_treino.shape)

X_teste = X_teste.reshape(-1, 28*28)

# Construindo o modelo

modelo = tf.keras.models.Sequential()

print(modelo)

modelo.add(tf.keras.layers.Dense(units = 128,
                                 activation = 'relu',
                                 input_shape = (784, )))
modelo.add(tf.keras.layers.Dropout(0.2))
modelo.add(tf.keras.layers.Dense(units = 10,
                                 activation = 'softmax'))
modelo.compile(optimizer = 'adam',
               loss = 'sparse_categorical_crossentropy',
               metrics = ['sparse_categorical_accuracy'])

# Treinando o modelo

print(modelo.summary())

modelo.fit(X_treino,
           y_treino,
           epochs = 100)

# Avaliando o modelo

margem_erro, margem_acerto = modelo.evaluate(X_teste,
                                             y_teste)

print(f'Margem de Precisão: {margem_acerto}')
print(f'Margem de Erro: {margem_erro}')
