# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import np_utils
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

(etreino,streino),(eteste,steste) = mnist.load_data()

plt.imshow(etreino[0], cmap = 'gray')
plt.title('Classe' + str(streino[0]))

etreino = etreino.reshape(etreino.shape[0],28,28,1)
eteste = eteste.reshape(eteste.shape[0],28,28,1)

etreino = etreino.astype('float32')
eteste = eteste.astype('float32')

etreino /= 255
eteste /= 255

streino = np_utils.to_categorical(streino, 10)
steste = np_utils.to_categorical(steste, 10)

classificador = Sequential()
classificador.add(Conv2D(32,
                         (3,3),
                         input_shape = (28,28,1),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Conv2D(32,
                         (3,3),
                         activation = 'relu'))
classificador.add(BatchNormalization())
classificador.add(MaxPooling2D(pool_size = (2,2)))
classificador.add(Flatten())
classificador.add(Dense(units = 128,
                        activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 128,
                        activation = 'relu'))
classificador.add(Dropout(0.2))
classificador.add(Dense(units = 10,
                        activation = 'softmax'))
classificador.compile(loss = 'categorical_crossentropy',
                      optimizer = 'adam',
                      metrics = ['accuracy'])
classificador.fit(etreino,
                  streino,
                  batch_size = 128,
                  epochs = 10,
                  validation_data = (eteste,steste))

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
