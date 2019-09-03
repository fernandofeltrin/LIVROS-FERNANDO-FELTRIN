from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
import numpy as np
from sklearn.model_selection import StratifiedKFold

seed = 5
np.random.seed(seed)

(etreino,streino), (eteste,steste) = mnist.load_data()
entradas = etreino.reshape(etreino.shape[0], 28, 28, 1)
entradas = entradas.astype('float32')
entradas /= 255
saidas = np_utils.to_categorical(streino, 10)

kfold = StratifiedKFold(n_splits = 5, shuffle = True, random_state = seed)
resultados = []

a = np.zeros(5)
b = np.zeros(shape = (saidas.shape[0], 1))

for evalcruzada,svalcruzada in kfold.split(entradas, 
                                            np.zeros(shape=(saidas.shape[0],1))):
    classificador = Sequential()
    classificador.add(Conv2D(32, (3,3), input_shape=(28,28,1), activation='relu'))
    classificador.add(MaxPooling2D(pool_size = (2,2)))
    classificador.add(Flatten())
    classificador.add(Dense(units = 128, activation = 'relu'))
    classificador.add(Dense(units = 10, activation = 'softmax'))
    classificador.compile(loss = 'categorical_crossentropy', optimizer='adam',
                          metrics = ['accuracy'])
    classificador.fit(entradas[evalcruzada], saidas[evalcruzada],
                      batch_size = 128, epochs = 5)
    precisao = classificador.evaluate(entradas[svalcruzada], saidas[svalcruzada])
    resultados.append(precisao[1])

media = sum(resultados) / len(resultados)

