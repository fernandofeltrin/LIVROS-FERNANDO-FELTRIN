import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

entradas = pd.read_csv('entradas-breast.csv')
saidas = pd.read_csv('saidas-breast.csv')

classificadorB = Sequential()
classificadorB.add(Dense(units = 8,
                         activation = 'relu',
                         kernel_initializer = 'normal',
                         input_dim = 30))
classificadorB.add(Dropout(0.2))
classificadorB.add(Dense(units = 8,
                         activation = 'relu',
                         kernel_initializer = 'normal',))
classificadorB.add(Dropout(0.2))
classificadorB.add(Dense(units = 1,
                         activation = 'sigmoid'))
classificadorB.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['binary_accuracy'])
classificadorB.fit(entradas,
                   saidas,
                   batch_size = 10,
                   epochs = 100)

classificador_json = classificadorB.to_json()

with open('classificador_binario.json', 'w') as json_file:
    json_file.write(classificador_json)
    
classificadorB.save_weights('classificador_binario_pesos.h5')

