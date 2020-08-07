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

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
