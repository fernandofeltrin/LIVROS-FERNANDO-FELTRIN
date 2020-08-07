import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

entradas = pd.read_csv('entradas-breast.csv')
saidas = pd.read_csv('saidas-breast.csv')

classificadorA = Sequential()
classificadorA.add(Dense(units = 8,
                         activation = 'relu',
                         kernel_initializer = 'normal',
                         input_dim = 30))
classificadorA.add(Dropout(0.2))
classificadorA.add(Dense(units = 8,
                         activation = 'relu',
                         kernel_initializer = 'normal',))
classificadorA.add(Dropout(0.2))
classificadorA.add(Dense(units = 1,
                         activation = 'sigmoid'))
classificadorA.compile(optimizer = 'adam',
                       loss = 'binary_crossentropy',
                       metrics = ['binary_accuracy'])
classificadorA.fit(entradas,
                   saidas,
                   batch_size = 10,
                   epochs = 100)

objeto = np.array([[17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,
                    0.1471,0.2419,0.07871,1095,0.9053,8589,153.4,
                    0.006399,0.04904,0.05373,0.01587,0.03003,0.006193,
                    25.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,
                    0.4601,0.1189]])

previsorA = classificadorA.predict(objeto)
previsorB = (previsorA > 0.5)

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
