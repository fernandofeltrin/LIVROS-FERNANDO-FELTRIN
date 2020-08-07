# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

base = pd.read_csv('iris.csv')

entradas = base.iloc[:, 0:4].values
saidas = base.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder

labelencoder = LabelEncoder()
saidas = labelencoder.fit_transform(saidas)

def valCruzada():
    classificadorValCruzada = Sequential()
    classificadorValCruzada.add(Dense(units = 4,
                            activation = 'relu',
                            input_dim = 4))
    classificadorValCruzada.add(Dense(units = 4,
                            activation = 'relu'))
    classificadorValCruzada.add(Dense(units = 3,
                            activation = 'softmax'))
    classificadorValCruzada.compile(optimizer = 'adam',
                          loss = 'categorical_crossentropy',
                          metrics = ['categorical_accuracy'])
    return classificadorValCruzada

from keras.wrappers.scikit_learn import KerasClassifier

classificadorValCruzada = KerasClassifier(buind_fn = valCruzada,
                                          epochs = 1000,
                                          batch_size = 10)

resultadoValCruzada = cross_val_score(estimator = classificadorValCruzada,
                                      X = entradas,
                                      y = saidas,
                                      cv = 10,
                                      scoring = 'accuracy')

media = resultadosValCruzada.mean()
desvio = resultadosValCruzada.std()

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
