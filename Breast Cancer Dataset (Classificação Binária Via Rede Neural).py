# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

entradas = pd.read_csv('entradas-breast.csv')
saidas = pd.read_csv('saidas-breast.csv')

etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas,
                                                    test_size = 0.25)

classificador = Sequential()
classificador.add(Dense(units = 16,
                        activation = 'relu',
                        kernel_initializer = 'random_uniform',
                        input_dim = 30))
classificador.add(Dense(units = 1,
                        activation = 'sigmoid'))
classificador.compile(optimizer = 'adam',
                      loss = 'binary_crossentropy',
                      metrics = ['binary_accuracy'])
classificador.fit(etreino,
                  streino,
                  batch_size = 10,
                  epochs = 100)

previsor = classificador.predict(eteste)
margem_acertos = accuracy_score(steste, previsor)
matriz_confusao = confusion_matrix(steste, previsor)
resultadoMatrizConfusao = classificador.evaluate(eteste,steste)

def valCruzada():
    classificadorValCruzada = Sequential()
    classificadorValCruzada.add(Dense(units = 16,
                                      activation = 'relu',
                                      kernel_initializer = 'random_uniform',
                                      input_dim = 30))
    classificadorValCruzada.add(Dropout(0.2))
    classificadorValCruzada.add(Dense(units = 16,
                                      activation = 'relu',
                                      kernel_initializer = 'random_uniform'))
    classificadorValCruzada.add(Dense(units = 1,
                                      activation = 'sigmoid'))
    classificadorValCruzada.compile(optimizer = 'adam',
                                    loss = 'binary_crossentropy',
                                    metrics = ['binary_accuracy'])
    return classificadorValCruzada

classificador = KerasClassifier(build_fn = valCruzada,
                                epochs = 100,
                                batch_size = 10)
resultadoValCruzada = cross_val_score(estimator = classificador,
                                      X = entradas,
                                      y = saidas,
                                      cv = 10,
                                      scoring = 'accuracy')

mediaValCruzada = resultadoValCruzada.mean()
desvioValCruzada = mediaValCruzada.std()

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
