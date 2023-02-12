import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from keras.wrappers.scikit_learn import KerasClassifier

base = pd.read_csv('iris.csv')
entradas = base.iloc[:, 0:4].values
saidas = base.iloc[:, 4].values

labelencoder = LabelEncoder()
saidas = labelencoder.fit_transform(saidas)
saidas_dummy = np_utils.to_categorical(saidas)

etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas_dummy,
                                                    test_size = 0.25)
classificador = Sequential()
classificador.add(Dense(units = 4,
                        activation = 'relu',
                        input_dim = 4))
classificador.add(Dense(units = 4,
                        activation = 'relu'))
classificador.add(Dense(units = 3,
                        activation = 'softmax'))
classificador.compile(optimizer = 'adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['categorical_accuracy'])
classificador.fit(etreino,
                  streino,
                  batch_size = 10,
                  epochs = 1000)

avalPerformance = classificador.evaluate(eteste, steste)
previsoes = classificador.predict(eteste)
previsoesVF = (previsoes > 0.5)
steste2 = [np.argmax(t) for t in steste]
previsoes2 = [np.argmax(t) for t in previsoes]
matrizConfusao = confusion_matrix(previsoes2, steste2)

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

classificadorValCruzada = KerasClassifier(build_fn = valCruzada,
                                          epochs = 1000,
                                          batch_size = 10)

resultadosValCruzada = cross_val_score(estimator = classificadorValCruzada,
                                      X = entradas,
                                      y = saidas,
                                      cv = 10,
                                      scoring = 'accuracy')
media = resultadosValCruzada.mean()
desvio = resultadosValCruzada.std()
