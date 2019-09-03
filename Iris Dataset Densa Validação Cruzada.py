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