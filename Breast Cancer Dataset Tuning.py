import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

entradas = pd.read_csv('entradas-breast.csv')
saidas = pd.read_csv('saidas-breast.csv')

def tuningClassificador(optimizer, loss, kernel_initializer, activation, neurons):
    classificadorTuning = Sequential()
    classificadorTuning.add(Dense(units = neurons,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer,
                                  input_dim = 30))
    classificadorTuning.add(Dropout(0.2))
    classificadorTuning.add(Dense(units = neurons,
                                  activation = activation,
                                  kernel_initializer = kernel_initializer))
    classificadorTuning.add(Dropout(0.2))
    classificadorTuning.add(Dense(units = 1,
                                  activation = 'sigmoid'))
    classificadorTuning.compile(optimizer = optimizer,
                                loss = loss,
                                metrics = ['binary_accuracy'])
    return classificadorTuning

classificadorTunado = KerasClassifier(build_fn = tuningClassificador)
parametros = {'batch_size':[10,30],
              'epochs':[50,100],
              'optimizer':['adam','sgd'],
              'loss':['binary_crossentropy','hinge'],
              'kernel_initializer':['random_uniform','normal'],
              'activation':['relu','tanh'],
              'neurons':[10,8]}
tunagem = GridSearchCV(estimator = classificadorTunado,
                           param_grid = parametros,
                           scoring = 'accuracy')
tunagem = tunagem.fit(entradas,saidas)
melhores_parametros = tunagem.best_params_
melhor_margem_precisao = tunagem.best_score_
