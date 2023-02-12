import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.layers import Conv1D
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

print(tf.__version__)

data = datasets.load_breast_cancer()

print(data.DESCR)

X = pd.DataFrame(data = data.data,
                 columns=data.feature_names)
print(X.head())

y = data.target
print(y)

print(data.target_names)

print(X.shape)

X_treino, X_teste, y_treino, y_teste = train_test_split(X,
                                                        y,
                                                        test_size = 0.2,
                                                        random_state = 0,
                                                        stratify = y)

print(X_treino.shape)
print(X_teste.shape)

escalonador = StandardScaler()

X_treino = escalonador.fit_transform(X_treino)
X_teste = escalonador.transform(X_teste)

X_treino = X_treino.reshape(455, 30, 1)
X_teste = X_teste.reshape(114, 30, 1)

epochs = 500

modelo = Sequential()
modelo.add(Conv1D(filters=32,
                  kernel_size=2,
                  activation='relu',
                  input_shape = (30, 1)))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.2))

modelo.add(Conv1D(filters=64,
                  kernel_size=2,
                  activation='relu'))
modelo.add(BatchNormalization())
modelo.add(Dropout(0.5))

modelo.add(Flatten())
modelo.add(Dense(64,
                 activation='relu'))
modelo.add(Dropout(0.5))

modelo.add(Dense(1,
                 activation='sigmoid'))

modelo.summary()

modelo.compile(optimizer=Adam(lr=0.001),
               loss ='binary_crossentropy',
               metrics=['accuracy'])

history = modelo.fit(X_treino,
                     y_treino,
                     epochs=epochs,
                     validation_data=(X_teste, y_teste),
                     verbose=1)

def plot_learningCurve(history, epoch):
  epoca = range(1, epoch + 1)
  plt.plot(epoca, history.history['accuracy'])
  plt.plot(epoca, history.history['val_accuracy'])
  plt.title('Precisão do Modelo')
  plt.ylabel('Margem de Acertos')
  plt.xlabel('Épocas')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

  plt.plot(epoca, history.history['loss'])
  plt.plot(epoca, history.history['val_loss'])
  plt.title('Margem de Erro')
  plt.ylabel('Erros')
  plt.xlabel('Épocas')
  plt.legend(['Train', 'Val'], loc='upper left')
  plt.show()

print(history.history)

curva_aprendizado = plot_learningCurve(history, epochs)
