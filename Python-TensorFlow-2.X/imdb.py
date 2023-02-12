import tensorflow as tf
from tensorflow.keras.datasets import imdb

(X_treino, y_treino), (X_teste, y_teste) = imdb.load_data(num_words=20000)

print(X_treino.shape)
print(X_treino)
print(X_treino[44])

print(y_treino)

print(len(X_treino[44]))

X_treino = tf.keras.preprocessing.sequence.pad_sequences(X_treino,
                                                         maxlen = 100)
X_teste = tf.keras.preprocessing.sequence.pad_sequences(X_teste,
                                                        maxlen = 100)

print(len(X_treino[44]))


modelo = tf.keras.Sequential()

modelo.add(tf.keras.layers.Embedding(input_dim = 20000,
                                     output_dim = 128,
                                     input_shape = (X_treino.shape[1], )))
modelo.add(tf.keras.layers.LSTM(units = 128,
                                activation = 'tanh'))
modelo.add(tf.keras.layers.Dense(units = 1,
                                 activation= 'sigmoid'))
modelo.compile(optimizer = 'rmsprop',
               loss = 'binary_crossentropy',
               metrics = ['accuracy'])

print(modelo.summary())

modelo.fit(X_treino,
           y_treino,
           epochs = 50,
           batch_size = 128,
           use_multiprocessing = True)


