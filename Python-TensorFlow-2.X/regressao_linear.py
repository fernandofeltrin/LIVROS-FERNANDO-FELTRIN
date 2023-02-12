import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('datasets/data.csv')

print(data.head())

plt.scatter(data['X'], data['y'])
plt.show()

modelo = tf.keras.Sequential()
modelo.add(tf.keras.layers.Dense(1, input_shape=[1]))
modelo.compile(loss = 'mean_squared_error',
               optimizer=tf.keras.optimizers.Adam(0.01))

print(modelo.summary())

modelo.fit(data['X'], data['y'], epochs=1000)

data['Previsao'] = modelo.predict(data['X'])

plt.scatter(data['X'], data['y'])
plt.plot(data['X'], data['Previsao'], color='r')
plt.show()
