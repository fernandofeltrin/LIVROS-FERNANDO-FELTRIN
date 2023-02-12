import tensorflow as tf
import numpy as np

# Parametros

taxa_aprendizado = 0.005
epocas = 500
batch_size = 600

# Importando MNIST

(x_treino, y_treino), (x_teste, y_teste) = tf.keras.datasets.mnist.load_data()

base = (tf.data.Dataset.from_tensor_slices((tf.reshape(x_treino,
                                                       [-1, 784]),
                                            y_treino))
        .batch(batch_size)
        .shuffle(1000))

base = (base.map(lambda x, y:
                      (tf.divide(tf.cast(x, tf.float32), 255.0),
                       tf.reshape(tf.one_hot(y, 10), (-1, 10)))))

# Set model weights
pesos = tf.Variable(tf.zeros([784, 10]))
bias = tf.Variable(tf.zeros([10]))

# Construct model
modelo = lambda x: tf.nn.softmax(tf.matmul(x, pesos) + bias) # Softmax


# Minimize error using cross entropy
margem_erro = lambda true, pred: tf.reduce_mean(tf.reduce_sum(tf.losses.binary_crossentropy(true, pred), axis=-1))


# caculate accuracy
margem_acerto = lambda true, pred: tf.reduce_mean(tf.keras.metrics.categorical_accuracy(true, pred))


# Gradient Descent
otimizador = tf.optimizers.Adam(taxa_aprendizado)

for epoch in range(epocas):
    for i, (x_, y_) in enumerate(base):
        with tf.GradientTape() as tape:
            pred = modelo(x_)
            loss = margem_erro(y_, pred)
        acc = margem_acerto(y_, pred)
        gradientes = tape.gradient(loss, [pesos, bias])
        otimizador.apply_gradients(zip(gradientes, [pesos, bias]))
        print(f'Margem de erro: {loss.numpy()}, Margem de acertos: {acc.numpy()}')
