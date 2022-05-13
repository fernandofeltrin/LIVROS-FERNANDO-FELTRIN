!pip install -q -U keras-tuner

from google.colab import drive

drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
import numpy as np
import matplotlib.pyplot as plt

(img_train, label_train), (img_test, label_test) = keras.datasets.fashion_mnist.load_data()

def model_builder(hp):
  hp_units = hp.Int('units', min_value = 16, max_value = 512, step = 16)
  hp_units_int = hp.Int('units_int', min_value = 64, max_value = 512, step = 64)
  activation = hp.Choice("activation", ["relu", "tanh"])
  activation_final = hp.Choice("activation_final", ["sigmoid", "softmax"])
  hp_learning_rate = hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])

  model = keras.Sequential()
  model.add(keras.layers.Flatten(input_shape = (28, 28)))
  model.add(keras.layers.Dense(units = hp_units,
                               activation = activation)) 
  model.add(keras.layers.Dense(units = hp_units_int,
                               activation = activation))
  if hp.Boolean("dropout"):
        model.add(Dropout(rate = 0.25))
  model.add(keras.layers.Dense(units = hp_units_int,
                               activation = activation))
  model.add(keras.layers.Dense(units = 10,
                               activation = activation_final))

  model.compile(optimizer = keras.optimizers.Adam(learning_rate = hp_learning_rate),
                loss = keras.losses.sparse_categorical_crossentropy,
                metrics = 'accuracy')

  return model

tuner = kt.Hyperband(model_builder,
                     objective = 'val_accuracy',
                     max_epochs = 100,
                     factor = 3,
                     directory = '/content/drive/MyDrive/Colab Notebooks/',
                     project_name = 'KerasTuner')

learning_rate = ReduceLROnPlateau(monitor = 'accuracy',
                                  factor = 0.2,
                                  patience = 2,
                                  min_lr = 0.000001,
                                  verbose = 1)

earlystop = EarlyStopping(monitor='loss',
                          min_delta = 0,
                          patience = 5,
                          verbose = 1,
                          mode = 'min')

tuner.search(img_train,
             label_train,
             steps_per_epoch = 100,
             epochs = 100,
             validation_split = 0.25,
             callbacks = [earlystop])

best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

best_model = tuner.get_best_models()[0]
#best_model = tuner.get_best_models(num_models=3)

print('Melhor nº de neurônios para a camada de entrada: {0}'.format(best_hps.get('units')))
print('Melhor método de ativação para as camadas intermediárias: {0}'.format(best_hps.get('activation')))
print('Melhor método de ativação para a camada de saída: {0}'.format(best_hps.get('activation_final')))
print('Melhor taxa de aprendizado definida: {0}'.format(best_hps.get('learning_rate')))

best_model.summary()

tuner.results_summary()

model = tuner.hypermodel.build(best_hps)

history = model.fit(img_train,
                    label_train,
                    steps_per_epoch = 100,
                    epochs = 100,
                    validation_split = 0.2,
                    shuffle = True)

evaluation = model.evaluate(img_test,
                            label_test)

print("[loss, accuracy]:", evaluation)
