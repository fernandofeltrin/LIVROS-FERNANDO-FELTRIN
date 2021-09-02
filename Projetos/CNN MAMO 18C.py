'''
Modelo de rede neural artificial convolucional treinado para interpretar mamografias e a partir das mesmas
classificar os achados, de acordo com a probabilidade definida, em 18 categorias diferentes, sendo três 
normais e 16 patológicas (Abscesso, Calcificações Pontuais, Calcificações Vasculares, Câncer e/ou Metástases,
Carcinoma, Cisto, Fibroadenoma, Fibromatose, Hamartoma, Hemangioma, Linfoma, Lipoma, Mastite e Sarcoma).
Modelo treinado com 4068 imagens físicas e 32.544 imagens aumentadas sinteticamente, atingindo 94% de margem
de acertos em classificação.
'''
################################################################################################################

#!pip install keras-sequential-ascii
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.preprocessing import image
from keras.callbacks import ReduceLROnPlateau
from keras_sequential_ascii import keras2ascii
from tensorflow.keras import initializers

gerador_treino = ImageDataGenerator(rescale = 1.0/255,
                                    rotation_range = 10,
                                    horizontal_flip = True,
                                    vertical_flip = True,
                                    shear_range = 0.5,
                                    width_shift_range=0.1,
                                    height_shift_range = 0.1,
                                    zoom_range = 0.5)
gerador_teste = ImageDataGenerator(rescale = 1.0/255)

base_treino = gerador_treino.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/dataset/train',
                                                 target_size = (512,512),
                                                 batch_size = 8,
                                                 class_mode = 'categorical')
base_teste = gerador_teste.flow_from_directory('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/dataset/test',
                                               target_size = (512,512),
                                               batch_size = 8,
                                               class_mode = 'categorical')

classificadorx2 = Sequential()
classificadorx2.add(Conv2D(256,
                         kernel_size = (3,3),
                         padding = 'same',
                         input_shape = (512,512,3),
                         activation = 'relu'))
classificadorx2.add(BatchNormalization())
classificadorx2.add(MaxPooling2D(pool_size = (2,2)))
classificadorx2.add(Conv2D(256,
                         kernel_size = (3,3),
                         padding = 'valid',
                         activation = 'relu'))
classificadorx2.add(BatchNormalization())
classificadorx2.add(MaxPooling2D(pool_size = (2,2)))
classificadorx2.add(Conv2D(256,
                         kernel_size = (3,3),
                         padding = 'same',
                         activation = 'relu'))
classificadorx2.add(BatchNormalization())
classificadorx2.add(MaxPooling2D(pool_size = (2,2)))
classificadorx2.add(Conv2D(256,
                         kernel_size = (3,3),
                         padding = 'valid',
                         activation = 'relu'))
classificadorx2.add(BatchNormalization())
classificadorx2.add(MaxPooling2D(pool_size = (2,2)))
classificadorx2.add(Conv2D(256,
                         kernel_size = (3,3),
                         padding = 'same',
                         activation = 'relu'))
classificadorx2.add(BatchNormalization())
classificadorx2.add(MaxPooling2D(pool_size = (2,2)))
classificadorx2.add(Flatten())
classificadorx2.add(Dense(units = 256,
                         activation = 'relu',
                         use_bias=True,
                         kernel_initializer=initializers.RandomNormal(stddev=0.01),
                         bias_initializer='zeros'))
classificadorx2.add(Dropout(0.1))
classificadorx2.add(Dense(units = 256,
                        activation = 'relu',
                        use_bias=True,
                        kernel_initializer=initializers.RandomNormal(stddev=0.01),
                        bias_initializer='zeros'))
classificadorx2.add(Dropout(0.1))
classificadorx2.add(Dense(units = 128,
                         activation = 'relu',
                         use_bias=True,
                         kernel_initializer=initializers.RandomNormal(stddev=0.01),
                         bias_initializer='zeros'))
classificadorx2.add(Dropout(0.05))
classificadorx2.add(Dense(units = 18,
                        activation = 'sigmoid'))
classificadorx2.compile(optimizer = 'Adam',
                      loss = 'categorical_crossentropy',
                      metrics = ['accuracy'])

classificadorx2.summary()

keras2ascii(classificadorx2)

epochs = 500

learning_rate = ReduceLROnPlateau(monitor='accuracy',
                                  factor=0.1,
                                  patience=2,
                                  verbose=1,
                                  mode="auto",
                                  min_delta=0.001,
                                  cooldown=0,
                                  min_lr=0.01)

h_x2 = classificadorx2.fit(base_treino,
                           steps_per_epoch = 100,
                           epochs = epochs,
                           validation_data = base_teste,
                           callbacks = [learning_rate],
                           verbose = 'auto',
                           validation_steps = 2)

print('Precisão do treino: {0:.2f}%'.format(max(h_x2.history['accuracy']) * 100))
print('Precisão da validacao: {0:.5f}'.format(max(h_x2.history['val_accuracy']) * 100))

classificadorx2.save_weights('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/CNN_MAMO_18C_weights.h5')
classificadorx2.save('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/CNN_MAMO_18C_model.h5')

plt.rcParams['figure.figsize'] = (20.0, 8.0)
plt.plot(h_x2.history['accuracy'])
plt.ylim(0.01, 0.99)
plt.legend(['Modelo CNN Mamo'], loc = 'lower right', fontsize = 'xx-large')
plt.xlabel('Epocas de processamento', fontsize=16)
plt.ylabel('Acuracia', fontsize=16)
plt.title('Avaliacao do Modelo', fontsize=18)
plt.show()

######################## TESTE EM IMAGENS ####################################

from keras.preprocessing import image
from keras.models import Sequential, load_model
from keras.preprocessing.image import load_img
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt

model = Sequential()
model = load_model('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/CNN_MAMO_18C_model.h5')


img_teste = load_img('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/dataset/test/Calcificacoes Vasculares/breast-benign-type-calcification(1).PNG',
                           target_size = (512, 512))

img_plot = PIL.Image.open('/content/drive/MyDrive/Colab Notebooks/CNN_MAMO_18C/dataset/test/Calcificacoes Vasculares/breast-benign-type-calcification(1).PNG')

plt.figure(figsize=(8,8))
plt.imshow(img_plot)
plt.show()

img_teste = image.img_to_array(img_teste)
img_teste /= 255
img_teste = np.expand_dims(img_teste, axis = 0)

resultado_teste = model.predict(img_teste)

resultado_final = resultado_teste

print('ACHADOS RADIOLÓGICOS: ')
if resultado_final[0,0] > 0.7:
    print('Abscesso')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,1]*100 - 5))
if resultado_final[0,1] > 0.7:
    print('Calcificações Pontuais')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,1]*100 - 5))
if resultado_final[0,2] > 0.7:
    print('Calcificações Vasculares')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,2]*100 - 5))
if resultado_final[0,3] > 0.7:
    print('Tumor Genérico ou Metástase')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,3]*100 - 5))
if resultado_final[0,4] > 0.7:
    print('Carcinoma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,4]*100 - 5))
if resultado_final[0,5] > 0.7:
    print('Cisto')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,5]*100 - 5))
if resultado_final[0,6] > 0.7:
    print('Fibroadenoma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,6]*100 - 5))
if resultado_final[0,7] > 0.7:
    print('Fibromatose')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,7]*100 - 5))
if resultado_final[0,8] > 0.7:
    print('Hamartoma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,8]*100 - 5))
if resultado_final[0,9] > 0.7:
    print('Hemangioma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,8]*100 - 5))
if resultado_final[0,10] > 0.7:
    print('Linfoma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))
if resultado_final[0,11] > 0.7:
    print('Lipoma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))
if resultado_final[0,12] > 0.7:
    print('Mastite')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5)) 
if resultado_final[0,13] > 0.7:
    print('Necrose')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))
if resultado_final[0,14] > 0.7:
    print('Normal')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))
if resultado_final[0,15] > 0.7:
    print('Normal Densa')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))
if resultado_final[0,16] > 0.7:
    print('Normal Lactante')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))
if resultado_final[0,17] > 0.7:
    print('Sarcoma')
    print('Probabilidade: {0:.2f}%'.format(resultado_final[0,10]*100 - 5))                           
else:
    pass

print(f'Sugere-se exame citopatológico para confirmar ou descartar o diagnóstico sugerido.')

