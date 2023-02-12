import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.preprocessing import image
import numpy as np
import glob, os

datagen = ImageDataGenerator(shear_range = 0.1,
                             zoom_range = 0.1,
                             horizontal_flip = True,
                             fill_mode = 'nearest')
                      
imagens_geradas = 'C:/Users/Fernando/Desktop/imagens_geradas/'

images = []
for img in os.listdir('C:/Users/Fernando/Desktop/CNN_CELULAS/dataset/train/EOSINOPHIL/'):
    img = os.path.join('C:/Users/Fernando/Desktop/CNN_CELULAS/dataset/train/EOSINOPHIL/', img)
    img = image.load_img(img)
    print(f'Imagem formato: {img.format} - Modo: {img.mode} - Dimensões: {img.size}')
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    images.append(img)
    print(f'{len(images)}')

images = np.vstack(images)
i = 0
for batch in datagen.flow(images, save_to_dir = imagens_geradas, save_prefix = 'EOSINOPHIL', save_format = 'jpg'):
    i += 1
    if i > 10:
        break
