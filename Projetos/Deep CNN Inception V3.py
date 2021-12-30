!pip install -q -U watermark
!pip install -q -U plaidml # biblioteca que permite criar modelos portáteis
!pip install -q plaidml.keras

import pandas as pd
import numpy as np
import os
import sys
import warnings
import sklearn
import itertools
import plaidml
import plaidml.keras
import tensorflow
import keras
import matplotlib.pyplot as plt
import matplotlib.cbook
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import shuffle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, MaxPool2D
from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling2D, Concatenate
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau
from keras.utils.np_utils import to_categorical
from keras.applications import InceptionV3
from keras.applications.inception_v3 import preprocess_input
from tensorflow.python.client import device_lib

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'

import tensorflow as tf

print(device_lib.list_local_devices())
print(len(tf.config.experimental.list_physical_devices('GPU')))

# Carregando a base de dados

dataset = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/Datasets/ISIC_2019_Training_GroundTruth.csv')

print(dataset)

print(f'Melanocytis nevus: {dataset.NV.value_counts()[1]} imagens')
print(f'Melanoma: {dataset.MEL.value_counts()[1]} imagens')
print(f'Basal cell carcinoma: {dataset.BCC.value_counts()[1]} imagens')
print(f'Actinic keratosis: {dataset.AK.value_counts()[1]} imagens')
print(f'Benign keratosis: {dataset.BKL.value_counts()[1]} imagens')
print(f'Dermatofibroma: {dataset.DF.value_counts()[1]} imagens')
print(f'Vascular lesion: {dataset.VASC.value_counts()[1]} imagens')
print(f'Squamous cell carcinoma: {dataset.SCC.value_counts()[1]} imagens')

# Tratamento dos dados

dataset.rename(columns = {'image':'image_id'}, inplace = True)

print(dataset.head(10))

dataset['tipo_cancer'] = dataset['UNK']

for i, j in dataset.iterrows():
  dataset['tipo_cancer'][i] = (j == 1).idxmax(axis = 1) 

print(dataset.head(10)) 

dataset = dataset.drop(['MEL', 'NV', 'AK', 'BCC', 'BKL', 'DF', 'VASC', 'SCC', 'UNK'], axis = 1)

print(dataset.shape)
print(dataset.head(10))

print(dataset['tipo_cancer'].value_counts()) 

# Label encoding

dataset['tipo_cancer_idx'] = pd.Categorical(dataset['tipo_cancer']).codes 

print(dataset.head(10))

# Preparação das imagens

imagens = os.listdir('/content/drive/MyDrive/Colab Notebooks/Datasets/ISIC_Training_Input/')

print(imagens[1:10])
print(f'Total de amostras: {len(imagens)}')

def extrai(imagem):
  arquivo = imagem + '.jpg'
  return os.path.join('/content/drive/MyDrive/Colab Notebooks/Datasets/ISIC_Training_Input/', arquivo) 

dataset['imagem'] = dataset['image_id'].apply(extrai)

print(dataset.head(10))

# Balanceamento das classes

print(f'Total de amostras: {dataset.shape[0]}')

print(dataset.tipo_cancer.value_counts())

print(f'{dataset.tipo_cancer.value_counts() / dataset.shape[0] * 100}')

# Undersampling

df_nv = dataset[dataset['tipo_cancer'] == 'NV']
df_nv = shuffle(df_nv)
df_nv = df_nv[0:2000]
df_nv = df_nv.reset_index(drop = True)

df_mel = dataset[dataset['tipo_cancer'] == 'MEL']
df_mel = shuffle(df_mel)
df_mel = df_mel[0:2000]
df_mel = df_mel.reset_index(drop = True)

df_bcc = dataset[dataset['tipo_cancer'] == 'BCC']
df_bcc = shuffle(df_bcc)
df_bcc = df_bcc[0:2000]
df_bcc = df_bcc.reset_index(drop = True)

df_bkl = dataset[dataset['tipo_cancer'] == 'BKL']
df_bkl = shuffle(df_bkl)
df_bkl = df_bkl[0:2000]
df_bkl = df_bkl.reset_index(drop = True)

df_ak = dataset[dataset['tipo_cancer'] == 'AK']
df_ak = shuffle(df_ak)
df_ak = df_ak[0:2000]
df_ak = df_ak.reset_index(drop = True)

df_scc = dataset[dataset['tipo_cancer'] == 'SCC']
df_scc = shuffle(df_scc)
df_scc = df_scc[0:2000]
df_scc = df_scc.reset_index(drop = True)

df_vasc = dataset[dataset['tipo_cancer'] == 'VASC']
df_vasc = shuffle(df_vasc)
df_vasc = df_vasc[0:2000]
df_vasc = df_vasc.reset_index(drop = True)

df_df = dataset[dataset['tipo_cancer'] == 'DF']
df_df = shuffle(df_df)
df_df = df_df[0:2000]
df_df = df_df.reset_index(drop = True)

dataset_final = pd.concat([df_nv, df_mel, df_bcc, df_bkl, df_ak, df_scc, df_vasc, df_df])

print(f'Número final de amostras: {dataset_final.shape[0]}')

print(f'TIPO    Nº de Amostras')
print(dataset_final.tipo_cancer.value_counts())

# Preparando as bases para treino e teste

keys = list(dataset_final.columns.values)

print(keys)

# Amostras para treino

df_treino = dataset_final

print(dataset_final.shape)

df_treino = shuffle(df_treino)
df_treino = df_treino.reset_index(drop = True)

i1 = df_treino.set_index(keys).index

print(i1[1:10])

# Amostras para teste

df_nv = dataset_final[dataset_final['tipo_cancer'] == 'NV']
df_nv = shuffle(df_nv)
df_nv = df_nv[0:200]
df_nv = df_nv.reset_index(drop = True)

df_mel = dataset_final[dataset_final['tipo_cancer'] == 'MEL']
df_mel = shuffle(df_mel)
df_mel = df_mel[0:200]
df_mel = df_mel.reset_index(drop = True)

df_bcc = dataset_final[dataset_final['tipo_cancer'] == 'BCC']
df_bcc = shuffle(df_bcc)
df_bcc = df_bcc[0:200]
df_bcc = df_bcc.reset_index(drop = True)

df_bkl = dataset_final[dataset_final['tipo_cancer'] == 'BKL']
df_bkl = shuffle(df_bkl)
df_bkl = df_bkl[0:200]
df_bkl = df_bkl.reset_index(drop = True)

df_ak = dataset_final[dataset_final['tipo_cancer'] == 'AK']
df_ak = shuffle(df_ak)
df_ak = df_ak[0:200]
df_ak = df_ak.reset_index(drop = True)

df_df = dataset_final[dataset_final['tipo_cancer'] == 'DF']
df_df = shuffle(df_df)
df_df = df_df[0:200]
df_df = df_df.reset_index(drop = True)

df_vasc = dataset_final[dataset_final['tipo_cancer'] == 'VASC']
df_vasc = shuffle(df_vasc)
df_vasc = df_vasc[0:200]
df_vasc = df_vasc.reset_index(drop = True)

df_scc = dataset_final[dataset_final['tipo_cancer'] == 'SCC']
df_scc = shuffle(df_scc)
df_scc = df_scc[0:200]
df_scc = df_scc.reset_index(drop = True)

df_teste = pd.concat([df_nv, df_mel, df_bcc, df_bkl, df_ak, df_scc, df_vasc, df_df])

df_teste = shuffle(df_teste)
df_teste = df_teste.reset_index(drop = True)

print(df_teste.tipo_cancer.value_counts())
print(df_teste.head(10))

print(f'Dataset completo possui {dataset_final.shape[0]} amostras')
print(f'Dataset final para teste possui {df_teste.shape[0]} amostras')

i2 = df_teste.set_index(keys).index

print(i2[1:10])

# Finalizando as bases de dados

df_treino = df_treino[~i1.isin(i2)] 

print(f'Dataset completo possui {dataset_final.shape[0]} amostras')
print(f'Dataset final para treino possui {df_treino.shape[0]} amostras')
print(f'Dataset final para teste possui {df_teste.shape[0]} amostras')

teste_duplicidade = df_teste['imagem'][10]

if (teste_duplicidade in df_treino.imagem.values) == False:
  print(f'Amostras únicas separadas com sucesso para treino e para teste')
else:
  print(f'Existem amostras duplicadas nas bases de dados para treino e para teste')

# Redimensionamento das imagens

caminho_imagem = df_teste['imagem'][10]
imagem = Image.open(caminho_imagem)

print(f'Objeto: {imagem}')
print(f'Imagem amostra: {caminho_imagem[-16:]}')
print(f'Largura da imagem: {imagem.width} pixels')
print(f'Altura da imagem: {imagem.height} pixels')

df_treino['imagem_res'] = df_treino['imagem'].map(lambda x: np.asarray(Image.open(x).resize((128, 128))))
df_teste['imagem_res'] = df_teste['imagem'].map(lambda x: np.asarray(Image.open(x).resize((128, 128))))

df_treino.set_index('image_id', inplace = True)
df_teste.set_index('image_id', inplace = True)

print(df_treino.head(10))

print(df_teste.head(10))

caminho_imagem2 = df_teste['imagem_res'][10]
imagem = Image.open(caminho_imagem2)

print(f'Objeto: {imagem}')
print(f'Imagem amostra: {caminho_imagem2[-16:]}')
print(f'Largura da imagem: {imagem.width} pixels')
print(f'Altura da imagem: {imagem.height} pixels')

print(f'Formato final de cada imagem: {df_treino.imagem_res[0].shape}')

# Plotando as imagens

n_samples = 8

fig, m_axs = plt.subplots(8, 8, figsize = (3*5, 3*5))

for n_axs, (type_name, type_rows) in zip(m_axs, df_treino.sort_values(['tipo_cancer']).groupby('tipo_cancer')):
  n_axs[0].set_title(type_name)
  for c_ax, (_, c_row) in zip(n_axs, type_rows.sample(n_samples, random_state = 1234).iterrows()):
    c_ax.imshow(c_row['imagem_res'])
    c_ax.axis('off')

# Divisão das bases de dados em treino e validação

df_treino = shuffle(df_treino)
df_teste = shuffle(df_teste)

y_treino = df_treino['tipo_cancer_idx']
y_teste = df_teste['tipo_cancer_idx']

print(y_treino.shape)
print(y_teste.shape)

print(y_treino.value_counts())
print(y_teste.value_counts())

y_treino = to_categorical(y_treino,
                            num_classes = 8)
y_teste = to_categorical(y_teste,
                           num_classes = 8)

print(f'Dataset para treino possui {y_treino.shape[0]} amostras divididas em {y_treino.shape[1]} categorias previsoras')
print(f'Dataset para treino possui {y_teste.shape[0]} amostras divididas em {y_teste.shape[1]} categorias previsoras')

# Preparando os dados de entrada

x_treino = np.asarray(df_treino['imagem_res'].tolist())
x_teste = np.asarray(df_teste['imagem_res'].tolist())

x_treino_mean = np.mean(x_treino)
x_treino_std = np.std(x_treino)
x_teste_mean = np.mean(y_teste)
x_teste_std = np.std(y_teste)

x_treino = (x_treino - x_treino_mean) / x_treino_std
x_teste = (x_teste - x_teste_mean) / x_teste_std

la = x_treino.shape[1]
al = x_treino.shape[2]

print(f'Dataset para treino possui {x_treino.shape[0]} imagens com {la, al} pixels e {x_treino.shape[3]} canais de cor')
print(f'Dataset para teste possui {x_teste.shape[0]} imagens com {la, al} pixels e {x_teste.shape[3]} canais de cor')

x_treino, x_valid, y_treino, y_valid = train_test_split(x_treino,
                                                        y_treino,
                                                        test_size = 0.2)

print(f'Número de amostras separadas para treino: {x_treino.shape[0]}')
print(f'Número de amostras separadas para validação: {x_valid.shape[0]}')

x_treino = x_treino.reshape(x_treino.shape[0], *(128, 128, 3))
x_valid = x_valid.reshape(x_valid.shape[0], *(128, 128, 3))

# Construção do modelo

modelo_base = tf.keras.applications.InceptionV3(weights = 'imagenet', 
                                                include_top = False, 
                                                input_shape = (128, 128, 3))

input_shape = (128, 128, 3)
num_classes = 8 
batch_size = 32
lr_param = 0.01

print(modelo_base.summary())

add_model = Sequential()
add_model.add(modelo_base)
add_model.add(GlobalAveragePooling2D())
add_model.add(Dropout(0.10))
add_model.add(BatchNormalization())
add_model.add(Dropout(0.10))
add_model.add(Dense(units = 8,
                    activation = 'sigmoid'))

modelo_final = add_model

print(modelo_final.summary())

otimizador = Adam(learning_rate = 0.001)

modelo_final.compile(optimizer = otimizador, 
                     loss = "categorical_crossentropy", 
                     metrics = ["accuracy"])

reduz_taxa_aprendizado = ReduceLROnPlateau(monitor = 'accuracy', 
                                           patience = 2, 
                                           verbose = 1, 
                                           factor = 0.2, 
                                           min_lr = 0.001,
                                           min_delta = 0.001,
                                           mode = "auto",
                                           cooldown = 0)

datagen = ImageDataGenerator(featurewise_center = False,  
                             samplewise_center = False,  
                             featurewise_std_normalization = False,  
                             samplewise_std_normalization = False, 
                             rotation_range = 5,
                             shear_range = 0.1,
                             zoom_range = 0.1, 
                             width_shift_range = 0.05,  
                             height_shift_range = 0.05,  
                             horizontal_flip = False,  
                             vertical_flip = False)

classificador = modelo_final.fit(datagen.flow(x_treino, 
                                        y_treino, 
                                        batch_size = 32),
                           epochs = 1000, 
                           validation_data = (x_valid, y_valid),
                           verbose = 1, 
                           steps_per_epoch = 100,
                           validation_steps = 25,
                           callbacks = [reduz_taxa_aprendizado])

# Salvando o modelo

modelo_final.save("/content/drive/MyDrive/Colab Notebooks/Datasets/modelo_cancer_pele.h5")

# Avaliação do modelo

plt.plot(classificador.history['accuracy'])
plt.plot(classificador.history['val_accuracy'])
plt.title('Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Epoca')
plt.legend(['Treino', 'Validação'], loc = 'upper left')
plt.show()

plt.plot(classificador.history['loss'])
plt.plot(classificador.history['val_loss'])
plt.title('Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Treino', 'Validação'], loc = 'upper left')
plt.show()
