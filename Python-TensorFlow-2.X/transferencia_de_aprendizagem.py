# pip install tqdm
# Download dataset https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

import os
import zipfile
import tensorflow as tf
from tqdm import tqdm_notebook
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Desempacotando a base de dados

raiz_dataset = "./cats_and_dogs_filtered.zip"
arquivo_zip = zipfile.ZipFile(file=raiz_dataset, mode="r")
arquivo_zip.extractall("./")
arquivo_zip.close()


# Configurando os caminhos/paths

diretorio = "./cats_and_dogs_filtered"
dir_treino = os.path.join(diretorio, "train")
dir_validacao = os.path.join(diretorio, "validation")


# Carregando o modelo pré-treinado (MobileNetV2)

formato_imagem = (128, 128, 3)
modelo_base = tf.keras.applications.MobileNetV2(input_shape = formato_imagem,
                                                include_top = False,
                                                weights = "imagenet")
print(modelo_base.summary())


# Congelando o modelo base

modelo_base.trainable = False


# Definindo o cabeçalho personalizado da rede neural

print(modelo_base.output)

camada_intermediaria = tf.keras.layers.GlobalAveragePooling2D()(modelo_base.output)

print(camada_intermediaria)

camada_previsora = tf.keras.layers.Dense(units = 1,
                                         activation = "sigmoid")(camada_intermediaria)


# Definindo o modelo

modelo = tf.keras.models.Model(inputs = modelo_base.input,
                               outputs = camada_previsora)
print(modelo.summary())

# Compilando o modelo

modelo.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001),
               loss="binary_crossentropy",
               metrics = ["accuracy"])


# Criando geradores de dados (Data Generators)

img_gen_treino = ImageDataGenerator(rescale=1 / 255.)
img_gen_validacao = ImageDataGenerator(rescale=1 / 255.)

gerador_treino = img_gen_treino.flow_from_directory(dir_treino,
                                                    target_size=(128,128),
                                                    batch_size=128,
                                                    class_mode="binary")
gerador_validacao = img_gen_treino.flow_from_directory(dir_validacao,
                                                       target_size=(128,128),
                                                       batch_size=128,
                                                       class_mode="binary")


# Treinando o modelo

modelo.fit_generator(gerador_treino,
                     epochs=5,
                     validation_data=gerador_validacao)


# Avaliando o modelo

margem_erro, margem_acerto = modelo.evaluate_generator(gerador_validacao)
print(margem_acerto)






# Fine Tuning

modelo_base.trainable = True # descongelando a parte inutilizada da base
len(modelo_base.layers)
fine_tuning = 100
for camada in modelo_base.layers[:fine_tuning]:
  camada.trainable = False

modelo.compile(optimizer=tf.keras.optimizers.RMSprop(lr = 0.0001),
               loss="binary_crossentropy",
               metrics=["accuracy"])
modelo.fit_generator(gerador_treino,
                     epochs=5,
                     validation_data=gerador_validacao)

margem_erro, margem_acerto = modelo.evaluate_generator(gerador_validacao)
print(margem_acerto)
