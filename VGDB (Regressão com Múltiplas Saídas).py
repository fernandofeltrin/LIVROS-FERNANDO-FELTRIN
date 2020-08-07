# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

base = pd.read_csv('games.csv')
base = base.drop('Other_Sales', axis = 1)
base = base.drop('Global_Sales', axis = 1)
base = base.drop('Developer', axis = 1)
base = base.dropna(axis = 0)
base = base.loc[base['NA_Sales'] > 1]
base = base.loc[base['EU_Sales'] > 1]
base['Name'].value_counts()
backupName = base.Name
base = base.drop('Name', axis = 1)

entradas = base.iloc[:, [0,1,2,3,7,8,9,10,11]].values
vendas_NA = base.iloc[:, 4].values
vendas_EU = base.iloc[:, 5].values
vendas_JP = base.iloc[:, 6].values

labelencoder = LabelEncoder()
entradas[:,0] = labelencoder.fit_transform(entradas[:,0])
entradas[:,2] = labelencoder.fit_transform(entradas[:,2])
entradas[:,3] = labelencoder.fit_transform(entradas[:,3])
entradas[:,8] = labelencoder.fit_transform(entradas[:,8])
onehotencoder = OneHotEncoder(categorical_features = [0,2,3,8])
entradas = onehotencoder.fit_transform(entradas).toarray()

camada_entrada = Input(shape = (61, ))
camada_oculta1 = Dense(units = 32,
                       activation = 'sigmoid')(camada_entrada)
camada_oculta2 = Dense(units = 32,
                       activation = 'sigmoid')(camada_oculta1)
camada_saida1 = Dense(units = 1,
                      activation = 'linear')(camada_oculta2)
camada_saida2 = Dense(units = 1,
                      activation = 'linear')(camada_oculta2)
camada_saida3 = Dense(units = 1,
                      activation = 'linear')(camada_oculta2)

regressor = Model(inputs = camada_entrada,
                  outputs = [camada_saida1,
                             camada_saida2,
                             camada_saida3])
regressor.compile(optimizer = 'adam',
                  loss = 'mse')
regressor.fit(entradas,
              [vendas_NA, vendas_EU, vendas_JP],
              epochs = 5000,
              batch_size = 100)
#previsor
previsao_NA, previsao_EU, previsao_JP = regressor.predict(entradas)

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
