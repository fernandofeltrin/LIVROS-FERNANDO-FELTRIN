# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

import pandas as pd
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler
from pylab import pcolor, colorbar

base = pd.read_csv('wines.csv')
entradas = base.iloc[:, 1:14].values
saidas = base.iloc[:, 0].values

normalizador = MinMaxScaler(feature_range = (0,1))
entradas = normalizador.fit_transform(entradas)

som = MiniSom(x = 8,
              y = 8,
              input_len = 13,
              sigma = 1.0,
              learning_rate = 0.5,
              random_seed = 2)
som.random_weights_init(entradas)
som.train_random(data = entradas,
                 num_iteration = 100)
som._weights
som._activation_map
agrupador = som.activation_response(entradas)

pcolor(som.distance_map().T)
colorbar()

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
