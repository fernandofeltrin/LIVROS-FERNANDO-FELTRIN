# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

from rbm import RBM
import numpy as np

rbm = RBM(num_visible = 6,
          num_hidden = 2)

base = np.array([[1,1,1,0,0,0],
                 [1,0,1,0,0,0],
                 [1,1,1,0,0,0],
                 [0,0,1,1,1,1],
                 [0,0,1,1,0,1],
                 [0,0,1,0,1,0]])

filmes = ['O Exorcista',
          'American Pie',
          'Matrix',
          'Forrest Gump',
          'Documentário X',
          'O Rei Leão']

rbm.train(base,
          max_epochs = 3000)
rbm.weights

usuario = np.array([[1,1,0,1,0,0]])

rbm.run_visible(usuario)
camada_oculta = np.array([[0,1]])
recomendacao = rbm.run_hidden(camada_oculta)

for i in range(len(usuario[0])):
    print(usuario[0,i])
    if usuario[0,i] == 0 and recomendacao[0,i] == 1:
        print(filmes[i])
        
# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
          
