# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

import numpy as np

entradas = np.array([[0,0],[0,1],[1,0],[1,1]])
saidas = np.array([0,0,0,1])
pesos = np.array([0.0,0.0])

taxaAprendizado = 0.5

def Soma(e,p):
    return e.dot(p)
  
s = Soma(entradas, pesos)

def stepFunction(soma):
  if (soma >= 1):
    return 1
  return 0

def calculoSaida(reg):
  s = reg.dot(pesos)
  return stepFunction(s)

def aprendeAtualiza():
  erroTotal = 1
  while (erroTotal != 0):
    erroTotal = 0
    for i in range (len(saidas)):
      calcSaida = calculoSaida(np.array(entradas[i]))
      erro = abs(saidas[i] - calcSaida)
      erroTotal += erro
      for j in range(len(pesos)):
        pesos[j] = pesos[j] + (taxaAprendizado * entradas[i][j] * erro)
        print('Pesos Atualizados> ' + str(pesos[j]))
    print('Total de Erros: ' +str(erroTotal))

aprendeAtualiza()

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
