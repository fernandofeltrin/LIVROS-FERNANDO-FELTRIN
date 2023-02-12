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
