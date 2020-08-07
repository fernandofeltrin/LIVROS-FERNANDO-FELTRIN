# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ

from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

base = load_iris()
entradas = base.data
saidas = base.target
rotulos = base.target_names

knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(entradas, saidas)
knn.predict([[5.1,3.1,1.4,0.2]])

especie = knn.predict([[5.9,3,5.1,1.8]])[0]
print(especie)
rotulos[especie]

etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas,
                                                    test_size = 0.25)
knn.fit(etreino, streino)
previsor = knn.predict(eteste)

margem_acertos = metrics.accuracy_score(steste, previsor)

valores_k = {}
k=1
while k < 25:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(etreino, streino)
    previsores_k = knn.predict(eteste)
    acertos = metrics.accuracy_score(steste, previsores_k)
    valores_k[k] = acertos
    k += 1

plt.plot(list(valores_k.keys()),
         list(valores_k.values()))
plt.xlabel('Valores de K')
plt.ylabel('Performance')
plt.show()

regl = LogisticRegression()
regl.fit(etreino, streino)
regl.predict([[6.2,3.4,5.4,2.3]])
regl.predict_proba([[6.2,3.4,5.4,2.3]])
previsor_regl = regl.predict(eteste)
margem_acertos_regl = metrics.accuracy_score(steste, previsor_regl)

# Livro Python do ZERO à Programação Orientada a Objetos - https://www.amazon.com.br/dp/B07P2VJVW5
# Livro Programação Orientada a Objetos com Python - https://www.amazon.com.br/dp/B083ZYRY9C
# Livro Tópicos Avançados em Python - https://www.amazon.com.br/dp/B08FBKBC9H
# Livro Ciência de Dados e Aprendizado de Máquina - https://www.amazon.com.br/dp/B07X1TVLKW
# Livro Inteligência Artificial com Python - Redes Neurais Intuitivas - https://www.amazon.com.br/dp/B087YSVVXW
# Livro Redes Neurais Artificiais - https://www.amazon.com.br/dp/B0881ZYYCJ
# Livro Análise Financeira com Python - https://www.amazon.com.br/dp/B08B6ZX6BB
# Livro Arrays com Python + Numpy - https://www.amazon.com.br/dp/B08BTN6V7Y
