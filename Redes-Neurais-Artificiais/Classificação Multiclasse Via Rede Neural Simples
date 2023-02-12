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
