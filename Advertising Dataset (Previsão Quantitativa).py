import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np

base = pd.read_csv('Advertising.csv')
entradas = base[['TV', 'radio', 'newspaper']]
saidas = base['sales']

sns = sns.pairplot(base,
                   x_vars = ['TV', 'radio', 'newspaper'],
                   y_vars = 'sales',
                   size = 5,
                   kind = 'reg')

from sklearn.model_selection import train_test_split

etreino, eteste, streino, steste = train_test_split(entradas,
                                                    saidas,
                                                    test_size = 0.3)
reglinear = LinearRegression()
reglinear.fit(etreino,streino)

print(list(zip(['TV', 'radio', 'newspaper'], reglinear.coef_)))
print(reglinear.predict([[230.1,37.8,69.2]]))

previsor = reglinear.predict(eteste)
print(previsor)

mae = metrics.mean_absolute_error(steste,previsor)
mse = metrics.mean_squared_error(steste,previsor)
rmse = np.sqrt(metrics.mean_squared_error(steste,previsor))
