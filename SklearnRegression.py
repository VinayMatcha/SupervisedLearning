import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor


N = 200
X = np.linspace(0, 10, N).reshape(N, 1)
Y = np.sin(X)

Ntrain = 20
idx = np.random.choice(N, Ntrain)
Xtrain = X[idx]
Ytrain = Y[idx]

kneigh = KNeighborsRegressor(n_neighbors=2, weights='distance')
kneigh.fit(Xtrain, Ytrain)
Yhat = kneigh.predict(X)

deci = DecisionTreeRegressor()
deci.fit(Xtrain, Ytrain)
YhatDeci = deci.predict(X)

plt.scatter(Xtrain, Ytrain)
plt.plot(X, Y)
plt.plot(X, Yhat, label = "Knn Regressor")
plt.plot(X, YhatDeci, label = "DecisionTree Regressor")
plt.legend()
plt.show()
plt.savefig("images/KnnDistAndDecisionTreeSklearn")




