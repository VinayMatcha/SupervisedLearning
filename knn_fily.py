import numpy as np
import pandas as pd
from KNN import KNN
import matplotlib.pyplot as plt

def get_data():
    height = 8
    width = 8
    n = height * width
    X = np.zeros((n,2))
    Y = np.zeros((n))
    n = 0
    start_t = 0
    for i in range(width):
        t = start_t
        for j in range(height):
            X[n] = [i, j]
            Y[n] = t
            n += 1
            t = (t+1)%2
        start_t = (start_t + 1)%2
    return X, Y


if __name__ == '__main__':
    X, Y = get_data()
    plt.scatter(X[:, 0], X[:, 1], s=100, c =Y, alpha=0.5)
    plt.show()
    plt.savefig("images/knnfail")
    knn = KNN(3)
    knn.fit(X, Y)
    print(" score is ", knn.score(X, Y))


