import numpy as np
import matplotlib.pyplot as plt
from KNN import KNN
from Util import get_xor


if __name__ == '__main__':
    X, Y = get_xor()
    plt.scatter(X[:, 0], X[:, 1], c=Y, alpha=0.5)
    plt.show()
    plt.savefig("images/Xor_fig")
    knn = KNN(3)
    knn.fit(X, Y)
    print("values of score is ", knn.score(X, Y))

