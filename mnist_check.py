import  numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



def get_data(limit = None):
    df = pd.read_csv("mnist_test.csv")
    data = df.values
    np.random.shuffle(data)
    X = data[:, 1:] / 255.0
    Y = data[:, 0]
    if limit is not None:
        return X[:limit], Y[:limit]
    return X, Y

if __name__ == '__main__':
    X, Y = get_data()
    # plt.plot(X[2].reshape(28, 28))
    plt.imshow(X[0].reshape(28,28))
    print(Y[0])
    plt.show()