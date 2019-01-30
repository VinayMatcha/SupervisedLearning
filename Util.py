import numpy as np
import pandas as pd
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



def get_xor():
    X = np.zeros((200, 2))
    X[:50] = np.random.random((50, 2)) / 2 + 0.5
    X[50:100] = np.random.random((50, 2)) / 2
    X[100:150] = np.random.random((50, 2)) / 2 + np.array([[0, 0.5]])
    X[150:] =  np.random.random((50, 2)) / 2 + np.array([[0.5, 0]])
    Y = np.array([0]*100 + [1]*100)
    return X, Y



def get_donut():
    N = 200
    r_inner = 5
    r_outer = 10
    R1 = np.random.randn(N//2) + r_inner
    theta = 2 * np.pi * np.random.random(N//2)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T
    R2 = np.random.randn(N//2) + r_outer
    theta = 2 * np.pi * np.random.random(N // 2)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

    X = np.concatenate([X_inner, X_outer])
    Y = np.concatenate(([0]*(N//2),[1]*(N//2)))
    return X, Y
