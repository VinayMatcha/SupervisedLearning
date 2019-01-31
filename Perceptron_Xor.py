import numpy as np
from perceptron import perceptron
from Util import get_simple_xor


if __name__ == '__main__':
    X, Y = get_simple_xor()
    Y[Y == 0] = -1
    model = perceptron()
    model.fit(X, Y)
    print(model.score(X, Y))

