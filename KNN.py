from datetime import datetime

import numpy as np
from sortedcontainers import SortedList
import pandas as pd
from Util import get_data, get_xor

class KNN:

    def __init__(self, K):
        self.k = K

    def fit(self, X, Y):
        self.X = X
        self.Y = Y

    def predict(self, X):
        y = np.zeros(len(X))
        for i, x in enumerate(X):
            sl = SortedList()
            for j, xl in enumerate(self.X):
                diff = x-xl
                dist = diff.dot(diff)
                if(len(sl) < self.k):
                    sl.add((dist, self.Y[j]))
                else:
                    if dist < sl[-1][0]:
                        del sl[-1]
                        sl.add((dist, self.Y[j]))
            res = {}
            for _, v in sl:
                res[v] = res.get(v,0) + 1
            max_votes = 0
            max_class = -1
            for v, count in res.items():
                if(max_votes < count):
                    max_votes = count
                    max_class = v
            y[i] = max_class
        return y


    def score(self, X, Y):
        p = self.predict(X)
        return np.mean(p == Y)

if __name__ == '__main__':
    X, Y = get_xor()
    Ntrain = 150
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]
    for k in (1,2,3,4,5):
        knn = KNN(k)
        start = datetime.now()
        knn.fit(Xtrain, Ytrain)
        score = knn.score(Xtrain, Ytrain)
        endTime = datetime.now()
        print(score, " is score in time", endTime-start)
        score = knn.score(Xtest, Ytest)
        start = datetime.now()
        print(score, " is score in time", start - endTime)




