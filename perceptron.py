import numpy as np
import matplotlib.pyplot as plt
from Util import get_data as mnist_data
from datetime import datetime

def get_data():
    w = np.array([-0.5, 0.5])
    b = 0.1
    X = np.random.random((300,2))*2 -1
    Y = np.sign(X.dot(w) + b)
    return X, Y

class perceptron:
    def fit(self, X, Y, learning_rate = 1.0, epochs=1000):
        N, D = X.shape
        self.w = np.random.random(D)
        self.b = 0
        costs = []
        for epoch in range(epochs):
            Yhat = self.predict(X)
            incorrect =  np.nonzero(Y != Yhat)[0]
            if len(incorrect) == 0:
                break
            i = np.random.choice(incorrect)
            self.w = self.w + learning_rate * Y[i] * X[i]
            self.b = self.b + learning_rate * Y[i]
            cost = float(len(incorrect))/N
            costs.append(cost)

        print("final w:", self.w, "final b:", self.b, "epochs:", (epoch + 1), "/", epochs)
        plt.plot(costs)
        plt.show()
        plt.savefig("images/Normal_Perceptron_cost1")


    def predict(self, X):
        return np.sign(X.dot(self.w) + self.b)


    def score(self, X, Y):
        Yhat = self.predict(X)
        return np.mean(Yhat == Y)

if __name__ == '__main__':
    # linearly separable data
    X, Y = get_data()
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
    Ntrain = len(Y)//2
    Xtrain = X[:Ntrain]
    Ytrain = Y[:Ntrain]
    Xtest = X[Ntrain:]
    Ytest = Y[Ntrain:]

    model = perceptron()
    startTime = datetime.now()
    model.fit(Xtrain, Ytrain)
    print("TIme taken ", datetime.now() - startTime)
    startTime = datetime.now()
    score = model.score(Xtest, Ytest)
    print("TIme taken to prediction", datetime.now() - startTime, " and effieciency is ", score)
    startTime = datetime.now()
    score = model.score(Xtrain, Ytrain)
    print("TIme taken to prediction", datetime.now() - startTime, " and effieciency is ", score)


