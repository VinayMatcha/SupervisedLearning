import numpy as np
from Util import get_data
from datetime import datetime
from scipy.stats import  norm
from scipy.stats import multivariate_normal as mvn


class Bayes(object):

    def fit(self, X, Y, smoothing=1e-3):
        N, D = X.shape
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for label in labels:
            current_x = X[Y == label]
            self.gaussians[label] = {
                'mean': current_x.mean(axis=0),
                'cov': np.cov(current_x.T) + np.eye(D) * smoothing,
            }
            self.priors[label] = float(len(Y[Y==label]))/len(Y)

    def score(self, X, Y):
        Yhat = self.predict(X)
        return np.mean(Y==Yhat)

    def predict(self, X):
        N , D = X.shape
        K = len(self.gaussians)
        Yhat = np.zeros((N, K))
        for label, gauss in self.gaussians.items():
            mean = gauss['mean']
            cov = gauss['cov']
            Yhat[:,label] = mvn.logpdf(X, mean=mean, cov = cov) + np.log(self.priors[label])
        return np.argmax(Yhat, axis=1)


if __name__ == '__main__':
    X, Y = get_data()
    Ntrain = len(Y)//2
    print(len(Y))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest =  X[Ntrain:], Y[Ntrain:]
    mnistNb = Bayes()
    startTime = datetime.now()
    mnistNb.fit(Xtrain, Ytrain)
    print("TIme taken ", datetime.now()-startTime)
    startTime = datetime.now()
    score = mnistNb.score(Xtest, Ytest)
    print("TIme taken to prediction", datetime.now() - startTime, " and effieciency is ", score)
    startTime = datetime.now()
    score = mnistNb.score(Xtrain, Ytrain)
    print("TIme taken to prediction", datetime.now() - startTime, " and effieciency is ", score)
