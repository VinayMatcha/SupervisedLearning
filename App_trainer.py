import numpy as np
import pickle
from Util import get_data
from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    X, Y = get_data()
    Ntrain = len(Y)//4
    Xtrain = X[:Ntrain]
    Ytrain = Y[:Ntrain]
    Xtest = X[Ntrain:]
    Ytest = Y[Ntrain:]

    model = RandomForestClassifier()
    model.fit(Xtrain, Ytrain)

    print("test Accuracy : ",model.score(Xtest, Ytest))
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)


