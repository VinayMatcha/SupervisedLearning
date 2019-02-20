import numpy as np

X = np.zeros((6, 4), dtype=np.float64)
X[0,] = [1,0,0,1]
X[1,] = [0,0,1,1]
X[2,] = [0,1,0,1]
X[3,] = [-1,0,0,1]
X[4,] = [0,-1,0,1]
X[5,] = [0,0,-1,1]

Y = np.array([1,1,1,0,0,0]).T
W = np.array([0, 0, 0, 0], dtype=np.float64)
lr = 0.1
costs = []
epsilon = 1e-5
check = 0.5
oldw = W
while(np.abs(check) > epsilon):
    Yhat = X.dot(W)
    diff = Yhat - Y
    # print(diff)
    oldw = np.copy(W)
    W -= lr * X.T.dot(diff)
    check = np.max(np.abs(oldw-W))
Yhat = X.dot(W)
print(Yhat)
print(W)

#[0.49996597 0.49996597 0.50003403 0.5       ] [0.49996597 0.49996597 0.49996597 0.5       ]