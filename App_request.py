import numpy as np
import matplotlib.pyplot as plt
from Util import get_data
import requests

if __name__ == '__main__':
    X, Y = get_data()
    N = len(Y)
    while True:
        i = np.random.choice(N)
        r = requests.post("http://localhost:8888/predictMnist", data={'input': X[i]})
        print("RESPONSE:")
        print(r.content)
        j = r.json()
        print("targets ", Y[i])

        plt.imshow(X[i].reshape(28, 28), cmap='gray')
        plt.title("Target: %d, Prediction: %d" % (Y[i], j['prediction']))
        plt.show()
        response = input("continue? yes | No\n")
        if(response in {'no', "No"}):
            break