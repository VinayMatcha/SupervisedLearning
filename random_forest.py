import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

NUMERICAL_COLS = [
  'crim', # numerical
  'zn', # numerical
  'nonretail', # numerical
  'nox', # numerical
  'rooms', # numerical
  'age', # numerical
  'dis', # numerical
  'rad', # numerical
  'tax', # numerical
  'ptratio', # numerical
  'b', # numerical
  'lstat', # numerical
]


NO_TRANSFORM = ['river']


class DataTransformer:

    def fit(self, df):
        self.scalers = {}
        for col in NUMERICAL_COLS:
            scaler = StandardScaler()
            scaler.fit(df[col].values.reshape(-1, 1))
            self.scalers[col] = scaler

    def transform(self, df):
        N, D = df.shape
        X = np.zeros((N, D))
        i = 0
        for col, scaler in self.scalers.items():
            X[:,i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            i += 1
        for col in NO_TRANSFORM:
            X[:,i] = df[col]
            i += 1
        return X

    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def get_data():
    df = pd.read_csv('housing.data', header=None, sep=r"\s+", engine='python')
    df.columns = [
        'crim', # numerical
        'zn', # numerical
        'nonretail', # numerical
        'river', # binary
        'nox', # numerical
        'rooms', # numerical
        'age', # numerical
        'dis', # numerical
        'rad', # numerical
        'tax', # numerical
        'ptratio', # numerical
        'b', # numerical
        'lstat', # numerical
        'medv', # numerical -- this is the target
    ]
    transform = DataTransformer()
    N = len(df)
    train_idx = np.random.choice(N, size=int(0.7*N), replace=False)
    test_idx = [i for i in range(N) if i not in train_idx]
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]
    X_train = transform.fit_transform(df_train)
    Y_train = np.log(df_train['medv'].values)
    X_test = transform.fit_transform(df_test)
    Y_test = np.log(df_test['medv'].values)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    X_train, Y_train, X_test, Y_test = get_data()
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, Y_train)
    predictions = model.predict(X_test)
    plt.scatter(Y_test, predictions)
    plt.xlabel("target")
    plt.ylabel("prediction")

    ymin = np.round(min(min(Y_test), min(predictions)))
    ymax = np.round(max(max(Y_test), max(predictions)))
    print("Ymin: ", ymin, "Ymax: ", ymax)
    r = range(int(ymin), int(ymax))
    plt.plot(r, r)
    plt.show()


    plt.plot(Y_test, label = 'targets')
    plt.plot(predictions, label = 'predictions')
    plt.legend()
    plt.show()

    baseline = LinearRegression()
    single_tree = DecisionTreeRegressor()
    print("CV single tree:", cross_val_score(single_tree, X_train, Y_train).mean())
    print("CV baseline:", cross_val_score(baseline, X_train, Y_train).mean())
    print("CV forest:", cross_val_score(model,X_train, Y_train).mean())

    # test score
    single_tree.fit(X_train, Y_train)
    baseline.fit(X_train, Y_train)
    print("test score single tree:", single_tree.score(X_test, Y_test))
    print("test score baseline:", baseline.score(X_test, Y_test))
    print("test score forest:", model.score(X_test, Y_test))