import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

NUMERICAL_COLS = ()
CATEGORICAL_COLS = np.arange(22) + 1


class DataTranformer:

    def fit(self, df):
        self.labelEncoders = {}
        self.scalers = {}
        for col in NUMERICAL_COLS:
            scaler = StandardScaler()
            scaler.fit(df[col].reshape(-1, 1))
            self.scalers[col] = scaler

        for col in CATEGORICAL_COLS:
            encoder = LabelEncoder()
            values = df[col].tolist()
            values.append('missing')
            encoder.fit(values)
            self.labelEncoders[col] = encoder

        self.D = len(NUMERICAL_COLS)
        for col, encoder in self.labelEncoders.items():
            self.D += len(encoder.classes_)
        print("Dimensionality is ", self.D)


    def transform(self, df):
        N, _ = df.shape
        X = np.zeros((N, self.D))
        i = 0
        for col, scaler in self.scalers.items():
            X[:, i] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            i += 1

        for col, encoder in self.labelEncoders.items():
            K = len(encoder.classes_)
            X[np.arange(N), encoder.transform(df[col])+i] = 1
            i += K

        return X


    def fit_transform(self, df):
        self.fit(df)
        return self.transform(df)


def replace_missing(df):
    for col in NUMERICAL_COLS:
        if np.any(df[col].isnull()):
            med = np.median(df[col][df[col].notnull()])
            df.loc[df[col][df[col].isnull()]] = med

    for col in CATEGORICAL_COLS:
        if np.any(df[col].isnull()):
            df.loc[df[col].isnull(), col] = 'missing'




def get_data():
    df = pd.read_csv('mushroom.data', header=None)
    df[0] = df.apply(lambda row:0 if row[0]=='e' else 1, axis=1)
    replace_missing(df)
    transform = DataTranformer()
    X = transform.fit_transform(df)
    Y = df[0].values
    return X, Y


if __name__ == '__main__':
    X, Y = get_data()
    baseline = LogisticRegression()
    print("CV baseline:", cross_val_score(baseline, X, Y, cv=8).mean())

    # single tree
    tree = DecisionTreeClassifier()
    print("CV one tree:", cross_val_score(tree, X, Y, cv=8).mean())

    model = RandomForestClassifier(n_estimators=20)  # try 10, 20, 50, 100, 200
    print("CV forest:", cross_val_score(model, X, Y, cv=8).mean())