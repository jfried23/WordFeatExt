import numpy as np
from sklearn import base
from sklearn import pipeline
from sklearn import preprocessing

class FeatureVectorizer(base.BaseEstimator, base.TransformerMixin):
    """
    Takes as list of data frame column names and outputs a numpy array
    with only those columns.
    """
    def __init__(self, colNames):
        self.colNames = colNames

    def fit(self, X, *_): return self

    def transform(self, X, *_):

        return np.vstack(X.loc[:, self.colNames].values)

    def fit_transform(self, X, *_):
        return self.transform(X)


def main():
    import pandas as pd
    df = pd.read_csv('./lyrics.csv', nrows=100)
    df.drop('index', axis=1)

    df=df[~df.lyrics.isnull()]

    fvTrans = FeatureVectorizer(['genre', 'year'])
    vectorized_features = fvTrans.transform(df)
    print(vectorized_features)

if __name__ == "__main__":
    main()

