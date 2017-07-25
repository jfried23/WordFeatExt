import numpy as np
from sklearn import base
from sklearn import pipeline
from sklearn import preprocessing

class WordLengthMeanStd(base.BaseEstimator,base.TransformerMixin):
    """
    This transformer takes a string and outputs the mean and standard deviation
    of the word lenghts.
    """
    def __init__(self, colName):
        self.colName = colName
        
    def fit(self, X, *_): return self

    def word_length_mean_std(self, s):
        words = s.split()
        wl = [len(word) for word in words]
        return np.array([np.mean(wl), np.std(wl)])
    
    def transform(self, X, *_):
        new_feature = np.atleast_2d(X[self.colName].apply(self.word_length_mean_std)).T
        X.loc[:,'WordLengthMeanStd'] = new_feature.tolist()
        return X

    def fit_transform(self,X, *_):
        return self.transform(X)


def main():
    import pandas as pd
    df = pd.read_csv('./lyrics.csv')
    df.drop('index', axis=1)

    df=df[~df.lyrics.isnull()]
    
    wlmsTransformer = WordLengthMeanStd('lyrics')
    df = wlmsTransformer.transform(df)
    print(df.head())

if __name__ == "__main__":
    main()


