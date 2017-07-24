import pandas as pd
import numpy as np
from sklearn import base
from sklearn import pipeline
from sklearn import preprocessing

class WordCount(base.BaseEstimator,base.TransformerMixin):
    """
    An Example custom transformer!
    WordCount takes the 'lyrics' column returns the number of words
    in the song.
    Can you make it better?
    Count the number of unique words instead?
    """
    def __init__(self, colName):
        self.colName = colName
        
    def fit(self, X, *_): return self

    def word_length_mean_std(self, s):
        words = s.split()
        wl = [len(word) for word in words]
        return np.array([np.mean(wl), np.std(wl)])
    
    def transform(self, X, *_):
        return np.atleast_2d(X[self.colName].apply(self.word_length_mean_std)).T

    def fit_transform(self,X, *_):
        return self.transform(X)
