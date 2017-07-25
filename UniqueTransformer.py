"""
    Unique Transformer:

    Transformer for use in scikit processing pipeline

    Class methods:

    fit(self, X, *_):
        Takes: Input dataframe, variable arguments (dropped)
        Returns: self

    transform(self, X, *__):
        Takes: Input dataframe, variable arguments (dropped)
        Returns: Unique word count of every specificied column in the dataframe

    fit_transform(self, X, *_):
        Takes: Input dataframe, variable arguments (dropped)
        Returns: self.transform(X)
"""

from collections import Counter
from sklearn import base
import re

class UniqueTransformer(base.BaseEstimator, base.TransformerMixin):
    def __init__(self, col_name):
        self.col_name = col_name
        
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        ret = []
        raw_str_list = X[self.col_name]
        for raw_str in raw_str_list:
            if type(raw_str) is str:
                word_list = re.sub(r"\W+", ' ', raw_str).lower().split()
                the_count = Counter()
                count_list = []
                for word in word_list:
                    if len(word) != 1 or (word == 'a' or word == 'i'):
                        count_list.append(word)
                    the_count.update(count_list)
                    
                ret.append(len(the_count))
            else:
                ret.append(0)

        X.loc[:,'unique_count'] = ret
        return X
    
    def fit_transform(self,X, *_):
        self.fit(X)
        return self.transform(X)