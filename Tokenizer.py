import string
from sklearn import feature_extraction, base

class Tokenizer(base.BaseEstimator,base.TransformerMixin):
    # pass in column name, data frame
    # returns list of sparse matrix-style strings of tfidf values
    def __init__(self,colName):
        self.colName = colName
        
    def fit(self, X, *_): return self
    
    def tokenize(self, X):
        exclude = set(string.punctuation)
        sent = ''.join(ch for ch in X if ch not in exclude)
        tokenize_sents = sent.replace("\n"," ").split(" ")
        return tokenize_sents

    def transform(self, X, *_):
        tfid=feature_extraction.text.TfidfVectorizer(max_features=500, max_df = .9, tokenizer = self.tokenize)
        tf = tfid.fit_transform(X[self.colName].tolist())
        tf_list = [tf[i] for i in range(tf.shape[0])]
        return tf_list

    def fit_transform(self,X, *_):
        return self.transform(X) 