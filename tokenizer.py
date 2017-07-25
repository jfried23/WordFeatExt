import string
from sklearn import feature_extraction, base

class Tokenizer(base.BaseEstimator,base.TransformerMixin):
    def __init__(self,colName):
        self.colName = colName
        
    def fit(self, X, *_): return self
    
    def tokenize(self, X):
        """
        sents = X[self.colName].tolist()
        exclude = set(string.punctuation)
        sents = [''.join(ch for ch in sent if ch not in exclude) for sent in sents]
        tokenize_sents = [sent.replace("\n"," ").split(" ") for sent in sents]        
        """
        exclude = set(string.punctuation)
        sent = ''.join(ch for ch in X if ch not in exclude)
        tokenize_sents = sent.replace("\n"," ").split(" ")
        return tokenize_sents

    def transform(self, X, *_):
        tfid=sklearn.feature_extraction.text.TfidfVectorizer(max_features=500, max_df = .9, tokenizer = self.tokenize)
        tf = tfid.fit_transform(X[self.colName].tolist())
        return tf

    def fit_transform(self,X, *_):
        return self.transform(X) 