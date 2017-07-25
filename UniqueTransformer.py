from collections import Counter

class UniqueTransformer(base.BaseEstimator, base.TransformerMixin):
    """
    Unique Transformer
    """
    def __init__(self, col_name):
        self.col_name = col_name
        self.mapping_dict = mapping_dict
        
    def fit(self, X, *_):
        return self
    
    def transform(self, X, *_):
        ret = []
        raw_str_list = X[self.col_name]
        for raw_str in raw_str_list:
            word_list = re.sub(r"\W+", ' ', raw_str).lower().split()
            the_count = Counter()
            count_list = []
            for word in word_list:
                if len(word) != 1 or (word == 'a' or word == 'i'):
                    count_list.append(word)
                the_count.update(count_list)
                
            ret.append(len(the_count))

        return ret
    
    def fit_transform(self,X, *_):
        self.fit(X)
        return self.transform(X)