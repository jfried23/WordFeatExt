class GenreGroups(base.BaseEstimator,base.TransformerMixin):
    """
    Input: 'genre' column, the series.
    Output: numpy array with Encoded Genres
    GenreGroups will group the genres into three classes:
    1. rock
    2. pop_hiphop
    3. other
    """
    def __init__(self):
        self._transformer = sklearn.preprocessing.LabelBinarizer()

    def groupGenres(self,i):
        genre='other'
        if i in set(['Rock','Hip-Hop']): genre='rock_hiphop'
        elif i in set(['Pop']): genre='pop'
        return genre

    def fit(self, X, *_):

        X = X.apply(self.groupGenres)
        self._transformer.fit(X)
        return self

    def transform(self, X, *_):
        X = X.apply(self.groupGenres)
        return self._transformer.transform(X)

    def fit_transform(self,X, *_):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self,array,*_):
        return self._transformer.inverse_transform(array)
