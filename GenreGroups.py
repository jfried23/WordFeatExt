from sklearn import base
import pandas as pd
from sklearn.preprocessing import LabelBinarizer

class GenreGroups(base.BaseEstimator,base.TransformerMixin):
    """
    Input: 'genre' column into object and then pass DataFrame.
    Output: numpy array with Encoded Genres
    GenreGroups will group the genres into three classes:
    1. rock
    2. pop_hiphop
    3. other
    """
    def __init__(self, colName):
        self.colName = colName
        self._transformer = LabelBinarizer()

    def groupGenres(self,i):
        genre='other'
        if i in set(['Rock','Hip-Hop']): genre='rock_hiphop'
        elif i in set(['Pop']): genre='pop'
        return genre

    def fit(self, X, *_):

        newLabels = X[self.colName].apply(self.groupGenres)
        self._transformer.fit(newLabels)
        return self

    def transform(self, X, *_):
        newLabels = X[self.colName].apply(self.groupGenres)
        newLabels = pd.DataFrame(data=self._transformer.transform(newLabels))
        X = pd.concat([X,newLabels],axis=1)
        return X

    def fit_transform(self,X, *_):
        self.fit(X[self.colName])
        return self.transform(X[self.colName])

    def inverse_transform(self,array,*_):
        return self._transformer.inverse_transform(array)

if __name__=="__main__":
    import sklearn
    import pandas as pd
    genre = GenreGroups('genre')
