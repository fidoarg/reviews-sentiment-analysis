import numpy as np

from pandas import DataFrame
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.base import TransformerMixin, BaseEstimator


class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, kind='bow'):
        self.kind = kind
        if kind == 'bow':
            self.vectorizer = CountVectorizer()
        elif kind == 'tfidf':
            self.vectorizer = TfidfVectorizer()
        else:
            raise ValueError('Unsupported vectorizer type')

    def fit(self, X, y=None):

        if isinstance(X, DataFrame):
            self.text_data = X.squeeze().values.astype(str)
        elif isinstance(X, np.ndarray):
            self.text_data = np.asarray(X).squeeze().astype(str)
        else:
            raise TypeError('Unsupported data type')

        self.vectorizer.fit(self.text_data)

        return self

    def transform(self, X, y=None):

        if isinstance(X, DataFrame):
            X = X.squeeze().values.astype(str)
        elif isinstance(X, np.ndarray):
            X = np.asarray(X).squeeze().astype(str)
        else:
            raise TypeError('Unsupported data type')

        return self.vectorizer.transform(X)
