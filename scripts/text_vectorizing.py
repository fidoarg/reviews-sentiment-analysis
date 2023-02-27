from gensim.models import Word2Vec
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pandas.core.frame import DataFrame
import numpy as np

class TextVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, kind='bow', w2vec=False):
        self.kind = kind
        self.w2vec = w2vec
        if not w2vec:
            if kind == 'bow':
                self.vectorizer = CountVectorizer()
            elif kind == 'tfidf':
                self.vectorizer = TfidfVectorizer()
            else:
                raise ValueError('Unsupported vectorizer type')
        else:
            self.w2v_model = None

    def fit(self, X, y=None):
        if not self.w2vec:
            if isinstance(X, DataFrame):
                self.text_data = X.squeeze().values.astype(str)
            elif isinstance(X, np.ndarray):
                self.text_data = np.asarray(X).squeeze().astype(str)
            else:
                raise TypeError('Unsupported data type')

            self.vectorizer.fit(self.text_data)
        else:
            # Train a Word2Vec model on the input data.
            sentences = [doc.split() for doc in X]
            self.w2v_model = Word2Vec(sentences, size=100, window=5, min_count=5, workers=3)

        return self

    def transform(self, X, y=None):
        if not self.w2vec:
            if isinstance(X, DataFrame):
                X = X.squeeze().values.astype(str)
            elif isinstance(X, np.ndarray):
                X = np.asarray(X).squeeze().astype(str)
            else:
                raise TypeError('Unsupported data type')

            return self.vectorizer.transform(X)
        else:
            # Convert each document to a vector by averaging its word vectors.
            X_vectors = []
            for doc in X:
                doc_vector = []
                for word in doc.split():
                    if word in self.w2v_model.wv.vocab:
                        doc_vector.append(self.w2v_model.wv[word])
                if doc_vector:
                    doc_vector = np.mean(doc_vector, axis=0)
                    X_vectors.append(doc_vector)
                else:
                    X_vectors.append(np.zeros(100))
            return np.array(X_vectors)