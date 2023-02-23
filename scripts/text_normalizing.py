import re
import nltk
import spacy
import unicodedata
import numpy as np

from pandas import DataFrame
from bs4 import BeautifulSoup
from scripts.contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer
from sklearn.base import BaseEstimator, TransformerMixin
from enum import Enum
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words("english")
nlp = spacy.load("en_core_web_sm")


class NormTechniques(Enum):
    HTML_STRIPPING = "html_stripping"
    CONTRACTION_EXPANSION = "contraction_expansion"
    ACCENTED_CHAR_REMOVAL = "accented_char_removal"
    TEXT_LOWER_CASE = "text_lower_case"
    TEXT_STEMMING = "text_stemming"
    TEXT_LEMMATIZATION = "text_lemmatization"
    SPECIAL_CHAR_REMOVAL = "special_char_removal"
    REMOVE_DIGITS = "remove_digits"
    STOPWORD_REMOVAL = "stopword_removal"


class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, norm_to_use: list) -> None:
        self.norm_to_use = norm_to_use
        self.html_stripping = True if NormTechniques.HTML_STRIPPING in norm_to_use else False,
        self.contraction_expansion = True if NormTechniques.CONTRACTION_EXPANSION in norm_to_use else False,
        self.accented_char_removal = True if NormTechniques.ACCENTED_CHAR_REMOVAL in norm_to_use else False,
        self.text_lower_case = True if NormTechniques.TEXT_LOWER_CASE in norm_to_use else False,
        self.text_stemming = True if NormTechniques.TEXT_STEMMING in norm_to_use else False,
        self.text_lemmatization = True if NormTechniques.TEXT_LEMMATIZATION in norm_to_use else False,
        self.special_char_removal = True if NormTechniques.SPECIAL_CHAR_REMOVAL in norm_to_use else False,
        self.remove_digits = True if NormTechniques.REMOVE_DIGITS in norm_to_use else False,
        self.stopword_removal = True if NormTechniques.STOPWORD_REMOVAL in norm_to_use else False,
        self.stopwords = stopword_list

    @classmethod
    def remove_html_tags(cls, text):
        parser = BeautifulSoup(text, "html.parser")
        text = parser.get_text()
        return text

    @classmethod
    def stem_text(cls, text):
        stemmer = PorterStemmer()
        words = text.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text

    @classmethod
    def lemmatize_text(cls, text):
        doc = nlp(text)
        lemmatized_text = " ".join([token.lemma_ for token in doc])

        return lemmatized_text

    @classmethod
    def expand_contractions(cls, text, contraction_mapping=CONTRACTION_MAP):
        pattern = re.compile(
            "({})".format("|".join(contraction_mapping.keys())), flags=re.IGNORECASE
        )

        def replace(match):
            return contraction_mapping[match.group(0).lower()]

        return pattern.sub(replace, text)

    @classmethod
    def remove_accented_chars(cls, text):
        return "".join(
            c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
        )

    @classmethod
    def remove_special_chars(cls, text, remove_digits=False):
        if remove_digits:
            pattern = r"[^a-zA-Z\s]"
        else:
            pattern = r"[^a-zA-Z0-9\s]"
        return re.sub(pattern, "", text)

    @classmethod
    def remove_stopwords(cls, text, is_lower_case=False, stopwords=stopword_list):
        words = ToktokTokenizer().tokenize(text)
        filtered_words = [
            word for word in words if word.lower() not in stopwords]
        return " ".join(filtered_words)

    @classmethod
    def remove_extra_new_lines(cls, text):
        return " ".join(line.strip() for line in text.splitlines())

    @classmethod
    def remove_extra_whitespace(cls, text):
        return " ".join(text.split())

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        if isinstance(X, DataFrame):
            X = X.to_numpy()
        elif not isinstance(X, np.ndarray):
            raise ValueError(
                f"Expected pd.DataFrame or np.ndarray, but got {type(X)}")

        norm_X = np.empty(X.shape, dtype=object)

        # Normalize each doc in the corpus
        for idx, row in enumerate(X):

            doc = row[0]
            # Remove HTML
            if self.html_stripping:
                doc = self.remove_html_tags(doc)

            # Remove extra newlines
            doc = self.remove_extra_new_lines(doc)

            # Remove accented chars
            if self.accented_char_removal:
                doc = self.remove_accented_chars(doc)

            # Expand contractions
            if self.contraction_expansion:
                doc = self.expand_contractions(doc)

            # Lemmatize text
            if self.text_lemmatization:
                doc = self.lemmatize_text(doc)

            # Stemming text
            if self.text_stemming and not self.text_lemmatization:
                doc = self.stem_text(doc)

            # Remove special chars and\or digits
            if self.special_char_removal:
                doc = self.remove_special_chars(
                    doc, remove_digits=self.remove_digits)

            # Remove extra whitespace
            doc = self.remove_extra_whitespace(doc)

            # Lowercase the text
            if self.text_lower_case:
                doc = doc.lower()

            # Remove stopwords
            if self.stopword_removal:
                doc = self.remove_stopwords(
                    doc, is_lower_case=self.text_lower_case, stopwords=self.stopwords
                )

            # Remove extra whitespace
            doc = self.remove_extra_whitespace(doc)
            doc = doc.strip()

            norm_X[idx] = doc

        return norm_X
