import re
import nltk
import spacy
import unicodedata

from bs4 import BeautifulSoup
from scripts.contractions import CONTRACTION_MAP
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

tokenizer = ToktokTokenizer()
nltk.download("stopwords")
stopword_list = nltk.corpus.stopwords.words("english")
nlp = spacy.load("en_core_web_sm")


def remove_html_tags(text):
    parser = BeautifulSoup(text, "html.parser")
    text = parser.get_text()
    return text


def stem_text(text):
    stemmer = PorterStemmer()
    words = text.split()
    stemmed_words = [stemmer.stem(word) for word in words]
    stemmed_text = " ".join(stemmed_words)
    return stemmed_text


def lemmatize_text(text):
    doc = nlp(text)
    lemmatized_text = " ".join([token.lemma_ for token in doc])

    return lemmatized_text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    pattern = re.compile(
        "({})".format("|".join(contraction_mapping.keys())), flags=re.IGNORECASE
    )

    def replace(match):
        return contraction_mapping[match.group(0).lower()]

    return pattern.sub(replace, text)


def remove_accented_chars(text):
    return "".join(
        c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c)
    )


def remove_special_chars(text, remove_digits=False):
    if remove_digits:
        pattern = r"[^a-zA-Z\s]"
    else:
        pattern = r"[^a-zA-Z0-9\s]"
    return re.sub(pattern, "", text)


def remove_stopwords(text, is_lower_case=False, stopwords=stopword_list):
    words = ToktokTokenizer().tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stopwords]
    return " ".join(filtered_words)


def remove_extra_new_lines(text):
    return " ".join(line.strip() for line in text.splitlines())


def remove_extra_whitespace(text):
    return " ".join(text.split())


def normalize_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lower_case=True,
    text_stemming=False,
    text_lemmatization=False,
    special_char_removal=True,
    remove_digits=True,
    stopword_removal=True,
    stopwords=stopword_list,
):
    normalized_corpus = []

    # Normalize each doc in the corpus
    for doc in corpus:
        # Remove HTML
        if html_stripping:
            doc = remove_html_tags(doc)

        # Remove extra newlines
        doc = remove_extra_new_lines(doc)

        # Remove accented chars
        if accented_char_removal:
            doc = remove_accented_chars(doc)

        # Expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)

        # Lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)

        # Stemming text
        if text_stemming and not text_lemmatization:
            doc = stem_text(doc)

        # Remove special chars and\or digits
        if special_char_removal:
            doc = remove_special_chars(doc, remove_digits=remove_digits)

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)

        # Lowercase the text
        if text_lower_case:
            doc = doc.lower()

        # Remove stopwords
        if stopword_removal:
            doc = remove_stopwords(
                doc, is_lower_case=text_lower_case, stopwords=stopwords
            )

        # Remove extra whitespace
        doc = remove_extra_whitespace(doc)
        doc = doc.strip()

        normalized_corpus.append(doc)

    return normalized_corpus
