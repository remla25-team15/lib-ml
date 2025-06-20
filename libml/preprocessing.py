import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
import nltk
nltk.download('stopwords')
from sklearn.pipeline import make_pipeline, make_union
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import FunctionTransformer

def _text_process(data):
    '''
    1. Remove punctuation and non-letter characters
    2. Do stemming of words
    3. Remove stop words (except 'not')
    4. Return list of clean stemmed text words
    '''
    # Remove all non-letter characters
    data = re.sub('[^a-zA-Z]', ' ', data)

    # Remove punctuation
    nopunc = [c for c in data if c not in string.punctuation]
    nopunc = ''.join(nopunc)

    # Tokenize and stem
    stemmer = SnowballStemmer('english')
    tokens = nopunc.lower().split()
    stemmed = [stemmer.stem(word) for word in tokens]

    # Remove stopwords (but keep 'not')
    stop_words = set(stopwords.words('english'))
    stop_words.discard('not')  # keep 'not'
    clean_msgs = [word for word in stemmed if word not in stop_words]

    return clean_msgs

def _extract_message_len(data):
    # return as np.array and reshape so that it works with make_union
    return np.array([len(review) for review in data]).reshape(-1, 1)


def _preprocess(messages, cv=None):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words (using CountVectorizer).
    2. Convert bag of words representation into tf-idf vectorized representation for each message.
    3. Add message length as a numerical feature.
    If `cv` (CountVectorizer) is provided, it will be used to transform the data.
    Otherwise, a new `cv` will be created and fitted.
    '''

    if cv is None:
        # 1. If CountVectorizer is not provided, create and fit it
        cv = CountVectorizer(analyzer=_text_process, max_features=1420)
        X_counts = cv.fit_transform(messages['Review'])
    else:
        # 2. If CountVectorizer is provided, just use it to transform the messages
        X_counts = cv.transform(messages['Review'])

    # 2. Convert counts to tf-idf representation
    tfidf_transformer = TfidfTransformer()
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    # 3. Add message length as a numerical feature
    message_length = _extract_message_len(messages['Review'])

    # Combine tf-idf features and message length
    preprocessed_data = np.hstack([X_tfidf.toarray(), message_length])

    # Return the preprocessed data and the CountVectorizer
    return preprocessed_data, cv

