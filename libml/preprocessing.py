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
from joblib import dump, load

def _load_data():
    message = pd.read_csv('../restaurantreviewsdata/a1_RestaurantReviews_HistoricDump.tsv',
                          delimiter='\t',
                          quoting=3,)
    return message

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

def _preprocess(messages):
    '''
    1. Convert word tokens from processed msgs dataframe into a bag of words (limited to 1420 features)
    2. Convert bag of words representation into tf-idf vectorized representation for each message
    3. Add message length as a numerical feature
    '''
    preprocessor = make_union(
        make_pipeline(
            CountVectorizer(analyzer=_text_process, max_features=1420),
            TfidfTransformer()
        ),
        FunctionTransformer(_extract_message_len, validate=False)
    )

    preprocessed_data = preprocessor.fit_transform(messages['Review'])
    dump(preprocessor, '../output/preprocessor.joblib')
    dump(preprocessed_data, '../output/preprocessed_data.joblib')
    return preprocessed_data

def prepare(message):
    preprocessor = load('../output/preprocessor.joblib')
    return preprocessor.transform([message])

def get_vectorizer():
    return CountVectorizer()

def main():
    messages = _load_data()
    print('\n################### Processed Messages ###################\n')
    with pd.option_context('expand_frame_repr', False):
        print(messages)
    _preprocess(messages)

if __name__ == "__main__":
    main()