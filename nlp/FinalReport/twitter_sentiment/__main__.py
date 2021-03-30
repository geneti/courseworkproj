import csv
import re
import pandas as pd
from CLEAN import clean
from LOAD import load_data_direct, compress_csv
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np


def __main__():
    # compress_csv('/Users/liuchang/Downloads/twitter_sentiment/Corona_NLP_train.csv')

    very_pos, pos, neutral, neg, very_neg = load_data_direct(
        '/Users/liuchang/Downloads/twitter_sentiment/Corona_NLP_train.csv')
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(very_pos + pos + neutral + neg + very_neg)
    X = X.toarray()
    y = np.asarray(
        [1 for i in range(0, len(very_pos))] + [2 for i in range(0, len(pos))] + [3 for i in range(0, len(neutral))] + [
            4 for i in range(0, len(neg))] + [5 for i in range(0, len(very_neg))])
    print('Bag of words representation for 5 labels (very_pos to very_neg): ', len(very_pos), ',', len(pos), ',',
          len(neutral), ',', len(neg), ',', len(very_neg))
    print('Yields a array, X of size:', np.size(X, 0), 'sentences x', np.size(X, 1), 'tokens')
    print(X)

__main__()
