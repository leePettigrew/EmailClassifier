"""
embeddings.py

Provides functions for converting text data into numerical feature vectors using TF-IDF.
This module isolates the text embedding process so that you can easily swap out
TF-IDF for another embedding method later if desired (probably)
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

def generate_tfidf_features(train_texts: pd.Series, test_texts: pd.Series, max_features: int, stop_words=None):
    """
    Fit a TF-IDF vectorizer on the training texts and transform both training and test texts.

    :param train_texts: pd.Series containing the training text data.
    :param test_texts: pd.Series containing the test text data.
    :param max_features: Maximum number of features to consider.
    :param stop_words: Stop words to exclude (e.g., "english").
    :return: A tuple (X_train, X_test, vectorizer) where X_train and X_test are the transformed feature matrices.
    """
    vectorizer = TfidfVectorizer(max_features=max_features, stop_words=stop_words)
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)
    return X_train, X_test, vectorizer
