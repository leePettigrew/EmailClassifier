"""
data.py

Defines a Data class that encapsulates the training/testing data for modeling,
including TF-IDF features, encoded labels, the vectorizer, and label encoders.
Also provides helper functions to:
  1. Split your preprocessed DataFrame into training and testing sets.
  2. Generate TF-IDF embeddings on the "CombinedText" column.
  3. Encode the label columns (y2, y3, y4) using LabelEncoder.
  4. Return a Data object that encapsulates these elements.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
import config
import embeddings
from sklearn.preprocessing import LabelEncoder

class Data:
    """
    Encapsulates the training/testing data for modeling.

    Attributes:
        X_train: TF-IDF feature matrix for training (sparse matrix).
        X_test:  TF-IDF feature matrix for testing (sparse matrix).
        y_train: DataFrame of encoded labels (y2, y3, y4) for training.
        y_test:  DataFrame of encoded labels (y2, y3, y4) for testing.
        vectorizer: The fitted TF-IDF vectorizer.
        label_encoders: Dictionary of LabelEncoders for each label column.
    """
    def __init__(self, X_train, X_test, y_train, y_test, vectorizer, label_encoders):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.vectorizer = vectorizer
        self.label_encoders = label_encoders

    def get_num_labels(self):
        return self.y_train.shape[1]

    def get_label_names(self):
        if hasattr(self.y_train, 'columns'):
            return list(self.y_train.columns)
        return ["y2", "y3", "y4"]

def encode_labels(labels: pd.DataFrame):
    """
    Encodes categorical labels using LabelEncoder for each column.

    :param labels: DataFrame with label columns (e.g., y2, y3, y4)
    :return: A tuple (encoded_labels, label_encoders)
    """
    label_encoders = {}
    encoded = labels.copy()
    for col in labels.columns:
        le = LabelEncoder()
        encoded[col] = le.fit_transform(labels[col])
        label_encoders[col] = le
    return encoded, label_encoders

def get_data_object(preprocessed_df: pd.DataFrame) -> Data:
    """
    Splits the preprocessed DataFrame into training and testing sets,
    generates TF-IDF embeddings on the CombinedText column, encodes the labels,
    and returns a Data object encapsulating these elements.

    :param preprocessed_df: The DataFrame output from preprocess.py.
    :return: Data object containing training/test splits, TF-IDF matrices, vectorizer, and label encoders.
    """
    # Extract the combined text (created in preprocess.py) and labels (y2, y3, y4)
    features = preprocessed_df["CombinedText"]
    labels = preprocessed_df[["y2", "y3", "y4"]]

    # Split the data; stratify on y2 if possible to maintain class balance
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        features, labels, test_size=config.TEST_SIZE, random_state=config.RANDOM_SEED, stratify=labels["y2"]
    )

    # Generate TF-IDF embeddings using the embeddings module
    X_train, X_test, vectorizer = embeddings.generate_tfidf_features(
        X_train_text, X_test_text, max_features=config.MAX_FEATURES, stop_words=config.STOP_WORDS
    )

    # Encode the label columns to numeric values
    y_train_encoded, label_encoders = encode_labels(y_train)
    y_test_encoded, _ = encode_labels(y_test)

    return Data(X_train, X_test, y_train_encoded, y_test_encoded, vectorizer, label_encoders)
