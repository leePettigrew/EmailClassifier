"""
randomforest.py

Implements the RandomForestChainClassifier class which performs chained multi-output
classification. It uses three RandomForest classifiers, one for each label (y2, y3, y4).
Predictions are chained so that:
  - The first classifier predicts y2.
  - The second classifier predicts y3 using original features augmented with predicted y2.
  - The third classifier predicts y4 using original features augmented with predicted y2 and y3.
This approach follows the design requirement for Chained Multi-Output Classification.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from base import BaseModel

class RandomForestChainClassifier(BaseModel):
    def __init__(self, n_estimators=100, random_state=None):
        """
        Initialize three RandomForest classifiers, one for each output.
        :param n_estimators: Number of trees for each RandomForest.
        :param random_state: Random seed for reproducibility.
        """
        self.model_y2 = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model_y3 = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self.model_y4 = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        self._predictions = None

    def train(self, data):
        """
        Train the three RandomForest models in a chained manner.
        :param data: Data object containing training features (X_train) and labels (y_train as DataFrame with columns y2, y3, y4).
        """
        X_train = data.X_train
        y_train = data.y_train  # Expected to be a DataFrame with columns 'y2', 'y3', 'y4'

        # Train first model on y2 only
        self.model_y2.fit(X_train, y_train['y2'])

        # For y3, augment features with true y2 values
        X_train_y2 = self._append_feature(X_train, y_train['y2'].values)
        self.model_y3.fit(X_train_y2, y_train['y3'])

        # For y4, augment features with true y2 and true y3 values
        X_train_y2_y3 = self._append_feature(X_train_y2, y_train['y3'].values)
        self.model_y4.fit(X_train_y2_y3, y_train['y4'])

    def predict(self, data):
        """
        Predict y2, y3, and y4 in a chained fashion for the test set.
        :param data: Data object containing testing features (X_test).
        :return: A numpy array of predictions with shape (n_samples, 3) corresponding to y2, y3, y4.
        """
        X_test = data.X_test

        # Predict y2
        pred_y2 = self.model_y2.predict(X_test)
        # Append predicted y2 as additional feature
        X_test_y2 = self._append_feature(X_test, pred_y2)

        # Predict y3 using original features + predicted y2
        pred_y3 = self.model_y3.predict(X_test_y2)
        # Append predicted y3 to the previous augmented features
        X_test_y2_y3 = self._append_feature(X_test_y2, pred_y3)

        # Predict y4 using original features + predicted y2 and predicted y3
        pred_y4 = self.model_y4.predict(X_test_y2_y3)

        # Combine predictions into a single numpy array with columns for y2, y3, y4
        self._predictions = np.column_stack((pred_y2, pred_y3, pred_y4))
        return self._predictions

    def print_results(self, data):
        """
        Evaluate the chained predictions against true labels using the specified evaluation strategy.
        Calculates:
          - Accuracy for y2,
          - Conditional accuracy for y3 (only when y2 is correct),
          - Conditional accuracy for y4 (only when both y2 and y3 are correct),
          - Overall chained accuracy per instance.
        :param data: Data object containing true labels (y_test).
        """
        if self._predictions is None:
            self.predict(data)
        preds = self._predictions
        true = data.y_test.values  # Expected shape (n_samples, 3)
        n = true.shape[0]

        # Evaluate accuracy of y2 predictions
        correct_y2 = (preds[:, 0] == true[:, 0])
        acc_y2 = correct_y2.mean()

        # Evaluate y3 conditional on y2 being correct
        correct_y3 = (preds[:, 1] == true[:, 1])
        if correct_y2.sum() > 0:
            acc_y3_given = (correct_y3 & correct_y2).sum() / correct_y2.sum()
        else:
            acc_y3_given = 0.0

        # Evaluate y4 conditional on both y2 and y3 being correct
        correct_y4 = (preds[:, 2] == true[:, 2])
        both_correct_y2_y3 = correct_y2 & correct_y3
        if both_correct_y2_y3.sum() > 0:
            acc_y4_given = (correct_y4 & both_correct_y2_y3).sum() / both_correct_y2_y3.sum()
        else:
            acc_y4_given = 0.0

        # Calculate chained accuracy per instance:
        # 0 if y2 is wrong; 1 if y2 correct but y3 wrong; 2 if y2 and y3 correct but y4 wrong; 3 if all correct.
        instance_scores = []
        for i in range(n):
            if not correct_y2[i]:
                score = 0
            elif not correct_y3[i]:
                score = 1
            elif not correct_y4[i]:
                score = 2
            else:
                score = 3
            instance_scores.append(score)
        instance_scores = np.array(instance_scores)
        chained_accuracy = instance_scores.mean() / 3.0  # normalized to a fraction

        # Print evaluation results
        print("Evaluation Results:")
        print(f"  Accuracy (y2 only)           : {acc_y2*100:.2f}%")
        print(f"  Accuracy (y3 | y2 correct)   : {acc_y3_given*100:.2f}%")
        print(f"  Accuracy (y4 | y2,y3 correct): {acc_y4_given*100:.2f}%")
        print(f"  Overall Chained Accuracy     : {chained_accuracy*100:.2f}%")

    def _append_feature(self, X, extra_feature):
        """
        Appends an extra feature (e.g., predicted labels) as a new column to the feature matrix X.
        Handles both sparse and dense X appropriately.

        :param X: Feature matrix (sparse or dense)
        :param extra_feature: Array of extra feature values (1D array)
        :return: New feature matrix with the extra column appended.
        """
        # Check if X is a sparse matrix (from TF-IDF)
        if 'sparse' in str(type(X)):
            from scipy import sparse
            # Ensure extra_feature is 2D and convert to a supported numeric type
            extra_feature = extra_feature.reshape(-1, 1).astype(float)
            extra_sparse = sparse.csr_matrix(extra_feature)
            return sparse.hstack([X, extra_sparse])
        else:
            # If X is a dense numpy array
            return np.hstack([X, extra_feature.reshape(-1, 1)])
