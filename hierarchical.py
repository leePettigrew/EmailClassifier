"""
hierarchical.py

Implements the HierarchicalClassifier for multi-label email classification using a hierarchical modeling approach.
In this design, rather than using one chained classifier that appends previous predictions as features,
we train separate classifiers at each level:
  - A global classifier for Type 2 on the entire training set.
  - For each class in Type 2, a classifier for Type 3 is trained on the corresponding subset.
  - For each (Type 2, Type 3) combination, a classifier for Type 4 is trained on the corresponding subset.
During prediction, each instance is processed in a hierarchical manner.
blahblahblah.
This design follows the project brief's requirement for Hierarchical Modeling.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from base import BaseModel
import logging


class HierarchicalClassifier(BaseModel):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2, random_state=None):
        """
        Initialize the hierarchical classifier with specified RandomForest hyperparameters.

        :param n_estimators: Number of trees for each RandomForest.
        :param max_depth: Maximum depth of each tree (None for no limit).
        :param min_samples_split: Minimum number of samples required to split an internal node.
        :param random_state: Random seed for reproducibility.
        """
        self.model_y2 = RandomForestClassifier(n_estimators=n_estimators,
                                               max_depth=max_depth,
                                               min_samples_split=min_samples_split,
                                               random_state=random_state)
        self.models_y3 = {}  # Dictionary to hold classifiers for y3, keyed by y2 class value.
        self.models_y4 = {}  # Dictionary to hold classifiers for y4, keyed by (y2, y3) tuple.
        self._predictions = None
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state

    def train(self, data):
        """
        Train the hierarchical classifiers.

        :param data: Data object containing training features (X_train) and labels (y_train with columns 'y2', 'y3', 'y4').
        """
        X_train = data.X_train
        y_train = data.y_train  # Expecting a DataFrame with columns 'y2', 'y3', 'y4'

        # Train global classifier for y2
        logging.info("Training Hierarchical Model: Global classifier for Type 2")
        self.model_y2.fit(X_train, y_train['y2'])

        # Train classifiers for y3 based on each unique y2 class
        unique_y2 = np.unique(y_train['y2'])
        for cls in unique_y2:
            indices = y_train[y_train['y2'] == cls].index
            if len(indices) == 0:
                continue
            X_subset = X_train[indices]
            y_subset_y3 = y_train.loc[indices, 'y3']
            logging.info("Training classifier for Type 3 for y2 class %s with %d samples", cls, len(indices))
            model_y3 = RandomForestClassifier(n_estimators=self.n_estimators,
                                              max_depth=self.max_depth,
                                              min_samples_split=self.min_samples_split,
                                              random_state=self.random_state)
            model_y3.fit(X_subset, y_subset_y3)
            self.models_y3[cls] = model_y3

        # Train classifiers for y4 based on each (y2, y3) combination
        for cls_y2 in unique_y2:
            indices_y2 = y_train[y_train['y2'] == cls_y2].index
            unique_y3 = np.unique(y_train.loc[indices_y2, 'y3'])
            for cls_y3 in unique_y3:
                indices = y_train[(y_train['y2'] == cls_y2) & (y_train['y3'] == cls_y3)].index
                if len(indices) == 0:
                    continue
                X_subset = X_train[indices]
                y_subset_y4 = y_train.loc[indices, 'y4']
                logging.info("Training classifier for Type 4 for (y2, y3) = (%s, %s) with %d samples", cls_y2, cls_y3,
                             len(indices))
                model_y4 = RandomForestClassifier(n_estimators=self.n_estimators,
                                                  max_depth=self.max_depth,
                                                  min_samples_split=self.min_samples_split,
                                                  random_state=self.random_state)
                model_y4.fit(X_subset, y_subset_y4)
                self.models_y4[(cls_y2, cls_y3)] = model_y4

    def predict(self, data):
        """
        Predicts Type 2, Type 3, and Type 4 labels for the test set using the hierarchical approach.

        :param data: Data object containing testing features (X_test).
        :return: A numpy array of predictions with shape (n_samples, 3) corresponding to (y2, y3, y4).
        """
        X_test = data.X_test
        n_samples = X_test.shape[0]

        # Predict y2 globally
        pred_y2 = self.model_y2.predict(X_test)
        pred_y3 = []
        pred_y4 = []

        # For each instance, use the corresponding y3 and y4 classifiers based on predicted y2 and y3
        for i in range(n_samples):
            # Extract single instance; ensure it's 2D for predict()
            x_instance = X_test[i]
            if len(x_instance.shape) == 1:
                x_instance = x_instance.reshape(1, -1)

            # Predict y3 using the classifier corresponding to predicted y2
            cls_y2 = pred_y2[i]
            if cls_y2 in self.models_y3:
                model_y3 = self.models_y3[cls_y2]
                pred_y3_instance = model_y3.predict(x_instance)[0]
            else:
                pred_y3_instance = None  # or a default class
            pred_y3.append(pred_y3_instance)

            # Predict y4 using the classifier corresponding to (predicted y2, predicted y3)
            key = (cls_y2, pred_y3_instance)
            if key in self.models_y4:
                model_y4 = self.models_y4[key]
                pred_y4_instance = model_y4.predict(x_instance)[0]
            else:
                pred_y4_instance = None  # or a default class
            pred_y4.append(pred_y4_instance)

        self._predictions = np.column_stack((pred_y2, pred_y3, pred_y4))
        return self._predictions

    def print_results(self, data):
        """
        Evaluates the hierarchical model's predictions against true labels.
        Calculates:
          - Accuracy for y2,
          - Conditional accuracy for y3 (only when y2 is correct),
          - Conditional accuracy for y4 (only when both y2 and y3 are correct),
          - Overall hierarchical accuracy (chained accuracy).

        :param data: Data object containing true labels (y_test).
        """
        if self._predictions is None:
            self.predict(data)
        preds = self._predictions
        true = data.y_test.values  # Expected shape (n_samples, 3)
        n = true.shape[0]

        # Evaluate accuracy for y2
        correct_y2 = (preds[:, 0] == true[:, 0])
        acc_y2 = correct_y2.mean()

        # Evaluate y3 accuracy only for instances where y2 is predicted correctly
        correct_y3 = (preds[:, 1] == true[:, 1])
        if correct_y2.sum() > 0:
            acc_y3_given = (correct_y3 & correct_y2).sum() / correct_y2.sum()
        else:
            acc_y3_given = 0.0

        # Evaluate y4 accuracy only for instances where both y2 and y3 are predicted correctly
        correct_y4 = (preds[:, 2] == true[:, 2])
        both_correct_y2_y3 = correct_y2 & correct_y3
        if both_correct_y2_y3.sum() > 0:
            acc_y4_given = (correct_y4 & both_correct_y2_y3).sum() / both_correct_y2_y3.sum()
        else:
            acc_y4_given = 0.0

        # Compute overall hierarchical (chained) accuracy per instance
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
        hierarchical_accuracy = np.mean(instance_scores) / 3.0  # Normalize

        print("Hierarchical Modeling Evaluation Results:")
        print(f"  Accuracy (Type 2 only)           : {acc_y2 * 100:.2f}%")
        print(f"  Accuracy (Type 3 | Type 2 correct): {acc_y3_given * 100:.2f}%")
        print(f"  Accuracy (Type 4 | Type 2,3 correct): {acc_y4_given * 100:.2f}%")
        print(f"  Overall Hierarchical Accuracy    : {hierarchical_accuracy * 100:.2f}%")
