"""
tuning.py

Provides stubs for:
1. Hyperparameter tuning of the RandomForestChainClassifier.
2. Fine-tuning the BERT-based SentenceTransformer embeddings on your own data.
These stubs are intended as starting points to help boost overall accuracy, god helping.
"""

import logging
import numpy as np
from randomforest import RandomForestChainClassifier
import config
from sklearn.model_selection import train_test_split
from data import Data, encode_labels
import pandas as pd


def grid_search_rf(data_obj, param_grid):
    """
    A simple grid search over RandomForestChainClassifier parameters.
    For each combination, it trains the chain model and computes overall chained accuracy.

    :param data_obj: Data object containing training and testing splits.
    :param param_grid: Dictionary with keys 'n_estimators', 'max_depth', 'min_samples_split'.
    :return: Best parameter combination (as dict) and its accuracy.
    """
    best_params = None
    best_accuracy = -np.inf

    # Unpack grid values (assuming lists)
    for n_est in param_grid.get("n_estimators", [config.NUM_TREES]):
        for max_depth in param_grid.get("max_depth", [config.MAX_DEPTH]):
            for min_split in param_grid.get("min_samples_split", [config.MIN_SAMPLES_SPLIT]):
                logging.info("Testing parameters: n_estimators=%s, max_depth=%s, min_samples_split=%s", n_est,
                             max_depth, min_split)
                # Initialize model with these parameters
                model = RandomForestChainClassifier(n_estimators=n_est, max_depth=max_depth,
                                                    min_samples_split=min_split, random_state=config.RANDOM_SEED)
                # Train the model on training data (you might consider doing a further train/validation split)
                model.train(data_obj)
                # Get predictions and compute overall chained accuracy
                preds = model.predict(data_obj)
                true = data_obj.y_test.values
                n = true.shape[0]
                instance_scores = []
                for i in range(n):
                    if preds[i, 0] != true[i, 0]:
                        score = 0
                    elif preds[i, 1] != true[i, 1]:
                        score = 1
                    elif preds[i, 2] != true[i, 2]:
                        score = 2
                    else:
                        score = 3
                    instance_scores.append(score)
                overall_acc = np.mean(instance_scores) / 3.0
                logging.info(
                    "Parameters: n_estimators=%s, max_depth=%s, min_samples_split=%s --> Overall Chained Accuracy: %.2f%%",
                    n_est, max_depth, min_split, overall_acc * 100)
                if overall_acc > best_accuracy:
                    best_accuracy = overall_acc
                    best_params = {"n_estimators": n_est, "max_depth": max_depth, "min_samples_split": min_split}
    logging.info("Best parameters: %s with Overall Chained Accuracy: %.2f%%", best_params, best_accuracy * 100)
    return best_params, best_accuracy


def fine_tune_bert(train_texts: pd.Series, validation_texts: pd.Series):
    """
    A stub to demonstrate fine-tuning a SentenceTransformer (BERT) model.
    In practice, fine-tuning requires constructing a dataset of InputExamples and using an appropriate loss.
    This function serves as a starting point.

    :param train_texts: pd.Series of training texts.
    :param validation_texts: pd.Series of validation texts.
    :return: A fine-tuned SentenceTransformer model.
    """
    from sentence_transformers import SentenceTransformer, InputExample, losses, LoggingHandler, evaluation
    from torch.utils.data import DataLoader
    import logging

    logging.info("Loading pre-trained SentenceTransformer for fine-tuning: %s", config.BERT_MODEL_NAME)
    model = SentenceTransformer(config.BERT_MODEL_NAME)

    # Create a dummy dataset for fine-tuning.
    # In a real scenario, you would prepare InputExamples with proper labels.
    train_examples = [InputExample(texts=[text], label=0.0) for text in train_texts]
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

    # Use a simple cosine similarity loss (this is just a stub!)
    train_loss = losses.CosineSimilarityLoss(model)

    logging.info("Starting fine-tuning for 1 epoch (this is a stub; adjust epochs and loss as needed)")
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=100)
    logging.info("Fine-tuning complete.")
    return model
