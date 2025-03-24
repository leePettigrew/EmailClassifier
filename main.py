"""
main.py

Main driver script for the Chained Multi-Output Email Classification project.
It orchestrates:
  1. Data loading and preprocessing (including deduplication and noise removal)
  2. TF-IDF embedding and train/test splitting (encapsulated in a Data object)
  3. Model training, prediction, and evaluation using a RandomForest chain classifier

This implementation strictly adheres to the project brief and ensures modularity.
"""

import preprocess
import data
import randomforest as rf
import hierarchical as hm  # Your hierarchical.py module
import config
import evaluation
import logging
import sys


def get_user_input():
    print("Welcome to the Email Classification Pipeline")

    # Select modeling approach
    approach = input("Select modeling approach (1) Chained Multi-Output, (2) Hierarchical [default: 1]: ").strip()
    if approach not in ["1", "2"]:
        print("Invalid or empty input. Defaulting to '1' (Chained Multi-Output).")
        approach = "1"

    # Select embedding method
    embed_method = input("Enter embedding method (tfidf/bert) [default: tfidf]: ").strip().lower()
    if embed_method not in ["tfidf", "bert"]:
        print("Invalid or empty input. Defaulting to 'tfidf'.")
        embed_method = "tfidf"
    config.EMBEDDING_METHOD = embed_method

    # Whether to compare embeddings
    compare_input = input("Compare embeddings? (y/n) [default: n]: ").strip().lower()
    config.COMPARE_EMBEDDINGS = (compare_input.startswith("y"))

    # Whether to plot confusion matrices
    plot_input = input("Plot confusion matrices? (y/n) [default: y]: ").strip().lower()
    config.PLOT_CONFUSION = (not plot_input.startswith("n"))

    # Optionally set RandomForest hyperparameters
    n_trees_input = input(f"Enter number of trees for RandomForest [default: {config.NUM_TREES}]: ").strip()
    if n_trees_input:
        try:
            config.NUM_TREES = int(n_trees_input)
        except ValueError:
            print("Invalid input. Using default.")

    max_depth_input = input(
        f"Enter max_depth for RandomForest (None for no limit) [default: {config.MAX_DEPTH}]: ").strip()
    if max_depth_input.lower() == "none" or max_depth_input == "":
        config.MAX_DEPTH = None
    else:
        try:
            config.MAX_DEPTH = int(max_depth_input)
        except ValueError:
            print("Invalid input. Using default (None).")
            config.MAX_DEPTH = None

    min_split_input = input(f"Enter min_samples_split for RandomForest [default: {config.MIN_SAMPLES_SPLIT}]: ").strip()
    if min_split_input:
        try:
            config.MIN_SAMPLES_SPLIT = int(min_split_input)
        except ValueError:
            print("Invalid input. Using default.")

    logging.info(
        "Configuration updated: Approach=%s, EMBEDDING_METHOD=%s, COMPARE_EMBEDDINGS=%s, PLOT_CONFUSION=%s, NUM_TREES=%s, MAX_DEPTH=%s, MIN_SAMPLES_SPLIT=%s",
        approach, config.EMBEDDING_METHOD, config.COMPARE_EMBEDDINGS, config.PLOT_CONFUSION, config.NUM_TREES,
        config.MAX_DEPTH,
        config.MIN_SAMPLES_SPLIT)

    return approach


def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Get user input interactively
    approach = get_user_input()

    # Step 1: Load raw data
    logging.info("Loading raw data...")
    df_raw = preprocess.get_input_data()

    # Step 2: Deduplicate email interaction content
    logging.info("Performing deduplication...")
    df_dedup = preprocess.de_duplication(df_raw)

    # Step 3: Remove noise from text
    logging.info("Removing noise from text...")
    df_clean = preprocess.noise_remover(df_dedup)

    # Save preprocessed output for debugging
    preprocessed_output_path = "preprocessed_data.csv"
    df_clean.to_csv(preprocessed_output_path, index=False)
    logging.info("Preprocessed data saved to %s", preprocessed_output_path)

    # Step 4: Run the pipeline based on user settings
    if config.COMPARE_EMBEDDINGS:
        original_method = config.EMBEDDING_METHOD
        for method in ["tfidf", "bert"]:
            config.EMBEDDING_METHOD = method
            logging.info("=== Running pipeline with %s embeddings ===", method.upper())
            logging.info("Creating data object with %s features...", method.upper())
            data_obj = data.get_data_object(df_clean)

            # Initialize the chosen model based on approach
            if approach == "1":
                logging.info("Initializing RandomForest Chain Classifier (Chained Approach)...")
                model = rf.RandomForestChainClassifier(n_estimators=config.NUM_TREES,
                                                       max_depth=config.MAX_DEPTH,
                                                       min_samples_split=config.MIN_SAMPLES_SPLIT,
                                                       random_state=config.RANDOM_SEED)
            else:
                logging.info("Initializing Hierarchical Classifier (Hierarchical Approach)...")
                model = hm.HierarchicalClassifier(n_estimators=config.NUM_TREES,
                                                  max_depth=config.MAX_DEPTH,
                                                  min_samples_split=config.MIN_SAMPLES_SPLIT,
                                                  random_state=config.RANDOM_SEED)

            logging.info("Training model...")
            model.train(data_obj)

            logging.info("Predicting on test set...")
            predictions = model.predict(data_obj)

            logging.info("Evaluating model performance for %s embedding:", method.upper())
            model.print_results(data_obj)
            if config.PLOT_CONFUSION:
                evaluation.plot_confusion_matrices(data_obj.y_test, predictions, data_obj.label_encoders,
                                                   title_suffix=f"{method.upper()}_{'CHAIN' if approach == '1' else 'HIER'}")
            logging.info("=== Completed pipeline with %s embeddings ===", method.upper())
        config.EMBEDDING_METHOD = original_method  # Restore original setting
    else:
        embed_name = "TF-IDF" if config.EMBEDDING_METHOD.lower() == "tfidf" else "BERT"
        logging.info("Creating data object with %s features...", embed_name)
        data_obj = data.get_data_object(df_clean)

        if approach == "1":
            logging.info("Initializing RandomForest Chain Classifier (Chained Approach)...")
            model = rf.RandomForestChainClassifier(n_estimators=config.NUM_TREES,
                                                   max_depth=config.MAX_DEPTH,
                                                   min_samples_split=config.MIN_SAMPLES_SPLIT,
                                                   random_state=config.RANDOM_SEED)
        else:
            logging.info("Initializing Hierarchical Classifier (Hierarchical Approach)...")
            model = hm.HierarchicalClassifier(n_estimators=config.NUM_TREES,
                                              max_depth=config.MAX_DEPTH,
                                              min_samples_split=config.MIN_SAMPLES_SPLIT,
                                              random_state=config.RANDOM_SEED)

        logging.info("Training model...")
        model.train(data_obj)

        logging.info("Predicting on test set...")
        predictions = model.predict(data_obj)

        logging.info("Evaluating model performance...")
        model.print_results(data_obj)
        if config.PLOT_CONFUSION:
            evaluation.plot_confusion_matrices(data_obj.y_test, predictions, data_obj.label_encoders,
                                               title_suffix=f"{config.EMBEDDING_METHOD.upper()}_{'CHAIN' if approach == '1' else 'HIER'}")


if __name__ == "__main__":
    main()