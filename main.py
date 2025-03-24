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
import config
import evaluation
import logging

def main():
    # Configure logging format and level
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Step 1: Load raw data
    logging.info("Loading raw data...")
    df_raw = preprocess.get_input_data()

    # Step 2: Deduplicate email interaction content
    logging.info("Performing deduplication...")
    df_dedup = preprocess.de_duplication(df_raw)

    # Step 3: Remove noise from text
    logging.info("Removing noise from text...")
    df_clean = preprocess.noise_remover(df_dedup)

    # (Optional translation step omitted for brevity; can be inserted here if needed)

    # Save preprocessed output for verification/debugging
    preprocessed_output_path = "preprocessed_data.csv"
    df_clean.to_csv(preprocessed_output_path, index=False)
    logging.info(f"Preprocessed data saved to {preprocessed_output_path}")

    # Step 4 onward: Depending on config, either compare both embeddings or run single pipeline
    if config.COMPARE_EMBEDDINGS:
        original_method = config.EMBEDDING_METHOD
        for method in ["tfidf", "bert"]:
            config.EMBEDDING_METHOD = method
            logging.info(f"--- Running pipeline with {method.upper()} embeddings ---")
            logging.info(f"Creating data object with {method.upper()} features...")
            data_obj = data.get_data_object(df_clean)

            logging.info("Initializing RandomForest Chain Classifier...")
            model = rf.RandomForestChainClassifier(n_estimators=config.NUM_TREES,
                                                   max_depth=config.MAX_DEPTH,
                                                   min_samples_split=config.MIN_SAMPLES_SPLIT,
                                                   random_state=config.RANDOM_SEED)
            logging.info("Training model...")
            model.train(data_obj)

            logging.info("Predicting on test set...")
            predictions = model.predict(data_obj)

            logging.info("Evaluating model performance...")
            logging.info(f"Results for {method.upper()} embedding:")
            model.print_results(data_obj)
            if config.PLOT_CONFUSION:
                evaluation.plot_confusion_matrices(data_obj.y_test, predictions, data_obj.label_encoders, title_suffix=method.upper())
        config.EMBEDDING_METHOD = original_method  # restore original setting if changed
    else:
        # Single embedding run as per EMBEDDING_METHOD
        embed_name = "TF-IDF" if config.EMBEDDING_METHOD.lower() == "tfidf" else "BERT"
        logging.info(f"Creating data object with {embed_name} features...")
        data_obj = data.get_data_object(df_clean)

        logging.info("Initializing RandomForest Chain Classifier...")
        model = rf.RandomForestChainClassifier(n_estimators=config.NUM_TREES,
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
            evaluation.plot_confusion_matrices(data_obj.y_test, predictions, data_obj.label_encoders, title_suffix=config.EMBEDDING_METHOD.upper())

if __name__ == "__main__":
    main()
