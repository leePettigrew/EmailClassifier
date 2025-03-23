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


def main():
    # Step 1: Load raw data
    print("Loading raw data...")
    df_raw = preprocess.get_input_data()

    # Step 2: Deduplicate email interaction content
    print("Performing deduplication...")
    df_dedup = preprocess.de_duplication(df_raw)

    # Step 3: Remove noise from text
    print("Removing noise from text...")
    df_clean = preprocess.noise_remover(df_dedup)

    # Optional: Translate texts to English if necessary (uncomment if needed)
    # print("Translating texts to English...")
    # df_clean[config.INTERACTION_CONTENT_COL] = preprocess.translate_to_en(df_clean[config.INTERACTION_CONTENT_COL].tolist())

    # Save preprocessed output for verification
    preprocessed_output_path = "preprocessed_data.csv"
    df_clean.to_csv(preprocessed_output_path, index=False)
    print(f"Preprocessed data saved to {preprocessed_output_path}")

    # Step 4: Create the Data object (splitting and TF-IDF embedding)
    print("Creating data object with TF-IDF features...")
    data_obj = data.get_data_object(df_clean)

    # Step 5: Initialize the RandomForestChainClassifier model
    print("Initializing RandomForest Chain Classifier...")
    model = rf.RandomForestChainClassifier(n_estimators=config.NUM_TREES, random_state=config.RANDOM_SEED)

    # Step 6: Train the model using the training data
    print("Training model...")
    model.train(data_obj)

    # Step 7: Generate predictions on the test set
    print("Predicting on test set...")
    predictions = model.predict(data_obj)

    # Step 8: Evaluate and print results
    print("Evaluating model performance...")
    model.print_results(data_obj)


if __name__ == "__main__":
    main()
