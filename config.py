"""
config.py

 - Separation of concerns: configuration is isolated from processing logic
 - Modularity: changes here dont require altering other modules
 - Maintainability: easy to add new config items for future expansions
 ish ~
"""

import os

# -----------------------------------------------------------------------------
# File paths
# -----------------------------------------------------------------------------

BASE_DIR = os.path.dirname(__file__)          # Directory containing config.py
DATA_DIR = os.path.join(BASE_DIR, "data")     # Data folder inside the project

# CSV files
DATA_FILE_1 = os.path.join(DATA_DIR, "AppGallery.csv")
DATA_FILE_2 = os.path.join(DATA_DIR, "Purchasing.csv")

# -----------------------------------------------------------------------------
# Data columns
# -----------------------------------------------------------------------------
# Text column name in the CSV files. Adjust if your CSV has a different column name for email text.
TICKET_ID_COL = "Ticket id"
INTERACTION_ID_COL = "Interaction id"
INTERACTION_DATE_COL = "Interaction date"
MAILBOX_COL = "Mailbox"
TICKET_SUMMARY_COL = "Ticket Summary"
INTERACTION_CONTENT_COL = "Interaction content"
TYPOLOGY_COL = "Innso TYPOLOGY_TICKET"

# Labels (we are ignoring Type1 as it has only one class)
TYPE1_COL = "Type 1"
TYPE2_COL = "Type 2"
TYPE3_COL = "Type 3"
TYPE4_COL = "Type 4"

# -----------------------------------------------------------------------------
# General settings
# -----------------------------------------------------------------------------
# Train/test split ratio
TEST_SIZE = 0.2          # 20% of data goes to test set
RANDOM_SEED = 42         # For reproducible shuffling & model behavior

# -----------------------------------------------------------------------------
# Embedding settings
# -----------------------------------------------------------------------------
# Choose embedding method: "tfidf" for TF-IDF, "bert" for BERT sentence embeddings
EMBEDDING_METHOD = "tfidf"

# Pre-trained model name for SentenceTransformer (used if EMBEDDING_METHOD="bert")
BERT_MODEL_NAME = "all-MiniLM-L6-v2"

# If True, run the pipeline twice to compare TF-IDF and BERT embeddings
COMPARE_EMBEDDINGS = False

# TF-IDF embedding parameters (only used if EMBEDDING_METHOD="tfidf")
MAX_FEATURES = 1000      # Limit on the number of most frequent terms in TF-IDF
STOP_WORDS = "english"   # Remove common English stopwords

# -----------------------------------------------------------------------------
# RandomForest hyperparameters
# -----------------------------------------------------------------------------
NUM_TREES = 100          # Number of trees in each RandomForest classifier (n_estimators)
MAX_DEPTH = None         # Maximum depth of each tree (None for no limit)
MIN_SAMPLES_SPLIT = 2    # Minimum samples required to split an internal node

# -----------------------------------------------------------------------------
# Output / Results
# -----------------------------------------------------------------------------
EXPORT_RESULTS = False            # If True, export final metrics to CSV
RESULTS_FILE = "evaluation_results.csv"
PLOT_CONFUSION = True             # If True, display confusion matrix plots for each label