# 📧 Multi‑Label Email Classification

A modular, production‑ready Python pipeline for **multi‑label email classification**, comparing two architectural strategies:

1. **Chained Multi‑Output Classification**  
2. **Hierarchical Modeling**

This project demonstrates clean separation of concerns, consistent data interfaces, and easy swapping of embedding methods (TF‑IDF vs BERT). It’s built to industry standards and fulfills the Continuous Assessment brief from National College of Ireland.

---

##  Key Features

- ✅ **Two Architectural Approaches**  
  - **Chained**: Single classifier chain predicts Type2 → Type3 → Type4  
  - **Hierarchical**: Separate classifiers per class at each stage  

- ✅ **Flexible Embeddings**  
  - TF‑IDF (fast, interpretable)  
  - BERT via SentenceTransformer (contextual, semantic)

- ✅ **Interactive CLI**  
  Configure embedding, hyperparameters, evaluation options at runtime

- ✅ **Robust Preprocessing**  
  Deduplication, noise removal, optional translation

- ✅ **Comprehensive Evaluation**  
  Stage‑wise accuracy, chained/hierarchical accuracy, confusion matrix visualizations

---

## 📁 Repository Structure

```
Email_classifier/
├── data/                   # Raw CSVs (AppGallery.csv, Purchasing.csv)
├── preprocess.py           # Data cleaning & deduplication
├── embeddings.py           # TF‑IDF & BERT embedding
├── data.py                 # Train/test split & label encoding
├── randomforest.py         # Chained multi-output model
├── hierarchical.py         # Hierarchical modeling
├── evaluation.py           # Accuracy & confusion matrix plots
├── main.py                 # Interactive driver script
├── config.py               # All configuration constants
├── README.md               # This documentation
└── requirements.txt        # Python dependencies
```

---

## ⚙️ Installation

1. Clone repo  
   ```bash
   git clone https://github.com/leePettigrew/EmailClassifier.git
   cd EmailClassifier
   ```
2. Create & activate virtual environment  
   ```bash
   python -m venv .venv
   .venv\Scripts\activate  # Windows
   ```
3. Install dependencies  
   ```bash
   pip install -r requirements.txt
   ```

---

##  Usage

Run the interactive CLI:

```bash
python main.py
```

You’ll be prompted to choose:

| Prompt | Example Input | Default |
|--------|---------------|---------|
| Modeling approach | `1` (Chained) or `2` (Hierarchical) | 1 |
| Embedding method | `tfidf` or `bert` | tfidf |
| Compare embeddings? | `y` / `n` | n |
| Plot confusion matrices? | `y` / `n` | y |
| RandomForest `n_estimators` | integer | 100 |
| RandomForest `max_depth` | integer or `None` | None |
| RandomForest `min_samples_split` | integer | 2 |

Results (accuracy & confusion matrices) appear in console and pop‑up charts.

---

##  Example Results

| Approach | Embedding | Type2 Accuracy | Type3∣Type2 | Type4∣Type2,3 | Overall Accuracy |
|----------|-----------|----------------|-------------|--------------|------------------|
| Chained  | TF‑IDF    | 76.19%         | 34.38%      | 9.09%        | 34.92%           |
| Chained  | BERT      | 73.81%         | 29.03%      | 11.11%       | 32.54%           |
| Hierarchical | TF‑IDF | 76.19%        | 37.50%      | 16.67%       | 36.51%           |
| Hierarchical | BERT   | 73.81%        | 29.03%      | 11.11%       | 32.54%           |

---
Made by Lee Pettigrew - X20730039