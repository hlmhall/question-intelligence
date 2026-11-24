# ARCHITECTURE.md

## 1. Project Overview

**Practical Foundations for Question Intelligence** is a Python-based learning analytics prototype designed to help educational trainers identify possible learning friction points by classifying learner questions, surfacing recurring topics, and tracking lesson trends. The application leverages NLP and clustering techniques, presenting results in a privacy-conscious, confidence-aware Streamlit dashboard. Its primary users are trainers and instructional designers seeking actionable insights for real-time feedback and intervention, while respecting data privacy and model limitations.

---

## 2. Tech Stack

- **Python 3.11+**  
  For modern language features, strong data science ecosystem, and broad compatibility.
- **pandas 2**  
  For efficient, flexible tabular data analysis and manipulation.
- **NumPy**  
  For fast numerical operations and foundational array support.
- **scikit-learn**  
  For robust machine learning, including TF-IDF vectorization, clustering, and model evaluation.
- **matplotlib / seaborn**  
  For high-quality, insightful data visualizations.
- **Jupyter**  
  For exploratory data analysis, prototyping, and iterative development.
- **pytest**  
  For structured, automated testing and validation of code and models.
- **Streamlit**  
  For rapid development of interactive, user-friendly analytics dashboards.

---

## 3. Directory Structure

```
question_intelligence_prototype/
├── app.py                         # Main Streamlit app entry point
├── requirements.txt
├── .gitignore
├── README.md                      # Project overview, setup, and usage instructions
├── ARCHITECTURE.md                # This document
├── data/
│   ├── raw/
│   │   ├── learner_questions.csv  # Synthetic sample of learner questions
│   │   ├── lessons.csv            # Synthetic sample of lesson metadata
│   │   └── users.csv              # Synthetic user metadata (anonymized)
│   ├── processed/                 # Output from data cleaning scripts
│   └── README.md                  # Data dictionary: files, columns, known issues
├── notebooks/
│   ├── 01_exploration.ipynb       # EDA of data/raw/
│   └── 02_modeling.ipynb          # Model training, evaluation, and validation
├── src/
│   ├── __init__.py
│   ├── config.py                  # RAW_DIR, PROCESSED_DIR, RANDOM_SEED
│   ├── data/
│   │   ├── load.py                # load_questions(), load_lessons(), load_users()
│   │   └── clean.py               # clean_questions(df), clean_all()
│   ├── features/
│   │   └── build_features.py      # build_tfidf_features(df), build_clustering_features(df)
│   ├── models/
│   │   ├── train.py               # train_classifier(X, y), save_model(model)
│   │   └── evaluate.py            # evaluate_model(model, X_test, y_test), get_confidence_scores(model, X)
│   ├── feedback/
│   │   └── feedback.py            # submit_feedback(question_id, user_feedback), aggregate_feedback()
│   ├── privacy/
│   │   └── privacy.py             # anonymize_data(df), check_privacy_compliance(df)
│   ├── intervention/
│   │   └── indicators.py          # compute_intervention_indicators(preds, confidences)
│   ├── limitations/
│   │   └── report.py              # generate_limitations_report(model, data)
│   ├── visualization/
│   │   └── plots.py               # plot_topic_clusters(df), plot_trends(df), plot_confidence(df)
│   └── utils.py                   # General-purpose utilities
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_features.py
│   ├── test_feedback.py
│   ├── test_privacy.py
│   └── test_visualization.py
├── .env.example                   # Sample environment variable file
├── test_plan_summary.md           # Manual test plan and acceptance criteria
├── milestone_board.md             # Build board with milestone criteria
└── reflection_journal.md          # Learner’s reflection and design rationale
```

---

## 4. Key Components

### `app.py`
- Streamlit application entry point: loads models, data, and visualizations; handles user interaction, feedback, and presents intervention indicators, limitations, and privacy notes.

### `src/config.py`
- **Exports:** `RAW_DIR`, `PROCESSED_DIR`, `RANDOM_SEED`
- Centralizes all configurable paths and random seeds for reproducibility.

### `src/data/load.py`
- **Exports:** `load_questions()`, `load_lessons()`, `load_users()`
- Loads raw CSV data into DataFrames for downstream processing.

### `src/data/clean.py`
- **Exports:** `clean_questions(df)`, `clean_all()`
- Cleans and preprocesses raw data, writes processed outputs.

### `src/features/build_features.py`
- **Exports:** `build_tfidf_features(df)`, `build_clustering_features(df)`
- Transforms text data into machine learning features (TF-IDF, clustering).

### `src/models/train.py`
- **Exports:** `train_classifier(X, y)`, `save_model(model)`
- Trains text classification models and persists them for inference.

### `src/models/evaluate.py`
- **Exports:** `evaluate_model(model, X_test, y_test)`, `get_confidence_scores(model, X)`
- Evaluates models, computes accuracy, confusion matrices, and prediction confidences.

### `src/feedback/feedback.py`
- **Exports:** `submit_feedback(question_id, user_feedback)`, `aggregate_feedback()`
- Handles user feedback submission and aggregates for model refinement.

### `src/privacy/privacy.py`
- **Exports:** `anonymize_data(df)`, `check_privacy_compliance(df)`
- Implements privacy safeguards and compliance checks for data handling.

### `src/intervention/indicators.py`
- **Exports:** `compute_intervention_indicators(preds, confidences)`
- Calculates actionable intervention indicators for trainers, factoring in model confidence.

### `src/limitations/report.py`
- **Exports:** `generate_limitations_report(model, data)`
- Produces a report summarizing known model limitations for trainer review.

### `src/visualization/plots.py`
- **Exports:** `plot_topic_clusters(df)`, `plot_trends(df)`, `plot_confidence(df)`
- Generates interactive visualizations for the Streamlit dashboard.

### `src/utils.py`
- General-purpose utility functions shared across modules.

### `notebooks/01_exploration.ipynb`
- Exploratory data analysis, initial data understanding, and hypothesis generation.

### `notebooks/02_modeling.ipynb`
- Model prototyping, validation, and iterative refinement.

### `tests/`
- Automated tests for data loading, feature engineering, modeling, feedback, privacy, and visualization modules.

---

## 5. Data Flow

```text
[User uploads/interacts via Streamlit app]
         |
         v
[app.py] -- loads --> [src/data/load.py] --> [src/data/clean.py]
         |
         v
[src/features/build_features.py] -- TF-IDF, clustering
         |
         v
[src/models/train.py] / [src/models/evaluate.py] -- predictions, confidence scores
         |
         v
[src/intervention/indicators.py] -- computes intervention indicators
         |
         v
[src/visualization/plots.py] -- generates dashboard visualizations
         |
         v
[app.py] -- presents results, limitations ([src/limitations/report.py]), privacy notes ([src/privacy/privacy.py]), and collects user feedback ([src/feedback/feedback.py])
         |
         v
[src/feedback/feedback.py] -- aggregates feedback for model improvement
```

**Summary Steps:**
1. Data loaded and cleaned.
2. Features built (TF-IDF, clustering).
3. Model trained or loaded; predictions and confidences generated.
4. Intervention indicators computed.
5. Visualizations and reports generated.
6. User interacts via Streamlit, provides feedback.
7. Feedback and privacy safeguards integrated for iterative improvement.

---

## 6. API Design

*Not applicable: All functionality is exposed via the Streamlit interface. There are no public REST or HTTP APIs.*

---

## 7. Environment Variables

| Variable         | Default                 | Required | Description                                         |
|------------------|------------------------|----------|-----------------------------------------------------|
| RAW_DIR          | data/raw/              | Yes      | Path to raw input data CSVs                         |
| PROCESSED_DIR    | data/processed/        | Yes      | Path to store processed data                        |
| RANDOM_SEED      | 42                     | No       | Random seed for reproducibility                     |
| FEEDBACK_STORE   | feedback_store.json    | No       | Path to user feedback storage                       |
| MODEL_PATH       | models/classifier.pkl  | Yes      | Path to save/load trained model                     |
| STREAMLIT_PORT   | 8501                   | No       | Port for running the Streamlit app                  |

*(Add these to `.env.example` as needed.)*

---

This architecture supports the lesson-by-lesson progression: from dataset exploration and cleaning, through feature engineering and modeling, to dashboard deployment, validation, and continuous improvement via user feedback—all underpinned by privacy and model limitations reporting, as required by the curriculum.