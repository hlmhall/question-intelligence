# Practical Foundations for Question Intelligence

A reproducible Python data science prototype for learning analytics, designed to help educational trainers and instructional designers identify learning friction points by classifying learner questions, surfacing recurring topics, and tracking lesson trends. The project leverages NLP, clustering, and confidence-aware modeling, all presented in a privacy-conscious Streamlit dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
  - [1. Data Exploration](#1-data-exploration)
  - [2. Data Cleaning](#2-data-cleaning)
  - [3. Feature Engineering & Modeling](#3-feature-engineering--modeling)
  - [4. Dashboard](#4-dashboard)
  - [5. User Feedback](#5-user-feedback)
- [Configuration](#configuration)
- [Testing](#testing)
- [Data Dictionary](#data-dictionary)
- [Model Limitations & Privacy](#model-limitations--privacy)
- [Milestones & Reflection](#milestones--reflection)
- [License](#license)

---

## Project Overview

**Practical Foundations for Question Intelligence** is a Python-based prototype that enables EdTech practitioners to:

- **Classify** learner questions by topic/type using NLP and scikit-learn.
- **Cluster** similar questions to reveal recurring themes.
- **Visualize** lesson trends and model confidence in an interactive Streamlit dashboard.
- **Safeguard privacy** by anonymizing user data and enforcing compliance.
- **Incorporate feedback** from trainers for iterative model improvement.
- **Report limitations** and uncertainties to avoid overreliance on automation.

All data is synthetic and anonymized for privacy and reproducibility.

---

## Features

- **NLP Text Classification** of learner questions
- **TF-IDF Vectorization** for feature extraction
- **Clustering** to surface recurring topics
- **Confidence-Aware Predictions** with clear uncertainty reporting
- **Streamlit Dashboard** for interactive analytics and feedback
- **Privacy Safeguards** and compliance checks
- **Model Limitations Reporting** in plain language
- **User Feedback Mechanism** for continuous improvement
- **Milestone-based Build Board** and manual test plan

---

## Project Structure

```
question_intelligence_prototype/
├── app.py
├── requirements.txt
├── .gitignore
├── README.md
├── ARCHITECTURE.md
├── data/
│   ├── raw/
│   │   ├── learner_questions.csv
│   │   ├── lessons.csv
│   │   └── users.csv
│   ├── processed/
│   └── README.md
├── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data/
│   │   ├── load.py
│   │   └── clean.py
│   ├── features/
│   │   └── build_features.py
│   ├── models/
│   │   ├── train.py
│   │   └── evaluate.py
│   ├── feedback/
│   │   └── feedback.py
│   ├── privacy/
│   │   └── privacy.py
│   ├── intervention/
│   │   └── indicators.py
│   ├── limitations/
│   │   └── report.py
│   ├── visualization/
│   │   └── plots.py
│   └── utils.py
├── tests/
│   ├── test_data.py
│   ├── test_models.py
│   ├── test_features.py
│   ├── test_feedback.py
│   ├── test_privacy.py
│   └── test_visualization.py
├── .env.example
├── test_plan_summary.md
├── milestone_board.md
└── reflection_journal.md
```

---

## Setup & Installation

### Requirements

- **Python**: 3.11 or higher
- **pip**: for package management

### 1. Clone the repository

```bash
git clone https://github.com/your-org/question_intelligence_prototype.git
cd question_intelligence_prototype
```

### 2. Create and activate a virtual environment

```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Copy `.env.example` to `.env` and adjust as needed:

```bash
cp .env.example .env
```

---

## Usage

### 1. Data Exploration

Open the exploratory notebook:

```bash
cd notebooks
jupyter notebook 01_exploration.ipynb
```

- The notebook will load and explore the synthetic data in `data/raw/`.

### 2. Data Cleaning

Clean and preprocess the data:

```bash
python -m src.data.clean
```

- This will read from `data/raw/` and write cleaned files to `data/processed/`.

### 3. Feature Engineering & Modeling

Open the modeling notebook:

```bash
cd notebooks
jupyter notebook 02_modeling.ipynb
```

- This notebook builds features, trains models, and evaluates performance.

### 4. Dashboard

Launch the Streamlit dashboard:

```bash
streamlit run app.py
```

- The dashboard loads processed data, displays classified questions, clusters, trends, confidence scores, privacy notes, and model limitations.
- Default port is `8501` (configurable via `STREAMLIT_PORT`).

### 5. User Feedback

- Trainers can submit feedback on predictions directly in the dashboard.
- Feedback is stored in the location specified by `FEEDBACK_STORE` (default: `feedback_store.json`).

---

## Configuration

All configurable paths and settings are managed via environment variables (see `.env.example`):

| Variable         | Default                 | Description                                         |
|------------------|------------------------|-----------------------------------------------------|
| RAW_DIR          | data/raw/              | Path to raw input data CSVs                         |
| PROCESSED_DIR    | data/processed/        | Path to store processed data                        |
| RANDOM_SEED      | 42                     | Random seed for reproducibility                     |
| FEEDBACK_STORE   | feedback_store.json    | Path to user feedback storage                       |
| MODEL_PATH       | models/classifier.pkl  | Path to save/load trained model                     |
| STREAMLIT_PORT   | 8501                   | Port for running the Streamlit app                  |

---

## Testing

Run all automated tests with:

```bash
pytest
```

- Tests cover data loading/cleaning, feature engineering, modeling, feedback, privacy, and visualization.
- Manual test plan and acceptance criteria are in `test_plan_summary.md`.

---

## Data Dictionary

See `data/README.md` for a full data dictionary.

### Main Data Files

#### `data/raw/learner_questions.csv`
| Column     | Description                        |
|------------|------------------------------------|
| id         | Unique question ID                 |
| user_id    | Anonymized user reference          |
| lesson_id  | Lesson context                     |
| text       | Question text                      |
| timestamp  | When asked (may be inconsistent)   |
| label      | Human/simulated label (optional)   |

#### `data/raw/lessons.csv`
| Column     | Description                        |
|------------|------------------------------------|
| id         | Lesson ID                          |
| title      | Lesson title                       |
| topic      | Main topic                         |
| date       | Lesson date                        |

#### `data/raw/users.csv`
| Column     | Description                        |
|------------|------------------------------------|
| id         | Anonymized user ID                 |
| role       | User role                          |
| cohort     | Cohort/group                       |

- Data includes deliberate quality issues (missing values, duplicates, inconsistent formats) for realism.

---

## Model Limitations & Privacy

- **Model Limitations:**  
  - Predictions are based on synthetic, limited data and may not generalize.
  - Confidence scores are provided; low-confidence predictions should be reviewed by trainers.
  - See dashboard and `src/limitations/report.py` for a full limitations report.

- **Privacy:**  
  - All user data is anonymized.
  - Privacy compliance is checked before any data is displayed or exported.
  - See dashboard and `src/privacy/privacy.py` for details.

---

## Milestones & Reflection

- **Milestone board:** See `milestone_board.md` for build progress and criteria.
- **Reflection journal:** See `reflection_journal.md` for design rationale and learning reflections.

---

## License

This project is for educational and demonstration purposes only.  
All data is synthetic and anonymized.  
See `LICENSE` for details.

---

**Questions or feedback?**  
Open an issue or contact the maintainers.