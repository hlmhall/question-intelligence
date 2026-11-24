# BLUEPRINT.md

## 1. Project Goals

**Practical Foundations for Question Intelligence** aims to equip educational trainers and instructional designers with actionable insights into learner questions by leveraging NLP and data science. The project delivers a Python-based prototype that classifies learner queries, surfaces recurring topics, and visualizes lesson trends, all within a privacy-conscious, confidence-aware Streamlit dashboard. This tool is intended for EdTech practitioners and analytics professionals seeking to identify learning friction points and inform instructional interventions—while explicitly communicating the limitations and uncertainties of automated analysis.

---

## 2. Features

### Core (MVP)
1. **Load and Clean Synthetic Data**
   - Import and preprocess learner questions, lesson metadata, and anonymized user data from CSVs in `data/raw/`.
2. **TF-IDF Text Vectorization**
   - Transform learner questions into TF-IDF features for downstream modeling.
3. **NLP Text Classification**
   - Train and apply a scikit-learn classifier to categorize learner questions by topic or type.
4. **Clustering of Questions**
   - Group similar questions using unsupervised clustering to reveal recurring topics.
5. **Streamlit Dashboard**
   - Interactive web app (`app.py`) showing classified questions, clusters, and lesson trends.
6. **Confidence-Aware Predictions**
   - Display model confidence scores for each prediction, highlighting uncertainty.
7. **Privacy Safeguards**
   - Anonymize user data and clearly communicate privacy practices within the dashboard.
8. **Model Limitations Reporting**
   - Present a summary of model limitations and appropriate use warnings to trainers.

### Secondary
9. **User Feedback Mechanism**
   - Allow trainers to submit feedback on predictions for future model refinement.
10. **Intervention Indicators**
    - Compute and display actionable indicators (with confidence scores) to prompt instructional review.
11. **Model Validation and Error Analysis**
    - Evaluate model performance, show error rates, and visualize confusion matrices in the dashboard.
12. **Dashboard Usability Review**
    - Incorporate peer/trainer review cycles for dashboard clarity and practical value.

### Future Enhancements
13. **Automated Feedback Incorporation**
    - Periodically retrain or fine-tune models with aggregated user feedback.
14. **Advanced Topic Modeling**
    - Integrate LDA or other topic modeling techniques for deeper insights.
15. **Role-based Access Control**
    - Enable differentiated access for trainers, admins, and learners.
16. **Exportable Reports**
    - Allow users to export insights and intervention prompts as PDF or CSV.

---

## 3. User Stories

- **As a trainer**, I want to see a dashboard of classified learner questions so that I can quickly identify common areas of confusion.
- **As a trainer**, I want to view confidence scores for each prediction so that I know how much to trust the model's suggestions.
- **As a trainer**, I want to submit feedback on incorrect classifications so that the system can improve over time.
- **As an instructional designer**, I want to see clusters of recurring question topics so that I can adjust lesson materials proactively.
- **As a data privacy steward**, I want to ensure that all user data is anonymized so that learner privacy is protected.
- **As a trainer**, I want to see clear explanations of the model's limitations so that I avoid overreliance on automated outputs.
- **As a dashboard reviewer**, I want to provide usability feedback so that the product is practical and user-friendly.

---

## 4. Implementation Order

1. **Set up project structure and environment**  
   *Rationale:* Establishes a reproducible, organized codebase for all subsequent work.
2. **Load and clean synthetic datasets (`src/data/load.py`, `src/data/clean.py`)**  
   *Rationale:* Provides reliable, well-structured data for all downstream tasks.
3. **Perform exploratory data analysis (`notebooks/01_exploration.ipynb`)**  
   *Rationale:* Builds understanding of data distributions and informs feature engineering.
4. **Build TF-IDF and clustering features (`src/features/build_features.py`)**  
   *Rationale:* Converts raw text into usable features for modeling and analysis.
5. **Develop and train initial classifier (`src/models/train.py`)**  
   *Rationale:* Enables automated question categorization for the MVP.
6. **Evaluate model and generate confidence scores (`src/models/evaluate.py`)**  
   *Rationale:* Quantifies model performance and supports confidence-aware predictions.
7. **Implement privacy safeguards (`src/privacy/privacy.py`)**  
   *Rationale:* Ensures compliance and builds user trust from the outset.
8. **Develop Streamlit dashboard (`app.py`, `src/visualization/plots.py`)**  
   *Rationale:* Provides the primary user interface for insight delivery.
9. **Add model limitations and privacy reporting (`src/limitations/report.py`, dashboard sections)**  
   *Rationale:* Fulfills the project's ethical and educational commitments.
10. **Incorporate user feedback mechanisms (`src/feedback/feedback.py`)**  
    *Rationale:* Enables iterative improvement and aligns with real-world workflows.
11. **Compute and display intervention indicators (`src/intervention/indicators.py`)**  
    *Rationale:* Bridges analytics to actionable instructional support.
12. **Conduct usability review and refine dashboard**  
    *Rationale:* Ensures the product meets practical needs and is ready for showcase.

---

## 5. Data Models

### Entities

#### Question
| Field         | Type    | Constraints           | Description                                 |
|---------------|---------|----------------------|---------------------------------------------|
| id            | String  | PK                   | Unique identifier for each question         |
| user_id       | String  | FK → User.id         | Anonymized reference to the learner         |
| lesson_id     | String  | FK → Lesson.id       | Lesson context                              |
| text          | String  | Required             | Raw question text                           |
| timestamp     | DateTime|                      | When the question was asked                 |
| label         | String  | Optional             | Human-assigned or predicted label           |
| cluster_id    | Integer | Optional             | Cluster assignment                          |

#### Lesson
| Field         | Type    | Constraints           | Description                                 |
|---------------|---------|----------------------|---------------------------------------------|
| id            | String  | PK                   | Unique lesson identifier                    |
| title         | String  |                      | Lesson title                                |
| topic         | String  |                      | Main topic                                  |
| date          | Date    |                      | Lesson date                                 |

#### User
| Field         | Type    | Constraints           | Description                                 |
|---------------|---------|----------------------|---------------------------------------------|
| id            | String  | PK                   | Anonymized user ID                          |
| role          | String  |                      | Role (e.g., 'learner')                      |
| cohort        | String  |                      | Cohort or group                             |

#### Feedback
| Field         | Type    | Constraints           | Description                                 |
|---------------|---------|----------------------|---------------------------------------------|
| id            | String  | PK                   | Unique feedback entry                       |
| question_id   | String  | FK → Question.id     | Associated question                         |
| user_feedback | String  |                      | Feedback text or label                      |
| timestamp     | DateTime|                      | When feedback was submitted                 |

---

### Data Files

#### `data/raw/learner_questions.csv`
- **Columns:**  
  - `id` (String): Unique question ID  
  - `user_id` (String): Anonymized user reference  
  - `lesson_id` (String): Lesson context  
  - `text` (String): Question text  
  - `timestamp` (DateTime): When asked  
  - `label` (String, optional): Human or simulated label  
- **Rows:** 60-100  
- **Data Quality:**  
  - Some missing `label` values  
  - Occasional duplicate `text`  
  - Inconsistent timestamp formats  
  - A few empty `text` fields

#### `data/raw/lessons.csv`
- **Columns:**  
  - `id` (String): Lesson ID  
  - `title` (String): Lesson title  
  - `topic` (String): Main topic  
  - `date` (Date): Lesson date  
- **Rows:** 10-20  
- **Data Quality:**  
  - Some missing `topic` values  
  - Duplicate `title` entries

#### `data/raw/users.csv`
- **Columns:**  
  - `id` (String): Anonymized user ID  
  - `role` (String): User role  
  - `cohort` (String): Cohort/group  
- **Rows:** 30-40  
- **Data Quality:**  
  - Some duplicate `id` values (simulate data entry errors)  
  - Occasional missing `cohort`

---

## 6. Business Logic

- **Data Loading & Cleaning (`src/data/load.py`, `src/data/clean.py`):**  
  - Robustly load CSVs, handle missing/duplicate values, standardize timestamps, and anonymize user data.
- **Feature Engineering (`src/features/build_features.py`):**  
  - Apply TF-IDF vectorization to question texts; implement clustering (e.g., KMeans) to group similar questions.
- **Model Training & Prediction (`src/models/train.py`, `src/models/evaluate.py`):**  
  - Train scikit-learn classifiers (e.g., LogisticRegression) on labeled data; produce predictions with associated confidence scores (e.g., `predict_proba`).
- **Dashboard Visualization (`src/visualization/plots.py`, `app.py`):**  
  - Plot clusters, lesson trends, and prediction confidences using matplotlib/seaborn in Streamlit.
- **User Feedback Handling (`src/feedback/feedback.py`):**  
  - Allow trainers to provide feedback on predictions; aggregate and store for later model refinement.
- **Privacy & Compliance (`src/privacy/privacy.py`):**  
  - Ensure all user data is anonymized; check for privacy compliance before any data is displayed or exported.
- **Intervention Indicators (`src/intervention/indicators.py`):**  
  - Compute actionable signals (e.g., spike in low-confidence questions) for trainer review, with clear confidence bounds.
- **Model Limitations Reporting (`src/limitations/report.py`):**  
  - Generate and display plain-language summaries of model uncertainty, data coverage, and appropriate use.

---

## 7. Error Handling Strategy

- **Validation Errors:**  
  - Surface clear, actionable error messages in Streamlit when input data is missing, malformed, or fails validation (e.g., empty question text, non-CSV uploads).
- **Not-Found Errors:**  
  - Display user-friendly messages if requested data (e.g., lesson or user) does not exist.
- **Server/Processing Errors:**  
  - Log exceptions and display generic error messages in the dashboard, with instructions for seeking support or retrying.
- **Feedback Submission Errors:**  
  - Confirm successful feedback submission; if errors occur, inform the user and suggest corrective steps.
- **Privacy Compliance Errors:**  
  - Block display/export of any non-compliant data and require user acknowledgement.

All errors are surfaced via Streamlit notifications or dashboard sections to ensure transparency and maintain trust.

---

## 8. Security Considerations

- **Authentication/Authorization:**  
  - For prototype, no user authentication; all data is anonymized and synthetic.
- **Input Validation:**  
  - Strict validation of all uploaded or entered data (CSV schema, text length, allowed values).
- **Rate Limiting:**  
  - Not applicable for MVP; consider in future if public deployment.
- **Secrets Management:**  
  - No sensitive secrets in source; environment variables managed via `.env.example`.
- **Privacy:**  
  - All user data is anonymized; dashboard includes privacy notes and compliance checks.

---

## 9. Testing Strategy

- **Unit Tests (`tests/`):**  
  - Data loaders/cleaners: handle missing, duplicate, and malformed data (`test_data.py`)
  - Feature builders: correct TF-IDF and clustering output (`test_features.py`)
  - Model training/evaluation: reproducible results, confidence score logic (`test_models.py`)
  - Feedback handling: correct storage and retrieval (`test_feedback.py`)
  - Privacy safeguards: anonymization and compliance checks (`test_privacy.py`)
  - Visualization: plots render without error (`test_visualization.py`)

- **Integration Tests:**  
  - End-to-end pipeline: from raw CSV to dashboard output, ensuring correct flow and error handling.

- **Acceptance Criteria:**  
  - All MVP features work as described.
  - Dashboard loads and displays all required insights.
  - Model predictions include confidence and limitations.
  - User feedback can be submitted and is stored.
  - Privacy safeguards are active and effective.
  - All tests pass; manual test plan (`test_plan_summary.md`) criteria are met.

---

This blueprint ensures that each lesson’s deliverables are mapped to concrete features and milestones, scaffolding learners to build, test, and reflect on a practical, ethical, and privacy-conscious question intelligence system.