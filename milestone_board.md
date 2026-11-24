# Milestone Board

**Project:** Practical Foundations for Question Intelligence  
**Purpose:** Track progress, deliverables, and acceptance criteria for each milestone in the data science project lifecycle.

---

## Milestone 1: Project Setup & Environment

**Objectives:**
- Establish reproducible project structure.
- Configure environment and dependencies.
- Prepare initial documentation.

**Deliverables:**
- Directory structure as per `ARCHITECTURE.md`.
- `requirements.txt` with pinned versions.
- `.env.example` with all required variables.
- `README.md` with setup and usage instructions.
- Initial `ARCHITECTURE.md` and `BLUEPRINT.md`.

**Acceptance Criteria:**
- Project installs and runs in a clean environment.
- All paths and config variables resolve correctly.
- Documentation is clear and matches the codebase.

---

## Milestone 2: Data Loading & Cleaning

**Objectives:**
- Load and validate all raw data files.
- Clean and preprocess data for downstream use.

**Deliverables:**
- `src/data/load.py` with `load_questions()`, `load_lessons()`, `load_users()`.
- `src/data/clean.py` with `clean_questions(df)`, `clean_all()`.
- Cleaned data written to `data/processed/`.
- `data/README.md` with data dictionary and known issues.

**Acceptance Criteria:**
- All raw CSVs load without error.
- Cleaning handles missing, duplicate, and malformed data as described.
- Processed data is ready for feature engineering.
- Unit tests in `tests/test_data.py` and `tests/test_features.py` pass.

---

## Milestone 3: Exploratory Data Analysis (EDA)

**Objectives:**
- Understand data distributions, quality, and patterns.
- Identify hypotheses for modeling.

**Deliverables:**
- `notebooks/01_exploration.ipynb` with:
  - Data summaries and visualizations.
  - Notes on data quality and initial insights.

**Acceptance Criteria:**
- Notebook runs top-to-bottom without error.
- Key data issues and trends are documented.
- EDA informs feature engineering and modeling steps.

---

## Milestone 4: Feature Engineering

**Objectives:**
- Transform text data into ML-ready features.
- Implement clustering for topic discovery.

**Deliverables:**
- `src/features/build_features.py` with `build_tfidf_features(df)`, `build_clustering_features(df)`.
- Feature matrices saved for modeling.
- Visualizations of clusters in EDA notebook.

**Acceptance Criteria:**
- TF-IDF and clustering features are correctly computed.
- Features align with model input requirements.
- Unit tests in `tests/test_features.py` pass.

---

## Milestone 5: Model Training & Evaluation

**Objectives:**
- Train initial text classification model.
- Evaluate performance and generate confidence scores.

**Deliverables:**
- `src/models/train.py` with `train_classifier(X, y)`, `save_model(model)`.
- `src/models/evaluate.py` with `evaluate_model(model, X_test, y_test)`, `get_confidence_scores(model, X)`.
- `notebooks/02_modeling.ipynb` with model results and error analysis.
- Trained model saved to `models/classifier.pkl`.

**Acceptance Criteria:**
- Model achieves reasonable accuracy on synthetic data.
- Confidence scores are available for all predictions.
- Evaluation metrics and confusion matrix are visualized.
- Unit tests in `tests/test_models.py` pass.

---

## Milestone 6: Privacy Safeguards

**Objectives:**
- Ensure all user data is anonymized and privacy-compliant.

**Deliverables:**
- `src/privacy/privacy.py` with `anonymize_data(df)`, `check_privacy_compliance(df)`.
- Privacy notes and compliance checks integrated into dashboard.

**Acceptance Criteria:**
- No personally identifiable information is exposed.
- Privacy checks block non-compliant data from display/export.
- Unit tests in `tests/test_privacy.py` pass.

---

## Milestone 7: Streamlit Dashboard MVP

**Objectives:**
- Build interactive dashboard for trainers and designers.

**Deliverables:**
- `app.py` as Streamlit entry point.
- Visualizations from `src/visualization/plots.py`.
- Dashboard sections: classified questions, clusters, lesson trends, confidence scores, privacy notes, model limitations.

**Acceptance Criteria:**
- Dashboard loads and displays all required insights.
- User can interact with data and view model outputs.
- Visualizations render without error.
- Unit tests in `tests/test_visualization.py` pass.

---

## Milestone 8: Model Limitations & Reporting

**Objectives:**
- Communicate model limitations and appropriate use.

**Deliverables:**
- `src/limitations/report.py` with `generate_limitations_report(model, data)`.
- Dashboard section summarizing limitations and uncertainties.

**Acceptance Criteria:**
- Limitations report is clear, accurate, and visible in dashboard.
- Trainers are warned against overreliance on automated outputs.

---

## Milestone 9: User Feedback Mechanism

**Objectives:**
- Enable trainers to submit feedback on predictions.

**Deliverables:**
- `src/feedback/feedback.py` with `submit_feedback(question_id, user_feedback)`, `aggregate_feedback()`.
- Feedback form and aggregation in dashboard.
- Feedback stored in `feedback_store.json` or configured path.

**Acceptance Criteria:**
- Feedback can be submitted and is persisted.
- Aggregated feedback is accessible for model refinement.
- Unit tests in `tests/test_feedback.py` pass.

---

## Milestone 10: Intervention Indicators

**Objectives:**
- Compute and display actionable intervention signals.

**Deliverables:**
- `src/intervention/indicators.py` with `compute_intervention_indicators(preds, confidences)`.
- Dashboard section highlighting intervention prompts with confidence scores.

**Acceptance Criteria:**
- Intervention indicators are computed and displayed.
- Trainers can identify lessons or topics needing review.

---

## Milestone 11: Usability Review & Refinement

**Objectives:**
- Ensure dashboard is practical, clear, and user-friendly.

**Deliverables:**
- Peer/trainer usability review session.
- Dashboard refinements based on feedback.
- Updated documentation and reflection in `reflection_journal.md`.

**Acceptance Criteria:**
- Review feedback is addressed.
- Dashboard meets user needs and is ready for demonstration.

---

## Milestone 12: Testing & Acceptance

**Objectives:**
- Validate all features and ensure project quality.

**Deliverables:**
- All unit and integration tests in `tests/` pass.
- Manual test plan in `test_plan_summary.md` completed.
- Final acceptance review.

**Acceptance Criteria:**
- All acceptance criteria from previous milestones are met.
- Project is reproducible, robust, and ready for showcase.

---

**Legend:**  
✅ = Complete 🟡 = In Progress ⬜ = Not Started

| Milestone | Status | Owner | Notes |
|-----------|--------|-------|-------|
| 1. Setup & Env           | ⬜      |       |       |
| 2. Data Load & Clean     | ⬜      |       |       |
| 3. EDA                   | ⬜      |       |       |
| 4. Feature Engineering   | ⬜      |       |       |
| 5. Model Train/Eval      | ⬜      |       |       |
| 6. Privacy Safeguards    | ⬜      |       |       |
| 7. Dashboard MVP         | ⬜      |       |       |
| 8. Limitations Reporting | ⬜      |       |       |
| 9. Feedback Mechanism    | ⬜      |       |       |
|10. Intervention Indics.  | ⬜      |       |       |
|11. Usability Review      | ⬜      |       |       |
|12. Testing & Acceptance  | ⬜      |       |       |

---

**Instructions:**  
- Update the status and notes columns as each milestone progresses.  
- Use this board to coordinate team efforts and ensure all acceptance criteria are met before moving to the next milestone.  
- For detailed test cases and manual checks, refer to `test_plan_summary.md`.  
- For design rationale and lessons learned, see `reflection_journal.md`.