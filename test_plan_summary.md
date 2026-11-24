# Manual Test Plan & Acceptance Criteria

**Project:** Practical Foundations for Question Intelligence  
**Purpose:** This document defines the manual test plan and acceptance criteria for the prototype, ensuring all core features, privacy safeguards, and usability requirements are met.

---

## 1. Test Environment

- **Python Version:** 3.11+
- **Dependencies:** As specified in `requirements.txt`
- **Data:** Synthetic CSVs in `data/raw/`
- **Configuration:** Environment variables as per `.env.example`
- **Interface:** Streamlit dashboard (`app.py`)
- **Test Tools:** Manual steps, plus automated tests in `tests/`

---

## 2. Test Scenarios & Steps

### 2.1 Data Loading & Cleaning

**Objective:** Ensure raw data loads correctly, is cleaned, and processed data is saved.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Delete all files in `data/processed/` (if any). | `data/processed/` is empty. |
| 2 | Run `src/data/load.py` functions via notebook or REPL. | DataFrames load with correct columns and row counts. |
| 3 | Run `clean_all()` from `src/data/clean.py`. | Cleaned CSVs appear in `data/processed/`. |
| 4 | Inspect processed files for: <br> - No duplicate questions/users <br> - Standardized timestamps <br> - No empty question texts | All issues resolved as per cleaning logic. |

### 2.2 Feature Engineering

**Objective:** Verify TF-IDF and clustering features are built as expected.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Call `build_tfidf_features(df)` on cleaned questions. | Returns TF-IDF matrix with correct shape. |
| 2 | Call `build_clustering_features(df)` on same data. | Returns cluster assignments; similar questions grouped. |

### 2.3 Model Training & Evaluation

**Objective:** Confirm classifier trains, predicts, and evaluates with confidence scores.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `train_classifier(X, y)` on labeled data. | Model trains without error; model object returned. |
| 2 | Save model with `save_model(model)`. | Model file appears at `MODEL_PATH`. |
| 3 | Run `evaluate_model(model, X_test, y_test)`. | Accuracy, confusion matrix, and metrics displayed. |
| 4 | Run `get_confidence_scores(model, X)`. | Confidence/probability scores returned for each prediction. |

### 2.4 Streamlit Dashboard

**Objective:** Ensure dashboard loads, displays all sections, and handles user interaction.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Start app: `streamlit run app.py` | Dashboard loads at `localhost:STREAMLIT_PORT`. |
| 2 | View classified questions, clusters, and lesson trends. | All sections render with data and visualizations. |
| 3 | Click on a question to view prediction and confidence. | Confidence score shown; low-confidence flagged. |
| 4 | Submit feedback on a prediction. | Confirmation message appears; feedback stored. |
| 5 | Review privacy and limitations sections. | Clear, accurate notes shown; privacy compliance indicated. |
| 6 | Trigger error (e.g., upload malformed CSV). | User-friendly error message displayed; app remains stable. |

### 2.5 Privacy Safeguards

**Objective:** Validate anonymization and privacy compliance.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Inspect displayed user data in dashboard. | No real names or identifying info shown. |
| 2 | Run `anonymize_data(df)` on raw user/question data. | All IDs anonymized; no PII present. |
| 3 | Run `check_privacy_compliance(df)` on processed data. | Returns True; dashboard indicates compliance. |
| 4 | Attempt to export or display non-compliant data (simulate). | Export blocked; warning shown to user. |

### 2.6 Feedback Handling

**Objective:** Ensure feedback can be submitted and aggregated.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Submit feedback via dashboard on a question. | Feedback stored in `FEEDBACK_STORE`. |
| 2 | Run `aggregate_feedback()`. | Aggregated feedback summary returned; visible in dashboard. |
| 3 | Submit invalid/empty feedback. | Error message prompts for correction. |

### 2.7 Intervention Indicators

**Objective:** Confirm actionable indicators are computed and displayed.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `compute_intervention_indicators(preds, confidences)`. | Returns indicators (e.g., spike in low-confidence questions). |
| 2 | View indicators in dashboard. | Clear, actionable prompts shown to trainer. |

### 2.8 Model Limitations Reporting

**Objective:** Ensure limitations are generated and visible.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Run `generate_limitations_report(model, data)`. | Plain-language summary of model limitations produced. |
| 2 | View limitations section in dashboard. | Limitations and appropriate use warnings displayed. |

### 2.9 Visualization

**Objective:** Validate all plots render and update as expected.

| Step | Action | Expected Result |
|------|--------|----------------|
| 1 | Call `plot_topic_clusters(df)` in dashboard. | Cluster plot renders; clusters are visually distinct. |
| 2 | Call `plot_trends(df)` and `plot_confidence(df)`. | Trend and confidence plots render without error. |
| 3 | Interact with dashboard filters (if present). | Plots update dynamically; no errors. |

---

## 3. Acceptance Criteria Checklist

- [ ] **Data loading and cleaning**: All raw data loads, is cleaned, and processed files are correct.
- [ ] **Feature engineering**: TF-IDF and clustering features are built and valid.
- [ ] **Model training and evaluation**: Classifier trains, predicts, and evaluates with confidence scores.
- [ ] **Dashboard usability**: All sections load, display correct data, and handle user interaction.
- [ ] **Privacy safeguards**: No PII is displayed; privacy compliance is enforced and communicated.
- [ ] **Feedback mechanism**: Feedback can be submitted, stored, and aggregated.
- [ ] **Intervention indicators**: Actionable indicators are computed and shown to trainers.
- [ ] **Model limitations**: Limitations are generated and clearly communicated.
- [ ] **Visualization**: All plots render and update as expected.
- [ ] **Error handling**: All errors are surfaced with user-friendly messages; app remains stable.

---

## 4. Edge Cases & Negative Tests

- Upload malformed or empty CSVs: App should reject and display error.
- Submit feedback with invalid data: App should prompt for correction.
- Remove required environment variables: App should fail gracefully with clear message.
- Attempt to display/export non-anonymized data: Operation blocked, warning shown.
- Simulate model with very low confidence: Dashboard should flag uncertainty.

---

## 5. Regression & Integration

- After any code or data change, re-run all steps above.
- Automated tests in `tests/` must pass.
- End-to-end: From raw data to dashboard, all features must work together.

---

## 6. Usability & Accessibility

- Dashboard sections are clearly labeled and logically organized.
- All interactive elements are accessible via keyboard.
- Privacy and limitations notes are prominent and easy to understand.
- Visualizations are colorblind-friendly and have descriptive legends.

---

## 7. Documentation

- All user-facing instructions (README.md, dashboard help) are accurate and up to date.
- Data dictionary in `data/README.md` matches actual data files.

---

**Testers:**  
- Primary: Project developer(s)  
- Secondary: Peer/trainer reviewer(s) for usability and clarity

**Sign-off:**  
- All acceptance criteria checked  
- All critical and major issues resolved  
- Usability review completed

---

*This test plan ensures the prototype is robust, privacy-conscious, and delivers actionable insights to its intended users.*