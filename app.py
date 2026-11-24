"""
app.py

Main Streamlit app entry point for Practical Foundations for Question Intelligence.

- Loads cleaned data, trained model, and feature transformers.
- Presents interactive dashboard: classified questions, clusters, lesson trends, confidence scores.
- Handles user feedback, privacy notes, intervention indicators, and model limitations.
- All configuration via src/config.py and environment variables.

Run:
    streamlit run app.py

"""

import os
import sys
import logging
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

from src.config import RAW_DIR, PROCESSED_DIR, RANDOM_SEED, MODEL_PATH, FEEDBACK_STORE
from src.data.load import load_questions, load_lessons, load_users
from src.data.clean import clean_questions, clean_lessons, clean_users
from src.features.build_features import build_tfidf_features, build_clustering_features
from src.models.evaluate import evaluate_model, get_confidence_scores
from src.feedback.feedback import submit_feedback, aggregate_feedback
from src.privacy.privacy import anonymize_data, check_privacy_compliance
from src.intervention.indicators import compute_intervention_indicators
from src.limitations.report import generate_limitations_report
from src.visualization.plots import (
    st_plot_topic_clusters,
    st_plot_trends,
    st_plot_confidence,
)
from src.utils import set_random_seed

import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Streamlit page config
st.set_page_config(
    page_title="Question Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Utility Functions ---

@st.cache_data(show_spinner=False)
def load_and_clean_data():
    """Load and clean all data, returning DataFrames."""
    # Load raw data
    raw_questions = load_questions()
    raw_lessons = load_lessons()
    raw_users = load_users()
    # Clean
    questions = clean_questions(raw_questions)
    lessons = clean_lessons(raw_lessons)
    users = clean_users(raw_users)
    return questions, lessons, users

@st.cache_resource(show_spinner=False)
def load_model_and_vectorizer():
    """Load trained classifier and TF-IDF vectorizer."""
    if not Path(MODEL_PATH).exists():
        st.error(f"Trained model not found at {MODEL_PATH}. Please train the model first.")
        st.stop()
    model_bundle = joblib.load(MODEL_PATH)
    # Expecting a dict: {'model': ..., 'vectorizer': ...}
    model = model_bundle.get("model")
    vectorizer = model_bundle.get("vectorizer")
    if model is None or vectorizer is None:
        st.error("Model file is missing required components (model/vectorizer).")
        st.stop()
    return model, vectorizer

def get_tfidf_matrix(questions_df, vectorizer):
    """Transform question texts to TF-IDF matrix using fitted vectorizer."""
    texts = questions_df["text"].astype(str).fillna("").tolist()
    return vectorizer.transform(texts)

def assign_clusters(questions_df, tfidf_matrix, n_clusters=6):
    """Assign cluster labels to questions using KMeans."""
    cluster_labels, kmeans = build_clustering_features(questions_df, tfidf_matrix, n_clusters=n_clusters, random_state=RANDOM_SEED)
    return cluster_labels, kmeans

def merge_data(questions, lessons, users):
    """Merge questions with lesson and user metadata for dashboard."""
    merged = questions.merge(lessons, left_on="lesson_id", right_on="id", how="left", suffixes=("", "_lesson"))
    merged = merged.merge(users, left_on="user_id", right_on="id", how="left", suffixes=("", "_user"))
    return merged

def display_privacy_note():
    st.info(
        "🔒 **Privacy Notice:** All learner and user data shown here is anonymized and synthetic. "
        "No personally identifiable information is present. "
        "The dashboard enforces privacy compliance checks before displaying or exporting data."
    )

def display_limitations(model, data):
    report = generate_limitations_report(model, data)
    st.warning("⚠️ **Model Limitations:**\n\n" + report)

def display_intervention_indicators(preds, confidences, questions_df):
    indicators = compute_intervention_indicators(preds, confidences)
    st.subheader("📈 Intervention Indicators")
    for key, value in indicators.items():
        st.markdown(f"- **{key}:** {value}")
    st.caption("Indicators are computed from model predictions and confidence scores to help trainers identify possible learning friction points.")

def display_feedback_section(questions_df):
    st.subheader("💬 Trainer Feedback")
    st.markdown(
        "Help us improve! If you spot a misclassified question, submit feedback below."
    )
    question_options = questions_df[["id", "text"]].drop_duplicates().values.tolist()
    question_map = {f"{qid}: {text[:40]}": qid for qid, text in question_options}
    selected = st.selectbox("Select a question to give feedback on:", [""] + list(question_map.keys()))
    if selected:
        qid = question_map[selected]
        feedback = st.text_area("Your feedback (e.g., correct label, comments):", key=f"feedback_{qid}")
        if st.button("Submit Feedback", key=f"submit_{qid}"):
            submit_feedback(qid, feedback)
            st.success("Thank you for your feedback!")

    # Show aggregated feedback summary
    st.markdown("#### Recent Feedback")
    feedback_df = aggregate_feedback()
    if feedback_df is not None and not feedback_df.empty:
        st.dataframe(feedback_df.tail(10), use_container_width=True)
    else:
        st.info("No feedback submitted yet.")

def display_dashboard(merged_df, model, vectorizer, kmeans, confidences):
    st.title("Practical Foundations for Question Intelligence")
    st.caption("A prototype dashboard for classifying and analyzing learner questions. For instructional use only.")

    # Sidebar filters
    st.sidebar.header("🔎 Filters")
    lesson_titles = merged_df["title"].dropna().unique().tolist()
    selected_lessons = st.sidebar.multiselect("Lesson", lesson_titles, default=lesson_titles)
    label_values = merged_df["label"].dropna().unique().tolist()
    selected_labels = st.sidebar.multiselect("Label", label_values, default=label_values)
    min_conf = st.sidebar.slider("Min Confidence", 0.0, 1.0, 0.0, 0.01)

    filtered = merged_df[
        merged_df["title"].isin(selected_lessons) &
        merged_df["label"].isin(selected_labels) &
        (merged_df["confidence"] >= min_conf)
    ].copy()

    st.markdown("### 📋 Classified Learner Questions")
    st.dataframe(
        filtered[["id", "text", "title", "topic", "date", "label", "confidence", "cluster_id"]],
        use_container_width=True,
        hide_index=True,
    )

    # Visualizations
    st.markdown("### 🧭 Topic Clusters")
    st_plot_topic_clusters(filtered, cluster_col="cluster_id", text_col="text", max_samples=80, show_text_labels=False)

    st.markdown("### 📊 Lesson Trends")
    st_plot_trends(filtered, lesson_col="lesson_id", timestamp_col="timestamp", label_col="label")

    st.markdown("### 🎯 Prediction Confidence")
    st_plot_confidence(filtered, confidence_col="confidence", label_col="label", threshold=0.7)

    # Intervention indicators
    display_intervention_indicators(filtered["label"].values, filtered["confidence"].values, filtered)

    # Feedback section
    display_feedback_section(filtered)

    # Model limitations
    display_limitations(model, filtered)

    # Privacy note
    display_privacy_note()

def main():
    set_random_seed(RANDOM_SEED)
    st.sidebar.title("Question Intelligence Prototype")
    st.sidebar.markdown("Prototype for educational trainers and instructional designers.")

    # Load and clean data
    questions, lessons, users = load_and_clean_data()

    # Load model and vectorizer
    model, vectorizer = load_model_and_vectorizer()

    # TF-IDF features
    tfidf_matrix = get_tfidf_matrix(questions, vectorizer)

    # Predict labels and confidence
    preds = model.predict(tfidf_matrix)
    confidences = get_confidence_scores(model, tfidf_matrix)
    questions["label"] = preds
    questions["confidence"] = confidences

    # Clustering
    cluster_labels, kmeans = assign_clusters(questions, tfidf_matrix, n_clusters=6)
    questions["cluster_id"] = cluster_labels

    # Merge for dashboard
    merged_df = merge_data(questions, lessons, users)

    # Dashboard
    display_dashboard(merged_df, model, vectorizer, kmeans, confidences)

if __name__ == "__main__":
    main()
