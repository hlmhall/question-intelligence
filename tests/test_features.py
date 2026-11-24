"""
tests/test_features.py

Unit tests for src/features/build_features.py:
    - build_tfidf_features
    - build_clustering_features

Uses pytest and temporary dataframes.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src.features.build_features import build_tfidf_features, build_clustering_features

# --- Fixtures ---

@pytest.fixture
def sample_questions_df():
    # Simulate a small, realistic set of learner questions
    data = {
        "id": ["q1", "q2", "q3", "q4", "q5", "q6"],
        "user_id": ["u1", "u2", "u1", "u3", "u2", "u4"],
        "lesson_id": ["l1", "l1", "l2", "l2", "l3", "l3"],
        "text": [
            "What is the difference between supervised and unsupervised learning?",
            "Can you explain overfitting with an example?",
            "What is the difference between supervised and unsupervised learning?",  # duplicate
            "How does regularization help in machine learning?",
            "",  # empty text
            "What are the main types of neural networks?"
        ],
        "timestamp": [
            "2024-05-01T10:00:00",
            "2024-05-01T10:05:00",
            "2024-05-01T10:10:00",
            "2024-05-02T09:00:00",
            "2024-05-02T09:05:00",
            "2024-05-03T11:00:00"
        ],
        "label": ["concept", "example", "concept", "mechanism", "", "concept"]
    }
    return pd.DataFrame(data)

@pytest.fixture
def minimal_questions_df():
    # Only two non-empty questions
    data = {
        "id": ["q1", "q2"],
        "user_id": ["u1", "u2"],
        "lesson_id": ["l1", "l1"],
        "text": [
            "What is a confusion matrix?",
            "How do you calculate accuracy?"
        ],
        "timestamp": [
            "2024-05-01T10:00:00",
            "2024-05-01T10:05:00"
        ],
        "label": ["concept", "metric"]
    }
    return pd.DataFrame(data)

# --- Tests for build_tfidf_features ---

def test_build_tfidf_features_basic(sample_questions_df):
    tfidf_matrix, vectorizer = build_tfidf_features(sample_questions_df, text_col="text", max_features=10, random_state=123)
    # Only 5 non-empty texts
    assert tfidf_matrix.shape[0] == 5
    assert isinstance(vectorizer, TfidfVectorizer)
    # Should have at most 10 features
    assert tfidf_matrix.shape[1] <= 10
    # Check that duplicate questions are not dropped (row count matches non-empty texts)
    texts = sample_questions_df["text"].astype(str).fillna("").tolist()
    non_empty_count = sum(1 for t in texts if t.strip())
    assert tfidf_matrix.shape[0] == non_empty_count

def test_build_tfidf_features_empty_texts():
    df = pd.DataFrame({"text": ["", "   ", None]})
    with pytest.raises(ValueError, match="No non-empty texts found for TF-IDF vectorization."):
        build_tfidf_features(df, text_col="text")

def test_build_tfidf_features_missing_column(sample_questions_df):
    with pytest.raises(ValueError, match="Column 'body' not found in DataFrame."):
        build_tfidf_features(sample_questions_df, text_col="body")

def test_build_tfidf_features_reproducibility(sample_questions_df):
    tfidf1, vec1 = build_tfidf_features(sample_questions_df, random_state=42)
    tfidf2, vec2 = build_tfidf_features(sample_questions_df, random_state=42)
    np.testing.assert_array_almost_equal(tfidf1.toarray(), tfidf2.toarray())
    assert vec1.get_feature_names_out().tolist() == vec2.get_feature_names_out().tolist()

# --- Tests for build_clustering_features ---

def test_build_clustering_features_basic(sample_questions_df):
    tfidf_matrix, _ = build_tfidf_features(sample_questions_df)
    cluster_labels, kmeans = build_clustering_features(sample_questions_df, tfidf_matrix, n_clusters=3, random_state=99)
    assert isinstance(cluster_labels, np.ndarray)
    assert isinstance(kmeans, KMeans)
    assert len(cluster_labels) == tfidf_matrix.shape[0]
    assert set(cluster_labels) <= set(range(3))

def test_build_clustering_features_too_few_rows(minimal_questions_df):
    tfidf_matrix, _ = build_tfidf_features(minimal_questions_df)
    # Only 2 rows, so n_clusters=2 is allowed, but 3 is not
    with pytest.raises(ValueError, match="At least 2 clusters required for clustering."):
        build_clustering_features(minimal_questions_df, tfidf_matrix, n_clusters=1)
    # n_clusters=2 should work
    cluster_labels, kmeans = build_clustering_features(minimal_questions_df, tfidf_matrix, n_clusters=2)
    assert len(cluster_labels) == 2

def test_build_clustering_features_empty_matrix():
    df = pd.DataFrame({"text": ["", ""]})
    with pytest.raises(ValueError, match="No non-empty texts found for TF-IDF vectorization."):
        tfidf_matrix, _ = build_tfidf_features(df)
    # If tfidf_matrix is empty, clustering should fail
    # (simulate empty matrix directly)
    with pytest.raises(ValueError, match="TF-IDF matrix is empty; cannot cluster."):
        build_clustering_features(df, np.zeros((0, 10)))

def test_build_clustering_features_reproducibility(sample_questions_df):
    tfidf_matrix, _ = build_tfidf_features(sample_questions_df, random_state=7)
    labels1, kmeans1 = build_clustering_features(sample_questions_df, tfidf_matrix, n_clusters=3, random_state=7)
    labels2, kmeans2 = build_clustering_features(sample_questions_df, tfidf_matrix, n_clusters=3, random_state=7)
    np.testing.assert_array_equal(labels1, labels2)
    np.testing.assert_array_almost_equal(kmeans1.cluster_centers_, kmeans2.cluster_centers_)

def test_build_clustering_features_n_clusters_larger_than_rows(sample_questions_df):
    tfidf_matrix, _ = build_tfidf_features(sample_questions_df)
    # Only 5 non-empty rows, so n_clusters should be capped at 5
    cluster_labels, kmeans = build_clustering_features(sample_questions_df, tfidf_matrix, n_clusters=10)
    assert len(np.unique(cluster_labels)) == tfidf_matrix.shape[0]

# --- End of file ---
