"""
tests/test_visualization.py

Unit tests for src/visualization/plots.py.
Covers:
    - plot_topic_clusters
    - plot_trends
    - plot_confidence

All tests use synthetic in-memory DataFrames and do not write to or read from project data directories.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for tests

from src.visualization import plots

@pytest.fixture
def sample_questions_df():
    # Create a DataFrame with columns: id, text, cluster_id, label, confidence, lesson_id, timestamp
    n = 30
    np.random.seed(42)
    df = pd.DataFrame({
        "id": [f"q{i:03d}" for i in range(n)],
        "text": [f"Sample question text {i}" for i in range(n)],
        "cluster_id": np.random.randint(0, 4, size=n),
        "label": np.random.choice(["clarification", "concept", "logistics"], size=n),
        "confidence": np.random.uniform(0.5, 1.0, size=n),
        "lesson_id": np.random.choice([f"lesson_{j}" for j in range(1, 4)], size=n),
        "timestamp": pd.date_range("2024-05-01", periods=n, freq="D").astype(str),
    })
    # Add some TF-IDF columns for cluster plotting
    for i in range(10):
        df[f"tfidf_{i}"] = np.random.rand(n)
    return df

def test_plot_topic_clusters_runs(sample_questions_df):
    fig = plots.plot_topic_clusters(sample_questions_df)
    assert fig is not None
    assert hasattr(fig, "savefig")
    # Test with text labels
    fig2 = plots.plot_topic_clusters(sample_questions_df, show_text_labels=True)
    assert fig2 is not None
    # Test with t-SNE
    fig3 = plots.plot_topic_clusters(sample_questions_df, method="tsne")
    assert fig3 is not None

def test_plot_topic_clusters_missing_columns(sample_questions_df):
    df = sample_questions_df.drop(columns=["cluster_id"])
    with pytest.raises(ValueError):
        plots.plot_topic_clusters(df)
    df2 = sample_questions_df.drop(columns=["text"])
    with pytest.raises(ValueError):
        plots.plot_topic_clusters(df2)

def test_plot_trends_runs(sample_questions_df):
    fig = plots.plot_trends(sample_questions_df)
    assert fig is not None
    # Test with missing label column (should fallback to counts)
    df = sample_questions_df.drop(columns=["label"])
    fig2 = plots.plot_trends(df)
    assert fig2 is not None
    # Test with some invalid timestamps
    df2 = sample_questions_df.copy()
    df2.loc[0, "timestamp"] = "not-a-date"
    fig3 = plots.plot_trends(df2)
    assert fig3 is not None

def test_plot_trends_missing_columns(sample_questions_df):
    df = sample_questions_df.drop(columns=["lesson_id"])
    with pytest.raises(ValueError):
        plots.plot_trends(df)
    df2 = sample_questions_df.drop(columns=["timestamp"])
    with pytest.raises(ValueError):
        plots.plot_trends(df2)

def test_plot_confidence_runs(sample_questions_df):
    fig = plots.plot_confidence(sample_questions_df)
    assert fig is not None
    # Test with missing label column (should plot histogram)
    df = sample_questions_df.drop(columns=["label"])
    fig2 = plots.plot_confidence(df)
    assert fig2 is not None
    # Test with custom threshold
    fig3 = plots.plot_confidence(sample_questions_df, threshold=0.8)
    assert fig3 is not None

def test_plot_confidence_missing_column(sample_questions_df):
    df = sample_questions_df.drop(columns=["confidence"])
    with pytest.raises(ValueError):
        plots.plot_confidence(df)

def test_plot_confidence_handles_nan(sample_questions_df):
    df = sample_questions_df.copy()
    df.loc[0:2, "confidence"] = np.nan
    fig = plots.plot_confidence(df)
    assert fig is not None

def test_plot_functions_return_matplotlib_figure(sample_questions_df):
    fig1 = plots.plot_topic_clusters(sample_questions_df)
    fig2 = plots.plot_trends(sample_questions_df)
    fig3 = plots.plot_confidence(sample_questions_df)
    import matplotlib.figure
    assert isinstance(fig1, matplotlib.figure.Figure)
    assert isinstance(fig2, matplotlib.figure.Figure)
    assert isinstance(fig3, matplotlib.figure.Figure)
