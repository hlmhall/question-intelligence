"""
src/visualization/plots.py

Generates interactive visualizations for the Streamlit dashboard.

Exports:
    plot_topic_clusters(df: pd.DataFrame, cluster_col: str = "cluster_id", text_col: str = "text", max_samples: int = 100)
    plot_trends(df: pd.DataFrame, lesson_col: str = "lesson_id", timestamp_col: str = "timestamp", label_col: str = "label")
    plot_confidence(df: pd.DataFrame, confidence_col: str = "confidence", label_col: str = "label", threshold: float = 0.7)

- Cluster visualization (2D projection of question clusters, with sample texts).
- Lesson trends (questions per lesson over time, optionally by label).
- Confidence distribution (histogram or violin plot of prediction confidences).
- All plots are Streamlit/Matplotlib/Seaborn compatible.
"""

import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)


def plot_topic_clusters(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    text_col: str = "text",
    max_samples: int = 100,
    method: str = "pca",
    random_state: int = 42,
    show_text_labels: bool = False,
    figsize: tuple = (8, 6),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Visualize question clusters in 2D using PCA or t-SNE.

    Args:
        df (pd.DataFrame): DataFrame with cluster assignments and text.
        cluster_col (str): Column with cluster labels.
        text_col (str): Column with question text.
        max_samples (int): Max number of samples to plot (for clarity).
        method (str): 'pca' or 'tsne' for dimensionality reduction.
        random_state (int): Random seed for reproducibility.
        show_text_labels (bool): Overlay sample texts on points.
        figsize (tuple): Figure size.
        ax (plt.Axes): Optional matplotlib axes.

    Returns:
        plt.Figure: The generated figure.
    """
    if cluster_col not in df.columns:
        raise ValueError(f"Column '{cluster_col}' not found in DataFrame.")
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    # Sample for clarity
    plot_df = df.copy()
    if len(plot_df) > max_samples:
        plot_df = plot_df.sample(n=max_samples, random_state=random_state)

    # Use TF-IDF features if present, else fallback to random projection
    feature_cols = [col for col in plot_df.columns if col.startswith("tfidf_")]
    if feature_cols:
        X = plot_df[feature_cols].values
    else:
        # Fallback: random projection for demo
        np.random.seed(random_state)
        X = np.random.normal(size=(len(plot_df), 10))

    # Dimensionality reduction
    if method == "tsne":
        reducer = TSNE(n_components=2, random_state=random_state, init="random", perplexity=10)
        X_2d = reducer.fit_transform(X)
    else:
        reducer = PCA(n_components=2, random_state=random_state)
        X_2d = reducer.fit_transform(X)

    plot_df["x"] = X_2d[:, 0]
    plot_df["y"] = X_2d[:, 1]

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    palette = sns.color_palette("tab10", n_colors=plot_df[cluster_col].nunique())
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        hue=cluster_col,
        palette=palette,
        s=60,
        alpha=0.85,
        ax=ax,
        legend="full",
    )
    ax.set_title("Question Topic Clusters (2D Projection)")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")

    if show_text_labels:
        for _, row in plot_df.iterrows():
            ax.text(
                row["x"],
                row["y"],
                str(row[text_col])[:30] + ("..." if len(str(row[text_col])) > 30 else ""),
                fontsize=7,
                alpha=0.7,
            )

    fig.tight_layout()
    return fig


def plot_trends(
    df: pd.DataFrame,
    lesson_col: str = "lesson_id",
    timestamp_col: str = "timestamp",
    label_col: str = "label",
    date_format: str = "%Y-%m-%d",
    figsize: tuple = (9, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot number of questions per lesson over time, optionally by label.

    Args:
        df (pd.DataFrame): DataFrame with lesson, timestamp, and label columns.
        lesson_col (str): Column with lesson IDs.
        timestamp_col (str): Column with timestamps.
        label_col (str): Column with predicted or true labels.
        date_format (str): Format for x-axis dates.
        figsize (tuple): Figure size.
        ax (plt.Axes): Optional matplotlib axes.

    Returns:
        plt.Figure: The generated figure.
    """
    if lesson_col not in df.columns:
        raise ValueError(f"Column '{lesson_col}' not found in DataFrame.")
    if timestamp_col not in df.columns:
        raise ValueError(f"Column '{timestamp_col}' not found in DataFrame.")

    plot_df = df.copy()
    # Parse timestamps to dates
    plot_df["date"] = pd.to_datetime(plot_df[timestamp_col], errors="coerce").dt.date
    # Drop rows with invalid dates
    plot_df = plot_df.dropna(subset=["date"])

    # Group by date and label (if available)
    if label_col in plot_df.columns and plot_df[label_col].notnull().any():
        grouped = plot_df.groupby(["date", label_col]).size().reset_index(name="count")
        pivot = grouped.pivot(index="date", columns=label_col, values="count").fillna(0)
    else:
        grouped = plot_df.groupby("date").size().reset_index(name="count")
        pivot = grouped.set_index("date")

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    if label_col in plot_df.columns and plot_df[label_col].notnull().any():
        pivot.plot(kind="bar", stacked=True, ax=ax, colormap="tab20")
        ax.legend(title="Label", bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        pivot["count"].plot(kind="bar", ax=ax, color="skyblue")
        ax.legend().set_visible(False)

    ax.set_title("Questions per Lesson Date")
    ax.set_xlabel("Date")
    ax.set_ylabel("Number of Questions")
    ax.set_xticklabels([d.strftime(date_format) if hasattr(d, "strftime") else str(d) for d in pivot.index], rotation=45, ha="right")
    fig.tight_layout()
    return fig


def plot_confidence(
    df: pd.DataFrame,
    confidence_col: str = "confidence",
    label_col: str = "label",
    threshold: float = 0.7,
    figsize: tuple = (8, 5),
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Plot distribution of prediction confidence scores, optionally by label.

    Args:
        df (pd.DataFrame): DataFrame with confidence and label columns.
        confidence_col (str): Column with confidence scores (0.0-1.0).
        label_col (str): Column with predicted or true labels.
        threshold (float): Threshold for low-confidence highlighting.
        figsize (tuple): Figure size.
        ax (plt.Axes): Optional matplotlib axes.

    Returns:
        plt.Figure: The generated figure.
    """
    if confidence_col not in df.columns:
        raise ValueError(f"Column '{confidence_col}' not found in DataFrame.")

    plot_df = df.copy()
    plot_df = plot_df[plot_df[confidence_col].notnull()]
    plot_df[confidence_col] = plot_df[confidence_col].astype(float).clip(0, 1)

    fig, ax = plt.subplots(figsize=figsize) if ax is None else (ax.figure, ax)
    if label_col in plot_df.columns and plot_df[label_col].notnull().any():
        sns.violinplot(
            data=plot_df,
            x=label_col,
            y=confidence_col,
            inner="quartile",
            palette="Set2",
            ax=ax,
        )
        ax.set_xlabel("Label")
    else:
        sns.histplot(
            plot_df[confidence_col],
            bins=20,
            kde=True,
            color="cornflowerblue",
            ax=ax,
        )
        ax.set_xlabel("Confidence Score")

    ax.axhline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Threshold ({threshold:.2f})") if label_col in plot_df.columns and plot_df[label_col].notnull().any() else ax.axvline(threshold, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Threshold ({threshold:.2f})")
    ax.set_title("Prediction Confidence Distribution")
    ax.set_ylabel("Density" if not (label_col in plot_df.columns and plot_df[label_col].notnull().any()) else "Confidence")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


# Optional: Streamlit wrappers for direct use in app.py
try:
    import streamlit as st

    def st_plot_topic_clusters(*args, **kwargs):
        fig = plot_topic_clusters(*args, **kwargs)
        st.pyplot(fig)

    def st_plot_trends(*args, **kwargs):
        fig = plot_trends(*args, **kwargs)
        st.pyplot(fig)

    def st_plot_confidence(*args, **kwargs):
        fig = plot_confidence(*args, **kwargs)
        st.pyplot(fig)

except ImportError:
    # Streamlit not installed; wrappers not available
    pass


if __name__ == "__main__":
    # Minimal CLI for demonstration/testing
    import sys
    import os

    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m src.visualization.plots <questions_csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    # Add dummy cluster and confidence columns if missing
    if "cluster_id" not in df.columns:
        df["cluster_id"] = np.random.randint(0, 4, size=len(df))
    if "confidence" not in df.columns:
        df["confidence"] = np.random.uniform(0.5, 1.0, size=len(df))
    # Plot and save figures
    fig1 = plot_topic_clusters(df)
    fig1.savefig("topic_clusters.png")
    print("Saved: topic_clusters.png")
    fig2 = plot_trends(df)
    fig2.savefig("trends.png")
    print("Saved: trends.png")
    fig3 = plot_confidence(df)
    fig3.savefig("confidence.png")
    print("Saved: confidence.png")
