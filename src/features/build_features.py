"""
src/features/build_features.py

Transforms text data into machine learning features (TF-IDF, clustering).

Exports:
    build_tfidf_features(df: pd.DataFrame, text_col: str = "text", max_features: int = 3000, random_state: int = 42)
        -> Tuple[scipy.sparse.csr_matrix, TfidfVectorizer]
    build_clustering_features(df: pd.DataFrame, tfidf_matrix: scipy.sparse.csr_matrix, n_clusters: int = 6, random_state: int = 42)
        -> Tuple[np.ndarray, KMeans]

- Applies TF-IDF vectorization to question texts.
- Performs clustering (KMeans) on TF-IDF features.
- Returns features and fitted transformers for downstream modeling and analysis.
"""

import logging
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from src.config import RANDOM_SEED
from src.utils import set_random_seed

logger = logging.getLogger(__name__)


def build_tfidf_features(
    df: pd.DataFrame,
    text_col: str = "text",
    max_features: int = 3000,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, TfidfVectorizer]:
    """
    Transform question texts into TF-IDF features.

    Args:
        df (pd.DataFrame): DataFrame with a text column.
        text_col (str): Name of the column containing text.
        max_features (int): Maximum number of TF-IDF features.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, TfidfVectorizer]: (TF-IDF matrix, fitted vectorizer)
    """
    set_random_seed(random_state)
    if text_col not in df.columns:
        raise ValueError(f"Column '{text_col}' not found in DataFrame.")

    texts = df[text_col].astype(str).fillna("").tolist()
    # Remove empty texts
    texts = [t for t in texts if t.strip()]
    if not texts:
        raise ValueError("No non-empty texts found for TF-IDF vectorization.")

    vectorizer = TfidfVectorizer(
        strip_accents="unicode",
        lowercase=True,
        stop_words="english",
        max_features=max_features,
        ngram_range=(1, 2),
        dtype=np.float32,
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    logger.info(f"TF-IDF features built: shape={tfidf_matrix.shape}")
    return tfidf_matrix, vectorizer


def build_clustering_features(
    df: pd.DataFrame,
    tfidf_matrix: np.ndarray,
    n_clusters: int = 6,
    random_state: int = RANDOM_SEED,
) -> Tuple[np.ndarray, KMeans]:
    """
    Cluster questions using KMeans on TF-IDF features.

    Args:
        df (pd.DataFrame): DataFrame of questions (for row count).
        tfidf_matrix (np.ndarray): TF-IDF feature matrix.
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, KMeans]: (cluster labels, fitted KMeans model)
    """
    set_random_seed(random_state)
    if tfidf_matrix.shape[0] == 0:
        raise ValueError("TF-IDF matrix is empty; cannot cluster.")

    n_clusters = min(n_clusters, tfidf_matrix.shape[0])
    if n_clusters < 2:
        raise ValueError("At least 2 clusters required for clustering.")

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init="auto" if hasattr(KMeans(), "n_init") and isinstance(KMeans().n_init, str) else 10,
        max_iter=300,
        verbose=0,
    )
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    logger.info(f"KMeans clustering complete: n_clusters={n_clusters}")
    return cluster_labels, kmeans


if __name__ == "__main__":
    import sys
    import os

    logging.basicConfig(level=logging.INFO)
    # Minimal CLI for demonstration/testing
    # Usage: python -m src.features.build_features [csv_path]
    if len(sys.argv) < 2:
        print("Usage: python -m src.features.build_features <questions_csv_path>")
        sys.exit(1)
    csv_path = sys.argv[1]
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    try:
        tfidf_matrix, vectorizer = build_tfidf_features(df)
        cluster_labels, kmeans = build_clustering_features(df, tfidf_matrix)
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"Cluster labels: {np.unique(cluster_labels)}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
