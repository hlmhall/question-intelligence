"""
src/models/train.py

Trains text classification models and persists them for inference.

Exports:
    train_classifier(X, y, model_type: str = "logreg", random_state: int = 42, **kwargs) -> Any
    save_model(model: Any, path: str) -> None

- Trains a scikit-learn classifier (default: LogisticRegression) on provided features and labels.
- Supports extensibility for other model types (e.g., RandomForest).
- Persists trained model to disk for later inference.
"""

import logging
import os
from typing import Any, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator
import joblib

from src.config import MODEL_PATH, RANDOM_SEED
from src.utils import safe_makedirs, set_random_seed

logger = logging.getLogger(__name__)


def train_classifier(
    X: np.ndarray,
    y: np.ndarray,
    model_type: str = "logreg",
    random_state: int = RANDOM_SEED,
    **kwargs
) -> BaseEstimator:
    """
    Train a text classifier on the provided features and labels.

    Args:
        X (np.ndarray): Feature matrix (e.g., TF-IDF).
        y (np.ndarray): Target labels.
        model_type (str): Type of classifier ("logreg" or "rf").
        random_state (int): Random seed for reproducibility.
        **kwargs: Additional model-specific parameters.

    Returns:
        BaseEstimator: Trained scikit-learn classifier.
    """
    set_random_seed(random_state)
    if X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("Feature matrix X is empty.")
    if len(y) != X.shape[0]:
        raise ValueError("Length of y does not match number of samples in X.")

    model_type = model_type.lower()
    if model_type == "logreg":
        model = LogisticRegression(
            solver="lbfgs",
            multi_class="auto",
            max_iter=300,
            random_state=random_state,
            n_jobs=1,
            **kwargs
        )
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1,
            **kwargs
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    logger.info(f"Training {model_type} classifier on {X.shape[0]} samples, {X.shape[1]} features.")
    model.fit(X, y)
    logger.info(f"Model training complete. Classes: {getattr(model, 'classes_', None)}")
    return model


def save_model(model: Any, path: Optional[str] = None) -> None:
    """
    Persist a trained model to disk using joblib.

    Args:
        model (Any): Trained model object.
        path (str, optional): Path to save the model. Defaults to MODEL_PATH.
    """
    if path is None:
        path = MODEL_PATH
    safe_makedirs(os.path.dirname(path))
    joblib.dump(model, path)
    logger.info(f"Model saved to {path}")


if __name__ == "__main__":
    import sys
    import pandas as pd

    logging.basicConfig(level=logging.INFO)

    # Minimal CLI for demonstration/testing
    # Usage: python -m src.models.train <features.npy> <labels.npy> [model_type] [output_path]
    if len(sys.argv) < 3:
        print("Usage: python -m src.models.train <features.npy> <labels.npy> [model_type] [output_path]")
        sys.exit(1)

    features_path = sys.argv[1]
    labels_path = sys.argv[2]
    model_type = sys.argv[3] if len(sys.argv) > 3 else "logreg"
    output_path = sys.argv[4] if len(sys.argv) > 4 else MODEL_PATH

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Feature or label file not found: {features_path}, {labels_path}")
        sys.exit(1)

    X = np.load(features_path)
    y = np.load(labels_path)
    try:
        model = train_classifier(X, y, model_type=model_type)
        save_model(model, output_path)
        print(f"Model ({model_type}) trained and saved to {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
