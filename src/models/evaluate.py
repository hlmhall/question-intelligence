"""
src/models/evaluate.py

Evaluates models, computes accuracy, confusion matrices, and prediction confidences.

Exports:
    evaluate_model(model, X_test, y_test, average: str = "weighted") -> dict
    get_confidence_scores(model, X) -> list[float]

- Evaluates a scikit-learn classifier on test data, returning accuracy, confusion matrix, and per-class metrics.
- Computes prediction confidence scores (probabilities) for each sample.
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
)
from sklearn.base import BaseEstimator

from src.utils import (
    get_confusion_matrix_elements,
    format_confidence_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: BaseEstimator,
    X_test: np.ndarray,
    y_test: np.ndarray,
    average: str = "weighted",
) -> Dict[str, Any]:
    """
    Evaluate a trained classifier on test data.

    Args:
        model (BaseEstimator): Trained scikit-learn classifier.
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): True labels for test set.
        average (str): Averaging method for multi-class metrics.

    Returns:
        dict: {
            "accuracy": float,
            "confusion_matrix": np.ndarray,
            "classification_report": dict,
            "per_class": dict,
            "confusion_elements": dict (if binary),
        }
    """
    if X_test.shape[0] == 0:
        raise ValueError("X_test is empty.")
    if len(y_test) != X_test.shape[0]:
        raise ValueError("Length of y_test does not match number of samples in X_test.")

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    labels = list(model.classes_) if hasattr(model, "classes_") else sorted(set(y_test) | set(y_pred))

    # Per-class metrics
    per_class = {}
    for label in labels:
        per_class[label] = {
            "precision": report.get(str(label), {}).get("precision", 0.0),
            "recall": report.get(str(label), {}).get("recall", 0.0),
            "f1-score": report.get(str(label), {}).get("f1-score", 0.0),
            "support": report.get(str(label), {}).get("support", 0),
        }

    # For binary, add confusion matrix elements
    confusion_elements = None
    if len(labels) == 2:
        confusion_elements = get_confusion_matrix_elements(np.array(y_test), np.array(y_pred), labels=labels)

    logger.info(f"Model evaluation: accuracy={accuracy:.3f}, classes={labels}")
    return {
        "accuracy": accuracy,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_class": per_class,
        "confusion_elements": confusion_elements,
    }


def get_confidence_scores(
    model: BaseEstimator,
    X: np.ndarray,
) -> List[float]:
    """
    Compute confidence scores (max predicted probability) for each sample.

    Args:
        model (BaseEstimator): Trained classifier supporting predict_proba.
        X (np.ndarray): Feature matrix.

    Returns:
        list[float]: List of confidence scores (0.0-1.0) for each sample.
    """
    if not hasattr(model, "predict_proba"):
        raise AttributeError("Model does not support predict_proba.")
    if X.shape[0] == 0:
        return []
    proba = model.predict_proba(X)
    confidences = proba.max(axis=1)
    return confidences.tolist()


if __name__ == "__main__":
    import sys
    import joblib

    logging.basicConfig(level=logging.INFO)

    # Minimal CLI for demonstration/testing
    # Usage: python -m src.models.evaluate <model_path> <X_test.npy> <y_test.npy>
    if len(sys.argv) < 4:
        print("Usage: python -m src.models.evaluate <model_path> <X_test.npy> <y_test.npy>")
        sys.exit(1)

    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]

    try:
        model = joblib.load(model_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        results = evaluate_model(model, X_test, y_test)
        print(f"Accuracy: {results['accuracy']:.3f}")
        print("Confusion matrix:")
        print(results["confusion_matrix"])
        print("Per-class metrics:")
        for label, metrics in results["per_class"].items():
            print(f"  {label}: precision={metrics['precision']:.2f}, recall={metrics['recall']:.2f}, f1={metrics['f1-score']:.2f}, support={metrics['support']}")
        if results["confusion_elements"]:
            print("Confusion elements (binary):", results["confusion_elements"])
        # Confidence scores demo
        confidences = get_confidence_scores(model, X_test)
        print("Sample confidence scores:", [format_confidence_score(c) for c in confidences[:5]])
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
