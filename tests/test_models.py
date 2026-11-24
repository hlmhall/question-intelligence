"""
tests/test_models.py

Unit tests for model training and evaluation modules:
- src/models/train.py: train_classifier, save_model
- src/models/evaluate.py: evaluate_model, get_confidence_scores

Uses pytest and temporary directories for file operations.
"""

import os
import tempfile
import numpy as np
import pytest
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.train import train_classifier, save_model
from src.models.evaluate import evaluate_model, get_confidence_scores

# Minimal synthetic data for classification
@pytest.fixture
def simple_classification_data():
    # 3-class, 2-feature, 12 samples
    X = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.9, 0.8],
        [0.8, 0.9],
        [0.5, 0.5],
        [0.51, 0.49],
        [0.15, 0.18],
        [0.85, 0.82],
        [0.48, 0.52],
        [0.52, 0.48],
        [0.13, 0.19],
        [0.87, 0.81],
    ])
    y = np.array([
        "A", "A", "B", "B", "C", "C",
        "A", "B", "C", "C", "A", "B"
    ])
    return X, y

def test_train_classifier_logreg(simple_classification_data):
    X, y = simple_classification_data
    model = train_classifier(X, y, model_type="logreg", random_state=123)
    assert isinstance(model, LogisticRegression)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    # Model should have seen all classes
    assert set(model.classes_) == set(y)
    # Model should fit without error and predict
    preds = model.predict(X)
    assert len(preds) == len(y)
    # Should be able to call predict_proba
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), len(model.classes_))

def test_train_classifier_rf(simple_classification_data):
    X, y = simple_classification_data
    model = train_classifier(X, y, model_type="rf", random_state=123)
    assert isinstance(model, RandomForestClassifier)
    assert hasattr(model, "predict")
    assert hasattr(model, "predict_proba")
    assert set(model.classes_) == set(y)
    preds = model.predict(X)
    assert len(preds) == len(y)
    proba = model.predict_proba(X)
    assert proba.shape == (len(y), len(model.classes_))

def test_train_classifier_invalid_type(simple_classification_data):
    X, y = simple_classification_data
    with pytest.raises(ValueError):
        train_classifier(X, y, model_type="unsupported")

def test_train_classifier_empty_X():
    X = np.empty((0, 2))
    y = np.array([])
    with pytest.raises(ValueError):
        train_classifier(X, y)

def test_train_classifier_y_length_mismatch():
    X = np.ones((5, 2))
    y = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        train_classifier(X, y)

def test_save_model_and_reload(simple_classification_data, tmp_path):
    X, y = simple_classification_data
    model = train_classifier(X, y, model_type="logreg", random_state=42)
    model_path = tmp_path / "test_model.pkl"
    save_model(model, str(model_path))
    assert os.path.exists(model_path)
    loaded = joblib.load(model_path)
    assert isinstance(loaded, LogisticRegression)
    # Loaded model should predict the same as original
    np.testing.assert_array_equal(model.predict(X), loaded.predict(X))

def test_evaluate_model_output(simple_classification_data):
    X, y = simple_classification_data
    model = train_classifier(X, y, model_type="logreg", random_state=42)
    results = evaluate_model(model, X, y)
    assert "accuracy" in results
    assert "confusion_matrix" in results
    assert "classification_report" in results
    assert "per_class" in results
    # Accuracy should be between 0 and 1
    assert 0.0 <= results["accuracy"] <= 1.0
    # Confusion matrix shape matches number of classes
    n_classes = len(set(y))
    assert results["confusion_matrix"].shape == (n_classes, n_classes)
    # Per-class metrics for each class
    for label in set(y):
        assert label in results["per_class"]
        metrics = results["per_class"][label]
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1-score" in metrics
        assert "support" in metrics

def test_evaluate_model_binary_confusion_elements():
    # 2-class case
    X = np.array([[0], [1], [0], [1]])
    y = np.array(["yes", "no", "yes", "no"])
    model = train_classifier(X, y, model_type="logreg", random_state=1)
    results = evaluate_model(model, X, y)
    assert "confusion_elements" in results
    # Should be a dict with TP, TN, FP, FN
    ce = results["confusion_elements"]
    assert all(k in ce for k in ("TP", "TN", "FP", "FN"))

def test_evaluate_model_empty_X():
    model = LogisticRegression()
    X_test = np.empty((0, 2))
    y_test = np.array([])
    with pytest.raises(ValueError):
        evaluate_model(model, X_test, y_test)

def test_evaluate_model_y_length_mismatch():
    model = LogisticRegression()
    X_test = np.ones((5, 2))
    y_test = np.array([1, 2])
    with pytest.raises(ValueError):
        evaluate_model(model, X_test, y_test)

def test_get_confidence_scores(simple_classification_data):
    X, y = simple_classification_data
    model = train_classifier(X, y, model_type="logreg", random_state=42)
    confidences = get_confidence_scores(model, X)
    assert isinstance(confidences, list)
    assert len(confidences) == len(y)
    # All confidences between 0 and 1
    assert all(0.0 <= c <= 1.0 for c in confidences)

def test_get_confidence_scores_no_predict_proba():
    class DummyModel:
        def predict(self, X):
            return np.zeros(X.shape[0])
    model = DummyModel()
    X = np.ones((3, 2))
    with pytest.raises(AttributeError):
        get_confidence_scores(model, X)

def test_get_confidence_scores_empty_X(simple_classification_data):
    X, y = simple_classification_data
    model = train_classifier(X, y, model_type="logreg", random_state=42)
    X_empty = np.empty((0, X.shape[1]))
    confidences = get_confidence_scores(model, X_empty)
    assert confidences == []
