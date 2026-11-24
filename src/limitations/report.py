"""
src/limitations/report.py

Produces a report summarizing known model limitations for trainer review.

Exports:
    generate_limitations_report(model, data) -> str

- Analyzes the provided model and data to generate a plain-language summary of key limitations.
- Surfaces issues such as data coverage, class imbalance, low-confidence predictions, and known risks.
- Intended for display in the Streamlit dashboard and for trainer awareness.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, Optional

from src.models.evaluate import evaluate_model, get_confidence_scores

LIMITATIONS_HEADER = "## Model Limitations & Appropriate Use"

def _summarize_class_imbalance(per_class: Dict[str, Dict[str, Any]]) -> Optional[str]:
    supports = {label: metrics["support"] for label, metrics in per_class.items()}
    if not supports or len(supports) < 2:
        return None
    max_support = max(supports.values())
    min_support = min(supports.values())
    if min_support == 0 or max_support / (min_support + 1e-6) > 3:
        major = [label for label, s in supports.items() if s == max_support]
        minor = [label for label, s in supports.items() if s == min_support]
        return (
            f"- **Class imbalance detected:** The dataset contains {max_support} samples for '{major[0]}' "
            f"but only {min_support} for '{minor[0]}'. Predictions for underrepresented classes may be less reliable."
        )
    return None

def _summarize_low_confidence(confidences: np.ndarray, threshold: float = 0.6) -> Optional[str]:
    if confidences is None or len(confidences) == 0:
        return None
    low_conf = np.sum(confidences < threshold)
    pct = 100.0 * low_conf / len(confidences)
    if pct > 20:
        return (
            f"- **Low-confidence predictions:** {low_conf} out of {len(confidences)} predictions "
            f"({pct:.1f}%) have confidence below {threshold:.2f}. These should be reviewed with caution."
        )
    return None

def _summarize_label_coverage(y_true: np.ndarray, y_pred: np.ndarray, model_classes: list) -> Optional[str]:
    missing_labels = set(model_classes) - set(y_true)
    if missing_labels:
        return (
            f"- **Limited label coverage:** The following classes are present in the model but not in the current data: "
            f"{', '.join(str(lbl) for lbl in sorted(missing_labels))}. The model may not generalize well to these."
        )
    return None

def _summarize_data_size(n_samples: int) -> Optional[str]:
    if n_samples < 50:
        return (
            f"- **Small dataset:** Only {n_samples} labeled samples were available for training/evaluation. "
            "Model performance may be unstable and not representative of real-world usage."
        )
    return None

def _summarize_known_risks() -> str:
    return (
        "- **Automated predictions are not always correct.** Use model outputs as guidance, not as definitive judgments.\n"
        "- **Model performance may degrade on new or out-of-domain questions.** Unexpected phrasing or topics may lead to misclassification.\n"
        "- **Feedback loop:** Trainer feedback is essential for improving model accuracy over time.\n"
        "- **Privacy:** All data is anonymized, but trainers should avoid uploading real learner identities.\n"
    )

def generate_limitations_report(model: Any, data: Dict[str, Any]) -> str:
    """
    Generate a plain-language report summarizing model limitations.

    Args:
        model: Trained scikit-learn classifier (must support predict/predict_proba).
        data: Dict with keys:
            - 'X_test': np.ndarray or pd.DataFrame (features)
            - 'y_test': np.ndarray or pd.Series (true labels)
            - 'df_questions': pd.DataFrame (optional, for context)

    Returns:
        str: Markdown-formatted limitations report.
    """
    lines = [LIMITATIONS_HEADER, ""]
    # Defensive: check required data
    X_test = data.get("X_test")
    y_test = data.get("y_test")
    df_questions = data.get("df_questions", None)
    n_samples = len(y_test) if y_test is not None else 0

    # Evaluate model if possible
    try:
        eval_result = evaluate_model(model, X_test, y_test)
        per_class = eval_result.get("per_class", {})
        y_pred = model.predict(X_test)
        model_classes = list(model.classes_) if hasattr(model, "classes_") else sorted(set(y_test) | set(y_pred))
        confidences = np.array(get_confidence_scores(model, X_test))
    except Exception:
        per_class = {}
        y_pred = np.array([])
        model_classes = []
        confidences = np.array([])
    
    # Data size
    ds = _summarize_data_size(n_samples)
    if ds:
        lines.append(ds)
    # Class imbalance
    ci = _summarize_class_imbalance(per_class)
    if ci:
        lines.append(ci)
    # Label coverage
    lc = _summarize_label_coverage(y_test, y_pred, model_classes) if n_samples > 0 else None
    if lc:
        lines.append(lc)
    # Low-confidence
    lc2 = _summarize_low_confidence(confidences)
    if lc2:
        lines.append(lc2)
    # Known risks
    lines.append(_summarize_known_risks())

    # If no limitations detected, still show known risks
    if len(lines) <= 2:
        lines.append("- No major limitations detected in current evaluation, but see below for general cautions.")
        lines.append(_summarize_known_risks())

    return "\n".join(lines)


if __name__ == "__main__":
    import joblib
    import sys
    import numpy as np

    # Minimal CLI for demonstration/testing
    # Usage: python -m src.limitations.report <model_path> <X_test.npy> <y_test.npy>
    if len(sys.argv) < 4:
        print("Usage: python -m src.limitations.report <model_path> <X_test.npy> <y_test.npy>")
        sys.exit(1)
    model_path = sys.argv[1]
    X_test_path = sys.argv[2]
    y_test_path = sys.argv[3]
    try:
        model = joblib.load(model_path)
        X_test = np.load(X_test_path)
        y_test = np.load(y_test_path)
        report = generate_limitations_report(model, {"X_test": X_test, "y_test": y_test})
        print(report)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
