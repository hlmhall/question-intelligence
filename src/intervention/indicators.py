"""
src/intervention/indicators.py

Calculates actionable intervention indicators for trainers, factoring in model confidence.

Exports:
    compute_intervention_indicators(preds, confidences, threshold: float = 0.7, min_count: int = 3) -> dict

- Computes summary statistics and actionable signals (e.g., spike in low-confidence questions, high-frequency topics).
- Designed for use in the Streamlit dashboard to prompt instructional review.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from src.utils import format_confidence_score, safe_float, safe_int

def compute_intervention_indicators(
    preds: List[str],
    confidences: List[float],
    threshold: float = 0.7,
    min_count: int = 3,
    question_texts: Optional[List[str]] = None,
    lesson_ids: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute actionable intervention indicators for trainers.

    Args:
        preds (List[str]): Model-predicted labels for each question.
        confidences (List[float]): Model confidence scores (0.0-1.0) for each prediction.
        threshold (float): Confidence threshold below which a prediction is considered low-confidence.
        min_count (int): Minimum count for a topic/label to be considered a "hot topic".
        question_texts (Optional[List[str]]): Raw question texts (optional, for reporting).
        lesson_ids (Optional[List[str]]): Lesson IDs for each question (optional, for lesson-level indicators).
        labels (Optional[List[str]]): List of all possible labels (optional, for completeness).

    Returns:
        Dict[str, Any]: {
            "low_confidence_count": int,
            "low_confidence_pct": float,
            "low_confidence_questions": List[dict],
            "hot_topics": List[dict],
            "topic_distribution": Dict[str, int],
            "lesson_alerts": List[dict],
            "summary": str,
        }
    """
    n = len(preds)
    if n == 0:
        return {
            "low_confidence_count": 0,
            "low_confidence_pct": 0.0,
            "low_confidence_questions": [],
            "hot_topics": [],
            "topic_distribution": {},
            "lesson_alerts": [],
            "summary": "No questions to analyze.",
        }
    # Convert to numpy arrays for efficient ops
    preds_arr = np.array(preds)
    conf_arr = np.array([safe_float(c, default=0.0) for c in confidences])
    if labels is None:
        labels = sorted(list(set(preds_arr)))
    # Low-confidence questions
    low_conf_mask = conf_arr < threshold
    low_conf_indices = np.where(low_conf_mask)[0]
    low_conf_count = int(low_conf_mask.sum())
    low_conf_pct = float(low_conf_count) / n if n > 0 else 0.0
    low_conf_questions = []
    for idx in low_conf_indices:
        q = {
            "index": int(idx),
            "pred_label": preds_arr[idx],
            "confidence": format_confidence_score(conf_arr[idx]),
        }
        if question_texts is not None and idx < len(question_texts):
            q["text"] = question_texts[idx]
        if lesson_ids is not None and idx < len(lesson_ids):
            q["lesson_id"] = lesson_ids[idx]
        low_conf_questions.append(q)
    # Topic distribution
    topic_counts = pd.Series(preds_arr).value_counts().to_dict()
    # Hot topics: topics with >= min_count questions
    hot_topics = []
    for label, count in topic_counts.items():
        if count >= min_count:
            hot_topics.append({
                "label": label,
                "count": int(count),
                "pct": round(100.0 * count / n, 1),
            })
    # Lesson-level alerts: lessons with high proportion of low-confidence or hot topics
    lesson_alerts = []
    if lesson_ids is not None and len(lesson_ids) == n:
        df = pd.DataFrame({
            "lesson_id": lesson_ids,
            "pred_label": preds_arr,
            "confidence": conf_arr,
        })
        lesson_groups = df.groupby("lesson_id")
        for lesson_id, group in lesson_groups:
            total = len(group)
            low_conf = (group["confidence"] < threshold).sum()
            hot_topic = group["pred_label"].value_counts().idxmax()
            hot_topic_count = group["pred_label"].value_counts().max()
            alert = {}
            if total >= min_count and (low_conf / total > 0.5 or hot_topic_count / total > 0.7):
                alert = {
                    "lesson_id": lesson_id,
                    "total_questions": int(total),
                    "low_conf_pct": round(100.0 * low_conf / total, 1),
                    "hot_topic": hot_topic,
                    "hot_topic_pct": round(100.0 * hot_topic_count / total, 1),
                }
                lesson_alerts.append(alert)
    # Summary string
    summary = (
        f"{low_conf_count} of {n} questions ({format_confidence_score(low_conf_pct)}) "
        f"were classified with low confidence (<{int(threshold*100)}%). "
        f"{len(hot_topics)} topic(s) had at least {min_count} questions."
    )
    if lesson_alerts:
        summary += f" {len(lesson_alerts)} lesson(s) flagged for review."
    return {
        "low_confidence_count": low_conf_count,
        "low_confidence_pct": low_conf_pct,
        "low_confidence_questions": low_conf_questions,
        "hot_topics": hot_topics,
        "topic_distribution": {str(k): int(v) for k, v in topic_counts.items()},
        "lesson_alerts": lesson_alerts,
        "summary": summary,
    }


if __name__ == "__main__":
    # Minimal CLI/test: simulate predictions/confidences and print indicators
    import sys
    import random

    # Simulate data
    preds = ["concept", "clarification", "process", "concept", "other", "concept", "clarification", "process", "concept", "other"]
    confidences = [0.92, 0.65, 0.81, 0.55, 0.99, 0.60, 0.45, 0.88, 0.73, 0.67]
    question_texts = [
        "What does this term mean?",
        "Can you explain step 2 again?",
        "How do I submit the assignment?",
        "I'm confused about the formula.",
        "Is this topic on the test?",
        "What is the difference between X and Y?",
        "Could you clarify the instructions?",
        "What happens if I miss a deadline?",
        "Why is this method used?",
        "Is attendance mandatory?",
    ]
    lesson_ids = ["L1", "L1", "L2", "L2", "L1", "L3", "L3", "L2", "L1", "L3"]

    indicators = compute_intervention_indicators(
        preds=preds,
        confidences=confidences,
        threshold=0.7,
        min_count=2,
        question_texts=question_texts,
        lesson_ids=lesson_ids,
    )
    print("=== Intervention Indicators ===")
    print(f"Summary: {indicators['summary']}")
    print(f"Low-confidence questions ({indicators['low_confidence_count']}):")
    for q in indicators["low_confidence_questions"]:
        print(f"  - [{q.get('lesson_id','')}] {q.get('text','')} (label: {q['pred_label']}, conf: {q['confidence']})")
    print("Hot topics:")
    for t in indicators["hot_topics"]:
        print(f"  - {t['label']}: {t['count']} ({t['pct']}%)")
    print("Lesson alerts:")
    for alert in indicators["lesson_alerts"]:
        print(f"  - Lesson {alert['lesson_id']}: {alert['low_conf_pct']}% low-confidence, {alert['hot_topic']} ({alert['hot_topic_pct']}%)")
    print("Topic distribution:", indicators["topic_distribution"])
