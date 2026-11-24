"""
src/feedback/feedback.py

Handles user feedback submission and aggregates for model refinement.

Exports:
    submit_feedback(question_id: str, user_feedback: str, user_id: str = None) -> bool
    aggregate_feedback() -> list[dict]

- Feedback is stored in a JSON file (path from FEEDBACK_STORE env variable or default).
- Each feedback entry includes a unique id, question_id, user_feedback, timestamp, and (optionally) user_id.
- Aggregation returns all feedback entries as a list of dicts.
"""

import os
import threading
from datetime import datetime
from typing import Optional, List, Dict, Any

from src.utils import (
    generate_id,
    to_isoformat,
    load_json,
    save_json,
    get_env_variable,
    safe_makedirs,
)

# Feedback storage path (default: feedback_store.json in project root)
FEEDBACK_STORE = get_env_variable("FEEDBACK_STORE", "feedback_store.json")
_FEEDBACK_LOCK = threading.Lock()


def _get_feedback_store_path() -> str:
    """
    Returns the absolute path to the feedback store file.
    """
    # If FEEDBACK_STORE is an absolute path, use as is; else, relative to project root
    if os.path.isabs(FEEDBACK_STORE):
        return FEEDBACK_STORE
    # Project root = two levels up from this file
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    return os.path.join(root, FEEDBACK_STORE)


def _load_feedback_entries() -> List[Dict[str, Any]]:
    """
    Loads all feedback entries from the feedback store.
    Returns an empty list if file does not exist.
    """
    path = _get_feedback_store_path()
    entries = load_json(path)
    if entries is None:
        return []
    if isinstance(entries, list):
        return entries
    # If file is corrupted or not a list, ignore and return empty
    return []


def _save_feedback_entries(entries: List[Dict[str, Any]]) -> None:
    """
    Saves all feedback entries to the feedback store.
    """
    path = _get_feedback_store_path()
    safe_makedirs(os.path.dirname(path))
    save_json(entries, path, indent=2)


def submit_feedback(
    question_id: str,
    user_feedback: str,
    user_id: Optional[str] = None,
) -> bool:
    """
    Submit user feedback for a given question.

    Args:
        question_id (str): The ID of the question being commented on.
        user_feedback (str): The feedback text or label.
        user_id (str, optional): Anonymized user ID (if available).

    Returns:
        bool: True if feedback was successfully stored, False otherwise.
    """
    if not isinstance(question_id, str) or not question_id.strip():
        return False
    if not isinstance(user_feedback, str) or not user_feedback.strip():
        return False

    feedback_entry = {
        "id": generate_id(prefix="fb_", length=10),
        "question_id": question_id.strip(),
        "user_feedback": user_feedback.strip(),
        "timestamp": to_isoformat(datetime.utcnow()),
    }
    if user_id and isinstance(user_id, str) and user_id.strip():
        feedback_entry["user_id"] = user_id.strip()

    try:
        with _FEEDBACK_LOCK:
            entries = _load_feedback_entries()
            entries.append(feedback_entry)
            _save_feedback_entries(entries)
        return True
    except Exception:
        return False


def aggregate_feedback() -> List[Dict[str, Any]]:
    """
    Aggregate all feedback entries.

    Returns:
        list[dict]: List of feedback entries, each with keys:
            - id
            - question_id
            - user_feedback
            - timestamp
            - user_id (optional)
    """
    try:
        with _FEEDBACK_LOCK:
            entries = _load_feedback_entries()
        return entries
    except Exception:
        return []


if __name__ == "__main__":
    import sys

    # Minimal CLI for demonstration/testing
    # Usage:
    #   python -m src.feedback.feedback submit <question_id> <user_feedback> [user_id]
    #   python -m src.feedback.feedback aggregate

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python -m src.feedback.feedback submit <question_id> <user_feedback> [user_id]")
        print("  python -m src.feedback.feedback aggregate")
        sys.exit(1)

    cmd = sys.argv[1].lower()
    if cmd == "submit":
        if len(sys.argv) < 4:
            print("Usage: python -m src.feedback.feedback submit <question_id> <user_feedback> [user_id]")
            sys.exit(1)
        question_id = sys.argv[2]
        user_feedback = sys.argv[3]
        user_id = sys.argv[4] if len(sys.argv) > 4 else None
        ok = submit_feedback(question_id, user_feedback, user_id)
        print("Feedback submitted." if ok else "Failed to submit feedback.")
    elif cmd == "aggregate":
        entries = aggregate_feedback()
        print(f"Total feedback entries: {len(entries)}")
        for entry in entries:
            print(entry)
    else:
        print("Unknown command.")
        sys.exit(1)
