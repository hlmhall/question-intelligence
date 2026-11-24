"""
tests/test_feedback.py

Unit tests for src/feedback/feedback.py:
- submit_feedback
- aggregate_feedback

All file I/O is redirected to a temp file using monkeypatch.
"""

import os
import tempfile
import json
import time
from datetime import datetime
import pytest

from src.feedback import feedback

# Utilities for patching feedback store path
def patch_feedback_store(monkeypatch, tmp_path):
    """
    Redirect feedback store to a temp file for isolation.
    """
    store_path = tmp_path / "feedback_store.json"
    monkeypatch.setattr(feedback, "FEEDBACK_STORE", str(store_path))
    # Patch _get_feedback_store_path to use the temp file
    monkeypatch.setattr(feedback, "_get_feedback_store_path", lambda: str(store_path))
    return store_path

def read_feedback_file(store_path):
    if not os.path.exists(store_path):
        return []
    with open(store_path, "r", encoding="utf-8") as f:
        return json.load(f)

def test_submit_feedback_valid(monkeypatch, tmp_path):
    """
    Test that valid feedback is stored and retrievable.
    """
    store_path = patch_feedback_store(monkeypatch, tmp_path)
    qid = "q123"
    feedback_text = "This was misclassified."
    user_id = "u456"
    # Submit feedback
    ok = feedback.submit_feedback(qid, feedback_text, user_id)
    assert ok is True
    # Check file contents
    entries = read_feedback_file(store_path)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["question_id"] == qid
    assert entry["user_feedback"] == feedback_text
    assert entry["user_id"] == user_id
    # Timestamp is ISO format
    dt = datetime.fromisoformat(entry["timestamp"])
    assert isinstance(dt, datetime)
    assert entry["id"].startswith("fb_")
    assert len(entry["id"]) > 3

def test_submit_feedback_minimal(monkeypatch, tmp_path):
    """
    Test feedback submission with only required fields.
    """
    patch_feedback_store(monkeypatch, tmp_path)
    qid = "q789"
    feedback_text = "Unclear question."
    ok = feedback.submit_feedback(qid, feedback_text)
    assert ok is True
    entries = feedback.aggregate_feedback()
    assert len(entries) == 1
    entry = entries[0]
    assert entry["question_id"] == qid
    assert entry["user_feedback"] == feedback_text
    assert "user_id" not in entry

def test_submit_feedback_invalid(monkeypatch, tmp_path):
    """
    Test that invalid feedback (empty question_id or feedback) is rejected.
    """
    patch_feedback_store(monkeypatch, tmp_path)
    # Empty question_id
    assert feedback.submit_feedback("", "Some feedback") is False
    # Empty feedback
    assert feedback.submit_feedback("q1", "") is False
    # Both empty
    assert feedback.submit_feedback("", "") is False
    # Whitespace only
    assert feedback.submit_feedback("   ", "   ") is False
    # No file should be created
    entries = feedback.aggregate_feedback()
    assert entries == []

def test_aggregate_feedback_multiple(monkeypatch, tmp_path):
    """
    Test aggregation returns all submitted feedback entries.
    """
    patch_feedback_store(monkeypatch, tmp_path)
    # Submit several feedbacks
    items = [
        ("q1", "Feedback 1", "u1"),
        ("q2", "Feedback 2", "u2"),
        ("q3", "Feedback 3", None),
    ]
    for qid, text, uid in items:
        ok = feedback.submit_feedback(qid, text, uid)
        assert ok is True
    entries = feedback.aggregate_feedback()
    assert len(entries) == 3
    # Entries are in order of submission
    for i, (qid, text, uid) in enumerate(items):
        entry = entries[i]
        assert entry["question_id"] == qid
        assert entry["user_feedback"] == text
        if uid:
            assert entry["user_id"] == uid
        else:
            assert "user_id" not in entry

def test_feedback_file_corruption(monkeypatch, tmp_path):
    """
    If the feedback file is corrupted (not a list), aggregation returns empty.
    """
    store_path = patch_feedback_store(monkeypatch, tmp_path)
    # Write invalid JSON (dict instead of list)
    with open(store_path, "w", encoding="utf-8") as f:
        json.dump({"bad": "data"}, f)
    entries = feedback.aggregate_feedback()
    assert entries == []
    # Submitting new feedback should overwrite the file
    ok = feedback.submit_feedback("q9", "Valid feedback")
    assert ok is True
    entries = feedback.aggregate_feedback()
    assert len(entries) == 1
    assert entries[0]["question_id"] == "q9"

def test_thread_safety(monkeypatch, tmp_path):
    """
    Simulate concurrent feedback submissions.
    """
    patch_feedback_store(monkeypatch, tmp_path)
    import threading

    def submitter(qid, text, uid):
        feedback.submit_feedback(qid, text, uid)

    threads = []
    for i in range(10):
        t = threading.Thread(
            target=submitter,
            args=(f"q{i}", f"Feedback {i}", f"u{i}"),
        )
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    entries = feedback.aggregate_feedback()
    assert len(entries) == 10
    qids = {e["question_id"] for e in entries}
    assert qids == {f"q{i}" for i in range(10)}

def test_feedback_store_env(monkeypatch, tmp_path):
    """
    Test that FEEDBACK_STORE env variable is respected.
    """
    env_path = tmp_path / "env_feedback.json"
    monkeypatch.setenv("FEEDBACK_STORE", str(env_path))
    # Re-import feedback to pick up env var (simulate reload)
    import importlib
    import src.feedback.feedback as fb_mod
    importlib.reload(fb_mod)
    ok = fb_mod.submit_feedback("qenv", "Env feedback", "uenv")
    assert ok is True
    with open(env_path, "r", encoding="utf-8") as f:
        entries = json.load(f)
    assert len(entries) == 1
    assert entries[0]["question_id"] == "qenv"

def test_feedback_entry_uniqueness(monkeypatch, tmp_path):
    """
    Each feedback entry should have a unique id.
    """
    patch_feedback_store(monkeypatch, tmp_path)
    ids = set()
    for i in range(5):
        ok = feedback.submit_feedback(f"q{i}", f"Feedback {i}", f"u{i}")
        assert ok is True
        entry = feedback.aggregate_feedback()[-1]
        assert entry["id"] not in ids
        ids.add(entry["id"])
    assert len(ids) == 5

def test_feedback_timestamp_order(monkeypatch, tmp_path):
    """
    Feedback entries should have increasing timestamps if submitted sequentially.
    """
    patch_feedback_store(monkeypatch, tmp_path)
    times = []
    for i in range(3):
        ok = feedback.submit_feedback(f"q{i}", f"Feedback {i}")
        assert ok is True
        entry = feedback.aggregate_feedback()[-1]
        times.append(datetime.fromisoformat(entry["timestamp"]))
        time.sleep(0.01)
    assert times[0] <= times[1] <= times[2]
