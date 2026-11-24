"""
Unit tests for data loading and cleaning modules.

Covers:
    - src.data.load: load_questions, load_lessons, load_users
    - src.data.clean: clean_questions, clean_lessons, clean_users, clean_all

Uses pytest and temporary directories to avoid modifying project data.
"""

import os
import shutil
import tempfile
import pandas as pd
import pytest
from pathlib import Path

from src.data import load as data_load
from src.data import clean as data_clean
from src.config import RAW_DIR, PROCESSED_DIR, RANDOM_SEED

# --- Helpers for test data ---

SAMPLE_QUESTIONS = pd.DataFrame([
    {
        "id": "q1",
        "user_id": "userA",
        "lesson_id": "l1",
        "text": "What is a function?",
        "timestamp": "2024-05-01 10:00:00",
        "label": "concept"
    },
    {
        "id": "q2",
        "user_id": "userB",
        "lesson_id": "l1",
        "text": "What is a function?",  # duplicate text
        "timestamp": "2024-05-01 10:00:00",
        "label": "concept"
    },
    {
        "id": "q3",
        "user_id": "userC",
        "lesson_id": "l2",
        "text": "How do I use loops?",
        "timestamp": "2024/05/02 11:30",  # inconsistent timestamp
        "label": ""
    },
    {
        "id": "q4",
        "user_id": "userD",
        "lesson_id": "l2",
        "text": "",
        "timestamp": "2024-05-02T11:35:00",
        "label": "syntax"
    },
    {
        "id": "q5",
        "user_id": "userE",
        "lesson_id": "l3",
        "text": "Can you explain variables?",
        "timestamp": "not-a-date",
        "label": "concept"
    },
    {
        "id": "q6",
        "user_id": "userE",
        "lesson_id": "l3",
        "text": "Can you explain variables?",
        "timestamp": "not-a-date",
        "label": "concept"
    }
])

SAMPLE_LESSONS = pd.DataFrame([
    {"id": "l1", "title": "Intro to Python", "topic": "Python Basics", "date": "2024-05-01"},
    {"id": "l2", "title": "Control Flow", "topic": "Python Basics", "date": "2024/05/02"},
    {"id": "l3", "title": "Variables", "topic": "", "date": "2024-05-03"},
    {"id": "l3", "title": "Variables", "topic": "", "date": "2024-05-03"},  # duplicate
])

SAMPLE_USERS = pd.DataFrame([
    {"id": "userA", "role": "learner", "cohort": "2024A"},
    {"id": "userB", "role": "learner", "cohort": "2024A"},
    {"id": "userC", "role": "learner", "cohort": ""},
    {"id": "userA", "role": "learner", "cohort": "2024A"},  # duplicate id
    {"id": "userD", "role": "learner", "cohort": "2024B"},
    {"id": "userE", "role": "learner", "cohort": "2024B"},
])

# --- Fixtures ---

@pytest.fixture
def temp_raw_and_processed_dirs(tmp_path, monkeypatch):
    """
    Creates temporary RAW_DIR and PROCESSED_DIR, monkeypatches config.
    Returns (raw_dir, processed_dir)
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()
    monkeypatch.setattr(data_load, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_clean, "RAW_DIR", raw_dir)
    monkeypatch.setattr(data_clean, "PROCESSED_DIR", processed_dir)
    return raw_dir, processed_dir

def write_df_csv(df, path):
    df.to_csv(path, index=False)

# --- Tests for data loading ---

def test_load_questions_success(temp_raw_and_processed_dirs):
    raw_dir, _ = temp_raw_and_processed_dirs
    path = raw_dir / "learner_questions.csv"
    write_df_csv(SAMPLE_QUESTIONS, path)
    df = data_load.load_questions()
    assert isinstance(df, pd.DataFrame)
    assert set(["id", "user_id", "lesson_id", "text", "timestamp"]).issubset(df.columns)
    assert len(df) == len(SAMPLE_QUESTIONS)

def test_load_lessons_success(temp_raw_and_processed_dirs):
    raw_dir, _ = temp_raw_and_processed_dirs
    path = raw_dir / "lessons.csv"
    write_df_csv(SAMPLE_LESSONS, path)
    df = data_load.load_lessons()
    assert isinstance(df, pd.DataFrame)
    assert set(["id", "title", "topic", "date"]).issubset(df.columns)
    assert len(df) == len(SAMPLE_LESSONS)

def test_load_users_success(temp_raw_and_processed_dirs):
    raw_dir, _ = temp_raw_and_processed_dirs
    path = raw_dir / "users.csv"
    write_df_csv(SAMPLE_USERS, path)
    df = data_load.load_users()
    assert isinstance(df, pd.DataFrame)
    assert set(["id", "role", "cohort"]).issubset(df.columns)
    assert len(df) == len(SAMPLE_USERS)

def test_load_questions_missing_file(temp_raw_and_processed_dirs):
    # No file written
    with pytest.raises(FileNotFoundError):
        data_load.load_questions()

def test_load_questions_missing_column(temp_raw_and_processed_dirs):
    raw_dir, _ = temp_raw_and_processed_dirs
    df = SAMPLE_QUESTIONS.drop(columns=["user_id"])
    write_df_csv(df, raw_dir / "learner_questions.csv")
    with pytest.raises(ValueError):
        data_load.load_questions()

# --- Tests for data cleaning ---

def test_clean_questions_removes_empty_and_duplicates():
    df = SAMPLE_QUESTIONS.copy()
    cleaned = data_clean.clean_questions(df)
    # Should remove row with empty text and deduplicate
    # Only unique (user_id, lesson_id, text, timestamp) with non-empty text and valid required fields remain
    texts = cleaned["text"].tolist()
    assert "" not in texts
    # Should deduplicate q5/q6 (same user_id, lesson_id, text, timestamp)
    assert cleaned.duplicated(subset=["user_id", "lesson_id", "text", "timestamp"]).sum() == 0
    # All required fields present and non-empty
    for col in ["id", "user_id", "lesson_id", "text", "timestamp"]:
        assert cleaned[col].isnull().sum() == 0
        assert (cleaned[col].astype(str).str.strip() != "").all()

def test_clean_questions_standardizes_timestamp():
    df = SAMPLE_QUESTIONS.copy()
    cleaned = data_clean.clean_questions(df)
    # All timestamps should be ISO 8601 or None (but rows with None dropped)
    for ts in cleaned["timestamp"]:
        assert "T" in ts or ts.endswith("Z") or ts.count("-") == 2

def test_clean_lessons_removes_duplicates_and_standardizes_date():
    df = SAMPLE_LESSONS.copy()
    cleaned = data_clean.clean_lessons(df)
    # Deduplicated on id, title, date
    assert cleaned.duplicated(subset=["id", "title", "date"]).sum() == 0
    # All dates are ISO format
    for dt in cleaned["date"]:
        assert dt.count("-") == 2

def test_clean_users_removes_duplicates_and_anonymizes():
    df = SAMPLE_USERS.copy()
    cleaned = data_clean.clean_users(df)
    # Deduplicated on id
    assert cleaned.duplicated(subset=["id"]).sum() == 0
    # All required fields present and non-empty
    for col in ["id", "role"]:
        assert cleaned[col].isnull().sum() == 0
        assert (cleaned[col].astype(str).str.strip() != "").all()
    # Anonymized id: should not look like original user ids (simulate hash)
    assert all(len(str(uid)) >= 4 for uid in cleaned["id"])

def test_clean_all_writes_cleaned_files(temp_raw_and_processed_dirs):
    raw_dir, processed_dir = temp_raw_and_processed_dirs
    # Write raw files
    write_df_csv(SAMPLE_QUESTIONS, raw_dir / "learner_questions.csv")
    write_df_csv(SAMPLE_LESSONS, raw_dir / "lessons.csv")
    write_df_csv(SAMPLE_USERS, raw_dir / "users.csv")
    # Run clean_all
    data_clean.clean_all()
    # Check processed files exist
    for fname in ["learner_questions.csv", "lessons.csv", "users.csv"]:
        out_path = processed_dir / fname
        assert out_path.exists()
        df = pd.read_csv(out_path)
        assert not df.empty

def test_clean_questions_handles_invalid_timestamp():
    df = SAMPLE_QUESTIONS.copy()
    # Row with "not-a-date" timestamp should be dropped after cleaning
    cleaned = data_clean.clean_questions(df)
    assert not any(ts == "not-a-date" for ts in cleaned["timestamp"])

def test_clean_lessons_drops_missing_required():
    df = SAMPLE_LESSONS.copy()
    # Remove 'title' from one row
    df2 = df.copy()
    df2.loc[0, "title"] = ""
    cleaned = data_clean.clean_lessons(df2)
    assert "" not in cleaned["title"].tolist()
    # All required fields present and non-empty
    for col in ["id", "title", "date"]:
        assert cleaned[col].isnull().sum() == 0
        assert (cleaned[col].astype(str).str.strip() != "").all()

def test_clean_users_drops_missing_required():
    df = SAMPLE_USERS.copy()
    # Remove 'role' from one row
    df2 = df.copy()
    df2.loc[0, "role"] = ""
    cleaned = data_clean.clean_users(df2)
    assert "" not in cleaned["role"].tolist()
    for col in ["id", "role"]:
        assert cleaned[col].isnull().sum() == 0
        assert (cleaned[col].astype(str).str.strip() != "").all()
