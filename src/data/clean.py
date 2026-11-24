"""
src/data/clean.py

Cleans and preprocesses raw data, writes processed outputs.

Exports:
    clean_questions(df: pd.DataFrame) -> pd.DataFrame
    clean_all() -> None

- Cleans learner questions, lessons, and users data.
- Handles missing values, deduplication, timestamp normalization, and privacy checks.
- Writes cleaned DataFrames to PROCESSED_DIR.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

from src.config import RAW_DIR, PROCESSED_DIR, RANDOM_SEED
from src.data.load import load_questions, load_lessons, load_users
from src.privacy.privacy import anonymize_data, check_privacy_compliance
from src.utils import (
    deduplicate_dataframe,
    filter_empty_texts,
    parse_datetime,
    to_isoformat,
    safe_write_csv,
    set_random_seed,
)

import os

# Configure module-level logger
logger = logging.getLogger(__name__)


def clean_questions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess learner questions DataFrame.

    Steps:
        - Remove rows with empty or null 'text'
        - Deduplicate on ['user_id', 'lesson_id', 'text', 'timestamp']
        - Standardize 'timestamp' to ISO 8601, set to None if invalid
        - Strip whitespace from all string columns
        - Optionally anonymize user_id if not already anonymized
        - Drop rows with missing required fields after cleaning

    Args:
        df (pd.DataFrame): Raw questions DataFrame

    Returns:
        pd.DataFrame: Cleaned questions DataFrame
    """
    # Remove empty or null question texts
    df = filter_empty_texts(df, text_col="text")

    # Strip whitespace from all string columns
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    # Deduplicate
    df = deduplicate_dataframe(df, subset=["user_id", "lesson_id", "text", "timestamp"])

    # Standardize timestamps
    def _fix_timestamp(ts):
        dt = parse_datetime(ts)
        return to_isoformat(dt) if dt else None

    df["timestamp"] = df["timestamp"].apply(_fix_timestamp)

    # Drop rows with missing required fields after cleaning
    required = ["id", "user_id", "lesson_id", "text", "timestamp"]
    df = df.dropna(subset=required)
    for col in required:
        df = df[df[col].astype(str).str.strip().astype(bool)]

    # Anonymize user_id if not already anonymized
    df = anonymize_data(df, id_columns=["user_id"])

    # Check privacy compliance
    if not check_privacy_compliance(df):
        logger.warning("Privacy compliance check failed for questions data.")

    # Reset index
    df = df.reset_index(drop=True)
    return df


def clean_lessons(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess lessons DataFrame.

    Steps:
        - Strip whitespace from all string columns
        - Deduplicate on ['id', 'title', 'date']
        - Standardize 'date' to ISO 8601, set to None if invalid
        - Drop rows with missing required fields after cleaning

    Args:
        df (pd.DataFrame): Raw lessons DataFrame

    Returns:
        pd.DataFrame: Cleaned lessons DataFrame
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = deduplicate_dataframe(df, subset=["id", "title", "date"])

    # Standardize date
    def _fix_date(dt):
        parsed = parse_datetime(dt)
        return parsed.date().isoformat() if parsed else None

    df["date"] = df["date"].apply(_fix_date)

    # Drop rows with missing required fields
    required = ["id", "title", "date"]
    df = df.dropna(subset=required)
    for col in required:
        df = df[df[col].astype(str).str.strip().astype(bool)]

    # Reset index
    df = df.reset_index(drop=True)
    return df


def clean_users(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess users DataFrame.

    Steps:
        - Strip whitespace from all string columns
        - Deduplicate on ['id']
        - Drop rows with missing required fields after cleaning
        - Anonymize id if not already anonymized
        - Check privacy compliance

    Args:
        df (pd.DataFrame): Raw users DataFrame

    Returns:
        pd.DataFrame: Cleaned users DataFrame
    """
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()

    df = deduplicate_dataframe(df, subset=["id"])

    # Drop rows with missing required fields
    required = ["id", "role"]
    df = df.dropna(subset=required)
    for col in required:
        df = df[df[col].astype(str).str.strip().astype(bool)]

    # Anonymize id if not already anonymized
    df = anonymize_data(df, id_columns=["id"])

    # Check privacy compliance
    if not check_privacy_compliance(df):
        logger.warning("Privacy compliance check failed for users data.")

    # Reset index
    df = df.reset_index(drop=True)
    return df


def clean_all() -> None:
    """
    Clean all raw data files and write cleaned outputs to PROCESSED_DIR.

    - Cleans learner_questions.csv, lessons.csv, users.csv
    - Writes cleaned CSVs to PROCESSED_DIR
    """
    set_random_seed(RANDOM_SEED)

    # Ensure output directory exists
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # Clean questions
    try:
        raw_questions = load_questions()
        cleaned_questions = clean_questions(raw_questions)
        safe_write_csv(cleaned_questions, os.path.join(PROCESSED_DIR, "learner_questions.csv"))
        logger.info(f"Cleaned questions written to {os.path.join(PROCESSED_DIR, 'learner_questions.csv')}")
    except Exception as e:
        logger.error(f"Failed to clean questions: {e}")

    # Clean lessons
    try:
        raw_lessons = load_lessons()
        cleaned_lessons = clean_lessons(raw_lessons)
        safe_write_csv(cleaned_lessons, os.path.join(PROCESSED_DIR, "lessons.csv"))
        logger.info(f"Cleaned lessons written to {os.path.join(PROCESSED_DIR, 'lessons.csv')}")
    except Exception as e:
        logger.error(f"Failed to clean lessons: {e}")

    # Clean users
    try:
        raw_users = load_users()
        cleaned_users = clean_users(raw_users)
        safe_write_csv(cleaned_users, os.path.join(PROCESSED_DIR, "users.csv"))
        logger.info(f"Cleaned users written to {os.path.join(PROCESSED_DIR, 'users.csv')}")
    except Exception as e:
        logger.error(f"Failed to clean users: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    clean_all()
