"""
Data loading utilities for Practical Foundations for Question Intelligence.

Exports:
    load_questions() -> pd.DataFrame
    load_lessons() -> pd.DataFrame
    load_users() -> pd.DataFrame

Loads raw CSV data from the paths specified in src/config.py.
Handles common loading errors and ensures consistent column names.
"""

import pandas as pd
from pathlib import Path
from typing import Optional
import logging

from src.config import RAW_DIR

# Configure module-level logger
logger = logging.getLogger(__name__)


def _load_csv(filename: str, required_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Load a CSV file from RAW_DIR, ensuring required columns exist.

    Args:
        filename (str): Name of the CSV file (e.g., 'learner_questions.csv').
        required_columns (list, optional): List of required column names.

    Returns:
        pd.DataFrame: Loaded DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = RAW_DIR / filename
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        raise FileNotFoundError(f"Raw data file not found: {csv_path}")

    try:
        df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    except Exception as e:
        logger.error(f"Error loading {csv_path}: {e}")
        raise

    if required_columns:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            logger.error(f"Missing columns in {filename}: {missing}")
            raise ValueError(f"Missing required columns in {filename}: {missing}")

    return df


def load_questions() -> pd.DataFrame:
    """
    Load learner questions from data/raw/learner_questions.csv.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - id
            - user_id
            - lesson_id
            - text
            - timestamp
            - label (optional)
    """
    required_columns = ["id", "user_id", "lesson_id", "text", "timestamp"]
    # 'label' is optional
    return _load_csv("learner_questions.csv", required_columns=required_columns)


def load_lessons() -> pd.DataFrame:
    """
    Load lesson metadata from data/raw/lessons.csv.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - id
            - title
            - topic
            - date
    """
    required_columns = ["id", "title", "topic", "date"]
    return _load_csv("lessons.csv", required_columns=required_columns)


def load_users() -> pd.DataFrame:
    """
    Load user metadata from data/raw/users.csv.

    Returns:
        pd.DataFrame: DataFrame with columns:
            - id
            - role
            - cohort
    """
    required_columns = ["id", "role", "cohort"]
    return _load_csv("users.csv", required_columns=required_columns)
