"""
src/utils.py

General-purpose utility functions shared across modules.
"""

import os
import json
import hashlib
import random
import string
from datetime import datetime
from typing import Any, Dict, Optional, List, Union

import numpy as np
import pandas as pd


def set_random_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility across numpy, random, and pandas.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass  # torch not required; ignore if not installed


def safe_makedirs(path: str) -> None:
    """
    Create a directory if it does not exist.
    """
    os.makedirs(path, exist_ok=True)


def hash_string(s: str, length: int = 12) -> str:
    """
    Return a short, deterministic hash of a string (for anonymization).
    """
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:length]


def generate_id(prefix: str = "", length: int = 8) -> str:
    """
    Generate a random alphanumeric ID with optional prefix.
    """
    chars = string.ascii_letters + string.digits
    rand_part = ''.join(random.choices(chars, k=length))
    return f"{prefix}{rand_part}" if prefix else rand_part


def parse_datetime(dt_str: str) -> Optional[datetime]:
    """
    Parse a datetime string into a datetime object.
    Returns None if parsing fails.
    """
    if not isinstance(dt_str, str) or not dt_str.strip():
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%m/%d/%Y %H:%M", "%m/%d/%Y"):
        try:
            return datetime.strptime(dt_str.strip(), fmt)
        except ValueError:
            continue
    return None


def to_isoformat(dt: Union[datetime, str, None]) -> Optional[str]:
    """
    Convert a datetime object or string to ISO 8601 format.
    Returns None if input is invalid.
    """
    if isinstance(dt, datetime):
        return dt.isoformat()
    elif isinstance(dt, str):
        parsed = parse_datetime(dt)
        if parsed:
            return parsed.isoformat()
    return None


def deduplicate_dataframe(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Remove duplicate rows from a DataFrame, optionally on a subset of columns.
    """
    return df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)


def safe_read_csv(path: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file into a DataFrame, returning an empty DataFrame if file not found.
    """
    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError:
        return pd.DataFrame()
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def safe_write_csv(df: pd.DataFrame, path: str, **kwargs) -> None:
    """
    Write a DataFrame to CSV, creating parent directories if needed.
    """
    safe_makedirs(os.path.dirname(path))
    df.to_csv(path, index=False, **kwargs)


def load_json(path: str) -> Any:
    """
    Load a JSON file, returning None if file does not exist.
    """
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """
    Save an object as JSON, creating parent directories if needed.
    """
    safe_makedirs(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def is_anonymized_id(s: str) -> bool:
    """
    Heuristically check if a string looks like an anonymized ID (hex or random).
    """
    if not isinstance(s, str):
        return False
    return len(s) >= 8 and all(c in string.hexdigits for c in s)


def filter_empty_texts(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Remove rows where the specified text column is empty or null.
    """
    return df[df[text_col].astype(str).str.strip().astype(bool)].reset_index(drop=True)


def get_confusion_matrix_elements(y_true: np.ndarray, y_pred: np.ndarray, labels: Optional[List[str]] = None) -> Dict[str, int]:
    """
    Compute TP, FP, TN, FN for binary classification.
    """
    if labels is None:
        labels = list(set(y_true) | set(y_pred))
    if len(labels) != 2:
        raise ValueError("Only binary classification supported for confusion matrix elements.")
    pos, neg = labels
    tp = int(np.sum((y_true == pos) & (y_pred == pos)))
    tn = int(np.sum((y_true == neg) & (y_pred == neg)))
    fp = int(np.sum((y_true == neg) & (y_pred == pos)))
    fn = int(np.sum((y_true == pos) & (y_pred == neg)))
    return {"TP": tp, "TN": tn, "FP": fp, "FN": fn}


def format_confidence_score(score: float, decimals: int = 2) -> str:
    """
    Format a confidence score as a percentage string.
    """
    if not isinstance(score, (float, int)):
        return ""
    pct = round(100 * score, decimals)
    return f"{pct:.{decimals}f}%"


def get_env_variable(key: str, default: Optional[str] = None) -> Optional[str]:
    """
    Get an environment variable, returning default if not set.
    """
    return os.environ.get(key, default)


def flatten_list(nested: List[Any]) -> List[Any]:
    """
    Flatten a nested list (one level).
    """
    return [item for sublist in nested for item in (sublist if isinstance(sublist, list) else [sublist])]


def safe_int(val: Any, default: int = 0) -> int:
    """
    Safely convert a value to int, returning default on failure.
    """
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def safe_float(val: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float, returning default on failure.
    """
    try:
        return float(val)
    except (ValueError, TypeError):
        return default
