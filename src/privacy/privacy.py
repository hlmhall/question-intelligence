"""
src/privacy/privacy.py

Implements privacy safeguards and compliance checks for data handling.

Exports:
    anonymize_data(df: pd.DataFrame, id_columns: list[str]) -> pd.DataFrame
    check_privacy_compliance(df: pd.DataFrame) -> bool

- Anonymizes specified ID columns using deterministic hashing.
- Checks for privacy compliance: no direct identifiers, no empty anonymized IDs, no PII leaks.
"""

import pandas as pd
import logging
from typing import List

from src.utils import hash_string, is_anonymized_id

logger = logging.getLogger(__name__)

# List of columns considered direct identifiers (should always be anonymized)
DIRECT_IDENTIFIER_COLUMNS = {"user_id", "id"}

# List of columns that may contain PII (expand as needed)
PII_LIKE_COLUMNS = {"email", "phone", "name", "full_name", "first_name", "last_name", "address"}

def anonymize_data(df: pd.DataFrame, id_columns: List[str]) -> pd.DataFrame:
    """
    Anonymize specified ID columns in the DataFrame using deterministic hashing.

    Args:
        df (pd.DataFrame): Input DataFrame.
        id_columns (list[str]): List of column names to anonymize.

    Returns:
        pd.DataFrame: DataFrame with anonymized ID columns.
    """
    df = df.copy()
    for col in id_columns:
        if col in df.columns:
            # Only hash values that are not already anonymized
            def _anonymize(val):
                if not isinstance(val, str) or not val.strip():
                    return ""
                if is_anonymized_id(val):
                    return val
                return hash_string(val)
            df[col] = df[col].apply(_anonymize)
    return df

def check_privacy_compliance(df: pd.DataFrame) -> bool:
    """
    Check if the DataFrame is privacy compliant:
        - No direct identifiers (e.g., raw emails, names, phone numbers)
        - All ID columns are anonymized (hex/random, not raw)
        - No empty anonymized IDs in required columns

    Args:
        df (pd.DataFrame): DataFrame to check.

    Returns:
        bool: True if compliant, False otherwise.
    """
    compliant = True

    # Check for PII-like columns
    found_pii_cols = [col for col in df.columns if col.lower() in PII_LIKE_COLUMNS]
    if found_pii_cols:
        logger.warning(f"Potential PII columns found: {found_pii_cols}")
        compliant = False

    # Check that all direct identifier columns are anonymized
    for col in DIRECT_IDENTIFIER_COLUMNS:
        if col in df.columns:
            # If any value is not anonymized, flag as non-compliant
            non_anon = df[col].dropna().astype(str).apply(lambda x: not is_anonymized_id(x))
            if non_anon.any():
                logger.warning(f"Column '{col}' contains non-anonymized values.")
                compliant = False
            # If any value is empty, flag as non-compliant
            empty_ids = df[col].astype(str).str.strip() == ""
            if empty_ids.any():
                logger.warning(f"Column '{col}' contains empty anonymized IDs.")
                compliant = False

    # Check for obvious PII patterns in string columns (very basic)
    for col in df.select_dtypes(include="object").columns:
        # Check for '@' in values (possible email)
        if df[col].astype(str).str.contains("@").any():
            logger.warning(f"Possible email addresses found in column '{col}'.")
            compliant = False
        # Check for phone-like patterns (very basic)
        if df[col].astype(str).str.contains(r"\d{3,}[-\s]?\d{3,}").any():
            logger.warning(f"Possible phone numbers found in column '{col}'.")
            compliant = False

    return compliant
