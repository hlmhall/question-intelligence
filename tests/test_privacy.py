"""
tests/test_privacy.py

Unit tests for privacy safeguard modules in src/privacy/privacy.py.
Covers anonymization and privacy compliance checks.

Requires:
    - pytest
    - pandas
    - src/privacy/privacy.py
    - src/utils.py
"""

import pytest
import pandas as pd

from src.privacy.privacy import anonymize_data, check_privacy_compliance

# --- Fixtures and helpers ---

@pytest.fixture
def sample_df():
    """A DataFrame with various ID and PII-like columns for testing."""
    return pd.DataFrame({
        "id": ["user1", "user2", "user3", ""],
        "user_id": ["user1", "user2", "user3", "user3"],
        "email": ["alice@example.com", "bob@example.com", "", None],
        "phone": ["123-456-7890", "", "555 123 4567", None],
        "name": ["Alice", "Bob", "Charlie", ""],
        "text": ["What is Python?", "How to use pandas?", "What is a DataFrame?", ""],
        "label": ["concept", "usage", "concept", ""],
    })


@pytest.fixture
def already_anonymized_df():
    """A DataFrame where IDs are already anonymized (hex strings)."""
    return pd.DataFrame({
        "id": [
            "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a1",
            "c3d4e5f6a7b8c9d0e1f2a3b4c5d6a1b2",
            ""
        ],
        "user_id": [
            "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6",
            "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6a1",
            "c3d4e5f6a7b8c9d0e1f2a3b4c5d6a1b2",
            "c3d4e5f6a7b8c9d0e1f2a3b4c5d6a1b2"
        ],
        "text": ["foo", "bar", "baz", ""],
        "label": ["x", "y", "z", ""],
    })


# --- Tests for anonymize_data ---

def test_anonymize_data_hashes_ids(sample_df):
    """Test that anonymize_data hashes id columns and leaves other columns unchanged."""
    id_columns = ["id", "user_id"]
    df_anon = anonymize_data(sample_df, id_columns)
    # IDs should be replaced with 32-char hex strings (from hash_string)
    for col in id_columns:
        for orig, anon in zip(sample_df[col], df_anon[col]):
            if orig.strip():
                assert isinstance(anon, str)
                assert len(anon) == 32
                assert anon != orig
            else:
                assert anon == ""
    # Non-ID columns should remain unchanged
    assert (df_anon["text"] == sample_df["text"]).all()
    assert (df_anon["label"] == sample_df["label"]).all()


def test_anonymize_data_idempotent(already_anonymized_df):
    """Test that anonymize_data does not re-hash already anonymized IDs."""
    id_columns = ["id", "user_id"]
    df_anon = anonymize_data(already_anonymized_df, id_columns)
    for col in id_columns:
        # Already-anonymized values should remain unchanged
        assert (df_anon[col] == already_anonymized_df[col]).all()


def test_anonymize_data_handles_missing_columns(sample_df):
    """Test that anonymize_data does not fail if a specified column is missing."""
    # 'nonexistent_id' does not exist in df
    df_anon = anonymize_data(sample_df, ["id", "nonexistent_id"])
    assert "id" in df_anon.columns
    assert "nonexistent_id" not in df_anon.columns


def test_anonymize_data_empty_and_nonstring(sample_df):
    """Test that anonymize_data handles empty strings and non-string values."""
    df = sample_df.copy()
    df.loc[0, "id"] = None
    df.loc[1, "user_id"] = 12345  # non-string
    df_anon = anonymize_data(df, ["id", "user_id"])
    assert df_anon.loc[0, "id"] == ""
    assert isinstance(df_anon.loc[1, "user_id"], str)
    assert len(df_anon.loc[1, "user_id"]) == 32


# --- Tests for check_privacy_compliance ---

def test_check_privacy_compliance_detects_pii(sample_df):
    """Test that check_privacy_compliance flags PII columns and values."""
    compliant = check_privacy_compliance(sample_df)
    assert compliant is False  # Should detect 'email', 'phone', 'name' columns and values


def test_check_privacy_compliance_passes_on_anon_ids(already_anonymized_df):
    """Test that check_privacy_compliance passes when IDs are anonymized and no PII columns."""
    compliant = check_privacy_compliance(already_anonymized_df)
    assert compliant is True


def test_check_privacy_compliance_flags_non_anonymized_ids(sample_df):
    """Test that check_privacy_compliance flags non-anonymized IDs in 'id' and 'user_id'."""
    # Remove PII-like columns to isolate ID check
    df = sample_df.drop(columns=["email", "phone", "name"])
    compliant = check_privacy_compliance(df)
    assert compliant is False


def test_check_privacy_compliance_flags_empty_ids(already_anonymized_df):
    """Test that check_privacy_compliance flags empty anonymized IDs."""
    df = already_anonymized_df.copy()
    df.loc[3, "id"] = ""
    compliant = check_privacy_compliance(df)
    assert compliant is False


def test_check_privacy_compliance_detects_email_pattern():
    """Test that check_privacy_compliance flags email-like values in non-PII columns."""
    df = pd.DataFrame({
        "id": ["a" * 32, "b" * 32],
        "user_id": ["c" * 32, "d" * 32],
        "text": ["Contact me at alice@example.com", "No email here"]
    })
    compliant = check_privacy_compliance(df)
    assert compliant is False


def test_check_privacy_compliance_detects_phone_pattern():
    """Test that check_privacy_compliance flags phone-like values in non-PII columns."""
    df = pd.DataFrame({
        "id": ["a" * 32, "b" * 32],
        "user_id": ["c" * 32, "d" * 32],
        "text": ["My number is 123-456-7890", "No phone here"]
    })
    compliant = check_privacy_compliance(df)
    assert compliant is False
