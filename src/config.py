"""
Centralized configuration for project paths and random seed.

Exports:
    RAW_DIR (Path): Path to raw input data CSVs.
    PROCESSED_DIR (Path): Path to store processed data.
    RANDOM_SEED (int): Random seed for reproducibility.

Reads from environment variables if set, otherwise uses defaults.
"""

import os
from pathlib import Path

# Project root is always the parent of this file's parent (i.e., question_intelligence_prototype/)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

def _get_env_path(var_name: str, default_relative: str) -> Path:
    """Get a Path from environment or default relative to project root."""
    env_val = os.environ.get(var_name)
    if env_val:
        return Path(env_val).expanduser().resolve()
    return (PROJECT_ROOT / default_relative).resolve()

def _get_env_int(var_name: str, default: int) -> int:
    """Get an int from environment or default."""
    env_val = os.environ.get(var_name)
    if env_val is not None:
        try:
            return int(env_val)
        except ValueError:
            raise ValueError(f"Environment variable {var_name} must be an integer.")
    return default

# Paths
RAW_DIR = _get_env_path("RAW_DIR", "data/raw/")
PROCESSED_DIR = _get_env_path("PROCESSED_DIR", "data/processed/")

# Random seed
RANDOM_SEED = _get_env_int("RANDOM_SEED", 42)
