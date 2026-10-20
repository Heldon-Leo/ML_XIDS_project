"""
utils.py
========
Shared helper utilities used across the ML-XIDS pipeline:
- logging setup
- timing decorator
- generic save/load helpers (joblib + json)
- memory-usage helper for dataframes
"""

import os
import json
import time
import logging
import functools

import joblib
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# LOGGING
# ---------------------------------------------------------------------------
def get_logger(name: str = "ML-XIDS") -> logging.Logger:
    """
    Returns a configured logger that prints timestamped, leveled messages
    to stdout. Reused across all modules for consistent pipeline logs.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:  # avoid duplicate handlers on repeated imports
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = get_logger()


# ---------------------------------------------------------------------------
# TIMING DECORATOR
# ---------------------------------------------------------------------------
def timeit(func):
    """Decorator that logs how long a pipeline step took to run."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        logger.info(f"--> Starting '{func.__name__}' ...")
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"<-- Finished '{func.__name__}' in {elapsed:.2f}s")
        return result

    return wrapper


# ---------------------------------------------------------------------------
# SAVE / LOAD HELPERS
# ---------------------------------------------------------------------------
def save_joblib(obj, path: str) -> None:
    """Persist any Python object (model, scaler, encoder) using joblib."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    logger.info(f"Saved artifact -> {path}")


def load_joblib(path: str):
    """Load a joblib-serialized object from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at: {path}")
    return joblib.load(path)


def save_json(obj, path: str) -> None:
    """Persist a JSON-serializable Python object to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    logger.info(f"Saved artifact -> {path}")


def load_json(path: str):
    """Load a JSON file from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Artifact not found at: {path}")
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# DATAFRAME HELPERS
# ---------------------------------------------------------------------------
def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """
    Downcasts numeric columns to the smallest safe dtype to reduce memory
    footprint. CSE-CIC-IDS2018 CSVs are large (multiple GB combined), so
    this materially helps on commodity hardware.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2

    for col in df.columns:
        col_type = df[col].dtype

        # Use an explicit numeric-dtype check rather than `!= object`, since
        # newer pandas versions (>= 2.x/3.x) may store text columns under a
        # dedicated StringDtype rather than the classic `object` dtype.
        if pd.api.types.is_numeric_dtype(df[col]):
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    if verbose:
        logger.info(
            f"Memory usage reduced from {start_mem:.2f} MB to {end_mem:.2f} MB "
            f"({100 * (start_mem - end_mem) / max(start_mem, 1e-9):.1f}% reduction)"
        )
    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names (CSE-CIC-IDS2018 has leading spaces)."""
    df.columns = [str(c).strip() for c in df.columns]
    return df
