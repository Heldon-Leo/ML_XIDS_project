"""
src/feature_stats.py
=====================
Per-feature training statistics in RAW (unscaled) units.

Why the live monitor needs this
-------------------------------
* Training imputed NaN/inf values with column medians. A live flow with an
  inf (e.g. Flow Byts/s when Flow Duration is 0) must be treated the same
  way, otherwise live predictions silently drift from what the model saw.
* Comparing live values against the training min/max is the most practical
  way to detect unit or definition mismatches between CICFlowMeter and the
  CSE-CIC-IDS2018 files (e.g. durations in seconds vs microseconds).

The statistics are rebuilt from the cached training split
(models/preprocessed_split.pkl) by inverting the saved MinMaxScaler, so no
re-training or --force run is needed. They are cached in
models/feature_stats.json.
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np

import config
from src.utils import get_logger, load_joblib

logger = get_logger("feature_stats")


def build_feature_stats(features: List[str]) -> Optional[Dict[str, Dict[str, float]]]:
    """Builds stats from the cached training split. Returns None if unavailable."""
    if not (os.path.exists(config.SPLIT_DATA_CACHE_PATH) and os.path.exists(config.SCALER_PATH)):
        return None

    data = load_joblib(config.SPLIT_DATA_CACHE_PATH)
    scaler = load_joblib(config.SCALER_PATH)

    cached_features = list(getattr(data, "feature_columns", []))
    if cached_features != list(features):
        logger.warning(
            "Cached split feature list differs from feature_columns.json; "
            "not building training statistics from it."
        )
        return None

    raw_train = scaler.inverse_transform(np.asarray(data.X_train))
    stats = {}
    for i, name in enumerate(features):
        col = raw_train[:, i]
        col = col[np.isfinite(col)]
        if col.size == 0:
            continue
        stats[name] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "median": float(np.median(col)),
            "p01": float(np.percentile(col, 1)),
            "p99": float(np.percentile(col, 99)),
        }
    return stats


def load_or_build_feature_stats(features: List[str]) -> Optional[Dict[str, Dict[str, float]]]:
    """Loads models/feature_stats.json, building it from the cached split if absent."""
    if os.path.exists(config.FEATURE_STATS_PATH):
        with open(config.FEATURE_STATS_PATH, encoding="utf-8") as f:
            stats = json.load(f)
        if all(name in stats for name in features):
            return stats
        logger.warning("feature_stats.json does not cover the current feature list; rebuilding.")

    stats = build_feature_stats(features)
    if stats is None:
        logger.warning(
            "Training statistics unavailable (no cached split / scaler). Live rows with "
            "missing values will be rejected and value-drift checks will be skipped."
        )
        return None

    with open(config.FEATURE_STATS_PATH, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved training feature statistics -> {config.FEATURE_STATS_PATH}")
    return stats
