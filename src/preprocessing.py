"""
preprocessing.py
=================
Cleans the merged CSE-CIC-IDS2018 DataFrame and prepares it for model
training:

1. Strip column names (defensive, also done in data_loader)
2. Drop identifier / metadata columns that are not real traffic features
3. Replace inf/-inf with NaN, then drop or impute NaNs
4. Encode the multi-class `Label` column with LabelEncoder
5. Scale numeric features with MinMaxScaler
6. Stratified train/test split (80/20)

Returns everything downstream steps need (train/test arrays, the fitted
encoder/scaler, and the final feature column list) so training,
evaluation, and explanation stay perfectly aligned.
"""

import os
import sys
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import (
    get_logger,
    timeit,
    save_joblib,
    save_json,
    clean_column_names,
)

logger = get_logger(__name__)


@dataclass
class PreprocessedData:
    """Container bundling every artifact produced by preprocessing."""

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_columns: List[str]
    label_encoder: LabelEncoder
    scaler: MinMaxScaler
    # Kept for SHAP / local explanations, which read raw (unscaled) values
    # so that the generated text is human-interpretable (e.g. actual packet
    # counts rather than a 0-1 normalized number).
    X_test_raw: pd.DataFrame


def _drop_identifier_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drops IP/timestamp/ID-style columns that are not generalizable features."""
    cols_to_drop = [c for c in config.DROP_COLUMNS if c in df.columns]
    if cols_to_drop:
        logger.info(f"Dropping identifier/metadata columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    return df


def _handle_missing_and_infinite(df: pd.DataFrame) -> pd.DataFrame:
    """
    CICFlowMeter output frequently contains inf/-inf (e.g. division by
    zero when computing rate features) and occasional NaNs. We replace
    inf with NaN, then drop rows that are still NaN in any feature column
    (a small fraction of the total dataset).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    n_inf_before = np.isinf(df[numeric_cols]).sum().sum()
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    logger.info(f"Replaced {n_inf_before} infinite values with NaN")

    n_before = len(df)
    n_na = df[numeric_cols].isna().sum().sum()
    if n_na > 0:
        # Impute rather than drop when NaNs are a small fraction, to avoid
        # discarding legitimate attack samples that are already rare.
        na_fraction = n_na / (df[numeric_cols].shape[0] * df[numeric_cols].shape[1])
        if na_fraction < 0.05:
            logger.info(
                f"Imputing {n_na} NaN values with column medians "
                f"({na_fraction:.4%} of numeric cells)"
            )
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        else:
            logger.info(f"Dropping rows containing NaN values ({n_na} NaN cells)")
            df = df.dropna(subset=numeric_cols)

    logger.info(f"Rows before cleaning: {n_before}, after cleaning: {len(df)}")
    return df


def _drop_constant_and_duplicate_columns(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    """
    Removes zero-variance columns (uninformative for the model and can
    break SHAP/scalers) and exact duplicate columns.
    """
    feature_cols = [c for c in df.columns if c != label_col]

    # Drop constant columns
    nunique = df[feature_cols].nunique()
    constant_cols = nunique[nunique <= 1].index.tolist()
    if constant_cols:
        logger.info(f"Dropping {len(constant_cols)} constant column(s): {constant_cols}")
        df = df.drop(columns=constant_cols)

    # Drop exact duplicate columns (keep first occurrence)
    df = df.loc[:, ~df.columns.duplicated()]

    return df


@timeit
def preprocess(df: pd.DataFrame) -> PreprocessedData:
    """Full preprocessing pipeline. Returns a PreprocessedData bundle."""

    df = clean_column_names(df)
    df = _drop_identifier_columns(df)

    label_col = config.LABEL_COLUMN
    if label_col not in df.columns:
        raise KeyError(f"Label column '{label_col}' missing from dataframe")

    # Keep only rows with a non-null label
    df = df.dropna(subset=[label_col])

    # Coerce all non-label columns to numeric (CICFlowMeter sometimes writes
    # numeric columns as strings, e.g. "Infinity" or blank strings)
    feature_cols_all = [c for c in df.columns if c != label_col]
    for col in feature_cols_all:
        # Covers both the classic `object` dtype and the dedicated
        # StringDtype used by newer pandas versions for text columns.
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = _handle_missing_and_infinite(df)
    df = _drop_constant_and_duplicate_columns(df, label_col)

    feature_columns = [c for c in df.columns if c != label_col]
    logger.info(f"Final feature count: {len(feature_columns)}")

    # ---------------------------------------------------------------
    # Label encoding (multi-class: Benign + every attack category)
    # ---------------------------------------------------------------
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[label_col].astype(str))
    logger.info(
        f"Encoded {len(label_encoder.classes_)} classes: "
        f"{list(label_encoder.classes_)}"
    )

    X = df[feature_columns].astype(np.float64)

    # ---------------------------------------------------------------
    # Train/test split BEFORE scaling to avoid data leakage
    # ---------------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )

    # Keep a raw (unscaled) copy of X_test for human-readable SHAP explanations
    X_test_raw = X_test.copy().reset_index(drop=True)

    # Assign a persistent, human-referenceable ID to every test-set row so a
    # specific flow/"user" can be looked up later (e.g. via predict_user.py)
    # without needing to know its positional index. IDs are stable for the
    # lifetime of this preprocessed split (i.e. as long as
    # models/preprocessed_split.pkl isn't regenerated).
    X_test_raw.insert(
        0, "User_ID", [f"USER-{i:05d}" for i in range(len(X_test_raw))]
    )

    # ---------------------------------------------------------------
    # Feature scaling (fit on train only, applied to both splits)
    # ---------------------------------------------------------------
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    logger.info(f"Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    # Persist artifacts needed for inference / explanation reproducibility
    save_joblib(label_encoder, config.LABEL_ENCODER_PATH)
    save_joblib(scaler, config.SCALER_PATH)
    save_json(feature_columns, config.FEATURE_LIST_PATH)

    return PreprocessedData(
        X_train=X_train_scaled,
        X_test=X_test_scaled,
        y_train=y_train,
        y_test=y_test,
        feature_columns=feature_columns,
        label_encoder=label_encoder,
        scaler=scaler,
        X_test_raw=X_test_raw,
    )


if __name__ == "__main__":
    from src.data_loader import load_and_merge_csvs

    raw_df = load_and_merge_csvs()
    data = preprocess(raw_df)
    print("X_train:", data.X_train.shape)
    print("X_test:", data.X_test.shape)
    print("Classes:", data.label_encoder.classes_)