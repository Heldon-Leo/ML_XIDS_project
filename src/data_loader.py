"""
data_loader.py
==============
Responsible for discovering all CSE-CIC-IDS2018 CSV files inside the
`data/` directory and merging them into a single pandas DataFrame,
while being memory-conscious (chunked reads + dtype downcasting).

The CSE-CIC-IDS2018 dataset (as redistributed on Mendeley) ships as
several multi-gigabyte CSVs, one per capture day/scenario. Each file
shares (mostly) the same 80+ CICFlowMeter-derived feature columns plus
a trailing `Label` column.
"""

import os
import glob
from typing import List, Optional

import pandas as pd

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import get_logger, timeit, reduce_memory_usage, clean_column_names

logger = get_logger(__name__)


def discover_csv_files(data_dir: str = config.DATA_DIR) -> List[str]:
    """
    Finds all CSV files inside `data_dir` (non-recursive by default, but
    falls back to a recursive search if nothing is found at the top level,
    since some CSE-CIC-IDS2018 mirrors nest files under sub-folders).
    """
    pattern = os.path.join(data_dir, config.CSV_GLOB_PATTERN)
    files = sorted(glob.glob(pattern))

    if not files:
        # Fall back to recursive search (e.g. data/Friday-.../file.csv)
        recursive_pattern = os.path.join(data_dir, "**", config.CSV_GLOB_PATTERN)
        files = sorted(glob.glob(recursive_pattern, recursive=True))

    if not files:
        raise FileNotFoundError(
            f"No CSV files found under '{data_dir}'. "
            f"Please download the CSE-CIC-IDS2018 dataset and place the "
            f"CSV files there (see README.md for instructions)."
        )

    logger.info(f"Discovered {len(files)} CSV file(s):")
    for f in files:
        size_mb = os.path.getsize(f) / (1024 ** 2)
        logger.info(f"  - {os.path.basename(f)} ({size_mb:.1f} MB)")

    return files


def _read_single_csv(filepath: str) -> pd.DataFrame:
    """
    Reads one CSV file, optionally in chunks to bound peak memory usage,
    downcasting numeric dtypes as it goes.
    """
    logger.info(f"Reading: {os.path.basename(filepath)}")

    if config.CSV_CHUNKSIZE:
        chunks = []
        for chunk in pd.read_csv(
            filepath,
            chunksize=config.CSV_CHUNKSIZE,
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="skip",
        ):
            chunk = clean_column_names(chunk)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
    else:
        df = pd.read_csv(filepath, low_memory=False, encoding="utf-8", on_bad_lines="skip")
        df = clean_column_names(df)

    if config.MAX_ROWS_PER_FILE is not None and len(df) > config.MAX_ROWS_PER_FILE:
        df = df.sample(
            n=config.MAX_ROWS_PER_FILE, random_state=config.RANDOM_STATE
        ).reset_index(drop=True)
        logger.info(
            f"  Sub-sampled {os.path.basename(filepath)} down to "
            f"{config.MAX_ROWS_PER_FILE} rows"
        )

    logger.info(f"  Loaded shape: {df.shape}")
    return df

@timeit
def load_and_merge_csvs(data_dir: Optional[str] = None) -> pd.DataFrame:
    """
    Main entry point: discovers all CSVs, reads them, aligns their
    columns, concatenates them, and optionally downsamples the
    combined result while preserving the Label column.
    """
    data_dir = data_dir or config.DATA_DIR
    files = discover_csv_files(data_dir)

    frames = []

    for filepath in files:
        try:
            df = _read_single_csv(filepath)
            frames.append(df)
        except Exception as exc:
            logger.warning(
                f"Skipping '{filepath}' due to read error: {exc}"
            )

    if not frames:
        raise RuntimeError("No CSV files could be read successfully.")

    logger.info("Merging all files into a single DataFrame ...")

    merged = pd.concat(
        frames,
        ignore_index=True,
        sort=False
    )

    del frames

    merged = clean_column_names(merged)

    logger.info(
        f"Merged dataset shape (before cleanup): {merged.shape}"
    )

    # Make sure Label exists BEFORE any sampling
    if config.LABEL_COLUMN not in merged.columns:
        raise KeyError(
            f"Expected label column '{config.LABEL_COLUMN}' not found "
            f"in merged dataset."
        )

    # ---------------------------------------------------------------
    # STRATIFIED DOWNSAMPLING
    # ---------------------------------------------------------------
    if (
        config.MAX_TOTAL_ROWS is not None
        and len(merged) > config.MAX_TOTAL_ROWS
    ):
        logger.info(
            f"Down-sampling merged dataset from {len(merged)} "
            f"to {config.MAX_TOTAL_ROWS} rows "
            f"(stratified by label) ..."
        )

        label_col = config.LABEL_COLUMN

        # Count samples in each class
        class_counts = merged[label_col].value_counts()

        # Calculate proportional number of samples
        target_counts = (
            class_counts
            / len(merged)
            * config.MAX_TOTAL_ROWS
        ).round().astype(int)

        # Keep at least one sample from every class
        target_counts = target_counts.clip(lower=1)

        sampled_parts = []

        for label, n_samples in target_counts.items():

            class_df = merged[
                merged[label_col] == label
            ]

            n_samples = min(
                int(n_samples),
                len(class_df)
            )

            sampled = class_df.sample(
                n=n_samples,
                random_state=config.RANDOM_STATE
            )

            sampled_parts.append(sampled)

        # Combine sampled classes
        merged = pd.concat(
            sampled_parts,
            ignore_index=True
        )

        # Shuffle final dataset
        merged = merged.sample(
            frac=1,
            random_state=config.RANDOM_STATE
        ).reset_index(drop=True)

        # Ensure maximum row limit
        if len(merged) > config.MAX_TOTAL_ROWS:
            merged = merged.sample(
                n=config.MAX_TOTAL_ROWS,
                random_state=config.RANDOM_STATE
            ).reset_index(drop=True)

        logger.info(
            f"Dataset shape after down-sampling: {merged.shape}"
        )

    # ---------------------------------------------------------------
    # VERIFY LABEL COLUMN
    # ---------------------------------------------------------------
    logger.info(
        f"Label column preserved: "
        f"{config.LABEL_COLUMN in merged.columns}"
    )

    if config.LABEL_COLUMN not in merged.columns:
        raise KeyError(
            f"Label column '{config.LABEL_COLUMN}' was lost "
            f"during dataset processing."
        )

    # ---------------------------------------------------------------
    # REDUCE MEMORY
    # ---------------------------------------------------------------
    merged = reduce_memory_usage(merged)

    # Final verification
    logger.info(
        f"Final dataset shape: {merged.shape}"
    )

    logger.info(
        f"Final columns: {len(merged.columns)}"
    )

    return merged


if __name__ == "__main__":
    df = load_and_merge_csvs()
    print(df.head())
    print(df[config.LABEL_COLUMN].value_counts())
