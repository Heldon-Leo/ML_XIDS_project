"""
main.py
=======
End-to-end ML-XIDS pipeline entry point.

Run with:
    python main.py                # uses cached model/data if available
    python main.py --force        # ignores cache, reruns everything from scratch

Pipeline stages:
    1. Load & merge all CSE-CIC-IDS2018 CSVs from data/
    2. Preprocess (clean, encode, scale, split)
    3. Train XGBoost multi-class model
    4. Evaluate (accuracy, precision/recall/F1, confusion matrix)
    5. Generate SHAP-based global + local explanations
    6. Save all outputs to outputs/ and the model to models/

Stages 1-3 are the expensive part (reading/merging large CSVs, scaling,
training). Once they've been run once, the trained model AND the
preprocessed train/test split are cached to disk under models/. On
subsequent runs, if both caches exist, main.py loads them directly and
jumps straight to evaluation + SHAP -- so re-running the pipeline to
regenerate outputs doesn't mean waiting through data loading and training
again every time. Pass --force to bypass the cache (e.g. after changing
files in data/, editing preprocessing logic, or tuning XGB_PARAMS).

This script intentionally stops after the Explainable AI stage --
no frontend/serving layer is included per project scope.
"""

import argparse
import os
import sys
import time

import config
from src.data_loader import load_and_merge_csvs
from src.preprocessing import preprocess, PreprocessedData
from src.train import train_model
from src.evaluate import evaluate_model
from src.explain import run_explainability_pipeline
from src.utils import get_logger, save_joblib, load_joblib

logger = get_logger("main")


def _cache_available() -> bool:
    """True if both the trained model and the preprocessed split are cached."""
    return os.path.exists(config.MODEL_PATH) and os.path.exists(
        config.SPLIT_DATA_CACHE_PATH
    )


def _load_from_cache():
    logger.info(
        f"Found cached model ({config.MODEL_PATH}) and cached preprocessed "
        f"split ({config.SPLIT_DATA_CACHE_PATH}) -- skipping data loading, "
        f"preprocessing, and training. Pass --force to redo them."
    )
    model = load_joblib(config.MODEL_PATH)
    data: PreprocessedData = load_joblib(config.SPLIT_DATA_CACHE_PATH)
    return model, data


def _run_from_scratch():
    # -----------------------------------------------------------------
    # STAGE 1: Data Loading
    # -----------------------------------------------------------------
    logger.info("\n[STAGE 1/5] Loading and merging CSV files ...")
    raw_df = load_and_merge_csvs(config.DATA_DIR)
    logger.info(f"Raw merged dataset shape: {raw_df.shape}")

    # -----------------------------------------------------------------
    # STAGE 2: Preprocessing
    # -----------------------------------------------------------------
    logger.info("\n[STAGE 2/5] Preprocessing data ...")
    data = preprocess(raw_df)
    del raw_df  # free memory; no longer needed

    # Cache the full split so the next run can skip stages 1-3 entirely.
    save_joblib(data, config.SPLIT_DATA_CACHE_PATH)

    # -----------------------------------------------------------------
    # STAGE 3: Model Training
    # -----------------------------------------------------------------
    logger.info("\n[STAGE 3/5] Training XGBoost model ...")
    model = train_model(data.X_train, data.y_train)

    return model, data


def parse_args():
    parser = argparse.ArgumentParser(description="ML-XIDS pipeline")
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Ignore any cached model/preprocessed data and rerun data "
            "loading, preprocessing, and training from scratch."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("ML-XIDS: Explainable Intrusion Detection System")
    logger.info("Dataset: CSE-CIC-IDS2018")
    logger.info("=" * 70)

    if not args.force and _cache_available():
        model, data = _load_from_cache()
    else:
        model, data = _run_from_scratch()

    # -----------------------------------------------------------------
    # STAGE 4: Evaluation
    # -----------------------------------------------------------------
    logger.info("\n[STAGE 4/5] Evaluating model ...")
    eval_results = evaluate_model(model, data.X_test, data.y_test, data.label_encoder)

    # -----------------------------------------------------------------
    # STAGE 5: Explainable AI (SHAP)
    # -----------------------------------------------------------------
    logger.info("\n[STAGE 5/5] Generating SHAP explanations ...")
    explain_results = run_explainability_pipeline(
        model,
        data.X_test,
        data.X_test_raw,
        data.feature_columns,
        data.label_encoder,
    )

    total_time = time.time() - pipeline_start

    # -----------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Total runtime: {total_time / 60:.2f} minutes")
    logger.info(f"Test Accuracy: {eval_results['accuracy']:.4f}")
    logger.info(f"Model saved to: {config.MODEL_PATH}")
    logger.info(f"Evaluation report: {config.EVAL_REPORT_PATH}")
    logger.info(f"Confusion matrix: {config.CONFUSION_MATRIX_PATH}")
    logger.info(f"SHAP summary plot: {config.SHAP_SUMMARY_PLOT_PATH}")
    logger.info(f"SHAP force plot: {config.SHAP_FORCE_PLOT_PATH}")
    logger.info(f"Feature importance CSV: {config.FEATURE_IMPORTANCE_CSV_PATH}")
    logger.info(
        f"Predictions with explanations: {config.PREDICTIONS_EXPLANATIONS_CSV_PATH}"
    )
    logger.info(f"Knowledge graph triples: {config.TRIPLES_OUTPUT_PATH}")
    logger.info(f"Unified alerts: {config.ALERTS_OUTPUT_PATH}")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception(f"Pipeline failed: {exc}")
        sys.exit(1)