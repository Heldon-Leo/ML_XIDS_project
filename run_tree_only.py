"""
run_tree_only.py
=================
Standalone tree visualization for the Benign-vs-attack decision tree.

Loads the already-trained model, feature list, and label encoder from
disk, skips data loading/preprocessing/training entirely.

Run from the project root with:
    python run_tree_only.py

Requires that you've already run the full pipeline at least once, so that
models/xgb_model.pkl, models/feature_columns.json, and
models/label_encoder.pkl exist.
"""

import json
import pickle

import joblib

import config
from src.visualize_tree import visualize_tree
from src.utils import get_logger

logger = get_logger("run_tree_only")

# Which class's tree to visualize. Change to any class name that exists
# in your label encoder (e.g. "DDoS attack-HOIC", "DoS attacks-Hulk", ...)
# to see the tree for that attack type instead.
TARGET_CLASS = "Benign"


def main():
    logger.info("Loading saved model, feature columns, and label encoder ...")

    with open(config.MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(config.FEATURE_LIST_PATH) as f:
        feature_columns = json.load(f)

    with open(config.LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = joblib.load(f)

    logger.info(f"Loaded model from {config.MODEL_PATH}")
    logger.info(f"Classes available: {list(label_encoder.classes_)}")

    tree_plot_path = visualize_tree(
        model,
        feature_columns,
        label_encoder=label_encoder,
        class_name=TARGET_CLASS,
    )
    logger.info(f"Done. Tree plot saved to: {tree_plot_path}")


if __name__ == "__main__":
    main()