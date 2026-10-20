"""
evaluate.py
===========
Evaluates the trained XGBoost model on the held-out test set:
- Accuracy
- Precision / Recall / F1 (per-class + weighted averages)
- Confusion matrix (saved as PNG heatmap)
- Full text report saved to outputs/evaluation_report.txt
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend, safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)


@timeit
def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray, label_encoder) -> dict:
    """
    Runs full evaluation of `model` on (X_test, y_test), writes a text
    report and a confusion-matrix heatmap to the outputs/ directory, and
    returns a dict summary for programmatic use (e.g. in main.py).
    """
    class_names = list(label_encoder.classes_)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report_str = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0
    )
    report_dict = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0, output_dict=True
    )

    logger.info(f"Test Accuracy: {acc:.4f}")
    logger.info("\n" + report_str)

    # -----------------------------------------------------------------
    # Persist text report
    # -----------------------------------------------------------------
    with open(config.EVAL_REPORT_PATH, "w") as f:
        f.write("ML-XIDS Model Evaluation Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Overall Accuracy: {acc:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    logger.info(f"Saved evaluation report -> {config.EVAL_REPORT_PATH}")

    # -----------------------------------------------------------------
    # Confusion matrix heatmap
    # -----------------------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.7)))
    sns.heatmap(
        cm,
        annot=len(class_names) <= 15,  # avoid unreadable clutter for many classes
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix - ML-XIDS")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(config.CONFUSION_MATRIX_PATH, dpi=150)
    plt.close()
    logger.info(f"Saved confusion matrix plot -> {config.CONFUSION_MATRIX_PATH}")

    return {
        "accuracy": acc,
        "report": report_dict,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    from src.data_loader import load_and_merge_csvs
    from src.preprocessing import preprocess
    from src.train import train_model

    raw_df = load_and_merge_csvs()
    data = preprocess(raw_df)
    model = train_model(data.X_train, data.y_train)
    results = evaluate_model(model, data.X_test, data.y_test, data.label_encoder)
    print("Accuracy:", results["accuracy"])
