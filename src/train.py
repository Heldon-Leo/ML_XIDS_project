"""
train.py
========
Trains a multi-class XGBoost classifier on the preprocessed
CSE-CIC-IDS2018 features and persists the fitted model to disk.
"""

import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import get_logger, timeit, save_joblib

logger = get_logger(__name__)


@timeit
def train_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """
    Trains an XGBClassifier configured for multi-class classification.
    `num_class` is inferred automatically by XGBoost from the labels,
    but we set the objective explicitly for clarity and reproducibility.
    """
    n_classes = len(np.unique(y_train))
    logger.info(f"Training XGBoost multi-class classifier on {n_classes} classes ...")

    params = dict(config.XGB_PARAMS)
    params["objective"] = "multi:softprob"
    params["num_class"] = n_classes

    model = XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=True)

    train_preds = model.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    logger.info(f"Training accuracy: {train_acc:.4f}")

    save_joblib(model, config.MODEL_PATH)
    return model


if __name__ == "__main__":
    from src.data_loader import load_and_merge_csvs
    from src.preprocessing import preprocess

    raw_df = load_and_merge_csvs()
    data = preprocess(raw_df)
    trained_model = train_model(data.X_train, data.y_train)
    print("Model trained and saved to:", config.MODEL_PATH)
