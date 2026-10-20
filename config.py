"""
config.py
=========
Central configuration file for the ML-XIDS project.

All paths, hyperparameters, and constants used across the pipeline
are defined here so that every module stays in sync and the project
remains easy to reconfigure (e.g. when moving from a laptop to a
GPU server, or when swapping dataset locations).
"""

import os

# ---------------------------------------------------------------------------
# BASE PROJECT PATHS
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")          # Raw CSE-CIC-IDS2018 CSVs go here
MODELS_DIR = os.path.join(BASE_DIR, "models")       # Trained model artifacts
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")     # Reports, plots, explanations

# Ensure output directories exist even on a fresh clone
for _dir in (DATA_DIR, MODELS_DIR, OUTPUTS_DIR):
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
# Glob pattern used to discover all CSE-CIC-IDS2018 CSV files inside DATA_DIR
CSV_GLOB_PATTERN = "*.csv"

# CSE-CIC-IDS2018 ships with a "Label" column that holds the attack name
# (e.g. Benign, DDoS-HOIC, DoS-Hulk, Bot, Infiltration, ...).
LABEL_COLUMN = "Label"

# Chunk size (rows) used when reading very large CSV files to keep memory
# consumption bounded. Set to None to disable chunked reading.
CSV_CHUNKSIZE = 200_000

# Cap on the number of rows kept per CSV after loading (None = no cap).
# Useful for prototyping on limited hardware; set to None for full-scale runs.
MAX_ROWS_PER_FILE = None

# Optional: cap total merged dataset size via stratified-ish random sampling
# (None = keep everything). This helps SHAP stay tractable on modest machines.
MAX_TOTAL_ROWS = 500_000

# ---------------------------------------------------------------------------
# PREPROCESSING
# ---------------------------------------------------------------------------
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Columns that are identifiers/metadata rather than real traffic features.
# These are dropped if present (CSE-CIC-IDS2018 sometimes includes them
# depending on the CICFlowMeter export version).
DROP_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Source IP",
    "Src Port",
    "Source Port",
    "Dst IP",
    "Destination IP",
    "Timestamp",
]

# ---------------------------------------------------------------------------
# MODEL TRAINING (XGBoost)
# ---------------------------------------------------------------------------
XGB_PARAMS = {
    "n_estimators": 300,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",       # fast, memory-efficient histogram method
    "eval_metric": "mlogloss",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

MODEL_PATH = os.path.join(MODELS_DIR, "xgb_model.pkl")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_LIST_PATH = os.path.join(MODELS_DIR, "feature_columns.json")

# Cached bundle of the full preprocessed train/test split. When this file AND
# MODEL_PATH both exist, main.py skips CSV loading, preprocessing, and
# training and jumps straight to evaluation/SHAP (use --force to redo all).
SPLIT_DATA_CACHE_PATH = os.path.join(MODELS_DIR, "preprocessed_split.pkl")

# Per-feature training statistics (min / max / median / p01 / p99 in RAW,
# unscaled units). Used by the live monitor to impute missing values exactly
# like training did and to detect unit/definition drift. Built automatically
# from the cached split the first time it is needed.
FEATURE_STATS_PATH = os.path.join(MODELS_DIR, "feature_stats.json")

# ---------------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------------
EVAL_REPORT_PATH = os.path.join(OUTPUTS_DIR, "evaluation_report.txt")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUTS_DIR, "confusion_matrix.png")

# ---------------------------------------------------------------------------
# EXPLAINABLE AI (SHAP)
# ---------------------------------------------------------------------------
# Number of test-set rows to explain locally (SHAP on the full test set can
# be slow for large multi-class datasets). Set to None to explain all rows.
SHAP_SAMPLE_SIZE = 2000

# Number of background rows used for TreeExplainer speed-ups (not strictly
# required for tree_path_dependent SHAP, kept for API compatibility).
SHAP_BACKGROUND_SIZE = 100

TOP_N_FEATURES = 3  # top contributing features per local explanation

SHAP_SUMMARY_PLOT_PATH = os.path.join(OUTPUTS_DIR, "shap_summary.png")
BENIGN_VS_ATTACK_PLOT_PATH = os.path.join(OUTPUTS_DIR, "benign_vs_attack_explanation.png")
BEHAVIOR_TREE_PLOT_PATH = os.path.join(OUTPUTS_DIR, "behavior_tree_explanation.png")

# How many top features to show per behavior category in the tree plot
BEHAVIOR_TREE_TOP_PER_CATEGORY = 3
SHAP_FORCE_PLOT_PATH = os.path.join(OUTPUTS_DIR, "shap_force_plot.html")
FEATURE_IMPORTANCE_CSV_PATH = os.path.join(OUTPUTS_DIR, "feature_importance.csv")
PREDICTIONS_EXPLANATIONS_CSV_PATH = os.path.join(
    OUTPUTS_DIR, "predictions_with_explanations.csv"
)

# Index of the sample used to render the single illustrative SHAP force plot
FORCE_PLOT_SAMPLE_INDEX = 0

# ---------------------------------------------------------------------------
# TREE VISUALIZATION
# ---------------------------------------------------------------------------
# For multi-class XGBoost, tree_idx = round_idx * num_class + class_idx.
TREE_VIS_INDEX = 0
TREE_PLOT_PATH = os.path.join(OUTPUTS_DIR, "xgb_tree.png")

# ---------------------------------------------------------------------------
# INTEGRATION-READY OUTPUTS (offline batch run: knowledge-graph triples + alerts)
# ---------------------------------------------------------------------------
TRIPLES_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "threat_graph_triples.json")
ALERTS_OUTPUT_PATH = os.path.join(OUTPUTS_DIR, "alerts.json")

# ---------------------------------------------------------------------------
# LIVE MONITORING (CICFlowMeter CSV -> XGBoost -> TreeSHAP -> detection events)
# ---------------------------------------------------------------------------
# The capture tool (CICFlowMeter) appends completed flows to this CSV.
CAPTURE_DIR = os.path.join(BASE_DIR, "capture")
LIVE_CAPTURE_FILE = os.path.join(CAPTURE_DIR, "live_flows.csv")

LIVE_OUTPUT_DIR = os.path.join(OUTPUTS_DIR, "live")
LIVE_EVENTS_JSONL = os.path.join(LIVE_OUTPUT_DIR, "events.jsonl")      # every processed flow
LIVE_ALERTS_JSONL = os.path.join(LIVE_OUTPUT_DIR, "alerts.jsonl")      # ALERT + REVIEW only
LIVE_ALERTS_CSV = os.path.join(LIVE_OUTPUT_DIR, "alerts.csv")          # flat view of the above
LIVE_TRIPLES_JSONL = os.path.join(LIVE_OUTPUT_DIR, "triples.jsonl")    # knowledge-graph triples
LIVE_REJECTED_JSONL = os.path.join(LIVE_OUTPUT_DIR, "rejected.jsonl")  # rows that failed validation
LIVE_STATS_JSON = os.path.join(LIVE_OUTPUT_DIR, "live_stats.json")     # counters / latency for the dashboard
SCHEMA_REPORT_PATH = os.path.join(LIVE_OUTPUT_DIR, "schema_report.json")

for _dir in (CAPTURE_DIR, LIVE_OUTPUT_DIR):
    os.makedirs(_dir, exist_ok=True)

LIVE_POLL_SECONDS = 2          # how often the monitor looks for new flow rows
LIVE_BATCH_MAX_ROWS = 500      # max rows inferred + explained per cycle

# A malicious prediction with confidence >= this is an ALERT; below it, REVIEW.
# This is a configurable starting point, NOT a validated security threshold.
LIVE_ALERT_THRESHOLD = 0.85

LIVE_SHAP_TOP_K = 5            # SHAP evidence items stored per event
LIVE_EXPLAIN_BENIGN = False    # SHAP for benign flows too (slower, bigger files)

# A flow with more than this fraction of model features missing/non-numeric is
# rejected instead of imputed.
LIVE_MAX_MISSING_FRACTION = 0.10

# Placeholder severity policy derived from model confidence (tune with your SOC).
LIVE_SEVERITY_HIGH = 0.95

# ---------------------------------------------------------------------------
# DETECTION-EVENT CONTRACT (shared with the ontology / graph / dashboard modules)
# ---------------------------------------------------------------------------
EVENT_SCHEMA_VERSION = "1.0"
EVENT_SCHEMA_PATH = os.path.join(BASE_DIR, "schemas", "detection_event.schema.json")
