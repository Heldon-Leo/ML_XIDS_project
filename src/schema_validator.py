"""
src/schema_validator.py
========================
Feature-schema validation between the TRAINED model and incoming flow data
(e.g. a CSV written by CICFlowMeter).

A model trained on CSE-CIC-IDS2018 only makes sense if the live flow table
supplies the same features, in the same units, with the same definitions.
Matching column *names* is necessary but not sufficient, so this module
performs three levels of checking:

1. ARTIFACT CONSISTENCY  - model / scaler / label encoder / feature list all
   come from the same training run (feature counts and class counts agree).
2. COLUMN RESOLUTION     - every model feature is located in the incoming
   file: exact name, normalized name (case/punctuation), or a documented
   alias (e.g. CIC-IDS2017-style "Total Fwd Packets" -> "Tot Fwd Pkts").
   Alias matches are reported as warnings because the *definition* still has
   to be verified. Missing features are an error - they are never zero-filled.
3. VALUE SANITY          - incoming values are compared with the training
   range of each feature. If most rows fall outside the range for many
   features, the extractor probably uses different units or definitions.

Nothing here modifies data; it only reports.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import config
from src.utils import get_logger, load_joblib

logger = get_logger("schema_validator")


# ---------------------------------------------------------------------------
# NAME NORMALIZATION + ALIASES
# ---------------------------------------------------------------------------
def normalize_name(name) -> str:
    """'Flow Byts/s' -> 'flow_byts_s', 'flow_byts_s' -> 'flow_byts_s'."""
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


# model feature name -> alternative normalized names used by other extractor
# versions (mainly the CIC-IDS2017 / Java CICFlowMeter header spelling).
_FEATURE_ALIASES: Dict[str, List[str]] = {
    "Dst Port": ["destination_port"],
    "Tot Fwd Pkts": ["total_fwd_packets"],
    "Tot Bwd Pkts": ["total_backward_packets", "total_bwd_packets"],
    "TotLen Fwd Pkts": ["total_length_of_fwd_packets"],
    "TotLen Bwd Pkts": ["total_length_of_bwd_packets"],
    "Fwd Pkt Len Max": ["fwd_packet_length_max"],
    "Fwd Pkt Len Min": ["fwd_packet_length_min"],
    "Fwd Pkt Len Mean": ["fwd_packet_length_mean"],
    "Fwd Pkt Len Std": ["fwd_packet_length_std"],
    "Bwd Pkt Len Max": ["bwd_packet_length_max"],
    "Bwd Pkt Len Min": ["bwd_packet_length_min"],
    "Bwd Pkt Len Mean": ["bwd_packet_length_mean"],
    "Bwd Pkt Len Std": ["bwd_packet_length_std"],
    "Flow Byts/s": ["flow_bytes_s"],
    "Flow Pkts/s": ["flow_packets_s"],
    "Fwd IAT Tot": ["fwd_iat_total"],
    "Bwd IAT Tot": ["bwd_iat_total"],
    "Fwd Header Len": ["fwd_header_length"],
    "Bwd Header Len": ["bwd_header_length"],
    "Fwd Pkts/s": ["fwd_packets_s"],
    "Bwd Pkts/s": ["bwd_packets_s"],
    "Pkt Len Min": ["min_packet_length"],
    "Pkt Len Max": ["max_packet_length"],
    "Pkt Len Mean": ["packet_length_mean"],
    "Pkt Len Std": ["packet_length_std"],
    "Pkt Len Var": ["packet_length_variance"],
    "FIN Flag Cnt": ["fin_flag_count"],
    "SYN Flag Cnt": ["syn_flag_count"],
    "RST Flag Cnt": ["rst_flag_count"],
    "PSH Flag Cnt": ["psh_flag_count"],
    "ACK Flag Cnt": ["ack_flag_count"],
    "URG Flag Cnt": ["urg_flag_count"],
    "ECE Flag Cnt": ["ece_flag_count"],
    "Pkt Size Avg": ["average_packet_size"],
    "Fwd Seg Size Avg": ["avg_fwd_segment_size"],
    "Bwd Seg Size Avg": ["avg_bwd_segment_size"],
    "Subflow Fwd Pkts": ["subflow_fwd_packets"],
    "Subflow Fwd Byts": ["subflow_fwd_bytes"],
    "Subflow Bwd Pkts": ["subflow_bwd_packets"],
    "Subflow Bwd Byts": ["subflow_bwd_bytes"],
    "Init Fwd Win Byts": ["init_win_bytes_forward"],
    "Init Bwd Win Byts": ["init_win_bytes_backward"],
    "Fwd Act Data Pkts": ["act_data_pkt_fwd"],
    "Fwd Seg Size Min": ["min_seg_size_forward"],
}

# Capture metadata: kept for the alert, never fed to the model.
_METADATA_CANDIDATES: Dict[str, List[str]] = {
    "src_ip": ["src_ip", "source_ip"],
    "dst_ip": ["dst_ip", "destination_ip"],
    "src_port": ["src_port", "source_port"],
    "flow_id": ["flow_id"],
    "timestamp": ["timestamp"],
    "label": ["label"],  # ground truth, only present in replayed dataset rows
}

MIN_ROWS_FOR_VALUE_CHECKS = 20
OUT_OF_RANGE_WARN_PCT = 20.0     # per-feature warning
OUT_OF_RANGE_BAD_PCT = 50.0      # feature counts as "mostly outside training range"
OUT_OF_RANGE_BAD_FEATURE_FRACTION = 0.25   # error when this share of features is "bad"


# ---------------------------------------------------------------------------
# DATA CLASSES
# ---------------------------------------------------------------------------
@dataclass
class ColumnResolution:
    mapping: Dict[str, str] = field(default_factory=dict)   # model feature -> file column
    exact: List[str] = field(default_factory=list)
    normalized: List[str] = field(default_factory=list)
    aliased: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    extra: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)  # logical name -> file column


@dataclass
class SchemaReport:
    ok: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    n_rows: int = 0
    n_model_features: int = 0
    resolution: Optional[ColumnResolution] = None
    problem_features: Dict[str, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        res = self.resolution
        return {
            "ok": self.ok,
            "errors": self.errors,
            "warnings": self.warnings,
            "n_rows_checked": self.n_rows,
            "n_model_features": self.n_model_features,
            "columns": None if res is None else {
                "exact": len(res.exact),
                "normalized": len(res.normalized),
                "aliased": {f: res.mapping[f] for f in res.aliased},
                "missing": res.missing,
                "extra": res.extra,
                "metadata": res.metadata,
            },
            "problem_features": self.problem_features,
        }


# ---------------------------------------------------------------------------
# 1. ARTIFACT CONSISTENCY
# ---------------------------------------------------------------------------
def check_artifacts() -> Tuple[bool, List[str], List[str]]:
    """
    Verifies model / scaler / encoder / feature list belong together.
    Returns (ok, errors, info_lines).
    """
    errors: List[str] = []
    info: List[str] = []

    for path in (config.MODEL_PATH, config.LABEL_ENCODER_PATH, config.SCALER_PATH, config.FEATURE_LIST_PATH):
        if not os.path.exists(path):
            errors.append(f"Missing artifact: {path}")
    if errors:
        return False, errors, info

    model = load_joblib(config.MODEL_PATH)
    encoder = load_joblib(config.LABEL_ENCODER_PATH)
    scaler = load_joblib(config.SCALER_PATH)
    with open(config.FEATURE_LIST_PATH, encoding="utf-8") as f:
        features = json.load(f)

    if not isinstance(features, list) or not features:
        errors.append("feature_columns.json must contain a non-empty list.")
        return False, errors, info
    if len(features) != len(set(features)):
        errors.append("feature_columns.json contains duplicate feature names.")

    n = len(features)
    info.append(f"Feature count (feature_columns.json): {n}")

    model_n = getattr(model, "n_features_in_", None)
    if model_n is not None and model_n != n:
        errors.append(f"Model expects {model_n} features but feature_columns.json lists {n}.")

    scaler_n = getattr(scaler, "n_features_in_", None)
    if scaler_n is not None and scaler_n != n:
        errors.append(f"Scaler was fit on {scaler_n} features but feature_columns.json lists {n}.")

    scaler_names = getattr(scaler, "feature_names_in_", None)
    if scaler_names is not None and list(scaler_names) != features:
        errors.append("Scaler feature names/order differ from feature_columns.json "
                      "(artifacts are from different training runs).")

    n_classes_model = getattr(model, "n_classes_", None)
    n_classes_enc = len(encoder.classes_)
    if n_classes_model is not None and n_classes_model != n_classes_enc:
        errors.append(f"Model has {n_classes_model} classes but the label encoder has {n_classes_enc}.")

    info.append(f"Classes ({n_classes_enc}): {list(encoder.classes_)}")
    return len(errors) == 0, errors, info


# ---------------------------------------------------------------------------
# 2. COLUMN RESOLUTION
# ---------------------------------------------------------------------------
def resolve_columns(columns, features: List[str]) -> ColumnResolution:
    """Locates every model feature (and capture metadata) among `columns`."""
    columns = [str(c).strip() for c in columns]
    by_norm: Dict[str, str] = {}
    for col in columns:
        by_norm.setdefault(normalize_name(col), col)

    res = ColumnResolution()
    used = set()

    for feature in features:
        if feature in columns:
            res.mapping[feature] = feature
            res.exact.append(feature)
        elif normalize_name(feature) in by_norm:
            res.mapping[feature] = by_norm[normalize_name(feature)]
            res.normalized.append(feature)
        else:
            found = None
            for alias in _FEATURE_ALIASES.get(feature, []):
                if alias in by_norm:
                    found = by_norm[alias]
                    break
            if found is not None:
                res.mapping[feature] = found
                res.aliased.append(feature)
            else:
                res.missing.append(feature)
        if feature in res.mapping:
            used.add(res.mapping[feature])

    for logical, candidates in _METADATA_CANDIDATES.items():
        for cand in candidates:
            if cand in by_norm:
                res.metadata[logical] = by_norm[cand]
                used.add(by_norm[cand])
                break

    res.extra = [c for c in columns if c not in used]
    return res


# ---------------------------------------------------------------------------
# 3. VALUE SANITY
# ---------------------------------------------------------------------------
def coerce_feature_frame(df: pd.DataFrame, resolution: ColumnResolution, features: List[str]) -> pd.DataFrame:
    """
    Returns a float DataFrame with exactly the model features, in model order.
    Non-numeric strings and +/-inf become NaN (the same cleaning training used).
    """
    out = pd.DataFrame(index=df.index)
    for feature in features:
        out[feature] = pd.to_numeric(df[resolution.mapping[feature]], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).astype(np.float64)


def check_values(X: pd.DataFrame, stats: Optional[dict], report: SchemaReport) -> None:
    """Adds value-sanity warnings/errors to `report` (training-range comparison)."""
    n = len(X)
    report.n_rows = n
    if n == 0:
        report.warnings.append("No data rows to check yet.")
        return

    nan_pct = X.isna().mean() * 100
    all_missing = [c for c in X.columns if nan_pct[c] == 100.0]
    if all_missing:
        report.warnings.append(
            f"{len(all_missing)} feature(s) are empty/non-numeric in every checked row: "
            f"{all_missing[:5]}{' ...' if len(all_missing) > 5 else ''}"
        )
    heavy_nan = [c for c in X.columns if 10.0 < nan_pct[c] < 100.0]
    if heavy_nan:
        report.warnings.append(
            f"{len(heavy_nan)} feature(s) have >10% missing/inf values: {heavy_nan[:5]}"
        )

    if stats is None:
        report.warnings.append("Training statistics unavailable: value-range drift check skipped.")
        return
    if n < MIN_ROWS_FOR_VALUE_CHECKS:
        report.warnings.append(
            f"Only {n} row(s) checked (< {MIN_ROWS_FOR_VALUE_CHECKS}); value-range drift check skipped."
        )
        return

    bad_features = []
    for feature in X.columns:
        s = stats.get(feature)
        if s is None:
            continue
        col = X[feature].dropna()
        if col.empty:
            continue
        span = max(s["max"] - s["min"], 1e-12)
        tol = 1e-9 * span
        oor = float(((col < s["min"] - tol) | (col > s["max"] + tol)).mean() * 100)
        if oor > OUT_OF_RANGE_WARN_PCT:
            report.problem_features[feature] = {
                "out_of_training_range_pct": round(oor, 1),
                "live_median": float(col.median()),
                "train_median": s["median"],
                "train_min": s["min"],
                "train_max": s["max"],
            }
        if oor > OUT_OF_RANGE_BAD_PCT:
            bad_features.append(feature)

    if report.problem_features:
        worst = sorted(report.problem_features.items(),
                       key=lambda kv: -kv[1]["out_of_training_range_pct"])[:5]
        report.warnings.append(
            f"{len(report.problem_features)} feature(s) have >{OUT_OF_RANGE_WARN_PCT:.0f}% of values outside "
            f"the training range. Worst: " + ", ".join(f"{k} ({v['out_of_training_range_pct']}%)" for k, v in worst)
        )

    if len(bad_features) / max(len(X.columns), 1) >= OUT_OF_RANGE_BAD_FEATURE_FRACTION:
        report.errors.append(
            f"{len(bad_features)}/{len(X.columns)} features are mostly outside the training range. "
            "This usually means different units or feature definitions between the capture tool "
            "and the training data (e.g. durations in seconds vs microseconds). "
            "Predictions would not be reliable."
        )


# ---------------------------------------------------------------------------
# PUBLIC ENTRY POINTS
# ---------------------------------------------------------------------------
def validate_dataframe(df: pd.DataFrame, features: List[str], stats: Optional[dict]) -> SchemaReport:
    """Full validation (columns + values) of a DataFrame of flow rows."""
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    report = SchemaReport(n_model_features=len(features))
    res = resolve_columns(df.columns, features)
    report.resolution = res

    if res.missing:
        report.errors.append(
            f"{len(res.missing)} required model feature(s) not found in the capture file: "
            f"{res.missing[:8]}{' ...' if len(res.missing) > 8 else ''}. "
            "Missing features are never zero-filled."
        )
    if res.aliased:
        report.warnings.append(
            f"{len(res.aliased)} feature(s) matched only through a name alias "
            f"(verify the definition is identical): {res.aliased[:6]}{' ...' if len(res.aliased) > 6 else ''}"
        )
    if "timestamp" not in res.metadata:
        report.warnings.append("No Timestamp column: events will carry only the detection time.")
    if "src_ip" not in res.metadata or "dst_ip" not in res.metadata:
        report.warnings.append("No Src/Dst IP columns: events will not contain IP addresses.")

    if not res.missing:
        X = coerce_feature_frame(df, res, features)
        check_values(X, stats, report)

    report.ok = not report.errors
    return report


def validate_capture_file(path: str, features: List[str], stats: Optional[dict], max_rows: int = 2000) -> SchemaReport:
    """Reads the header and up to `max_rows` rows of a capture CSV and validates them."""
    report = SchemaReport(n_model_features=len(features))
    if not os.path.exists(path):
        report.ok = False
        report.errors.append(f"Capture file not found: {path}")
        return report
    try:
        df = pd.read_csv(path, nrows=max_rows, low_memory=False, encoding="utf-8-sig", on_bad_lines="skip")
    except Exception as exc:  # unreadable / empty file
        report.ok = False
        report.errors.append(f"Could not read capture file: {exc}")
        return report
    # drop repeated header rows (seen when CSVs are concatenated)
    first = str(df.columns[0]).strip()
    df = df[df.iloc[:, 0].astype(str).str.strip() != first]
    return validate_dataframe(df, features, stats)


def format_report(report: SchemaReport) -> str:
    """Human-readable console summary."""
    lines = []
    res = report.resolution
    if res is not None:
        total = report.n_model_features
        lines.append(f"Model features required : {total}")
        lines.append(f"  exact name match      : {len(res.exact)}")
        lines.append(f"  normalized name match : {len(res.normalized)}")
        lines.append(f"  alias match           : {len(res.aliased)}")
        lines.append(f"  MISSING               : {len(res.missing)}")
        lines.append(f"Extra columns ignored   : {len(res.extra)}")
        if res.metadata:
            lines.append("Metadata columns        : " + ", ".join(f"{k}='{v}'" for k, v in res.metadata.items()))
    lines.append(f"Rows checked            : {report.n_rows}")
    for w in report.warnings:
        lines.append(f"[WARN]  {w}")
    for e in report.errors:
        lines.append(f"[ERROR] {e}")
    lines.append("RESULT: " + ("PASSED" if report.ok else "FAILED"))
    return "\n".join(lines)
