"""
src/live_inference.py
=====================
LiveDetector: raw CICFlowMeter rows -> detection events.

Pipeline per batch
  1. resolve columns (schema_validator), coerce to numeric, inf -> NaN
  2. impute NaN with TRAINING medians (same rule as training); reject rows with
     too many missing features instead of guessing
  3. scale with the saved MinMaxScaler, predict_proba with XGBoost
  4. TreeSHAP for non-benign rows (native XGBoost pred_contribs; falls back to `shap`)
  5. build events (src/events.py), write JSONL / CSV / triples / stats
"""
import csv
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import config
from src import events as ev
from src.feature_stats import load_or_build_feature_stats
from src.schema_validator import (check_artifacts, check_values, coerce_feature_frame,
                                  resolve_columns, SchemaReport)
from src.utils import get_logger, load_joblib, load_json

logger = get_logger("live")

ALERT_CSV_COLUMNS = ["timestamp_utc", "event_id", "status", "severity", "predicted_class", "confidence",
                     "src_ip", "dst_ip", "src_port", "dst_port", "protocol", "mitre_id", "top_evidence"]


class LiveDetector:
    def __init__(self, explain_benign: Optional[bool] = None):
        ok, errs, warns = check_artifacts()
        for w in warns:
            logger.info(w)
        if not ok:
            raise RuntimeError("Model artifacts inconsistent: " + "; ".join(errs))

        self.features: List[str] = load_json(config.FEATURE_LIST_PATH)
        self.model = load_joblib(config.MODEL_PATH)
        self.scaler = load_joblib(config.SCALER_PATH)
        self.encoder = load_joblib(config.LABEL_ENCODER_PATH)
        self.classes = [str(c) for c in self.encoder.classes_]
        self.stats = load_or_build_feature_stats(self.features)
        self.explain_benign = config.LIVE_EXPLAIN_BENIGN if explain_benign is None else explain_benign
        self.benign_idx = next((i for i, c in enumerate(self.classes) if c.lower() == "benign"), None)
        self.resolution = None
        self.row_counter = 0
        self.seen: set = set()
        self.counters = {"processed": 0, "alerts": 0, "reviews": 0, "benign": 0,
                         "rejected": 0, "duplicates": 0, "invalid_events": 0}
        self.by_class: Dict[str, int] = {}
        self.latency_ms_last = 0.0
        self.started = time.time()
        self.explain_method = None
        self._load_seen()
        os.makedirs(config.LIVE_OUTPUT_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    def _load_seen(self):
        if os.path.exists(config.LIVE_EVENTS_JSONL):
            with open(config.LIVE_EVENTS_JSONL, encoding="utf-8") as f:
                for line in f:
                    try:
                        e = json.loads(line)
                    except ValueError:
                        continue
                    self.seen.add(e["event_id"])
                    self.row_counter += 1
                    self._count(e)
            logger.info(f"Resumed: {len(self.seen)} previous events found (will not be duplicated).")

    def _count(self, e):
        self.counters["processed"] += 1
        key = {"ALERT": "alerts", "REVIEW": "reviews", "BENIGN": "benign"}[e["status"]]
        self.counters[key] += 1
        self.by_class[e["predicted_class"]] = self.by_class.get(e["predicted_class"], 0) + 1

    # ------------------------------------------------------------------
    def _shap(self, X_scaled: pd.DataFrame) -> np.ndarray:
        """Returns contributions shaped (n, n_classes, n_features) (bias dropped)."""
        try:
            import xgboost as xgb
            booster = self.model.get_booster()
            contrib = booster.predict(xgb.DMatrix(X_scaled, feature_names=list(X_scaled.columns)),
                                      pred_contribs=True)
            self.explain_method = "TreeSHAP (XGBoost pred_contribs, log-odds margin)"
            return np.asarray(contrib)[:, :, :-1]
        except Exception as exc:
            logger.warning(f"Native pred_contribs failed ({exc}); falling back to shap.TreeExplainer.")
            import shap
            sv = shap.TreeExplainer(self.model).shap_values(X_scaled)
            self.explain_method = "TreeSHAP (shap library, log-odds margin)"
            if isinstance(sv, list):
                sv = np.stack(sv, axis=1)
            else:
                sv = np.asarray(sv)
                if sv.shape[1] == len(self.features):   # (n, f, C) -> (n, C, f)
                    sv = np.transpose(sv, (0, 2, 1))
            return sv

    # ------------------------------------------------------------------
    def process(self, raw: pd.DataFrame) -> List[dict]:
        if raw is None or raw.empty:
            return []
        t0 = time.time()
        raw = raw.copy()
        raw.columns = [str(c).strip() for c in raw.columns]
        res = resolve_columns(raw.columns, self.features)
        if res.missing:
            self._reject(raw, f"missing model features: {res.missing[:5]}")
            return []
        if self.resolution is None:
            self.resolution = res
            if res.aliased:
                logger.warning(f"{len(res.aliased)} features matched via alias; verify definitions.")

        X = coerce_feature_frame(raw, res, self.features)
        nan_mask = X.isna()
        miss_frac = nan_mask.mean(axis=1)
        bad = miss_frac > config.LIVE_MAX_MISSING_FRACTION
        if bad.any():
            self._reject(raw[bad], f"more than {config.LIVE_MAX_MISSING_FRACTION:.0%} of features missing/invalid")
        if self.stats is None:
            bad = bad | nan_mask.any(axis=1)   # no medians available -> cannot impute safely
        keep = ~bad
        if not keep.any():
            return []
        raw, X, nan_mask = raw[keep], X[keep], nan_mask[keep]

        if self.stats is not None:
            med = pd.Series({f: self.stats[f]["median"] for f in self.features})
            X = X.fillna(med)
        imputed = nan_mask.sum(axis=1).to_numpy()

        lo = np.array([self.stats[f]["min"] for f in self.features]) if self.stats else None
        hi = np.array([self.stats[f]["max"] for f in self.features]) if self.stats else None
        oor = ((X.to_numpy() < lo) | (X.to_numpy() > hi)).sum(axis=1) if self.stats else np.zeros(len(X), int)

        Xs = pd.DataFrame(self.scaler.transform(X.to_numpy()), columns=self.features, index=X.index)
        proba = self.model.predict_proba(Xs)
        pred = proba.argmax(axis=1)

        explain_rows = np.ones(len(pred), bool) if self.explain_benign else (pred != self.benign_idx)
        contrib = self._shap(Xs[explain_rows]) if explain_rows.any() else None
        contrib_pos = {int(i): k for k, i in enumerate(np.where(explain_rows)[0])}

        events: List[dict] = []
        for i in range(len(pred)):
            label = self.classes[pred[i]]
            top_idx = np.argsort(proba[i])[::-1][:3]
            probs_top = {self.classes[j]: proba[i][j] for j in top_idx}
            evidence = []
            if i in contrib_pos:
                sv = contrib[contrib_pos[i], pred[i], :]
                for j in np.argsort(-np.abs(sv))[: config.LIVE_SHAP_TOP_K]:
                    evidence.append({"feature": self.features[j],
                                     "value": ev._num(X.iloc[i, j]),
                                     "shap_value": round(float(sv[j]), 6),
                                     "effect": "supports" if sv[j] > 0 else "opposes"})
            meta = {k: raw.iloc[i][col] for k, col in res.metadata.items()}
            for k, feat in (("dst_port", "Dst Port"), ("protocol", "Protocol")):
                if feat in res.mapping:
                    meta[k] = raw.iloc[i][res.mapping[feat]]
            self.row_counter += 1
            e = ev.build_event(meta=meta, label=label, confidence=proba[i][pred[i]], probs_top=probs_top,
                               evidence=evidence, n_classes=len(self.classes), imputed=imputed[i],
                               out_of_range=oor[i], explanation_method=self.explain_method if evidence else None,
                               row_key=self.row_counter if not meta.get("flow_id") and not meta.get("timestamp") else raw.index[i])
            if e["event_id"] in self.seen:
                self.counters["duplicates"] += 1
                continue
            errs = ev.validate_event(e)
            if errs:
                self.counters["invalid_events"] += 1
                logger.error(f"Event failed schema validation: {errs[:2]}")
                continue
            self.seen.add(e["event_id"])
            events.append(e)

        self._write(events)
        self.latency_ms_last = (time.time() - t0) * 1000
        self.write_stats(len(events))
        return events

    # ------------------------------------------------------------------
    def _reject(self, rows: pd.DataFrame, reason: str):
        self.counters["rejected"] += len(rows)
        with open(config.LIVE_REJECTED_JSONL, "a", encoding="utf-8") as f:
            for _, r in rows.head(50).iterrows():
                f.write(json.dumps({"reason": reason, "row_preview": {k: str(v) for k, v in list(r.items())[:6]}}) + "\n")
        logger.warning(f"Rejected {len(rows)} row(s): {reason}")

    def _write(self, events: List[dict]):
        if not events:
            return
        new_csv = not os.path.exists(config.LIVE_ALERTS_CSV)
        with open(config.LIVE_EVENTS_JSONL, "a", encoding="utf-8") as fe, \
             open(config.LIVE_ALERTS_JSONL, "a", encoding="utf-8") as fa, \
             open(config.LIVE_TRIPLES_JSONL, "a", encoding="utf-8") as ft, \
             open(config.LIVE_ALERTS_CSV, "a", newline="", encoding="utf-8") as fc:
            w = csv.DictWriter(fc, fieldnames=ALERT_CSV_COLUMNS)
            if new_csv:
                w.writeheader()
            for e in events:
                self._count(e)
                fe.write(json.dumps(e) + "\n")
                if e["status"] != "BENIGN":
                    fa.write(json.dumps(e) + "\n")
                    w.writerow({"timestamp_utc": e["timestamp_utc"], "event_id": e["event_id"],
                                "status": e["status"], "severity": e["severity"],
                                "predicted_class": e["predicted_class"], "confidence": e["confidence"],
                                "src_ip": e["src_ip"], "dst_ip": e["dst_ip"], "src_port": e["src_port"],
                                "dst_port": e["dst_port"], "protocol": e["protocol"],
                                "mitre_id": (e["mitre"] or {}).get("technique_id"),
                                "top_evidence": "; ".join(f"{x['feature']}({x['effect'][:3]})" for x in e["shap_evidence"][:3])})
                    for t in ev.event_to_triples(e):
                        ft.write(json.dumps(t) + "\n")

    def write_stats(self, last_batch: int = 0):
        stats = dict(self.counters)
        stats.update({"by_class": self.by_class, "last_batch_events": last_batch,
                      "last_batch_latency_ms": round(self.latency_ms_last, 1),
                      "uptime_s": round(time.time() - self.started, 1),
                      "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        tmp = config.LIVE_STATS_JSON + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp, config.LIVE_STATS_JSON)
