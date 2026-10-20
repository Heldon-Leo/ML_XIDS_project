"""
src/events.py
=============
Builds / validates the standard detection event (schemas/detection_event.schema.json)
and converts events to knowledge-graph triples for the ontology module.
"""
import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import config
from src.mitre_mapping import get_mitre_info

_MODEL_VERSION_CACHE: Optional[str] = None


def model_version() -> str:
    """Short hash of the model file, so every event is traceable to a model build."""
    global _MODEL_VERSION_CACHE
    if _MODEL_VERSION_CACHE is None:
        h = hashlib.sha1()
        try:
            with open(config.MODEL_PATH, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            _MODEL_VERSION_CACHE = h.hexdigest()[:12]
        except OSError:
            _MODEL_VERSION_CACHE = "unknown"
    return _MODEL_VERSION_CACHE


def make_event_id(parts: List[Any]) -> str:
    """Deterministic id: the same flow always yields the same id (dedupe-safe)."""
    raw = "|".join("" if p is None else str(p) for p in parts)
    return "evt-" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and abs(f) != float("inf") else None


def _int(v):
    f = _num(v)
    return None if f is None else int(f)


def severity_for(status: str, confidence: float) -> str:
    if status == "ALERT":
        return "high" if confidence >= config.LIVE_SEVERITY_HIGH else "medium"
    if status == "REVIEW":
        return "medium"
    return "none"


def build_explanation(label: str, confidence: float, evidence: List[dict], method: str) -> str:
    if not evidence:
        return f"Classified as {label} ({confidence:.1%}). No per-feature explanation was computed."
    sup = [e for e in evidence if e["effect"] == "supports"][:3]
    opp = [e for e in evidence if e["effect"] == "opposes"][:2]
    txt = f"Classified as {label} ({confidence:.1%})."
    if sup:
        txt += " Pushing towards this class: " + ", ".join(f"{e['feature']}={e['value']:.6g}" for e in sup if e["value"] is not None) + "."
    if opp:
        txt += " Pushing against: " + ", ".join(f"{e['feature']}={e['value']:.6g}" for e in opp if e["value"] is not None) + "."
    return txt


def build_event(*, meta: Dict[str, Any], label: str, confidence: float, probs_top: Dict[str, float],
                evidence: List[dict], n_classes: int, imputed: int, out_of_range: int,
                explanation_method: Optional[str], row_key: Any) -> Dict[str, Any]:
    is_mal = label.strip().lower() != "benign"
    if not is_mal:
        status = "BENIGN"
    elif confidence >= config.LIVE_ALERT_THRESHOLD:
        status = "ALERT"
    else:
        status = "REVIEW"
    tid, tname = get_mitre_info(label)
    event = {
        "schema_version": config.EVENT_SCHEMA_VERSION,
        "event_id": make_event_id([meta.get("flow_id"), meta.get("src_ip"), meta.get("dst_ip"),
                                   meta.get("src_port"), meta.get("dst_port"),
                                   meta.get("timestamp"), row_key]),
        "source_module": "ML-XIDS",
        "evidence_type": "ML_NETWORK_FLOW_PREDICTION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "capture_timestamp": meta.get("timestamp"),
        "flow_id": meta.get("flow_id"),
        "src_ip": meta.get("src_ip"),
        "dst_ip": meta.get("dst_ip"),
        "src_port": _int(meta.get("src_port")),
        "dst_port": _int(meta.get("dst_port")),
        "protocol": _int(meta.get("protocol")),
        "predicted_class": label,
        "is_malicious": is_mal,
        "confidence": round(float(confidence), 6),
        "class_probabilities": {k: round(float(v), 6) for k, v in probs_top.items()},
        "status": status,
        "severity": severity_for(status, confidence),
        "mitre": {"technique_id": tid, "technique_name": tname} if is_mal else None,
        "shap_evidence": evidence,
        "explanation": build_explanation(label, confidence, evidence, explanation_method or ""),
        "explanation_method": explanation_method,
        "data_quality": {"imputed_features": int(imputed), "out_of_range_features": int(out_of_range)},
        "model": {"name": "XGBoost", "version": model_version(), "trained_classes": int(n_classes)},
        "ontology_status": "PENDING" if is_mal else "NOT_REQUIRED",
    }
    if meta.get("label") not in (None, ""):
        event["ground_truth"] = str(meta["label"])
    return event


_SCHEMA = None


def validate_event(event: dict) -> List[str]:
    """Returns a list of schema errors (empty = valid). Needs `jsonschema`; skipped if missing."""
    global _SCHEMA
    try:
        import jsonschema
    except ImportError:
        return []
    if _SCHEMA is None:
        with open(config.EVENT_SCHEMA_PATH, encoding="utf-8") as f:
            _SCHEMA = json.load(f)
    v = jsonschema.Draft7Validator(_SCHEMA)
    return [f"{'/'.join(map(str, e.path))}: {e.message}" for e in v.iter_errors(event)]


def event_to_triples(event: dict) -> List[dict]:
    """Subject-predicate-object triples for malicious events (input for Module 2)."""
    if not event["is_malicious"]:
        return []
    eid = event["event_id"]
    src = event.get("src_ip") or "unknown_source"
    dst = event.get("dst_ip") or "unknown_target"
    out = [
        {"subject": src, "predicate": "performed", "object": event["predicted_class"], "event_id": eid},
        {"subject": src, "predicate": "targeted", "object": dst, "event_id": eid},
    ]
    m = event.get("mitre")
    if m and m.get("technique_id"):
        out.append({"subject": event["predicted_class"], "predicate": "maps_to_mitre",
                    "object": m["technique_id"], "event_id": eid})
    for ev in event["shap_evidence"]:
        if ev["effect"] == "supports":
            out.append({"subject": event["predicted_class"], "predicate": "explained_by",
                        "object": ev["feature"], "event_id": eid})
    return out
