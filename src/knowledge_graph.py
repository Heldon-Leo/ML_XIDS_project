"""
src/knowledge_graph.py
=======================
Converts ML-XIDS predictions + SHAP explanations into two integration-ready
outputs, so this module is ready to plug into a larger platform later
without needing another rewrite:

1. KNOWLEDGE GRAPH TRIPLES (subject, predicate, object) — consumable by a
   threat-graph / ontology module (e.g. Neo4j import, OWL population).
2. UNIFIED ALERTS — one consistent JSON schema per prediction, the kind of
   shape a shared dashboard would expect from every module in a larger
   system, not just this one.

Both are written as plain JSON: no assumption about which graph database
or dashboard framework eventually consumes them.
"""

import json
from typing import List

import numpy as np
import pandas as pd

import config
from src.utils import get_logger
from src.mitre_mapping import get_mitre_info

logger = get_logger("knowledge_graph")


def _describe_direction(raw_val: float, median_val: float) -> str:
    return "high" if raw_val >= median_val else "low"


def build_triples_and_alerts(
    model,
    shap_values_per_class: List[np.ndarray],
    X_scaled: np.ndarray,
    X_raw: pd.DataFrame,
    feature_names: List[str],
    label_encoder,
    top_n: int = None,
):
    """
    Walks the same sampled rows used for SHAP explanations and produces:
      - triples: list of {"subject", "predicate", "object"} dicts
      - alerts:  list of unified alert dicts, one per row

    Returns (triples, alerts).
    """
    top_n = top_n or config.TOP_N_FEATURES

    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)
    feature_medians = X_raw[feature_names].median()

    triples = []
    alerts = []
    unmapped_labels = set()

    n_samples = X_scaled.shape[0]
    for i in range(n_samples):
        predicted_class_idx = int(y_pred[i])
        predicted_label = str(label_encoder.inverse_transform([predicted_class_idx])[0])
        confidence = float(y_proba[i, predicted_class_idx])
        is_malicious = predicted_label.strip().lower() != "benign"

        user_id = (
            str(X_raw.iloc[i]["User_ID"])
            if "User_ID" in X_raw.columns
            else f"ROW-{i:05d}"
        )

        sample_shap = shap_values_per_class[predicted_class_idx][i]
        order = np.argsort(-np.abs(sample_shap))[:top_n]

        contributing_features = []
        for idx in order:
            fname = feature_names[idx]
            shap_val = float(sample_shap[idx])
            raw_val = float(X_raw.iloc[i][fname])
            median_val = float(feature_medians[fname])
            direction = _describe_direction(raw_val, median_val)

            contributing_features.append(
                {
                    "feature": fname,
                    "shap_value": round(shap_val, 4),
                    "raw_value": raw_val,
                    "direction": direction,
                }
            )
            # --- knowledge graph triples: entity -> triggeredBy -> feature
            triples.append(
                {"subject": user_id, "predicate": "triggeredBy", "object": fname}
            )

        # --- knowledge graph triples: entity -> hasPrediction -> label
        triples.append(
            {"subject": user_id, "predicate": "hasPrediction", "object": predicted_label}
        )
        triples.append(
            {"subject": user_id, "predicate": "hasConfidence", "object": round(confidence, 4)}
        )

        mitre_id, mitre_name = get_mitre_info(predicted_label)
        if is_malicious and mitre_id is None:
            unmapped_labels.add(predicted_label)
        if mitre_id is not None:
            triples.append(
                {"subject": predicted_label, "predicate": "mapsToMITRE", "object": mitre_id}
            )
            triples.append(
                {"subject": mitre_id, "predicate": "hasName", "object": mitre_name}
            )

        alerts.append(
            {
                "alert_id": f"ALERT-{user_id}",
                "source_module": "ML-XIDS",
                "entity_id": user_id,
                "prediction": predicted_label,
                "is_malicious": is_malicious,
                "confidence": round(confidence, 4),
                "mitre_technique_id": mitre_id,
                "mitre_technique_name": mitre_name,
                "contributing_features": contributing_features,
            }
        )

    if unmapped_labels:
        logger.warning(
            f"No MITRE ATT&CK mapping found for labels: {sorted(unmapped_labels)}. "
            f"Add them to src/mitre_mapping.py's CICIDS2018_TO_MITRE dict."
        )

    # Deduplicate: entity-level triples (hasPrediction, triggeredBy,
    # hasConfidence) are unique per row already, but label-level facts
    # (mapsToMITRE, hasName) get re-emitted once per row sharing that
    # prediction — collapse those back down to one copy each.
    seen = set()
    deduped_triples = []
    for t in triples:
        key = (t["subject"], t["predicate"], t["object"])
        if key not in seen:
            seen.add(key)
            deduped_triples.append(t)

    logger.info(
        f"Built {len(triples)} raw triples, deduplicated to {len(deduped_triples)}."
    )

    return deduped_triples, alerts


def save_triples(triples: list, output_path: str = None) -> str:
    output_path = output_path or config.TRIPLES_OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump(triples, f, indent=2)
    logger.info(f"Saved {len(triples)} knowledge graph triples to: {output_path}")
    return output_path


def save_alerts(alerts: list, output_path: str = None) -> str:
    output_path = output_path or config.ALERTS_OUTPUT_PATH
    with open(output_path, "w") as f:
        json.dump(alerts, f, indent=2)
    logger.info(f"Saved {len(alerts)} unified alerts to: {output_path}")
    return output_path