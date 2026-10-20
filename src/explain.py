"""
explain.py
==========
Explainable AI module (the core deliverable of this project).

Uses SHAP's TreeExplainer (TreeSHAP) — the exact, fast algorithm for
tree ensembles like XGBoost — to produce:

GLOBAL explanations:
    - SHAP summary plot (beeswarm), saved as PNG
    - Global feature importance ranking (mean |SHAP value|), saved as CSV

LOCAL explanations (per test sample):
    - Top-N contributing features for that sample's predicted class
    - A human-readable sentence describing why the model made that call
    - One illustrative SHAP force plot, saved as an interactive HTML file

STRUCTURED OUTPUT:
    - predictions_with_explanations.csv with columns:
      Prediction, Confidence, Top_Features, Explanation_Text
"""

import os
import sys
from typing import List, Tuple

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src.utils import get_logger, timeit

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# SHAP VALUE EXTRACTION (version-robust)
# ---------------------------------------------------------------------------
def _compute_shap_values(model, X: np.ndarray, feature_names: List[str]):
    """
    Builds a TreeExplainer for the trained XGBoost model and computes SHAP
    values for X. Handles the two shapes the `shap` library has used for
    multi-class output across versions:
      - list of length n_classes, each (n_samples, n_features)          [older shap]
      - single ndarray of shape (n_samples, n_features, n_classes)      [newer shap]
    Returns a standardized list-of-arrays: shap_values[class_idx] -> (n, f)
    """
    explainer = shap.TreeExplainer(model)
    raw_shap = explainer.shap_values(X)

    if isinstance(raw_shap, list):
        shap_values_per_class = raw_shap
    else:
        raw_shap = np.asarray(raw_shap)
        if raw_shap.ndim == 3:
            # (n_samples, n_features, n_classes) -> list of (n_samples, n_features)
            n_classes = raw_shap.shape[-1]
            shap_values_per_class = [raw_shap[:, :, c] for c in range(n_classes)]
        else:
            # Binary classification / single-output model
            shap_values_per_class = [raw_shap]

    return explainer, shap_values_per_class


# ---------------------------------------------------------------------------
# GLOBAL EXPLANATIONS
# ---------------------------------------------------------------------------
@timeit
def generate_global_explanations(
    shap_values_per_class: List[np.ndarray],
    X: pd.DataFrame,
    feature_names: List[str],
    class_names: List[str],
) -> pd.DataFrame:
    """
    Produces:
      1. A SHAP summary (beeswarm) plot averaged across classes, saved as PNG.
      2. A global feature-importance CSV ranked by mean(|SHAP value|)
         aggregated across all classes.
    """
    # Aggregate absolute SHAP values across all classes to get one global
    # importance score per feature (standard approach for multi-class SHAP).
    stacked = np.stack([np.abs(sv) for sv in shap_values_per_class], axis=0)  # (C, N, F)
    mean_abs_shap_per_class_sample = stacked.mean(axis=0)  # (N, F) averaged over classes
    global_importance = mean_abs_shap_per_class_sample.mean(axis=0)  # (F,)

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Mean_Abs_SHAP_Value": global_importance})
        .sort_values("Mean_Abs_SHAP_Value", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv(config.FEATURE_IMPORTANCE_CSV_PATH, index=False)
    logger.info(f"Saved global feature importance -> {config.FEATURE_IMPORTANCE_CSV_PATH}")

    # ---------------------------------------------------------------
    # Summary (beeswarm) plot. For multi-class models, shap.summary_plot
    # accepts a list of arrays directly and produces a stacked bar-style
    # summary of per-class mean |SHAP| — clear and standard for XIDS reports.
    # ---------------------------------------------------------------
    plt.figure()
    shap.summary_plot(
        shap_values_per_class,
        X,
        feature_names=feature_names,
        class_names=class_names,
        show=False,
        max_display=20,
    )
    plt.tight_layout()
    plt.savefig(config.SHAP_SUMMARY_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved SHAP summary plot -> {config.SHAP_SUMMARY_PLOT_PATH}")

    return importance_df


# ---------------------------------------------------------------------------
# LOCAL EXPLANATIONS
# ---------------------------------------------------------------------------
def _describe_feature_direction(feature_value: float, feature_median: float) -> str:
    """Returns 'high' or 'low' depending on where the value sits vs. the median."""
    return "high" if feature_value >= feature_median else "low"


def _build_explanation_sentence(
    top_features: List[Tuple[str, float, float, float]], predicted_label: str
) -> str:
    """
    Builds a human-readable sentence from the top contributing features.

    `top_features` is a list of tuples:
        (feature_name, shap_value, raw_feature_value, feature_median)

    Example output:
        "High Flow Duration, low Fwd Packet Length Max, and high SYN Flag
         Count contributed to DDoS-HOIC classification."
    """
    descriptors = []
    for name, shap_val, raw_val, median_val in top_features:
        direction = _describe_feature_direction(raw_val, median_val)
        # Only mention features that push TOWARD the predicted class
        # (positive SHAP value for that class) to keep the narrative causal.
        descriptors.append(f"{direction} {name}")

    if len(descriptors) == 1:
        joined = descriptors[0]
    elif len(descriptors) == 2:
        joined = f"{descriptors[0]} and {descriptors[1]}"
    else:
        joined = ", ".join(descriptors[:-1]) + f", and {descriptors[-1]}"

    return f"{joined.capitalize()} contributed to {predicted_label} classification."


@timeit
def generate_local_explanations(
    model,
    shap_values_per_class: List[np.ndarray],
    X_scaled: np.ndarray,
    X_raw: pd.DataFrame,
    feature_names: List[str],
    label_encoder,
    top_n: int = config.TOP_N_FEATURES,
) -> pd.DataFrame:
    """
    For every sampled test row, extracts the top-N SHAP-contributing
    features toward the model's *predicted* class and builds a
    human-readable explanation string.

    Returns a DataFrame with columns:
        Prediction, Confidence, Top_Features, Explanation_Text
    """
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)

    feature_medians = X_raw[feature_names].median()

    records = []
    n_samples = X_scaled.shape[0]

    for i in range(n_samples):
        predicted_class_idx = int(y_pred[i])
        predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
        confidence = float(y_proba[i, predicted_class_idx])

        # SHAP values for THIS sample, toward THIS predicted class
        sample_shap = shap_values_per_class[predicted_class_idx][i]

        # Prefer features that push POSITIVELY toward the predicted class
        # (a positive SHAP value = evidence supporting this classification),
        # so the generated sentence reads as a causal explanation rather
        # than mixing in features that actually argued against the call.
        # If fewer than `top_n` positive contributors exist, backfill with
        # the remaining highest-magnitude features.
        positive_idx = np.where(sample_shap > 0)[0]
        positive_sorted = positive_idx[np.argsort(-sample_shap[positive_idx])]

        if len(positive_sorted) >= top_n:
            order = positive_sorted[:top_n]
        else:
            remaining_needed = top_n - len(positive_sorted)
            other_idx = np.array(
                [j for j in np.argsort(-np.abs(sample_shap)) if j not in positive_sorted]
            )
            order = np.concatenate([positive_sorted, other_idx[:remaining_needed]])

        top_features_info = []
        top_features_readable = []
        for idx in order:
            fname = feature_names[idx]
            shap_val = float(sample_shap[idx])
            raw_val = float(X_raw.iloc[i][fname])
            median_val = float(feature_medians[fname])
            top_features_info.append((fname, shap_val, raw_val, median_val))
            direction = _describe_feature_direction(raw_val, median_val)
            top_features_readable.append(f"{fname} ({direction}, SHAP={shap_val:.4f})")

        explanation_text = _build_explanation_sentence(top_features_info, predicted_label)

        user_id = X_raw.iloc[i]["User_ID"] if "User_ID" in X_raw.columns else i

        records.append(
            {
                "User_ID": user_id,
                "Prediction": predicted_label,
                "Confidence": round(confidence, 4),
                "Top_Features": "; ".join(top_features_readable),
                "Explanation_Text": explanation_text,
            }
        )

    result_df = pd.DataFrame(records)
    result_df.to_csv(config.PREDICTIONS_EXPLANATIONS_CSV_PATH, index=False)
    logger.info(
        f"Saved {len(result_df)} local explanations -> "
        f"{config.PREDICTIONS_EXPLANATIONS_CSV_PATH}"
    )
    return result_df


# ---------------------------------------------------------------------------
# BENIGN vs. ATTACK EXPLANATION PLOT
# ---------------------------------------------------------------------------
def _get_benign_class_idx(label_encoder) -> int:
    """Finds the class index corresponding to 'Benign' (case-insensitive).
    Falls back to 0 if no exact match is found (and logs a warning)."""
    classes = list(label_encoder.classes_)
    for i, c in enumerate(classes):
        if str(c).strip().lower() == "benign":
            return i
    logger.warning(
        f"No class named 'Benign' found in label_encoder classes {classes}; "
        f"defaulting to class index 0 for the benign-vs-attack plot."
    )
    return 0


@timeit
def generate_benign_vs_attack_plot(
    explainer,
    shap_values_per_class: List[np.ndarray],
    X_scaled: np.ndarray,
    feature_names: List[str],
    model,
    label_encoder,
    sample_index: int,
    output_path: str,
    top_n: int = 10,
) -> str:
    """
    For one specific flow (sample_index), plots the top-N features that
    pushed its SHAP value toward or away from the 'Benign' class.

    Reads as a single coherent story:
      - Green bar (positive)  -> this feature's value made the flow look
        MORE like normal/benign traffic.
      - Red bar (negative)    -> this feature's value made the flow look
        LESS like benign traffic, i.e. more like an attack.

    This directly answers "why was this flow flagged as benign or
    malicious?", unlike the global summary plot which averages across
    all 15 classes at once.
    """
    benign_idx = _get_benign_class_idx(label_encoder)

    predicted_class_idx = int(model.predict(X_scaled[sample_index : sample_index + 1])[0])
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = float(
        model.predict_proba(X_scaled[sample_index : sample_index + 1])[0][predicted_class_idx]
    )

    # SHAP values for THIS sample, toward the BENIGN class specifically.
    # Positive = evidence for benign; negative = evidence against benign
    # (i.e. toward whichever attack class actually won).
    sample_shap_benign = shap_values_per_class[benign_idx][sample_index]

    top_idx = np.argsort(-np.abs(sample_shap_benign))[:top_n]
    feats = [feature_names[i] for i in top_idx]
    vals = [float(sample_shap_benign[i]) for i in top_idx]

    # Sort ascending so the largest-magnitude bars end up at the top of the chart
    order = np.argsort(vals)
    feats = [feats[i] for i in order]
    vals = [vals[i] for i in order]
    colors = ["#1E7A50" if v > 0 else "#B23A2E" for v in vals]

    is_benign = predicted_label.strip().lower() == "benign"
    outcome_str = "BENIGN" if is_benign else f"MALICIOUS ({predicted_label})"

    plt.figure(figsize=(9, 6))
    plt.barh(feats, vals, color=colors)
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title(
        f"Why this flow was classified as {outcome_str}\n"
        f"(model confidence: {confidence:.1%})",
        fontsize=12,
    )
    plt.xlabel(
        "SHAP value toward Benign\n"
        "(green = looks normal, red = looks malicious)"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(
        f"Saved benign-vs-attack explanation for sample #{sample_index} "
        f"(predicted: {predicted_label}, confidence: {confidence:.1%}) -> {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# BEHAVIOR TREE EXPLANATION PLOT
# ---------------------------------------------------------------------------
# Groups features into human-meaningful traffic-behavior categories so the
# tree reads as "why did this user's packets/timing/ports/flags look
# benign or malicious", rather than a flat list of 70 raw column names.
_BEHAVIOR_CATEGORY_RULES = [
    ("Packet Behavior", ["pkt", "pkts", "len", "size"]),
    ("Destination / Port", ["port", "dst", "protocol"]),
    ("Timing", ["duration", "iat", "active", "idle"]),
    ("Flags / Protocol", ["flag", "urg", "psh", "syn", "ack", "fin", "rst", "ece", "cwe"]),
]


def _categorize_feature(feature_name: str) -> str:
    fname = feature_name.lower()
    for category, keywords in _BEHAVIOR_CATEGORY_RULES:
        if any(k in fname for k in keywords):
            return category
    return "Other"


@timeit
def generate_behavior_tree_plot(
    explainer,
    shap_values_per_class: List[np.ndarray],
    X_scaled: np.ndarray,
    X_raw: pd.DataFrame,
    feature_names: List[str],
    model,
    label_encoder,
    sample_index: int,
    output_path: str,
    top_per_category: int = None,
) -> str:
    """
    Renders a hierarchical tree diagram for one flow:

        [Prediction outcome] -> [Behavior category] -> [individual feature]

    Categories group the 70 raw CICFlowMeter columns into readable traffic-
    behavior buckets (Packet Behavior, Destination/Port, Timing,
    Flags/Protocol), so the diagram tells a behavioral story ("this flow's
    packet counts, destination port, timing, and flags all looked
    malicious") instead of a flat bar chart of column names.

    Node/branch color = direction of each category's/feature's SHAP value
    toward the Benign class (green = looks normal, red = looks malicious).
    Leaf box width scales with the magnitude of that feature's SHAP value.
    """
    top_per_category = top_per_category or config.BEHAVIOR_TREE_TOP_PER_CATEGORY
    benign_idx = _get_benign_class_idx(label_encoder)

    predicted_class_idx = int(model.predict(X_scaled[sample_index : sample_index + 1])[0])
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]
    confidence = float(
        model.predict_proba(X_scaled[sample_index : sample_index + 1])[0][predicted_class_idx]
    )
    is_benign = predicted_label.strip().lower() == "benign"

    sample_shap = shap_values_per_class[benign_idx][sample_index]
    raw_row = X_raw.iloc[sample_index][feature_names].values

    # Group features into categories, keep the top-N by |SHAP| per category
    groups = {}
    for i, fname in enumerate(feature_names):
        cat = _categorize_feature(fname)
        groups.setdefault(cat, []).append((fname, float(sample_shap[i]), float(raw_row[i])))

    cat_order = []
    for cat, _ in _BEHAVIOR_CATEGORY_RULES + [("Other", [])]:
        if cat in groups and groups[cat]:
            feats = sorted(groups[cat], key=lambda t: -abs(t[1]))[:top_per_category]
            cat_total = sum(v for _, v, _ in groups[cat])
            cat_order.append((cat, feats, cat_total))

    # --- render ---
    fig, ax = plt.subplots(figsize=(13, max(6, 1.6 * len(cat_order))))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    outcome_str = "BENIGN" if is_benign else f"MALICIOUS\n({predicted_label})"
    root_color = "#1E7A50" if is_benign else "#A32D2D"
    root_xy = (0.8, 5)
    ax.add_patch(
        FancyBboxPatch(
            (root_xy[0] - 0.7, root_xy[1] - 0.6), 1.4, 1.2,
            boxstyle="round,pad=0.05,rounding_size=0.15",
            linewidth=1.5, edgecolor="white", facecolor=root_color,
        )
    )
    ax.text(
        root_xy[0], root_xy[1], f"{outcome_str}\n{confidence:.0%}",
        ha="center", va="center", color="white", fontsize=11, fontweight="bold",
    )

    n_cat = len(cat_order)
    cat_y_positions = np.linspace(8.5, 1.5, n_cat) if n_cat > 1 else [5]

    for (cat, feats, cat_total), cat_y in zip(cat_order, cat_y_positions):
        cat_x, cat_w = 4.2, 2.4
        cat_color = "#1E7A50" if cat_total >= 0 else "#A32D2D"
        ax.add_patch(
            FancyBboxPatch(
                (cat_x - cat_w / 2, cat_y - 0.45), cat_w, 0.9,
                boxstyle="round,pad=0.04,rounding_size=0.12",
                linewidth=1.2, edgecolor="#333333", facecolor=cat_color, alpha=0.85,
            )
        )
        ax.text(cat_x, cat_y, cat, ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.add_patch(
            FancyArrowPatch(
                root_xy, (cat_x - cat_w / 2, cat_y), connectionstyle="arc3,rad=0.15",
                arrowstyle="-", color="#999999", linewidth=1.3,
            )
        )

        n_feat = len(feats)
        feat_spacing = 0.85
        feat_y_start = cat_y + (n_feat - 1) * feat_spacing / 2
        for j, (fname, shap_val, raw_val) in enumerate(feats):
            feat_y = feat_y_start - j * feat_spacing
            feat_x = 7.8
            color = "#2E9B6B" if shap_val > 0 else "#C24A3C"
            width = min(abs(shap_val) * 2.2 + 1.4, 3.2)
            ax.add_patch(
                FancyBboxPatch(
                    (feat_x - width / 2, feat_y - 0.3), width, 0.6,
                    boxstyle="round,pad=0.03,rounding_size=0.1",
                    linewidth=1, edgecolor="#222222", facecolor=color,
                )
            )
            ax.text(
                feat_x, feat_y, f"{fname}\nSHAP={shap_val:+.3f}  (val={raw_val:.3g})",
                ha="center", va="center", color="white", fontsize=8,
            )
            ax.add_patch(
                FancyArrowPatch(
                    (cat_x + cat_w / 2, cat_y), (feat_x - width / 2, feat_y),
                    connectionstyle="arc3,rad=0.12", arrowstyle="-", color="#bbbbbb", linewidth=1,
                )
            )

    plt.title(
        f"Behavior profile tree \u2014 why this flow is {outcome_str.replace(chr(10), ' ')}",
        fontsize=13, pad=15,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    logger.info(
        f"Saved behavior tree explanation for sample #{sample_index} "
        f"(predicted: {predicted_label}, confidence: {confidence:.1%}) -> {output_path}"
    )
    return output_path


# ---------------------------------------------------------------------------
# SHAP FORCE PLOT (single illustrative sample, saved as standalone HTML)
# ---------------------------------------------------------------------------
@timeit
def generate_force_plot(
    explainer,
    shap_values_per_class: List[np.ndarray],
    X_scaled: np.ndarray,
    feature_names: List[str],
    model,
    label_encoder,
    sample_index: int = config.FORCE_PLOT_SAMPLE_INDEX,
) -> None:
    """
    Generates an interactive SHAP force plot for one representative test
    sample (for its predicted class) and saves it as a standalone HTML file.
    """
    predicted_class_idx = int(model.predict(X_scaled[sample_index : sample_index + 1])[0])
    predicted_label = label_encoder.inverse_transform([predicted_class_idx])[0]

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = base_value[predicted_class_idx]

    force_plot = shap.force_plot(
        base_value,
        shap_values_per_class[predicted_class_idx][sample_index],
        X_scaled[sample_index],
        feature_names=feature_names,
    )

    shap.save_html(config.SHAP_FORCE_PLOT_PATH, force_plot)
    logger.info(
        f"Saved SHAP force plot for sample #{sample_index} "
        f"(predicted: {predicted_label}) -> {config.SHAP_FORCE_PLOT_PATH}"
    )


# ---------------------------------------------------------------------------
# ORCHESTRATOR
# ---------------------------------------------------------------------------
@timeit
def run_explainability_pipeline(
    model,
    X_test: np.ndarray,
    X_test_raw: pd.DataFrame,
    feature_names: List[str],
    label_encoder,
) -> dict:
    """
    Runs the full explainability pipeline:
      1. Sub-samples the test set (SHAP_SAMPLE_SIZE) for tractability
      2. Computes TreeSHAP values
      3. Generates global explanations (summary plot + importance CSV)
      4. Generates local explanations (predictions_with_explanations.csv)
      5. Generates one illustrative SHAP force plot (HTML)
    """
    n_total = X_test.shape[0]
    sample_size = config.SHAP_SAMPLE_SIZE or n_total
    sample_size = min(sample_size, n_total)

    rng = np.random.RandomState(config.RANDOM_STATE)
    sample_idx = rng.choice(n_total, size=sample_size, replace=False)
    sample_idx.sort()

    X_sample = X_test[sample_idx]
    X_sample_raw = X_test_raw.iloc[sample_idx].reset_index(drop=True)
    class_names = list(label_encoder.classes_)

    logger.info(f"Computing SHAP values for {sample_size} test samples ...")
    explainer, shap_values_per_class = _compute_shap_values(
        model, X_sample, feature_names
    )

    X_sample_df = pd.DataFrame(X_sample, columns=feature_names)

    importance_df = generate_global_explanations(
        shap_values_per_class, X_sample_df, feature_names, class_names
    )

    explanations_df = generate_local_explanations(
        model,
        shap_values_per_class,
        X_sample,
        X_sample_raw,
        feature_names,
        label_encoder,
    )

    generate_force_plot(
        explainer,
        shap_values_per_class,
        X_sample,
        feature_names,
        model,
        label_encoder,
        sample_index=config.FORCE_PLOT_SAMPLE_INDEX,
    )

    # ---------------------------------------------------------------
    # Benign-vs-Attack explanation plots: one for a flow predicted
    # Benign, one for a flow predicted as an attack, so the two can be
    # compared directly ("why normal" vs "why flagged").
    # ---------------------------------------------------------------
    y_pred_sample = model.predict(X_sample)
    benign_class_idx = _get_benign_class_idx(label_encoder)

    benign_rows = np.where(y_pred_sample == benign_class_idx)[0]
    attack_rows = np.where(y_pred_sample != benign_class_idx)[0]

    if len(benign_rows) > 0:
        benign_plot_path = config.BENIGN_VS_ATTACK_PLOT_PATH.replace(
            ".png", "_benign_example.png"
        )
        generate_benign_vs_attack_plot(
            explainer,
            shap_values_per_class,
            X_sample,
            feature_names,
            model,
            label_encoder,
            sample_index=int(benign_rows[0]),
            output_path=benign_plot_path,
        )

    if len(attack_rows) > 0:
        attack_plot_path = config.BENIGN_VS_ATTACK_PLOT_PATH.replace(
            ".png", "_attack_example.png"
        )
        generate_benign_vs_attack_plot(
            explainer,
            shap_values_per_class,
            X_sample,
            feature_names,
            model,
            label_encoder,
            sample_index=int(attack_rows[0]),
            output_path=attack_plot_path,
        )

    # ---------------------------------------------------------------
    # Behavior tree diagrams: same benign/attack pair, but as a richer
    # hierarchical tree grouped by traffic-behavior category (packets,
    # destination/port, timing, flags) rather than a flat bar chart.
    # ---------------------------------------------------------------
    if len(benign_rows) > 0:
        generate_behavior_tree_plot(
            explainer,
            shap_values_per_class,
            X_sample,
            X_sample_raw,
            feature_names,
            model,
            label_encoder,
            sample_index=int(benign_rows[0]),
            output_path=config.BEHAVIOR_TREE_PLOT_PATH.replace(".png", "_benign_example.png"),
        )

    if len(attack_rows) > 0:
        generate_behavior_tree_plot(
            explainer,
            shap_values_per_class,
            X_sample,
            X_sample_raw,
            feature_names,
            model,
            label_encoder,
            sample_index=int(attack_rows[0]),
            output_path=config.BEHAVIOR_TREE_PLOT_PATH.replace(".png", "_attack_example.png"),
        )

    # ---------------------------------------------------------------
    # Knowledge graph triples + unified alerts: integration-ready
    # outputs for a larger platform (threat graph, ontology, dashboard
    # modules). Reuses the SHAP values already computed above for the
    # sampled rows, rather than recomputing them.
    # ---------------------------------------------------------------
    from src.knowledge_graph import build_triples_and_alerts, save_triples, save_alerts

    triples, alerts = build_triples_and_alerts(
        model,
        shap_values_per_class,
        X_sample,
        X_sample_raw,
        feature_names,
        label_encoder,
    )
    triples_path = save_triples(triples)
    alerts_path = save_alerts(alerts)

    return {
        "importance_df": importance_df,
        "explanations_df": explanations_df,
        "triples": triples,
        "alerts": alerts,
        "triples_path": triples_path,
        "alerts_path": alerts_path,
    }


if __name__ == "__main__":
    from src.data_loader import load_and_merge_csvs
    from src.preprocessing import preprocess
    from src.train import train_model

    raw_df = load_and_merge_csvs()
    data = preprocess(raw_df)
    model = train_model(data.X_train, data.y_train)
    run_explainability_pipeline(
        model, data.X_test, data.X_test_raw, data.feature_columns, data.label_encoder
    )