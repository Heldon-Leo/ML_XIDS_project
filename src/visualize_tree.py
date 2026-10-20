"""
src/visualize_tree.py
=====================
Given one row of preprocessed traffic features, runs it through the trained
XGBoost model, finds the predicted class, and renders the selected boosting
tree with the row's actual decision path highlighted.

Can be called from main.py or standalone through a caller script.
"""

import numpy as np
import pandas as pd
import graphviz
import xgboost as xgb

import config
from src.utils import get_logger


logger = get_logger("visualize_tree")

DEPTH_COLORS = [
    "#534AB7",  # depth 0
    "#0F6E56",  # depth 1
    "#993C1D",  # depth 2+
]
LEAF_COLOR = "#993C1D"
PATH_HIGHLIGHT = "#F2C744"
OUTCOME_BENIGN_COLOR = "#0F6E56"
OUTCOME_ATTACK_COLOR = "#A32D2D"

MAX_DEPTH_TO_RENDER = 2


def _node_depth(node_id: str) -> int:
    """Return the binary-tree depth encoded by an XGBoost node ID."""
    node_num = int(node_id.split("-")[1])
    depth = 0
    start = 0
    width = 1

    while node_num >= start + width:
        start += width
        width *= 2
        depth += 1

    return depth


def _trace_decision_path(df, row, root_id):
    """
    Follow the same yes/no/missing branch logic used by XGBoost
    for one row and return the node IDs visited.
    """
    path = [root_id]
    node_id = root_id

    while True:
        if node_id not in df.index:
            break

        row_data = df.loc[node_id]

        if row_data["Feature"] == "Leaf":
            break

        feat = row_data["Feature"]
        split = row_data["Split"]
        value = row.get(feat, np.nan)

        # XGBoost sends missing values through its Missing branch.
        if pd.isna(value):
            next_id = row_data["Missing"]
        elif value < split:
            next_id = row_data["Yes"]
        else:
            next_id = row_data["No"]

        if next_id not in df.index:
            break

        path.append(next_id)
        node_id = next_id

    return path


def _normalize_row(row, feature_columns):
    """Convert dict/Series/DataFrame input to a one-row DataFrame."""
    if isinstance(row, dict):
        return pd.DataFrame([row])[feature_columns]

    if isinstance(row, pd.Series):
        return row.to_frame().T[feature_columns]

    if isinstance(row, pd.DataFrame):
        if len(row) != 1:
            raise ValueError("row DataFrame must contain exactly one row.")
        return row[feature_columns]

    raise TypeError(
        "row must be a dict, pandas Series, or one-row pandas DataFrame."
    )


def predict_and_visualize_tree(
    model,
    feature_columns,
    row,
    label_encoder=None,
    round_idx=0,
    max_depth=None,
    output_path=None,
):
    """
    Predict a class for one preprocessed row and render the selected
    XGBoost tree with the actual decision path highlighted.

    Parameters
    ----------
    model : xgboost.XGBClassifier
        Trained XGBoost model.
    feature_columns : list[str]
        Features in the same order used during training.
    row : dict | pandas.Series | 1-row pandas.DataFrame
        One row of preprocessed features.
    label_encoder : sklearn.preprocessing.LabelEncoder, optional
        Used to convert class index to the original class name.
    round_idx : int
        Boosting round to visualize.
    max_depth : int, optional
        Maximum tree depth to render.
    output_path : str, optional
        Output PNG path.

    Returns
    -------
    dict
        Output path, predicted class, tree index, and decision path.
    """
    if max_depth is None:
        max_depth = MAX_DEPTH_TO_RENDER

    if output_path is None:
        output_path = config.TREE_PLOT_PATH

    row_df = _normalize_row(row, feature_columns)

    # Prediction comes from the trained XGBoost model.
    pred_class_idx = int(model.predict(row_df)[0])
    pred_proba = float(model.predict_proba(row_df)[0][pred_class_idx])

    if label_encoder is not None:
        pred_class_name = label_encoder.inverse_transform([pred_class_idx])[0]
    else:
        pred_class_name = str(pred_class_idx)

    logger.info(
        "Row predicted as '%s' (class_idx=%d, confidence=%.3f)",
        pred_class_name,
        pred_class_idx,
        pred_proba,
    )

    # For multiclass XGBoost, trees are arranged by boosting round/class.
    num_class = len(model.classes_)
    tree_idx = round_idx * num_class + pred_class_idx

    booster = model.get_booster()

    # Make sure tree feature names match the input columns.
    booster.feature_names = feature_columns

    tree_df = booster.trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == tree_idx].set_index("ID")

    root_id = f"{tree_idx}-0"
    decision_path = _trace_decision_path(
        tree_df,
        row_df.iloc[0],
        root_id=root_id,
    )

    path_set = set(decision_path)

    logger.info(
        "Decision path (%d nodes): %s",
        len(decision_path),
        decision_path,
    )

    # -------------------------
    # Render tree using Graphviz
    # -------------------------
    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", bgcolor="#0d0d0d")
    dot.attr(
        "node",
        fontname="Helvetica",
        fontcolor="white",
        style="filled,rounded",
    )
    dot.attr(
        "edge",
        color="#888888",
        fontcolor="#aaaaaa",
        fontname="Helvetica",
    )

    for node_id, r in tree_df.iterrows():
        depth = _node_depth(node_id)

        if depth > max_depth:
            continue

        is_leaf = r["Feature"] == "Leaf"

        fill = (
            LEAF_COLOR
            if is_leaf
            else DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)]
        )

        if is_leaf:
            label = f"leaf = {r['Gain']:.3f}"
        else:
            label = f"{r['Feature']} < {r['Split']:.4g}"

        on_path = node_id in path_set
        pen_width = "4" if on_path else "0"
        border_color = PATH_HIGHLIGHT if on_path else "none"

        dot.node(
            node_id,
            label=label,
            shape="box",
            fillcolor=fill,
            color=border_color,
            penwidth=pen_width,
        )

        if not is_leaf and depth < max_depth:
            yes_id = r["Yes"]
            no_id = r["No"]

            def edge_color(next_id):
                if node_id in path_set and next_id in path_set:
                    return PATH_HIGHLIGHT
                return "#888888"

            if yes_id in tree_df.index:
                color = edge_color(yes_id)
                dot.edge(
                    node_id,
                    yes_id,
                    label="yes",
                    color=color,
                    penwidth="2" if color == PATH_HIGHLIGHT else "1",
                )

            if no_id in tree_df.index:
                color = edge_color(no_id)
                dot.edge(
                    node_id,
                    no_id,
                    label="no",
                    color=color,
                    penwidth="2" if color == PATH_HIGHLIGHT else "1",
                )

    # -------------------------
    # Add final prediction node
    # -------------------------
    last_rendered_path_node = None

    for node_id in decision_path:
        if _node_depth(node_id) <= max_depth:
            last_rendered_path_node = node_id
        else:
            break

    is_benign = str(pred_class_name).strip().lower() == "benign"

    outcome_color = (
        OUTCOME_BENIGN_COLOR
        if is_benign
        else OUTCOME_ATTACK_COLOR
    )

    if is_benign:
        outcome_label = (
            f"Prediction: Benign\\n"
            f"confidence {pred_proba:.2%}"
        )
    else:
        outcome_label = (
            f"Prediction: Attacker\\n"
            f"({pred_class_name})\\n"
            f"confidence {pred_proba:.2%}"
        )

    outcome_id = "outcome"

    dot.node(
        outcome_id,
        label=outcome_label,
        shape="box",
        fillcolor=outcome_color,
        color=PATH_HIGHLIGHT,
        penwidth="3",
    )

    if last_rendered_path_node is not None:
        dot.edge(
            last_rendered_path_node,
            outcome_id,
            color=PATH_HIGHLIGHT,
            penwidth="2",
            style="dashed",
        )

    # -------------------------
    # Save PNG
    # -------------------------
    output_path = str(output_path)
    render_base = output_path.rsplit(".", 1)[0]

    try:
        dot.render(
            filename=render_base,
            format="png",
            cleanup=True,
        )
        logger.info("Tree plot saved to: %s", output_path)

    except graphviz.backend.execute.ExecutableNotFound:
        logger.warning(
            "Graphviz 'dot' executable was not found on PATH. "
            "Falling back to matplotlib."
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(30, 15))
        xgb.plot_tree(
            booster,
            tree_idx=tree_idx,
            ax=ax,
        )
        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        logger.info(
            "Tree plot saved to: %s (unstyled fallback)",
            output_path,
        )

    return {
        "output_path": output_path,
        "predicted_class_idx": pred_class_idx,
        "predicted_class_name": pred_class_name,
        "tree_idx": tree_idx,
        "decision_path": decision_path,
    }


def visualize_tree(
    model,
    feature_columns,
    tree_idx=None,
    max_depth=None,
    output_path=None,
):
    """
    Render a tree by raw XGBoost tree index without using a row.
    """
    if tree_idx is None:
        tree_idx = config.TREE_VIS_INDEX

    if max_depth is None:
        max_depth = MAX_DEPTH_TO_RENDER

    if output_path is None:
        output_path = config.TREE_PLOT_PATH

    booster = model.get_booster()
    booster.feature_names = feature_columns

    tree_df = booster.trees_to_dataframe()
    tree_df = tree_df[tree_df["Tree"] == tree_idx].set_index("ID")

    dot = graphviz.Digraph()
    dot.attr(rankdir="TB", bgcolor="#0d0d0d")
    dot.attr(
        "node",
        fontname="Helvetica",
        fontcolor="white",
        style="filled,rounded",
    )
    dot.attr(
        "edge",
        color="#888888",
        fontcolor="#aaaaaa",
        fontname="Helvetica",
    )

    for node_id, r in tree_df.iterrows():
        depth = _node_depth(node_id)

        if depth > max_depth:
            continue

        is_leaf = r["Feature"] == "Leaf"

        fill = (
            LEAF_COLOR
            if is_leaf
            else DEPTH_COLORS[min(depth, len(DEPTH_COLORS) - 1)]
        )

        if is_leaf:
            label = f"leaf = {r['Gain']:.3f}"
        else:
            label = f"{r['Feature']} < {r['Split']:.4g}"

        dot.node(
            node_id,
            label=label,
            shape="box",
            fillcolor=fill,
            color="none",
        )

        if not is_leaf and depth < max_depth:
            yes_id = r["Yes"]
            no_id = r["No"]

            if yes_id in tree_df.index:
                dot.edge(node_id, yes_id, label="yes")

            if no_id in tree_df.index:
                dot.edge(node_id, no_id, label="no")

    output_path = str(output_path)
    render_base = output_path.rsplit(".", 1)[0]

    try:
        dot.render(
            filename=render_base,
            format="png",
            cleanup=True,
        )
        logger.info("Tree plot saved to: %s", output_path)

    except graphviz.backend.execute.ExecutableNotFound:
        logger.warning(
            "Graphviz 'dot' not found on PATH; "
            "falling back to matplotlib."
        )

        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(30, 15))
        xgb.plot_tree(
            booster,
            tree_idx=tree_idx,
            ax=ax,
        )
        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)

        logger.info("Tree plot saved to: %s", output_path)

    return output_path
