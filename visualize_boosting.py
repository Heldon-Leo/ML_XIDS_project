"""
visualize_boosting.py - how XGBoost classifies ONE flow, round by round and layer by layer.

    python visualize_boosting.py                       # first attack row of the cached test split
    python visualize_boosting.py --index 123           # a specific test row
    python visualize_boosting.py --label "DoS attacks-Hulk"
    python visualize_boosting.py --csv capture/live_flows.csv --index 0   # a live/captured row

Output: outputs/boosting_walkthrough.png with three panels
  A. probability of the top classes after 1, 2, ... N boosting rounds
  B. what each round's tree added to the predicted class (log-odds), plus the running total
  C. the layer-by-layer path (root -> leaf) the row takes in the strongest tree of the predicted class
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb

import config
from src.utils import load_joblib, load_json

BG, FG, MUTED, GRID = "#0b1220", "#e5edf7", "#94a3b8", "#1e2b45"
C_PRED, C_2ND, C_3RD, C_OTHER, POS, NEG = "#22d3ee", "#fbbf24", "#c084fc", "#475569", "#f87171", "#22d3ee"


def pick_row(a, features, classes):
    if a.csv:
        from src.schema_validator import resolve_columns, coerce_feature_frame
        from src.feature_stats import load_or_build_feature_stats
        raw = pd.read_csv(a.csv, low_memory=False, encoding="utf-8-sig")
        raw.columns = [str(c).strip() for c in raw.columns]
        res = resolve_columns(raw.columns, features)
        if res.missing:
            raise SystemExit(f"Missing features in {a.csv}: {res.missing[:5]}")
        X = coerce_feature_frame(raw, res, features)
        stats = load_or_build_feature_stats(features)
        if stats:
            X = X.fillna(pd.Series({f: stats[f]["median"] for f in features}))
        scaler = load_joblib(config.SCALER_PATH)
        Xs = pd.DataFrame(scaler.transform(X.to_numpy()), columns=features)
        return Xs.iloc[[a.index]], X.iloc[[a.index]], None
    data = load_joblib(config.SPLIT_DATA_CACHE_PATH)
    Xs = data.X_test if getattr(data, "X_test", None) is not None else data.X_train
    y = data.y_test if getattr(data, "X_test", None) is not None else data.y_train
    Xs = pd.DataFrame(np.asarray(Xs), columns=features)
    raw = getattr(data, "X_test_raw", None)
    raw = raw.reset_index(drop=True) if isinstance(raw, pd.DataFrame) and len(raw) == len(Xs) else None
    idx = a.index
    if idx is None and getattr(a, "random", False):
        rng = getattr(a, "rng", None) or np.random.default_rng()
        yy = None if y is None else np.asarray(y)
        if a.label and yy is not None:
            pool = np.where(yy == classes.index(a.label))[0]
        elif getattr(a, "attack_only", False) and yy is not None:
            pool = np.where(yy != next(i for i, c in enumerate(classes) if c.lower() == "benign"))[0]
        else:
            pool = np.arange(len(Xs))
        idx = int(rng.choice(pool))
    elif idx is None:
        want = [i for i, c in enumerate(classes) if (a.label and c == a.label) or (not a.label and c.lower() != "benign")]
        idx = int(np.where(np.isin(np.asarray(y), want))[0][0]) if y is not None else 0
    truth = classes[int(np.asarray(y)[idx])] if y is not None else None
    a.chosen_index = idx
    return Xs.iloc[[idx]], (raw.iloc[[idx]] if raw is not None else None), truth


def style(ax, title):
    ax.set_facecolor(BG)
    ax.set_title(title, color=FG, loc="left", fontsize=12, fontweight="bold")
    ax.tick_params(colors=MUTED, labelsize=9)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.grid(color=GRID, lw=0.6, alpha=0.8)
    ax.set_axisbelow(True)


def tree_path(booster, tree_idx, x, features):
    df = booster.trees_to_dataframe()
    t = df[df["Tree"] == tree_idx].set_index("ID")
    node, steps = f"{tree_idx}-0", []
    while t.loc[node, "Feature"] != "Leaf":
        r = t.loc[node]
        name = r["Feature"]
        if name not in features and name[:1] == "f" and name[1:].isdigit():
            name = features[int(name[1:])]        # booster trained on a bare numpy array
        fv = float(x[features.index(name)])
        go_yes = fv < r["Split"]          # XGBoost: "Yes" branch is value < split
        steps.append((name, fv, float(r["Split"]), go_yes))
        node = r["Yes"] if go_yes else r["No"]
    return steps, float(t.loc[node, "Gain"])   # Gain of a leaf row = leaf value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "boosting_walkthrough.png"))
    a = p.parse_args()
    if a.csv and a.index is None:
        a.index = 0

    features = load_json(config.FEATURE_LIST_PATH)
    model = load_joblib(config.MODEL_PATH)
    classes = [str(c) for c in load_joblib(config.LABEL_ENCODER_PATH).classes_]
    booster = model.get_booster()
    n_cls, rounds = len(classes), booster.num_boosted_rounds()

    row_s, row_raw, truth = pick_row(a, features, classes)
    d = xgb.DMatrix(row_s, feature_names=list(row_s.columns))
    margins = np.vstack([booster.predict(d, output_margin=True, iteration_range=(0, k)).reshape(-1)
                         for k in range(1, rounds + 1)])            # (rounds, classes)
    e = np.exp(margins - margins.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    final = probs[-1]
    order = np.argsort(final)[::-1]
    pred = int(order[0])
    step = np.diff(np.vstack([np.zeros(n_cls), margins]), axis=0)[:, pred]

    # strongest tree of the predicted class
    later = int(np.argmax(np.abs(step[1:]))) + 1 if rounds > 1 else 0
    shown = [0] + ([later] if later else [])          # first tree + strongest later tree
    paths = [(r, *tree_path(booster, r * n_cls + pred, row_s.iloc[0].to_numpy(), features)) for r in shown]

    fig = plt.figure(figsize=(15, 9.5), facecolor=BG)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.38, wspace=0.18)
    x = np.arange(1, rounds + 1)

    ax = fig.add_subplot(gs[0, 0])
    style(ax, "A. Probability as boosting rounds are added")
    for j in range(n_cls):
        if j not in order[:3]:
            ax.plot(x, probs[:, j], color=C_OTHER, lw=0.8, alpha=0.6)
    for rank, col in zip(range(3), (C_PRED, C_2ND, C_3RD)):
        j = order[rank]
        ax.plot(x, probs[:, j], color=col, lw=2.2, label=f"{classes[j]}  ({final[j]:.1%})")
        ax.annotate(f"{final[j]:.0%}", (rounds, final[j]), color=col, fontsize=9, xytext=(5, 0), textcoords="offset points", va="center")
    ax.axhline(1 / n_cls, color=MUTED, lw=0.8, ls=":")
    ax.text(rounds * 0.5, 1 / n_cls + 0.01, "dotted line = all 15 classes equally likely", color=MUTED, fontsize=8, va="bottom", ha="center")
    ax.set_xlabel("boosting rounds used (each round = one tree per class)", color=MUTED)
    ax.set_ylabel("probability", color=MUTED); ax.set_ylim(0, 1.02)
    ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=9, loc="center right")

    ax = fig.add_subplot(gs[0, 1])
    style(ax, f"B. What each round added to “{classes[pred]}” (log-odds)")
    ax.bar(x, step, color=[POS if v > 0 else NEG for v in step], width=1.0, lw=0)
    ax2 = ax.twinx()
    ax2.plot(x, margins[:, pred], color=FG, lw=1.8)
    ax2.tick_params(colors=MUTED, labelsize=9); ax2.set_ylabel("running total (line)", color=MUTED)
    for s in ax2.spines.values():
        s.set_color(GRID)
    if rounds > 2:   # round 1 also carries the starting score and would hide every later bar
        m = float(np.abs(step[1:]).max()) * 1.4
        ax.set_ylim(-m, m)
        ax.text(0.01, 0.97, "round 1 bar clipped (includes starting score)", transform=ax.transAxes, color=MUTED, fontsize=8, va="top")
    ax.set_xlabel("boosting round", color=MUTED); ax.set_ylabel("this round's tree output (bars)", color=MUTED)

    for col, (r, steps, leaf_val) in enumerate(paths):
        ax = fig.add_subplot(gs[1, col])
        ax.set_facecolor(BG); ax.axis("off")
        ax.set_title(f"C{col + 1}. Layer-by-layer path, round {r + 1} tree for “{classes[pred]}”"
                     + ("  (first tree)" if r == 0 else "  (strongest later tree)"),
                     color=FG, loc="left", fontsize=11, fontweight="bold")
        n = len(steps)
        for i, (feat, val, thr, yes) in enumerate(steps):
            y = 0.88 - i * min(0.13, 0.8 / max(n, 1))
            ax.text(0.0, y, f"Layer {i + 1}", color=MUTED, fontsize=10, va="center", transform=ax.transAxes)
            ax.text(0.14, y, feat, color=FG, fontsize=10.5, fontweight="bold", va="center", transform=ax.transAxes)
            ax.text(0.14, y - 0.045, f"{val:.4g} {'<' if yes else '≥'} {thr:.4g}  →  {'yes (left)' if yes else 'no (right)'}",
                    color=C_2ND, fontsize=9.5, va="center", transform=ax.transAxes)
        ax.text(0.0, 0.0, f"Leaf adds {leaf_val:+.3f} to this class's score"
                + (" (round 1 total also includes the model's starting score)" if r == 0 else ""),
                color=MUTED, fontsize=9, transform=ax.transAxes)
    fig.text(0.07, 0.015, "Feature values are in the scaled (0–1) space the model was trained on. Positive tree output pushes towards the predicted class.",
             color=MUTED, fontsize=9)

    head = f"XGBoost walkthrough · predicted {classes[pred]} ({final[pred]:.1%})"
    if truth:
        head += f" · true label: {truth}"
    fig.suptitle(head, color=FG, fontsize=15, fontweight="bold", x=0.07, ha="left", y=0.98)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=130, facecolor=BG, bbox_inches="tight")
    print(f"Saved {a.out}\nPredicted: {classes[pred]} ({final[pred]:.1%}), true: {truth}")


if __name__ == "__main__":
    main()