"""
visualize_boosting_trees.py - draws the ACTUAL XGBoost trees round by round for one flow.

For the predicted class it shows the tree of round 1, 2, 3 ... and the last round. In each tree
the path the flow takes is highlighted; grey stubs are the branches it did not take. Under each
tree: the leaf value added, the running probability, and the error still left. Each next tree is
trained to correct what the previous trees still got wrong, so the error shrinks left -> right.

    python visualize_boosting_trees.py                    # first attack row of the test split
    python visualize_boosting_trees.py --index 123
    python visualize_boosting_trees.py --label "DoS attacks-Hulk"
    python visualize_boosting_trees.py --csv capture/live_flows.csv --index 0
    python visualize_boosting_trees.py --rounds 1 2 3 10 300
Output: outputs/boosting_trees.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import xgboost as xgb

import config
from src.utils import load_joblib, load_json
from visualize_boosting import pick_row

BG, FG, MUTED, GRID = "#0b1220", "#e5edf7", "#94a3b8", "#1e2b45"
CYAN, AMBER, RED, PANEL = "#22d3ee", "#fbbf24", "#f87171", "#111a2e"


def subtree_size(t, node):
    """Number of nodes below (and including) `node`."""
    n, stack = 0, [node]
    while stack:
        cur = stack.pop(); n += 1
        if t.loc[cur, "Feature"] != "Leaf":
            stack += [t.loc[cur, "Yes"], t.loc[cur, "No"]]
    return n


def walk(booster_df, tree_idx, x, features):
    t = booster_df[booster_df["Tree"] == tree_idx].set_index("ID")
    node, path = f"{tree_idx}-0", []
    while t.loc[node, "Feature"] != "Leaf":
        r = t.loc[node]
        name = r["Feature"]
        if name not in features and name[:1] == "f" and name[1:].isdigit():
            name = features[int(name[1:])]
        v = float(x[features.index(name)])
        yes = v < r["Split"]
        taken, other = (r["Yes"], r["No"]) if yes else (r["No"], r["Yes"])
        path.append(dict(name=name, v=v, thr=float(r["Split"]), yes=yes, other_size=subtree_size(t, other)))
        node = taken
    return path, float(t.loc[node, "Gain"])


def box(ax, x, y, w, h, text, fc, ec, tc=FG, fs=8, bold=False):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.4))
    ax.text(x, y, text, color=tc, fontsize=fs, ha="center", va="center", fontweight="bold" if bold else "normal")


def draw_tree(ax, cx, top, path, leaf, color):
    dy, side = 1.05, 1.5
    y = top
    for d in path:
        box(ax, cx, y, 2.05, 0.62, f"{d['name']}\n{d['v']:.3g} {'<' if d['yes'] else '≥'} {d['thr']:.3g} ?", PANEL, color, fs=7.5)
        ny = y - dy
        sg = -1 if d["yes"] else 1                 # yes = left, no = right (stub sits on the NOT-taken side)
        ax.plot([cx, cx], [y - 0.31, ny + 0.31], color=color, lw=2.4)
        ax.text(cx + 0.14, (y + ny) / 2, "yes" if d["yes"] else "no", color=color, fontsize=7, ha="left", va="center")
        ox = cx - sg * side
        ax.plot([cx - sg * 0.3, ox], [y - 0.31, ny], color=GRID, lw=1.1, ls="--")
        ax.add_patch(FancyBboxPatch((ox - 0.4, ny - 0.17), 0.8, 0.34, boxstyle="round,pad=0.01,rounding_size=0.06", fc=BG, ec=GRID, lw=1))
        ax.text(ox, ny, f"{d['other_size']} nodes", color=MUTED, fontsize=6.5, ha="center", va="center")
        y = ny
    box(ax, cx, y, 1.35, 0.5, f"leaf {leaf:+.3f}", color + "33", color, fs=9, bold=True)
    return cx, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--rounds", type=int, nargs="+", default=None, help="1-based rounds to draw (disables random rounds)")
    p.add_argument("--n-rounds", type=int, default=5, help="how many trees to draw (minimum 5)")
    p.add_argument("--attack-only", action="store_true", help="random row must be an attack")
    p.add_argument("--seed", type=int, default=None, help="repeat an earlier random choice")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 1_000_000)
    a.rng = np.random.default_rng(seed)
    a.random = a.index is None            # no --index => a random flow every run
    a.n_rounds = max(5, a.n_rounds)
    if a.csv and a.index is None:
        a.index = 0

    features = load_json(config.FEATURE_LIST_PATH)
    model = load_joblib(config.MODEL_PATH)
    classes = [str(c) for c in load_joblib(config.LABEL_ENCODER_PATH).classes_]
    booster = model.get_booster()
    n_cls, rounds = len(classes), booster.num_boosted_rounds()
    row_s, _, truth = pick_row(a, features, classes)
    x = row_s.iloc[0].to_numpy()

    d = xgb.DMatrix(row_s, feature_names=list(row_s.columns))
    margins = np.vstack([booster.predict(d, output_margin=True, iteration_range=(0, k)).reshape(-1) for k in range(1, rounds + 1)])
    e = np.exp(margins - margins.max(1, keepdims=True))
    probs = e / e.sum(1, keepdims=True)
    pred = int(np.argmax(probs[-1]))
    if a.rounds:
        shown = a.rounds
    else:   # always the first and last tree, the rest random -> different rounds every run
        k = min(a.n_rounds, rounds)
        mid = a.rng.choice(np.arange(2, rounds), size=max(0, k - 2), replace=False) if rounds > 2 else []
        shown = sorted({1, rounds, *[int(m) for m in mid]})
    shown = [r for r in shown if 1 <= r <= rounds]
    tdf = booster.trees_to_dataframe()

    paths = [(r, *walk(tdf, (r - 1) * n_cls + pred, x, features)) for r in shown]
    max_depth = max(len(pt) for _, pt, _ in paths)
    colw, n = 4.4, len(paths)
    H = max_depth * 1.05 + 5.0
    fig, ax = plt.subplots(figsize=(3.9 * n + 1, 0.62 * H + 1.2), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(-colw * 0.55, colw * (n - 0.45)); ax.set_ylim(-max_depth * 1.05 - 3.0, 2.0)

    prev_p = 1 / n_cls
    for i, (r, path, leaf) in enumerate(paths):
        cx = i * colw
        prob = probs[r - 1, pred]
        ax.text(cx, 1.55, f"Round {r} tree", color=FG, fontsize=12, fontweight="bold", ha="center")
        ax.text(cx, 1.15, f"class: {classes[pred]}", color=MUTED, fontsize=8.5, ha="center")
        col = CYAN if leaf >= 0 else AMBER
        lx, ly = draw_tree(ax, cx, 0.5, path, leaf, col)
        by = -max_depth * 1.05 - 0.9
        ax.plot([lx, cx], [ly - 0.25, by + 0.6], color=GRID, lw=1, ls=":")
        box(ax, cx, by, 3.7, 1.2,
            f"score after round {r}: {margins[r - 1, pred]:+.2f}\n"
            f"P({classes[pred]}) = {prob:.1%}\n"
            f"error still left: {1 - prob:.1%}",
            PANEL, GRID, fs=8.5)
        bar_y = by - 0.9
        ax.add_patch(plt.Rectangle((cx - 1.85, bar_y), 3.7, 0.16, fc=GRID, ec="none"))
        ax.add_patch(plt.Rectangle((cx - 1.85, bar_y), 3.7 * prob, 0.16, fc=CYAN, ec="none"))
        if i < n - 1:
            gap = "…" if paths[i + 1][0] - r > 1 else ""
            ax.add_patch(FancyArrowPatch((cx + 1.95, by), (cx + colw - 1.95, by), arrowstyle="-|>", mutation_scale=14, color=MUTED, lw=1.4))
            ax.text(cx + colw / 2, by + 0.22, ("next trees" if gap else "next tree") + "\ncorrect the error", color=MUTED, fontsize=7.5, ha="center", va="bottom")
        prev_p = prob

    title = f"XGBoost boosting, tree by tree · {classes[pred]} ({probs[-1, pred]:.1%} after {rounds} rounds)"
    if truth:
        title += f" · true label: {truth}"
    fig.suptitle(title, color=FG, fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.005, "Highlighted path = how this flow moves through each tree. Dashed grey = branch not taken. Cyan leaf = pushes towards this class, amber = pushes away. Leaf values add up (log-odds) "
             "into the class score; softmax turns scores into probabilities. Feature values are scaled 0–1.", color=MUTED, fontsize=8)
    if a.out is None:
        a.out = os.path.join(config.OUTPUTS_DIR, f"boosting_trees_row{getattr(a, 'chosen_index', 0)}_seed{seed}.png")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=120, facecolor=BG, bbox_inches="tight")
    print(f"Saved {a.out}\nrow {getattr(a, 'chosen_index', '?')} | seed {seed} (repeat with --seed {seed}) | true label {truth} | predicted {classes[pred]} {probs[-1, pred]:.1%} | rounds drawn {shown}")


if __name__ == "__main__":
    main()