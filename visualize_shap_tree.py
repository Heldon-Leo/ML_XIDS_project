"""
visualize_shap_tree.py - a SHAP "explanation tree" for one flow (left -> right).

    prediction  ->  supports / opposes  ->  behaviour group  ->  feature (+ SHAP bar, value, meaning)

* Root: the predicted class and probability.
* Branch 1: features that pushed TOWARDS the prediction vs. features that pushed AGAINST it.
* Branch 2: the behaviour group (ports, timing, packet size, volume/rate, TCP flags, window).
* Leaves: the feature, its real (unscaled) value compared with the training median, and its SHAP value.
* Bottom line: starting score + all SHAP contributions = final score -> probability.
Line thickness = total SHAP of that branch. SHAP is in log-odds units of the predicted class.

    python visualize_shap_tree.py                     # random flow
    python visualize_shap_tree.py --attack-only
    python visualize_shap_tree.py --index 123 --top-k 14
    python visualize_shap_tree.py --csv capture/live_flows.csv --index 0
    python visualize_shap_tree.py --seed 4242         # repeat an earlier random pick
Output: outputs/shap_tree_row<N>_seed<S>.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle
from matplotlib.path import Path
import numpy as np
import xgboost as xgb

import config
from src.feature_stats import load_or_build_feature_stats
from src.utils import load_joblib, load_json
from visualize_boosting import pick_row

BG, FG, MUTED, GRID, PANEL = "#0b1220", "#e5edf7", "#94a3b8", "#1e2b45", "#111a2e"
SUP, OPP = "#f87171", "#22d3ee"      # supports prediction / opposes prediction

GROUP_RULES = [   # first match wins
    ("Ports & protocol", ["port", "protocol"]),
    ("TCP flags", ["flag", "fin ", "syn ", "rst ", "psh ", "ack ", "urg ", "cwe", "ece"]),
    ("Window / handshake", ["win byts", "act data", "seg size min"]),
    ("Timing (IAT / duration)", ["iat", "duration", "active", "idle"]),
    ("Packet size", ["pkt len", "pkt size", "seg size", "header len", "len max", "len min", "len mean", "len std", "len var"]),
    ("Volume & rate", ["pkts", "byts", "/s", "totlen", "subflow", "bulk", "blk", "down/up"]),
]


def group_of(name: str) -> str:
    n = name.lower() + " "
    for g, keys in GROUP_RULES:
        if any(k in n for k in keys):
            return g
    return "Other"


def fmt(v):
    if v is None:
        return "n/a"
    a = abs(v)
    if a >= 1e6 or (0 < a < 1e-3):
        return f"{v:.3g}"
    return f"{v:,.4g}" if a < 1000 else f"{v:,.0f}"


def vs_typical(v, st):
    if not st:
        return ""
    m, p01, p99 = st["median"], st["p01"], st["p99"]
    if v > p99:
        return "extremely high vs training data"
    if v < p01:
        return "extremely low vs training data"
    if m == 0:
        return "typical" if v == 0 else "above typical (median 0)"
    r = v / m
    if r > 2:
        return f"{r:.0f}x above typical" if r < 100 else "far above typical"
    if r < 0.5:
        return "well below typical"
    return "near typical"


def curve(ax, x0, y0, x1, y1, color, lw, alpha=0.9):
    mx = (x0 + x1) / 2
    ax.add_patch(PathPatch(Path([(x0, y0), (mx, y0), (mx, y1), (x1, y1)],
                                [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]),
                           fc="none", ec=color, lw=lw, alpha=alpha, capstyle="round", zorder=1))


def node(ax, x, y, w, h, text, ec, fs=9, fc=PANEL, bold=True, tc=FG):
    ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=fc, ec=ec, lw=1.6, zorder=3))
    ax.text(x + w / 2, y, text, color=tc, fontsize=fs, ha="center", va="center", fontweight="bold" if bold else "normal", zorder=4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--label", default=None)
    p.add_argument("--csv", default=None)
    p.add_argument("--attack-only", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    seed = a.seed if a.seed is not None else int(np.random.SeedSequence().entropy % 1_000_000)
    a.rng = np.random.default_rng(seed)
    a.random = a.index is None

    features = load_json(config.FEATURE_LIST_PATH)
    model = load_joblib(config.MODEL_PATH)
    scaler = load_joblib(config.SCALER_PATH)
    classes = [str(c) for c in load_joblib(config.LABEL_ENCODER_PATH).classes_]
    stats = load_or_build_feature_stats(features) or {}
    row_s, _, truth = pick_row(a, features, classes)
    raw = scaler.inverse_transform(row_s.to_numpy())[0]

    booster = model.get_booster()
    d = xgb.DMatrix(row_s, feature_names=list(row_s.columns))
    contrib = booster.predict(d, pred_contribs=True)[0]            # (classes, features+1)
    proba = model.predict_proba(row_s)[0]
    pred = int(np.argmax(proba))
    sv, bias = contrib[pred, :-1], float(contrib[pred, -1])
    margin = bias + float(sv.sum())

    order = np.argsort(-np.abs(sv))[: a.top_k]
    rest = np.setdiff1d(np.arange(len(sv)), order)
    rest_sum = float(sv[rest].sum())
    tree = {"sup": {}, "opp": {}}
    for j in order:
        tree["sup" if sv[j] > 0 else "opp"].setdefault(group_of(features[j]), []).append(j)

    # ---- vertical layout (top -> bottom) -------------------------------------------------
    ROW = 1.0
    y = 0.0
    layout = {}                     # (dir, group) -> [(feature j, y)]
    dir_y, grp_y = {}, {}
    for dkey in ("sup", "opp"):
        ys_d = []
        for g, js in sorted(tree[dkey].items(), key=lambda kv: -sum(abs(sv[j]) for j in kv[1])):
            js = sorted(js, key=lambda j: -abs(sv[j]))
            ys = []
            for j in js:
                layout[j] = y; ys.append(y); y -= ROW
            grp_y[(dkey, g)] = float(np.mean(ys)); ys_d.append(grp_y[(dkey, g)])
            y -= 0.25
        if ys_d:
            dir_y[dkey] = float(np.mean(ys_d))
        y -= 0.45
    root_y = float(np.mean(list(dir_y.values()))) if dir_y else 0.0
    ymin, ymax = y, 1.5

    H = (ymax - ymin) * 0.52 + 2.4
    fig, ax = plt.subplots(figsize=(17, H), facecolor=BG)
    ax.set_facecolor(BG); ax.axis("off")
    X_ROOT, X_DIR, X_GRP, X_FEAT, X_BAR = 0.0, 3.6, 7.2, 10.8, 17.6
    ax.set_xlim(-0.3, 25.6); ax.set_ylim(ymin - 2.4, ymax + 0.6)

    for x, t in ((X_ROOT, "PREDICTION"), (X_DIR, "DIRECTION"), (X_GRP, "BEHAVIOUR GROUP"), (X_FEAT, "FEATURE  (real value · vs typical)"), (X_BAR, "SHAP VALUE (log-odds)")):
        ax.text(x, ymax, t, color=MUTED, fontsize=9, fontweight="bold")

    total_abs = float(np.abs(sv[order]).sum()) or 1.0
    lw_of = lambda s: 1.2 + 11 * abs(s) / total_abs
    maxabs = float(np.abs(sv[order]).max()) or 1.0
    BAR_W = 7.2

    # root
    node(ax, X_ROOT, root_y, 3.0, 1.5, f"{classes[pred]}\n{proba[pred]:.1%}", SUP if classes[pred].lower() != "benign" else "#34d399", fs=12)
    for dkey, title, col in (("sup", "SUPPORTS\nthe prediction", SUP), ("opp", "OPPOSES\nthe prediction", OPP)):
        if dkey not in dir_y:
            continue
        dsum = sum(sv[j] for g in tree[dkey].values() for j in g)
        curve(ax, X_ROOT + 3.0, root_y, X_DIR, dir_y[dkey], col, lw_of(dsum))
        node(ax, X_DIR, dir_y[dkey], 2.9, 1.2, f"{title}\n{dsum:+.2f}", col, fs=10)
        for g, js in tree[dkey].items():
            gsum = sum(sv[j] for j in js)
            curve(ax, X_DIR + 2.9, dir_y[dkey], X_GRP, grp_y[(dkey, g)], col, lw_of(gsum))
            node(ax, X_GRP, grp_y[(dkey, g)], 3.0, 0.9, f"{g}\n{gsum:+.2f}", col, fs=9)
            for j in js:
                curve(ax, X_GRP + 3.0, grp_y[(dkey, g)], X_FEAT, layout[j], col, lw_of(sv[j]), 0.75)
                st = stats.get(features[j])
                ax.add_patch(FancyBboxPatch((X_FEAT, layout[j] - 0.4), 6.3, 0.8, boxstyle="round,pad=0.02,rounding_size=0.08", fc=PANEL, ec=col, lw=1.2, zorder=3))
                ax.text(X_FEAT + 0.15, layout[j] + 0.14, features[j], color=FG, fontsize=9.5, fontweight="bold", va="center", zorder=4)
                ax.text(X_FEAT + 0.15, layout[j] - 0.17, f"{fmt(raw[j])}  ·  {vs_typical(raw[j], st)}", color=MUTED, fontsize=8, va="center", zorder=4)
                w = BAR_W * abs(sv[j]) / maxabs
                ax.add_patch(Rectangle((X_BAR, layout[j] - 0.16), BAR_W, 0.32, fc=GRID, ec="none", zorder=2))
                ax.add_patch(Rectangle((X_BAR, layout[j] - 0.16), w, 0.32, fc=col, ec="none", zorder=3))
                ax.text(X_BAR + BAR_W + 0.15, layout[j], f"{sv[j]:+.3f}", color=FG, fontsize=9, va="center")

    # bottom: how the score is assembled
    by = ymin - 1.0
    ax.add_patch(FancyBboxPatch((0, by - 0.7), 24.6, 1.4, boxstyle="round,pad=0.02,rounding_size=0.1", fc=PANEL, ec=GRID, lw=1.2))
    ax.text(0.3, by + 0.22, "How the score adds up", color=FG, fontsize=10.5, fontweight="bold", va="center")
    ax.text(0.3, by - 0.22,
            f"starting score {bias:+.2f}   +   top {len(order)} features {float(sv[order].sum()):+.2f}   +   other {len(rest)} features {rest_sum:+.2f}"
            f"   =   final score {margin:+.2f}   →   softmax over 15 classes   →   P({classes[pred]}) = {proba[pred]:.1%}",
            color=MUTED, fontsize=9.5, va="center")

    title = f"SHAP explanation tree · predicted {classes[pred]} ({proba[pred]:.1%})"
    if truth:
        title += f" · true label: {truth}"
    fig.suptitle(title, color=FG, fontsize=15, fontweight="bold", x=0.02, ha="left", y=0.995)
    fig.text(0.02, 0.003, "Red = pushes towards the predicted class, cyan = pushes away from it. SHAP values are in log-odds for the predicted class; "
             "line thickness = size of the branch's total SHAP. 'Typical' = training-set median.", color=MUTED, fontsize=8.5)
    if a.out is None:
        a.out = os.path.join(config.OUTPUTS_DIR, f"shap_tree_row{getattr(a, 'chosen_index', 0)}_seed{seed}.png")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    fig.savefig(a.out, dpi=115, facecolor=BG, bbox_inches="tight")
    print(f"Saved {a.out}\nrow {getattr(a, 'chosen_index', '?')} | seed {seed} (repeat with --seed {seed}) | true label {truth} | predicted {classes[pred]} {proba[pred]:.1%}")


if __name__ == "__main__":
    main()