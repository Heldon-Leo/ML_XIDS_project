"""
explain_simple.py - explain 5 (or 10) random network connections to someone who knows NOTHING about cybersecurity.

    python explain_simple.py                 # 5 random connections (a mix of normal and attacks)
    python explain_simple.py --n 10          # 10 connections
    python explain_simple.py --seed 7        # repeat an earlier pick
    python explain_simple.py --pure-random   # no guaranteed mix
    python explain_simple.py --label "DoS attacks-Hulk"   # that attack type (plus normal ones)

Output (outputs/simple_explain/seed_<n>/): one story page per connection + summary.png
Story page: 1 What the connection looked like | 2 How the answer builds up, team by team |
3 One decision tree, drawn question by question | 4 The arithmetic done by hand: points -> percentages |
5 SHAP: why, as a tug of war.
Needs the files from `python main.py` (model, scaler, label encoder, feature list, cached split).
The hand arithmetic is checked against XGBoost's own output on every page.
"""
import argparse
import os
import textwrap
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
import xgboost as xgb

import config
from src.utils import load_joblib, load_json

INK, MUTED, LINE, BG = "#1f2937", "#6b7280", "#e5e7eb", "#ffffff"
TOWARD, AWAY, BENIGN_C, ATTACK_C, SOFT = "#ea580c", "#2563eb", "#16a34a", "#dc2626", "#f3f4f6"

# plain label, unit, what it means  (all 70 model features)
F = {
 "Dst Port": ("Door number on the target", "", "Computers have numbered 'doors' (ports). 80 = websites, 22 = remote login, 21 = file transfer."),
 "Protocol": ("Type of language used", "", "6 = TCP (reliable, like a phone call), 17 = UDP (fire and forget, like a postcard)."),
 "Flow Duration": ("How long the connection lasted", "microseconds", "Very short = a quick knock; very long = someone staying connected."),
 "Tot Fwd Pkts": ("Messages sent to the target", "", "A packet is one small message."),
 "Tot Bwd Pkts": ("Messages the target sent back", "", "Replies from the target."),
 "TotLen Fwd Pkts": ("Total size of messages sent", "bytes", "How much data went towards the target."),
 "TotLen Bwd Pkts": ("Total size of replies", "bytes", "How much data came back."),
 "Fwd Pkt Len Max": ("Biggest message sent", "bytes", ""), "Fwd Pkt Len Min": ("Smallest message sent", "bytes", ""),
 "Fwd Pkt Len Mean": ("Average message size sent", "bytes", ""), "Fwd Pkt Len Std": ("How much sent message sizes vary", "bytes", "Low = all messages look alike (robot-like)."),
 "Bwd Pkt Len Max": ("Biggest reply", "bytes", ""), "Bwd Pkt Len Min": ("Smallest reply", "bytes", ""),
 "Bwd Pkt Len Mean": ("Average reply size", "bytes", ""), "Bwd Pkt Len Std": ("How much reply sizes vary", "bytes", ""),
 "Flow Byts/s": ("Data speed", "bytes/second", "How fast data moved."), "Flow Pkts/s": ("Message speed", "messages/second", "How many messages per second. Floods are very fast."),
 "Flow IAT Mean": ("Average pause between messages", "microseconds", "IAT = inter-arrival time, the gap between two messages."),
 "Flow IAT Std": ("How much the pauses vary", "microseconds", ""), "Flow IAT Max": ("Longest pause", "microseconds", ""), "Flow IAT Min": ("Shortest pause", "microseconds", ""),
 "Fwd IAT Tot": ("Total time spent sending", "microseconds", ""), "Fwd IAT Mean": ("Average pause between sent messages", "microseconds", ""),
 "Fwd IAT Std": ("Variation in pauses (sending)", "microseconds", ""), "Fwd IAT Max": ("Longest pause while sending", "microseconds", ""), "Fwd IAT Min": ("Shortest pause while sending", "microseconds", ""),
 "Bwd IAT Tot": ("Total time spent replying", "microseconds", ""), "Bwd IAT Mean": ("Average pause between replies", "microseconds", ""),
 "Bwd IAT Std": ("Variation in pauses (replies)", "microseconds", ""), "Bwd IAT Max": ("Longest pause between replies", "microseconds", ""), "Bwd IAT Min": ("Shortest pause between replies", "microseconds", ""),
 "Fwd PSH Flags": ("'Send now' flags on sent messages", "", ""), "Fwd URG Flags": ("'Urgent' flags on sent messages", "", ""),
 "Fwd Header Len": ("Envelope size of sent messages", "bytes", "Each message has an envelope (header) with addressing details."),
 "Bwd Header Len": ("Envelope size of replies", "bytes", ""), "Fwd Pkts/s": ("Messages sent per second", "per second", ""), "Bwd Pkts/s": ("Replies per second", "per second", ""),
 "Pkt Len Min": ("Smallest message overall", "bytes", ""), "Pkt Len Max": ("Biggest message overall", "bytes", ""), "Pkt Len Mean": ("Average message size", "bytes", ""),
 "Pkt Len Std": ("Variation in message size", "bytes", ""), "Pkt Len Var": ("Spread of message sizes", "", ""),
 "FIN Flag Cnt": ("'Goodbye' signals", "", "A polite end of a connection."), "SYN Flag Cnt": ("'Hello' signals", "", "Used to start a connection."),
 "RST Flag Cnt": ("'Reset' signals", "", "Abruptly cutting a connection off."), "PSH Flag Cnt": ("'Send now' signals", "", ""),
 "ACK Flag Cnt": ("'Got it' signals", "", ""), "URG Flag Cnt": ("'Urgent' signals", "", ""), "CWE Flag Count": ("Congestion flag (CWE)", "", ""), "ECE Flag Cnt": ("Congestion flag (ECE)", "", ""),
 "Down/Up Ratio": ("Download-to-upload ratio", "", ""), "Pkt Size Avg": ("Average message size (overall)", "bytes", ""),
 "Fwd Seg Size Avg": ("Average chunk size sent", "bytes", ""), "Bwd Seg Size Avg": ("Average chunk size replied", "bytes", ""),
 "Subflow Fwd Pkts": ("Messages sent (per sub-part)", "", ""), "Subflow Fwd Byts": ("Bytes sent (per sub-part)", "bytes", ""),
 "Subflow Bwd Pkts": ("Replies (per sub-part)", "", ""), "Subflow Bwd Byts": ("Bytes replied (per sub-part)", "bytes", ""),
 "Init Fwd Win Byts": ("Sender's starting 'buffer' size", "bytes", "How much data the sender says it can receive at once when the call starts. Different attack tools set unusual values."),
 "Init Bwd Win Byts": ("Target's starting 'buffer' size", "bytes", ""), "Fwd Act Data Pkts": ("Messages that carried real data", "", "Not just 'hello' or 'got it'."),
 "Fwd Seg Size Min": ("Smallest chunk size the sender uses", "bytes", "A fingerprint of the software that made the connection."),
 "Active Mean": ("Average busy time", "microseconds", ""), "Active Std": ("Variation in busy time", "microseconds", ""), "Active Max": ("Longest busy spell", "microseconds", ""), "Active Min": ("Shortest busy spell", "microseconds", ""),
 "Idle Mean": ("Average quiet time", "microseconds", ""), "Idle Std": ("Variation in quiet time", "microseconds", ""), "Idle Max": ("Longest quiet spell", "microseconds", ""), "Idle Min": ("Shortest quiet spell", "microseconds", ""),
}
CLASS_PLAIN = {
 "benign": "Normal, harmless traffic (browsing, email, updates).",
 "bot": "A hijacked computer taking orders from a remote controller.",
 "brute force -web": "Guessing passwords on a website again and again.",
 "brute force -xss": "Guessing passwords while also trying to inject hidden scripts into a website.",
 "ddos attack-hoic": "Many machines flooding a website with traffic so real visitors cannot get in.",
 "ddos attack-loic-udp": "Many machines flooding a target with a stream of 'postcard' (UDP) messages.",
 "ddos attacks-loic-http": "Many machines flooding a website with page requests.",
 "dos attacks-goldeneye": "One machine overloading a website with requests built to look like real visitors.",
 "dos attacks-hulk": "One machine flooding a website with huge numbers of unique page requests.",
 "dos attacks-slowhttptest": "Holding a website's connections open very slowly so nobody else fits.",
 "dos attacks-slowloris": "Opening many connections and never finishing them, using up the site's capacity.",
 "ftp-bruteforce": "Guessing passwords on a file-transfer server again and again.",
 "infilteration": "An intruder quietly inside the network, trying to look like normal traffic.",
 "sql injection": "Sending trick text to a website to make its database reveal or change data.",
 "ssh-bruteforce": "Guessing passwords on a remote-login service again and again.",
}
def plain(f): return F.get(f, (f, "", ""))[0]
def cdesc(c): return CLASS_PLAIN.get(c.lower(), "A type of malicious traffic.")
def is_benign(c): return c.lower() == "benign"
def level(s):
    return "very low" if s < .1 else "low" if s < .35 else "medium" if s < .65 else "high" if s < .9 else "very high"
def fmt(v):
    if abs(v) >= 1e7 or (0 < abs(v) < 1e-3): return f"{v:.2e}"
    return f"{v:,.0f}" if abs(v - round(v)) < 1e-9 or abs(v) >= 1000 else f"{v:.3g}"
def wrap(t, w): return "\n".join(textwrap.wrap(t, w)) or ""



def tree_lookup(booster):
    df = booster.trees_to_dataframe(); lf = df[df["Feature"] == "Leaf"]
    return {(int(t), int(n)): float(g) for t, n, g in zip(lf["Tree"], lf["Node"], lf["Gain"])}


def tree_path(booster, tree_idx, xs, features, df):
    t = df[df["Tree"] == tree_idx].set_index("ID")
    node, steps, ids = f"{tree_idx}-0", [], [f"{tree_idx}-0"]
    while t.loc[node, "Feature"] != "Leaf":
        r = t.loc[node]; name = r["Feature"]
        if name not in features and name[:1] == "f" and name[1:].isdigit():
            name = features[int(name[1:])]
        v = float(xs[features.index(name)]); yes = v < r["Split"]
        steps.append((name, v, float(r["Split"]), yes))
        node = r["Yes"] if yes else r["No"]; ids.append(node)
    return steps, float(t.loc[node, "Gain"]), ids


def to_raw(scaler, thr, j):
    try: return (thr - scaler.min_[j]) / scaler.scale_[j]
    except Exception: return None


def softmax(m):
    e = np.exp(m - m.max()); return e / e.sum()


def analyse(booster, features, classes, xs, leafmap, tdf):
    n_cls, rounds = len(classes), booster.num_boosted_rounds()
    dm = xgb.DMatrix(xs.reshape(1, -1), feature_names=booster.feature_names)
    margin = booster.predict(dm, output_margin=True).reshape(-1)
    leaf_ids = booster.predict(dm, pred_leaf=True).reshape(rounds, n_cls)
    leaves = np.array([[leafmap[(r * n_cls + k, int(leaf_ids[r, k]))] for k in range(n_cls)] for r in range(rounds)])
    start = margin - leaves.sum(0)                                   # the model's starting score per type
    run = start[None, :] + np.cumsum(leaves, 0)                        # running total after each team
    final = softmax(run[-1]); pred = int(np.argmax(final))
    diff = float(np.abs(run[-1] - margin).max())
    cps = sorted({1, min(3, rounds), min(10, rounds), min(50, rounds), min(150, rounds), rounds})
    probs = np.array([softmax(run[k - 1]) for k in cps])
    contr = booster.predict(dm, pred_contribs=True).reshape(n_cls, -1)[pred]
    later = int(np.argmax(np.abs(leaves[1:, pred]))) + 1 if rounds > 1 else 0
    trees = [(r, r * n_cls + pred, *tree_path(booster, r * n_cls + pred, xs, features, tdf)) for r in dict.fromkeys([0, later])]
    return dict(cps=cps, probs=probs, final=final, pred=pred, contr=contr, trees=trees, leaves=leaves,
                start=start, run=run, diff=diff, rounds=rounds)


class Page:
    def __init__(self, H):
        self.H = H; self.fig = plt.figure(figsize=(12, H), facecolor="#f9fafb")
    def card(self, top, h, title, sub):
        ax = self.fig.add_axes([.04, 1 - (top + h) / self.H, .92, h / self.H]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.008", fc=BG, ec=LINE, lw=1.2, transform=ax.transAxes))
        ax.text(.015, 1 - .25 / h, title, fontsize=15, fontweight="bold", color=INK, va="top")
        ax.text(.015, 1 - .75 / h, sub, fontsize=10.5, color=MUTED, va="top", linespacing=1.35)
        return ax
    def sub(self, top, h, x0, y_in, w, h_in):
        return self.fig.add_axes([x0, 1 - (top + y_in + h_in) / self.H, w, h_in / self.H])


ITER = "iteration"   # word used for one boosting step (one tree per class)


def build_view(tdf, tree_idx, path_ids, features):
    """Pruned view of a real tree: every node on the decision path, plus the sibling of each path node."""
    t = tdf[tdf["Tree"] == tree_idx].set_index("ID")

    def nm(n):
        f = t.loc[n, "Feature"]
        if f not in features and f[:1] == "f" and f[1:].isdigit(): f = features[int(f[1:])]
        return f

    def nleaves(n):
        return 1 if t.loc[n, "Feature"] == "Leaf" else nleaves(t.loc[n, "Yes"]) + nleaves(t.loc[n, "No"])

    def rec(n, on):
        r = t.loc[n]; leaf = r["Feature"] == "Leaf"
        d = dict(id=n, on=on, leaf=leaf, ch=[])
        if leaf: d["w"] = float(r["Gain"])
        else:
            d["f"] = nm(n); d["thr"] = float(r["Split"])
            if on:
                d["ch"] = [rec(r["Yes"], r["Yes"] in path_ids), rec(r["No"], r["No"] in path_ids)]
            else:
                d["collapsed"] = nleaves(n)
        return d
    return rec(f"{tree_idx}-0", True)


def draw_full_tree(ax, root, xs, raw_vals, features, scaler, pc, col):
    ax.axis("off")
    slot = [0]; nodes = []

    def lay(d, depth):
        if not d["ch"]:
            d["x"] = slot[0]; slot[0] += 1
        else:
            for c in d["ch"]: lay(c, depth + 1)
            d["x"] = np.mean([c["x"] for c in d["ch"]])
        d["y"] = depth; nodes.append(d)
    lay(root, 0)
    D = max(n["y"] for n in nodes); W = slot[0]
    ax.set_xlim(-.6, W - .4); ax.set_ylim(-(D + .7), .7)
    bw, bh = .94, .80
    for d in nodes:
        for c in d["ch"]:
            yes = c is d["ch"][0]; hot = c["on"]
            cc = (TOWARD if yes else AWAY) if hot else LINE
            ax.plot([d["x"], c["x"]], [-d["y"] - bh / 2, -c["y"] + bh / 2], color=cc, lw=2.6 if hot else 1.1, zorder=1, solid_capstyle="round")
            mx, my = (d["x"] + c["x"]) / 2, (-d["y"] - c["y"]) / 2
            ax.text(mx, my, "yes" if yes else "no", fontsize=10 if hot else 8.5, fontweight="bold" if hot else None, color=cc if hot else MUTED,
                    ha="center", va="center", zorder=3, bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    for d in nodes:
        x, y = d["x"], -d["y"]
        if d["leaf"]:
            if d["on"]:
                ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.1", fc=col, ec=col, lw=2, zorder=4))
                ax.text(x, y + .09, "LEAF", fontsize=10, fontweight="bold", color="white", ha="center", va="center", zorder=5)
                ax.text(x, y - .17, f"w = {d['w']:+.3f}", fontsize=10.5, fontweight="bold", color="white", ha="center", va="center", zorder=5)
            else:
                ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2 + .15), bw, bh - .3, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#ecfdf5", ec=LINE, lw=1, zorder=2))
                ax.text(x, y, f"leaf w={d['w']:+.3f}", fontsize=8.5, color=MUTED, ha="center", va="center", zorder=3)
            continue
        j = features.index(d["f"]); rt = to_raw(scaler, d["thr"], j)
        thr = fmt(rt) if rt is not None else f"{d['thr']:.4f}"
        if d["on"]:
            yes = raw_vals[j] is not None and xs[j] < d["thr"]
            ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2), bw, bh, boxstyle="round,pad=0.02,rounding_size=0.08", fc="#fff7ed" if d["id"].endswith("-0") is False else "#fef3c7", ec=TOWARD, lw=1.8, zorder=4))
            ax.text(x, y + .22, ("ROOT: " if d["id"].endswith("-0") else "") + wrap(d["f"], 13), fontsize=9.5, fontweight="bold", color=INK, ha="center", va="center", zorder=5, linespacing=1.0)
            ax.text(x, y - .04, f"< {thr} ?", fontsize=9.5, color=INK, ha="center", va="center", zorder=5)
            ax.text(x, y - .27, f"x = {fmt(raw_vals[j])}", fontsize=9.5, color=TOWARD if yes else AWAY, fontweight="bold", ha="center", va="center", zorder=5)
        else:
            ax.add_patch(FancyBboxPatch((x - bw / 2, y - bh / 2 + .02), bw, bh - .04, boxstyle="round,pad=0.02,rounding_size=0.08", fc=SOFT, ec=LINE, lw=1, ls="--", zorder=2))
            ax.text(x, y + .12, wrap(d["f"], 15) + f"\n< {thr}", fontsize=8, color=MUTED, ha="center", va="center", zorder=3)
            ax.text(x, y - .24, f"subtree: {d['collapsed']} leaves", fontsize=8, color=MUTED, ha="center", va="center", style="italic", zorder=3)


tdf_global = [None]


def story(a, classes, features, scaler, raw_vals, xs, truth, idx, out):
    pred = a["pred"]; pc = classes[pred]; conf = a["final"][pred]; attack = not is_benign(pc)
    col = ATTACK_C if attack else BENIGN_C; R = a["rounds"]; n_cls = len(classes)
    TH = 9.6; nT = len(a["trees"])
    H = 2.6 + 4.2 + nT * (TH + .2) + 7.4 + .3
    pg = Page(H); fig = pg.fig
    fig.text(.04, 1 - .4 / H, f"Flow #{idx}: XGBoost classification", fontsize=22, fontweight="bold", color=INK, va="top")
    ax = fig.add_axes([.04, 1 - 2.4 / H, .92, 1.5 / H]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", fc=col, ec="none", alpha=.12))
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.03", fc="none", ec=col, lw=2))
    ax.text(.02, .68, f"{'ATTACK' if attack else 'BENIGN'}:  {pc}   (p = {conf:.3f})", fontsize=19, fontweight="bold", color=col, va="center")
    ax.text(.02, .25, f"softmax over {n_cls} classes, {R} boosting iterations ({R * n_cls} trees)" + (f"   |   ground truth: {truth}" if truth else ""), fontsize=12, color=INK, va="center")

    # STEP 1  features
    top = 2.6; h = 4.2
    ax = pg.card(top, h, "STEP 1  -  Input features", "Top 6 features by |SHAP|. Orange = pushes towards the predicted class, blue = away.")
    order = np.argsort(np.abs(a["contr"][:-1]))[::-1][:6]
    for xx, t in ((.02, "Feature"), (.37, "Raw value"), (.55, "Scaled x [0,1]"), (.85, "SHAP")):
        ax.text(xx, .66, t, fontsize=12, fontweight="bold", color=MUTED)
    for r, j in enumerate(order):
        y = .53 - r * .095
        ax.text(.02, y, features[j], fontsize=13, fontweight="bold", color=INK, va="center")
        ax.text(.37, y, fmt(raw_vals[j]), fontsize=13, color=INK, va="center")
        sv = float(np.clip(xs[j], 0, 1)); ax.add_patch(plt.Rectangle((.55, y - .025), .24, .05, fc=LINE, ec="none"))
        ax.add_patch(plt.Rectangle((.55, y - .025), .24 * sv, .05, fc=TOWARD if a["contr"][j] > 0 else AWAY, ec="none"))
        ax.text(.85, y, f"{a['contr'][j]:+.2f}", fontsize=13, color=TOWARD if a["contr"][j] > 0 else AWAY, va="center", fontweight="bold")

    # STEP 2  full trees
    top += h + .1
    for t_i, (rnd, tidx, steps, leaf, ids) in enumerate(a["trees"]):
        ax = pg.card(top, TH, f"STEP 2.{t_i + 1}  -  Tree {tidx}  (class '{pc}', {ITER} {rnd + 1})",
                     f"Root to leaf: coloured path = this flow ({len(steps)} splits). Grey dashed = branches not taken.\n"
                     + ("First tree of the ensemble." if t_i == 0 else "Tree with the largest leaf value for this class."))
        tx = pg.sub(top, TH, .03, 1.25, .94, TH - 1.9)
        draw_full_tree(tx, build_view(tdf_global[0], tidx, ids, features), xs, raw_vals, features, scaler, pc, col)
        ax.text(.5, .02, f"Leaf value w = {leaf:+.4f} (log-odds) is added to F_{pc}", fontsize=12, ha="center", color=col, fontweight="bold")
        top += TH + .2

    # STEP 3  SHAP (plain language)
    h = 7.4
    ax = pg.card(top, h, "STEP 3  -  Why did the AI say this? (SHAP)",
                 "Each clue pulls the score towards the AI's answer (orange) or away from it (blue).\nStart at the neutral score, add every pull, and you reach the final score.")
    c = a["contr"]; idxs = np.argsort(np.abs(c[:-1]))[::-1][:6]
    rows = [("Starting point", c[-1])] + [(plain(features[j]) + f" = {fmt(raw_vals[j])}", c[j]) for j in idxs]
    rows.append((f"All {len(features) - 6} other clues", c[:-1].sum() - c[idxs].sum()))
    cum = 0; ys = np.arange(len(rows))[::-1]
    wf = pg.sub(top, h, .33, 1.5, .60, 3.3); wf.set_facecolor("white")
    for y, (lab, v) in zip(ys, rows):
        wf.barh(y, v, left=cum, color=(MUTED if lab.startswith("Start") else TOWARD if v > 0 else AWAY), height=.62)
        right = v >= 0 or lab.startswith("Start")
        wf.text(max(cum, cum + v) + .08 if right else cum + v - .08, y, f"{v:+.1f}", fontsize=11, va="center", ha="left" if right else "right", color=INK)
        cum += v
    wf.barh(-1, cum, color=col, height=.62); wf.text(cum + .08, -1, f"{cum:.1f}", fontsize=12, fontweight="bold", va="center", color=col)
    wf.set_yticks(list(ys) + [-1]); wf.set_yticklabels([wrap(r[0], 26) for r in rows] + ["FINAL SCORE"], fontsize=10.5)
    wf.axvline(0, color=MUTED, lw=.8)
    for s_ in ("top", "right"): wf.spines[s_].set_visible(False)
    wf.tick_params(axis="x", labelsize=10)
    ups = [(plain(features[j]), c[j]) for j in idxs if c[j] > 0][:3]; dn = [(plain(features[j]), c[j]) for j in idxs if c[j] < 0][:2]
    t = f"In plain English: the AI says '{pc}' mainly because of " + (", ".join(f"'{n.lower()}' (+{v:.1f})" for n, v in ups) or "no single strong clue") + "."
    if dn: t += " Pulling the other way: " + ", ".join(f"'{n.lower()}' ({v:.1f})" for n, v in dn) + "."
    j0 = idxs[0]; t += f" Without the biggest clue ('{plain(features[j0]).lower()}'), the score would drop by about {abs(c[j0]):.1f} points."
    ax.text(.03, .26, wrap(t, 88), fontsize=12.5, color=INK, va="top", linespacing=1.45)
    ax.text(.03, .035, wrap("SHAP shows what the AI relied on. It is a hint for a human analyst, not proof that the connection is malicious.", 100), fontsize=10.5, color=MUTED, va="bottom", style="italic")
    fig.savefig(out, dpi=110, facecolor=fig.get_facecolor()); plt.close(fig)


def summary(items, out):
    fig = plt.figure(figsize=(12, 2.2 + 1.5 * len(items)), facecolor="#f9fafb")
    fig.text(.04, .98, f"{len(items)} random connections: what the AI decided and why", fontsize=19, fontweight="bold", color=INK, va="top")
    ax = fig.add_axes([.04, .02, .92, .88]); ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    for i, (idx, verdict, conf, truth, why, attack) in enumerate(items):
        y = 1 - (i + .5) / len(items); c = ATTACK_C if attack else BENIGN_C; hh = .42 / len(items)
        ax.add_patch(FancyBboxPatch((0, y - hh), 1, 2 * hh, boxstyle="round,pad=0,rounding_size=0.01", fc="white", ec=LINE))
        ax.add_patch(plt.Rectangle((0, y - hh), .008, 2 * hh, fc=c, ec="none"))
        ax.text(.02, y + hh * .3, f"#{idx}  {'ATTACK' if attack else 'NORMAL'}: {verdict}", fontsize=13, fontweight="bold", color=c, va="center")
        ax.text(.02, y - hh * .35, f"{conf:.0%} sure" + (f"   |   real answer: {truth}" if truth else ""), fontsize=10, color=MUTED, va="center")
        ax.text(.38, y, wrap(why, 80), fontsize=10.5, color=INK, va="center", linespacing=1.35)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor()); plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=5); p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pure-random", action="store_true"); p.add_argument("--label", default=None)
    p.add_argument("--out", default=os.path.join(config.OUTPUTS_DIR, "simple_explain"))
    a = p.parse_args()
    seed = a.seed if a.seed is not None else int(time.time()) % 100000
    rng = np.random.default_rng(seed)
    features = load_json(config.FEATURE_LIST_PATH); model = load_joblib(config.MODEL_PATH)
    classes = [str(c) for c in load_joblib(config.LABEL_ENCODER_PATH).classes_]; scaler = load_joblib(config.SCALER_PATH)
    booster = model.get_booster(); tdf = booster.trees_to_dataframe(); tdf_global[0] = tdf; leafmap = tree_lookup(booster)
    data = load_joblib(config.SPLIT_DATA_CACHE_PATH)
    Xs = np.asarray(data.X_test, dtype=float); y = np.asarray(data.y_test)
    raw = getattr(data, "X_test_raw", None)
    raw = raw.reset_index(drop=True) if isinstance(raw, pd.DataFrame) and len(raw) == len(Xs) else None
    ben = next(i for i, c in enumerate(classes) if is_benign(c))
    pool_b = np.where(y == ben)[0]
    pool_a = np.where(y != ben)[0] if not a.label else np.where(y == classes.index(a.label))[0]
    if a.pure_random: picks = rng.choice(len(Xs), a.n, replace=False)
    else:
        nb = max(1, a.n // 3); picks = np.concatenate([rng.choice(pool_b, nb, replace=False), rng.choice(pool_a, a.n - nb, replace=False)]); rng.shuffle(picks)
    out_dir = os.path.join(a.out, f"seed_{seed}"); os.makedirs(out_dir, exist_ok=True)
    items = []
    for n, i in enumerate(picks, 1):
        xs = Xs[i]
        rawv = raw[features].iloc[i].to_numpy(dtype=float) if raw is not None else scaler.inverse_transform(xs.reshape(1, -1))[0]
        an = analyse(booster, features, classes, xs, leafmap, tdf)
        pc = classes[an["pred"]]; truth = classes[int(y[i])]
        path = os.path.join(out_dir, f"flow_{n}_{pc.replace(' ', '_').replace('/', '_')}.png")
        story(an, classes, features, scaler, rawv, xs, truth, n, path)
        c = an["contr"]; top = np.argsort(np.abs(c[:-1]))[::-1][:2]
        why = "SHAP says: " + "; ".join(f"{plain(features[j]).lower()} = {fmt(rawv[j])} ({'for' if c[j] > 0 else 'against'})" for j in top) + "."
        items.append((n, pc, an["final"][an["pred"]], truth, why, not is_benign(pc)))
        print(f"#{n} test row {i}: AI says {pc} ({an['final'][an['pred']]:.0%}), true {truth}, hand-sum check diff {an['diff']:.5f} -> {path}")
    sp = os.path.join(out_dir, "summary.png"); summary(items, sp); print("Summary:", sp)


if __name__ == "__main__":
    main()