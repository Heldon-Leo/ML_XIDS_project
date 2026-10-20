"""
walkthrough.py  -  Learn how XGBoost TRAINS and how SHAP EXPLAINS, on a tiny dataset.

Everything is computed BY HAND (plain numpy), printed stage by stage, and then
checked against the real xgboost / shap libraries.

    python walkthrough.py                      # built-in 15-flow toy dataset
    python walkthrough.py --sample 3           # explain a different row
    python walkthrough.py --rounds 3 --depth 2 # more boosting rounds
    python walkthrough.py --csv data/ --classes Benign "DoS attacks-Hulk" Bot --per-class 5
                                               # tiny sample from the real CSE-CIC-IDS2018 CSVs

Stages printed (also saved to outputs/walkthrough/walkthrough_report.txt):
  0 raw input      1 scaling      2 how boosting works (formulas)
  3 hand-training  round by round: gradients -> split search -> tree -> leaf weights -> new scores
  4 check          hand-built trees vs. XGBoost's own trees
  5 prediction     one sample walked through every tree -> softmax
  6 SHAP by hand   every feature coalition, Shapley weights, additivity
  7 check + plain-English reason, then global importance
"""
import argparse
import glob
import itertools
import json
import os
import sys
from math import factorial

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# ----------------------------------------------------------------------------
# small printing helpers (ASCII only so Windows consoles never break)
# ----------------------------------------------------------------------------
class Tee:
    def __init__(self, *s): self.s = s
    def write(self, x): [t.write(x) for t in self.s]
    def flush(self): [t.flush() for t in self.s]


def H1(t): print("\n" + "=" * 90 + f"\n{t}\n" + "=" * 90)
def H2(t): print(f"\n--- {t} ---")


def rv(v):
    """compact number formatting for raw feature values"""
    v = float(v)
    if abs(v) >= 1000: return f"{v:,.0f}"
    if abs(v - round(v)) < 1e-9: return str(int(round(v)))
    return f"{v:.4g}"


# ----------------------------------------------------------------------------
# DATA: built-in toy flows, or a tiny sample of the real CSVs
# ----------------------------------------------------------------------------
TOY_FEATURES = ["Flow Duration", "Tot Fwd Pkts", "Flow Pkts/s", "SYN Flag Cnt"]
TOY_ROWS = [  # duration(us), fwd packets, packets/s, SYN count, label
    (2_400_000, 24, 10.0, 0, "Benign"), (850_000, 12, 14.1, 0, "Benign"), (5_100_000, 40, 7.8, 1, "Benign"),
    (60_000, 12, 200.0, 1, "Benign"), (1_700_000, 18, 10.6, 1, "Benign"),
    (310_000, 95, 306.5, 1, "DoS-Hulk"), (150_000, 70, 466.7, 1, "DoS-Hulk"), (890_000, 160, 179.8, 2, "DoS-Hulk"),
    (2_800_000, 150, 53.6, 2, "DoS-Hulk"), (60_000, 45, 750.0, 2, "DoS-Hulk"),
    (900, 2, 2222.2, 1, "PortScan"), (1_200, 2, 1666.7, 1, "PortScan"), (350, 1, 2857.1, 1, "PortScan"),
    (20_000, 4, 200.0, 1, "PortScan"), (700, 2, 2857.1, 1, "PortScan"),
]


def toy_data():
    df = pd.DataFrame(TOY_ROWS, columns=TOY_FEATURES + ["Label"])
    return df, TOY_FEATURES, 8      # row 8 = a slow DoS: interesting, features disagree


def csv_data(path, classes, per_class, features, n_feat, seed):
    """Collect a few rows of each requested class from the real CSV(s)."""
    files = sorted(glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)) if os.path.isdir(path) else [path]
    if not files: sys.exit(f"No CSV files found at {path}")
    pool = {c.lower(): [] for c in classes}
    for f in files:
        for ch in pd.read_csv(f, chunksize=200_000, low_memory=False):
            ch.columns = [c.strip() for c in ch.columns]
            if "Label" not in ch.columns: break
            lab = ch["Label"].astype(str).str.strip().str.lower()
            for c in pool:
                if sum(len(d) for d in pool[c]) < 300 and (lab == c).any():
                    pool[c].append(ch[lab == c].head(300))
            if all(sum(len(d) for d in v) >= 300 for v in pool.values()): break
        if all(sum(len(d) for d in v) >= 300 for v in pool.values()): break
    parts = []
    for c, orig in zip(pool, classes):
        if not pool[c]: sys.exit(f"Class '{orig}' not found. Check spelling against the Label column.")
        d = pd.concat(pool[c]).head(300)
        parts.append(d.sample(n=min(per_class, len(d)), random_state=seed))
    df = pd.concat(parts, ignore_index=True)
    num = [c for c in df.columns if c != "Label"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if features is None:   # auto-pick the features that separate the chosen classes best
        from sklearn.feature_selection import f_classif
        cand = [c for c in num if df[c].nunique() > 1 and df[c].notna().all()]
        sc = np.nan_to_num(f_classif(df[cand].values, df["Label"].values)[0])
        features = [cand[i] for i in np.argsort(-sc)[:n_feat]]
    missing = [f for f in features if f not in df.columns]
    if missing: sys.exit(f"Unknown feature(s) {missing}. Available: {num[:15]} ...")
    df = df[features + ["Label"]].dropna().reset_index(drop=True)
    return df, features, len(df) // 2


# ----------------------------------------------------------------------------
# MINI-XGBOOST (exact greedy, softmax objective) written from scratch
# ----------------------------------------------------------------------------
def softmax(F):
    e = np.exp(F - F.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def score(G, H, lam): return G * G / (H + lam)      # "quality" of a node


def find_split(X, g, h, idx, P):
    """Try every feature and every midpoint between neighbouring values; return best + all candidates."""
    G, H = g[idx].sum(), h[idx].sum()
    parent = score(G, H, P["lam"])
    best, cands = None, []
    for f in range(X.shape[1]):
        order = idx[np.argsort(X[idx, f], kind="stable")]
        xs = X[order, f]; cg = np.cumsum(g[order]); ch = np.cumsum(h[order])
        for k in range(len(order) - 1):
            if xs[k] == xs[k + 1]: continue
            GL, HL = cg[k], ch[k]; GR, HR = G - GL, H - HL
            if HL < P["mcw"] or HR < P["mcw"]: continue
            thr = float(np.float32((np.float32(xs[k]) + np.float32(xs[k + 1])) * np.float32(0.5)))
            gain = score(GL, HL, P["lam"]) + score(GR, HR, P["lam"]) - parent
            c = dict(f=f, thr=thr, gain=gain, GL=GL, HL=HL, GR=GR, HR=HR, nL=k + 1, nR=len(order) - k - 1)
            cands.append(c)
            if best is None or gain > best["gain"]: best = c
    return best, cands


def build_tree(X, g, h, P):
    """Grow one tree level by level (breadth-first, same node numbering as XGBoost)."""
    def new(idx, depth):
        G, H = g[idx].sum(), h[idx].sum()
        return dict(idx=idx, depth=depth, G=G, H=H, n=len(idx), leaf=True,
                    w=-G / (H + P["lam"]) * P["eta"], feat=None, thr=None, left=None, right=None, gain=None, cands=None)
    nodes = [new(np.arange(len(X)), 0)]
    q = [0]
    while q:
        i = q.pop(0); nd = nodes[i]
        if nd["depth"] >= P["max_depth"] or nd["n"] < 2: continue
        best, cands = find_split(X, g, h, nd["idx"], P)
        if best is None or best["gain"] <= max(1e-6, P["gamma"]): continue
        mask = X[nd["idx"], best["f"]] < best["thr"]            # "yes" branch = value < threshold
        nd.update(leaf=False, feat=best["f"], thr=best["thr"], gain=best["gain"], cands=cands)
        nodes.append(new(nd["idx"][mask], nd["depth"] + 1)); nd["left"] = len(nodes) - 1
        nodes.append(new(nd["idx"][~mask], nd["depth"] + 1)); nd["right"] = len(nodes) - 1
        q += [nd["left"], nd["right"]]
    return nodes


def tree_output(nodes, n):
    out = np.zeros(n)
    for nd in nodes:
        if nd["leaf"]: out[nd["idx"]] = nd["w"]
    return out


def tree_predict(nodes, x):
    i = 0
    while not nodes[i]["leaf"]:
        i = nodes[i]["left"] if x[nodes[i]["feat"]] < nodes[i]["thr"] else nodes[i]["right"]
    return nodes[i]["w"]


def tree_path(nodes, x):
    i, steps = 0, []
    while not nodes[i]["leaf"]:
        nd = nodes[i]; yes = x[nd["feat"]] < nd["thr"]
        steps.append((i, nd["feat"], x[nd["feat"]], nd["thr"], yes))
        i = nd["left"] if yes else nd["right"]
    return steps, i


def show_tree(nodes, feats, lo, span, P, ind="    "):
    raw = lambda f, t: t * span[f] + lo[f]
    def rec(i, pre):
        nd = nodes[i]
        if nd["leaf"]:
            print(f"{pre}LEAF {i}: samples={nd['n']}  G={nd['G']:+.4f}  H={nd['H']:.4f}")
            print(f"{pre}        w = -G/(H+lambda)*eta = -({nd['G']:+.4f})/({nd['H']:.4f}+{P['lam']:g})*{P['eta']:g} = {nd['w']:+.4f}")
        else:
            print(f"{pre}NODE {i}: is {feats[nd['feat']]} < {nd['thr']:.4f} (raw {rv(raw(nd['feat'], nd['thr']))}) ?   "
                  f"[samples={nd['n']}, gain={nd['gain']:.4f}]")
            print(f"{pre}  yes ->"); rec(nd["left"], pre + "        ")
            print(f"{pre}  no  ->"); rec(nd["right"], pre + "        ")
    rec(0, ind)


# ----------------------------------------------------------------------------
# SHAP by hand: exact Shapley values for a tree ensemble (path-dependent value function)
# ----------------------------------------------------------------------------
def tree_value(nodes, x, S, i=0):
    """Expected tree output when only the features in S are 'known'. Unknown split -> cover-weighted average."""
    nd = nodes[i]
    if nd["leaf"]: return nd["w"]
    if nd["feat"] in S:
        return tree_value(nodes, x, S, nd["left"] if x[nd["feat"]] < nd["thr"] else nd["right"])
    l, r = nodes[nd["left"]], nodes[nd["right"]]
    return (l["H"] * tree_value(nodes, x, S, nd["left"]) + r["H"] * tree_value(nodes, x, S, nd["right"])) / (l["H"] + r["H"])


def shapley(v, M):
    phi = np.zeros(M)
    for i in range(M):
        others = [j for j in range(M) if j != i]
        for r in range(M):
            w = factorial(r) * factorial(M - r - 1) / factorial(M)
            for S in itertools.combinations(others, r):
                phi[i] += w * (v(set(S) | {i}) - v(set(S)))
    return phi


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv"); ap.add_argument("--classes", nargs="+"); ap.add_argument("--per-class", type=int, default=5)
    ap.add_argument("--features", nargs="+"); ap.add_argument("--n-features", type=int, default=4)
    ap.add_argument("--rounds", type=int, default=2); ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--eta", type=float, default=0.3); ap.add_argument("--sample", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", default=os.path.join("outputs", "walkthrough"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rep = open(os.path.join(a.out, "walkthrough_report.txt"), "w", encoding="utf-8")
    sys.stdout = Tee(sys.__stdout__, rep)

    P = dict(lam=1.0, gamma=0.0, mcw=0.0, eta=a.eta, max_depth=a.depth)   # mcw=0: default 1 is too strict for ~15 rows
    R = a.rounds

    if a.csv:
        if not a.classes: sys.exit("--classes is required with --csv")
        df, feats, dflt = csv_data(a.csv, a.classes, a.per_class, a.features, a.n_features, a.seed)
    else:
        df, feats, dflt = toy_data()
    s = dflt if a.sample is None else a.sample
    M, n = len(feats), len(df)

    # ------------------------------------------------------------------ STAGE 0
    H1("STAGE 0 - RAW INPUT (what the model is given)")
    print(f"{n} network flows, {M} features, label column = 'Label'.  Row numbers are used later to pick a sample.\n")
    print(df.to_string())

    # ------------------------------------------------------------------ STAGE 1
    H1("STAGE 1 - PRE-PROCESSING: label encoding + MinMax scaling (same as the main project)")
    le = LabelEncoder(); y = le.fit_transform(df["Label"].astype(str)); classes = list(le.classes_); K = len(classes)
    print("Label encoding:", {c: int(i) for i, c in enumerate(classes)})
    sc = MinMaxScaler(); X = sc.fit_transform(df[feats].values.astype(float)).astype(np.float32)
    lo, span = sc.data_min_, sc.data_max_ - sc.data_min_
    print("\nscaled = (raw - min) / (max - min)   -> every feature now lies in [0,1]\n")
    print(pd.DataFrame({"min": lo, "max": sc.data_max_}, index=feats).to_string(float_format=lambda v: rv(v)))
    print("\nScaled matrix X (this is what the trees see):\n")
    print(pd.DataFrame(X, columns=feats).round(4).to_string())
    print("\nNote: trees only compare 'value < threshold', so scaling does not change the trees; it is kept for consistency.")

    # real XGBoost model (trained silently; used only for checking and for its starting score)
    model = xgb.XGBClassifier(n_estimators=R, max_depth=a.depth, learning_rate=a.eta, reg_lambda=1.0, gamma=0.0,
                              min_child_weight=0.0, tree_method="exact", objective="multi:softprob",
                              n_jobs=1, random_state=a.seed)
    model.fit(X, y)
    booster = model.get_booster(); tdf = booster.trees_to_dataframe()
    leafval = {(int(t), int(nn)): float(g) for t, nn, g in zip(tdf["Tree"], tdf["Node"], tdf["Gain"]) if True}
    dm1 = xgb.DMatrix(X[:1]); mg = booster.predict(dm1, output_margin=True)[0]
    lv = booster.predict(dm1, pred_leaf=True).reshape(R, K)
    base = mg - np.array([[leafval[(r * K + k, int(lv[r, k]))] for k in range(K)] for r in range(R)]).sum(0)

    # ------------------------------------------------------------------ STAGE 2
    H1("STAGE 2 - HOW GRADIENT BOOSTING TRAINS (the formulas used below)")
    print(f"""Model = base score + sum of small trees.  With K={K} classes, every boosting ROUND adds {K} trees (one per class).
Each class k keeps a running score F_k (called 'margin' or log-odds).  Probabilities = softmax(F).

For every round:
  1. p = softmax(F)                       current predicted probabilities
  2. g = p - y        (y = 1 for the true class, else 0)   'gradient'  = how wrong we are (sign = direction)
  3. h = 2*p*(1-p)                         'hessian'   = how confident/curved the loss is
  4. For each class, grow ONE tree on (g, h):
       - a node with samples S has  G = sum(g), H = sum(h)
       - a split sends 'value < threshold' left, others right
       - gain = G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G^2/(H+lambda)       (bigger = better split)
       - every leaf gets weight  w = -G/(H+lambda) * eta
  5. F_k += w   for the leaf each sample falls into

Settings: lambda={P['lam']:g} (regularisation), eta={P['eta']:g} (learning rate), max_depth={P['max_depth']}, rounds={R}, min_child_weight=0.
Starting score of XGBoost (read from the real model): {np.round(base, 4).tolist()}  -> every F_k starts here.""")

    # ------------------------------------------------------------------ STAGE 3
    H1("STAGE 3 - TRAINING BY HAND, ROUND BY ROUND")
    Y = np.eye(K)[y]; F = np.tile(base, (n, 1)).astype(float); hand = []
    X64 = X.astype(float)
    for r in range(R):
        H2(f"ROUND {r + 1} of {R}")
        Pm = softmax(F); g = Pm - Y; h = np.maximum(2 * Pm * (1 - Pm), 1e-16)
        if n <= 30:
            t = pd.DataFrame({"label": df["Label"]})
            for k in range(K): t[f"p[{classes[k][:8]}]"] = Pm[:, k]
            for k in range(K): t[f"g{k}"] = g[:, k]
            for k in range(K): t[f"h{k}"] = h[:, k]
            print("Step 1-3: probabilities p, gradients g=p-y, hessians h=2p(1-p) (index k = class code; "
                  + ", ".join(f"{k}={c}" for k, c in enumerate(classes)) + ")\n")
            print(t.round(4).to_string())
        trees = []
        for k in range(K):
            print(f"\n>> Tree {r * K + k}  (round {r + 1}, class {k} = '{classes[k]}'): fit g{k}, h{k}")
            nodes = build_tree(X64, g[:, k], h[:, k], P)
            root = nodes[0]
            print(f"   Root: {root['n']} samples, G={root['G']:+.4f}, H={root['H']:.4f}, quality G^2/(H+lambda)={score(root['G'], root['H'], P['lam']):.4f}")
            if root["cands"]:
                print("   Split search at the root - best 5 of all candidate (feature, threshold) pairs:")
                top = sorted(root["cands"], key=lambda c: -c["gain"])[:5]
                print(f"   {'feature':<16}{'threshold(raw)':>16}{'nL/nR':>8}{'G_L':>9}{'H_L':>8}{'G_R':>9}{'H_R':>8}{'gain':>9}")
                for c in top:
                    print(f"   {feats[c['f']]:<16}{rv(c['thr'] * span[c['f']] + lo[c['f']]):>16}{str(c['nL']) + '/' + str(c['nR']):>8}"
                          f"{c['GL']:>+9.3f}{c['HL']:>8.3f}{c['GR']:>+9.3f}{c['HR']:>8.3f}{c['gain']:>9.4f}")
                print("   -> the highest gain wins; the same search is repeated inside each child until depth/gain limits stop it.")
            else:
                print("   No valid split -> the tree is a single leaf.")
            print("   Resulting tree:")
            show_tree(nodes, feats, lo, span, P)
            F[:, k] += tree_output(nodes, n)
            trees.append(nodes)
        hand.append(trees)
        Pn = softmax(F); loss = -np.log(Pn[np.arange(n), y]).mean(); acc = (Pn.argmax(1) == y).mean()
        print(f"\nStep 5: scores updated.  After round {r + 1}:  train log-loss = {loss:.4f},  train accuracy = {acc:.0%}")

    # ------------------------------------------------------------------ STAGE 4
    H1("STAGE 4 - CHECK: hand-built trees vs. real XGBoost trees")
    bad = 0
    for r in range(R):
        for k in range(K):
            t = r * K + k; d = tdf[tdf["Tree"] == t].set_index("Node"); nodes = hand[r][k]; probs = []
            if len(d) != len(nodes): probs.append(f"node count {len(nodes)} vs {len(d)}")
            else:
                for i, nd in enumerate(nodes):
                    row = d.loc[i]
                    if nd["leaf"]:
                        if row["Feature"] != "Leaf" or abs(float(row["Gain"]) - nd["w"]) > 1e-4: probs.append(f"leaf {i}")
                    else:
                        fn = f"f{nd['feat']}"
                        if row["Feature"] not in (fn, feats[nd["feat"]]) or abs(float(row["Split"]) - nd["thr"]) > 1e-5 \
                                or abs(float(row["Gain"]) - nd["gain"]) > 1e-3: probs.append(f"split {i}")
                    if abs(float(row["Cover"]) - nd["H"]) > 1e-3 * max(1, nd["H"]): probs.append(f"cover {i}")
            bad += len(probs)
            print(f"tree {t:>2} (round {r + 1}, '{classes[k]}'): " + ("MATCH (features, thresholds, gains, leaf weights, covers)" if not probs else "DIFFERENT: " + ", ".join(probs)))
    pm = booster.predict(xgb.DMatrix(X), output_margin=True)
    print(f"\nFinal scores F (hand) vs XGBoost margins: max |difference| = {np.abs(F - pm).max():.2e}")
    same_out = np.abs(F - pm).max() < 1e-4
    print("RESULT:", "hand-built training reproduces XGBoost exactly." if bad == 0 and same_out
          else "predictions are identical, but a tree differs: two different (feature, threshold) pairs produced the SAME gain "
               "(a tie, common on tiny data where two features separate the samples identically). XGBoost broke the tie differently; "
               "SHAP credit below can therefore differ slightly from the library." if same_out
          else "outputs differ - check the DIFFERENT lines above.")
    print("Cover = sum of hessians H in the node; SHAP uses it below.")

    # ------------------------------------------------------------------ STAGE 5
    H1(f"STAGE 5 - PREDICTION FOR ONE SAMPLE (row {s}, true label = {df['Label'][s]})")
    x = X64[s]
    print("Raw values   : " + ", ".join(f"{f}={rv(df[f][s])}" for f in feats))
    print("Scaled values: " + ", ".join(f"{f}={x[i]:.4f}" for i, f in enumerate(feats)) + "\n")
    run = base.astype(float).copy()
    print(f"Start: F = {np.round(run, 4).tolist()}   (classes: {classes})")
    for r in range(R):
        for k in range(K):
            steps, leaf = tree_path(hand[r][k], x); w = hand[r][k][leaf]["w"]; run[k] += w
            way = "; ".join(f"{feats[f]}={x[f]:.3f} {'<' if yes else '>='} {thr:.3f} -> {'yes' if yes else 'no'}" for _, f, _, thr, yes in steps) or "(single leaf)"
            print(f"  tree {r * K + k:>2} ('{classes[k]}'): {way}  => leaf {leaf}, w={w:+.4f}   F[{classes[k]}]={run[k]:+.4f}")
    p = softmax(run[None, :])[0]
    print(f"\nFinal scores F = {np.round(run, 4).tolist()}")
    e = np.exp(run - run.max())
    print("softmax: p_k = exp(F_k - Fmax) / sum_j exp(F_j - Fmax)")
    for k in range(K): print(f"   {classes[k]:<14} exp = {e[k]:.4f}   p = {e[k]:.4f}/{e.sum():.4f} = {p[k]:.4f}")
    pred = int(p.argmax()); xp = model.predict_proba(X[s:s + 1])[0]
    print(f"Predicted class = argmax = '{classes[pred]}' (p={p[pred]:.4f}).  XGBoost predict_proba says {np.round(xp, 4).tolist()}  "
          f"(max diff {np.abs(xp - p).max():.1e})")

    # ------------------------------------------------------------------ STAGE 6
    H1("STAGE 6 - SHAP BY HAND (exact Shapley values)")
    print(f"""Question SHAP answers: 'how much of the gap between the average score and THIS sample's score is due to each feature?'
Idea: a feature's credit = its average marginal contribution over ALL orders in which features can be revealed.
 - v(S) = score of class k when only the features in S are known.  Unknown features at a split are handled by
   sending the sample down BOTH branches and averaging, weighted by cover (hessian sum) of each branch.
 - phi_i = sum over coalitions S not containing i of  |S|!(M-|S|-1)!/M! * [ v(S+i) - v(S) ]
 - Sanity law (additivity):  base + sum(phi) = final score F_k.
With M={M} features there are 2^{M}={2 ** M} coalitions - small enough to list every one.
SHAP below explains class '{classes[pred]}' (the predicted class), in margin (log-odds) units, not probabilities.""")

    def V(k, S):    # v(S) for class k, including the starting score
        return base[k] + sum(tree_value(hand[r][k], x, S) for r in range(R))

    names = [f.split()[0][:6] + (f.split()[1][:3] if len(f.split()) > 1 else "") for f in feats]
    H2(f"Table of v(S) for class '{classes[pred]}' (score when only features in S are known)")
    print(f"{'known features S':<46}{'v(S)':>10}")
    for r_ in range(M + 1):
        for S in itertools.combinations(range(M), r_):
            lab = "{" + ", ".join(feats[i] for i in S) + "}" if S else "{} (nothing known = average score)"
            print(f"{lab:<46}{V(pred, set(S)):>+10.4f}" + ("   <- base value" if not S else "   <- full score F" if len(S) == M else ""))
    phi = shapley(lambda S: V(pred, S), M)
    top_i = int(np.argmax(np.abs(phi)))
    H2(f"Worked example: phi for '{feats[top_i]}' (largest effect), class '{classes[pred]}'")
    print(f"{'S (others known)':<40}{'v(S)':>9}{'v(S+i)':>9}{'diff':>9}{'weight':>9}{'w*diff':>9}")
    others = [j for j in range(M) if j != top_i]; tot = 0
    for r_ in range(M):
        w = factorial(r_) * factorial(M - r_ - 1) / factorial(M)
        for S in itertools.combinations(others, r_):
            a0, a1 = V(pred, set(S)), V(pred, set(S) | {top_i}); tot += w * (a1 - a0)
            print(f"{'{' + ', '.join(names[i] for i in S) + '}':<40}{a0:>+9.4f}{a1:>+9.4f}{a1 - a0:>+9.4f}{w:>9.4f}{w * (a1 - a0):>+9.4f}")
    print(f"{'phi = sum of last column':<40}{'':>27}{tot:>+18.4f}")

    # ------------------------------------------------------------------ STAGE 7
    H1("STAGE 7 - CHECK AGAINST XGBoost / shap LIBRARY, AND THE PLAIN-ENGLISH REASON")
    PHI = np.array([shapley(lambda S, k=k: V(k, S), M) for k in range(K)])        # (K, M)
    BASE = np.array([V(k, set()) for k in range(K)])
    xc = booster.predict(xgb.DMatrix(X[s:s + 1]), pred_contribs=True).reshape(K, M + 1)
    print(f"max |hand phi - xgboost pred_contribs| over all classes/features = {np.abs(PHI - xc[:, :M]).max():.2e}")
    print(f"max |hand base - xgboost bias|                                   = {np.abs(BASE - xc[:, M]).max():.2e}")
    try:
        import shap
        sv = shap.TreeExplainer(model).shap_values(X[s:s + 1]); sv = np.asarray(sv if not isinstance(sv, list) else np.stack(sv, -1))
        sv = sv.reshape(M, K).T if sv.size == M * K else None
        if sv is not None: print(f"max |hand phi - shap.TreeExplainer|                              = {np.abs(PHI - sv).max():.2e}")
    except Exception as ex:
        print(f"(shap library comparison skipped: {type(ex).__name__})")
    print(f"additivity: base + sum(phi) = {BASE[pred]:+.4f} + {PHI[pred].sum():+.4f} = {BASE[pred] + PHI[pred].sum():+.4f}   vs final F = {run[pred]:+.4f}")

    H2("SHAP table for this sample (rows = features, columns = classes; units = score/log-odds)")
    t = pd.DataFrame(PHI.T, index=[f"{f} = {rv(df[f][s])}" for f in feats], columns=classes)
    t.loc["base value (average)"] = BASE; t.loc["= final score F"] = run
    t.loc["probability"] = p
    print(t.round(4).to_string())

    H2("Plain-English explanation")
    o = np.argsort(-np.abs(PHI[pred]))
    pro = [(feats[i], rv(df[feats[i]][s]), PHI[pred][i]) for i in o if PHI[pred][i] > 0]
    con = [(feats[i], rv(df[feats[i]][s]), PHI[pred][i]) for i in o if PHI[pred][i] < 0]
    rival = int(np.argsort(-run)[1])
    print(f"1. The model predicts '{classes[pred]}' with probability {p[pred]:.2f} (true label: {df['Label'][s]}).")
    print(f"2. Before looking at the flow, the average score for '{classes[pred]}' is {BASE[pred]:+.3f}; this flow ends at {run[pred]:+.3f}.")
    if pro: print("3. Pushing UP towards this class: " + "; ".join(f"{f} = {v} ({c:+.3f})" for f, v, c in pro) + ".")
    if con: print("4. Pushing DOWN (evidence against):  " + "; ".join(f"{f} = {v} ({c:+.3f})" for f, v, c in con) + ".")
    print(f"5. Runner-up class is '{classes[rival]}' (score {run[rival]:+.3f}); '{classes[pred]}' wins because its score is higher by {run[pred] - run[rival]:.3f}, "
          f"which softmax turns into probabilities {p[pred]:.2f} vs {p[rival]:.2f}.")
    print("6. Reading rule: SHAP value > 0 = this feature value raised the score of the class; < 0 = lowered it; size = strength.")
    print("   Caution: SHAP describes what the MODEL relied on, not proof that the traffic is truly malicious.")

    H2("Global view: mean |phi| of each feature for the predicted class of every sample (all rows)")
    allphi = np.zeros((n, M))
    for i in range(n):
        xi = X64[i]; pk = int(np.argmax(base + np.array([sum(tree_predict(hand[r][k], xi) for r in range(R)) for k in range(K)])))
        allphi[i] = shapley(lambda S, k=pk, xi=xi: base[k] + sum(tree_value(hand[r][k], xi, S) for r in range(R)), M)
    g_ = pd.Series(np.abs(allphi).mean(0), index=feats).sort_values(ascending=False)
    print(g_.round(4).to_string())
    print("\nThis is exactly what the 'shap_summary.png' bar plot in the main project shows, just on tiny data.")
    print(f"\nReport saved to {os.path.join(a.out, 'walkthrough_report.txt')}")
    sys.stdout = sys.__stdout__; rep.close()


if __name__ == "__main__":
    main()
