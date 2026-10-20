"""
simulate_traffic.py - replay rows from the CSE-CIC-IDS2018 files (data/*.csv) into the
live capture file in small batches, so the monitor + dashboard can be demonstrated
without a real capture. The Label column is kept as `Label` (ground truth only; the
model never sees it).

    python simulate_traffic.py                      # 20 rows every 2 s, ~25% attack
    python simulate_traffic.py --batch 50 --interval 1 --attack-ratio 0.4 --total 2000
    python simulate_traffic.py --reset              # empty the capture file first
"""
import argparse
import glob
import os
import random
import time
import uuid
from datetime import datetime

import pandas as pd

import config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batch", type=int, default=20)
    p.add_argument("--interval", type=float, default=2.0)
    p.add_argument("--attack-ratio", type=float, default=0.25)
    p.add_argument("--total", type=int, default=0, help="stop after N rows (0 = until Ctrl+C)")
    p.add_argument("--pool", type=int, default=20000, help="rows sampled from the dataset")
    p.add_argument("--capture", default=config.LIVE_CAPTURE_FILE)
    p.add_argument("--reset", action="store_true")
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files in {config.DATA_DIR}")
    frames = []
    for f in files:
        frames.append(pd.read_csv(f, nrows=a.pool // len(files) + 1, low_memory=False))
    pool = pd.concat(frames, ignore_index=True)
    pool.columns = [str(c).strip() for c in pool.columns]
    pool = pool[pool.iloc[:, 0].astype(str) != pool.columns[0]]
    label_col = next((c for c in pool.columns if c.lower() == "label"), None)
    if label_col is None:
        raise SystemExit("Dataset has no Label column")
    is_benign = pool[label_col].astype(str).str.strip().str.lower() == "benign"
    benign, attack = pool[is_benign], pool[~is_benign]
    print(f"Pool: {len(benign)} benign, {len(attack)} attack rows from {len(files)} file(s)")

    if a.reset and os.path.exists(a.capture):
        os.remove(a.capture)
    written, header_needed = 0, not os.path.exists(a.capture) or os.path.getsize(a.capture) == 0
    try:
        while a.total == 0 or written < a.total:
            n_att = int(round(a.batch * a.attack_ratio))
            parts = [benign.sample(a.batch - n_att, replace=True)]
            if n_att and len(attack):
                parts.append(attack.sample(n_att, replace=True))
            batch = pd.concat(parts).sample(frac=1).copy()
            # Simulation only: the 2018 files carry no IPs / unique ids, so add them.
            # Attack rows come from a few "attacker" hosts, benign from many clients.
            is_b = batch[label_col].astype(str).str.strip().str.lower() == "benign"
            batch["Src IP"] = [f"10.0.0.{random.randint(10, 80)}" if b else f"172.31.{random.randint(60, 62)}.{random.randint(2, 6)}" for b in is_b]
            batch["Dst IP"] = "10.0.0.5"
            batch["Src Port"] = [random.randint(1025, 65000) for _ in range(len(batch))]
            batch["Flow ID"] = [uuid.uuid4().hex[:16] for _ in range(len(batch))]
            batch["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            batch.to_csv(a.capture, mode="a", header=header_needed, index=False)
            header_needed = False
            written += len(batch)
            print(f"wrote {len(batch)} rows (total {written})")
            time.sleep(a.interval)
    except KeyboardInterrupt:
        print("Stopped.")


if __name__ == "__main__":
    main()
