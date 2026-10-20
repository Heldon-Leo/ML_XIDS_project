"""
make_demo_snapshot.py - save a small copy of a real live-monitor run so the dashboard can run on Streamlit Cloud.

Streamlit Cloud cannot run the simulator, the monitor or the 4,500-tree model next to the dashboard, so the cloud
app replays a saved snapshot of your own run.

    python make_demo_snapshot.py                 # uses outputs/live/, keeps up to 1500 alert events
    python make_demo_snapshot.py --max-events 800

Writes demo_data/live/alerts.jsonl and demo_data/live/live_stats.json. Commit the demo_data/ folder to GitHub.
"""
import argparse
import json
import os
import shutil
from collections import defaultdict
from datetime import datetime, timezone

import config

DEMO_DIR = os.path.join(config.BASE_DIR, "demo_data", "live")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-events", type=int, default=1500, help="maximum alert/review events to keep")
    ap.add_argument("--src", default=config.LIVE_OUTPUT_DIR, help="folder with alerts.jsonl and live_stats.json")
    a = ap.parse_args()

    alerts_path = os.path.join(a.src, "alerts.jsonl")
    stats_path = os.path.join(a.src, "live_stats.json")
    if not (os.path.exists(alerts_path) and os.path.exists(stats_path)):
        raise SystemExit(f"Nothing to snapshot in {a.src}. Run simulate_traffic.py + live_monitor.py first.")

    events = []
    with open(alerts_path, encoding="utf-8") as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except ValueError:
                pass
    if not events:
        raise SystemExit("alerts.jsonl is empty: let the monitor process some attack flows first.")

    # keep every REVIEW event, and spread the ALERT budget evenly over the attack classes
    keep_ids = {e["event_id"] for e in events if e.get("status") == "REVIEW"}
    budget = max(0, a.max_events - len(keep_ids))
    by_cls = defaultdict(list)
    for e in events:
        if e["event_id"] not in keep_ids:
            by_cls[e["predicted_class"]].append(e)
    while budget > 0 and any(by_cls.values()):
        for cls in list(by_cls):
            if by_cls[cls] and budget > 0:
                keep_ids.add(by_cls[cls].pop()["event_id"])     # newest first
                budget -= 1
    kept = [e for e in events if e["event_id"] in keep_ids]

    os.makedirs(DEMO_DIR, exist_ok=True)
    with open(os.path.join(DEMO_DIR, "alerts.jsonl"), "w", encoding="utf-8") as f:
        for e in kept:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(stats_path, encoding="utf-8") as f:
        stats = json.load(f)
    stats["snapshot"] = {"source_events": len(events), "kept": len(kept),
                         "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")}
    with open(os.path.join(DEMO_DIR, "live_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    size = sum(os.path.getsize(os.path.join(DEMO_DIR, n)) for n in os.listdir(DEMO_DIR)) / 1e6
    print(f"Kept {len(kept)} of {len(events)} events -> {DEMO_DIR} ({size:.1f} MB)")
    if size > 50:
        print("WARNING: over 50 MB. GitHub rejects files above 100 MB; re-run with a smaller --max-events.")


if __name__ == "__main__":
    main()
