# ML-XIDS Live Monitoring (Module 4 extension)

```
CICFlowMeter -> capture/live_flows.csv -> live_monitor.py -> outputs/live/*  -> dashboard.py
                                           (schema check, scaler, XGBoost, TreeSHAP)
                                                   |
                                                   +-> events.jsonl = contract for the ontology module (Module 2)
```

## One-time prerequisites
1. `pip install -r requirements-full.txt`
2. `models/` must contain xgb_model.pkl, scaler.pkl, label_encoder.pkl, feature_columns.json.
3. Run `python main.py --force` once so `models/preprocessed_split.pkl` exists. The monitor uses it to learn the
   training medians / min / max (saved to `models/feature_stats.json`).

## Step 1 - check that your capture matches the model
    python check_schema.py capture/live_flows.csv
Exit code 1 = a required feature is missing, or values are far outside the training range (often a units
mismatch). Fix this BEFORE trusting predictions. Alias matches (2017-style names) only warn: verify the
definitions are really identical.

## Step 2 - run
Demo (replays rows from `data/*.csv`, adds simulated IPs/ids):

    python simulate_traffic.py --reset --batch 20 --interval 2     # terminal 1
    python live_monitor.py --from-start                             # terminal 2
    streamlit run dashboard.py                                      # terminal 3

Real traffic: write CICFlowMeter's CSV to `capture/live_flows.csv` (or use `--capture <path>`), then run the
monitor and dashboard the same way. Options: `--threshold 0.9`, `--explain-benign`, `--once`, `--poll 1`.

## Real capture on Windows (not verified here - check on your machine)
* Install Npcap; use the Java CICFlowMeter GUI or the Python `cicflowmeter` package
  (`cicflowmeter -i <interface> -c flows.csv`).
* Column names/units differ between CICFlowMeter versions and CSV flushing may be delayed until flows end.
  `check_schema.py` tells you. A model trained on the 2018 testbed may not transfer to home traffic: watch
  `out_of_range_features` in each event.

## Outputs (`outputs/live/`)
| File | Content |
|---|---|
| events.jsonl | every processed flow as a detection event |
| alerts.jsonl / alerts.csv | ALERT and REVIEW events only |
| triples.jsonl | attacker performed / targeted / maps_to_mitre / explained_by triples |
| rejected.jsonl | rows dropped (missing or invalid features) |
| live_stats.json | counters + latency for the dashboard |

ALERT = attack class with confidence >= threshold (default 0.85); REVIEW = attack predicted below it; BENIGN.
SHAP values are log-odds and explain the predicted class. MITRE mapping is best effort and unaudited.
Restarting the monitor never duplicates events (deterministic event_id).

## Event contract
`schemas/detection_event.schema.json` v1.0; every event is validated before it is written. Building the ontology
is Module 2's job; `ontology_status` is PENDING for malicious events.
