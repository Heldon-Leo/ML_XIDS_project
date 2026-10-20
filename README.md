# ML-XIDS: Explainable Machine Learning Intrusion Detection System

A production-style pipeline that trains a multi-class **XGBoost** intrusion
detection model on the **CSE-CIC-IDS2018** dataset and explains every
prediction using **SHAP (TreeSHAP)** — both globally (which features matter
most overall) and locally (why *this specific flow* was flagged as an attack).

This project stops at the Explainable AI stage by design: there is no API
server or frontend, only a reproducible offline pipeline you run with a
single command.

---

## 1. Project Structure

```
ML-XIDS/
│
├── data/                          # Place CSE-CIC-IDS2018 CSVs here
│   └── README.md
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py             # Discover + merge all CSVs
│   ├── preprocessing.py           # Clean, encode, scale, split
│   ├── train.py                   # XGBoost training
│   ├── evaluate.py                # Metrics + confusion matrix
│   ├── explain.py                 # SHAP global + local explanations
│   └── utils.py                   # Logging, IO helpers, memory reduction
│
├── models/
│   ├── xgb_model.pkl              # Trained model (created by main.py)
│   ├── label_encoder.pkl          # Fitted LabelEncoder
│   ├── scaler.pkl                 # Fitted MinMaxScaler
│   └── feature_columns.json       # Final feature column order
│
├── outputs/
│   ├── shap_summary.png                     # Global SHAP beeswarm plot
│   ├── shap_force_plot.html                 # Local SHAP force plot (interactive)
│   ├── feature_importance.csv               # Global feature ranking
│   ├── predictions_with_explanations.csv    # Structured local explanations
│   ├── confusion_matrix.png
│   └── evaluation_report.txt
│
├── requirements.txt
├── main.py                        # Full pipeline entry point
├── config.py                      # All paths & hyperparameters
└── README.md
```

---

## 2. Dataset: CSE-CIC-IDS2018

**Download link:** https://data.mendeley.com/datasets/29hdbdzx2r/1

### Steps to download

1. Open the link above in your browser.
2. Click **Download All (X GB)** (or download files individually).
3. Extract the archive if it is zipped.
4. Copy every `*.csv` file (e.g. `Friday-02-03-2018_TrafficForML_CICFlowMeter.csv`,
   `Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv`, etc.) into this
   project's `data/` folder:

```
ML-XIDS/data/
├── Friday-02-03-2018_TrafficForML_CICFlowMeter.csv
├── Friday-16-02-2018_TrafficForML_CICFlowMeter.csv
├── ...
└── Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv
```

> The full dataset is several gigabytes across ~10 files. You do not need
> every file to run the pipeline — even a subset of CSVs will work, since
> `data_loader.py` merges whatever `.csv` files it finds in `data/`.

### Note on scale

`config.py` includes two safety valves for large-scale/limited-hardware
runs:
- `MAX_ROWS_PER_FILE` — cap rows read per CSV (default: `None`, i.e. all rows)
- `MAX_TOTAL_ROWS` — stratified cap on the final merged dataset
  (default: `500_000`, so SHAP stays tractable on a laptop)

Set both to `None` in `config.py` for a full-scale, uncapped run on a
sufficiently large machine.

---

## 3. Setup

### Requirements
- Python 3.9+
- ~4 GB+ free RAM (more for full-scale, uncapped runs)

### Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

---

## 4. Running the Pipeline

After placing dataset CSVs in `data/`:

```bash
python main.py
```

This runs, in order:

1. **Data Loading** — discovers and merges all CSVs in `data/`
2. **Preprocessing** — strips column names, fixes inf/NaN values,
   label-encodes the multi-class `Label` column, MinMax-scales features,
   and performs an 80/20 stratified train/test split
3. **Training** — fits an `XGBClassifier` (multi-class, `tree_method="hist"`)
   and saves it to `models/xgb_model.pkl`
4. **Evaluation** — computes accuracy, precision/recall/F1 per class, and
   a confusion matrix; writes `outputs/evaluation_report.txt` and
   `outputs/confusion_matrix.png`
5. **Explainability (SHAP)**:
   - Global: `outputs/shap_summary.png`, `outputs/feature_importance.csv`
   - Local: `outputs/predictions_with_explanations.csv`,
     `outputs/shap_force_plot.html`

Progress and timing for each stage are logged to the console.

### Running individual stages

Each module can also run standalone for debugging/inspection, e.g.:

```bash
python -m src.data_loader
python -m src.preprocessing
python -m src.train
python -m src.evaluate
python -m src.explain
```

---

## 5. Expected Outputs

| File | Description |
|---|---|
| `models/xgb_model.pkl` | Trained multi-class XGBoost model |
| `models/label_encoder.pkl` | Encoder mapping class names <-> integer labels |
| `models/scaler.pkl` | Fitted MinMaxScaler for feature normalization |
| `models/feature_columns.json` | Ordered list of feature columns used for training |
| `outputs/evaluation_report.txt` | Accuracy + full classification report |
| `outputs/confusion_matrix.png` | Confusion matrix heatmap |
| `outputs/shap_summary.png` | Global SHAP beeswarm summary across all classes |
| `outputs/feature_importance.csv` | Features ranked by mean(\|SHAP value\|) |
| `outputs/shap_force_plot.html` | Interactive local explanation for one sample |
| `outputs/predictions_with_explanations.csv` | Structured explanation per test sample |

### Example row from `predictions_with_explanations.csv`

| Prediction | Confidence | Top_Features | Explanation_Text |
|---|---|---|---|
| DDoS-HOIC | 0.98 | Flow Duration (low, SHAP=0.41); SYN Flag Count (high, SHAP=0.33); Fwd Packets/s (high, SHAP=0.21) | Low flow duration, high syn flag count, and high fwd packets/s contributed to DDoS-HOIC classification. |

---

## 6. Reproducibility

- All random operations (`train_test_split`, sampling, XGBoost) use
  `random_state=42` (see `config.RANDOM_STATE`).
- Fitted preprocessing artifacts (`label_encoder.pkl`, `scaler.pkl`,
  `feature_columns.json`) are saved so the exact same transformation can
  be replayed later without retraining.

---

## 7. Packaging as a ZIP

From the parent directory of `ML-XIDS/`:

```bash
zip -r ML-XIDS.zip ML-XIDS -x "ML-XIDS/data/*.csv" "ML-XIDS/models/*.pkl" "ML-XIDS/outputs/*"
```

(The exclusions keep the archive small — the raw dataset, trained model
binary, and generated outputs are all reproducible by running `main.py`
and don't need to travel with the source code.) Remove the `-x` filters
if you want a fully self-contained archive including the dataset and
generated artifacts.

---

## 8. Scope Note

This project intentionally stops at the Explainable AI stage. There is no
REST API, dashboard, or real-time inference service included — the
pipeline produces static, auditable artifacts (model + explanations)
suitable for offline analysis or as the foundation for a future serving
layer.
