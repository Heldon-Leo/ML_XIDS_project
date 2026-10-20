"""
check_schema.py - verify a CICFlowMeter CSV is compatible with the trained model
BEFORE starting live monitoring.

    python check_schema.py                      # checks capture/live_flows.csv
    python check_schema.py path/to/flows.csv

Exit code 0 = compatible, 1 = not compatible.
"""
import json
import sys

import config
from src.schema_validator import check_artifacts, format_report, validate_capture_file
from src.feature_stats import load_or_build_feature_stats
from src.utils import load_json


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else config.LIVE_CAPTURE_FILE
    print("=" * 64)
    print("ML-XIDS schema check")
    print("=" * 64)

    ok, errs, warns = check_artifacts()
    for w in warns:
        print(f"[INFO]  {w}")
    for e in errs:
        print(f"[ERROR] {e}")
    if not ok:
        print("RESULT: FAILED (model artifacts inconsistent)")
        return 1

    features = load_json(config.FEATURE_LIST_PATH)
    stats = load_or_build_feature_stats(features)
    print(f"File: {path}\n")
    report = validate_capture_file(path, features, stats)
    print(format_report(report))
    with open(config.SCHEMA_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
