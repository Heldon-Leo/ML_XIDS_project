"""
live_monitor.py - tail a CICFlowMeter CSV and write detection events.

    python live_monitor.py                         # follow capture/live_flows.csv (new rows only)
    python live_monitor.py --from-start            # also process rows already in the file
    python live_monitor.py --once --from-start     # process what's there and exit
    python live_monitor.py --capture my.csv --threshold 0.9
"""
import argparse
import time

import config
from src.flow_stream import IncrementalCsvReader
from src.live_inference import LiveDetector
from src.utils import get_logger

logger = get_logger("monitor")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capture", default=config.LIVE_CAPTURE_FILE)
    p.add_argument("--poll", type=float, default=config.LIVE_POLL_SECONDS)
    p.add_argument("--threshold", type=float, default=None, help="ALERT confidence threshold")
    p.add_argument("--from-start", action="store_true", help="process existing rows too")
    p.add_argument("--once", action="store_true", help="drain the file then exit")
    p.add_argument("--explain-benign", action="store_true")
    a = p.parse_args()
    if a.threshold is not None:
        config.LIVE_ALERT_THRESHOLD = a.threshold

    det = LiveDetector(explain_benign=a.explain_benign or None)
    reader = IncrementalCsvReader(a.capture, from_start=a.from_start or a.once)
    logger.info(f"Monitoring {a.capture} (threshold={config.LIVE_ALERT_THRESHOLD}, "
                f"{'from start' if a.from_start or a.once else 'new rows only'}). Ctrl+C to stop.")
    try:
        while True:
            drained = False
            while True:
                raw = reader.read_new(config.LIVE_BATCH_MAX_ROWS)
                if raw.empty:
                    break
                events = det.process(raw)
                drained = True
                alerts = [e for e in events if e["status"] != "BENIGN"]
                logger.info(f"batch: {len(raw)} rows -> {len(events)} events, {len(alerts)} alert/review "
                            f"({det.latency_ms_last:.0f} ms)")
                for e in alerts[:5]:
                    logger.info(f"  {e['status']:6} {e['predicted_class']} {e['confidence']:.2f} "
                                f"{e['src_ip']}->{e['dst_ip']}:{e['dst_port']}")
            if a.once:
                break
            if not drained:
                det.write_stats()
            time.sleep(a.poll)
    except KeyboardInterrupt:
        logger.info("Stopped.")
    det.write_stats()
    logger.info(f"Totals: {det.counters}")


if __name__ == "__main__":
    main()
