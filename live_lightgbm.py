#!/usr/bin/env python3
"""Run the existing telemetry collector and score each new row with LightGBM."""

import argparse
import csv
import json
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import pandas as pd


FEATURE_COLUMNS = [
    "interval_s", "cpu_pct", "read_iops", "write_iops",
    "read_bytes_per_s", "write_bytes_per_s", "read_mb_per_s",
    "write_mb_per_s", "write_iops_fraction", "net_rx_bytes_per_s",
    "net_tx_bytes_per_s",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description="Live LightGBM scoring for VM telemetry.")
    parser.add_argument("--vm-name", default="lin")
    parser.add_argument("--model", type=Path, default=Path("models/ransomware_lightgbm_50_50.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--prediction-log", type=Path, default=Path("data/live_lightgbm_predictions.jsonl"))
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.83)
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"model not found: {args.model}")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.prediction_log.parent.mkdir(parents=True, exist_ok=True)
    model = lgb.Booster(model_file=str(args.model))
    existing_files = set(args.output_dir.glob("telemetry_linux_*.csv"))
    command = [
        sys.executable, "vm_telemetry_collector.py",
        "--vm-name", args.vm_name,
        "--interval", str(args.interval),
        "--output-dir", str(args.output_dir),
    ]
    collector = subprocess.Popen(command)
    stopped = False

    def stop(_signal, _frame):
        nonlocal stopped
        stopped = True
        collector.send_signal(signal.SIGINT)

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    output_path = None
    while not stopped and collector.poll() is None:
        candidates = sorted(
            (path for path in args.output_dir.glob("telemetry_linux_*.csv") if path not in existing_files),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            output_path = candidates[-1]
            break
        time.sleep(0.2)
    if output_path is None:
        collector.wait()
        return collector.returncode

    print(f"[LIVE LIGHTGBM] reading {output_path}", flush=True)
    seen_rows = 0
    with args.prediction_log.open("a", encoding="utf-8", buffering=1) as log_file:
        while not stopped:
            try:
                frame = pd.read_csv(output_path)
            except (FileNotFoundError, pd.errors.EmptyDataError):
                frame = pd.DataFrame()
            if len(frame) > seen_rows:
                new_rows = frame.iloc[seen_rows:]
                for _, row in new_rows.iterrows():
                    features = pd.DataFrame([[row[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
                    probability = float(model.predict(features)[0])
                    record = {
                        "telemetry_timestamp": row["timestamp"],
                        "host_scored_ts": utc_now(),
                        "ransomware_probability": probability,
                        "verdict": int(probability >= args.threshold),
                    }
                    if record["verdict"]:
                        record["message"] = "RANSOMWARE DETECTED"
                    log_file.write(json.dumps(record) + "\n")
                    if record["verdict"]:
                        print(
                            f"[LIVE LIGHTGBM] RANSOMWARE DETECTED "
                            f"timestamp={record['telemetry_timestamp']} probability={probability:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[LIVE LIGHTGBM] {record['telemetry_timestamp']} "
                            f"probability={probability:.4f} verdict=0",
                            flush=True,
                        )
                seen_rows = len(frame)
            if collector.poll() is not None:
                break
            time.sleep(0.2)

    if collector.poll() is None:
        collector.send_signal(signal.SIGINT)
    collector.wait()
    print(f"[LIVE LIGHTGBM] predictions -> {args.prediction_log}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())