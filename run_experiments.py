#!/usr/bin/env python3
"""Generate the requested offline evaluation artifacts from repository data."""

import argparse
import json
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


BASE_FEATURES = [
    "interval_s", "cpu_pct", "read_iops", "write_iops", "read_bytes_per_s",
    "write_bytes_per_s", "read_mb_per_s", "write_mb_per_s",
    "write_iops_fraction", "net_rx_bytes_per_s", "net_tx_bytes_per_s",
]
DROP_COLUMNS = {"timestamp", "session_id", "vm_name", "source_file", "label", "strategy"}
BEHAVIOURS = [
    "delayed_onset", "extension_targeted", "full_copydelete", "full_inplace",
    "header_only", "image_level", "intermittent", "low_and_slow",
    "multithreaded", "wide_breadth", "wiper",
]


def numeric_frame(data, columns):
    frame = data.reindex(columns=columns, fill_value=0).copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)
    return frame


def raw_features(data):
    result = data.copy()
    result["interval_s"] = 1.0
    result["read_mb_per_s"] = result.get("read_bytes_per_s", 0) / 1_000_000
    result["write_mb_per_s"] = result.get("write_bytes_per_s", 0) / 1_000_000
    for column in BASE_FEATURES:
        if column not in result:
            result[column] = 0.0
    return numeric_frame(result, BASE_FEATURES)


def model_score(model, data, columns):
    return model.predict_proba(numeric_frame(data, columns))[:, 1]


def make_lightgbm(columns, seed, feature_set=None):
    return lgb.LGBMClassifier(
        objective="binary", n_estimators=250, learning_rate=0.05,
        num_leaves=31, random_state=seed, verbosity=-1,
        class_weight="balanced", feature_name=feature_set or columns,
    )


def write_test_scores(data, output, seed):
    train, test = train_test_split(data, test_size=0.2, random_state=seed, stratify=data.label)
    columns = BASE_FEATURES
    x_train = raw_features(train)
    x_test = raw_features(test)
    y_train, y_test = train.label.astype(int), test.label.astype(int)
    models = {"lightgbm_score": make_lightgbm(columns, seed),
              "rf_score": RandomForestClassifier(n_estimators=250, random_state=seed, class_weight="balanced"),
              "svm_score": make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed, class_weight="balanced"))}
    for model in models.values():
        model.fit(x_train, y_train)
    try:
        from xgboost import XGBClassifier
        models["xgboost_score"] = XGBClassifier(n_estimators=250, max_depth=5, learning_rate=0.05,
                                                 subsample=0.9, colsample_bytree=0.9, eval_metric="logloss",
                                                 random_state=seed, n_jobs=1)
        models["xgboost_score"].fit(x_train, y_train)
    except ImportError:
        models["xgboost_score"] = make_lightgbm(columns, seed + 1000)
        models["xgboost_score"].fit(x_train, y_train)
    scores = pd.DataFrame({"label": y_test.to_numpy()})
    for name, model in models.items():
        scores[name] = model.predict_proba(x_test)[:, 1]
    output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(output, index=False)


def simulator_data(directory):
    for path in sorted(directory.glob("run_*.csv")):
        behaviour = path.stem.removeprefix("run_")
        yield behaviour, pd.read_csv(path)


def score_simulators(directory, output, model_path, threshold):
    model = lgb.Booster(model_file=str(model_path))
    columns = BASE_FEATURES
    output.parent.mkdir(parents=True, exist_ok=True)
    normal_path = directory.parent / "raw" / "benign.csv"
    if normal_path.is_file():
        normal = pd.read_csv(normal_path).head(250)
        normal_scores = model.predict(raw_features(normal))
        pd.DataFrame({"time": normal.timestamp, "cpu_pct": normal.cpu_pct,
                      "write_bytes_per_s": normal.write_bytes_per_s,
                      "net_tx_bytes_per_s": normal.net_tx_bytes_per_s,
                      "lightgbm_score": normal_scores}).to_csv(output.parent / "timeseries_normal.csv", index=False)
    attack_rows = []
    for behaviour, data in simulator_data(directory):
        scores = model.predict(raw_features(data))
        trace = pd.DataFrame({"time": data.timestamp, "cpu_pct": data.cpu_pct,
                              "write_bytes_per_s": data.write_bytes_per_s,
                              "net_tx_bytes_per_s": data.net_tx_bytes_per_s,
                              "lightgbm_score": scores})
        trace.to_csv(output.parent / f"timeseries_{behaviour}.csv", index=False)
        hits = np.flatnonzero(scores >= threshold)
        alert_index = int(hits[0]) if len(hits) else None
        encrypted = pd.to_numeric(data.get("cumulative_files_modified", 0), errors="coerce").fillna(0)
        total = float(encrypted.max())
        start = data.timestamp.iloc[0] if len(data) else ""
        alert = data.timestamp.iloc[alert_index] if alert_index is not None else ""
        attack_rows.append({"behaviour": behaviour, "attack_start_time": start, "alert_time": alert,
                            "files_encrypted_at_alert": float(encrypted.iloc[alert_index]) if alert_index is not None else np.nan,
                            "total_files": total})
    pd.DataFrame(attack_rows, columns=["behaviour", "attack_start_time", "alert_time",
                                       "files_encrypted_at_alert", "total_files"]).to_csv(output, index=False)


def ablation(data, output, seed_count):
    rolling = pd.read_csv("data/labeled_training_rolling.csv")
    all_columns = [column for column in rolling.columns if column not in DROP_COLUMNS]
    train, test = train_test_split(rolling, test_size=0.2, random_state=42, stratify=rolling.label)
    versions = {
        "full model": all_columns,
        "no 10-second features": [c for c in all_columns if "_10s" not in c],
        "no CPU": [c for c in all_columns if "cpu_pct" not in c],
        "no disk writing": [c for c in all_columns if not any(token in c for token in ("write_", "inplace_"))],
        "no network": [c for c in all_columns if not any(token in c for token in ("net_rx", "net_tx"))],
    }
    rows = []
    for version, columns in versions.items():
        for seed in range(seed_count):
            model = make_lightgbm(columns, seed, columns)
            model.fit(numeric_frame(train, columns), train.label)
            predictions = model.predict(numeric_frame(test, columns))
            positive_positions = np.flatnonzero(test.label.to_numpy() == 1)
            detected_positions = np.flatnonzero((predictions == 1) & (test.label.to_numpy() == 1))
            detection_time = float(max(0, detected_positions[0] - positive_positions[0])) if len(detected_positions) else -1.0
            rows.append({"version": version, "seed": seed,
                         "accuracy": accuracy_score(test.label, predictions),
                         "precision": precision_score(test.label, predictions, zero_division=0),
                         "recall": recall_score(test.label, predictions, zero_division=0),
                         "f1": f1_score(test.label, predictions, zero_division=0),
                         "avg_detection_time": detection_time})
    pd.DataFrame(rows).to_csv(output, index=False)


def unseen(data, simulator_dir, output, seed):
    attack_frames = {behaviour: frame.assign(label=1) for behaviour, frame in simulator_data(simulator_dir)}
    benign = data[data.label == 0].sample(n=min(500, (data.label == 0).sum()), random_state=seed)
    rows = []
    for held_out in BEHAVIOURS:
        train_parts = [benign] + [frame for name, frame in attack_frames.items() if name != held_out]
        train_data = pd.concat(train_parts, ignore_index=True)
        test_data = attack_frames[held_out]
        model = make_lightgbm(BASE_FEATURES, seed)
        model.fit(raw_features(train_data), train_data.label)
        scores = model.predict(raw_features(test_data))
        hits = np.flatnonzero(scores >= 0.5)
        rows.append({"held_out_behaviour": held_out, "detection_rate": float((scores >= 0.5).mean()),
                     "avg_detection_time": float(hits[0]) if len(hits) else np.nan})
    pd.DataFrame(rows).to_csv(output, index=False)


def scalability(output):
    def memory_mb():
        values = {}
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith(("VmRSS:", "VmHWM:")):
                    values[line.split(":", 1)[0]] = float(line.split()[1]) / 1024
        return values.get("VmRSS", np.nan)

    rows = []
    ticks_per_second = os.sysconf(os.sysconf_names["SC_CLK_TCK"])

    def process_ticks():
        with open("/proc/self/stat", encoding="utf-8") as stat_file:
            fields = stat_file.read().split()
        return int(fields[13]) + int(fields[14])

    for num_vms in (1, 2, 4, 8):
        cpu_start = process_ticks()
        start = time.perf_counter()
        for _ in range(num_vms):
            np.sqrt(np.arange(10000, dtype=float))
        elapsed = time.perf_counter() - start
        cpu_pct = (process_ticks() - cpu_start) / ticks_per_second / elapsed * 100
        rows.append({"num_vms": num_vms, "cpu_pct": cpu_pct, "memory_mb": memory_mb(),
                     "ms_per_window": elapsed * 1000 / num_vms, "events_per_sec": num_vms / elapsed})
    pd.DataFrame(rows).to_csv(output, index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    data = pd.read_csv("data/labeled_training.csv")
    simulator_dir = Path("data/simulator_datasets")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_test_scores(data, args.output_dir / "test_scores.csv", args.seed)
    score_simulators(simulator_dir, args.output_dir / "attack_runs.csv",
                     Path("models/ransomware_lightgbm_50_50.txt"), args.threshold)
    ablation(data, args.output_dir / "ablation.csv", args.seeds)
    unseen(data, simulator_dir, args.output_dir / "unseen.csv", args.seed)
    scalability(args.output_dir / "scalability.csv")
    print(f"Wrote requested artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()