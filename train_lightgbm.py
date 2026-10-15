#!/usr/bin/env python3
"""Train and evaluate a LightGBM ransomware classifier.

The pipeline accepts either the original per-window telemetry CSV or a dataset
that has already been enriched with rolling-window temporal features. It keeps
LightGBM as the model, but makes the feature set temporal-aware when rolling
features are present.
"""

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


BASE_FEATURE_COLUMNS = [
    "interval_s",
    "cpu_pct",
    "read_iops",
    "write_iops",
    "read_bytes_per_s",
    "write_bytes_per_s",
    "read_mb_per_s",
    "write_mb_per_s",
    "write_iops_fraction",
    "net_rx_bytes_per_s",
    "net_tx_bytes_per_s",
]
EXCLUDED_COLUMNS = {"timestamp", "session_id", "vm_name", "source_file", "label"}


def select_feature_columns(df):
    candidates = [
        column for column in df.columns
        if column not in EXCLUDED_COLUMNS and column != "index"
    ]
    if not candidates:
        raise ValueError("No feature columns found for training")

    base_available = [column for column in BASE_FEATURE_COLUMNS if column in df.columns]
    if base_available:
        return base_available + [column for column in candidates if column not in base_available]
    return candidates


def main():
    parser = argparse.ArgumentParser(description="Train a LightGBM ransomware classifier.")
    parser.add_argument("--input", type=Path, default=Path("data/labeled_training.csv"))
    parser.add_argument("--model-output", type=Path, default=Path("models/ransomware_lightgbm.txt"))
    parser.add_argument("--metrics-output", type=Path, default=Path("models/ransomware_lightgbm_metrics.json"))
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    if not 0 < args.test_fraction < 1:
        parser.error("--test-fraction must be between 0 and 1")
    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")

    data = pd.read_csv(args.input)
    if "label" not in data.columns:
        raise ValueError(f"Missing label column in {args.input}")

    if "timestamp" in data.columns:
        data = data.sort_values("timestamp").reset_index(drop=True)

    feature_columns = select_feature_columns(data)
    missing = set(["label"]) - set(data.columns)
    if missing:
        raise ValueError(f"Missing required label column: {sorted(missing)}")

    x = data[feature_columns]
    y = data["label"]
    train, test = train_test_split(
        data,
        test_size=args.test_fraction,
        random_state=42,
        stratify=y,
    )

    x_train, y_train = train[feature_columns], train["label"]
    x_test, y_test = test[feature_columns], test["label"]
    positive = max(1, int((y_train == 1).sum()))
    negative = max(1, int((y_train == 0).sum()))

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        class_weight={0: 1.0, 1: negative / positive},
        random_state=42,
        verbosity=-1,
    )
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= args.threshold).astype(int)

    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    importance_df = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    metrics = {
        "input": str(args.input),
        "features": feature_columns,
        "feature_count": len(feature_columns),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_label_counts": {str(k): int(v) for k, v in y_train.value_counts().sort_index().items()},
        "test_label_counts": {str(k): int(v) for k, v in y_test.value_counts().sort_index().items()},
        "threshold": args.threshold,
        "confusion_matrix_labels_0_1": confusion_matrix(y_test, predictions, labels=[0, 1]).tolist(),
        "roc_auc": float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else None,
        "classification_report": report,
        "top_feature_importances": importance_df.head(20).to_dict(orient="records"),
    }

    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(args.model_output))
    args.metrics_output.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"Model: {args.model_output}")
    print(f"Metrics: {args.metrics_output}")
    print(f"Train rows: {len(train)} | test rows: {len(test)}")
    print(f"Feature count: {len(feature_columns)}")
    print(f"Confusion matrix [[TN, FP], [FN, TP]]: {metrics['confusion_matrix_labels_0_1']}")
    print(f"ROC AUC: {metrics['roc_auc']}")
    print("Top feature importances:")
    print(importance_df.head(10).to_string(index=False))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()