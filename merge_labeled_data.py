#!/usr/bin/env python3
"""Merge benign and ransomware telemetry CSVs with binary labels."""

import argparse
import csv
from pathlib import Path


BENIGN_FILE = "benign.csv"
RANSOMWARE_FILES = (
    "full_copy_delete.csv",
    "fullinplace.csv",
    "low_and_slow.csv",
    "run_full_copy_delete.csv",
    "wide_breath.csv",
)
FILE_ACTIVITY_FIELDS = (
    "files_modified", "files_created", "files_renamed", "files_deleted",
    "directories_affected", "max_path_depth", "files_changed_per_depth",
    "change_rate_files_per_s", "file_extensions_changed", "rename_patterns",
    "affected_paths", "file_write_bytes_per_s", "file_read_bytes_per_s",
)
JSON_ACTIVITY_FIELDS = {
    "files_changed_per_depth", "file_extensions_changed", "rename_patterns", "affected_paths",
}


def read_rows(path, label):
    with path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"{path} has no CSV header")
        for row in reader:
            row["label"] = label
            yield row, reader.fieldnames


def merge(input_dir, output_path):
    sources = [(input_dir / BENIGN_FILE, 0)]
    sources.extend((input_dir / name, 1) for name in RANSOMWARE_FILES)

    rows = []
    source_fields = []
    counts = {0: 0, 1: 0}
    for source_path, label in sources:
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        for row, fields in read_rows(source_path, label):
            rows.append(row)
            source_fields.extend(field for field in fields if field not in source_fields)
            counts[label] += 1

    if not rows:
        raise ValueError("No data rows found")

    fieldnames = [*source_fields]
    for field in FILE_ACTIVITY_FIELDS:
        if field not in fieldnames:
            fieldnames.append(field)
    fieldnames.append("label")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            for field in FILE_ACTIVITY_FIELDS:
                if field not in row:
                    row[field] = "{}" if field in JSON_ACTIVITY_FIELDS else 0
            writer.writerow(row)
    return counts


def main():
    parser = argparse.ArgumentParser(description="Create a labelled ML training CSV.")
    parser.add_argument("--input-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output", type=Path, default=Path("data/labeled_training.csv"))
    args = parser.parse_args()
    counts = merge(args.input_dir, args.output)
    print(f"Wrote {sum(counts.values())} rows to {args.output}")
    print(f"label 0 (benign): {counts[0]}")
    print(f"label 1 (ransomware): {counts[1]}")
    print("Excluded: creatingfiles.csv")


if __name__ == "__main__":
    main()