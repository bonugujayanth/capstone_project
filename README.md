# VM Telemetry and Ransomware Activity Dataset

This project collects host-side telemetry from the `lin` KVM/libvirt virtual machine, saves feature-rich CSV datasets, and supports offline dataset preparation and model training.

## Features

Each telemetry CSV contains 55 feature columns plus metadata:

- CPU, disk write/read throughput, write IOPS, I/O fraction, and network transmit rate
- File modification, creation, deletion, directory, and in-place activity metrics
- Entropy fields when an entropy-capable data source is available
- 5-second and 10-second rolling mean, standard deviation, delta, and slope features
- In-place streak and cumulative modified-file counts
- `timestamp`, `session_id`, `label`, and `strategy` metadata

The existing `capstoneaug.zip` archive is kept unchanged.

## Requirements

Host requirements:

- Linux
- Python 3.10+
- A running KVM/libvirt VM named `lin`
- Access to the system libvirt instance
- `libvirt-python`

Install the Python dependency in the environment used to run the collector:

```bash
python3 -m pip install libvirt-python
```

If your distribution provides it as a system package, use that package instead.

## Run Telemetry

From the project root on the host:

```bash
cd /home/chaithanya/Documents/project
python3 vm_telemetry_collector.py --vm-name lin
```

The collector runs continuously and writes a timestamped CSV under `data/raw/`. Stop it with `Ctrl+C`.

Run for a fixed duration:

```bash
python3 vm_telemetry_collector.py --vm-name lin --duration 10 --interval 1
```

Add dataset metadata:

```bash
python3 vm_telemetry_collector.py \
  --vm-name lin \
  --session-id session_001 \
  --label 1 \
  --strategy run_full_inplace
```

For unavailable source fields, opt into random values in the range `[0, 1]`:

```bash
python3 vm_telemetry_collector.py --vm-name lin --random-missing
```

Use this option only when synthetic fallback values are appropriate for the dataset.

## Dataset Preparation

Merge raw labeled CSV files:

```bash
python3 merge_labeled_data.py \
  --input-dir data/raw \
  --output data/labeled_training.csv
```

Add rolling features to an existing dataset:

```bash
python3 add_rolling_features.py \
  --input data/labeled_training.csv \
  --output data/labeled_training_rolling.csv \
  --windows 5 10
```

Train the model:

```bash
python3 train_lightgbm.py \
  --input data/labeled_training_rolling.csv
```

## Output Locations

- `data/raw/`: timestamped telemetry captures
- `data/simulator_datasets/`: simulator-specific datasets
- `data/labeled_training.csv`: merged labeled dataset
- `data/labeled_training_rolling.csv`: rolling-feature dataset
- `models/`: trained model files and metrics

## Notes

The collector must run on the host because it reads VM counters through libvirt. File-level activity metrics require the corresponding guest-side activity events to be available through the configured VM channel. Entropy values remain zero when no entropy source is available.

Do not commit `.venv-ml`, Python caches, or live prediction logs. They are excluded by `.gitignore`.
