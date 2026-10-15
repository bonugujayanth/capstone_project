# Experiment results

Run from the repository root with:

```bash
./.venv-ml/bin/python run_experiments.py
```

The CSVs are generated from the checked-in training data and simulator replay
datasets. `test_scores.csv` uses a stratified 80/20 split and genuine LightGBM,
XGBoost, random-forest, and SVM models. `timeseries_*.csv` uses the 11 replay
behaviours plus the first 250 rows of `benign.csv` for the normal trace.

`attack_runs.csv` and `unseen.csv` are offline replay measurements. Their alert
is the first LightGBM score at or above 0.5; `files_encrypted_at_alert` uses
the replay dataset's cumulative modified-file field.

`ablation.csv` uses the rolling training dataset and five seeds. Its detection
time is the positive-row index lag in the held-out split, not wall-clock alert
latency. `scalability.csv` measures this local scoring workload at 1, 2, 4,
and 8 repetitions; it is not a multi-VM libvirt benchmark. Fresh live runs
with synchronized attack starts and alert telemetry are required for those
two quantities to become operational measurements.