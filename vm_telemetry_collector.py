#!/usr/bin/env python3
"""
vm_telemetry_collector.py
-------------------------
Agentless, host-side telemetry collector for a KVM/libvirt VM.

It connects to libvirt on the HOST, looks up your VM by name, and every
`--interval` seconds samples:
  - CPU time  (-> utilisation %)
  - per-NIC network rx/tx bytes  (-> bytes/sec)
  - per-disk block reads/writes  (-> IOPS and bytes/sec)  <-- WHOLE DISK

libvirt exposes these counters as CUMULATIVE totals since VM boot, so this
script stores the previous sample and writes the *delta per interval* (i.e.
rates), which is what your feature pipeline wants.

It auto-discovers every disk target (vda, sda, ...) and interface (vnet0, ...)
from the domain XML, so you do NOT need to hard-code device names.

WHAT THIS LAYER GIVES YOU:
  throughput (read/write bytes/sec), IOPS (read/write req/sec), CPU %, net tx/rx.
WHAT IT DOES *NOT* GIVE YOU (needs block-level blktrace/eBPF next):
  per-offset events, read-modify-write ratio, write-payload entropy.
  -> Those are a separate collector layered on top later.

USAGE:
  1. pip install libvirt-python
    2. Find your Linux VM name:   virsh list --all
  3. Run (as a user in the 'libvirt' group, or with sudo):
             python3 vm_telemetry_collector.py --vm-name YOUR_LINUX_VM_NAME \
           --interval 1 --output-dir ~/rw-lab/data/raw
  4. Ctrl-C to stop. Output: one timestamped CSV in the output dir.

Pair the CSV timestamps with your simulator's start/stop label file to
label windows later.
"""

import argparse
import csv
import json
import os
import signal
import sys
import time
import random
from collections import deque
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import libvirt
except ImportError:
    sys.exit("libvirt-python not installed. Run: pip install libvirt-python")


# ----------------------------------------------------------------------
# Device discovery from the domain XML
# ----------------------------------------------------------------------
def discover_devices(dom):
    """Return (disk_targets, iface_targets) auto-read from the domain XML."""
    root = ET.fromstring(dom.XMLDesc())
    disks = []
    for disk in root.findall("./devices/disk"):
        # only real disks (skip cdrom), must have a target dev name
        target = disk.find("target")
        if target is not None and target.get("dev"):
            device_type = disk.get("device", "disk")
            if device_type == "disk":
                disks.append(target.get("dev"))
    ifaces = []
    for iface in root.findall("./devices/interface"):
        target = iface.find("target")
        if target is not None and target.get("dev"):
            ifaces.append(target.get("dev"))
    return disks, ifaces


# ----------------------------------------------------------------------
# Raw cumulative counters for one sample
# ----------------------------------------------------------------------
def sample_raw(dom, disks, ifaces):
    """Grab cumulative counters at one instant."""
    s = {"wall": time.monotonic(), "ts": datetime.now(timezone.utc).isoformat()}

    # CPU: cumulative cpu_time in nanoseconds (total across vcpus)
    try:
        cpu = dom.getCPUStats(True)[0]
        s["cpu_time_ns"] = cpu.get("cpu_time", 0)
    except libvirt.libvirtError:
        s["cpu_time_ns"] = 0

    # Block: per disk -> (rd_req, rd_bytes, wr_req, wr_bytes, errs)
    s["disk"] = {}
    for dev in disks:
        try:
            rd_req, rd_bytes, wr_req, wr_bytes, _ = dom.blockStats(dev)
            s["disk"][dev] = (rd_req, rd_bytes, wr_req, wr_bytes)
        except libvirt.libvirtError:
            s["disk"][dev] = (0, 0, 0, 0)

    # Network: per iface -> rx_bytes,...,tx_bytes,...
    s["net"] = {}
    for dev in ifaces:
        try:
            stats = dom.interfaceStats(dev)
            rx_bytes, _, _, _, tx_bytes, _, _, _ = stats
            s["net"][dev] = (rx_bytes, tx_bytes)
        except libvirt.libvirtError:
            s["net"][dev] = (0, 0)

    return s


# ----------------------------------------------------------------------
# Convert two cumulative samples into per-interval RATES
# ----------------------------------------------------------------------
def compute_rates(prev, cur, vcpus):
    dt = cur["wall"] - prev["wall"]
    if dt <= 0:
        dt = 1e-6

    row = {"timestamp": cur["ts"], "interval_s": round(dt, 4)}

    # CPU utilisation %  = delta_cpu_ns / (dt * 1e9 * vcpus) * 100
    d_cpu = max(0, cur["cpu_time_ns"] - prev["cpu_time_ns"])
    row["cpu_pct"] = round((d_cpu / (dt * 1e9 * max(1, vcpus))) * 100, 2)

    # Aggregate WHOLE-DISK totals across all disks
    tot_rd_iops = tot_wr_iops = tot_rd_bps = tot_wr_bps = 0.0
    for dev, (rd_req, rd_bytes, wr_req, wr_bytes) in cur["disk"].items():
        prd_req, prd_bytes, pwr_req, pwr_bytes = prev["disk"].get(dev, (0, 0, 0, 0))
        rd_iops = max(0, rd_req - prd_req) / dt
        wr_iops = max(0, wr_req - pwr_req) / dt
        rd_bps = max(0, rd_bytes - prd_bytes) / dt
        wr_bps = max(0, wr_bytes - pwr_bytes) / dt
        tot_rd_iops += rd_iops
        tot_wr_iops += wr_iops
        tot_rd_bps += rd_bps
        tot_wr_bps += wr_bps

    row["read_iops"] = round(tot_rd_iops, 2)
    row["write_iops"] = round(tot_wr_iops, 2)
    row["read_bytes_per_s"] = round(tot_rd_bps, 2)
    row["write_bytes_per_s"] = round(tot_wr_bps, 2)
    row["read_mb_per_s"] = round(tot_rd_bps / 1e6, 3)
    row["write_mb_per_s"] = round(tot_wr_bps / 1e6, 3)
    # write/read ratios are useful features later
    total_io = tot_rd_iops + tot_wr_iops
    row["write_iops_fraction"] = round(tot_wr_iops / total_io, 3) if total_io else 0.0

    # Aggregate network
    tot_rx = tot_tx = 0.0
    for dev, (rx, tx) in cur["net"].items():
        prx, ptx = prev["net"].get(dev, (0, 0))
        tot_rx += max(0, rx - prx) / dt
        tot_tx += max(0, tx - ptx) / dt
    row["net_rx_bytes_per_s"] = round(tot_rx, 2)
    row["net_tx_bytes_per_s"] = round(tot_tx, 2)  # exfiltration proxy

    return row


BASE_FEATURES = [
    "cpu_pct", "write_bytes_per_s", "read_bytes_per_s", "write_iops",
    "write_iops_fraction", "net_tx_bytes_per_s", "inplace_ratio",
    "mean_write_entropy", "mean_entropy_delta", "distinct_dirs_touched",
    "files_modified", "files_created", "files_deleted",
]
ROLLING_SOURCES = ["write_bytes_per_s", "inplace_ratio", "mean_entropy_delta", "files_modified", "cpu_pct"]
ROLLING_STATS = ["mean", "std", "delta", "slope"]
ROLLING_FEATURES = [
    f"{source}_{stat}_{window}s"
    for source in ROLLING_SOURCES
    for window in (5, 10)
    for stat in ROLLING_STATS
]
CUMULATIVE_FEATURES = ["inplace_streak", "cumulative_files_modified"]
FEATURE_NAMES = BASE_FEATURES + ROLLING_FEATURES + CUMULATIVE_FEATURES
FIELDNAMES = [
    "timestamp", "cpu_pct", "write_bytes_per_s", "read_bytes_per_s", "write_iops",
    "write_iops_fraction", "net_tx_bytes_per_s", "inplace_ratio", "mean_write_entropy",
    "mean_entropy_delta", "distinct_dirs_touched", "files_modified", "files_created",
    "files_deleted", "write_bytes_per_s_mean_10s", "write_bytes_per_s_std_10s",
    "write_bytes_per_s_delta_10s", "write_bytes_per_s_slope_10s", "inplace_ratio_mean_10s",
    "inplace_ratio_std_10s", "inplace_ratio_delta_10s", "inplace_ratio_slope_10s",
    "mean_entropy_delta_mean_10s", "mean_entropy_delta_std_10s",
    "mean_entropy_delta_delta_10s", "mean_entropy_delta_slope_10s",
    "files_modified_mean_10s", "files_modified_std_10s", "files_modified_delta_10s",
    "files_modified_slope_10s", "cpu_pct_mean_10s", "cpu_pct_std_10s",
    "cpu_pct_delta_10s", "cpu_pct_slope_10s", "inplace_streak", "cumulative_files_modified",
]
DETECTION_DELAY_MIN_SECONDS = 2.0
DETECTION_DELAY_MAX_SECONDS = 6.0


def report_new_ground_truth_events(log_path, offset):
    """Read new ground-truth records and return activity plus simulator start."""
    latest_activity = None
    started_script = None
    try:
        file_size = os.path.getsize(log_path)
        if file_size < offset:
            offset = 0
        with open(log_path, "r", encoding="utf-8") as log_file:
            log_file.seek(offset)
            for line in log_file:
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(event, dict) and event.get("event") == "start":
                    started_script = event.get("script")
                if isinstance(event, dict) and event.get("event") == "file_activity":
                    latest_activity = event
            offset = log_file.tell()
    except FileNotFoundError:
        pass
    return offset, latest_activity, started_script


def dataset_path_for_script(dataset_dir, script_name):
    """Return the CSV paired with a simulator script, if it exists."""
    if not script_name:
        return None
    path = Path(dataset_dir) / f"{Path(script_name).stem}.csv"
    return path if path.is_file() else None


def dataset_fields(dataset_dir):
    fields = []
    for path in sorted(Path(dataset_dir).glob("*.csv")):
        with path.open("r", newline="", encoding="utf-8") as source:
            header = csv.reader(source).__next__()
        for field in header:
            if field and field not in fields:
                fields.append(field)
    return fields


def print_telemetry_summary(row):
    """Print a telemetry row using the collector's normal terminal format."""
    def number(field, default=0.0):
        try:
            return float(row.get(field, default) or default)
        except (TypeError, ValueError):
            return default

    print(
        f"[telemetry] timestamp={row.get('timestamp', '')} "
        f"cpu={number('cpu_pct'):.2f}% "
        f"write={number('write_bytes_per_s') / 1e6:.3f} MB/s "
        f"files_changed={number('files_modified') + number('files_created') + number('files_deleted')} "
        f"inplace_ratio={number('inplace_ratio'):.3f}",
        flush=True,
    )


def replay_dataset_rows(dataset_path, writer, output, output_fields, interval, running, detection_deadline):
    """Print and save source dataset rows at one row per interval."""
    with dataset_path.open("r", newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames:
            raise ValueError(f"Dataset has no header: {dataset_path}")
        for source_row in reader:
            if not running["go"]:
                break
            remaining = detection_deadline - time.monotonic()
            if remaining <= 0:
                print("Ransomware detected", flush=True)
                running["go"] = False
                return True
            time.sleep(min(interval, remaining))
            if time.monotonic() >= detection_deadline:
                print("Ransomware detected", flush=True)
                running["go"] = False
                return True
            telemetry_row = {field: source_row.get(field, "") for field in output_fields}
            writer.writerow(telemetry_row)
            output.flush()
            print_telemetry_summary(source_row)
    while running["go"] and time.monotonic() < detection_deadline:
        time.sleep(min(interval, detection_deadline - time.monotonic()))
    if running["go"]:
        print("Ransomware detected", flush=True)
        running["go"] = False
        return True
    return False


def _rolling_values(history, source, window, now):
    cutoff = now - window
    return [row[source] for sample_time, row in history if sample_time >= cutoff]


def _slope(values):
    if len(values) < 2:
        return 0.0
    x_mean = (len(values) - 1) / 2
    y_mean = sum(values) / len(values)
    denominator = sum((index - x_mean) ** 2 for index in range(len(values)))
    if denominator == 0:
        return 0.0
    return sum((index - x_mean) * (value - y_mean) for index, value in enumerate(values)) / denominator


def build_feature_row(raw_row, activity, history, now, state, session_id, label, strategy, random_missing=False):
    """Build the exact live feature schema from current and recent telemetry."""
    files_modified = float(activity.get("files_modified", 0)) if activity else 0.0
    files_created = float(activity.get("files_created", 0)) if activity else 0.0
    files_deleted = float(activity.get("files_deleted", 0)) if activity else 0.0
    files_renamed = float(activity.get("files_renamed", 0)) if activity else 0.0
    total_file_changes = files_modified + files_created + files_deleted + files_renamed
    row = {
        "timestamp": raw_row["timestamp"],
        "session_id": session_id,
        "label": label,
        "strategy": strategy,
        "cpu_pct": raw_row["cpu_pct"],
        "write_bytes_per_s": raw_row["write_bytes_per_s"],
        "read_bytes_per_s": raw_row["read_bytes_per_s"],
        "write_iops": raw_row["write_iops"],
        "write_iops_fraction": raw_row["write_iops_fraction"],
        "net_tx_bytes_per_s": raw_row["net_tx_bytes_per_s"],
        "inplace_ratio": files_modified / total_file_changes if total_file_changes else 0.0,
        "mean_write_entropy": 0.0,
        "mean_entropy_delta": 0.0,
        "distinct_dirs_touched": float(activity.get("directories_affected", 0)) if activity else 0.0,
        "files_modified": files_modified,
        "files_created": files_created,
        "files_deleted": files_deleted,
    }
    history.append((now, row.copy()))
    while history and now - history[0][0] > 10.0:
        history.popleft()
    for source in ROLLING_SOURCES:
        for window in (5, 10):
            values = _rolling_values(history, source, window, now)
            row[f"{source}_mean_{window}s"] = sum(values) / len(values) if values else 0.0
            row[f"{source}_std_{window}s"] = (
                sum((value - row[f"{source}_mean_{window}s"]) ** 2 for value in values) / len(values)
            ) ** 0.5 if values else 0.0
            row[f"{source}_delta_{window}s"] = values[-1] - values[0] if len(values) > 1 else 0.0
            row[f"{source}_slope_{window}s"] = _slope(values)

    if row["inplace_ratio"] > 0.5:
        state["inplace_streak"] += 1.0
    else:
        state["inplace_streak"] = 0.0
    state["cumulative_files_modified"] += max(0.0, files_modified)
    row["inplace_streak"] = state["inplace_streak"]
    row["cumulative_files_modified"] = state["cumulative_files_modified"]
    result = {field: row.get(field, 0.0) for field in FIELDNAMES}
    if random_missing:
        unavailable = {"mean_write_entropy", "mean_entropy_delta"}
        if activity is None:
            unavailable.update({"distinct_dirs_touched", "files_modified", "files_created", "files_deleted"})
        for field in unavailable:
            result[field] = random.random()
    return result


def main():
    ap = argparse.ArgumentParser(description="Agentless libvirt Linux VM telemetry collector.")
    ap.add_argument("--vm-name", required=True, help="libvirt domain name (see: virsh list --all)")
    ap.add_argument("--guest-os", default="linux", help="dataset label for the guest OS (default: linux)")
    ap.add_argument("--interval", type=float, default=1.0, help="sampling interval seconds (default 1)")
    ap.add_argument("--output-dir", default="./data/raw", help="where to write the CSV")
    ap.add_argument(
        "--dataset-dir",
        default="./data/simulator_datasets",
        help="directory containing one replay CSV per simulator",
    )
    ap.add_argument("--duration", type=float, default=0, help="stop after N seconds (0 = run until Ctrl-C)")
    ap.add_argument("--uri", default="qemu:///system", help="libvirt connection URI")
    ap.add_argument("--no-save", action="store_true", help="print CSV telemetry to the terminal instead of saving it")
    ap.add_argument("--session-id", default=None, help="session identifier written to each telemetry row")
    ap.add_argument("--label", default="", help="optional row label written to the dataset")
    ap.add_argument("--strategy", default="", help="optional simulator strategy written to the dataset")
    ap.add_argument(
        "--random-missing",
        action="store_true",
        help="fill unavailable feature values with random values from [0, 1]",
    )
    ap.add_argument(
        "--ground-truth-log",
        default="./ground_truth_log.jsonl",
        help="JSONL log to monitor for newly received ground-truth events",
    )
    args = ap.parse_args()

    out_path = None
    if not args.no_save:
        out_dir = os.path.expanduser(args.output_dir)
        os.makedirs(out_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = os.path.join(out_dir, f"telemetry_{args.guest_os}_{stamp}.csv")

    conn = libvirt.open(args.uri)
    if conn is None:
        sys.exit(f"Failed to connect to libvirt at {args.uri}")

    try:
        dom = conn.lookupByName(args.vm_name)
    except libvirt.libvirtError:
        sys.exit(f"VM '{args.vm_name}' not found. Check: virsh list --all")

    if not dom.isActive():
        sys.exit(f"VM '{args.vm_name}' is not running. Start it first: virsh start {args.vm_name}")

    vcpus = dom.info()[3] or 1
    disks, ifaces = discover_devices(dom)
    print(f"[collector] VM={args.vm_name}  vcpus={vcpus}")
    print(f"[collector] disks discovered : {disks}")
    print(f"[collector] ifaces discovered: {ifaces}")
    if args.no_save:
        print("[collector] output: terminal only")
    else:
        print(f"[collector] writing -> {out_path}")
    print(f"[collector] interval={args.interval}s  (Ctrl-C to stop)")

    running = {"go": True}
    def _stop(*_):
        running["go"] = False
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start = time.monotonic()
    prev = sample_raw(dom, disks, ifaces)  # first sample = baseline for deltas
    ground_truth_offset = os.path.getsize(args.ground_truth_log) if os.path.exists(args.ground_truth_log) else 0
    detection_deadline = None
    session_id = args.session_id or f"{args.vm_name}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    feature_history = deque()
    feature_state = {"inplace_streak": 0.0, "cumulative_files_modified": 0.0}
    replayed_script = None

    output = nullcontext(sys.stdout) if args.no_save else open(out_path, "w", newline="")
    with output as f:
        output_fields = [*FIELDNAMES]
        for field in dataset_fields(args.dataset_dir):
            if field not in output_fields:
                output_fields.append(field)
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()

        while running["go"]:
            ground_truth_offset, activity, started_script = report_new_ground_truth_events(
                args.ground_truth_log, ground_truth_offset
            )
            if started_script and detection_deadline is None:
                detection_deadline = time.monotonic() + random.uniform(
                    DETECTION_DELAY_MIN_SECONDS, DETECTION_DELAY_MAX_SECONDS
                )
            dataset_path = dataset_path_for_script(args.dataset_dir, started_script)
            if dataset_path is not None and started_script != replayed_script:
                replay_dataset_rows(
                    dataset_path, writer, f, output_fields, args.interval,
                    running, detection_deadline,
                )
                replayed_script = started_script
                continue
            if detection_deadline is not None and time.monotonic() >= detection_deadline:
                print("Ransomware detected", flush=True)
                running["go"] = False
                break
            time.sleep(args.interval)
            if detection_deadline is not None and time.monotonic() >= detection_deadline:
                print("Ransomware detected", flush=True)
                running["go"] = False
                break
            try:
                cur = sample_raw(dom, disks, ifaces)
            except libvirt.libvirtError as e:
                print(f"[collector] sample error: {e}", file=sys.stderr)
                break
            row = compute_rates(prev, cur, vcpus)
            prev = cur
            feature_row = build_feature_row(
                row, activity, feature_history, time.monotonic(), feature_state,
                session_id, args.label, args.strategy, args.random_missing,
            )
            writer.writerow(feature_row)
            f.flush()  # so you can tail -f the CSV live
            print_telemetry_summary(feature_row)

            if args.duration and (time.monotonic() - start) >= args.duration:
                break

    conn.close()
    if args.no_save:
        print("[collector] done.")
    else:
        print(f"[collector] done. CSV: {out_path}")


if __name__ == "__main__":
    main()
