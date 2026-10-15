#!/usr/bin/env python3
"""Run an allowlisted ransomware simulator and emit ground-truth events."""

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path


DEVICE = Path("/dev/virtio-ports/org.capstone.ransomsim.0")
ALLOWED_SCRIPTS = {
    "run_full_inplace.py",
    "run_wide_breadth.py",
    "run_full_copydelete.py",
    "run_intermittent.py",
    "run_low_and_slow.py",
    "run_image_level.py",
    "run_header_only.py",
    "run_extension_targeted.py",
    "run_wiper.py",
    "run_multithreaded.py",
    "run_delayed_onset.py",
}
DEFAULT_WATCH_ROOT = Path("/home/rw-lab/testdata")


def send_event(event, script):
    message = {"event": event, "script": script, "ts": time.time()}
    with DEVICE.open("w", encoding="utf-8") as channel:
        channel.write(json.dumps(message) + "\n")
        channel.flush()


def send_event_payload(message):
    with DEVICE.open("w", encoding="utf-8") as channel:
        channel.write(json.dumps(message) + "\n")
        channel.flush()


def snapshot_tree(root):
    snapshot = {}
    if not root.is_dir():
        return snapshot
    for path in root.rglob("*"):
        try:
            stat = path.stat()
        except OSError:
            continue
        relative = str(path.relative_to(root))
        if path.is_file():
            snapshot[relative] = ("file", stat.st_ino, stat.st_size, stat.st_mtime_ns)
        elif path.is_dir():
            snapshot[relative] = ("dir", stat.st_ino, 0, stat.st_mtime_ns)
    return snapshot


def collect_file_activity(root, previous, interval):
    current = snapshot_tree(root)
    created = {
        path for path in set(current) - set(previous)
        if current[path][0] == "file"
    }
    deleted = {
        path for path in set(previous) - set(current)
        if previous[path][0] == "file"
    }
    modified = {
        path for path in set(current) & set(previous)
        if current[path] != previous[path] and current[path][0] == "file"
    }
    deleted_by_inode = {
        previous[path][1]: path for path in deleted if previous[path][0] == "file"
    }
    renamed = {}
    for path in list(created):
        old_path = deleted_by_inode.get(current[path][1])
        if old_path is not None and current[path][0] == "file":
            renamed[old_path] = path
            created.remove(path)
            deleted.remove(old_path)

    changed_paths = set(created) | set(deleted) | modified | set(renamed.values())
    depth_counts = {}
    extensions = {}
    directories = set()
    write_bytes = 0
    for path in changed_paths:
        path_obj = Path(path)
        depth = len(path_obj.parts)
        depth_counts[str(depth)] = depth_counts.get(str(depth), 0) + 1
        directories.add(str(path_obj.parent))
        extension = path_obj.suffix.lower() or "[no_extension]"
        extensions[extension] = extensions.get(extension, 0) + 1
    for path in modified | set(created):
        if path in current and current[path][0] == "file":
            old_size = previous.get(path, (None, 0, 0, 0))[2]
            write_bytes += max(0, current[path][2] - old_size)

    activity = {
        "event": "file_activity",
        "watch_root": str(root),
        "files_modified": len(modified),
        "files_created": len(created),
        "files_renamed": len(renamed),
        "files_deleted": len(deleted),
        "directories_affected": len(directories),
        "max_path_depth": max((len(Path(path).parts) for path in changed_paths), default=0),
        "files_changed_per_depth": depth_counts,
        "change_rate_files_per_s": round(len(changed_paths) / interval, 2),
        "file_extensions_changed": extensions,
        "rename_patterns": [f"{old}->{new}" for old, new in renamed.items()],
        "affected_paths": sorted(changed_paths),
        "file_write_bytes_per_s": round(write_bytes / interval, 2),
        "file_read_bytes_per_s": 0.0,
        "ts": time.time(),
    }
    return activity, current


def monitor_files(root, script, stop_event):
    previous = snapshot_tree(root)
    interval = 1.0
    while not stop_event.wait(interval):
        activity, previous = collect_file_activity(root, previous, interval)
        changed = ("files_modified", "files_created", "files_renamed", "files_deleted")
        if any(activity[field] for field in changed):
            activity["script"] = script
            send_event_payload(activity)


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {Path(sys.argv[0]).name} SCRIPT.py [arguments...]")
        return 2

    watch_root = DEFAULT_WATCH_ROOT
    if "--watch-root" in sys.argv:
        index = sys.argv.index("--watch-root")
        if index + 1 >= len(sys.argv):
            print("--watch-root requires a directory", file=sys.stderr)
            return 2
        watch_root = Path(sys.argv[index + 1]).expanduser().resolve()
        del sys.argv[index:index + 2]

    script_path = Path(sys.argv[1]).expanduser().resolve()
    script_name = script_path.name
    if script_name not in ALLOWED_SCRIPTS:
        print(f"Refusing non-simulator script: {script_name}", file=sys.stderr)
        return 2
    if not script_path.is_file():
        print(f"Script not found: {script_path}", file=sys.stderr)
        return 1

    send_event("start", script_name)
    stop_monitor = threading.Event()
    monitor = threading.Thread(
        target=monitor_files,
        args=(watch_root, script_name, stop_monitor),
        daemon=True,
    )
    monitor.start()
    result = None
    try:
        result = subprocess.run(
            [sys.executable, str(script_path), *sys.argv[2:]],
            cwd=str(script_path.parent),
        )
    finally:
        stop_monitor.set()
        monitor.join(timeout=2)
        send_event("stop", script_name)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())