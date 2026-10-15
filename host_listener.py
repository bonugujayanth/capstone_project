#!/usr/bin/env python3
"""Receive guest simulation labels over the private virtio-serial socket."""

import argparse
import json
import socket
import sys
import time
from datetime import datetime, timezone


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def receive_events(socket_path, log_path, reconnect_delay):
    print(f"[GROUND TRUTH] listening on {socket_path}", flush=True)
    with open(log_path, "a", encoding="utf-8", buffering=1) as log_file:
        while True:
            try:
                print(f"[GROUND TRUTH] connecting to {socket_path}", flush=True)
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                    connection.connect(socket_path)
                    print("[GROUND TRUTH] connected", flush=True)
                    with connection.makefile("r", encoding="utf-8", newline="") as stream:
                        for line in stream:
                            if not line.strip():
                                continue
                            try:
                                event = json.loads(line)
                                if not isinstance(event, dict):
                                    raise ValueError("message is not a JSON object")
                            except (json.JSONDecodeError, ValueError) as error:
                                print(f"[GROUND TRUTH] ignored invalid message: {error}", file=sys.stderr, flush=True)
                                continue

                            record = dict(event)
                            record["host_received_ts"] = utc_now()
                            log_file.write(json.dumps(record, sort_keys=True) + "\n")
                            log_file.flush()
                            print(f"[GROUND TRUTH] {json.dumps(record, sort_keys=True)}", flush=True)
            except KeyboardInterrupt:
                print("[GROUND TRUTH] stopped", flush=True)
                return
            except (FileNotFoundError, ConnectionRefusedError, PermissionError, OSError) as error:
                print(f"[GROUND TRUTH] connection unavailable: {error}; retrying", file=sys.stderr, flush=True)
                try:
                    time.sleep(reconnect_delay)
                except KeyboardInterrupt:
                    print("[GROUND TRUTH] stopped", flush=True)
                    return


def main():
    parser = argparse.ArgumentParser(description="Listen for ransomware simulation ground-truth events.")
    parser.add_argument("--socket", default="/var/lib/libvirt/qemu/ransomsim.sock")
    parser.add_argument("--log", default="ground_truth_log.jsonl")
    parser.add_argument("--reconnect-delay", type=float, default=2.0)
    args = parser.parse_args()
    if args.reconnect_delay <= 0:
        parser.error("--reconnect-delay must be positive")
    receive_events(args.socket, args.log, args.reconnect_delay)


if __name__ == "__main__":
    main()