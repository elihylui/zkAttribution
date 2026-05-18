#!/usr/bin/env python3
"""Parse a wandb *offline* run's binary datastore into per-step metrics.

Reads the `run-*.wandb` datastore directly — no `wandb sync` or cloud account
needed. Useful for inspecting training runs that logged with WANDB_MODE=offline.

Usage:
    uv run python scripts/parse_wandb_run.py <run_dir_or_.wandb_file> [--csv OUT]

Without --csv, prints a sampled summary of the key metrics over training.
"""

import argparse
import csv
import glob
import json
import os

from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal.datastore import DataStore


# Cooperation / training metrics worth surfacing in the summary, in display order.
_SUMMARY_KEYS = [
    "env_step",
    "update_steps",
    "cleaned_water",
    "clean_action_info",
    "returned_episode_returns",
    "returned_episode_lengths",
    "original_rewards",
    "shaped_rewards",
]


def find_wandb_file(path: str) -> str:
    """Resolve a run directory or direct file path to the run-*.wandb file."""
    if os.path.isfile(path) and path.endswith(".wandb"):
        return path
    matches = glob.glob(os.path.join(path, "run-*.wandb"))
    if not matches:
        # Fall back to a recursive search (e.g. path is a run dir whose
        # datastore lives under wandb/offline-run-*/).
        matches = glob.glob(os.path.join(path, "**", "run-*.wandb"), recursive=True)
    if not matches:
        raise FileNotFoundError(f"no run-*.wandb datastore found under {path}")
    return sorted(matches)[-1]


def read_history(wandb_file: str) -> list[dict]:
    """Return a list of per-step metric dicts from the run datastore."""
    ds = DataStore()
    ds.open_for_scan(wandb_file)
    rows: list[dict] = []
    while True:
        rec = ds.scan_data()
        if rec is None:
            break
        record = pb.Record()
        record.ParseFromString(rec)
        if record.WhichOneof("record_type") != "history":
            continue
        row: dict = {}
        for item in record.history.item:
            key = item.key or ".".join(item.nested_key)
            try:
                row[key] = json.loads(item.value_json)
            except (json.JSONDecodeError, ValueError):
                row[key] = item.value_json
        rows.append(row)
    # Sort by wandb step so the trajectory is in training order.
    rows.sort(key=lambda r: r.get("_step", 0))
    return rows


def print_summary(rows: list[dict]) -> None:
    if not rows:
        print("no history records found")
        return
    print(f"{len(rows)} history records (update steps)\n")
    present = [k for k in _SUMMARY_KEYS if any(k in r for r in rows)]
    # Sample at 0%, 25%, 50%, 75%, 100% of training.
    n = len(rows)
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1})
    col_w = max(24, *(len(k) for k in present))
    header = "metric".ljust(col_w) + "".join(f"step {rows[i].get('_step', i):>10}" for i in idxs)
    print(header)
    print("-" * len(header))
    for k in present:
        line = k.ljust(col_w)
        for i in idxs:
            v = rows[i].get(k)
            line += (f"{v:>15.3f}" if isinstance(v, (int, float)) else f"{str(v):>15}")
        print(line)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="wandb offline run directory or .wandb file")
    ap.add_argument("--csv", help="write full per-step metrics to this CSV path")
    args = ap.parse_args()

    wandb_file = find_wandb_file(args.path)
    rows = read_history(wandb_file)

    if args.csv:
        keys: list[str] = []
        for r in rows:
            for k in r:
                if k not in keys:
                    keys.append(k)
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        print(f"wrote {len(rows)} rows x {len(keys)} cols -> {args.csv}")
    else:
        print_summary(rows)


if __name__ == "__main__":
    main()
