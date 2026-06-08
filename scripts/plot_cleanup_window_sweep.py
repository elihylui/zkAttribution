#!/usr/bin/env python3
"""Cleanup attribution window sweep — return + alpha_std vs training, per window.

Paired single-seed design: same seed, only the α-window changes (w=25/50/100).
Two stacked panels share the x-axis (env_step):
    1. episode return  — does the basin lift, and does it hold?
    2. alpha_std       — cross-agent α-spread (the mechanism signal)

Usage:
    uv run python scripts/plot_cleanup_window_sweep.py \
        --run 25:/tmp/wsweep/w25/run-tjcsrung.wandb \
        --run 50:cleanup/sweep_results/attr_w50_partial/ippo_cleanup_individual_t300000000_ps0_e128_attrV2_w50_seed0 \
        --run 100:/tmp/wsweep/w100/run-rrrg7ajh.wandb \
        --out cleanup/sweep_results/attr_w50_partial/cleanup_window_sweep_seed0.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from parse_wandb_run import find_wandb_file, read_history

RISE_LO, RISE_HI = 3.0e7, 1.35e8
FLOOR, CEIL = 139.65, 1463.26
COLORS = {25: "#e08214", 50: "#1b7837", 100: "#762a83"}


def load(path):
    wf = path if os.path.isfile(path) else find_wandb_file(path)
    rows = read_history(wf)
    cols = ("env_step", "alpha_std", "returned_episode_returns")
    recs = [[r[k] for k in cols] for r in rows if all(k in r for k in cols)]
    a = np.array(recs, dtype=float)
    a = a[np.argsort(a[:, 0])]
    return dict(es=a[:, 0], astd=a[:, 1], ret=a[:, 2])


def smooth(y, w=15):
    if w <= 1:
        return y
    pad = np.pad(y, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[: len(y)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", action="append", required=True, help="w:path (file or run dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--smooth", type=int, default=15)
    args = ap.parse_args()

    runs = {}
    for spec in args.run:
        w, path = spec.split(":", 1)
        runs[int(w)] = load(path)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), dpi=150, sharex=True)
    for w in sorted(runs):
        d = runs[w]
        c = COLORS.get(w, None)
        axes[0].plot(d["es"], smooth(d["ret"], args.smooth), color=c, lw=2, label=f"w={w}")
        axes[1].plot(d["es"], smooth(d["astd"], args.smooth), color=c, lw=2, label=f"w={w}")

    axes[0].axhline(FLOOR, color="gray", ls=":", lw=1)
    axes[0].text(axes[0].get_xlim()[1], FLOOR, " individual floor", va="bottom",
                 ha="right", color="gray", fontsize=8)
    for ax in axes:
        ax.axvspan(RISE_LO, RISE_HI, color="gray", alpha=0.10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", frameon=True)
    axes[0].set_ylabel("Episode return (mean/agent)")
    axes[0].set_title("Cleanup attribution window sweep (seed 0): return", fontsize=11, loc="left")
    axes[1].set_ylabel("std(α) over agents")
    axes[1].set_title("alpha_std (cross-agent α-spread)", fontsize=11, loc="left")
    axes[1].set_xlabel("Environment steps")
    axes[1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle("Attribution α-window sweep — w=25 / 50 / 100 (seed 0)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out)
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
