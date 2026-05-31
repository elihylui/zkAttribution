#!/usr/bin/env python3
"""Cleanup attribution: α-spread vs cooperative basin depth (seed 0 vs seed 1).

Tests the hypothesis: *the better basin (higher return) is reached when the
per-peer cooperation-rate signal α carries more cross-agent information* —
operationalised as a higher cross-agent std of α (`alpha_std`) during the
learning-critical rise.

Both attribution runs logged `alpha` (grand-mean cooperation rate) and
`alpha_std` (cross-agent std of α, averaged over the rollout and the 128
parallel envs) at every update step — so we can pull this retroactively with
no re-run.

Three stacked panels share the x-axis (env_step):
    1. episode return         — where the seeds diverge
    2. alpha_std (α-spread)    — the signal-information metric
    3. alpha (mean coop rate)  — context (note the measurement caveat in docs)

Usage:
    uv run python scripts/plot_cleanup_alpha_spread.py \
        --base cleanup/sweep_results/attr_w50_partial \
        --out  cleanup/sweep_results/attr_w50_partial/cleanup_alpha_spread.png
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from parse_wandb_run import find_wandb_file, read_history

_RUN = "ippo_cleanup_individual_t300000000_ps0_e128_attrV2_w50_seed{seed}"

# Learning-critical "rise" window (env_step) — where seed0 climbs to its basin.
RISE_LO, RISE_HI = 3.0e7, 1.35e8


def load(base, seed):
    d = os.path.join(base, _RUN.format(seed=seed))
    rows = read_history(find_wandb_file(d))
    cols = ("env_step", "alpha", "alpha_std", "returned_episode_returns")
    recs = [[r[k] for k in cols] for r in rows if all(k in r for k in cols)]
    a = np.array(recs, dtype=float)
    a = a[np.argsort(a[:, 0])]
    return dict(es=a[:, 0], alpha=a[:, 1], astd=a[:, 2], ret=a[:, 3])


def smooth(y, w=15):
    if w <= 1:
        return y
    pad = np.pad(y, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[: len(y)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="cleanup/sweep_results/attr_w50_partial")
    ap.add_argument("--out", default="cleanup/sweep_results/attr_w50_partial/cleanup_alpha_spread.png")
    ap.add_argument("--smooth", type=int, default=15)
    args = ap.parse_args()

    S = {s: load(args.base, s) for s in (0, 1)}
    styles = {0: ("#1b7837", "seed 0 (deep basin)"), 1: ("#762a83", "seed 1 (shallow basin)")}

    # --- summary stats ---
    print("window                |  seed0 alpha_std   seed1 alpha_std   (s0 - s1)")
    print("-" * 72)
    for name, lo, hi in [
        ("rise (3e7-1.35e8)", RISE_LO, RISE_HI),
        ("plateau (>70%)",    None,    None),
        ("whole run",         0.0,     np.inf),
    ]:
        cells = []
        for s in (0, 1):
            es, astd = S[s]["es"], S[s]["astd"]
            if name.startswith("plateau"):
                m = es >= 0.70 * es[-1]
            else:
                m = (es >= lo) & (es < hi)
            cells.append(astd[m].mean())
        print(f"  {name:<19} | {cells[0]:14.5f}   {cells[1]:14.5f}   {cells[0]-cells[1]:+.5f}")

    # crossover: last env_step where smoothed s0 astd >= s1 astd
    grid = np.linspace(max(S[0]['es'][0], S[1]['es'][0]), min(S[0]['es'][-1], S[1]['es'][-1]), 1000)
    d0 = np.interp(grid, S[0]['es'], smooth(S[0]['astd'], args.smooth))
    d1 = np.interp(grid, S[1]['es'], smooth(S[1]['astd'], args.smooth))
    above = d0 >= d1
    cross = grid[np.where(above[:-1] & ~above[1:])[0]]
    print(f"\n  alpha_std crossover (s0 falls below s1): "
          f"{cross[-1]:.3e}" if len(cross) else "\n  no crossover (s0 always >= s1)")

    # --- figure ---
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), dpi=150, sharex=True)
    panels = [
        ("ret",  "Episode return", "Episode return (mean/agent)"),
        ("astd", "alpha_std  (cross-agent spread of α)", "std(α) over agents"),
        ("alpha", "alpha  (mean cooperation rate)", "mean α"),
    ]
    for ax, (key, title, ylab) in zip(axes, panels):
        for s in (0, 1):
            c, lab = styles[s]
            ax.plot(S[s]["es"], smooth(S[s][key], args.smooth), color=c, lw=2, label=lab)
        ax.axvspan(RISE_LO, RISE_HI, color="gray", alpha=0.10)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="upper right", frameon=True)
    axes[0].text(
        (RISE_LO + RISE_HI) / 2, axes[0].get_ylim()[1] * 0.18,
        "learning-critical\nrise window", ha="center", color="dimgray", fontsize=8,
    )
    axes[-1].set_xlabel("Environment steps")
    axes[-1].ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    fig.suptitle(
        "Cleanup attribution (individual reward + α obs): "
        "α-spread vs basin depth",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(args.out)
    print(f"\nwrote plot -> {args.out}")


if __name__ == "__main__":
    main()
