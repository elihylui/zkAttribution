#!/usr/bin/env python3
"""Cleanup attribution — whole-picture overview (2 panels).

Panel A (training): episode return vs env_step for the four seed-0 arms —
  individual floor / verified per-agent α / spread-collapse (mean α) / common
  ceiling. Shows that the *aggregate* α (spread-collapse) trains HIGHER than the
  *per-agent* α (verified): the per-agent breakdown hurts.

Panel B (inference ablation): return of the trained verified policy when the α
  it sees at inference is swapped (zero / real / permuted / random), annotated
  with clean-action rate and dirt — the converged policy does best with NO
  signal, and every perturbation raises (futile) cleaning + dirt while lowering
  return (coordination degradation).

Usage:
    uv run python scripts/plot_cleanup_attr_overview.py --out cleanup/sweep_results/attr_w50_partial/cleanup_attr_overview.png
"""
import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from parse_wandb_run import read_history

_BASE = "cleanup/sweep_results"
ARMS = [  # label, color, run-dir (any *.wandb under it is used)
    ("individual (floor)", "#7f7f7f",
     f"{_BASE}/baseline_3e8_complete/ippo_cleanup_individual_t300000000_ps0_e128_seed0"),
    ("verified — per-agent α", "#1f4ea1",
     f"{_BASE}/attr_w50_partial/ippo_cleanup_individual_t300000000_ps0_e128_attrV2_w50_seed0"),
    ("spread-collapse — mean α", "#e08214",
     f"{_BASE}/attr_w50_partial/ippo_cleanup_individual_t300000000_ps0_e128_attrV2_w50_sc_seed0"),
    ("common (ceiling)", "#222222",
     f"{_BASE}/baseline_3e8_complete/ippo_cleanup_common_t300000000_ps0_e128_seed0"),
]

# Inference α-usage ablation (verified w=50 seed0, all-attr pop, 24 eps).
ABLATION = [  # label, return, clean-action rate, dirt, color
    ("zero",        914, 0.078,  55.8, "#1b7837"),
    ("real",        643, 0.100,  64.6, "#1f4ea1"),
    ("perm/window", 361, 0.107,  73.5, "#e08214"),
    ("random",        0, 0.066, 138.8, "#c5283d"),
]


def load(path):
    wf = path if path.endswith(".wandb") else (
        sorted(glob.glob(os.path.join(path, "**", "*.wandb"), recursive=True)) or [None])[0]
    if not wf:
        return None, None
    rows = read_history(wf)
    recs = [(r["env_step"], r["returned_episode_returns"]) for r in rows
            if "env_step" in r and "returned_episode_returns" in r]
    if not recs:
        return None, None
    a = np.array(recs, dtype=float)
    a = a[a[:, 0].argsort()]
    return a[:, 0], a[:, 1]


def smooth(y, w=15):
    if w <= 1 or len(y) < w:
        return y
    pad = np.pad(y, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{_BASE}/attr_w50_partial/cleanup_attr_overview.png")
    ap.add_argument("--smooth", type=int, default=15)
    args = ap.parse_args()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=150,
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # --- Panel A: training curves ---
    for label, color, path in ARMS:
        es, ret = load(path)
        if es is None:
            print(f"  SKIP {label}: no data ({path})")
            continue
        ls = "--" if "ceiling" in label else "-"
        axA.plot(es, smooth(ret, args.smooth), color=color, lw=2.2, ls=ls,
                 label=f"{label}  (~{smooth(ret, args.smooth)[int(len(ret)*0.9):].mean():.0f})")
        print(f"  load {label}: {len(es)} pts, plateau ~{ret[int(len(ret)*0.9):].mean():.0f}")
    axA.set_xlabel("Environment steps")
    axA.set_ylabel("Episode return (mean/agent)")
    axA.set_title("A. Training: aggregate α (spread-collapse) > per-agent α (verified)",
                  fontsize=11, loc="left")
    axA.grid(True, alpha=0.3)
    axA.legend(loc="upper left", frameon=True, fontsize=9)
    axA.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # --- Panel B: inference ablation ---
    labels = [a[0] for a in ABLATION]
    rets = [a[1] for a in ABLATION]
    colors = [a[4] for a in ABLATION]
    x = np.arange(len(labels))
    axB.bar(x, rets, color=colors, width=0.65)
    for i, (lab, ret, cr, dl, c) in enumerate(ABLATION):
        axB.text(i, ret + 18, f"{ret}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        axB.text(i, max(ret * 0.5, 40), f"clean {cr:.2f}\ndirt {dl:.0f}",
                 ha="center", va="center", fontsize=8, color="white" if ret > 200 else "black")
    axB.set_xticks(x)
    axB.set_xticklabels(labels)
    axB.set_ylabel("Return/agent at inference")
    axB.set_title("B. Verified policy under α-swaps:\nbest with NO signal; perturbing → more dirt, less return",
                  fontsize=11, loc="left")
    axB.set_xlabel("α shown to the (verified-trained) policy")
    axB.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Cleanup attribution (seed 0, w=50): less α information is better",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
