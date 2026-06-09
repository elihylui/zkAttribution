#!/usr/bin/env python3
"""S2 trained saboteur — the ZK-positive bookend (2 panels).

A malicious focal (reward = -victim welfare) trained vs the common+α reciprocator
population, two arms: self-report (claim usable) vs verified (claim overridden).

A. Victim welfare over training: with the claim UNVERIFIED the saboteur drives the
   victims to ~0; VERIFIED, it can only env-sabotage and the reciprocators absorb
   it (welfare stays ~1180). The gap = trained ZK marginal value (~+1180, ~86%).
B. The saboteur LEARNS to lie: claimed α rises in the self-report arm (issue-2 does
   not bite for a malicious objective).

Usage: uv run python scripts/plot_selfreport_saboteur.py
"""
import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np

NUM_STEPS = 1000  # victim_mean is per-step; ×NUM_STEPS = episode victim welfare


def _load(path):
    rows = list(csv.DictReader(open(path)))
    return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}


def _smooth(y, w=11):
    if len(y) < w:
        return y
    pad = np.pad(y, w // 2, mode="edge")
    return np.convolve(pad, np.ones(w) / w, mode="valid")[:len(y)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfreport", default="cleanup/selfreport/saboteur_selfreport_seed0.csv")
    ap.add_argument("--verified", default="cleanup/selfreport/saboteur_verified_seed0.csv")
    ap.add_argument("--out", default="cleanup/selfreport/selfreport_saboteur.png")
    args = ap.parse_args()
    S, V = _load(args.selfreport), _load(args.verified)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)

    # --- Panel A: victim welfare ---
    axA.plot(V["env_step"], _smooth(V["victim_mean"] * NUM_STEPS), color="#1b7837", lw=2.4,
             label="verified (claim overridden)")
    axA.plot(S["env_step"], _smooth(S["victim_mean"] * NUM_STEPS), color="#c5283d", lw=2.4,
             label="self-report (claim usable)")
    axA.fill_between(V["env_step"], _smooth(S["victim_mean"] * NUM_STEPS),
                     _smooth(V["victim_mean"] * NUM_STEPS), color="#1b7837", alpha=0.10)
    axA.text(2.5e7, 600, "ZK marginal value\n≈ +1180 (~86% of\nthe commons saved)",
             ha="center", fontsize=9.5, color="#1b7837")
    axA.set_xlabel("environment steps")
    axA.set_ylabel("victim welfare (6 partners' return)")
    axA.set_title("A. Verification saves the commons from a malicious saboteur\n"
                  "(self-report → ~0; verified → reciprocators absorb env-sabotage)",
                  fontsize=11, loc="left")
    axA.grid(True, alpha=0.3); axA.legend(loc="center right", fontsize=9, frameon=True)
    axA.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    # --- Panel B: claimed α (learns to lie) ---
    axB.plot(S["env_step"], _smooth(S["claimed_alpha"]), color="#c5283d", lw=2.4,
             label="self-report (claim acts on partners)")
    axB.plot(V["env_step"], _smooth(V["claimed_alpha"]), color="#7f7f7f", lw=1.8, ls="--",
             label="verified (claim overridden → no gradient pressure)")
    axB.plot(S["env_step"], _smooth(S["true_alpha_focal"]), color="#1f4ea1", lw=1.4, alpha=0.7,
             label="self-report TRUE α (focal barely cleans)")
    axB.set_xlabel("environment steps")
    axB.set_ylabel("focal's claimed α")
    axB.set_title("B. The saboteur LEARNS to lie\n(claimed α rises; true α ~0 → inflation)",
                  fontsize=11, loc="left")
    axB.grid(True, alpha=0.3); axB.legend(loc="upper left", fontsize=8.5, frameon=True)
    axB.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))

    fig.suptitle("S2 trained saboteur: against a reciprocity-using population, lying is the "
                 "dominant attack — and ZK neutralizes it",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
