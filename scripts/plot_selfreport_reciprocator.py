#!/usr/bin/env python3
"""Stage-4 reciprocator arc (2 panels).

A. α-response by regime: common+α genuinely RECIPROCATES (real α > zero α), the
   reverse of individual+α (zero > real). So a real reciprocator exists.
B. Exploiting it: a member of the common+α reciprocity population sweeping its
   self-reported claim — over-reporting (claim>0) CATASTROPHICALLY collapses the
   commons (dirt 42→103, return 964→90). Lying is maximally self-defeating; the
   more a population uses the signal, the more a lie destroys what the liar needs.

Usage: uv run python scripts/plot_selfreport_reciprocator.py
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np

# Homogeneous α-response (all-attr ablation), return/agent at real vs zero α.
RESP = {
    "individual+α\n(anti-uses α)": {"real": 643, "zero": 914},   # zero > real
    "common+α\n(reciprocates)":    {"real": 1312, "zero": 690},  # real > zero
}
# Closing exploit sweep: common+α member focal, per-agent α, claim 0→1.
SW_CLAIM = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SW_RET = [964.2, 90.1, 38.5, 65.6, 71.8, 83.1, 76.1, 60.4, 42.0, 42.9, 62.1]
SW_DIRT = [41.8, 103.1, 103.0, 103.2, 98.4, 97.8, 97.1, 93.1, 94.0, 93.8, 96.5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="cleanup/selfreport/selfreport_reciprocator.png")
    args = ap.parse_args()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=150)

    # --- Panel A: α-response by regime ---
    labels = list(RESP)
    x = np.arange(len(labels))
    w = 0.36
    real = [RESP[k]["real"] for k in labels]
    zero = [RESP[k]["zero"] for k in labels]
    axA.bar(x - w / 2, real, w, color="#1f4ea1", label="real α (signal on)")
    axA.bar(x + w / 2, zero, w, color="#7f7f7f", label="zero α (signal off)")
    for i in range(len(labels)):
        axA.text(x[i] - w / 2, real[i] + 20, str(real[i]), ha="center", fontsize=9)
        axA.text(x[i] + w / 2, zero[i] + 20, str(zero[i]), ha="center", fontsize=9)
    axA.annotate("real > zero\n→ USES the signal\n(reciprocator)", (1, 1312),
                 textcoords="offset points", xytext=(-2, -64), fontsize=8.5, color="#1b7837", ha="center")
    axA.annotate("zero > real\n→ anti-uses", (0, 914), textcoords="offset points",
                 xytext=(0, 6), fontsize=8.5, color="#c5283d", ha="center")
    axA.set_xticks(x); axA.set_xticklabels(labels)
    axA.set_ylabel("homogeneous return/agent")
    axA.set_title("A. common+α genuinely reciprocates (real α > zero)\n— a real reciprocator exists",
                  fontsize=11, loc="left")
    axA.legend(loc="upper left", fontsize=9, frameon=True)
    axA.grid(True, axis="y", alpha=0.3)

    # --- Panel B: exploiting the reciprocity population ---
    axB.plot(SW_CLAIM, SW_RET, "o-", color="#1f4ea1", lw=2.4, label="focal return")
    axB.axvspan(0.05, 1.0, color="#c5283d", alpha=0.08)
    axB.annotate("honest / min\n(commons intact)", (0.0, 964), textcoords="offset points",
                 xytext=(24, -6), fontsize=9, color="#1b7837")
    axB.text(0.55, 500, "any over-report →\ncommons COLLAPSE", ha="center", color="#c5283d", fontsize=9.5)
    axB.set_xlabel("focal's self-reported claim")
    axB.set_ylabel("focal return", color="#1f4ea1")
    axB.set_title("B. A member lying collapses the reciprocity population\n(return 964→90, a cliff)",
                  fontsize=11, loc="left")
    axB.grid(True, alpha=0.3)
    axd = axB.twinx()
    axd.plot(SW_CLAIM, SW_DIRT, "s--", color="#c5283d", lw=1.6, alpha=0.8)
    axd.set_ylabel("river dirt tiles", color="#c5283d")

    fig.suptitle("Stage-4 reciprocator: lying is self-defeating even against a signal-using "
                 "population — verification only matters vs a *malicious* saboteur",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
