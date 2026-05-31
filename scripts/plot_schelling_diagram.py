#!/usr/bin/env python3
"""Plot a Schelling diagram from `run_schelling_rollouts.py` output.

Reads the CSV (cols: n_other_coops, coop_payoff_mean/stderr,
def_payoff_mean/stderr, avg_payoff_mean/stderr, n_episodes) and renders the
three lines in the paper's convention:

    blue solid  : cooperator payoff vs number of other cooperators
    red  solid  : defector payoff
    green dashed: population-average payoff

Usage:
    uv run python scripts/plot_schelling_diagram.py <csv> --out plot.png
"""

import argparse
import csv

import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv_path", help="CSV produced by run_schelling_rollouts.py")
    ap.add_argument("--out", default="schelling_diagram.png", help="output PNG path")
    ap.add_argument("--title", default="Territory — Schelling diagram", help="plot title")
    ap.add_argument(
        "--ymin", type=float, default=None, help="y-axis minimum (auto if unset)"
    )
    ap.add_argument(
        "--ymax", type=float, default=None, help="y-axis maximum (auto if unset)"
    )
    args = ap.parse_args()

    rows = []
    with open(args.csv_path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if v not in ("", "nan") else float("nan"))
                         for k, v in r.items()})
    rows.sort(key=lambda r: r["n_other_coops"])

    xs = np.array([r["n_other_coops"] for r in rows])
    coop = np.array([r["coop_payoff_mean"] for r in rows])
    coop_se = np.array([r["coop_payoff_stderr"] for r in rows])
    deff = np.array([r["def_payoff_mean"] for r in rows])
    def_se = np.array([r["def_payoff_stderr"] for r in rows])
    avg = np.array([r["avg_payoff_mean"] for r in rows])
    avg_se = np.array([r["avg_payoff_stderr"] for r in rows])
    # Optional attribution curve (present only for the 3-curve attr CSV).
    has_attr = "attr_payoff_mean" in rows[0]
    if has_attr:
        attr = np.array([r["attr_payoff_mean"] for r in rows])
        attr_se = np.array([r["attr_payoff_stderr"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=150)

    ax.plot(xs, coop, color="#1f4ea1", lw=2.4, label="cooperator (common-reward)")
    ax.fill_between(xs, coop - coop_se, coop + coop_se, color="#1f4ea1", alpha=0.18)

    ax.plot(xs, deff, color="#c5283d", lw=2.4, label="defector (individual-reward)")
    ax.fill_between(xs, deff - def_se, deff + def_se, color="#c5283d", alpha=0.18)

    if has_attr:
        ax.plot(xs, attr, color="#7b3fa0", lw=2.4, label="attribution (individual + α)")
        ax.fill_between(xs, attr - attr_se, attr + attr_se, color="#7b3fa0", alpha=0.18)

    ax.plot(xs, avg, color="#2a8c4a", lw=2.0, linestyle="--", label="average")
    ax.fill_between(xs, avg - avg_se, avg + avg_se, color="#2a8c4a", alpha=0.12)

    ax.set_xlabel("number of other cooperators")
    ax.set_ylabel("individual payoff")
    ax.set_title(args.title)
    ax.set_xlim(xs.min(), xs.max())
    if args.ymin is not None or args.ymax is not None:
        ax.set_ylim(args.ymin, args.ymax)
    ax.set_xticks(np.arange(int(xs.min()), int(xs.max()) + 1))
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", frameon=True)

    fig.tight_layout()
    fig.savefig(args.out)
    print(f"wrote plot -> {args.out}")


if __name__ == "__main__":
    main()
