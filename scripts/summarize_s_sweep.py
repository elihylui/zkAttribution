#!/usr/bin/env python3
"""Summarise an s-sweep results directory into a cooperation table.

Walks a directory of run subdirs named `{regime}_s{stag}_t{tt}_seed{seed}`,
reads each run's final `returned_episode_returns` from its wandb-offline
datastore, and tabulates, per (regime, S), the per-seed final returns and how
many seeds reached cooperation.

The s-sweep outcome is sharply bimodal — cooperating runs end ~850-1400,
collapsed runs ~0-5 — so a return threshold anywhere in the gap classifies
each run cleanly.

Usage:
    uv run python scripts/summarize_s_sweep.py <results_dir>
"""

import argparse
import os
import re
from collections import defaultdict

from parse_wandb_run import find_wandb_file, read_history

# A run "cooperated" if its final episode return clears this threshold (chosen
# in the bimodal gap between collapsed ~0-5 and cooperating ~850+).
COOP_RETURN_THRESHOLD = 100.0

_RUN_RE = re.compile(r"^(?P<regime>.+)_s(?P<stag>\d+)_t\d+_seed(?P<seed>\d+)$")


def final_return(run_dir: str):
    """Final `returned_episode_returns` for one run dir, or None."""
    rows = read_history(find_wandb_file(run_dir))
    if not rows:
        return None
    return rows[-1].get("returned_episode_returns")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="dir of {regime}_s{stag}_t{tt}_seed{n} run dirs")
    args = ap.parse_args()

    # (regime, S) -> {seed: final_return}
    cells: dict = defaultdict(dict)
    for name in sorted(os.listdir(args.results_dir)):
        path = os.path.join(args.results_dir, name)
        m = _RUN_RE.match(name)
        if not (os.path.isdir(path) and m):
            continue
        try:
            cells[(m["regime"], int(m["stag"]) / 100.0)][int(m["seed"])] = final_return(path)
        except FileNotFoundError:
            print(f"  (skip {name}: no wandb datastore)")

    regimes = sorted({r for r, _ in cells})
    svals = sorted({s for _, s in cells}, reverse=True)

    print(f"\n{'regime':<16}{'S':>6}   {'final return per seed':<46}{'cooperated':>12}")
    print("-" * 80)
    for regime in regimes:
        for s in svals:
            seeds = cells.get((regime, s))
            if not seeds:
                continue
            rets = [seeds[k] for k in sorted(seeds)]
            n_coop = sum(1 for r in rets if r is not None and r > COOP_RETURN_THRESHOLD)
            rets_str = "[" + " ".join(
                f"{r:>6.0f}" if r is not None else "   n/a" for r in rets
            ) + "]"
            print(f"{regime:<16}{s:>6.2f}   {rets_str:<46}{n_coop:>8} / {len(rets)}")
        print()


if __name__ == "__main__":
    main()
