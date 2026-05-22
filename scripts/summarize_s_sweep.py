#!/usr/bin/env python3
"""Summarise an s-sweep results directory: per-run trajectory + cooperation.

Walks run subdirs named `{regime}_s{stag}_t{tt}_seed{seed}`, reads each run's
episode-return trajectory from its wandb-offline datastore, and prints, per
run, the return at 50% / 75% / 100% of training plus a verdict.

The mid-run points expose whether a cooperating run has *plateaued* or is
still *climbing* at the horizon — the key check for whether the training
budget is long enough. (The 3e7 confirm was confounded precisely because
every run was still climbing at the cutoff.)

Usage:
    uv run python scripts/summarize_s_sweep.py <results_dir>
"""

import argparse
import os
import re
from collections import defaultdict

from parse_wandb_run import find_wandb_file, read_history

# A run "cooperated" if its final episode return clears this threshold.
COOP_RETURN_THRESHOLD = 100.0
# A cooperating run is "still climbing" (not plateaued) if its final return
# exceeds its 75%-mark return by more than this factor.
CLIMB_FACTOR = 1.15

_RUN_RE = re.compile(r"^(?P<regime>.+)_s(?P<stag>\d+)_t\d+_seed(?P<seed>\d+)$")


def trajectory(run_dir: str):
    """Episode return at ~50%, ~75%, and 100% of training (or None)."""
    rows = read_history(find_wandb_file(run_dir))
    if not rows:
        return None

    def at(frac: float):
        return rows[min(int(frac * len(rows)), len(rows) - 1)].get(
            "returned_episode_returns"
        )

    return at(0.50), at(0.75), at(1.0)


def verdict(traj) -> str:
    """'fail' / 'coop (plateau)' / 'coop (climbing)' from a (50,75,100) triple."""
    if traj is None:
        return "no data"
    _, r75, r100 = (x if x is not None else 0.0 for x in traj)
    if r100 <= COOP_RETURN_THRESHOLD:
        return "fail"
    if r100 > CLIMB_FACTOR * max(r75, 1.0):
        return "coop (climbing)"
    return "coop (plateau)"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="dir of {regime}_s{stag}_t{tt}_seed{n} run dirs")
    args = ap.parse_args()

    cells: dict = defaultdict(dict)  # (regime, S) -> {seed: traj}
    for name in sorted(os.listdir(args.results_dir)):
        path = os.path.join(args.results_dir, name)
        m = _RUN_RE.match(name)
        if not (os.path.isdir(path) and m):
            continue
        try:
            cells[(m["regime"], int(m["stag"]) / 100.0)][int(m["seed"])] = trajectory(path)
        except FileNotFoundError:
            print(f"  (skip {name}: no wandb datastore)")

    regimes = sorted({r for r, _ in cells})
    svals = sorted({s for _, s in cells}, reverse=True)

    print(f"\n{'regime':<16}{'S':>6}{'seed':>6}   {'return 50% / 75% / final':<28}{'verdict':>16}")
    print("-" * 74)
    for regime in regimes:
        for s in svals:
            seeds = cells.get((regime, s))
            if not seeds:
                continue
            n_coop = n_climb = 0
            for seed in sorted(seeds):
                traj = seeds[seed]
                v = verdict(traj)
                n_coop += v.startswith("coop")
                n_climb += "climbing" in v
                t = traj or (None, None, None)
                tstr = " / ".join(f"{x:>6.0f}" if x is not None else "   n/a" for x in t)
                print(f"{regime:<16}{s:>6.2f}{seed:>6}   {tstr:<28}{v:>16}")
            note = f"{n_coop}/{len(seeds)} cooperated"
            if n_climb:
                note += f" ({n_climb} still climbing at the horizon)"
            print(f"{' ':<30}-> {note}\n")


if __name__ == "__main__":
    main()
