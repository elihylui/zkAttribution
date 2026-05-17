#!/usr/bin/env python3
"""Confound measurement: v1 (global dirt delta) vs v2 (per-agent beam-hit).

Rolls out Cleanup with random actions and tallies, per agent-step:
  - zap_clean fired
  - v1 credit  (zap AND global dirt count strictly decreased)
  - v2 credit  (zap AND the agent's beam covered a dirt tile)

If v2 credits substantially more than v1, the v1 global-delta predicate was
systematically under-crediting genuine cleans — the confound that drove the
logged cooperation rate alpha to ~0 in the 2e5 diagnostic run.

Usage:
    uv run python scripts/confound_measurement.py [--steps 1000] [--seed 0]
"""

import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_REPO, "src"))
sys.path.insert(0, os.path.join(_REPO, "external", "SocialJax"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import socialjax  # noqa: E402

from zkattribution.predicate import (  # noqa: E402
    CLEAN_ACTION,
    cleanup_events_batch,
    cleanup_events_batch_global_delta,
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--num-agents", type=int, default=7)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    env = socialjax.make(
        "clean_up", num_agents=args.num_agents, num_inner_steps=args.steps
    )
    n_actions = env.action_space(0).n
    rng = jax.random.PRNGKey(args.seed)
    rng, reset_key = jax.random.split(rng)
    _obs, state = env.reset(reset_key)

    n_zap = n_v1 = n_v2 = n_both = 0
    prev = state
    for _ in range(args.steps):
        rng, *agent_keys = jax.random.split(rng, args.num_agents + 1)
        actions = [
            jax.random.randint(agent_keys[i], (), 0, n_actions)
            for i in range(args.num_agents)
        ]
        rng, step_key = jax.random.split(rng)
        _obs, state, _r, _d, _info = env.step_env(step_key, prev, actions)

        actions_arr = jnp.array(actions)
        v1 = cleanup_events_batch_global_delta(prev, state, actions_arr)
        v2 = cleanup_events_batch(prev, state, actions_arr)

        n_zap += int(jnp.sum(actions_arr == CLEAN_ACTION))
        n_v1 += int(jnp.sum(v1))
        n_v2 += int(jnp.sum(v2))
        n_both += int(jnp.sum((v1 == 1) & (v2 == 1)))
        prev = state

    print(
        f"Confound measurement — Cleanup, random policy, "
        f"{args.steps} steps x {args.num_agents} agents\n"
    )
    print(f"  zap_clean actions fired         : {n_zap}")
    print(
        f"  v1 (global dirt delta) credits  : {n_v1:5d}"
        f"   ({100 * n_v1 / max(n_zap, 1):.1f}% of zaps)"
    )
    print(
        f"  v2 (per-agent beam-hit) credits : {n_v2:5d}"
        f"   ({100 * n_v2 / max(n_zap, 1):.1f}% of zaps)"
    )
    if n_v2 > 0:
        recall = 100 * n_both / n_v2
        print()
        print(
            f"  Of {n_v2} genuine cleans (v2), v1 also credited {n_both} "
            f"({recall:.1f}%) — v1's recall of real cleans."
        )
        print(
            f"  -> v1 missed {n_v2 - n_both} genuine cleans "
            f"({100 - recall:.1f}%): the confound."
        )


if __name__ == "__main__":
    main()
