#!/usr/bin/env python3
"""Uniform-random-policy rollout on a SocialJax env — an experimental control.

Measures `cleaned_water` and collective return under uniform-random actions.
Used to interpret a trained policy: if the trained policy keeps the river
*dirtier* than random, it has learned anti-cooperative behaviour (defection);
if it matches random, training has not produced a meaningful policy.

Usage:
    uv run python scripts/random_rollout.py [--env clean_up] [--episodes 3] [--steps 1000]
"""

import argparse
import os
import statistics
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), os.pardir, "external", "SocialJax")
)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import socialjax  # noqa: E402


def rollout(env, num_agents: int, n_steps: int, seed: int):
    """One episode of uniform-random actions. Returns (mean_cw, first_cw, last_cw, return)."""
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    _obs, state = env.reset(reset_key)
    n_actions = env.action_space(0).n

    cleaned_water = []
    collective_return = 0.0
    for _ in range(n_steps):
        rng, *agent_keys = jax.random.split(rng, num_agents + 1)
        actions = [
            jax.random.randint(agent_keys[i], (), 0, n_actions) for i in range(num_agents)
        ]
        rng, step_key = jax.random.split(rng)
        _obs, state, reward, _done, info = env.step_env(step_key, state, actions)
        cleaned_water.append(float(jnp.ravel(info["cleaned_water"])[0]))
        collective_return += float(jnp.sum(reward))

    return (
        statistics.mean(cleaned_water),
        cleaned_water[0],
        cleaned_water[-1],
        collective_return,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--env", default="clean_up")
    ap.add_argument("--num-agents", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    env = socialjax.make(
        args.env, num_agents=args.num_agents, num_inner_steps=args.steps
    )
    print(
        f"Uniform-random policy on '{args.env}': "
        f"{args.episodes} episodes x {args.steps} steps, {args.num_agents} agents\n"
    )
    means, returns = [], []
    for ep in range(args.episodes):
        mean_cw, first_cw, last_cw, ret = rollout(
            env, args.num_agents, args.steps, seed=ep
        )
        means.append(mean_cw)
        returns.append(ret)
        print(
            f"  ep {ep}: cleaned_water mean={mean_cw:7.2f}  "
            f"(first={first_cw:.0f}, last={last_cw:.0f})  collective_return={ret:.2f}"
        )

    print(
        f"\n  cleaned_water  — across episodes: mean={statistics.mean(means):.2f}"
        f"  (min={min(means):.2f}, max={max(means):.2f})"
    )
    print(
        f"  collective_return — across episodes: mean={statistics.mean(returns):.2f}"
    )


if __name__ == "__main__":
    main()
