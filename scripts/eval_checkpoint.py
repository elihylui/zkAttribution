#!/usr/bin/env python3
"""Roll out a trained MAPPO checkpoint and measure cooperation metrics.

Loads a saved Actor checkpoint, rolls it out on the bare (unwrapped) SocialJax
Cleanup env for full episodes, and reports `cleaned_water` + collective return.

This is the decisive test for interpreting a near-zero-return training run:
compare the trained policy's cleaned_water against the uniform-random control
(see scripts/random_rollout.py).
  - trained clearly  < random  -> learned anti-cooperative behaviour (defection)
  - trained         ~= random  -> undertrained, behaving ~randomly

Usage:
    uv run python scripts/eval_checkpoint.py checkpoints/clean_up_seed30_reward_MAPPO.pkl
"""

import argparse
import importlib.util
import os
import pickle
import statistics
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, os.path.join(_REPO, "external", "SocialJax"))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import socialjax  # noqa: E402


def _load_actor_class():
    """Import the exact Actor class that produced the baseline checkpoint."""
    path = os.path.join(
        _REPO, "external/SocialJax/algorithms/MAPPO/mappo_cnn_cleanup.py"
    )
    spec = importlib.util.spec_from_file_location("_sj_mappo_cleanup", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Actor


def rollout(env, apply_fn, params, num_agents, n_steps, seed):
    """One episode under the trained policy (sampled actions). Returns metrics."""
    rng = jax.random.PRNGKey(seed)
    rng, reset_key = jax.random.split(rng)
    obs, state = env.reset(reset_key)
    obs_shape = env.observation_space()[0].shape

    cleaned_water = []
    collective_return = 0.0
    for _ in range(n_steps):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(-1, *obs_shape)
        pi = apply_fn(params, obs_batch)
        rng, action_key = jax.random.split(rng)
        actions = pi.sample(seed=action_key)
        rng, step_key = jax.random.split(rng)
        obs, state, reward, _done, info = env.step_env(
            step_key, state, [actions[i] for i in range(num_agents)]
        )
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
    ap.add_argument("checkpoint", help="path to a saved Actor .pkl checkpoint")
    ap.add_argument("--num-agents", type=int, default=7)
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1000)
    args = ap.parse_args()

    with open(args.checkpoint, "rb") as f:
        params = pickle.load(f)
    params = jax.tree_util.tree_map(lambda x: jnp.array(x), params)

    actor_cls = _load_actor_class()
    env = socialjax.make(
        "clean_up", num_agents=args.num_agents, num_inner_steps=args.steps
    )
    actor = actor_cls(action_dim=env.action_space().n, activation="relu")
    apply_fn = jax.jit(actor.apply)

    print(f"Trained checkpoint: {args.checkpoint}")
    print(f"  {args.episodes} episodes x {args.steps} steps, {args.num_agents} agents\n")
    means, returns = [], []
    for ep in range(args.episodes):
        mean_cw, first_cw, last_cw, ret = rollout(
            env, apply_fn, params, args.num_agents, args.steps, seed=ep
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
    print("\n  Compare against scripts/random_rollout.py (uniform-random control).")


if __name__ == "__main__":
    main()
