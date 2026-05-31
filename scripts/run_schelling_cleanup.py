#!/usr/bin/env python3
"""Schelling-diagram co-play rollouts for SocialJax Cleanup (clean_up).

Cleanup analogue of `run_schelling_rollouts.py` (which targets Territory). Two
key differences:

  1. Env = `clean_up` (7 agents, 9 actions, 1000 inner steps). No upstream-bug
     monkeypatch needed — `Clean_up.__init__` sets inequity_aversion/svo
     correctly.
  2. Supports **PS=False position-matched** policies: each role is a set of 7
     per-agent `.pkl`s (one per spawn slot), and agent slot i runs the policy
     trained for slot i. So a "cooperator at slot i" uses common[i]; a
     "defector at slot i" uses individual[i]. This respects the
     position-specialisation of PS=False training (common[i] != common[j]).

Cooperator = common-reward IPPO (learns to clean); defector = individual-reward
IPPO (free-rides). For each x in 0..N-1 (number of OTHER cooperators):

  - rollout A (focal=cooperator): x+1 coop slots [0..x] + (N-1-x) def slots
      -> coop_payoff[x] = mean individual return over the x+1 coop agents
  - rollout B (focal=defector):   x coop slots [0..x-1] + (N-x) def slots
      -> def_payoff[x] = mean individual return over the N-x def agents
      -> avg_payoff[x] = mean over all agents

**Eval reward accounting**: we instantiate the env with `shared_rewards=False`
so each agent's reward is its OWN apples (the env's `else` branch yields
own-apples * num_agents). This is what makes the cooperator-vs-defector payoff
gap visible — under shared rewards every agent gets the team sum and the gap
vanishes. The policies' *behaviour* is what we evaluate; the reward accounting
is only the measurement.

Output: a CSV consumable by `scripts/plot_schelling_diagram.py`.

Usage:
    uv run python scripts/run_schelling_cleanup.py \
        --coop-dir cleanup/schelling/policies/common_seed0 \
        --def-dir  cleanup/schelling/policies/individual_seed0 \
        --episodes 32 \
        --out      cleanup/schelling/schelling_cleanup_seed0.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import pickle
import sys
import time
from functools import partial
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

# Make SocialJax importable when run from repo root.
_SOCIALJAX_DIR = Path(__file__).resolve().parents[1] / "external" / "SocialJax"
if str(_SOCIALJAX_DIR) not in sys.path:
    sys.path.insert(0, str(_SOCIALJAX_DIR))

import socialjax  # noqa: E402


# --- Network — copied verbatim from algorithms/IPPO/ippo_cnn_cleanup.py so the
#     loaded params apply. (CNN 32x(5,5)+32x(3,3)+32x(3,3)+Dense64; actor/critic.)
class CNN(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(32, (5, 5), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = nn.Conv(32, (3, 3), kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = act(x)
        return x


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        act = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNN(self.activation)(x)
        a = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        a = act(a)
        a = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        pi = distrax.Categorical(logits=a)
        c = nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(embedding)
        c = act(c)
        c = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(c)
        return pi, jnp.squeeze(c, axis=-1)


def load_agent_params(role_dir: str, num_agents: int):
    """Load num_agents per-agent pkls (clean_up_seed*_<i>.pkl) as a list."""
    out = []
    for i in range(num_agents):
        matches = glob.glob(os.path.join(role_dir, f"*_{i}.pkl"))
        if not matches:
            raise FileNotFoundError(f"no per-agent pkl for slot {i} in {role_dir}")
        with open(sorted(matches)[0], "rb") as f:
            p = pickle.load(f)
        out.append(jax.tree_util.tree_map(jnp.asarray, p))
    return out


def make_env(num_agents: int, num_steps: int):
    """Cleanup with individual-reward accounting for per-agent eval payoffs."""
    return socialjax.make(
        "clean_up",
        num_agents=num_agents,
        num_inner_steps=num_steps,
        num_outer_steps=1,
        shared_rewards=False,   # -> each agent's reward = own apples * num_agents
        inequity_aversion=False,
        svo=False,
        cnn=True,
        jit=True,
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--coop-dir", required=True, help="dir of cooperator per-agent pkls (common)")
    ap.add_argument("--def-dir", required=True, help="dir of defector per-agent pkls (individual)")
    ap.add_argument("--episodes", type=int, default=32, help="episodes per composition")
    ap.add_argument("--num-steps", type=int, default=1000, help="steps per episode")
    ap.add_argument("--num-agents", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    N = args.num_agents
    env = make_env(N, args.num_steps)
    network = ActorCritic(env.action_space().n)
    agents = list(env.agents)

    coop = load_agent_params(args.coop_dir, N)
    deff = load_agent_params(args.def_dir, N)
    # Stack per-slot param trees -> leaves get a leading (N,) axis.
    coop_stk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *coop)
    def_stk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *deff)

    def select_params(roles):
        """roles: (N,) int {1=coop,0=def} -> per-slot batched params."""
        def pick(c, d):
            r = roles.reshape((N,) + (1,) * (c.ndim - 1))
            return jnp.where(r == 1, c, d)
        return jax.tree_util.tree_map(pick, coop_stk, def_stk)

    def act(batched_params, obs_batch, key):
        """obs_batch: (N,H,W,C) -> sampled actions (N,)."""
        def one(p, o):
            pi, _ = network.apply(p, o[None])
            return pi.logits[0]
        logits = jax.vmap(one)(batched_params, obs_batch)  # (N, A)
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    @partial(jax.jit, static_argnums=())
    def run_episode(batched_params, rng):
        rng, kr = jax.random.split(rng)
        obs, state = env.reset(kr)

        def body(carry, _):
            state, obs, rng, ret = carry
            obs_batch = jnp.stack([obs[a] for a in agents])  # (N,H,W,C)
            rng, ka, ks = jax.random.split(rng, 3)
            actions = act(batched_params, obs_batch, ka)
            obs2, state2, rewards, done, info = env.step_env(ks, state, actions)
            ret = ret + rewards.astype(jnp.float32)
            return (state2, obs2, rng, ret), None

        init = (state, obs, rng, jnp.zeros(N, dtype=jnp.float32))
        (state, obs, rng, ret), _ = jax.lax.scan(body, init, None, length=args.num_steps)
        return ret  # (N,) per-agent episode return

    run_episodes = jax.jit(jax.vmap(run_episode, in_axes=(None, 0)))

    def stats(xs):
        m = float(np.mean(xs))
        se = float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
        return m, se

    started = time.time()
    print(
        f"Cleanup Schelling rollouts: N={N} episodes={args.episodes} "
        f"steps={args.num_steps}\n  coop={args.coop_dir}\n  def ={args.def_dir}"
    )
    rows = []
    base = jax.random.PRNGKey(args.seed)
    for x in range(N):  # number of OTHER cooperators
        rolesA = jnp.array([1] * (x + 1) + [0] * (N - 1 - x), dtype=jnp.int32)  # focal=coop
        rolesB = jnp.array([1] * x + [0] * (N - x), dtype=jnp.int32)            # focal=def
        kA, kB = jax.random.split(jax.random.fold_in(base, x))
        keysA = jax.random.split(kA, args.episodes)
        keysB = jax.random.split(kB, args.episodes)

        retsA = np.asarray(run_episodes(select_params(rolesA), keysA))  # (E,N)
        retsB = np.asarray(run_episodes(select_params(rolesB), keysB))

        coop_per_ep = retsA[:, : x + 1].mean(axis=1)   # x+1 coop agents
        def_per_ep = retsB[:, x:].mean(axis=1)         # N-x def agents
        avg_per_ep = retsB.mean(axis=1)

        coop_m, coop_se = stats(coop_per_ep)
        def_m, def_se = stats(def_per_ep)
        avg_m, avg_se = stats(avg_per_ep)
        print(
            f"  x={x}: coop={coop_m:7.2f}±{coop_se:5.2f} | "
            f"def={def_m:7.2f}±{def_se:5.2f} | avg={avg_m:7.2f}±{avg_se:5.2f}"
        )
        rows.append({
            "n_other_coops": x,
            "coop_payoff_mean": coop_m, "coop_payoff_stderr": coop_se,
            "def_payoff_mean": def_m, "def_payoff_stderr": def_se,
            "avg_payoff_mean": avg_m, "avg_payoff_stderr": avg_se,
            "n_episodes": args.episodes,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out}  ({(time.time()-started)/60:.1f} min)")


if __name__ == "__main__":
    main()
