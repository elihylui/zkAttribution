#!/usr/bin/env python3
"""Attribution Schelling diagram for SocialJax Cleanup (clean_up).

Extends `run_schelling_cleanup.py` with a THIRD curve: the α-augmented
individual-reward policy ("attr"). The scientific question — does
observation-layer attribution (each agent sees a per-peer cooperation-rate
vector α) induce *reciprocity*? I.e. does the attr policy behave like a
defector when few others cooperate (α low) and shift toward cooperator-like
behaviour as more others cooperate (α high) — earning the cooperator/"sucker"
payoff and/or lowering the tipping point?

Three policy roles, all PS=False position-matched (7 per-agent pkls each):
  - cooperator (coop) = common-reward IPPO      (19-channel obs)
  - defector  (def)   = individual-reward IPPO  (19-channel obs)
  - attr              = individual-reward IPPO + AttributionWrapper
                        (26-channel obs: 19 base + 7 α channels)

To co-play 19ch and 26ch policies inside ONE vmapped rollout, we ZERO-PAD the
coop/def Conv_0 kernel from (5,5,19,32) to (5,5,26,32). A conv with zero weights
on the 7 α input-channels ignores them — mathematically identical to feeding
only the 19 base channels. So all three policies become structurally identical
(26ch) and the per-slot vmap/select design carries over unchanged.

The eval env is wrapped with `AttributionWrapper(predicate=cleanup_events_batch,
window_size=50)` so α is computed online from the ACTUAL cleaning events of the
co-players. All agents receive the same lagged α vector; coop/def ignore it
(zero-padded), the attr focal reads it.

**Eval reward accounting**: env instantiated with `shared_rewards=False` so each
agent's reward is its OWN apples (the env's `else` branch yields own-apples *
num_agents). This makes the cooperator-vs-defector payoff gap visible.

Curves (x = number of OTHER cooperators, 0..N-1):
  - coop[x]: rollout A, x+1 cooperators -> mean over the x+1 coop slots = R_c(x)
  - def[x] : rollout B, x cooperators   -> mean over the N-x def slots   = R_d(x)
  - avg[x] : rollout B population mean
  - attr[x]: rollout C, attr focal at the MARGINAL slot x, with x coop
             (slots 0..x-1) + (N-1-x) def (slots x+1..N-1) -> return[x] = R_attr(x)
    The attr focal sees exactly x cooperators among its peers — the same
    background a defector faces in rollout B — so attr[x] vs def[x] is an
    apples-to-apples "attribution vs plain free-rider" comparison.

We also record each role's mean cooperation rate (time-averaged α from the
wrapper's predicate) as a behavioural cross-check: coop slots should clean
(α high), def slots should not (α low), and attr's α-vs-x slope is the direct
reciprocity signal.

Output: a CSV consumable by `scripts/plot_schelling_diagram.py` (which draws the
attr curve when the `attr_payoff_*` columns are present).

Usage:
    uv run python scripts/run_schelling_cleanup_attr.py \
        --coop-dir cleanup/schelling/policies/common_seed0 \
        --def-dir  cleanup/schelling/policies/individual_seed0 \
        --attr-dir cleanup/schelling/policies/attr_seed0 \
        --episodes 32 \
        --out      cleanup/schelling/schelling_cleanup_attr_seed0.csv
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
import warnings
from functools import partial
from pathlib import Path

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

# Make SocialJax + zkattribution importable when run from repo root.
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "external" / "SocialJax", _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import socialjax  # noqa: E402
from zkattribution.predicate import cleanup_events_batch  # noqa: E402
from zkattribution.wrappers import AttributionWrapper  # noqa: E402

# Cleanup env scatter casts int32 actions -> int16 (values 0-8 fit); benign.
warnings.filterwarnings("ignore", message=".*int32.*int16.*")


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


def pad_conv0(p, target_in: int):
    """Zero-pad ActorCritic CNN_0/Conv_0 kernel input-channels up to target_in.

    A conv with zero weights on the appended channels ignores them, so a 19ch
    policy padded to 26ch behaves identically while becoming structurally
    compatible with the 26ch attr policy (so all three can share one vmap).
    """
    k = jnp.asarray(p["params"]["CNN_0"]["Conv_0"]["kernel"])  # (kh, kw, cin, cout)
    cin = k.shape[2]
    if cin >= target_in:
        return p
    pad = jnp.zeros(k.shape[:2] + (target_in - cin,) + k.shape[3:], dtype=k.dtype)
    k2 = jnp.concatenate([k, pad], axis=2)
    # Shallow-copy only the path we mutate; other leaves are shared (read-only).
    new_params = dict(p["params"])
    new_cnn = dict(new_params["CNN_0"])
    new_conv0 = dict(new_cnn["Conv_0"])
    new_conv0["kernel"] = k2
    new_cnn["Conv_0"] = new_conv0
    new_params["CNN_0"] = new_cnn
    return {"params": new_params}


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
    ap.add_argument("--coop-dir", required=True, help="cooperator per-agent pkls (common, 19ch)")
    ap.add_argument("--def-dir", required=True, help="defector per-agent pkls (individual, 19ch)")
    ap.add_argument("--attr-dir", required=True, help="attribution per-agent pkls (individual+α, 26ch)")
    ap.add_argument("--episodes", type=int, default=32, help="episodes per composition")
    ap.add_argument("--num-steps", type=int, default=1000, help="steps per episode")
    ap.add_argument("--num-agents", type=int, default=7)
    ap.add_argument("--window-size", type=int, default=50, help="α window (must match training)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output CSV path")
    args = ap.parse_args()

    N = args.num_agents
    base_env = make_env(N, args.num_steps)
    wenv = AttributionWrapper(
        base_env, predicate=cleanup_events_batch, window_size=args.window_size
    )
    network = ActorCritic(base_env.action_space().n)
    agents = list(base_env.agents)

    # Load; pad coop/def to 26ch so all three role stacks match the attr policy.
    coop = [pad_conv0(p, 26) for p in load_agent_params(args.coop_dir, N)]
    deff = [pad_conv0(p, 26) for p in load_agent_params(args.def_dir, N)]
    attr = load_agent_params(args.attr_dir, N)  # already 26ch
    coop_stk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *coop)
    def_stk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *deff)
    attr_stk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *attr)

    def select_params(roles):
        """roles: (N,) int {2=attr,1=coop,0=def} -> per-slot batched params (26ch)."""
        def pick(a, c, d):
            r = roles.reshape((N,) + (1,) * (c.ndim - 1))
            return jnp.where(r == 2, a, jnp.where(r == 1, c, d))
        return jax.tree_util.tree_map(pick, attr_stk, coop_stk, def_stk)

    def act(batched_params, obs_batch, key):
        """obs_batch: (N,H,W,26) -> sampled actions (N,)."""
        def one(p, o):
            pi, _ = network.apply(p, o[None])
            return pi.logits[0]
        logits = jax.vmap(one)(batched_params, obs_batch)  # (N, A)
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    @jax.jit
    def run_episode(batched_params, rng):
        rng, kr = jax.random.split(rng)
        obs, state = wenv.reset(kr)

        def body(carry, _):
            state, obs, rng, ret, asum = carry
            obs_batch = jnp.stack([obs[a] for a in agents])  # (N,H,W,26)
            rng, ka, ks = jax.random.split(rng, 3)
            actions = act(batched_params, obs_batch, ka)
            obs2, state2, rewards, done, info = wenv.step(ks, state, actions)
            ret = ret + rewards.astype(jnp.float32)
            asum = asum + info["alpha"].astype(jnp.float32)  # (N,) lagged windowed α
            return (state2, obs2, rng, ret, asum), None

        init = (state, obs, rng, jnp.zeros(N), jnp.zeros(N))
        (state, obs, rng, ret, asum), _ = jax.lax.scan(
            body, init, None, length=args.num_steps
        )
        return ret, asum / args.num_steps  # per-agent return, per-agent mean α

    run_episodes = jax.jit(jax.vmap(run_episode, in_axes=(None, 0)))

    def stats(xs):
        m = float(np.mean(xs))
        se = float(np.std(xs, ddof=1) / math.sqrt(len(xs))) if len(xs) > 1 else 0.0
        return m, se

    started = time.time()
    print(
        f"Cleanup ATTRIBUTION Schelling rollouts: N={N} episodes={args.episodes} "
        f"steps={args.num_steps} window={args.window_size}\n"
        f"  coop={args.coop_dir}\n  def ={args.def_dir}\n  attr={args.attr_dir}"
    )
    rows = []
    base = jax.random.PRNGKey(args.seed)
    for x in range(N):  # number of OTHER cooperators
        rolesA = jnp.array([1] * (x + 1) + [0] * (N - 1 - x), dtype=jnp.int32)  # coop group
        rolesB = jnp.array([1] * x + [0] * (N - x), dtype=jnp.int32)            # def group
        rolesC = jnp.array([1] * x + [2] + [0] * (N - 1 - x), dtype=jnp.int32)  # attr focal @ slot x
        kA, kB, kC = jax.random.split(jax.random.fold_in(base, x), 3)
        keysA = jax.random.split(kA, args.episodes)
        keysB = jax.random.split(kB, args.episodes)
        keysC = jax.random.split(kC, args.episodes)

        retsA, alphA = run_episodes(select_params(rolesA), keysA)  # (E,N) each
        retsB, alphB = run_episodes(select_params(rolesB), keysB)
        retsC, alphC = run_episodes(select_params(rolesC), keysC)
        retsA, retsB, retsC = map(np.asarray, (retsA, retsB, retsC))
        alphA, alphB, alphC = map(np.asarray, (alphA, alphB, alphC))

        coop_per_ep = retsA[:, : x + 1].mean(axis=1)   # x+1 coop agents
        def_per_ep = retsB[:, x:].mean(axis=1)         # N-x def agents
        avg_per_ep = retsB.mean(axis=1)
        attr_per_ep = retsC[:, x]                      # attr focal at slot x

        # Behavioural cross-check: mean cleaning rate (α) per role.
        coop_cr = alphA[:, : x + 1].mean() if x + 1 > 0 else float("nan")
        def_cr = alphB[:, x:].mean()
        attr_cr = alphC[:, x].mean()

        coop_m, coop_se = stats(coop_per_ep)
        def_m, def_se = stats(def_per_ep)
        avg_m, avg_se = stats(avg_per_ep)
        attr_m, attr_se = stats(attr_per_ep)
        print(
            f"  x={x}: coop={coop_m:7.2f}±{coop_se:5.2f} | def={def_m:7.2f}±{def_se:5.2f}"
            f" | attr={attr_m:7.2f}±{attr_se:5.2f} | avg={avg_m:7.2f}"
            f"  ||  cleanrate coop={coop_cr:.3f} def={def_cr:.3f} attr={attr_cr:.3f}"
        )
        rows.append({
            "n_other_coops": x,
            "coop_payoff_mean": coop_m, "coop_payoff_stderr": coop_se,
            "def_payoff_mean": def_m, "def_payoff_stderr": def_se,
            "attr_payoff_mean": attr_m, "attr_payoff_stderr": attr_se,
            "avg_payoff_mean": avg_m, "avg_payoff_stderr": avg_se,
            "coop_cleanrate": coop_cr, "def_cleanrate": def_cr, "attr_cleanrate": attr_cr,
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
