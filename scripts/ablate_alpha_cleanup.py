#!/usr/bin/env python3
"""α-usage ablation for the trained w=50 seed0 attribution policy.

Run the homogeneous all-attr population (7 PS=False policies) in the
AttributionWrapper env and measure mean per-agent episode return under three
inference views of the α channels (env dynamics identical in all three — α
never affects transitions, only what the policy sees):

    real        : the wrapper's true α (should reproduce the ~553 training plateau)
    zero        : α channels set to 0
    rand U[0,1] : large noise — OOD *magnitude* (original, confounded control)
    rand U[0,.1]: small noise, magnitude-matched to real α (~0.005-0.05)
    perm/step   : real α permuted across peers EVERY step (value distn exact, but
                  adds temporal jitter vs the window-stable training signal)
    perm/window : real α permuted across peers ONCE per 50-step window — cadence-
                  matched to real α, so isolates the cross-agent *information* effect

If return stays ~553 under zero/random, the policy ignores α and the training
lift came from something else (e.g. the extra channels acting as regularizer).

Usage:
    uv run python scripts/ablate_alpha_cleanup.py --attr-dir cleanup/schelling/policies/attr_seed0 --episodes 24
"""
from __future__ import annotations
import argparse, glob, math, os, pickle, sys, warnings
from functools import partial
from pathlib import Path

import distrax, flax.linen as nn, jax, jax.numpy as jnp, numpy as np
from flax.linen.initializers import constant, orthogonal

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "external" / "SocialJax", _REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import socialjax  # noqa: E402
from zkattribution.predicate import cleanup_events_batch  # noqa: E402
from zkattribution.wrappers import AttributionWrapper  # noqa: E402
warnings.filterwarnings("ignore", message=".*int32.*int16.*")


class CNN(nn.Module):
    activation: str = "relu"
    @nn.compact
    def __call__(self, x):
        act = nn.relu
        for k in ((5, 5), (3, 3), (3, 3)):
            x = act(nn.Conv(32, k, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))
        x = x.reshape((x.shape[0], -1))
        return act(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x))


class ActorCritic(nn.Module):
    action_dim: int
    @nn.compact
    def __call__(self, x):
        emb = CNN()(x)
        a = nn.relu(nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(emb))
        a = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(a)
        return distrax.Categorical(logits=a), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attr-dir", default="cleanup/schelling/policies/attr_seed0")
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--num-steps", type=int, default=1000)
    ap.add_argument("--num-agents", type=int, default=7)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    N, NS = args.num_agents, args.num_steps

    base = socialjax.make("clean_up", num_agents=N, num_inner_steps=NS, num_outer_steps=1,
                          shared_rewards=False, inequity_aversion=False, svo=False, cnn=True, jit=True)
    wenv = AttributionWrapper(base, predicate=cleanup_events_batch, window_size=50)
    net = ActorCritic(base.action_space().n)
    agents = list(base.agents)

    pols = []
    for i in range(N):
        with open(sorted(glob.glob(os.path.join(args.attr_dir, f"*_{i}.pkl")))[0], "rb") as f:
            pols.append(jax.tree_util.tree_map(jnp.asarray, pickle.load(f)))
    stk = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *pols)

    def act(obs_batch, key):
        logits = jax.vmap(lambda p, o: net.apply(p, o[None])[0].logits[0])(stk, obs_batch)
        return jax.random.categorical(key, logits, axis=-1).astype(jnp.int32)

    def make_run(mode):  # 0 real, 1 zero, 2 U[0,1], 3 U[0,.1], 4 perm/step, 5 perm/window
        @jax.jit
        def run(rng):
            rng, kr, kperm = jax.random.split(rng, 3)
            obs, st = wenv.reset(kr)
            def body(c, t):
                st, obs, rng, ret, clean, dirt = c
                ob = jnp.stack([obs[a] for a in agents])  # (N,11,11,26)
                rng, kr2, ka, ks = jax.random.split(rng, 4)
                if mode == 1:
                    ob = ob.at[..., 19:].set(0.0)
                elif mode == 2:
                    r = jax.random.uniform(kr2, (N,))           # one α vector, broadcast like the wrapper
                    ob = ob.at[..., 19:].set(jnp.broadcast_to(r, ob[..., 19:].shape))
                elif mode == 3:                                  # magnitude-matched noise
                    r = jax.random.uniform(kr2, (N,)) * 0.1
                    ob = ob.at[..., 19:].set(jnp.broadcast_to(r, ob[..., 19:].shape))
                elif mode == 4:                                  # permute real α across peers, EVERY step
                    av = ob[0, 0, 0, 19:]                        # the broadcast α vector (same for all agents)
                    avp = av[jax.random.permutation(kr2, N)]
                    ob = ob.at[..., 19:].set(jnp.broadcast_to(avp, ob[..., 19:].shape))
                elif mode == 5:                                  # permute real α ONCE per 50-step window
                    av = ob[0, 0, 0, 19:]                        # 50 = wrapper window_size (cadence-matched)
                    perm = jax.random.permutation(jax.random.fold_in(kperm, t // 50), N)
                    ob = ob.at[..., 19:].set(jnp.broadcast_to(av[perm], ob[..., 19:].shape))
                a = act(ob, ka)
                obs2, st2, rew, done, info = wenv.step(ks, st, a)
                clean = clean + (a == 8).astype(jnp.float32)   # CLEAN_ACTION=8: chose to clean?
                dirt = dirt + jnp.sum(st2.env_state.potential_dirt_and_dirt_label == 8).astype(jnp.float32)
                return (st2, obs2, rng, ret + rew.astype(jnp.float32), clean, dirt), None
            init = (st, obs, rng, jnp.zeros(N), jnp.zeros(N), jnp.float32(0.0))
            (st, obs, rng, ret, clean, dirt), _ = jax.lax.scan(body, init, jnp.arange(NS))
            return ret, clean / NS, dirt / NS  # return(N,), clean-rate(N,), mean dirt tiles
        return jax.jit(jax.vmap(run))

    bkeys = jax.random.split(jax.random.PRNGKey(args.seed), args.episodes)
    print(f"all-attr ablation: N={N} episodes={args.episodes} steps={NS}  attr={args.attr_dir}")
    for mode, name in [(0, "real       "), (1, "zero       "), (2, "rand U[0,1] "),
                       (3, "rand U[0,.1]"), (4, "perm/step  "), (5, "perm/window")]:
        rets, clean, dirt = make_run(mode)(bkeys)            # (E,N),(E,N),(E,)
        rets, clean, dirt = map(np.asarray, (rets, clean, dirt))
        per_ep = rets.mean(axis=1)
        m = per_ep.mean(); se = per_ep.std(ddof=1) / math.sqrt(len(per_ep))
        print(f"  α={name}: return/agent = {m:7.2f} ± {se:5.2f}  | "
              f"clean-action rate = {clean.mean():.3f}  | mean dirt tiles = {dirt.mean():5.1f}")


if __name__ == "__main__":
    main()
