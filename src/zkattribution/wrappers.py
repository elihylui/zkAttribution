"""Stage 3: AttributionWrapper for oracle-attribution augmented observations.

Wraps a SocialJax `MultiAgentEnv` (or any env with the same interface) so each
agent's observation is augmented with a per-peer cooperation-rate vector
`alpha`. `alpha` is computed from global state via a Stage-2 batched predicate
and updated at window boundaries (every `window_size` steps).

The augmentation: `alpha` (shape `(num_agents,)`) is broadcast as additional
constant channels onto each agent's local obs along the trailing channel axis.
An obs of shape `(..., C)` becomes `(..., C + num_agents)`. The CNN policy
adapts automatically — its first conv layer's `in_channels` is inferred at
init time.

`alpha` is constant within a window — agents see the *previous* window's
cooperation rates, not a rolling estimate. For the first window (steps
`0..N-1`), `alpha = zeros`.

The wrapper follows SocialJax's `JaxMARLWrapper` pattern: it delegates unknown
attribute access to the inner env via `__getattr__`, overrides `reset` and
`step`, and overrides `observation_space` to report the augmented shape.

See `docs/stage3_design.md` for design rationale.
"""

from functools import partial
from types import SimpleNamespace
from typing import Callable

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class AttributionState:
    """State for AttributionWrapper.

    Attributes:
        env_state: Inner SocialJax env state (delegated).
        event_buffer: shape (window_size, num_agents) int8 — per-step e_k for
            the current window. Overwritten as a new window fills.
        step_in_window: scalar int32 in [0, window_size) — position in
            event_buffer where the next event will be written. Wraps to 0 at
            window boundaries.
        current_alpha: shape (num_agents,) float32 — broadcast onto each
            agent's obs at every step until the next window boundary.
    """

    env_state: object
    event_buffer: jnp.ndarray
    step_in_window: jnp.ndarray
    current_alpha: jnp.ndarray


class AttributionWrapper:
    """Augment a SocialJax env's observations with per-peer cooperation rates."""

    def __init__(
        self,
        env,
        predicate: Callable,
        window_size: int = 100,
        predicate_uses_actions: bool = True,
    ):
        self._env = env
        self._predicate = predicate
        self._window_size = window_size
        self._predicate_uses_actions = predicate_uses_actions

    def __getattr__(self, name: str):
        # Delegate unknown attrs to the inner env (JaxMARLWrapper pattern).
        # NB: __getattr__ is only called when normal attribute lookup fails,
        # so it doesn't interfere with _env / _predicate / etc.
        return getattr(self._env, name)

    def observation_space(self, *args, **kwargs):
        """Augmented observation space — channel axis grows by num_agents.

        SocialJax envs return `(spaces.Box, shape_tuple)`. We construct an
        equivalent pair with the augmented trailing dim. Only `.shape` is
        consumed by MAPPO; we expose `.dtype` and the original `low`/`high`
        for completeness.
        """
        inner = self._env.observation_space(*args, **kwargs)
        if isinstance(inner, tuple) and len(inner) == 2:
            inner_space, inner_shape = inner
        else:
            inner_space = inner
            inner_shape = inner.shape
        n = self._env.num_agents
        new_shape = tuple(inner_shape[:-1]) + (int(inner_shape[-1]) + n,)
        augmented_space = SimpleNamespace(
            shape=new_shape,
            dtype=getattr(inner_space, "dtype", jnp.float32),
            low=getattr(inner_space, "low", None),
            high=getattr(inner_space, "high", None),
        )
        return augmented_space, new_shape

    def _augment_obs(self, obs, alpha: jnp.ndarray):
        """Concat alpha as constant channels onto each agent's obs (channels-last).

        Works whether obs is a single ndarray (SocialJax convention) or a
        dict-per-agent (other MARL envs).
        """

        def _aug_one(o):
            leading = o.shape[:-1]
            alpha_chan = jnp.broadcast_to(alpha.astype(o.dtype), leading + (alpha.shape[0],))
            return jnp.concatenate([o, alpha_chan], axis=-1)

        return jax.tree_util.tree_map(_aug_one, obs)

    def _compute_events(self, state_before, state_after, actions_arr: jnp.ndarray) -> jnp.ndarray:
        if self._predicate_uses_actions:
            return self._predicate(state_before, state_after, actions_arr).astype(jnp.int8)
        return self._predicate(state_before, state_after).astype(jnp.int8)

    def _coerce_actions(self, action) -> jnp.ndarray:
        """Convert MAPPO's action format (dict / list / array) to (num_agents,) array.

        MAPPO post-unbatchify hands actions as a list of `(num_envs, 1)` tensors,
        which inside `jax.vmap` reduces to a list of `(1,)` scalars. Stacking
        gives `(num_agents, 1)`; we reshape to `(num_agents,)` so the predicate
        sees the canonical shape.
        """
        if isinstance(action, dict):
            arr = jnp.stack([action[a] for a in self._env.agents])
        elif isinstance(action, (list, tuple)):
            arr = jnp.stack(list(action))
        else:
            arr = jnp.asarray(action)
        return arr.reshape(self._env.num_agents)

    @partial(jax.jit, static_argnums=0)
    def reset(self, key):
        obs, env_state = self._env.reset(key)
        state = AttributionState(
            env_state=env_state,
            event_buffer=jnp.zeros(
                (self._window_size, self._env.num_agents), dtype=jnp.int8
            ),
            step_in_window=jnp.int32(0),
            current_alpha=jnp.zeros(self._env.num_agents, dtype=jnp.float32),
        )
        return self._augment_obs(obs, state.current_alpha), state

    @partial(jax.jit, static_argnums=0)
    def step(self, key, state: AttributionState, action):
        env_state_before = state.env_state
        # Calls inner env's step, which itself does auto-reset on done.
        obs, env_state_after, reward, done, info = self._env.step(
            key, env_state_before, action
        )

        # Predicate operates on raw (pre-reset) state pair — but inner `step`
        # may have auto-reset env_state_after when done. We mask the events to
        # zero on done to avoid spurious cooperation credit at episode bounds.
        actions_arr = self._coerce_actions(action)
        events = self._compute_events(env_state_before, env_state_after, actions_arr)
        ep_done = self._extract_done_scalar(done)
        events = jnp.where(ep_done, jnp.zeros_like(events), events)

        # Write events, advance step counter, recompute alpha at window boundary.
        buffer_next = state.event_buffer.at[state.step_in_window].set(events)
        step_next = state.step_in_window + 1
        at_boundary = step_next >= self._window_size
        new_alpha = buffer_next.mean(axis=0, dtype=jnp.float32)
        alpha_next = jnp.where(at_boundary, new_alpha, state.current_alpha)
        step_in_window_next = jnp.where(
            at_boundary, jnp.int32(0), step_next
        ).astype(jnp.int32)

        state_next = AttributionState(
            env_state=env_state_after,
            event_buffer=buffer_next,
            step_in_window=step_in_window_next,
            current_alpha=alpha_next,
        )

        # Surface alpha into `info` so training loops log the attribution signal.
        # `alpha` (per-agent) -> logged as mean alpha; `alpha_std` -> cross-agent
        # spread (is the signal varied, or flat/useless?).
        info = dict(info)
        info["alpha"] = alpha_next
        info["alpha_std"] = jnp.full(
            (self._env.num_agents,), jnp.std(alpha_next), dtype=jnp.float32
        )

        return self._augment_obs(obs, alpha_next), state_next, reward, done, info

    @staticmethod
    def _extract_done_scalar(done) -> jnp.ndarray:
        """SocialJax `done` is dict-like with '__all__'; older returns a scalar."""
        if isinstance(done, dict):
            return jnp.asarray(done.get("__all__", False))
        return jnp.asarray(done)

    def render(self, state):
        """Delegate to inner env's render, unwrapping AttributionState if needed."""
        if isinstance(state, AttributionState):
            return self._env.render(state.env_state)
        return self._env.render(state)
