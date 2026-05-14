"""Stage 3: AttributionWrapper for oracle-attribution augmented observations.

Wraps a SocialJax `MultiAgentEnv` so each agent's observation is augmented with
a per-peer cooperation-rate vector `alpha`. `alpha` is computed from global state
via a Stage-2 batched predicate, and updated at window boundaries (every
`window_size` steps).

The augmentation: alpha (shape `(num_agents,)`) is broadcast as additional
constant channels onto each agent's local obs (which has trailing channel
dimension). So an obs of shape `(H, W, C)` becomes `(H, W, C + num_agents)`.
The CNN policy adapts automatically — first conv layer's `in_channels` is
inferred at init time.

`alpha` is constant within a window — agents see the *previous* window's
cooperation rates, not a rolling estimate. For the first window (steps
`0..N-1`), `alpha = zeros`.

See `docs/stage3_design.md` for design rationale.
"""

from typing import Callable

import jax
import jax.numpy as jnp
from flax.struct import dataclass


@dataclass
class WrappedState:
    """State for AttributionWrapper.

    Attributes:
        env_state: Inner SocialJax env state (delegated).
        event_buffer: shape (window_size, num_agents) int8 — per-step e_k for the
            current window, written in order. Overwritten when a new window starts.
        step_in_window: scalar int32 — position in event_buffer where the next
            event will be written (range 0..window_size, wraps to 0 on boundary).
        current_alpha: shape (num_agents,) float32 — broadcast onto each agent's
            obs at every step until the next window boundary.
    """

    env_state: object
    event_buffer: jnp.ndarray
    step_in_window: jnp.ndarray
    current_alpha: jnp.ndarray


class AttributionWrapper:
    """Augment a SocialJax env's observation with per-peer cooperation rates.

    Constructor args:
        env: a SocialJax env (e.g. `socialjax.make("clean_up", ...)`).
        predicate: a JAX-native batched predicate function. Signature:
            `(state_before, state_after, actions) -> (num_agents,) int8`.
            Use one of `zkattribution.predicate.{cleanup_events_batch,
            harvest_events_batch}` (the latter ignores the `actions` arg).
        window_size: number of steps between alpha updates. Default 100.
        predicate_uses_actions: if False, `predicate` is called with
            `(state_before, state_after)` instead of including actions. Useful
            for Harvest:Open where harvesting is detected from inventory change
            rather than a specific action id.
    """

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
        # Delegate basic env metadata.
        self.num_agents = env.num_agents
        if hasattr(env, "agents"):
            self.agents = env.agents

    # Pass through env interface for methods MAPPO uses.
    def observation_space(self, agent_id):
        # NOTE: shape grows by `num_agents` along the channel axis. We don't try
        # to construct a proper spaces.Box here — MAPPO's CNN policy infers the
        # input shape at init time from a sample observation.
        return self._env.observation_space(agent_id)

    def action_space(self, agent_id):
        return self._env.action_space(agent_id)

    def _augment_obs(self, obs, alpha: jnp.ndarray):
        """Concat alpha as constant channels onto each agent's obs (channels-last)."""

        def _aug_one(o):
            leading = o.shape[:-1]
            alpha_chan = jnp.broadcast_to(alpha.astype(o.dtype), leading + (alpha.shape[0],))
            return jnp.concatenate([o, alpha_chan], axis=-1)

        return jax.tree_util.tree_map(_aug_one, obs)

    def _compute_events(self, state_before, state_after, actions):
        if self._predicate_uses_actions:
            return self._predicate(state_before, state_after, actions)
        return self._predicate(state_before, state_after)

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        wrapped = WrappedState(
            env_state=env_state,
            event_buffer=jnp.zeros((self._window_size, self.num_agents), dtype=jnp.int8),
            step_in_window=jnp.int32(0),
            current_alpha=jnp.zeros(self.num_agents, dtype=jnp.float32),
        )
        return self._augment_obs(obs, wrapped.current_alpha), wrapped

    def step_env(self, key, wrapped: WrappedState, actions):
        env_state_before = wrapped.env_state
        obs, env_state_after, reward, done, info = self._env.step_env(
            key, env_state_before, actions
        )

        # actions arrives as list/dict; predicate expects a jnp array of shape (num_agents,).
        if isinstance(actions, dict):
            actions_arr = jnp.stack([actions[a] for a in self.agents])
        elif isinstance(actions, (list, tuple)):
            actions_arr = jnp.stack(list(actions))
        else:
            actions_arr = jnp.asarray(actions)

        events = self._compute_events(env_state_before, env_state_after, actions_arr)
        events = events.astype(jnp.int8)

        # Write events at current position; advance step counter.
        buffer_next = wrapped.event_buffer.at[wrapped.step_in_window].set(events)
        step_next = wrapped.step_in_window + 1

        # At window boundary: recompute alpha and reset position to 0.
        at_boundary = step_next >= self._window_size
        new_alpha = buffer_next.mean(axis=0, dtype=jnp.float32)
        alpha_next = jnp.where(at_boundary, new_alpha, wrapped.current_alpha)
        step_in_window_next = jnp.where(at_boundary, jnp.int32(0), step_next).astype(jnp.int32)

        wrapped_next = WrappedState(
            env_state=env_state_after,
            event_buffer=buffer_next,
            step_in_window=step_in_window_next,
            current_alpha=alpha_next,
        )

        return self._augment_obs(obs, alpha_next), wrapped_next, reward, done, info
