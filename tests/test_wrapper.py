"""Unit tests for AttributionWrapper.

Uses minimal mock env + fake predicates to test wrapper logic in isolation
from SocialJax. End-to-end tests against the real env live in
tests/test_wrapper_e2e.py (Stage 3c).
"""

import jax
import jax.numpy as jnp
import pytest
from flax.struct import dataclass as flax_dataclass

from zkattribution.wrappers import AttributionWrapper, AttributionState


@flax_dataclass
class FakeInnerState:
    """Minimal flax-struct state for the mock env so it's PyTree-registered."""

    counter: jnp.ndarray


class FakeEnv:
    """Minimal SocialJax-shaped env for wrapper unit tests."""

    def __init__(self, num_agents: int = 3, obs_shape=(2, 2, 4)):
        self.num_agents = num_agents
        self.agents = [f"agent_{i}" for i in range(num_agents)]
        self.obs_shape = obs_shape

    def reset(self, key):
        obs = {a: jnp.zeros(self.obs_shape, dtype=jnp.float32) for a in self.agents}
        return obs, FakeInnerState(counter=jnp.int32(0))

    def step(self, key, state, actions):
        obs = {a: jnp.zeros(self.obs_shape, dtype=jnp.float32) for a in self.agents}
        new_state = FakeInnerState(counter=state.counter + 1)
        return obs, new_state, jnp.zeros(self.num_agents), False, {}

    def observation_space(self, agent_id):
        return None

    def action_space(self, agent_id):
        class _Spec:
            n = 9

        return _Spec()


def _all_events(state_before, state_after, actions):
    return jnp.ones(actions.shape[0], dtype=jnp.int8)


def _no_events(state_before, state_after, actions):
    return jnp.zeros(actions.shape[0], dtype=jnp.int8)


def _only_agent_zero(state_before, state_after, actions):
    e = jnp.zeros(actions.shape[0], dtype=jnp.int8)
    return e.at[0].set(1)


class TestReset:
    def test_initial_alpha_is_zeros(self):
        env = FakeEnv(num_agents=3)
        wrapper = AttributionWrapper(env, _all_events, window_size=4)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        assert ws.current_alpha.tolist() == [0.0, 0.0, 0.0]
        assert int(ws.step_in_window) == 0
        assert ws.event_buffer.shape == (4, 3)
        assert int(ws.event_buffer.sum()) == 0

    def test_obs_augmented_with_alpha_channels(self):
        env = FakeEnv(num_agents=3, obs_shape=(2, 2, 4))
        wrapper = AttributionWrapper(env, _all_events)
        obs, _ = wrapper.reset(jax.random.PRNGKey(0))
        # Original (2, 2, 4) + num_agents=3 alpha channels → (2, 2, 7).
        for v in obs.values():
            assert v.shape == (2, 2, 7)


class TestWindowDynamics:
    def test_alpha_constant_within_first_window(self):
        env = FakeEnv(num_agents=3)
        wrapper = AttributionWrapper(env, _all_events, window_size=5)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        for _ in range(4):
            _, ws, _, _, _ = wrapper.step(
                jax.random.PRNGKey(1), ws, jnp.array([0, 1, 2])
            )
        # Still inside the first window (step_in_window = 4, not yet 5).
        assert ws.current_alpha.tolist() == [0.0, 0.0, 0.0]
        assert int(ws.step_in_window) == 4

    def test_alpha_updates_at_window_boundary_all_one(self):
        env = FakeEnv(num_agents=3)
        wrapper = AttributionWrapper(env, _all_events, window_size=5)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        for _ in range(5):
            _, ws, _, _, _ = wrapper.step(
                jax.random.PRNGKey(1), ws, jnp.array([0, 1, 2])
            )
        assert ws.current_alpha.tolist() == [1.0, 1.0, 1.0]
        assert int(ws.step_in_window) == 0  # wrapped back to 0

    def test_alpha_zero_when_no_events(self):
        env = FakeEnv(num_agents=3)
        wrapper = AttributionWrapper(env, _no_events, window_size=5)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        for _ in range(5):
            _, ws, _, _, _ = wrapper.step(
                jax.random.PRNGKey(1), ws, jnp.array([0, 1, 2])
            )
        assert ws.current_alpha.tolist() == [0.0, 0.0, 0.0]

    def test_alpha_differs_per_agent(self):
        env = FakeEnv(num_agents=3)
        wrapper = AttributionWrapper(env, _only_agent_zero, window_size=5)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        for _ in range(5):
            _, ws, _, _, _ = wrapper.step(
                jax.random.PRNGKey(1), ws, jnp.array([0, 1, 2])
            )
        assert ws.current_alpha.tolist() == [1.0, 0.0, 0.0]


class TestEventBuffer:
    def test_buffer_records_events_in_order(self):
        env = FakeEnv(num_agents=3)
        wrapper = AttributionWrapper(env, _only_agent_zero, window_size=5)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        for step in range(3):
            _, ws, _, _, _ = wrapper.step(
                jax.random.PRNGKey(1), ws, jnp.array([0, 1, 2])
            )
            # Slot just written should match predicate output.
            assert ws.event_buffer[step].tolist() == [1, 0, 0]
        assert int(ws.step_in_window) == 3


class TestAlphaInObs:
    def test_alpha_appears_as_obs_channels_after_boundary(self):
        env = FakeEnv(num_agents=3, obs_shape=(2, 2, 4))
        wrapper = AttributionWrapper(env, _only_agent_zero, window_size=3)
        _, ws = wrapper.reset(jax.random.PRNGKey(0))
        # First 3 steps fill the window; the 3rd returns obs with updated alpha.
        for _ in range(3):
            obs, ws, _, _, _ = wrapper.step(
                jax.random.PRNGKey(1), ws, jnp.array([0, 1, 2])
            )
        # alpha = [1, 0, 0]. Each agent's obs has trailing 3 channels = alpha.
        for v in obs.values():
            # Take the last 3 channels at any spatial position; they should equal alpha.
            assert v[0, 0, -3:].tolist() == [1.0, 0.0, 0.0]
            assert v[1, 1, -3:].tolist() == [1.0, 0.0, 0.0]
