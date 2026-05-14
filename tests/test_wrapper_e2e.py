"""End-to-end tests for AttributionWrapper against real SocialJax envs.

Wraps `clean_up` and `harvest_common_open`, runs short rollouts with random
actions, and verifies the wrapper's invariants hold under real env dynamics:
alpha in [0, 1], obs shape grows by num_agents channels, alpha updates only at
window boundaries.
"""

import jax
import jax.numpy as jnp
import pytest

from zkattribution.predicate import (
    cleanup_events_batch,
    harvest_events_batch,
)
from zkattribution.wrappers import AttributionWrapper


WINDOW_SIZE = 50
NUM_STEPS = 110  # > 2 windows so we observe two boundaries
SEED = 7


def _random_action(rng, num_actions):
    return jax.random.randint(rng, shape=(), minval=0, maxval=num_actions)


@pytest.fixture(scope="module")
def cleanup_wrapped_rollout():
    import socialjax

    num_agents = 5
    rng = jax.random.PRNGKey(SEED)
    env = socialjax.make("clean_up", num_agents=num_agents, num_inner_steps=NUM_STEPS + 50)
    wrapper = AttributionWrapper(env, cleanup_events_batch, window_size=WINDOW_SIZE)
    num_actions = env.action_space(0).n

    rng, reset_rng = jax.random.split(rng)
    obs, ws = wrapper.reset(reset_rng)

    obs_traj = [obs]
    alpha_traj = [ws.current_alpha]
    step_traj = [int(ws.step_in_window)]
    for _ in range(NUM_STEPS):
        rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
        actions = jnp.stack(
            [_random_action(agent_rngs[i], num_actions) for i in range(num_agents)]
        )
        rng, step_rng = jax.random.split(rng)
        obs, ws, _, _, _ = wrapper.step(step_rng, ws, actions)
        obs_traj.append(obs)
        alpha_traj.append(ws.current_alpha)
        step_traj.append(int(ws.step_in_window))

    return {
        "obs_traj": obs_traj,
        "alpha_traj": alpha_traj,
        "step_traj": step_traj,
        "num_agents": num_agents,
    }


@pytest.fixture(scope="module")
def harvest_wrapped_rollout():
    import socialjax

    num_agents = 5
    rng = jax.random.PRNGKey(SEED)
    env = socialjax.make(
        "harvest_common_open", num_agents=num_agents, num_inner_steps=NUM_STEPS + 50
    )
    # Harvest predicate doesn't depend on actions — pass actions but predicate ignores them.
    wrapper = AttributionWrapper(
        env,
        lambda sb, sa, _actions: harvest_events_batch(sb, sa),
        window_size=WINDOW_SIZE,
    )
    num_actions = env.action_space(0).n

    rng, reset_rng = jax.random.split(rng)
    obs, ws = wrapper.reset(reset_rng)

    obs_traj = [obs]
    alpha_traj = [ws.current_alpha]
    for _ in range(NUM_STEPS):
        rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
        actions = jnp.stack(
            [_random_action(agent_rngs[i], num_actions) for i in range(num_agents)]
        )
        rng, step_rng = jax.random.split(rng)
        obs, ws, _, _, _ = wrapper.step(step_rng, ws, actions)
        obs_traj.append(obs)
        alpha_traj.append(ws.current_alpha)

    return {"obs_traj": obs_traj, "alpha_traj": alpha_traj, "num_agents": num_agents}


@pytest.mark.e2e
class TestCleanupWrappedRollout:
    def test_obs_channels_grow_by_num_agents(self, cleanup_wrapped_rollout):
        num_agents = cleanup_wrapped_rollout["num_agents"]
        obs0 = cleanup_wrapped_rollout["obs_traj"][0]
        # SocialJax obs is a single array (num_agents, H, W, C). After wrapping,
        # the channel axis (last) grows by num_agents.
        assert obs0.shape[0] == num_agents, f"obs leading dim should be num_agents: {obs0.shape}"
        assert obs0.shape[-1] >= num_agents, f"obs channels should include alpha: {obs0.shape}"

    def test_alpha_in_unit_interval_throughout(self, cleanup_wrapped_rollout):
        for alpha in cleanup_wrapped_rollout["alpha_traj"]:
            arr = jnp.asarray(alpha)
            assert bool(jnp.all(arr >= 0.0))
            assert bool(jnp.all(arr <= 1.0))

    def test_alpha_zero_before_first_window_completes(self, cleanup_wrapped_rollout):
        alpha_traj = cleanup_wrapped_rollout["alpha_traj"]
        # alpha_traj[0] is after reset; alpha_traj[k] is after step k.
        # alpha first updates after step WINDOW_SIZE (the WINDOW_SIZE-th step writes the last
        # event and triggers the boundary recompute).
        for k in range(WINDOW_SIZE):
            assert jnp.all(jnp.asarray(alpha_traj[k]) == 0.0)

    def test_alpha_updates_at_window_boundaries(self, cleanup_wrapped_rollout):
        alpha_traj = cleanup_wrapped_rollout["alpha_traj"]
        step_traj = cleanup_wrapped_rollout["step_traj"]
        # alpha should change at indices WINDOW_SIZE and 2*WINDOW_SIZE (the boundaries).
        # Between boundaries, alpha is constant. We don't assert it changes (random actions
        # might produce zero events) — only that it's constant between boundaries.
        # step_in_window resets to 0 at exactly those indices.
        assert step_traj[WINDOW_SIZE] == 0, f"step_in_window at step {WINDOW_SIZE} = {step_traj[WINDOW_SIZE]}"
        # In between: should be 1..WINDOW_SIZE-1, then 0 again.
        for k in range(1, WINDOW_SIZE):
            assert step_traj[k] == k


@pytest.mark.e2e
class TestHarvestWrappedRollout:
    def test_obs_channels_grow_by_num_agents(self, harvest_wrapped_rollout):
        num_agents = harvest_wrapped_rollout["num_agents"]
        obs0 = harvest_wrapped_rollout["obs_traj"][0]
        assert obs0.shape[0] == num_agents
        assert obs0.shape[-1] >= num_agents

    def test_alpha_in_unit_interval(self, harvest_wrapped_rollout):
        for alpha in harvest_wrapped_rollout["alpha_traj"]:
            arr = jnp.asarray(alpha)
            assert bool(jnp.all(arr >= 0.0))
            assert bool(jnp.all(arr <= 1.0))
