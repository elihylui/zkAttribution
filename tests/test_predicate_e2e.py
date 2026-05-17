"""End-to-end tests for predicate state extractors against real SocialJax episodes.

These tests import `socialjax` from `external/SocialJax/` (see pyproject.toml
pytest pythonpath) and run a short episode with random actions, then verify
that the predicate extractors produce values consistent with the live env.

Marked `e2e` because they involve JAX compilation and take ~30s on first run.
"""

import jax
import jax.numpy as jnp
import pytest

from zkattribution.predicate import (
    CLEAN_ACTION,
    cleanup_dirt_count,
    cleanup_event_from_state,
    cleanup_events_batch,
    cooperation_rate,
    harvest_event_from_state,
    harvest_events_batch,
)


NUM_STEPS = 100
SEED = 42


def _random_action(rng, num_actions):
    return jax.random.randint(rng, shape=(), minval=0, maxval=num_actions)


@pytest.fixture(scope="module")
def cleanup_rollout():
    """Run NUM_STEPS of Cleanup with uniform-random actions; return state sequence and actions."""
    import socialjax

    num_agents = 5
    rng = jax.random.PRNGKey(SEED)
    env = socialjax.make(
        "clean_up",
        num_agents=num_agents,
        num_inner_steps=NUM_STEPS + 50,
    )
    num_actions = env.action_space(0).n

    rng, reset_rng = jax.random.split(rng)
    _obs, state = env.reset(reset_rng)

    states = [state]
    actions_per_step = []
    for _ in range(NUM_STEPS):
        rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
        actions = [_random_action(agent_rngs[i], num_actions) for i in range(num_agents)]
        rng, step_rng = jax.random.split(rng)
        _obs, state, _r, _d, _info = env.step_env(step_rng, state, actions)
        states.append(state)
        actions_per_step.append(actions)

    return {
        "states": states,
        "actions_per_step": actions_per_step,
        "num_agents": num_agents,
    }


@pytest.fixture(scope="module")
def harvest_rollout():
    """Run NUM_STEPS of Harvest:Open with uniform-random actions."""
    import socialjax

    num_agents = 5
    rng = jax.random.PRNGKey(SEED)
    env = socialjax.make(
        "harvest_common_open",
        num_agents=num_agents,
        num_inner_steps=NUM_STEPS + 50,
    )
    num_actions = env.action_space(0).n

    rng, reset_rng = jax.random.split(rng)
    _obs, state = env.reset(reset_rng)

    states = [state]
    actions_per_step = []
    for _ in range(NUM_STEPS):
        rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
        actions = [_random_action(agent_rngs[i], num_actions) for i in range(num_agents)]
        rng, step_rng = jax.random.split(rng)
        _obs, state, _r, _d, _info = env.step_env(step_rng, state, actions)
        states.append(state)
        actions_per_step.append(actions)

    return {
        "states": states,
        "actions_per_step": actions_per_step,
        "num_agents": num_agents,
    }


@pytest.mark.e2e
class TestCleanupEndToEnd:
    def test_alpha_in_unit_interval(self, cleanup_rollout):
        states = cleanup_rollout["states"]
        actions_per_step = cleanup_rollout["actions_per_step"]
        num_agents = cleanup_rollout["num_agents"]

        for i in range(num_agents):
            events = [
                cleanup_event_from_state(
                    states[t], states[t + 1], i, actions_per_step[t][i]
                )
                for t in range(NUM_STEPS)
            ]
            alpha = cooperation_rate(events)
            assert 0.0 <= alpha <= 1.0, f"agent {i} alpha={alpha} outside [0, 1]"

    def test_non_clean_action_never_yields_event(self, cleanup_rollout):
        """If agent didn't fire zap_clean at step t, e_k(i, t) must be 0."""
        states = cleanup_rollout["states"]
        actions_per_step = cleanup_rollout["actions_per_step"]
        num_agents = cleanup_rollout["num_agents"]

        for t in range(NUM_STEPS):
            for i in range(num_agents):
                action = int(actions_per_step[t][i])
                if action != CLEAN_ACTION:
                    e_k = cleanup_event_from_state(states[t], states[t + 1], i, action)
                    assert e_k == 0, (
                        f"agent {i} step {t}: action {action} (not zap_clean) but e_k={e_k}"
                    )

    def test_dirt_count_is_nonnegative_and_bounded(self, cleanup_rollout):
        states = cleanup_rollout["states"]
        total_slots = int(states[0].potential_dirt_and_dirt_label.shape[0])
        for s in states:
            d = cleanup_dirt_count(s)
            assert 0 <= d <= total_slots


@pytest.mark.e2e
class TestHarvestEndToEnd:
    def test_alpha_in_unit_interval(self, harvest_rollout):
        states = harvest_rollout["states"]
        num_agents = harvest_rollout["num_agents"]

        for i in range(num_agents):
            events = [
                harvest_event_from_state(states[t], states[t + 1], agent_i=i)
                for t in range(NUM_STEPS)
            ]
            alpha = cooperation_rate(events)
            assert 0.0 <= alpha <= 1.0

    def test_no_inventory_growth_means_zero_event(self, harvest_rollout):
        """If agent's inventory didn't grow, e_k(i, t) must be 0 regardless of neighbors."""
        states = harvest_rollout["states"]
        num_agents = harvest_rollout["num_agents"]

        for t in range(NUM_STEPS):
            for i in range(num_agents):
                inv_before = int(jnp.sum(states[t].agent_invs[i]))
                inv_after = int(jnp.sum(states[t + 1].agent_invs[i]))
                if inv_after <= inv_before:
                    e_k = harvest_event_from_state(states[t], states[t + 1], agent_i=i)
                    assert e_k == 0, (
                        f"agent {i} step {t}: inv unchanged ({inv_before}->{inv_after}) "
                        f"but e_k={e_k}"
                    )


@pytest.mark.e2e
class TestBatchedPredicatesJit:
    """Confirm the JAX-native batched predicates trace under jit and agree with
    the host-side reference functions on real SocialJax states."""

    def test_cleanup_events_batch_jit_matches_reference(self, cleanup_rollout):
        states = cleanup_rollout["states"]
        actions_per_step = cleanup_rollout["actions_per_step"]
        num_agents = cleanup_rollout["num_agents"]

        jitted = jax.jit(cleanup_events_batch)

        # Sample 10 evenly-spaced steps.
        for t in range(0, NUM_STEPS, NUM_STEPS // 10):
            actions = jnp.stack(actions_per_step[t])
            batch_out = jitted(states[t], states[t + 1], actions)
            reference = [
                cleanup_event_from_state(states[t], states[t + 1], i, int(actions[i]))
                for i in range(num_agents)
            ]
            assert batch_out.tolist() == reference, (
                f"step {t}: jit batch {batch_out.tolist()} != reference {reference}"
            )

    def test_harvest_events_batch_jit_matches_reference(self, harvest_rollout):
        states = harvest_rollout["states"]
        num_agents = harvest_rollout["num_agents"]

        jitted = jax.jit(harvest_events_batch)

        for t in range(0, NUM_STEPS, NUM_STEPS // 10):
            batch_out = jitted(states[t], states[t + 1])
            reference = [
                harvest_event_from_state(states[t], states[t + 1], agent_i=i)
                for i in range(num_agents)
            ]
            assert batch_out.tolist() == reference, (
                f"step {t}: jit batch {batch_out.tolist()} != reference {reference}"
            )
