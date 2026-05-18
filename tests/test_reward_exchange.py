"""Tests for the Willis reward-exchange map and RewardExchangeWrapper (Track B).

`reward_exchange` is a pure function — its correctness is fully determined by
arithmetic, so the unit tests pin exact values (no training, no randomness).
The wrapper tests use a mock env with a known fixed reward; the e2e test wraps
real SocialJax Cleanup to confirm the wrapper integrates without crashing and
that the exchange conserves total reward under real env dynamics.
"""

import jax
import jax.numpy as jnp
import pytest
from flax.struct import dataclass as flax_dataclass

from zkattribution.wrappers import RewardExchangeWrapper, reward_exchange


# --- reward_exchange: pure-function tests -----------------------------------


class TestRewardExchangePure:
    def test_s_equals_one_is_identity(self):
        r = jnp.array([3.0, 0.0, 0.0, 1.0])
        out = reward_exchange(r, s=1.0, num_agents=4)
        assert out.tolist() == pytest.approx([3.0, 0.0, 0.0, 1.0])

    def test_s_equals_one_over_n_is_uniform_mean(self):
        # s = 1/n -> every agent receives the mean reward.
        r = jnp.array([3.0, 0.0, 0.0, 1.0])  # total 4, mean 1.0
        out = reward_exchange(r, s=0.25, num_agents=4)
        assert out.tolist() == pytest.approx([1.0, 1.0, 1.0, 1.0])

    def test_intermediate_s_hand_computed(self):
        # n=3, r=[6,0,0], s=0.5, n-1=2:
        #   R'_0 = .5*6 + .5*(6-6)/2 = 3
        #   R'_1 = .5*0 + .5*(6-0)/2 = 1.5
        #   R'_2 = .5*0 + .5*(6-0)/2 = 1.5
        r = jnp.array([6.0, 0.0, 0.0])
        out = reward_exchange(r, s=0.5, num_agents=3)
        assert out.tolist() == pytest.approx([3.0, 1.5, 1.5])

    def test_total_conserved_for_arbitrary_s(self):
        r = jnp.array([5.0, 2.0, 0.0, 1.0, 7.0])
        for s in (1.0, 0.7, 0.4, 0.2):
            out = reward_exchange(r, s=s, num_agents=5)
            assert float(jnp.sum(out)) == pytest.approx(float(jnp.sum(r)))

    def test_uniform_reward_unchanged(self):
        # If everyone already has equal reward, exchange is a no-op for any s.
        r = jnp.array([2.0, 2.0, 2.0])
        for s in (1.0, 0.6, 1 / 3):
            out = reward_exchange(r, s=s, num_agents=3)
            assert out.tolist() == pytest.approx([2.0, 2.0, 2.0])

    def test_batched_input_handled_elementwise(self):
        # A leading batch axis (e.g. num_envs) is exchanged row by row, never
        # mixed across the batch. At s=1/n each row collapses to its own mean;
        # rows with different totals must give different means.
        r = jnp.array([[6.0, 0.0, 0.0], [9.0, 0.0, 0.0]])
        out = reward_exchange(r, s=1 / 3, num_agents=3)
        expected = jnp.array([[2.0, 2.0, 2.0], [3.0, 3.0, 3.0]])
        assert bool(jnp.allclose(out, expected))


# --- RewardExchangeWrapper: mock-env tests ----------------------------------


@flax_dataclass
class _FakeState:
    """Minimal flax-struct state so the mock env's state is PyTree-registered."""

    counter: jnp.ndarray


class _FixedRewardEnv:
    """Minimal SocialJax-shaped env whose `step` returns a fixed reward vector."""

    def __init__(self, reward_vec):
        self._reward = jnp.asarray(reward_vec, dtype=jnp.float32)
        self.num_agents = int(self._reward.shape[0])
        self.agents = [f"agent_{i}" for i in range(self.num_agents)]

    def reset(self, key):
        return jnp.zeros((self.num_agents, 3)), _FakeState(counter=jnp.int32(0))

    def step(self, key, state, action):
        obs = jnp.zeros((self.num_agents, 3))
        new_state = _FakeState(counter=state.counter + 1)
        return obs, new_state, self._reward, False, {"k": jnp.int32(7)}


class TestRewardExchangeWrapper:
    def test_s_one_is_reward_passthrough(self):
        env = _FixedRewardEnv([4.0, 0.0, 0.0])
        w = RewardExchangeWrapper(env, s=1.0)
        _, st = w.reset(jax.random.PRNGKey(0))
        _, _, reward, _, _ = w.step(jax.random.PRNGKey(1), st, jnp.zeros(3))
        assert reward.tolist() == pytest.approx([4.0, 0.0, 0.0])

    def test_s_one_over_n_gives_mean(self):
        env = _FixedRewardEnv([4.0, 0.0, 0.0])  # total 4, mean 4/3
        w = RewardExchangeWrapper(env, s=1 / 3)
        _, st = w.reset(jax.random.PRNGKey(0))
        _, _, reward, _, _ = w.step(jax.random.PRNGKey(1), st, jnp.zeros(3))
        assert reward.tolist() == pytest.approx([4 / 3, 4 / 3, 4 / 3])

    def test_total_conserved_through_wrapper(self):
        env = _FixedRewardEnv([1.0, 2.0, 3.0, 4.0])  # total 10
        w = RewardExchangeWrapper(env, s=0.4)
        _, st = w.reset(jax.random.PRNGKey(0))
        _, _, reward, _, _ = w.step(jax.random.PRNGKey(1), st, jnp.zeros(4))
        assert float(jnp.sum(reward)) == pytest.approx(10.0)

    def test_state_and_info_pass_through_untouched(self):
        # The wrapper is stateless: inner state and info are unchanged.
        env = _FixedRewardEnv([1.0, 2.0, 3.0])
        w = RewardExchangeWrapper(env, s=0.5)
        _, st = w.reset(jax.random.PRNGKey(0))
        _, st_next, _, done, info = w.step(jax.random.PRNGKey(1), st, jnp.zeros(3))
        assert int(st_next.counter) == 1
        assert int(info["k"]) == 7
        assert not bool(done)

    def test_rejects_out_of_range_s(self):
        env = _FixedRewardEnv([1.0, 1.0])
        with pytest.raises(ValueError):
            RewardExchangeWrapper(env, s=1.5)
        with pytest.raises(ValueError):
            RewardExchangeWrapper(env, s=0.0)


# --- e2e: wrap real SocialJax Cleanup ---------------------------------------


@pytest.mark.e2e
class TestRewardExchangeCleanupE2E:
    def test_wraps_cleanup_and_steps(self):
        import socialjax

        num_agents = 5
        env = socialjax.make(
            "clean_up",
            num_agents=num_agents,
            num_inner_steps=200,
            shared_rewards=False,  # B1 must see raw per-agent rewards
        )
        w = RewardExchangeWrapper(env, s=0.5)
        num_actions = env.action_space(0).n

        rng = jax.random.PRNGKey(0)
        rng, reset_rng = jax.random.split(rng)
        _, st = w.reset(reset_rng)

        for _ in range(30):
            rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
            actions = jnp.stack(
                [
                    jax.random.randint(agent_rngs[i], (), 0, num_actions)
                    for i in range(num_agents)
                ]
            )
            rng, step_rng = jax.random.split(rng)
            _, st, reward, _, _ = w.step(step_rng, st, actions)
            # Reward keeps the per-agent shape and stays finite.
            assert reward.shape == (num_agents,)
            assert bool(jnp.all(jnp.isfinite(reward)))

    def test_exchange_conserves_total_vs_passthrough(self):
        # Same env, same seed, same actions: s=1 (passthrough) and s=0.5 must
        # produce the same per-step reward total — the exchange redistributes
        # reward but conserves it.
        import socialjax

        num_agents = 5

        def _rollout(s):
            env = socialjax.make(
                "clean_up",
                num_agents=num_agents,
                num_inner_steps=200,
                shared_rewards=False,
            )
            w = RewardExchangeWrapper(env, s=s)
            n_act = env.action_space(0).n
            rng = jax.random.PRNGKey(42)
            rng, reset_rng = jax.random.split(rng)
            _, st = w.reset(reset_rng)
            totals = []
            for _ in range(60):
                rng, *agent_rngs = jax.random.split(rng, num_agents + 1)
                actions = jnp.stack(
                    [
                        jax.random.randint(agent_rngs[i], (), 0, n_act)
                        for i in range(num_agents)
                    ]
                )
                rng, step_rng = jax.random.split(rng)
                _, st, reward, _, _ = w.step(step_rng, st, actions)
                totals.append(float(jnp.sum(reward)))
            return totals

        passthrough = _rollout(1.0)
        exchanged = _rollout(0.5)
        for tp, te in zip(passthrough, exchanged):
            assert te == pytest.approx(tp)
