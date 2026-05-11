from dataclasses import dataclass

import jax.numpy as jnp
import pytest

from zkattribution.predicate import (
    APPLE_LABEL,
    CLEAN_ACTION,
    DIRT_LABEL,
    apples_in_radius,
    cleanup_dirt_count,
    cleanup_event,
    cleanup_event_from_state,
    cooperation_rate,
    harvest_event,
    harvest_event_from_state,
)


@dataclass
class FakeCleanupState:
    potential_dirt_and_dirt_label: jnp.ndarray


@dataclass
class FakeHarvestState:
    grid: jnp.ndarray
    agent_invs: jnp.ndarray
    agent_locs: jnp.ndarray


class TestCleanupEvent:
    def test_clean_action_and_dirt_decreased(self):
        assert cleanup_event(True, 10, 9) == 1

    def test_clean_action_but_dirt_unchanged(self):
        # Beam fired but missed every dirt tile.
        assert cleanup_event(True, 10, 10) == 0

    def test_no_clean_action_even_if_dirt_decreased(self):
        # Another agent cleaned; this one didn't attempt.
        assert cleanup_event(False, 10, 9) == 0

    def test_clean_action_no_dirt_to_clean(self):
        assert cleanup_event(True, 0, 0) == 0

    def test_no_clean_action_no_change(self):
        assert cleanup_event(False, 10, 10) == 0

    def test_dirt_increased_is_not_cooperation(self):
        # Defensive: pollution spawn could fire same step; not a cooperative event.
        assert cleanup_event(True, 5, 6) == 0


class TestHarvestEvent:
    def test_harvested_with_apples_remaining(self):
        assert harvest_event(inv_sum_before=0, inv_sum_after=1, apples_in_radius_after=3) == 1

    def test_harvested_at_threshold_boundary(self):
        # Exactly threshold remaining still counts.
        assert harvest_event(inv_sum_before=0, inv_sum_after=1, apples_in_radius_after=1) == 1

    def test_harvested_but_emptied_patch(self):
        # Destructive harvest — patch is exhausted.
        assert harvest_event(inv_sum_before=0, inv_sum_after=1, apples_in_radius_after=0) == 0

    def test_did_not_harvest(self):
        assert harvest_event(inv_sum_before=0, inv_sum_after=0, apples_in_radius_after=5) == 0

    def test_inventory_unchanged(self):
        assert harvest_event(inv_sum_before=5, inv_sum_after=5, apples_in_radius_after=5) == 0

    def test_harvest_later_in_episode(self):
        assert harvest_event(inv_sum_before=5, inv_sum_after=6, apples_in_radius_after=2) == 1

    def test_custom_threshold_strict(self):
        assert harvest_event(0, 1, apples_in_radius_after=1, threshold=2) == 0
        assert harvest_event(0, 1, apples_in_radius_after=2, threshold=2) == 1


class TestCooperationRate:
    def test_basic_fraction(self):
        assert cooperation_rate([1, 0, 1, 1, 0]) == pytest.approx(0.6)

    def test_all_zero(self):
        assert cooperation_rate([0, 0, 0]) == 0.0

    def test_all_one(self):
        assert cooperation_rate([1, 1, 1, 1]) == 1.0

    def test_single_step_window(self):
        assert cooperation_rate([1]) == 1.0
        assert cooperation_rate([0]) == 0.0

    def test_empty_window_raises(self):
        with pytest.raises(ValueError):
            cooperation_rate([])


class TestCleanupDirtCount:
    def test_counts_dirt_labels_only(self):
        # Label array mixes potential_dirt (7) and dirt (8).
        labels = jnp.array([DIRT_LABEL, 7, DIRT_LABEL, 7, 7])
        assert cleanup_dirt_count(FakeCleanupState(labels)) == 2

    def test_no_dirt(self):
        labels = jnp.array([7, 7, 7])
        assert cleanup_dirt_count(FakeCleanupState(labels)) == 0

    def test_all_dirt(self):
        labels = jnp.array([DIRT_LABEL] * 4)
        assert cleanup_dirt_count(FakeCleanupState(labels)) == 4


class TestCleanupEventFromState:
    def test_clean_action_and_dirt_decreased(self):
        before = FakeCleanupState(jnp.array([DIRT_LABEL, DIRT_LABEL, DIRT_LABEL, 7]))
        after = FakeCleanupState(jnp.array([DIRT_LABEL, DIRT_LABEL, 7, 7]))
        assert cleanup_event_from_state(before, after, action_i=CLEAN_ACTION) == 1

    def test_no_clean_action_even_if_dirt_decreased(self):
        before = FakeCleanupState(jnp.array([DIRT_LABEL, DIRT_LABEL, DIRT_LABEL]))
        after = FakeCleanupState(jnp.array([DIRT_LABEL, DIRT_LABEL, 7]))
        assert cleanup_event_from_state(before, after, action_i=4) == 0  # up, not clean

    def test_clean_action_but_dirt_unchanged(self):
        labels = jnp.array([DIRT_LABEL, DIRT_LABEL, 7])
        before = FakeCleanupState(labels)
        after = FakeCleanupState(labels)
        assert cleanup_event_from_state(before, after, action_i=CLEAN_ACTION) == 0


class TestApplesInRadius:
    def test_radius_1_around_centre(self):
        grid = jnp.array(
            [
                [APPLE_LABEL, 0, 0, 0, APPLE_LABEL],
                [0, 0, 0, 0, 0],
                [0, 0, APPLE_LABEL, 0, 0],
                [0, 0, 0, 0, 0],
                [APPLE_LABEL, 0, 0, 0, APPLE_LABEL],
            ]
        )
        assert apples_in_radius(grid, 2, 2, radius=1) == 1  # only centre apple

    def test_radius_2_around_centre(self):
        grid = jnp.array(
            [
                [APPLE_LABEL, 0, 0, 0, APPLE_LABEL],
                [0, 0, 0, 0, 0],
                [0, 0, APPLE_LABEL, 0, 0],
                [0, 0, 0, 0, 0],
                [APPLE_LABEL, 0, 0, 0, APPLE_LABEL],
            ]
        )
        assert apples_in_radius(grid, 2, 2, radius=2) == 5  # all 5

    def test_edge_clipping_at_corner(self):
        grid = jnp.zeros((3, 3), dtype=jnp.int16).at[0, 0].set(APPLE_LABEL)
        assert apples_in_radius(grid, 0, 0, radius=1) == 1

    def test_no_apples(self):
        grid = jnp.zeros((4, 4), dtype=jnp.int16)
        assert apples_in_radius(grid, 2, 2, radius=2) == 0


class TestHarvestEventFromState:
    def _make_before(self, agent_loc, inv):
        # Pre-step grid is irrelevant for the predicate; only after-state matters.
        return FakeHarvestState(
            grid=jnp.zeros((5, 5), dtype=jnp.int16),
            agent_invs=jnp.array([inv]),
            agent_locs=jnp.array([agent_loc]),
        )

    def test_harvest_with_neighbors_remaining(self):
        # Agent walked from (1,1) to (1,2). Apples remain at (1,4) and (2,2).
        grid_after = jnp.zeros((5, 5), dtype=jnp.int16)
        grid_after = grid_after.at[1, 2].set(99)  # agent value
        grid_after = grid_after.at[1, 4].set(APPLE_LABEL)
        grid_after = grid_after.at[2, 2].set(APPLE_LABEL)

        before = self._make_before(agent_loc=[1, 1, 0], inv=[0, 0])
        after = FakeHarvestState(
            grid=grid_after,
            agent_invs=jnp.array([[1, 0]]),
            agent_locs=jnp.array([[1, 2, 0]]),
        )
        assert harvest_event_from_state(before, after, agent_i=0) == 1

    def test_harvest_emptied_patch(self):
        # Agent harvests the last apple; no others nearby.
        grid_after = jnp.zeros((4, 4), dtype=jnp.int16).at[1, 2].set(99)
        before = self._make_before(agent_loc=[1, 1, 0], inv=[0, 0])
        after = FakeHarvestState(
            grid=grid_after,
            agent_invs=jnp.array([[1, 0]]),
            agent_locs=jnp.array([[1, 2, 0]]),
        )
        assert harvest_event_from_state(before, after, agent_i=0) == 0

    def test_did_not_harvest(self):
        grid_after = jnp.zeros((3, 3), dtype=jnp.int16).at[0, 1].set(APPLE_LABEL)
        before = self._make_before(agent_loc=[2, 1, 0], inv=[0, 0])
        after = FakeHarvestState(
            grid=grid_after,
            agent_invs=jnp.array([[0, 0]]),
            agent_locs=jnp.array([[2, 1, 0]]),
        )
        assert harvest_event_from_state(before, after, agent_i=0) == 0
