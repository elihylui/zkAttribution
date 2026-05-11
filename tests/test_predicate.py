import pytest

from zkattribution.predicate import (
    cleanup_event,
    cooperation_rate,
    harvest_event,
)


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
