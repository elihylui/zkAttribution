"""Cooperative-event predicate for SocialJax environments.

The predicate e_k(i) returns 1 if agent i performed a cooperative event at
step k, else 0. Cooperation rate over a window of N steps is

    alpha(i) = (1 / N) * sum_k e_k(i)

See docs/predicate.md for the precise specification and known limitations.
"""

CLEAN_ACTION = 8
"""Actions.zap_clean in external/SocialJax/socialjax/environments/cleanup/clean_up.py."""

DIRT_LABEL = 8
"""Items.dirt label in SocialJax cleanup state."""

APPLE_NEIGHBOR_RADIUS = 2
"""Matches SocialJax's regrowth neighborhood radius for Harvest:Open."""


def cleanup_event(
    used_clean_action: bool,
    dirt_count_before: int,
    dirt_count_after: int,
) -> int:
    """e_k for Cleanup: agent fired zap_clean AND total dirt strictly decreased."""
    if used_clean_action and dirt_count_after < dirt_count_before:
        return 1
    return 0


def harvest_event(
    inv_sum_before: int,
    inv_sum_after: int,
    apples_in_radius_after: int,
    threshold: int = 1,
) -> int:
    """e_k for Harvest:Open: inventory grew AND >= `threshold` apples remain in radius."""
    if inv_sum_after > inv_sum_before and apples_in_radius_after >= threshold:
        return 1
    return 0


def cooperation_rate(events) -> float:
    """alpha = (1/N) * sum e_k over a single-agent window."""
    n = len(events)
    if n == 0:
        raise ValueError("cooperation_rate requires at least one event in the window")
    return sum(events) / n
