"""Cooperative-event predicate for SocialJax environments.

The predicate e_k(i) returns 1 if agent i performed a cooperative event at
step k, else 0. Cooperation rate over a window of N steps is

    alpha(i) = (1 / N) * sum_k e_k(i)

This module has two layers:

1. Pure-logic predicates (cleanup_event, harvest_event) — take precomputed
   counts/booleans and return 0/1. Easy to unit-test against synthetic data.
2. State extractors (cleanup_event_from_state, harvest_event_from_state) —
   pull the relevant counts from a pair of SocialJax `State` objects and
   delegate to the pure predicates. These are called off the JAX hot loop.

See docs/predicate.md for the precise specification and known limitations.
"""

import jax.numpy as jnp

CLEAN_ACTION = 8
"""Actions.zap_clean in external/SocialJax/socialjax/environments/cleanup/clean_up.py."""

DIRT_LABEL = 8
"""Items.dirt label in SocialJax cleanup state (also used in potential_dirt_and_dirt_label)."""

APPLE_LABEL = 3
"""Items.apple label in SocialJax grid (both Cleanup and Harvest:Open)."""

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


# ---------------------------------------------------------------------------
# State extractors: pull predicate inputs from SocialJax State objects.
# Expected fields (see external/SocialJax/socialjax/environments/{cleanup,common_harvest}/):
#   Cleanup: state.potential_dirt_and_dirt_label, state.agent_invs, state.agent_locs
#   Harvest: state.grid (H, W), state.agent_invs (N, 2), state.agent_locs (N, 3)
# ---------------------------------------------------------------------------


def cleanup_dirt_count(state) -> int:
    """Count of cells labelled Items.dirt in a cleanup state."""
    return int(jnp.sum(state.potential_dirt_and_dirt_label == DIRT_LABEL))


def cleanup_event_from_state(state_before, state_after, action_i) -> int:
    """Compute Cleanup e_k(i) from a pair of SocialJax cleanup states."""
    used_clean = int(action_i) == CLEAN_ACTION
    return cleanup_event(
        used_clean,
        cleanup_dirt_count(state_before),
        cleanup_dirt_count(state_after),
    )


def apples_in_radius(grid, row: int, col: int, radius: int = APPLE_NEIGHBOR_RADIUS) -> int:
    """Count Items.apple tiles in the Chebyshev radius around (row, col)."""
    h, w = grid.shape
    row_lo = max(0, row - radius)
    row_hi = min(h, row + radius + 1)
    col_lo = max(0, col - radius)
    col_hi = min(w, col + radius + 1)
    return int(jnp.sum(grid[row_lo:row_hi, col_lo:col_hi] == APPLE_LABEL))


def harvest_event_from_state(
    state_before,
    state_after,
    agent_i: int,
    radius: int = APPLE_NEIGHBOR_RADIUS,
    threshold: int = 1,
) -> int:
    """Compute Harvest:Open e_k(i) from a pair of SocialJax harvest states.

    Agent's post-step location (state_after.agent_locs[agent_i]) defines the
    neighborhood centre — that is the tile the agent just stepped onto. The
    just-harvested apple itself is no longer in the grid (replaced by the
    agent's value), so `apples_in_radius` counts *other* nearby apples.
    """
    inv_before = int(jnp.sum(state_before.agent_invs[agent_i]))
    inv_after = int(jnp.sum(state_after.agent_invs[agent_i]))
    loc_after = state_after.agent_locs[agent_i]
    row, col = int(loc_after[0]), int(loc_after[1])
    apples_after = apples_in_radius(state_after.grid, row, col, radius=radius)
    return harvest_event(inv_before, inv_after, apples_after, threshold=threshold)
