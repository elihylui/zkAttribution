"""Cooperative-event predicate for SocialJax environments.

The predicate e_k(i) returns 1 if agent i performed a cooperative event at
step k, else 0. Cooperation rate over a window of N steps is

    alpha(i) = (1 / N) * sum_k e_k(i)

Cleanup predicate — **v2, per-agent beam attribution**. An agent's `zap_clean`
is credited iff its beam footprint covers a dirt tile in the pre-step grid.
Because Cleanup's clean mechanic converts *any* in-beam dirt tile to clean
(clean_up.py `_interact`), "beam covers dirt" is equivalent to "this agent
cleans a tile". This replaced v1 (global dirt-count decrease), which was
confounded by the environment's per-step pollution spawning — an agent could
clean a tile yet see total dirt rise because dirt spawned elsewhere. The v1
functions are kept with a `_global_delta` suffix solely so the confound can be
measured against v2.

Harvest:Open's predicate is already per-agent (inventory growth + local apple
density) and is unchanged.

See docs/predicate.md for the precise specification.
"""

import jax
import jax.numpy as jnp

CLEAN_ACTION = 8
"""Actions.zap_clean in external/SocialJax/socialjax/environments/cleanup/clean_up.py."""

DIRT_LABEL = 8
"""Items.dirt label in SocialJax cleanup grid."""

APPLE_LABEL = 3
"""Items.apple label in SocialJax grid (both Cleanup and Harvest:Open)."""

APPLE_NEIGHBOR_RADIUS = 2
"""Matches SocialJax's regrowth neighborhood radius for Harvest:Open."""

_STEP_RC = jnp.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=jnp.int32)
"""(row, col) displacement per orientation {0:up, 1:right, 2:down, 3:left}.
Mirrors the STEP table in clean_up.py."""


# ---------------------------------------------------------------------------
# Pure predicates — precomputed inputs in, 0/1 out. Easy to unit-test.
# ---------------------------------------------------------------------------


def cleanup_event(
    used_clean_action: bool,
    dirt_count_before: int,
    dirt_count_after: int,
) -> int:
    """v1 pure logic (FLAWED): zap_clean fired AND global dirt count strictly
    decreased. Confounded by pollution spawning — see module docstring. Kept
    only for the confound measurement."""
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
# Cleanup beam geometry — mirrors clean_up.py `_interact`.
# ---------------------------------------------------------------------------


def cleanup_beam_tiles(loc, height: int, width: int):
    """The tiles a zap_clean beam covers for an agent at loc=(row, col, orient).

    Mirrors clean_up.py: one-step-forward, two-step-forward, forward-right,
    forward-left — with the right/left tiles collapsing to one-step-forward
    when out of bounds.

    Returns (rows, cols, in_bounds), each shape (4,). rows/cols are clipped
    into the grid so they are always safe to index with; in_bounds[k] is False
    when tile k was actually off-grid, so callers can mask it out (an off-grid
    tile cannot hold dirt).
    """
    r, c, o = loc[0], loc[1], loc[2]
    rc = jnp.array([r, c], dtype=jnp.int32)
    fwd = _STEP_RC[o]
    one = rc + fwd
    two = rc + 2 * fwd
    right = rc + fwd + _STEP_RC[(o + 1) % 4]
    left = rc + fwd + _STEP_RC[(o - 1) % 4]

    def _collapse(tile):
        oob = (
            (tile[0] < 0)
            | (tile[0] > height - 1)
            | (tile[1] < 0)
            | (tile[1] > width - 1)
        )
        return jnp.where(oob, one, tile)

    tiles = jnp.stack([one, two, _collapse(right), _collapse(left)])  # (4, 2)
    in_bounds = (
        (tiles[:, 0] >= 0)
        & (tiles[:, 0] <= height - 1)
        & (tiles[:, 1] >= 0)
        & (tiles[:, 1] <= width - 1)
    )
    rows = jnp.clip(tiles[:, 0], 0, height - 1)
    cols = jnp.clip(tiles[:, 1], 0, width - 1)
    return rows, cols, in_bounds


def cleanup_beam_hits_dirt(grid, loc) -> jnp.ndarray:
    """True iff the zap_clean beam from `loc` covers >= 1 in-bounds dirt tile."""
    h, w = grid.shape
    rows, cols, in_bounds = cleanup_beam_tiles(loc, h, w)
    return jnp.any((grid[rows, cols] == DIRT_LABEL) & in_bounds)


# ---------------------------------------------------------------------------
# State extractors — single-agent, host-side. Called off the JAX hot loop.
# ---------------------------------------------------------------------------


def cleanup_dirt_count(state) -> int:
    """Count of cells labelled Items.dirt in a cleanup state."""
    return int(jnp.sum(state.potential_dirt_and_dirt_label == DIRT_LABEL))


def cleanup_event_from_state(state_before, state_after, agent_i: int, action_i) -> int:
    """Cleanup e_k(i), v2 — agent fired zap_clean AND its beam covered a dirt tile.

    `state_after` is unused (kept for signature parity with the v1 variant and
    the wrapper's predicate protocol).
    """
    if int(action_i) != CLEAN_ACTION:
        return 0
    loc = state_before.agent_locs[agent_i]
    return int(cleanup_beam_hits_dirt(state_before.grid, loc))


def cleanup_event_from_state_global_delta(state_before, state_after, action_i) -> int:
    """v1 reference (FLAWED — confounded). Kept for the confound measurement."""
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
    """Compute Harvest:Open e_k(i) from a pair of SocialJax harvest states."""
    inv_before = int(jnp.sum(state_before.agent_invs[agent_i]))
    inv_after = int(jnp.sum(state_after.agent_invs[agent_i]))
    loc_after = state_after.agent_locs[agent_i]
    row, col = int(loc_after[0]), int(loc_after[1])
    apples_after = apples_in_radius(state_after.grid, row, col, radius=radius)
    return harvest_event(inv_before, inv_after, apples_after, threshold=threshold)


# ---------------------------------------------------------------------------
# JAX-native batched predicates: jit-/vmap-friendly. The env wrapper (Stage 3)
# uses these inside the JAX hot loop. Each returns shape (num_agents,) int8.
# ---------------------------------------------------------------------------


def cleanup_events_batch(state_before, state_after, actions: jnp.ndarray) -> jnp.ndarray:
    """Per-step, per-agent Cleanup e_k, v2 — beam-hit attribution.

    e_k(i) = 1 iff agent i fired zap_clean AND its beam covers >= 1 dirt tile in
    the pre-step grid. Not confounded by pollution spawning. `state_after` is
    unused (kept for wrapper signature compatibility). Returns (num_agents,) int8.
    """
    grid = state_before.grid
    locs = state_before.agent_locs
    used_clean = actions == CLEAN_ACTION

    beam_hits = jax.vmap(lambda loc: cleanup_beam_hits_dirt(grid, loc))(locs)
    return (used_clean & beam_hits).astype(jnp.int8)


def cleanup_events_batch_global_delta(
    state_before, state_after, actions: jnp.ndarray
) -> jnp.ndarray:
    """v1 batched predicate (FLAWED — confounded by pollution spawn; see
    docs/predicate.md). Kept only for the confound measurement vs v2."""
    used_clean = actions == CLEAN_ACTION
    dirt_before = jnp.sum(state_before.potential_dirt_and_dirt_label == DIRT_LABEL)
    dirt_after = jnp.sum(state_after.potential_dirt_and_dirt_label == DIRT_LABEL)
    return (used_clean & (dirt_after < dirt_before)).astype(jnp.int8)


def apples_in_radius_batch(
    grid: jnp.ndarray,
    locs: jnp.ndarray,
    radius: int = APPLE_NEIGHBOR_RADIUS,
) -> jnp.ndarray:
    """Vectorised apple count in Chebyshev radius around each agent.

    grid: (H, W) env grid
    locs: (num_agents, 2) or (num_agents, 3) — only the first two dims are used
    Returns: (num_agents,) int32
    """
    apple_mask = (grid == APPLE_LABEL).astype(jnp.int32)
    padded = jnp.pad(apple_mask, radius)
    patch_size = 2 * radius + 1
    locs_rc = locs[:, :2].astype(jnp.int32)

    def _count_at(loc):
        patch = jax.lax.dynamic_slice(padded, loc, (patch_size, patch_size))
        return jnp.sum(patch)

    return jax.vmap(_count_at)(locs_rc)


def harvest_events_batch(
    state_before,
    state_after,
    radius: int = APPLE_NEIGHBOR_RADIUS,
    threshold: int = 1,
) -> jnp.ndarray:
    """Per-step, per-agent Harvest:Open e_k. Returns shape (num_agents,) int8."""
    inv_before = jnp.sum(state_before.agent_invs, axis=-1)
    inv_after = jnp.sum(state_after.agent_invs, axis=-1)
    inv_grew = inv_after > inv_before
    apples_in_r = apples_in_radius_batch(state_after.grid, state_after.agent_locs, radius)
    return (inv_grew & (apples_in_r >= threshold)).astype(jnp.int8)
