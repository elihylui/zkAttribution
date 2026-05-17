# Predicate Specification (Stage 2)

The cooperative-event predicate `e_k(i)` returns 1 if agent `i` performs a cooperative event at step `k`, else 0. Per-agent cooperation rate over a window of `N` steps is

```
α(i) = (1 / N) · Σ_k e_k(i)
```

This document defines `e_k` for both target environments. Implementation: [`src/zkattribution/predicate.py`](../src/zkattribution/predicate.py). Tests: [`tests/test_predicate.py`](../tests/test_predicate.py).

## Cleanup

**Definition (v2 — per-agent beam attribution):**

```
e_k(i) = 1   iff   action_i == zap_clean   AND   agent i's beam covers ≥1 dirt tile (pre-step)
```

Where:

- `zap_clean` is `Actions.zap_clean = 8` in [`external/SocialJax/socialjax/environments/cleanup/clean_up.py`](../external/SocialJax/socialjax/environments/cleanup/clean_up.py).
- An agent's **beam footprint** is the 4 tiles its `zap_clean` covers — one-step-forward, two-step-forward, forward-right, forward-left — mirrored from `clean_up.py`'s `_interact` (STEP table + target geometry). Right/left collapse to one-step-forward when out of bounds; off-grid tiles are masked out.
- "Beam covers dirt" is equivalent to "this agent cleans a tile": Cleanup's clean mechanic converts *any* in-beam dirt tile (`grid == Items.dirt = 8`) to clean. So beam-hit is a faithful, per-agent "successful clean".

**Why v2 replaced v1.** The v1 predicate was `action_i == zap_clean AND total dirt count strictly decreased`. The total dirt count is a *global* quantity, and Cleanup spawns pollution every step (`dirtSpawnProbability = 0.5`) — so an agent can successfully clean a tile yet see `dirt_after ≥ dirt_before` because dirt spawned elsewhere. A 2e5-timestep diagnostic confirmed this: logged α peaked at **0.033** (signal in the noise floor). v1 is kept in code as `cleanup_events_batch_global_delta` / `cleanup_event_from_state_global_delta` *only* so the confound can be measured; it is not the live predicate.

**Known limitation (v2):** if two agents' beams overlap the same dirt tile, both are credited `e_k = 1`. Acceptable — both contributed a zap that would clean it.

## Harvest:Open

**Definition (v1 — pragmatic):**

```
e_k(i) = 1   iff   inv_sum_after_i > inv_sum_before_i
              AND  apples_in_radius_2(agent_i_loc_after) ≥ threshold       (threshold default = 1)
```

Where:

- "Inventory grew" detects passive apple collection — Harvest:Open has no explicit harvest action; collection happens when an agent steps onto a tile with an apple.
- `apples_in_radius_2(loc)` counts apples in the Chebyshev radius-2 neighborhood of `loc` in `state.apples` after the step.
- Radius 2 matches SocialJax's regrowth neighborhood.

**Known limitation:** the env's regrowth is *probabilistic*, scaling with neighborhood count — there is no hard "threshold" baked into the env. Our `threshold = 1` is the weakest sustainability condition (at least one apple remains nearby). v2 could tighten to `threshold = 2` or weight by neighborhood count to better approximate "regrowth-positive harvest."

## Cooperation rate

```python
def cooperation_rate(events: list[int]) -> float:
    return sum(events) / len(events)
```

Computed per-agent over the window of `N` steps (default `N = 100`, parameterised per the brief).

## Convention

All `e_k` definitions in this spec use **successful** events, not attempted, per the brief's gotcha note ("Use successful (it's harder to inflate)").
