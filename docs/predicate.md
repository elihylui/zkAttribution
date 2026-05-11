# Predicate Specification (Stage 2)

The cooperative-event predicate `e_k(i)` returns 1 if agent `i` performs a cooperative event at step `k`, else 0. Per-agent cooperation rate over a window of `N` steps is

```
α(i) = (1 / N) · Σ_k e_k(i)
```

This document defines `e_k` for both target environments. Implementation: [`src/zkattribution/predicate.py`](../src/zkattribution/predicate.py). Tests: [`tests/test_predicate.py`](../tests/test_predicate.py).

## Cleanup

**Definition (v1 — pragmatic):**

```
e_k(i) = 1   iff   action_i == zap_clean   AND   dirt_count_after < dirt_count_before
```

Where:

- `zap_clean` is `Actions.zap_clean = 8` in [`external/SocialJax/socialjax/environments/cleanup/clean_up.py`](../external/SocialJax/socialjax/environments/cleanup/clean_up.py).
- `dirt_count = Σ [state.potential_dirt_and_dirt_label == Items.dirt]`, where `Items.dirt = 8`.

**Known limitation:** if multiple agents fire `zap_clean` in the same step and the global dirt count decreases, all such agents receive `e_k = 1` — even though only one (or some) actually cleaned. Strict per-agent attribution would require reading the beam-hit logic in `clean_up.py` (around lines 1056–1061) or adding a per-agent `cleaned_count[i]` field to `info`. Deferred to v2.

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
