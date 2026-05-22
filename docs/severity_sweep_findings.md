# Severity Sweep — Findings (2026-05-18 → 19)

> **⚠ Largely superseded (2026-05-21).** A later 1e8 re-run showed the
> experimental conclusions in this note — the dirtSpawn "cliff", the Phase-1
> cooperation cliff `S*∈(0.14,0.28)`, the oracle scout's "attribution helps" —
> are **3e7-horizon artifacts**: at 3e7 every dirt0.7 run was caught
> mid-takeoff. At the proper 1e8 horizon, dirt0.7's no-attribution baseline
> cooperates — there is no failing baseline. See `docs/premise_findings.md`.
> Only the machinery (`RewardExchangeWrapper`, `train_mappo_attribution.py`,
> the orchestrators) and the gate-check motivation remain accurate; the rest
> is kept as a record of the reasoning at the time.

Follows on from `gate_check_findings.md`. The gate-check was inconclusive
because default Cleanup is too easy a dilemma; this note records how we found
a genuinely hard one and put dilemma severity on a principled axis.

## Why

The gate-check verdict: oracle attribution ≈ no-attribution on default Cleanup
(cleaned_water 91.6 vs 90.3, return 945 vs 845 — within 2-seed noise). The
cause: the no-attribution baseline cooperates *on its own*, even at maximum
self-interest. With no cooperation deficit there is nothing for an attribution
signal to close — so the Stage-3 gate ("does attribution help?") cannot be
asked meaningfully.

Two things were needed:

1. A Cleanup regime where the no-attribution baseline **genuinely fails**.
2. Dilemma severity expressed on **Willis et al.'s self-interest axis** (the
   brief's chosen severity coordinate), not raw environment parameters.

## Track A — hardening the environment

### Calibration

A scale check came first. The gate-check ran at 1e8 timesteps; a coarse sweep
wants something cheaper.

- **1e7 is too short.** At 1e7 the `dirt0.5` control — an env *known* to be
  solvable — still showed ~0 return. Undertrained, so "low cooperation" could
  not be told apart from "hard dilemma."
- Reading the gate-check's full 1e8 curve, training **takes off between 10M
  and 25M** timesteps (return 0 → 611 by 25M, plateau ~800 by 50M).
- **3e7 is the working scale** — safely past the takeoff, ~40–45 min / ~$1.4
  per A100 run. Publication-quality 1e8 was overkill for a coarse sweep.

### The dirtSpawnProbability sweep (3e7, 2 seeds)

`dirtSpawnProbability` (pollution spawn rate) is the most direct dilemma dial.
Final values, both seeds agreeing:

| dirtSpawn | cleaned_water | episode return | verdict |
|---|---|---|---|
| 0.5 (default) | ~85 | ~550 | baseline trains & cooperates |
| 0.7 | ~2 | ~0 | baseline fails |
| 0.9 | ~1 | ~0 | baseline fails |

A **sharp cliff between 0.5 and 0.7**. The `dirt0.5` control trained at the
*same* 3e7 budget, which rules out undertraining — the failure at 0.7/0.9 is
the environment, not the compute. **`dirt0.7` is the hardened env** (0.9's
harder failure risks being *impossible* rather than merely *hard*).

Orchestration: `experiments/modal_severity_sweep.py`.

## The self-interest machinery (B1 / B2)

To dial severity on the self-interest axis:

- **`RewardExchangeWrapper`** (`src/zkattribution/wrappers.py`) — the Willis
  reward-exchange map `R'_i = S·r_i + (1−S)·mean_{j≠i} r_j`. `S=1` is
  individual rewards (identity); `S=1/num_agents` is fully utilitarian (every
  agent receives the mean). Total-conserving. `S` *is* the self-interest level.
- **`experiments/train_mappo_attribution.py`** — trains the no-attribution and
  oracle regimes (an `ATTRIBUTION` flag) at any `S`.
- **`experiments/modal_self_interest_sweep.py`** — the Phase 1 / Phase 2
  orchestrator.

## Phase 1 — the s-sweep

The no-attribution baseline on `dirt0.7`, sweeping `S`, 1 seed, 3e7. Final
values:

| S (self-interest) | cleaned_water | episode return | verdict |
|---|---|---|---|
| 1.00 | ~2 | ~0 | fails |
| 0.85 | 0.8 | 0 | fails |
| 0.70 | 14 | ~1 | fails |
| 0.55 | 11 | 0 | fails |
| 0.40 | 5 | 0 | fails |
| 0.28 | 15 | ~1 | fails |
| **0.14** | **101** | **1044** | **cooperates** |

### Verdict 1 — `dirt0.7` is a genuine social dilemma

At `S=0.14` (≈1/7, near-full reward-sharing) the baseline trains fully —
return ~1044, river kept clean. **Cooperation on `dirt0.7` is achievable.** The
failures at higher `S` are therefore rational defection under a real incentive
conflict, not a physically impossible task. `dirt0.7` is a textbook sequential
social dilemma — the right environment for the Stage-3 experiment.

### Verdict 2 — the cooperation cliff `S* ∈ (0.14, 0.28)`

The baseline fails at every `S ≥ 0.28` (return ≤ ~1, negligible) and recovers
only at `S=0.14`. `dirt0.7` is a *harsh* dilemma — agents need nearly full
reward-sharing before cooperation is individually rational. This `S*` range
overlaps Willis et al.'s estimate for Melting Pot Clean Up (`s*≈0.25–0.29`),
an external check that `dirt0.7` sits at a credible dilemma severity.

Caveat: 1 seed, so the cliff is *bracketed* to `(0.14, 0.28)`, not pinned.

## The oracle scout — the Stage-3 signal

The same s-sweep, now for the **oracle** regime — oracle attribution on
`dirt0.7`, the same 6-value S-grid, 1 seed, 3e7 — so the oracle cooperation
curve overlays the Phase 1 no-attribution curve directly. Final episode return:

| S | no-attribution | oracle |
|---|---|---|
| 0.85 | fail · r≈0 | fail · r≈4 |
| 0.70 | fail · r≈1 | fail · r≈5 |
| 0.55 | fail · r≈0 | **coop · r≈1298** |
| 0.40 | fail · r≈0 | fail · r≈2 |
| 0.28 | fail · r≈1 | **coop · r≈1402** |
| 0.14 | coop · r≈1044 | coop · r≈857 |

### Read 1 — oracle attribution helps

Oracle cooperates strongly at **S=0.55 and S=0.28** — return ~1300–1400, river
cleaned — two severities where no-attribution categorically failed. Across the
whole failing region [0.28, 0.85], no-attribution cooperated 0/5; oracle
cooperated 2/5, and hard. Attribution demonstrably opens cooperation where the
baseline has none. Unlike the gate-check (oracle ≈ no-attribution on default
Cleanup), this is a real difference — the **Stage-3 gate passes**.

### Read 2 — the outcome is bimodal

The oracle curve is non-monotonic: cooperate at 0.55, fail at 0.40, cooperate
at 0.28. `S=0.40` is *less* selfish than `S=0.55`, so this is not a cliff — it
is **bimodality**. Under oracle attribution the mid-S region has two basins, a
cooperative one and a collapsed one, and which one a run reaches is
seed-dependent. The takeoffs are genuine (return climbs cleanly past 1300, not
noise) and the collapses are genuine (~0); 1 seed per S is one draw per S, so
the jagged pattern is the bimodality signature, not a bug.

Consequence: the multi-seed confirmation must **estimate P(cooperate | S)** for
each regime — the mid-S outcome is a coin-flip, so ≥5 seeds per cell are needed
to estimate a probability rather than denoise a point estimate.

### Caveat — the S=0.14 control

Both regimes cooperate at `S=0.14`, but oracle (r≈857) came in below
no-attribution (r≈1044) — likely 1-seed variance plus oracle's run still
climbing at 30M (not converged). The confirmation should check oracle is not
paying a real cost where the baseline already cooperates.

## Status & next

The Stage-3 gate has passed qualitatively: oracle attribution produces
cooperation where the no-attribution baseline cannot. What remains is to
quantify it.

- **The confirmation (`confirm` mode):** both regimes, 5 seeds, at
  `S ∈ {0.70, 0.55, 0.40, 0.28, 0.14}`, 3e7. Output: P(cooperate) vs S for each
  regime — the Stage-3 figure, with the bimodality characterised rather than
  hidden.
- **The confirmation is the Stage-3 verdict.** If oracle's P(cooperate) curve
  sits clearly above no-attribution's, the cryptographic layer (Stage 5) is
  motivated.
- **Then Stage 4** — the two-phase lying experiment (pretrain under oracle to
  establish reciprocity, then fork a self-reported continuation so the claim
  head has a payoff-relevant gradient), forking off the confirmation's oracle
  checkpoints.

Out of scope here: the cryptographic layer (Stages 5–7).
