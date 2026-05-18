# Gate-Check Findings — Stage 3/4 Cloud Run (2026-05-18)

A retrospective record of the first publication-scale (1e8-timestep) cloud run.
Read this before deciding what to run next.

## Why we ran it

The 7-stage plan is sequenced to **fail fast**: if oracle attribution (Stage 3)
does not improve cooperation, the cryptographic layer (Stage 5) is moot. CPU dev
runs at 1e6 timesteps were 100× below published scale and inconclusive by
construction — agents stay undertrained and near-random, so the measured
cooperation rate (α ≈ 0.03) is just arithmetic, not signal. So we rented GPUs to
run the three non-crypto regimes at full scale and answer two questions:

- **Stage 3 gate** — does oracle attribution beat no-attribution on cooperation?
- **Stage 4** — do agents inflate self-reported α claims?

## Setup

| | |
|---|---|
| Env | SocialJax Cleanup |
| Algorithm | MAPPO (CTDE), `NUM_ENVS=64` |
| Rewards | **individual** (`ENV_KWARGS.shared_rewards=False`) — shared rewards mute the dilemma |
| Scale | 1e8 timesteps/run |
| Design | 3 regimes (no-attribution / oracle / self-reported) × 2 seeds = 6 runs |
| Hardware | Modal, A100-40GB, ~83 min/run |
| Cost | ~$26 of $30 free Modal credits (~$4 remaining) |
| Orchestration | `experiments/modal_gate_check.py`; parsed via `scripts/parse_wandb_run.py` |

Operational note: A10G workers preempted long runs mid-training; A100 runs are
shorter and more reliable. A `RUN_COMPLETE` sentinel guards the skip-logic so a
preempted partial run re-trains rather than being falsely treated as done.

## Results

| Regime | `cleaned_water` | Episode return | Notes |
|---|---|---|---|
| no-attribution | 90.3 | 845 | baseline |
| oracle | 91.6 | 945 | ≈ baseline, within 2-seed noise |
| self-reported | 12.4 | ~0 | river effectively never cleaned |

Self-reported also logged `claimed_alpha` ≈ 0.50 (flat) and `true_alpha` ≈ 0.01.

## Verdict 1 — Stage 3 gate NOT cleanly shown

Oracle ≈ no-attribution. The 91.6-vs-90.3 / 945-vs-845 gaps sit within 2-seed
noise, so H1 ("verified attribution improves cooperation") is **not
demonstrated**. This is the gate result that blocks crypto.

Two non-exclusive suspects:

1. **Dilemma too mild** at Cleanup's default severity — if free-riding is barely
   tempting, there is little cooperation deficit for attribution to close.
2. **Weak α signal** — the oracle's true α sat around ~0.04 with low variance.
   If the signal an agent receives barely moves, the policy has nothing to
   condition on, regardless of whether the channel is honest.

## Verdict 2 — the self-reported "collapse" was an artifact, not a result

The headline-looking result — self-reported cooperation collapsing to ~0 while
the other regimes reach ~90 — is tempting to read as "unverified claims destroy
cooperation." **It is not that.** Scrutiny traced it to a training artifact:

- The env-action policy collapsed early and never recovered (return ~0).
- `claimed_alpha` was dead-flat at the uniform-prior mean (0.50) — the claim
  head never learned anything.
- The oracle regime uses the **same 26-channel augmented observation** but a
  **single-head** policy, and trained fine.

That isolates the cause to the **two-head architecture**, not the information
channel. The Stage-4 Actor has a second Categorical "claim" head sharing the CNN
trunk; an unlearnable head backpropagating into a shared trunk injects gradient
noise that degrades the env-action policy (negative transfer / gradient
interference).

**Fix (commit `411eaa9`):** `jax.lax.stop_gradient` on the trunk features
feeding the claim head. The claim head still reads the shared representation but
cannot backprop into the trunk. Smoke-tested on CPU; not yet re-verified at 1e8
scale.

## Open problems

1. **Weak oracle α signal.** Either the predicate produces too little
   discriminable variation, or the default severity is too mild. Candidate
   responses: harsher severity (Willis et al. self-interest level), longer or
   more windows, more seeds to resolve the noise, or a predicate rethink.
2. **The claim head has no learning signal (Stage-4 "issue 2").** The
   stop-gradient fix un-breaks the regime but does not make *lying* measurable.
   Claiming barely affects an agent's reward, so there is no gradient pressure
   for the claim head to learn — honestly or strategically. Studying lying needs
   claiming to be **incentivised**: peers' policies must condition on claims in a
   payoff-relevant way. This is a MARL design problem, not a bug.

## Status against the brief

Per the Stage-3 gate ("oracle must improve cooperation over Stage 1 — if not,
stop and investigate before adding crypto"), status = **investigate before
crypto**. Stage 5 is **not** started.

## What's next (not yet decided)

Both open problems are **design** problems, not compute problems — re-running
today would only re-confirm "oracle ≈ no-attribution" and "claim head flat". The
remaining ~$4 of credits cannot answer either question. Recommended order:

1. Resolve the oracle signal: choose a severity / window / seed-count change and
   justify it before spending on another sweep.
2. Design a claim incentive for Stage 4 so lying can actually emerge.
3. Only then cost and scope the next cloud sweep.
