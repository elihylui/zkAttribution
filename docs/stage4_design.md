# Stage 4 Design: Self-Reported Attribution Baseline

## Goal

Per the brief: each agent broadcasts a **self-claimed α** — not the oracle's true α. No verification. Peers receive the claim in their next observation. Measure whether agents **learn to lie** (claimed α vs true α) and whether that lying corrupts cooperation.

This is the baseline that motivates zkAttribution: if unverified self-reports get gamed by rational agents, that is exactly the failure the cryptographic verification (Stage 5) is designed to prevent.

## Relationship to Stage 3

| | Stage 3 — oracle | Stage 4 — self-reported |
|---|---|---|
| α source | simulator computes true α from global state | each agent emits its own claimed α |
| verification | n/a (trusted) | none |
| can agents lie? | no | **yes** — that's the point |
| obs augmentation | per-peer true-α vector | per-peer *claimed*-α vector |
| policy change | none | **new "claim" output head** |

The env-side augmentation mechanism is the same channel-concat as `AttributionWrapper`. The new piece is that the α now comes from a *learned policy output*, so the policy itself must change.

## Components

### 1. The claim — a new policy output

Each agent's policy gains a **claim** output: a value in `[0, 1]` it reports as its cooperation rate. It must be a *learned action* — that is the only way "learning to lie" can emerge.

**Design decision — claim representation:**

- **Option A — discrete-bucketed (recommended for v1).** The claim is a second `Categorical` head over `K` buckets, e.g. `K = 11` → {0.0, 0.1, …, 1.0}. Reuses all existing discrete-MAPPO machinery (Categorical log-prob, the PPO loss). Minimal surgery. 11-level resolution is ample to detect inflation.
- **Option B — continuous (Beta head).** The claim is sampled from a `Beta(a, b)` distribution on `[0, 1]`; the Actor outputs `(a, b)`. Faithful to the brief's "continuous in [0, 1]", but adds a continuous log-prob term to the MAPPO loss — more surgery, more failure modes (NaN gradients, entropy tuning).

**Recommendation: A for v1.** It answers the scientific question ("do agents inflate their claims?") with far less risk. Continuous is a later refinement if 11 buckets prove too coarse.

### 2. `SelfReportWrapper`

Analogous to `AttributionWrapper`. Each step:

- Receives `(discrete_action, claim)` per agent.
- At window boundaries (every `N` steps) snapshots each agent's most recent claim → the **broadcast claim vector**.
- Augments each agent's obs with the broadcast claim vector (peers see claims) — same channels-last concat as Stage 3.
- **Also computes the true α** via the Stage-2 v2 predicate — for **logging only**. Agents never observe true α.
- Logs `claimed_alpha`, `true_alpha`, and `inflation = claimed − true`.

### 3. Training script

`experiments/train_mappo_self_reported.py` — copy of the MAPPO script with:

- `Actor` extended with the claim head.
- Action sampling produces `(discrete, claim)`.
- The PPO loss sums both log-probs against the shared advantage.
- `SelfReportWrapper` inserted in the env stack.

## Stage 4 exit criterion (the brief)

Quantitative measurement of whether agents learn to lie and whether lying corrupts cooperation. Logged / derived:

- `claimed_alpha` (mean), `true_alpha` (mean), `inflation` (claimed − true).
- Post-hoc: correlation(claimed, true); inflation rate over training.
- Comparison vs the oracle regime — does an unverifiable channel degrade cooperation relative to a trusted one?

## Decisions

1. **Claim representation:** discrete-bucketed (A, recommended) vs continuous Beta (B). ← needs a call before 4c.
2. **Claim timing:** the agent emits a claim every step; the wrapper broadcasts the window-end claim. Simple, recommended.
3. **Reward unchanged** — Stage 4, like Stage 3, is an information-layer intervention only.

## Test plan

- `SelfReportWrapper` unit tests: claim snapshot at window boundary, obs augmentation, true-α computed and logged but absent from agent obs.
- Policy: `Actor` emits both heads; sampled action has the right structure; log-prob sums correctly.
- e2e smoke: wrap real Cleanup, run a short rollout, verify claimed + true α both logged.

## Out of scope

- Verification / crypto (Stage 5).
- The adversarial forgery study (Stage 6) — though Stage 4's "do agents lie" measurement is its conceptual precursor.
