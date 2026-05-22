# Premise Findings — the Missing Cooperation Deficit (2026-05-21)

Follows on from `gate_check_findings.md` and `severity_sweep_findings.md`. This
note records two linked discoveries that put the project's core hypothesis
premise in question.

## Background

H1 (the brief): *verified attribution improves cooperation over
no-attribution.* For H1 to be testable, the no-attribution baseline must
genuinely **fail** to cooperate — a cooperation deficit for attribution to
close. The gate-check showed default Cleanup's baseline cooperates unaided.
Track A hardened the env (`dirtSpawnProbability` 0.5 → 0.7), which *appeared*
to produce a failing baseline; the whole s-sweep (`severity_sweep_findings.md`)
was built on that.

## Discovery 1 — the 3e7 horizon confound

The 50-run `confirm` (5 S-values × {no-attribution, oracle} × 5 seeds, 3e7
timesteps) came back non-monotonic — cooperation jumping around with S in a way
no severity ordering explains.

Trajectory diagnosis settled it. Sampling 8 runs' training curves (episode
return at 15M / 22.5M / 30M timesteps):

| run | 15M | 22.5M | 30M |
|---|---|---|---|
| oracle S=0.40 seed0 | 8 | 843 | 1435 |
| no-attr S=0.28 seed1 | 0 | 11 | 1410 |
| no-attr S=0.40 seed4 | 0 | 0 | 447 |

**Every run is flat near zero until ~15–25M, then takes off in a steep climb
that is still rising at the 30M cutoff.** The smear of final returns (0 → 1435)
is a smear of *takeoff timing*, not of outcomes — a run's value at 30M just
records how early it happened to take off. The "return > 100 = cooperated" cut
was really measuring *"did this run take off before the cutoff."*

**3e7 is too short on dirt0.7.** Phase 1 and the oracle scout ran at the same
3e7 — their cooperation "cliffs" (`S*∈(0.14,0.28)`, the scout's "oracle helps")
are the same artifact. The experimental conclusions of
`severity_sweep_findings.md` are superseded.

Root cause: the Track-A calibration validated 3e7 on `dirt0.5` — an easy
control that plateaued within budget. `dirt0.7` is harder, so its takeoff is
later, and 3e7 catches it mid-flight. The "hard dilemma vs not-yet-trained"
confound was ruled out for the easy cell but not the hard one.

## Discovery 2 — dirt0.7 has no failing baseline (the 1e8 calibration)

We re-ran at **1e8** — SocialJax's own default training horizon for MAPPO
Cleanup (`mappo_cnn_cleanup.yaml`). Calibration: 20 runs, both regimes ×
{S=0.70, 0.40} × 5 seeds.

| regime | S=0.70 | S=0.40 |
|---|---|---|
| **no-attribution** | **4/5 cooperate** (return ≈ 1200–1400) | **5/5** (return ≈ 1700–2000) |
| oracle | 5/5 | 5/5 |

**The no-attribution baseline cooperates at 1e8** — 9 of 10 runs reach full
cooperation. dirt0.7's "failure" was the same horizon artifact: at the proper
horizon, plain MAPPO solves cooperation on dirt0.7.

(1e8 is an adequate horizon — 13/20 runs cleanly plateaued — though a
takeoff-timing tail extends past 75M for a minority.)

## The premise problem

The cooperation deficit has now dissolved **three times**:

1. **Gate-check** — default Cleanup's no-attribution baseline cooperates.
2. **Track A** — `dirt0.7` "fails" — but at 3e7, a confounded horizon.
3. **1e8 calibration** — `dirt0.7`'s baseline cooperates at the proper horizon.

The pattern is consistent: **Cleanup + MAPPO, given a fair training budget,
solves cooperation.** H1's premise — *no-attribution fails, verified
attribution rescues it* — does not hold for this environment + algorithm. It is
a **premise** problem, not a parameter: the env was already hardened, and
S=0.70 (near the maximally-selfish end) still cooperates 4/5.

## Options

- **(A) Much harsher env** — `dirtSpawnProbability` 0.9+, or tighten
  `thresholdDepletion`. May create a 1e8-persistent failure, or may also just
  cooperate later. ~$30–60 of cloud to find out.
- **(B) Reframe the metric** — attribution's effect on the *speed and
  reliability* of cooperation emergence, not the final rate. The 1e8
  calibration weakly hints at this (oracle 10/10 vs no-attr 9/10; a real
  takeoff-timing tail). A weaker, different claim than H1 as written.
- **(C) Weaker learner — IPPO instead of MAPPO.** Independent learners are the
  classic setting where cooperation *fails* in sequential social dilemmas;
  MAPPO's centralised critic is itself a coordination aid. SocialJax ships
  `ippo_cnn_cleanup.py`. Deviates from the brief's MAPPO spec.
- **(D) Reconsider H1 / the environment** — including Harvest:Open, set aside
  earlier but a sharper free-riding dilemma by design.

## Status

Per the brief's "investigate before crypto" gate, the project is **paused for a
strategic decision**. No further cloud spend until the path is chosen. The
immediate, free next step is a literature + SocialJax-paper check on
MAPPO-vs-IPPO cooperation on Cleanup, which directly informs option (C).

What stands regardless: the machinery — `RewardExchangeWrapper`,
`train_mappo_attribution.py`, the Modal orchestrators, `summarize_s_sweep.py` —
works; the training horizon is now understood (1e8, not 3e7); and Cleanup +
MAPPO is characterised precisely. The setback is in the experimental premise,
not the tooling.
