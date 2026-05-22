# Territory — Scouting Brief

A scoped investigation to decide whether **Territory** (SocialJax) should become a
third zkAttribution environment alongside Cleanup and Harvest:Open. Follows on
from `severity_sweep_findings.md`. **This is a go/no-go scout, not a full build.**

## The decision this brief resolves

Is Territory **Type (a)** — cooperation is achievable but unreachable by vanilla
IPPO — or **Type (b)** — cooperation is not achievable at all?

- If **Type (a)**: attribution can supply the missing signal and Territory becomes
  potentially our *strongest* environment (see "Why Territory" below). Build it out.
- If **Type (b)**: no information signal can create an equilibrium that doesn't
  exist. Territory becomes a "future work" mention and we ship with Cleanup +
  Harvest.

**Do not commit predicate/circuit/training work to Territory until this
go/no-go is settled.** The whole brief exists to answer it cheaply.

## Why Territory is worth scouting

The SocialJax paper reports that IPPO **fails to cooperate in Territory** — common-
reward IPPO does *worse* than individual-reward IPPO. Critically, the authors
attribute this to **credit assignment**, not to the absence of a cooperative
optimum:

> "Due to the difficulties of credit assignment in Territory, IPPO with common
> rewards fails to learn effective policies... We believe that this environment
> should be a social dilemma... but more advanced algorithms or training
> methodologies are required to find effective cooperative policies."

This is the key asymmetry vs. Cleanup:

- **Cleanup's** failure mode is *incentive* (free-riding). Reward-sharing fixes it
  (our S-sweep: cooperation emerges at S=0.14). Attribution's role is to *enable
  reciprocity* — an indirect, learning-dynamics argument.
- **Territory's** failure mode is *credit assignment*. Reward-sharing makes it
  *worse*. Attribution's role is to *directly supply the per-agent credit signal
  that common-reward training lacks* — a direct causal argument.

If attribution unlocks Territory cooperation, the result is **categorical**
("attribution enables cooperation that standard MARL cannot achieve") rather than
marginal ("attribution shifts the cliff by X%"), and it directly answers the open
question the benchmark's own authors posed. That's a stronger FAccT story than
Cleanup, and a cleaner causal claim.

## Exact Territory mechanics (from the SocialJax paper)

- Players paint walls their own color (claim) by touching or flinging paint.
- Paint **dries after 25 steps**, then the wall brightens and pays reward **every
  subsequent timestep**. More dried walls held = more reward.
- Walls can be **zapped twice to permanently destroy** them → become non-claimable
  rubble (deadweight loss for everyone). This is the social cost / "tragedy."
- Players can zap each other; a player is **permanently removed after two hits**.
- **9 players**, competitive, uneven resource distribution, 11×11 partial-obs
  window (grid-level, like the other SocialJax envs).

Verify all of the above against `external/SocialJax/socialjax/environments/territory/`
before relying on it — especially the exact action set (paint vs. zap vs. move),
the destruction counter mechanic, and the player-removal mechanic.

## Validation plan — staged, cheapest first

### Step 1 — Scripted-cooperator check (hours, ~no compute)

The definitive Type-(a) vs Type-(b) test. **Do this first.**

- Hand-code 9 scripted agents that each claim a region, let paint dry, **never
  zap**, and hold territory peacefully (a hard-coded peaceful partition).
- Measure collective return over an episode.
- Compare against the SocialJax IPPO individual-reward return (the published
  baseline).

**Decision:**
- If scripted-coop return **>> IPPO return** → a cooperative equilibrium exists and
  is simply unreachable by learning → **Type (a)** → proceed to Step 2.
- If scripted-coop return **≈ or < IPPO return** → cooperation may not be a genuine
  improvement → likely **Type (b)** → stop, write up as future work.

This step needs no training and settles the core question. It also gives you the
"cooperative ceiling" return number you'll compare everything else against.

### Step 2 — Oracle-attribution training run (does double duty)

Only if Step 1 confirms Type (a).

- Reuse `AttributionWrapper` (the Stage-3 oracle mechanism) with a Territory
  predicate (see below). Inject the true per-agent α into observations.
- Train IPPO + oracle attribution in Territory at the calibrated budget (start at
  3e7, the scale that worked for Cleanup; confirm takeoff timing for Territory
  separately — it may differ).
- Compare cooperation/return against the no-attribution IPPO baseline.

**Decision:**
- If oracle reaches cooperation where baseline IPPO fails → this *is* the Territory
  Stage-3 result: it proves the basin is reachable with credit info AND that
  attribution is the mechanism. Build out Territory (predicate finalization →
  self-reported → crypto).
- If oracle also fails → ambiguous (Type b, or attribution signal too coarse).
  Before abandoning, try a higher-resolution attribution signal (normalized
  territory-held rather than binary event-rate) — credit assignment in a 9-player
  destructive game may need more than a scalar event rate. If richer signal still
  fails, shelve Territory.

## Predicate (use only after Step 1 passes)

The signal should be **contribution-based** (addresses credit assignment), not
purely restraint-based. Recommended:

### Option D — productive non-aggression (recommended)

```
e_k(i) = 1  iff  agent i painted a neutral or own wall at step k
                 (productive claim, NOT overwriting a peer's wall)
                 AND did not zap a wall or a player
         = 0  iff  agent i zapped (wall or player),
                 OR painted over another agent's wall (contested/aggressive),
                 OR was idle
```

- **α represents:** rate of productive, non-destructive territory claiming.
- **Why it fits Territory:** it's a *contribution* signal (credit assignment) that
  also penalizes destructive defection (zapping). Symmetric with Cleanup's "active
  beneficial action" and Harvest's "active sustainable harvest."
- **Circuit-encodability:** moderate. Checks (a) action is a paint action, (b) beam
  landed on a neutral/own wall (readable from local obs), (c) action is not a zap.
  Comparable to Cleanup's beam predicate plus a colour check.
- **Privacy:** strong. α hides which walls (the agent's territory map), movement,
  and zap targets — exactly the strategic info a competitor would exploit.

### Option B1 — pure restraint (fallback, simplest circuit)

```
e_k(i) = 1  iff  agent i did not fire a zap beam at step k
```

- Trivial circuit (action-only check). Captures the destruction/aggression axis.
- **But** it's a "what you didn't do" signal — doesn't directly address credit
  assignment, so it's a weaker fit with the Territory framing. Use only if Option
  D's circuit complexity or sparsity (below) proves problematic.

## Known risks (all real for Territory specifically)

1. **Steady-state sparsity (Option D).** Paint doesn't decay — once dried, a wall
   pays forever until destroyed. In a perfectly peaceful partition, painting
   activity → 0 after the land-grab, so α collapses. *Mitigation:* with 9 players +
   uneven resources there's likely ongoing contest/re-painting. **Measure the
   painting rate in a cooperative run before trusting Option D.** If it's too
   sparse, consider normalized territory-held as the signal instead.
2. **Cooperative basin may not exist (Type b).** This is what Step 1 rules out.
   Don't skip Step 1.
3. **9 agents is expensive.** α vector length 9, 9 proofs/window, 9-channel obs
   augmentation, larger privacy-attack surface. Roughly 2× Cleanup's crypto cost.
   Relevant for Stage 5+, not for the scout.
4. **Player elimination edge case.** Players are permanently removed after two
   hits. Define what α means for a removed agent (drop from the attribution vector
   after removal, or use a fixed sentinel). Complicates windowing + per-agent
   indexing in the wrapper and (later) the circuit.

## How this fits the existing pipeline

- `AttributionWrapper` (Stage 3) is env-agnostic — it takes an injected predicate.
  Reuse it; just supply a Territory predicate function.
- `predicate.py` already houses Cleanup + Harvest predicates. Add a Territory
  predicate (JAX-native, vmapped, returns `(num_agents,)` of {0,1}), unit-tested
  against hand-constructed Territory states.
- `RewardExchangeWrapper` may be useful for Step 2 severity tuning, but note
  Territory's twist: individual rewards (S=1) *outperform* common rewards here, so
  the reward-exchange knob behaves differently than in Cleanup. Don't assume the
  Cleanup intuition transfers.
- Modal orchestration (`modal_*_sweep.py`) + the `RUN_COMPLETE` sentinel pattern
  carry over directly.

## Success / failure criteria

- **Step 1 pass:** scripted peaceful-partition collective return clearly exceeds
  IPPO individual-reward return → Type (a) confirmed → proceed.
- **Step 2 pass:** oracle attribution produces cooperation (peaceful partition,
  high collective return, low resource-destruction rate) where no-attribution IPPO
  does not → Territory is a flagship environment → build out.
- **Either step fails:** document cleanly, shelve Territory as future work, ship
  with Cleanup + Harvest. A clean negative ("we tested Territory; cooperation is
  not reachable / attribution is insufficient at this signal resolution") is itself
  a reportable finding, not a wasted effort.

## Sequencing

Run this **after** the Cleanup confirmation (5-seed P(cooperate) curve) and Harvest
calibration are locked down. Territory is a high-upside side-quest, not a
replacement for either. Step 1 is cheap enough to slot in opportunistically; only
commit to Step 2 and beyond if Step 1 says Type (a).
