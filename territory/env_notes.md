# Territory — Environment Notes (Phase 0, verified 2026-05-22)

Verification of `external/SocialJax/socialjax/environments/territory/territory_env.py`
(class `Territory_open`) and the IPPO config, done so the predicate, the
scripted-cooperator check, and the experiment are grounded in SocialJax's
actual code — not the Melting Pot paper's description. Corrects two points in
`territory_scout.md`.

## Environment

- Class `Territory_open`, env name `territory_open`. **9 agents**, **23×39
  grid**, **11×11** partial observation, **1000-step** episodes.
- **Actions** (9): `turn_left, turn_right, left, right, up, down, stay,
  zap_forward, claim`.
  - `claim` (action 8) — paint/claim the resource cell the agent faces (within
    interaction range).
  - `zap_forward` (action 7) — the zap beam.
- **Resources** — "R" cells in the ASCII map, scattered in uneven clusters.
  Start `resource_unclaimed`; a `claim` sets the grid cell to the claiming
  agent's colour value (`1000 + agent_id`).
- **Reward.** A claimed cell pays its owner **0.01 per step — but only once it
  has been held for ≥ 100 consecutive steps** (`claimed_indicator_time_matrix
  >= 100`). The hold-counter resets to 0 whenever the cell's owner changes.
  Per-agent per-step reward = `0.01 × (number of "dried" cells you own)`.
- **Reward modes** (`ENV_KWARGS.shared_rewards`):
  - `True` — common: every agent receives the **mean** of all agents' rewards.
  - `False` — individual: each agent receives its own dried-cells reward.
  - (inequity-aversion / SVO shaping branches also exist; unused here.)
- **Zap.** `zap_forward` freezes a hit agent (`freeze` penalty) and then
  **respawns** it (`reborn_locs`) — the same freeze+respawn mechanic as
  Cleanup. Players are **not** permanently removed.

## Corrections to `territory_scout.md`

The scout brief described Melting Pot's Territory; SocialJax's implementation
differs in two ways that matter:

1. **Dry threshold is 100 steps, not 25.** A wall must be held 100 consecutive
   steps before it pays anything (the `>= 25` line is commented out in favour
   of `>= 100`). That is 10% of a 1000-step episode per wall — claiming has a
   real lead time, and contested re-painting is very costly (any owner change
   resets the 100-step counter to 0).
2. **No permanent elimination.** Zapped players freeze, then respawn — they do
   not leave the game. The scout's "player-elimination edge case" (risk 4)
   therefore does **not** arise: `num_agents` is constant, so
   `AttributionWrapper`'s fixed-length α vector needs no special handling.

Still unverified (does not block the predicate or the experiment): whether
zapping a *wall* destroys it (scout's "zap ×2 → rubble"). Affects only the
social-cost framing.

## IPPO setup (`algorithms/IPPO/ippo_cnn_territory_open.py` + config)

This is the no-attribution baseline — off the shelf, the paper's exact setup.

- Config: `LR 5e-4`, `NUM_ENVS 256`, `NUM_STEPS 500`, **`TOTAL_TIMESTEPS 3e8`**,
  `PARAMETER_SHARING True` (one shared policy across the 9 agents),
  `ANNEAL_LR True`.
- `ENV_KWARGS.shared_rewards` toggles common (`True`) vs individual (`False`);
  the config ships individual.
- `TUNE: True` in the config → pass `TUNE=False` for single runs.
- The config carries reward-shaping keys (`SHAPING_BEGIN 1e6`,
  `REW_SHAPING_HORIZON 5e6`); whether the IPPO script actually applies them is
  to be confirmed when the script is read in Phase 2.

## Implications for the plan

- **Horizon is 3e8** — 3× Cleanup's 1e8 — and Territory is the slowest
  SocialJax env (~3–4× slower per step than Coins). Territory runs are
  materially more expensive than the Cleanup runs were. **Phase 1a needs a
  one-run cost calibration before the full multi-seed / multi-mode baseline.**
- The experiment's no-attribution baseline = **common-reward IPPO**
  (`shared_rewards=True`) — the regime the paper shows fails on Territory due
  to credit assignment. Oracle = common-reward IPPO + `AttributionWrapper`.
- The cooperative-ceiling analytic max (Phase 1b) ≈ `(#resource cells) × 0.01 ×
  ~900` — Phase 1b computes `#resource cells` exactly from the ASCII map.
- `PARAMETER_SHARING=True` — note for the eventual Stage-4 lying experiment
  (the shared-policy / symmetric-equilibrium wrinkle).
