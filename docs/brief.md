# zkAttribution — Implementation Brief

## Goal

Build an experimental pipeline to test two hypotheses:

- **H1 (cooperation):** verified attribution improves cooperation over no-attribution and self-reported attribution baselines in two SSDs.
- **H2 (feasibility):** the cryptographic protocol's overhead is small enough to integrate into a JAX MARL training loop.

Target venue: FAccT 2027 (deadline ~Jan 2027). Aim for working pipeline + initial results within ~3 months; full sweep + adversarial validation by ~6 months.

## Stack

- **Environments:** SocialJax (https://github.com/cooperativex/SocialJax) — Cleanup and Harvest:Open.
- **MARL:** MAPPO under CTDE, as implemented in SocialJax. Use shared hyperparameters across all regimes.
- **Crypto:** circom for circuits, snarkjs for Groth16 proving/verification, Pedersen commitments on BabyJubJub inside BN254, EdDSA signatures for environment attestation. Proof generation off the JAX hot loop on host CPU; batch via `multiprocessing.Pool`.
- **Compute:** single-GPU initially for MARL; CPU for proof generation. Plan for parallel envs (~32–128).

## Staged build plan

Build in this order — **do not try to implement crypto first.**

### Stage 1 — Reproduce no-attribution baseline (1–2 weeks)

- Get SocialJax + MAPPO running locally. Reproduce published cooperation rates on Cleanup and Harvest:Open with default hyperparameters and 3 dilemma severity levels (use Willis et al.'s self-interest level parameterisation).
- Establish logging infrastructure: cleaning rate, sustainable-harvest rate, collective return, individual return, plus per-step wall clock.
- Multiple seeds (≥5), report mean ± 95% CI.

**Exit criterion:** baseline cooperation curves match published numbers within reasonable tolerance.

### Stage 2 — Implement the cooperative-event predicate (1 week)

- **Cleanup:** `e_k = 1` iff agent fired clean-action while standing on a polluted aquifer tile AND the tile's pollution state actually decreased afterward. Else 0.
- **Harvest:Open:** `e_k = 1` iff agent harvested AND local apple density remained above the regrowth threshold afterward. Else 0.
- Compute true `α = (1/N) Σ e_k` from each agent's trajectory using the global state available during training.

**Exit criterion:** unit tests for both predicates against hand-constructed trajectories.

### Stage 3 — Wire α into augmented observations as oracle attribution (1 week)

- Concatenate the per-peer `α` vector from Stage 2 onto each agent's local observation at the start of every window (interval `N` — start with `N = 100`, parameterise).
- This is the "trusted oracle" baseline: simulator computes `α` from global state and broadcasts it directly. **No crypto yet.**
- Run the full training sweep (both envs × 3 severity levels × ≥5 seeds).

**Exit criterion:** if oracle attribution does NOT improve cooperation over Stage 1, stop and investigate before adding crypto. If it does, proceed.

### Stage 4 — Self-reported attribution baseline (1 week)

- Add a new action dimension where each agent broadcasts a self-claimed `α` (continuous in `[0, 1]`) at the end of each window.
- Peers receive the broadcast as part of their next observation (no verification).
- Run training sweep.
- Log: claimed `α` vs. true `α` (correlation, inflation rate), peer policies' causal sensitivity to received reports.

**Exit criterion:** quantitative measurement of whether agents learn to lie and whether that lying corrupts cooperation.

### Stage 5 — Cryptographic layer (3–4 weeks)

This is the heaviest stage. Implement in sub-stages with unit tests at each.

#### 5a. Environment signing

- Generate `(sk_E, pk_E)` at simulator init.
- For each observation delivered to each agent, compute `σ = EdDSA-Sign(sk_E, (i, t, o_t^i))` and attach.
- Each agent stores `(o, a, σ)` triples in its trajectory buffer.
- Verify signatures correctly **outside the circuit first**.

#### 5b. Predicate circuit in circom

- Inputs: trajectory window (private), claimed `α` (public), commitment `c` (public), `pk_E` (public).
- Subcircuits:
  - Pedersen commitment opening verification.
  - EdDSA signature verification per timestep.
  - Predicate evaluation (sum of `e_k` values, divide by `N`).
- Compile, generate Groth16 trusted setup parameters via a small ceremony (or use snarkjs's mock setup for development; do a real ceremony before camera-ready).
- Profile constraint count; expect ~200K–400K for `N = 100` dominated by signature verification.

#### 5c. Per-window prove + verify integration

- At end of each window, agent computes commitment, generates proof, broadcasts `(α, c, π)`.
- Peers run `Groth16.Verify` in milliseconds; valid proofs feed `α` into augmented observation, invalid proofs zero it out.
- Run proof generation on host CPU off the JAX loop.

#### 5d. Performance optimization

- Batch proof generation across parallel environments.
- Profile end-to-end overhead: proof gen latency, verification latency, total wall-clock training overhead vs. Stage 3 baseline.

**Exit criterion:** zkAttribution training run completes with overhead < ~2× baseline wall-clock time.

### Stage 6 — Adversarial validation (1–2 weeks)

Three forgery scenarios, each as a separate experiment:

1. **False α with forged proof.** Adversary tries to broadcast `α = 0.9` with a fake proof. Verify rejection rate.
2. **Fabricated trajectory (no/replayed signatures).** Adversary commits to a trajectory it never actually had. Verify rejection rate.
3. **Stolen signatures.** Adversary attempts to use peer's signed observations as its own. Verify rejection rate.

For each, also measure adversary's discounted return relative to honest peers, and compare against the same forgery attempt under the self-reported baseline (where the forgery succeeds).

### Stage 7 — Full experimental sweep + paper analysis (2–4 weeks)

Run all 4 regimes (no-attribution, self-reported, oracle-attribution, zkAttribution) × 2 environments × 3 severity levels × ≥5 seeds. Plus adversarial study at one severity level.

Final metrics to report:

- **Cooperation:** cleaning rate, sustainable-harvest rate, collective return, individual return.
- **Computational:** proof-gen latency, verification latency, per-step overhead, total training wall-clock overhead.
- **Privacy leakage:** train an inference-attack model that tries to predict (agent position, harvest tile, cleaning status) from `α` + local observation. Report attack accuracy vs. random baseline.
- **Cross-play:** pair regime A's agents with regime B's agents, measure cooperation under mismatch.
- **Schelling diagrams:** cooperator/defector payoffs vs. group composition.

## Key implementation gotchas

- **Predicate-event ambiguity.** Be explicit about whether `e_k` requires successful event vs. attempted event. Use **successful** (it's harder to inflate). Document choice in the predicate spec.
- **Proof generation off the JAX loop.** Do NOT try to fuse proof generation into the JAX-compiled training loop. Run it on host CPU, batch across parallel envs, write proofs into the next observation buffer asynchronously.
- **Circuit constraint count is the bottleneck.** Most of the cost is signature verification, not the predicate itself. If overhead is too high, consider sampled signature verification (verify a random subset per proof) — this is a soundness trade-off and should be flagged in limitations.
- **Trusted setup for development.** Use snarkjs's mock setup during development. A real multi-party ceremony is only needed for the final reported numbers.
- **Self-reported baseline action space.** Agents need a continuous-valued action dimension for the self-report. Make sure MAPPO handles this (it does, with appropriate continuous action heads).
- **Severity parameterisation.** Use Willis et al.'s computed self-interest level as the x-axis for severity, not raw environment parameters.

## Reference materials

- Consolidated protocol spec (the document with notation table + 4-phase walkthrough).
- SocialJax codebase + paper.
- Groth16 paper (Groth, EUROCRYPT 2016).
- Pedersen commitments (Pedersen, CRYPTO 1991).
- circom + snarkjs documentation.
- MAPPO paper (Yu et al. 2022).

## What success looks like

After Stage 7:

- Cooperation gap between zkAttribution and no-attribution > Y% across both envs at high severity (validates H1).
- Cooperation gap between zkAttribution and self-reported demonstrates verification's specific contribution (validates the contribution argument).
- Per-step protocol overhead measured at < few hundred ms with practical `N` (validates H2).
- Adversarial study shows 100% forgery rejection and adversary return drop matching honest-defector return.
- Privacy leakage attack accuracy near random (validates low-leakage claim).

That's enough for a defensible FAccT submission. Anything beyond this (richer predicates, fully decentralized peer attestation, optimistic-with-reputation hybrid) is future work.
