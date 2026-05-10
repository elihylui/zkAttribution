# zkAttribution

Zero-knowledge cryptographic attribution for cooperation in multi-agent reinforcement learning. Each agent broadcasts a single scalar — its cooperation rate `α` over the last `N` steps — together with a Groth16 proof that `α` is the correct output of a public scoring function applied to the agent's actual (private) trajectory. Peers verify in milliseconds and filter out invalid claims before they reach their observation.

**Target venue:** FAccT 2027.

## Hypotheses

- **H1 (cooperation):** verified attribution improves cooperation over no-attribution and self-reported baselines in Cleanup and Harvest:Open.
- **H2 (feasibility):** the protocol's overhead is small enough to integrate into a JAX MARL training loop.

## Status

Stage 1 — reproducing the no-attribution MAPPO baseline on SocialJax (Cleanup, Harvest:Open). See [`docs/brief.md`](docs/brief.md) for the full 7-stage build plan.

## Repository layout

```
docs/                  implementation brief, research proposal, design notes
src/zkattribution/     main Python package (env wrappers, predicate, training)
experiments/           configs + run scripts per stage / regime
circuits/              circom circuits and snarkjs artifacts
tests/                 unit tests (esp. cooperative-event predicate)
notebooks/             exploratory analysis
scripts/               utilities
```

Directories are created as each stage produces content.

## Documents

- [`docs/brief.md`](docs/brief.md) — implementation brief (stages, exit criteria, gotchas)
- [`docs/proposal.md`](docs/proposal.md) — research proposal (problem, protocol, contributions)

## Stack

- **Environments:** [SocialJax](https://github.com/cooperativex/SocialJax) — Cleanup, Harvest:Open
- **MARL:** MAPPO under CTDE
- **Crypto:** circom + snarkjs + Groth16; Pedersen commitments on BabyJubJub in BN254; EdDSA for environment attestation
- **Compute:** single-GPU for MARL; host CPU for proof generation (off the JAX hot loop)
