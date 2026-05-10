# zkAttribution — Research Proposal

## Problem Statement

Cooperation in multi-agent systems requires agents to know who contributed what to collective outcomes. The three natural options for providing this information all fail along different axes:

- **Self-reported attribution:** agents may lie.
- **Centralized attribution:** comes with a trust assumption which probably does not hold in real-life set up (e.g. in a situation where agents are deployed from different companies in an open & potentially hostile environment).
- **Full trajectory disclosure** (every agent broadcasts its raw history): agents may exploit exposed strategies of other agents.

Thus, what we need for attribution in MAS is something **truthful** (resists strategic manipulation), **decentralized** (no trusted verifier), and **low-leakage** (exposes only the cooperation summary, not the underlying behavior). To this end, we propose zkAttribution.

## Our Proposal: zkAttribution

zkAttribution is an **information-layer intervention**: it modifies only the observation function, leaving rewards and actions unchanged. It does not force cooperation; it makes reciprocity-based learning policies both **learnable** (signal is honest) and **credible** (signal is binding), which empirically moves trained policies toward cooperative equilibria that no-attribution baselines fail to reach.

Specifically, each agent computes a single scalar `α`, its cooperation rate over the last `N` steps, and broadcasts it to peers along with a zero-knowledge proof. The proof certifies that `α` is the correct output of running a public scoring function over the agent's actual trajectory. The trajectory itself stays hidden. Peers verify in milliseconds. Claims that fail verification are filtered out before reaching anyone's observation.

We call the property this achieves **cryptographic structural trustworthiness**.

## Hypotheses

- **H1 (cooperation):** zkAttribution improves cooperation over the no-attribution baseline across both Cleanup and Harvest:Open and across the tested range of dilemma severity. This isolates whether honest, decentralized attribution is a useful intervention for MARL cooperation.
- **H2 (feasibility):** Per-agent proof generation and verification latencies are small enough to integrate zkAttribution into a JAX-based MARL training loop without prohibitive wall-clock overhead, and the overhead scales reasonably with agent count and attribution interval. This isolates whether the protocol is practically deployable, not just theoretically sound.

## Protocol Mechanics in Context

### Cleanup (public goods)

- **Cooperative event:** `e_k = 1` if at step `k` the agent performed clean action while standing on a polluted aquifer tile, else `e_k = 0`.
- **What α represents:** the fraction of last `N` steps in which the agent contributed to the public good of keeping the aquifer clean.
- **What stays hidden:** which aquifer cells the agent cleaned, when the agent was harvesting apples and from where etc.
- **Why hiding matters here:** an agent's trajectory is strategic information; if revealed, it may be exploited by other agents. Restricting disclosure to `α` preserves competitive substructure while supporting reciprocity.

### Harvest:Open (commons)

- **Cooperative event:** `e_k = 1` iff at step `k` the agent harvested AND local apple density remained above the regrowth threshold afterwards; `e_k = 0` in all other cases (i.e. no harvest, or destructive harvest).
- **What α represents:** the rate of active sustainable harvesting — the fraction of steps where the agent both produced individual reward and preserved the commons.
- **What stays hidden:** which tiles the agent did harvest from, the agent's foraging routes, the high-density patches the agent has discovered.
- **Why hiding matters here:** location intelligence is the competitive edge in the game. Revealing only the restraint rate supports reciprocity without giving away the resource map.

### The shared commit–prove–verify cycle

#### Set up (one-time; before training)

- All agents agree upon the public parameters that define the proof. The problem: given a trajectory, did the agent actually achieve cooperation rate `α`?
- The public parameters are distributed to all agents, used to generate proofs (if they have the valid witness) or verify the proof (any outside auditor — even the public — can do so).
- The environment/simulator generates a signing keypair:
  - Public key is distributed to all agents in order to check signatures later.
  - Private key stays private.

#### Per-step attestation (every timestep `t`)

- Simulator has the global state — a complete snapshot of everything happening in the game.
- As this is a partially observable game, no single agent sees the complete global state. Instead, the simulator runs an observation function `O_i` for each agent.
- Before delivering the observation to an agent, the simulator stamps it with a digital signature using its private key. The signature is over three things:
  - Observation content
  - Timestamp
  - Agent ID
- Each agent picks an action, just like in a regular MARL setting.
- Each agent appends a triple to its private trajectory buffer:
  - Local observation
  - At this step, what the agent chose to do
  - Environment signature

#### Per-window commit and prove (every `N` steps)

- Each agent gathers its entire window's buffer into one big private object called the **witness**.
- Each agent computes its own cooperation rate.
- Each agent then commits to the witness:
  - Picks a random number.
  - Computes a Pedersen commitment, with the witness (the buffer) and the random number.
- The agent runs the Groth16 prover with two kinds of inputs:
  - **Public inputs:**
    - Its own cooperation rate.
    - The Pedersen commitment just computed.
    - The environment's public verification key.
  - **Private witness** (things that only that agent sees):
    - The full trajectory of triples.
    - The random number it picked for the Pedersen commitment.
- The proof attests to three statements:
  - **Opening consistency:** "the trajectory I'm proving things about really is what's inside the envelope I committed to."
  - **Signature validity:** "every observation in my trajectory was actually delivered by the environment to me, at the timestep I claim, with the content I claim."
  - **Predicate correctness:** "applying the public scoring function to my real trajectory gives exactly this cooperation rate, neither more nor less."
- Anyone can later verify the proof (the three statements above) without knowing the agents' private trajectories.

#### Phase 4 — Verify (per peer, per claim)

- Each agent runs the Groth16 verifier on others' claims.
- If the proof is valid, the agent accepts it as a verified claim and stores it for use in the next observation.
- If not, it does not reach the agent's policy — basically filtered out.

## Experiment setup

- **Environments:** Cleanup and Harvest:Open from SocialJax.
- **Algorithm:** MAPPO under CTDE, shared hyperparameters across regimes.
- **Regimes compared:** no-attribution baseline vs. zkAttribution.
- **Metrics:**
  - **Cooperation:** cleaning rate / sustainable-harvest rate, collective return, individual return.
  - **Computational performance:** proof generation latency, verification latency, total per-step overhead, wall-clock training overhead.
  - **Privacy leakage:** inference-attack models attempting to recover hidden trajectory features from `α` should perform near random.
  - **Extra:** red-team attack in 3 forgery scenarios & measure rejection rate:
    - Claim a false `α` with a forged proof.
    - Commit to a fabricated trajectory with no/replayed signatures.
    - Attempt to reuse another agent's signed observations.

## Contributions

- **Conceptual:** cryptographic structural trustworthiness as a security property of communication channels in multi-agent learning, formalizing when a channel is robust against strategic manipulation by rational agents.
- **Protocol:** zkAttribution, the first end-to-end ZKP-verified attribution mechanism for sequential social dilemmas, with proofs of completeness, soundness, and zero-knowledge under standard cryptographic assumptions.
- **Empirical:** the first measurement of ZKP-verified attribution in MARL — cooperation gains across two canonical SSDs without a trusted verifier.
- **Systems:** the cooperation–overhead trade-off characterised on commodity hardware, demonstrating practical deployability in realistic training pipelines.
