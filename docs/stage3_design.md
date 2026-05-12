# Stage 3 Design: Oracle Attribution via Augmented Observations

## What we're building

A wrapper that sits between SocialJax and the MAPPO trainer, adding a per-agent cooperation rate `α(i)` to each agent's observation. The simulator computes `α` from global state (no crypto yet — that's Stage 5). All other env mechanics (rewards, actions, dynamics) are unchanged.

Brief reference: Stage 3, "trusted oracle baseline."

## Schema

For each agent `i`, the augmented observation is:

```
augmented_obs[i] = concat(local_obs[i], alpha_vector)
```

where `alpha_vector` is a length-`num_agents` `float32` array — agent `i` sees every peer's `α` plus its own. (Simpler than per-peer masking; the policy can learn to attend to relevant entries.)

`α` is updated **at window boundaries** (every `N` steps), not as a rolling window. For steps `0..N-1`, `alpha = zeros`. At step `N`, we compute `α` from the buffered events of steps `0..N-1` and broadcast it for the next window.

## State

The wrapper carries an internal state alongside the env state:

```python
@dataclass
class WrappedState:
    env_state: SocialJaxState       # delegated to inner env
    event_buffer: jnp.ndarray       # shape (N, num_agents) int8, rolling
    step_in_window: int             # 0..N-1
    current_alpha: jnp.ndarray      # shape (num_agents,) float32
```

## API

```python
class AttributionWrapper:
    def __init__(self, env, predicate, window_size: int = 100):
        ...

    def reset(self, key):
        obs, env_state = self._env.reset(key)
        ws = WrappedState(env_state, zeros_buffer, 0, zeros_alpha)
        return self._augment(obs, ws.current_alpha), ws

    def step_env(self, key, ws, actions):
        obs, env_state_next, reward, done, info = self._env.step_env(key, ws.env_state, actions)
        events = self._predicate(ws.env_state, env_state_next, actions)        # (num_agents,)
        buffer_next = ws.event_buffer.at[ws.step_in_window].set(events)
        step_next = ws.step_in_window + 1
        # at window boundary, recompute alpha and reset buffer index
        at_boundary = step_next >= self._window_size
        alpha_next = jnp.where(at_boundary, buffer_next.mean(axis=0), ws.current_alpha)
        step_in_window_next = jnp.where(at_boundary, 0, step_next)
        ws_next = WrappedState(env_state_next, buffer_next, step_in_window_next, alpha_next)
        return self._augment(obs, alpha_next), ws_next, reward, done, info
```

## Predicate interface

`AttributionWrapper` is env-agnostic. The predicate function is injected:

```python
def cleanup_predicate(state_before, state_after, actions) -> jnp.ndarray:
    """Returns shape (num_agents,) of {0, 1}."""
    ...

def harvest_predicate(state_before, state_after, actions) -> jnp.ndarray:
    ...
```

These will be **JAX-native, vmapped** versions of the Stage 2 state extractors (no `int()` coercions, no Python loops). Stage 2's `cleanup_event_from_state` / `harvest_event_from_state` are the reference impl for unit-testing against.

## Integration with MAPPO

The MAPPO script (`external/SocialJax/algorithms/MAPPO/mappo_cnn_cleanup.py`) builds its env and wraps it with `MAPPOWorldStateWrapper`. We'll insert `AttributionWrapper` between SocialJax's `make()` and `MAPPOWorldStateWrapper`:

```
env = socialjax.make("clean_up", ...)
env = AttributionWrapper(env, predicate=cleanup_predicate, window_size=N)
env = MAPPOWorldStateWrapper(env)
env = LogWrapper(env)
```

The CNN policy already accepts arbitrary obs shapes (last channels concatenated), so the `+num_agents` augmented dimensions go through without architectural change. We may need to confirm by reading `MAPPOWorldStateWrapper`.

## Open questions / decisions before implementing

1. **JAX-native predicates.** Stage 2's extractors use `int()` for host-side checks. For tracing, we'll need `jnp.where`-based versions returning `(num_agents,)` arrays. Straightforward but takes a small refactor.
2. **Window alignment across the parallel envs.** `NUM_ENVS=64` runs vectorise the env; the wrapper state needs to be vmapped along the env-batch axis. JAX handles this naturally via `pmap`/`vmap`, but we need to keep `step_in_window` per env, not global.
3. **Per-peer vs per-self.** Including own `α` in the observation is simpler and probably fine. If it ends up trivially fitting (agent uses own α as a constant), we'll mask.
4. **First-window α.** Defaults to `zeros`. Alternative: don't augment until first window completes. Zeros is simpler.

## Test plan

1. JAX-native predicate functions: verify equal to Stage 2 reference on hand-constructed inputs.
2. `AttributionWrapper.step_env` unit test: 1 env, 3 agents, `N=4`, synthetic events `[1,0,1,1]` → α=[0.75, ...] after step 4.
3. Window-reset unit test: after step 4, buffer is overwritten and step_in_window resets to 0.
4. End-to-end smoke test: wrap real `clean_up` env, run 250 steps, verify obs shape grows by `num_agents` and alpha values track e_k accumulation.

## Out of scope (Stage 5 territory)

- Cryptographic proofs.
- Per-agent verification of received α.
- Self-reported / adversarial regimes (Stage 4 and 6 respectively).
