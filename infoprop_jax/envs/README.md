# Environments

This package separates the **environment-agnostic Infoprop core** from **environment-specific MJX
simulators**. There are three pieces:

| File | Class | Role |
|---|---|---|
| `infoprop_wrappable_env.py` | `InfopropWrappable(PipelineEnv)` | The contract a real MJX env implements to be Infoprop-compatible. **Start here to add a new env.** |
| `infoprop_env.py` | `InfopropEnv(brax Wrapper)` | The generic Infoprop core (decode, fusion, Kalman, entropy, cutoffs, sampling). Written once; never re-implemented per env. |
| `wheelbot/wheelbot_brax_mjx.py` | `WheelbotEnv(InfopropWrappable)` | The example env: real MuJoCo/MJX physics **and** the Infoprop hooks, in one class. |
| `humanoid/humanoid_mjx.py` | `HumanoidEnv(InfopropWrappable)` | MuJoCo MJX tutorial Humanoid with `context_size = 0`; validates the identity prediction path. |

A real env is used directly for data collection / evaluation; the model environment is built by
wrapping a real env: `InfopropEnv(WheelbotEnv(cfg), min_log_var=…, max_log_var=…, fast_model_rollout=…)`.
Because `InfopropEnv` is a Brax `Wrapper`, it forwards the underlying env's interface (`dt`,
`action_size`, sizes, the data contract) automatically and composes with the training wrappers in
`../algorithms/util/custom_wrapper.py`.

## The contract (`InfopropWrappable`)

A new environment sets these **attributes** in `__init__`: `model_state_size`, `context_size`,
`full_state_size`, `obs_history`, `act_history` (`dt`/`action_size` come from `PipelineEnv`).

**Required hooks**

| Hook | Signature |
|---|---|
| `preprocess` | `(state, action) -> (nn_input, curr_model_state, curr_context, applied_action, processed_action)` |
| `postprocess` | `(state, applied_action, next_model_state, next_context, processed_action, build_pipeline_state) -> State` |
| `reset_from_buffer` | `(rng, init_transition, build_pipeline_state) -> State` |
| `_get_obs`, `_get_rew` | observation / `(reward, done, reward_metrics)` |
| `reset`, `step` | standard `PipelineEnv` (real physics) |

**Optional hooks** (identity defaults): `augment_prediction` — override only when `context_size > 0`
to append integrated dims and propagate their variance *before* fusion.

**Data contract** (lets the framework carry env-owned state without knowing its key names):
`dummy_physics_transition` (sizes the physics buffer + declares context fields),
`extract_physics_transition(prev, next, policy_extras)` (builds the model-training transition),
optional `context_from_transition(transition)` (reads context from buffer extras for cutoff
evaluation; default returns an empty vector), and
`reset_carry_keys` (the env-owned dynamic info keys the auto-reset wrapper reverts on `done`).

The generic history utilities `shift_phys` / `shift_action` are provided by the base.

## State vectors

- **`model_state`** (`model_state_size`) — the dims the ensemble predicts deltas of.
- **`context`** (`context_size`) — extra dims reconstructed by integration in `augment_prediction`;
  `0` if the NN predicts the whole next state.
- **`full_state`** = `model_state + context` (`full_state_size`) — the entropy / cutoff space.

## `info`-key ownership

`InfopropEnv` writes/reads only **framework-owned** keys (fixed names): `model`, `model_obs_mean/std`,
`next_state_delta_mean/std`, `per_step_cutoff`, `accumulated_cutoff`, `binning_entropy`,
`accumulated/current_conditional_entropy`, `rng`, `info_cutoff`. Every other key (the model-state,
history, context, task id) is **env-owned** — named freely and touched only by the env's hooks.

The model environment carries these uncertainty quantities in `state.info` during a rollout:

| Key | Description |
|---|---|
| `accumulated_conditional_entropy` | Sum of per-step H(s̃) over the rollout so far |
| `current_conditional_entropy` | Per-step H(s̃) for the most recent step |
| `per_step_cutoff` / `accumulated_cutoff` | λ₁ / λ₂ thresholds (computed from the real-data buffer) |
| `info_cutoff` | 1 when the rollout was terminated by an entropy violation this step |

## File map

| Path | Role |
|---|---|
| `infoprop_wrappable_env.py` | The env contract + generic history utilities. |
| `infoprop_env.py` | The generic Infoprop core wrapper. |
| `wheelbot/` | Example env: simulator, trajectory logic, docs, and assets. |
| `humanoid/` | Example env adapted from the MuJoCo MJX tutorial. |
