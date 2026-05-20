# Environments

This package separates general Infoprop rollout logic from environment-specific ground-truth simulators.

## Training Pair

Training currently uses two Brax `PipelineEnv` implementations with the same interface:

| File | Role |
|---|---|
| `infoprop_env.py` | Model-based environment used for parallel Infoprop rollouts. It still uses the Wheelbot state/action/trajectory helpers in this refactor step. |
| `wheelbot/wheelbot_brax_mjx.py` | Wheelbot MuJoCo/MJX ground-truth environment used for data collection and evaluation. |

During training, `training_scripts/brax_infoprop_train.py` registers both under different Brax names and passes them to `algorithms.infoprop.train()`.

## Infoprop State

The model environment carries extra per-step uncertainty metrics in `state.info`:

| Key | Description |
|---|---|
| `accumulated_conditional_entropy` | Sum of per-step H(s̃) over the rollout so far |
| `current_conditional_entropy` | Per-step H(s̃) for the most recent step |
| `per_step_cutoff` | λ₁ threshold (computed from real-data buffer) |
| `accumulated_cutoff` | λ₂ threshold |
| `kalman_gain` | K per state dimension |
| `conditional_var` | (1 − K) · Sigma_GT per state dimension |
| `fused_mean` / `fused_var` | Precision-weighted ensemble fusion |
| `epist_var` | Epistemic variance (ensemble disagreement) |

## File Map

| File | Role |
|---|---|
| `infoprop_env.py` | Model-based environment. Used for parallel training rollouts. |
| `wheelbot/` | Wheelbot-specific simulator, trajectory logic, docs, and assets. |
