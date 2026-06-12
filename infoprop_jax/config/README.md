# Configuration reference

All training and evaluation is configured through [Hydra](https://hydra.cc/). This document
describes every key; the YAML files themselves carry only short reminders.

## Composition

[`main.yaml`](main.yaml) composes one file from each group, in this order:

```yaml
defaults:
  - _self_
  - algorithm: infoprop   # algorithm/infoprop.yaml
  - env: wheelbot         # env/<name>.yaml
  - eval: video_eval      # eval/video_eval.yaml
```

Because `env` composes **after** `algorithm`, an env file may force algorithm values
(e.g. `env/ant.yaml` sets `algorithm.discounting`, `algorithm.target_entropy`). Anything can be
overridden last on the CLI:

```bash
python -m infoprop_jax.main env=ant algorithm.num_model_envs=1000 seed=3
```

## `main.yaml` — run-level keys

| Key | Meaning |
|---|---|
| `video_eval` | `false` = train; `true` = render a checkpoint instead (see [eval](#evalvideo_evalyaml)) |
| `seed` | Global JAX/training seed |
| `experiment` | Name used in the output path and W&B |
| `wandb_project` / `wandb_run_id` | W&B project; set `wandb_run_id` to resume a specific run |
| `root_dir` + `hydra.run.dir` | Output layout: `exp/<experiment>/<seed>/<date>/<time>` |
| `agent_dir`, `dataset_dir`, `metrics_dir` | Paths used by data-transfer / offline tooling |

## `algorithm/infoprop.yaml` — training loop

The outer loop runs `num_trials` iterations of: collect real data → train model → derive rollout
cutoffs → train SAC on model rollouts.

### Real-data collection

| Key | Meaning |
|---|---|
| `episode_length` | Real-env episode length (steps before truncation) |
| `num_trials` | Outer-loop iterations |
| `real_steps_per_trial` | Real transitions collected per trial (must exceed `model_batch_size`, or the model gets zero updates) |
| `num_real_train_envs` | Parallel real envs; collection runs ceil(`real_steps_per_trial` / this) scan steps |
| `physics_buffer_size` | Real-data buffer capacity; `null` = `num_trials * real_steps_per_trial` (keep everything) |
| `random_init` | `True`: first trial uses uniform random actions; `False`: the untrained policy |

### Model rollouts (synthetic data)

| Key | Meaning |
|---|---|
| `num_model_envs` | Parallel imagined rollouts, each branched from a real buffer state |
| `model_steps_per_epoch` | Model env steps per SAC training epoch |
| `epochs_per_trial` | Epochs per trial; each epoch re-branches all rollouts from fresh real states |
| `model_subsampling` | Fraction of each step's `num_model_envs` transitions inserted into the SAC buffer, in (0, 1] |
| `keep_past_epoch` | SAC buffer holds the whole trial (`True`) or only the current epoch (`False`) |

The SAC replay buffer size is fully derived:
`(epochs_per_trial if keep_past_epoch else 1) * model_steps_per_epoch * round(model_subsampling * num_model_envs)`.
Training logs the resulting replay ratio and warns above 50 (critic-overestimation risk — lower
`utd_ratio` or raise `model_subsampling` / `num_model_envs`).

### Rollout cutoffs (InfoProp)

| Key | Meaning |
|---|---|
| `max_rollout_length` | Hard cap on imagined rollout length |
| `lower_quantile` | Quantile of real-data entropy → λ₂ (accumulated information-loss cutoff) |
| `upper_quantile` | Quantile of real-data entropy → λ₁ (per-step cutoff) |
| `action_repeat` | Env action repeat (applies to real and eval envs) |

### Agent (SAC)

| Key | Meaning |
|---|---|
| `agent_learning_rate`, `agent_batch_size`, `agent_hidden_layer_sizes` | Standard SAC hyperparameters |
| `policy_network_layer_norm`, `q_network_layer_norm` | Layer norm per network. Critic layer norm prevents unbounded Q growth on high-dimensional action spaces (humanoid) |
| `utd_ratio` | SAC gradient updates per model env step |
| `target_entropy` | Entropy target; env files usually set it to −action_dim |
| `tau` | Target-network EMA coefficient |
| `tune_entropy` / `alpha` | `True`: learn temperature starting at `alpha`; `False`: fix it at `alpha` |
| `reset_agent_per_trial`, `reset_model_per_trial`, `reset_model_replay_buffer` | Re-initialize the agent / model / SAC buffer at each trial boundary |

### Model training (ensemble)

| Key | Meaning |
|---|---|
| `model_learning_rate`, `model_weight_decay`, `model_batch_size`, `model_hidden_layer_sizes`, `model_layer_norm` | Ensemble optimizer/architecture |
| `patience` | Early stopping: epochs without validation-loss improvement |
| `min_log_var`, `max_log_var` | Clamp on predicted per-dimension log-variance |

### Correlated exploration noise (`exploration_noise`)

SAC normally samples actions as `tanh(mu + sigma * eps)` with `eps ~ N(0, I)` drawn independently
every step (temporally *white* noise). This block replaces `eps` with noise that has the **same
N(0, 1) per-step marginal** but is correlated over time, so exploration applies coherent,
sustained perturbations instead of step-to-step dithering. Per-step action distributions, SAC
losses, deterministic evaluation, and the uniform-random initial prefill are all unaffected —
only the *sequence* of exploration samples changes. Implementation:
[`algorithms/util/exploration_noise.py`](../algorithms/util/exploration_noise.py).

| Key | Meaning |
|---|---|
| `type` | `none` (baseline white noise, zero overhead), `ar1`, or `pink` |
| `beta` | `ar1` only: smoothing factor. Correlation time ≈ `1/beta` agent steps (`beta=1` is white noise, `beta=0.01` gives ~100-step pushes) |
| `num_filters` | `pink` only: number of AR(1) filters at octave-spaced timescales τ = 1, 2, 4, …, 2^(k−1) steps; equal-weight summation gives an approximate 1/f spectrum |
| `apply_real` | Use correlated noise for real-env collection (the main benefit: more diverse real states for model and agent) |
| `apply_model` | Use correlated noise in imagined rollouts (smaller effect — info cutoffs keep rollouts short relative to the slow filters) |

Guidance:

- `pink` is the robust default choice: it covers fast and slow timescales simultaneously, so it
  needs no tuning beyond `num_filters`. Choose `num_filters` so the slowest filter
  (2^(k−1) steps) matches the longest maneuver worth exploring — k=8 → ~128 steps; going beyond
  `episode_length` (or the typical model rollout length, for `apply_model`) buys nothing.
- `ar1` exposes a single timescale and is the cleaner ablation: sweep `beta ∈ {0.3, 0.1, 0.03}`
  to find *which* correlation time helps.
- Noise state is re-drawn from its stationary distribution at every episode (auto)reset, so
  correlation never leaks across episode boundaries.
- Trade-off: longer correlation times explore better but make behavior data more off-policy,
  which can slow critic learning; pink (half its energy at short timescales) is the middle ground.

### Miscellaneous

| Key | Meaning |
|---|---|
| `num_real_eval_envs` | Parallel envs for the deterministic evaluation runs |
| `discounting` | SAC discount factor |
| `normalize_observations` | Running-statistics observation normalization |

## `env/<name>.yaml` — environment selection

Each file uses `# @package _global_` and writes two subtrees: `env:` (constructor kwargs for the
class found via `ENV_REGISTRY` in [`envs/__init__.py`](../envs/__init__.py)) and optionally
`algorithm:` overrides the env forces. Keys shared by all envs:

| Key | Meaning |
|---|---|
| `name` / `env_name` | Display name / registry key |
| `obs_history`, `act_history` | How many past observations/actions form the model-learning context |

All remaining keys are env-specific (reward weights, controller gains, noise models, …) and are
documented in the env files themselves and in the env guide,
[`envs/README.md`](../envs/README.md).

## `eval/video_eval.yaml`

Used when running `python -m infoprop_jax.main video_eval=true`:

| Key | Meaning |
|---|---|
| `log_dir` | **Required**: Hydra training output dir, e.g. `exp/test/27752/2026.05.05/135429` |
| `iteration` | Checkpoint index; `null` = latest `brax_policy_*` found |
| `track_seed` | Track selection (wheelbot/humanoid_race only) |
| `output_dir` | `null` = `<log_dir>/video_eval/iter_<N>/` |
| `no_model` | Skip the model rollout (e.g. no model checkpoint saved) |
| `max_steps` | Max steps for the real rollout |
| `rng_seed` | RNG seed for the evaluation |
