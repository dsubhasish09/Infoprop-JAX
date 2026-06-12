# Infoprop JAX

JAX/Brax implementation of [Infoprop Dyna](https://arxiv.org/abs/2501.16918), structured as a
**reusable model-based RL framework**: the Infoprop algorithm lives in environment-agnostic code,
and any [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) environment can be plugged in
by implementing a small set of methods. Bundled examples: a **Mini Wheelbot** racing around procedurally
generated tracks, the MJX-tutorial **Humanoid**, a **Humanoid racing** variant on scaled Wheelbot
tracks, and the stock Brax **Ant** made compatible via `DefaultInfopropWrappable` without writing
any env-specific methods.

The full physics simulation runs on MJX; policy training uses massively parallel *imagined* model
rollouts on GPU via [Brax](https://github.com/google/brax).

---

## Algorithm overview

Infoprop Dyna is a model-based RL algorithm designed for reliable long-horizon rollouts. Standard
model-based RL rolls out an ensemble of learned dynamics models, but epistemic (model) uncertainty
compounds and long rollouts drift into garbage. Infoprop fixes this by treating each ensemble
member's prediction as a **noisy observation** of the true next state and applying a **Kalman
update**, which separates:

- **aleatoric** uncertainty (inherent noise - propagated), from
- **epistemic** uncertainty (model disagreement - measured and used to *terminate* rollouts early).

The training cycle alternates between:

1. **Real data collection** - the current policy runs on the MJX simulator; transitions go into a
   physics replay buffer.
2. **Model training** - a probabilistic ensemble (`E = 8` members) is fit to the buffer by negative
   log-likelihood; each member predicts a Gaussian over next-state deltas.
3. **Cutoff computation** - the ensemble is evaluated on the buffer to derive rollout-termination
   thresholds λ₁ (per-step) and λ₂ (accumulated) from the conditional entropy of the Kalman-filtered
   estimate (paper eq. 12).
4. **Policy training (SAC)** - many parallel imagined rollouts branch from real initial states.
   Rollouts terminate when accumulated information loss exceeds λ₂, and the policy is updated
   repeatedly on the synthetic transitions.

> Frauenknecht et al., *On Rollouts in Model-Based Reinforcement Learning*, 2025 -
> https://arxiv.org/abs/2501.16918

---

## Architecture

The code is organised so that **the Infoprop algorithm never mentions a specific robot**. The
pieces:

```
InfopropWrappable                       ── the methods/attributes a new env must define
        ▲ subclasses
MyEnv(PipelineEnv, InfopropWrappable)   ── ONE class: real MJX physics + the Infoprop methods
        ▲ wrapped by
InfopropEnv(brax Wrapper)               ── generic, written once: the fixed Infoprop core math
```

- [`InfopropWrappable`](infoprop_jax/envs/infoprop_wrappable_env.py) - a small base class with no
  Brax functionality of its own; it declares the env-specific methods Infoprop calls. A new
  environment inherits from it together with `PipelineEnv` (own physics) or
  `brax.envs.base.Wrapper` (building on an existing env).
- A concrete env (e.g. [`WheelbotEnv`](infoprop_jax/envs/wheelbot/wheelbot_brax_mjx.py) or
  [`HumanoidEnv`](infoprop_jax/envs/humanoid/humanoid_mjx.py)) - a single class playing **two
  roles**: the ground-truth env for data collection / evaluation (via its real `step`), and the env
  that `InfopropEnv` wraps for imagined rollouts (via the Infoprop methods). One
  observation/reward/state layout serves both.
- [`InfopropEnv`](infoprop_jax/envs/infoprop_env.py) - a generic Brax `Wrapper` holding **all the
  fixed math** (ensemble fusion + Kalman update, conditional entropy, sampling, cutoffs,
  ensemble-trainer setup). You never rewrite it; you construct it:
  `InfopropEnv(MyEnv(cfg), min_log_var=…, max_log_var=…)`.
- [`DefaultInfopropWrappable`](infoprop_jax/envs/default_wrappable.py) - a ready-made class that
  makes any stock Brax env with a flat observation Infoprop-wrappable: model state == observation,
  `context_size = 0`, with reward in imagined rollouts supplied as an obs-based
  `reward_fn(obs, action, next_obs) -> (reward, done)`. See
  [`envs/quadruped/ant.py`](infoprop_jax/envs/quadruped/ant.py) for the example.

Because `InfopropEnv` is a `Wrapper`, it stacks uniformly with the Infoprop training wrappers in
[`algorithms/util/custom_wrapper.py`](infoprop_jax/algorithms/util/custom_wrapper.py); the model
rollout stack is:

```
CustomAutoResetWrapper → CustomEpisodeWrapper → VmapInfopropWrapper → InfopropEnv → MyEnv
```

### One model step

A single imagined step runs this pipeline (the env author writes only the `(env)` stages; the rest
is fixed in `InfopropEnv`):

```
preprocess → NN forward → decode → augment_prediction → infoprop_core → postprocess
   (env)    (InfopropEnv) (InfopropEnv)  (env, optional)  (InfopropEnv)      (env)
```

`decode` un-normalises the predicted delta and integrates it (`next = curr + delta·dt`);
`infoprop_core` does precision-weighted fusion → Kalman gain → conditional variance → entropy →
sampling. Both are fixed, parametrised only by `dt` and the running delta-normalisation statistics.

### Three state vectors

- **`model_state`** (`model_state_size`) - what the neural net predicts deltas of.
- **`context`** (`context_size`) - extra dims you reconstruct *by integration* from the model_state
  (typically world-frame odometry, as in the Wheelbot and Humanoid envs). **Set `context_size = 0`
  if the NN predicts the entire next state** - then `augment_prediction` does nothing.
- **`full_state`** = `model_state + context` (`full_state_size`) - the entropy/cutoff space.

A **history window** (`obs_history`, `act_history`) handles partial observability: the NN input is
the last `obs_history` model-states + `act_history` actions, assembled by `preprocess`.

### Who manages which `info` keys

- Keys **managed by the training code** keep fixed names and you never touch them: `model`
  (ensemble params), `model_obs_mean/std`, `next_state_delta_mean/std`, `per_step_cutoff`,
  `accumulated_cutoff`, `binning_entropy`, `accumulated/current_conditional_entropy`, `rng`,
  episode bookkeeping.
- **Your environment's own** keys (the model-state, history, context, task id) are named freely;
  only your env's methods read or write them. Two small declarations let the generic wrappers
  carry/reset them without knowing the names (see below).

---

## Repository structure

```
infoprop-jax/
├── infoprop_jax/
│   ├── main.py                          # Main script (train / video-eval selection via Hydra)
│   ├── algorithms/
│   │   ├── infoprop.py                  # Generic Infoprop Dyna training loop (SAC + model + data)
│   │   └── util/
│   │       ├── nn/                      # GaussianEnsembleModel + shared MLP backbone
│   │       ├── model_learning/          # Ensemble trainer, NLL step, physics replay buffer
│   │       ├── agent_learning/          # SAC networks + losses
│   │       ├── custom_evaluator.py      # Parallel evaluation (mean/std returns)
│   │       └── custom_wrapper.py        # Infoprop training wrappers (vmap/episode/auto-reset)
│   ├── envs/
│   │   ├── infoprop_wrappable_env.py    # InfopropWrappable: what a new env must define (START HERE)
│   │   ├── infoprop_env.py              # InfopropEnv: the generic Infoprop core (env-agnostic)
│   │   ├── default_wrappable.py         # Makes stock Brax envs compatible (model state == obs)
│   │   ├── contract_validation.py       # Startup checks on new envs (jax.eval_shape, no physics)
│   │   ├── __init__.py                  # ENV_REGISTRY: config name -> env class
│   │   ├── wheelbot/                    # Example env: WheelbotEnv + trajectory + assets
│   │   ├── humanoid/                    # HumanoidEnv + HumanoidRaceEnv + scaled race tracks
│   │   └── quadruped/                   # AntEnv: stock Brax ant via DefaultInfopropWrappable
│   ├── training_scripts/
│   │   └── brax_infoprop_train.py       # Builds the envs + W&B and calls infoprop.train()
│   ├── eval_scripts/
│   │   ├── video_eval.py                # Render real vs. model rollouts for a checkpoint
│   │   ├── video_eval_humanoid.py       # Humanoid/humanoid-race rendering variant
│   │   └── eval_utils.py                # Shared checkpoint-evaluation helpers
│   └── config/                          # Hydra configs (main / algorithm / env / eval)
├── jobscript*.sh                        # SLURM launch scripts (train / video eval)
├── pyproject.toml                       # Direct dependencies
└── uv.lock                              # Resolved lockfile
```

---

## Installation

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create a Python 3.13 environment and install:
   ```bash
   uv venv --python 3.13
   source .venv/bin/activate
   uv lock && uv sync
   uv pip check
   ```
   On a cluster with a read-only cache: `UV_CACHE_DIR=/tmp/uv-cache uv sync`.
3. Verify:
   ```bash
   python -m infoprop_jax.envs.wheelbot.wheelbot_brax_mjx
   python -m infoprop_jax.envs.humanoid.humanoid_mjx
   ```

### Changing the JAX/CUDA stack
The current pin is `jax[cuda12]==0.8.0` (see `pyproject.toml`). To change it, change only the
direct requirement and let `uv` re-resolve:
```bash
uv add "jax[cuda12]==<version>" && uv sync && uv pip check
```
(Quote `jax[cuda12]` in `zsh`.)

---

## Running training

Training is managed by [Hydra](https://hydra.cc/); the main script is `infoprop_jax/main.py`:

```bash
python -m infoprop_jax.main
```

Hydra writes a timestamped directory under `exp/` and logs to
[Weights & Biases](https://wandb.ai/) (project set via `wandb_project` in `config/main.yaml`).
Override config on the CLI:

```bash
python -m infoprop_jax.main experiment=my_run \
    algorithm.num_model_envs=1000 algorithm.max_rollout_length=1000
```

Select the environment with `env=` (default: `wheelbot`, per `config/main.yaml`):
```bash
python -m infoprop_jax.main env=humanoid        # also: humanoid_race, ant
```
Each name is looked up in the `ENV_REGISTRY` dictionary in
[`infoprop_jax/envs/__init__.py`](infoprop_jax/envs/__init__.py) via the `env_name` key of the
matching `config/env/<name>.yaml`.

Render a checkpoint (real vs. model rollout; humanoid checkpoints are routed automatically to
`video_eval_humanoid.py`):
```bash
python -m infoprop_jax.main video_eval=true eval.log_dir=exp/<run_dir> eval.iteration=<N>
```

---

## Adding a new environment

There are two routes, both documented step by step in the canonical guide,
[`infoprop_jax/envs/README.md`](infoprop_jax/envs/README.md):

- **Default wrapping** - for any stock Brax env with a flat observation: subclass
  `DefaultInfopropWrappable` with an obs-based reward function and add it to the env dictionary.
  No env-specific Infoprop methods to write. Example:
  [`envs/quadruped/ant.py`](infoprop_jax/envs/quadruped/ant.py).
- **Full implementation** - for envs with their own physics, a control prior, or a
  model_state/context factorisation: implement the `InfopropWrappable` methods yourself.
  Reference implementations: [`envs/wheelbot/`](infoprop_jax/envs/wheelbot/) and
  [`envs/humanoid/`](infoprop_jax/envs/humanoid/).

Either way, `validate_infoprop_contract(env)` checks at startup that your environment defines
every required method with the right shapes (under `jax.eval_shape`), so mistakes fail fast with
a pointed error instead of an opaque scan/vmap trace.

---

## Configuration

Hydra configs compose under `infoprop_jax/config/`:

- `config/main.yaml` - composition + run metadata (seed, experiment, W&B project, output paths).
- `config/algorithm/infoprop.yaml` - model, SAC, rollout, and training hyperparameters.
- `config/env/wheelbot.yaml` - Wheelbot control, reward, history, and noise settings.
- `config/env/humanoid.yaml` - MJX tutorial Humanoid reward, reset, and history settings.
- `config/env/humanoid_race.yaml` - Humanoid racing reward, trajectory, and reset settings.
- `config/env/ant.yaml` - stock Brax ant via `DefaultInfopropWrappable`.
- `config/eval/video_eval.yaml` - checkpoint/track/output settings for rendering.

The training schedule in `config/algorithm/infoprop.yaml` is hierarchical - trials → epochs →
steps. Each of the `num_trials` outer iterations collects `real_steps_per_trial` real transitions,
refits the ensemble, and runs `epochs_per_trial` agent-training epochs; every epoch re-initialises
the model envs from real states and takes `model_steps_per_epoch` model env steps, with `utd_ratio`
SAC gradient updates per step. `random_init` controls how the very first dataset (before the first
model fit) is collected: uniform random actions (`True`, default) or the untrained policy
(`False`). `model_subsampling` sets the fraction of each step's transitions
kept in the SAC replay buffer, and `keep_past_epoch` controls whether the buffer holds the whole
trial or just the current epoch (buffer sizes are derived from these knobs). Network options
include `policy_network_layer_norm` and `q_network_layer_norm` (critic layer norm prevents
unbounded Q-value growth on high-dimensional action spaces). Per-parameter details live as
comments in the YAML itself.

---

## References

- **Infoprop**: Frauenknecht et al., 2025 - https://arxiv.org/abs/2501.16918
- **Mini Wheelbot**: Hose et al., 2025 - https://arxiv.org/abs/2502.04582
- **Brax**: Freeman et al., 2021 - https://github.com/google/brax
- **MuJoCo MJX**: https://mujoco.readthedocs.io/en/stable/mjx.html
- **SAC**: Haarnoja et al., 2018 - https://arxiv.org/abs/1812.05905
