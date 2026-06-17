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

Training alternates real-data collection on the MJX simulator, fitting a probabilistic ensemble to
the buffer, deriving entropy-based rollout-termination thresholds (λ₁ per-step, λ₂ accumulated) from
it, and SAC updates on many parallel imagined rollouts that branch from real states and stop when
accumulated information loss exceeds the threshold. The per-step fusion / Kalman / entropy math and
the thresholds are detailed in [`algorithms/README.md`](infoprop_jax/algorithms/README.md).

> Frauenknecht et al., *On Rollouts in Model-Based Reinforcement Learning*, 2025 -
> https://arxiv.org/abs/2501.16918

---

## Architecture

The code is organised so that **the Infoprop algorithm never mentions a specific robot**:

```
InfopropWrappable                       ── the methods/attributes a new env must define
        ▲ subclasses
MyEnv(PipelineEnv, InfopropWrappable)   ── ONE class: real MJX physics + the Infoprop methods
        ▲ wrapped by
InfopropEnv(brax Wrapper)               ── generic, written once: the fixed Infoprop core math
```

A concrete env (e.g. [`WheelbotEnv`](infoprop_jax/envs/wheelbot/wheelbot_brax_mjx.py)) plays **two
roles** from one observation/reward/state layout: the ground-truth env for data collection /
evaluation, and the env that [`InfopropEnv`](infoprop_jax/envs/infoprop_env.py) wraps for imagined
rollouts. `InfopropEnv` is a generic Brax `Wrapper` holding **all the fixed math** (ensemble fusion +
Kalman update, conditional entropy, sampling, cutoffs); you never rewrite it, you construct it:
`InfopropEnv(MyEnv(cfg), min_log_var=…, max_log_var=…)`. Being a `Wrapper`, it stacks uniformly with
the training wrappers in
[`algorithms/util/custom_wrapper.py`](infoprop_jax/algorithms/util/custom_wrapper.py):

```
CustomAutoResetWrapper → CustomEpisodeWrapper → VmapInfopropWrapper → InfopropEnv → MyEnv
```

[`DefaultInfopropWrappable`](infoprop_jax/envs/default_wrappable.py) is a ready-made class for any
stock Brax env with a flat observation (model state == observation, no context, obs-based reward);
see [`envs/quadruped/ant.py`](infoprop_jax/envs/quadruped/ant.py).

The one model step (`preprocess → NN → decode → augment_prediction → infoprop_core → postprocess`),
the three state vectors (`model_state` / `context` / `full_state`), the history window, and which
`info` keys belong to the training code versus your env are all documented in
[`infoprop_jax/envs/README.md`](infoprop_jax/envs/README.md).

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
│   │   ├── video_eval_ant.py            # Ant variant: real video + model rollout (no model video)
│   │   └── eval_utils.py                # Shared checkpoint-evaluation helpers
│   └── config/                          # Hydra configs (main / algorithm / env / eval) + README
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

### Hardware requirements

Training requires an NVIDIA GPU with CUDA 12 support (the stack is pinned to `jax[cuda12]`).
Compute capability ≥ 8.0 (Ampere) or newer is recommended; development used Hopper (H100) and
Blackwell (RTX PRO 6000) cards.

GPU memory depends on the environment and on `algorithm.num_model_envs` (default 10000).
Approximate peak VRAM for the bundled examples:

| Environment   | Peak VRAM |
|---------------|-----------|
| wheelbot      | ~12 GB    |
| ant           | ~18 GB    |
| humanoid_race | ~29 GB    |
| humanoid      | ~35 GB    |

- **Minimum:** 16 GB (runs wheelbot).
- **Recommended:** 40 GB+ to run every bundled env (humanoid is the heaviest).
- Lower `algorithm.num_model_envs` to reduce memory on smaller GPUs.

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
See [`infoprop_jax/config/README.md`](infoprop_jax/config/README.md) for config composition, the
training schedule, and how `env=` resolves to a class.

Render a checkpoint (real vs. model rollout; humanoid and ant checkpoints are routed automatically
to `video_eval_humanoid.py` / `video_eval_ant.py`, the latter rendering only the real rollout since
default-wrapped envs have no model `pipeline_state`):
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

[Hydra](https://hydra.cc/) configs compose under [`infoprop_jax/config/`](infoprop_jax/config/):
`main.yaml` sets the defaults and run metadata; `algorithm/infoprop.yaml` holds the hyperparameters;
one `env/<name>.yaml` per environment; `eval/video_eval.yaml` for rendering. Composition order, the
trials → epochs → steps schedule, and CLI override examples are in
[`infoprop_jax/config/README.md`](infoprop_jax/config/README.md); per-parameter detail lives in the
YAML comments.

---

## References

- **Infoprop**: Frauenknecht et al., 2025 - https://arxiv.org/abs/2501.16918
- **Mini Wheelbot**: Hose et al., 2025 - https://arxiv.org/abs/2502.04582
- **Brax**: Freeman et al., 2021 - https://github.com/google/brax
- **MuJoCo MJX**: https://mujoco.readthedocs.io/en/stable/mjx.html
- **SAC**: Haarnoja et al., 2018 - https://arxiv.org/abs/1812.05905

---

## Acknowledgements

Aside from the core Infoprop pipeline, a substantial portion of the surrounding code and
documentation was written with the help of AI coding tools, especially [Claude Code](https://claude.com/claude-code).
