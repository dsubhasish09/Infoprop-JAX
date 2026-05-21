# Infoprop JAX

JAX/Brax implementation of [Infoprop Dyna](https://arxiv.org/abs/2501.16918).
The current example trains a Mini Wheelbot to race around procedurally generated tracks.
The full physics simulation runs on [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html); policy training uses massively parallel model rollouts on GPU via [Brax](https://github.com/google/brax).

## Algorithm Overview

Infoprop Dyna is a model-based RL algorithm designed for reliable long-horizon rollouts.
The core idea is to replace standard ensemble sampling (which compounds epistemic uncertainty) with a Kalman-filtered step that separates aleatoric and epistemic uncertainty.

The training cycle alternates between:

1. **Real data collection** — the current policy runs on the MJX simulator and stores collected transitions in a replay buffer.
2. **Model training** — a probabilistic ensemble of neural networks (`E = 8` members) is trained on the replay buffer via negative log-likelihood. Each member predicts a Gaussian over next-state deltas.
3. **Cutoff computation** — the ensemble is evaluated on the full buffer to derive rollout termination thresholds λ₁ (per-step) and λ₂ (accumulated) from the conditional entropy of the Kalman-filtered state estimate (paper eq. 12).
4. **Policy training (SAC)** — many parallel model rollouts branch from real initial states. Rollouts terminate when accumulated information loss exceeds λ₂, and the policy is updated repeatedly on each model step.

For the full technical treatment see:
> Frauenknecht et al., *On Rollouts in Model-Based Reinforcement Learning*, 2025.
> https://arxiv.org/abs/2501.16918

## Repository Structure

```
infoprop-jax/
├── infoprop_jax/
│   ├── main.py                        # Top-level Hydra entry point
│   ├── algorithms/
│   │   ├── infoprop.py                # Full Infoprop Dyna training loop (SAC + model)
│   │   ├── README.md                  # Algorithm deep-dive
│   │   └── util/
│   │       ├── nn/
│   │       │   ├── gaussian_env_model.py   # Probabilistic ensemble {p_e}
│   │       │   └── mlp.py                  # Shared MLP backbone
│   │       ├── model_learning/
│   │       │   ├── model_trainer.py        # Ensemble training orchestrator
│   │       │   ├── model_update_step.py    # NLL gradient step
│   │       │   └── model_dataset.py        # Physics replay buffer + dataset
│   │       ├── custom_evaluator.py    # Parallel evaluation wrapper
│   │       └── custom_wrapper.py      # Episode-tracking Brax wrappers
│   ├── envs/
│   │   ├── infoprop_env.py            # Model-based environment (Infoprop rollouts)
│   │   ├── README.md                  # General environment architecture
│   │   └── wheelbot/
│   │       ├── wheelbot_brax_mjx.py   # Ground-truth MJX physics environment
│   │       ├── trajectory.py          # Track state: cross-track error, lookahead
│   │       ├── utils.py               # Wheelbot geometry helpers
│   │       ├── README.md              # Wheelbot environment details
│   │       └── assets/
│   │           ├── mjcf/              # Wheelbot MuJoCo XML and meshes
│   │           └── track/             # Track generation code and saved .npz tracks
│   ├── training_scripts/
│   │   └── brax_infoprop_train.py     # Hydra entry point + domain randomisation
│   └── config/
│       ├── main.yaml                  # Hydra composition + run metadata
│       ├── algorithm/infoprop.yaml    # Model and SAC hyperparameters
│       └── env/wheelbot.yaml          # Robot control, reward, noise config
├── pyproject.toml                     # Direct project dependencies
└── uv.lock                            # Resolved dependency lockfile
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd infoprop-jax
   ```

2. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (recommended package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create a virtual environment (Python 3.13):
   ```bash
   uv venv --python 3.13
   source .venv/bin/activate
   ```

4. Resolve and install dependencies:
   ```bash
   uv lock
   uv sync
   ```

   `pyproject.toml` lists the packages this project directly depends on.
   `uv.lock` records the exact compatible versions that `uv` resolved, including transitive packages such as `jaxlib`, CUDA plugins, `chex`, and `orbax-checkpoint`.

   If `uv` fails with a cache or read-only filesystem error on a cluster, point the cache at a writable temporary directory:
   ```bash
   UV_CACHE_DIR=/tmp/uv-cache uv sync
   ```

5. Verify the environment:
   ```bash
   uv pip check
   ```

6. Verify installation:
   ```bash
   python -m infoprop_jax.envs.wheelbot.wheelbot_brax_mjx
   ```

### Changing JAX Versions

To test a different JAX/CUDA stack, change only the direct JAX requirement and let `uv` resolve the rest:

```bash
uv add "jax[cuda12]==0.9.2"
uv sync
uv pip check
```

The quotes around `jax[cuda12]` are important when using `zsh`, because square brackets are otherwise treated as filename patterns.

To inspect the resolved JAX-related package set:

```bash
uv pip list | rg '^(jax|jaxlib|jax-cuda|flax|optax|chex|orbax|brax)'
```

## Running Training

Training is managed by [Hydra](https://hydra.cc/). The top-level entry point is `infoprop_jax/main.py`.

```bash
python -m infoprop_jax.main
```

Hydra will create a timestamped output directory under `exp/` and log metrics to [Weights & Biases](https://wandb.ai/) (project `JAX_Mini_Wheelbot` by default).

**Override config values on the command line:**
```bash
python -m infoprop_jax.main \
    experiment=my_run \
   algorithm.num_model_envs=<value> \
   algorithm.max_rollout_length=<value>
```

## Key Configuration

Configuration is split across YAML files under `infoprop_jax/config/` that compose via Hydra defaults. The concrete values live in those YAMLs, not in this README.

### `infoprop_jax/config/main.yaml`

Composes the algorithm and environment configs and stores run metadata such as the seed, experiment name, W&B project, and output paths.

### `infoprop_jax/config/algorithm/infoprop.yaml`

Defines the Infoprop Dyna model, SAC, rollout, and training hyperparameters.

### `infoprop_jax/config/env/wheelbot.yaml`

Defines the Wheelbot control, reward, observation-history, and noise settings.

### `infoprop_jax/config/eval/video_eval.yaml`

Defines evaluation settings for video rendering (checkpoint path, track seed, output directory).

## References

- **Infoprop paper**: Frauenknecht et al., 2025 — https://arxiv.org/abs/2501.16918
- **Mini Wheelbot paper**: Hose et al., 2025 — https://arxiv.org/abs/2502.04582
- **Brax**: Freeman et al., 2021 — https://github.com/google/brax
- **MuJoCo MJX**: https://mujoco.readthedocs.io/en/stable/mjx.html
- **SAC**: Haarnoja et al., 2018 — https://arxiv.org/abs/1812.05905
