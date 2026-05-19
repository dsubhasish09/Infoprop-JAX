# Wheelbot Racing — Infoprop Dyna

JAX/Brax implementation of [Infoprop Dyna](https://arxiv.org/abs/2501.16918) for training a miniature two-wheeled robot to race around procedurally-generated tracks.
The full physics simulation runs on [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html); policy training uses massively parallel model rollouts on GPU via [Brax](https://github.com/google/brax).

## Algorithm Overview

Infoprop Dyna is a model-based RL algorithm designed for reliable long-horizon rollouts.
The core idea is to replace standard ensemble sampling (which compounds epistemic uncertainty) with a Kalman-filtered step that separates aleatoric and epistemic uncertainty.

The training cycle alternates between:

1. **Real data collection** — the current policy runs on the MJX simulator and stores collected transitions in a replay buffer.
2. **Model training** — a probabilistic ensemble of neural networks (`E = 8` members) is trained on the replay buffer via negative log-likelihood. Each member predicts a Gaussian over next-state deltas.
3. **Cutoff computation** — the ensemble is evaluated on the full buffer to derive rollout termination thresholds λ₁ (per-step) and λ₂ (accumulated) from the conditional entropy of the Kalman-filtered state estimate (paper eq. 12).
4. **Policy training (SAC)** — many parallel model rollouts branch from real initial states. Rollouts terminate when accumulated information loss exceeds λ₂, and the policy is updated repeatedly on each model step.

**Dynamics invariances** are exploited to augment initial states: the robot's global XY position, yaw, and wheel angles are invariant to the learned dynamics, so rollouts can start from arbitrary positions on any of the pre-generated tracks — not just those visited in real data.

**Partial observability** is handled by conditioning the dynamics model on a history of `H = 20` past states and actions rather than a single state.

For the full technical treatment see:
> Frauenknecht et al., *Infoprop: Propagating Uncertainty in Model-Based Reinforcement Learning*, 2025.
> https://arxiv.org/abs/2501.16918

## Repository Structure

```
wheelbot-racing/
├── wheelbot_sim_python/
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
│   │   ├── wheelbot_brax_mjx.py       # Ground-truth MJX physics environment
│   │   ├── wheelbot_brax_infoprop.py  # Model-based environment (Infoprop rollouts)
│   │   ├── trajectory.py              # Track state: cross-track error, lookahead
│   │   ├── utils.py                   # Geometry helpers
│   │   └── README.md                  # Environment deep-dive
│   ├── track/
│   │   ├── generator.py               # Procedural track generation (Catmull-Rom)
│   │   └── utils.py                   # Contour discretisation
│   ├── training_scripts/
│   │   └── brax_infoprop_train.py     # Hydra entry point + domain randomisation
│   └── config/
│       ├── main.yaml                  # Hydra composition + run metadata
│       ├── algorithm/infoprop.yaml    # Model and SAC hyperparameters
│       └── env/wheelbot.yaml          # Robot control, reward, noise config
├── saved_tracks/                      # 200 pre-generated track .npz files
├── requirements/
│   └── requirements.txt
└── setup.py
```

## Installation

1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd wheelbot-racing
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

4. Install dependencies:
   ```bash
   uv pip install -r requirements/requirements.txt
   ```

5. Install the package in editable mode:
   ```bash
   uv pip install -e .
   ```

6. Verify installation:
   ```bash
   python -m wheelbot_sim_python.envs.wheelbot_brax_mjx
   ```

## Running Training

Training is managed by [Hydra](https://hydra.cc/). The top-level entry point is `wheelbot_sim_python/main.py`.

```bash
python -m wheelbot_sim_python.main
```

Hydra will create a timestamped output directory under `exp/` and log metrics to [Weights & Biases](https://wandb.ai/) (project `JAX_Mini_Wheelbot` by default).

**Override config values on the command line:**
```bash
python -m wheelbot_sim_python.main \
    experiment=my_run \
   algorithm.num_model_envs=<value> \
   algorithm.max_rollout_length=<value>
```

**Resume from a checkpoint** (automatic if a checkpoint exists in the Hydra output dir):
```bash
python -m wheelbot_sim_python.main \
    hydra.run.dir=exp/2025-01-01_12-00-00 \
    algorithm.auto_resume=true
```

## Key Configuration

Configuration is split across three YAML files that compose via Hydra defaults. The concrete values live in those YAMLs, not in this README.

### `config/main.yaml`

Composes the algorithm and environment configs and stores run metadata such as the seed, experiment name, W&B project, and output paths.

### `config/algorithm/infoprop.yaml`

Defines the Infoprop Dyna model, SAC, rollout, and training hyperparameters.

### `config/env/wheelbot.yaml`

Defines the Wheelbot control, reward, observation-history, and noise settings.

### `config/eval/video_eval.yaml`

Defines evaluation settings for video rendering (checkpoint path, track seed, output directory).

## References

- **Infoprop paper**: Frauenknecht et al., 2025 — https://arxiv.org/abs/2501.16918
- **Mini Wheelbot paper**: The Mini Wheelbot: A Testbed for Learning-based Balancing, Flips, and Articulated Driving — https://arxiv.org/abs/2502.04582
- **Brax**: Freeman et al., 2021 — https://github.com/google/brax
- **MuJoCo MJX**: https://mujoco.readthedocs.io/en/stable/mjx.html
- **SAC**: Haarnoja et al., 2018 — https://arxiv.org/abs/1812.05905
