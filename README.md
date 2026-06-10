# Infoprop JAX

JAX/Brax implementation of [Infoprop Dyna](https://arxiv.org/abs/2501.16918), structured as a
**reusable model-based RL framework**: the Infoprop algorithm lives in environment-agnostic code,
and any [MuJoCo MJX](https://mujoco.readthedocs.io/en/stable/mjx.html) environment can be plugged in
by implementing a small interface. The bundled example trains a **Mini Wheelbot** to race around
procedurally generated tracks.

The full physics simulation runs on MJX; policy training uses massively parallel *imagined* model
rollouts on GPU via [Brax](https://github.com/google/brax).

---

## Algorithm overview

Infoprop Dyna is a model-based RL algorithm designed for reliable long-horizon rollouts. Standard
model-based RL rolls out an ensemble of learned dynamics models, but epistemic (model) uncertainty
compounds and long rollouts drift into garbage. Infoprop fixes this by treating each ensemble
member's prediction as a **noisy observation** of the true next state and applying a **Kalman
update**, which separates:

- **aleatoric** uncertainty (inherent noise — propagated), from
- **epistemic** uncertainty (model disagreement — measured and used to *terminate* rollouts early).

The training cycle alternates between:

1. **Real data collection** — the current policy runs on the MJX simulator; transitions go into a
   physics replay buffer.
2. **Model training** — a probabilistic ensemble (`E = 8` members) is fit to the buffer by negative
   log-likelihood; each member predicts a Gaussian over next-state deltas.
3. **Cutoff computation** — the ensemble is evaluated on the buffer to derive rollout-termination
   thresholds λ₁ (per-step) and λ₂ (accumulated) from the conditional entropy of the Kalman-filtered
   estimate (paper eq. 12).
4. **Policy training (SAC)** — many parallel imagined rollouts branch from real initial states.
   Rollouts terminate when accumulated information loss exceeds λ₂, and the policy is updated
   repeatedly on the synthetic transitions.

> Frauenknecht et al., *On Rollouts in Model-Based Reinforcement Learning*, 2025 —
> https://arxiv.org/abs/2501.16918

---

## Architecture

The code is organised so that **the Infoprop algorithm never mentions a specific robot**. There are
three pieces:

```
InfopropWrappable(PipelineEnv)      ── the contract: hooks a new env implements
        ▲ subclasses
WheelbotEnv(InfopropWrappable)      ── ONE class: real MJX physics + the Infoprop hooks
        ▲ wrapped by
InfopropEnv(brax Wrapper)           ── generic, written once: the fixed Infoprop core math
```

- [`InfopropWrappable`](infoprop_jax/envs/infoprop_wrappable_env.py) — a `PipelineEnv` subclass that
  declares the env-specific hooks Infoprop needs (and ships the generic history utilities). A new
  environment subclasses this.
- [`WheelbotEnv`](infoprop_jax/envs/wheelbot/wheelbot_brax_mjx.py) — a single class playing **two
  roles**: the ground-truth env for data collection / evaluation (via its real `step`), and the env
  that `InfopropEnv` wraps for imagined rollouts (via its hooks). One observation/reward/state
  layout serves both.
- [`InfopropEnv`](infoprop_jax/envs/infoprop_env.py) — a generic Brax `Wrapper` holding **all the
  fixed math** (decode, ensemble fusion + Kalman update, conditional entropy, sampling, cutoffs,
  ensemble-trainer wiring). You never rewrite it; you construct it:
  `InfopropEnv(WheelbotEnv(cfg), min_log_var=…, max_log_var=…, fast_model_rollout=…)`.

Because `InfopropEnv` is a `Wrapper`, it composes uniformly with the Infoprop training wrappers in
[`algorithms/util/custom_wrapper.py`](infoprop_jax/algorithms/util/custom_wrapper.py); the model
rollout stack is:

```
CustomAutoResetWrapper → CustomEpisodeWrapper → VmapInfopropWrapper → InfopropEnv → WheelbotEnv
```

### One model step

A single imagined step runs this pipeline (the env author writes only the **bold** stages; the rest
is fixed in `InfopropEnv`):

```
preprocess → NN forward → decode → augment_prediction → infoprop_core → postprocess
 (env, bold)  (InfopropEnv) (InfopropEnv) (env, optional)   (InfopropEnv)  (env, bold)
```

`decode` un-normalises the predicted delta and integrates it (`next = curr + delta·dt`);
`infoprop_core` does precision-weighted fusion → Kalman gain → conditional variance → entropy →
sampling. Both are fixed, parametrised only by `dt` and the running delta-normalisation statistics.

### Three state vectors

- **`model_state`** (`model_state_size`) — what the neural net predicts deltas of.
- **`context`** (`context_size`) — extra dims you reconstruct *by integration* from the model_state
  (the Wheelbot's odometry). **Set `context_size = 0` if the NN predicts the entire next state** —
  then `augment_prediction` does nothing.
- **`full_state`** = `model_state + context` (`full_state_size`) — the entropy/cutoff space.

A **history window** (`obs_history`, `act_history`) handles partial observability: the NN input is
the last `obs_history` model-states + `act_history` actions, assembled by `preprocess`.

### `info`-key ownership

- **Framework-owned** keys keep fixed names and you never touch them: `model` (ensemble params),
  `model_obs_mean/std`, `next_state_delta_mean/std`, `per_step_cutoff`, `accumulated_cutoff`,
  `binning_entropy`, `accumulated/current_conditional_entropy`, `rng`, episode bookkeeping.
- **Env-owned** keys (the model-state, history, context, task id) are named freely by the env author;
  only your hooks read or write them. Two small declarations let the generic wrappers carry/reset
  them without knowing the names (see below).

---

## Repository structure

```
infoprop-jax/
├── infoprop_jax/
│   ├── main.py                          # Hydra entry point (train / video-eval dispatch)
│   ├── algorithms/
│   │   ├── infoprop.py                  # Generic Infoprop Dyna training loop (SAC + model + data)
│   │   └── util/
│   │       ├── nn/                      # GaussianEnsembleModel + shared MLP backbone
│   │       ├── model_learning/          # Ensemble trainer, NLL step, physics replay buffer
│   │       ├── agent_learning/          # SAC networks + losses
│   │       ├── custom_evaluator.py      # Parallel evaluation (mean/std returns)
│   │       └── custom_wrapper.py        # Infoprop training wrappers (vmap/episode/auto-reset)
│   ├── envs/
│   │   ├── infoprop_wrappable_env.py    # InfopropWrappable: the env contract (NEW-ENV ENTRY POINT)
│   │   ├── infoprop_env.py              # InfopropEnv: the generic Infoprop core (env-agnostic)
│   │   ├── wheelbot/                    # Example env: WheelbotEnv + trajectory + assets
│   │   └── humanoid/                    # Example env: MuJoCo MJX tutorial Humanoid
│   ├── training_scripts/
│   │   └── brax_infoprop_train.py       # Builds the envs + W&B and calls infoprop.train()
│   ├── eval_scripts/video_eval.py       # Render real vs. model rollouts for a checkpoint
│   └── config/                          # Hydra configs (main / algorithm / env / eval)
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
Change only the direct requirement and let `uv` re-resolve:
```bash
uv add "jax[cuda12]==0.9.2" && uv sync && uv pip check
```
(Quote `jax[cuda12]` in `zsh`.)

---

## Running training

Training is managed by [Hydra](https://hydra.cc/); the entry point is `infoprop_jax/main.py`:

```bash
python -m infoprop_jax.main
```

Hydra writes a timestamped directory under `exp/` and logs to
[Weights & Biases](https://wandb.ai/) (project `JAX_Mini_Wheelbot`). Override config on the CLI:

```bash
python -m infoprop_jax.main experiment=my_run \
    algorithm.num_model_envs=1000 algorithm.max_rollout_length=1000
```

Train the bundled MJX tutorial Humanoid example:
```bash
python -m infoprop_jax.main env=humanoid
```

Render a checkpoint (real vs. model rollout):
```bash
python -m infoprop_jax.main video_eval=true eval.log_dir=exp/<run_dir> eval.iteration=<N>
```

---

## Adding a new MJX environment

To run Infoprop Dyna on your own MJX system you write **one** class,
`class MyEnv(InfopropWrappable)`, plus a small entry point. The framework supplies the entire
algorithm — fusion, Kalman, entropy, cutoffs, normalization, history, SAC, replay buffers, and
auto-reset — so you only describe *your* dynamics representation.

### 1. Subclass `InfopropWrappable` and set the sizes

In `__init__`, build your MJX `sys` and call `super().__init__(sys, n_frames=…, backend='mjx')`,
then set:

```python
self.model_state_size = …    # the dim the NN predicts deltas of
self.context_size     = …    # extra dims you integrate yourself (0 if the NN predicts everything)
self.full_state_size  = self.model_state_size + self.context_size
self.obs_history, self.act_history = …
```

### 2. Standard physics + observation/reward

Implement the usual Brax interface with your real dynamics: `reset(self, rng)`,
`step(self, state, action)` (must store the model-state, history and context into `state.info`),
`_get_obs(...)`, and `_get_rew(self, state, action) -> (reward, done, reward_metrics)`.

### 3. The Infoprop hooks (the contract `InfopropEnv` calls)

```python
def preprocess(self, state, action):
    # -> (nn_input, curr_model_state, curr_context, applied_action, processed_action)
    # nn_input  : what the ensemble eats (history + action)
    # curr_*    : current model_state / context (the decode integrates onto curr_model_state)
    # applied_action  : action sent to the dynamics (inject a control prior here, if any)
    # processed_action: action used for obs/reward (e.g. the clipped RL action)

def postprocess(self, state, applied_action, next_model_state, next_context,
                processed_action, build_pipeline_state):
    # rebuild a valid Brax State: obs, env-owned info, reward, done. Build the MJX
    # pipeline_state only if build_pipeline_state is True. rng + entropy accumulation
    # are already set for you. Return the new State.

def reset_from_buffer(self, rng, init_transition, build_pipeline_state):
    # turn a sampled physics transition (history in .observation, context in
    # .extras['state_extras']) into an imagined-rollout initial State.
```

Optional hooks, with identity defaults (override only when needed):

```python
def augment_prediction(self, member_mean, member_var, curr_model_state, curr_context):
    # default identity (context_size == 0). Override to append integrated dims AND
    # propagate their variance — single rollout sample: [E, model_state_size] in.
```

### 4. The data contract (lets the framework carry your state without knowing its names)

```python
@property
def dummy_physics_transition(self):       # sizes the physics buffer + declares context fields
def extract_physics_transition(self, prev_state, next_state, policy_extras):  # builds it each real step
def context_from_transition(self, transition):  # optional; context for cutoff evaluation
@property
def reset_carry_keys(self):               # your dynamic info keys the auto-reset wrapper reverts on done
```

### 5. Wire it up

```python
real_env  = MyEnv(cfg)
model_env = InfopropEnv(MyEnv(cfg), min_log_var=…, max_log_var=…, fast_model_rollout=…)
infoprop.train(environment=real_env, model_environment=model_env, eval_environment=…, …)
```

The training loop reads every dimension and `dt` from the envs — nothing is hardcoded. See
[`training_scripts/brax_infoprop_train.py`](infoprop_jax/training_scripts/brax_infoprop_train.py)
for a complete example entry point.

### The minimal case
If the NN predicts the entire next state directly: set `context_size = 0`, skip
`augment_prediction` (identity default), make `preprocess` pass the action through
(`applied_action = processed_action = action`), and `postprocess` simply writes the next state +
obs and calls `_get_rew`. You still get fusion, Kalman, entropy, cutoffs, normalization, history,
SAC, buffers and auto-reset for free.

### What you never write
`decode`, `infoprop_core`, the model-step orchestration, the cutoff math, `init_NN_trainer`, the SAC
loop, the replay buffers, and the wrappers. The single enforced contract is the **core step's I/O**:
`preprocess` hands the core a model_state + NN input; `postprocess` turns the sampled next full-state
back into a `State`.

---

## Configuration

Hydra configs compose under `infoprop_jax/config/`:

- `config/main.yaml` — composition + run metadata (seed, experiment, W&B project, output paths).
- `config/algorithm/infoprop.yaml` — model, SAC, rollout, and training hyperparameters.
- `config/env/wheelbot.yaml` — Wheelbot control, reward, history, and noise settings.
- `config/env/humanoid.yaml` — MJX tutorial Humanoid reward, reset, and history settings.
- `config/eval/video_eval.yaml` — checkpoint/track/output settings for rendering.

---

## References

- **Infoprop**: Frauenknecht et al., 2025 — https://arxiv.org/abs/2501.16918
- **Mini Wheelbot**: Hose et al., 2025 — https://arxiv.org/abs/2502.04582
- **Brax**: Freeman et al., 2021 — https://github.com/google/brax
- **MuJoCo MJX**: https://mujoco.readthedocs.io/en/stable/mjx.html
- **SAC**: Haarnoja et al., 2018 — https://arxiv.org/abs/1812.05905
