# Configuration

[Hydra](https://hydra.cc/) configs compose into one global config at startup. `main.yaml` sets the
defaults; everything else is an override group.

```
config/
├── main.yaml                 # composition + run metadata (seed, experiment, W&B, output paths)
├── algorithm/infoprop.yaml   # model, SAC, rollout, and training hyperparameters
├── env/                      # one file per environment (selected with env=<name>)
│   ├── wheelbot.yaml         # Wheelbot control, reward, history, noise
│   ├── humanoid.yaml         # MJX tutorial Humanoid reward, reset, history
│   ├── humanoid_race.yaml    # Humanoid racing reward, trajectory, reset
│   └── ant.yaml              # stock Brax ant via DefaultInfopropWrappable
└── eval/video_eval.yaml      # checkpoint / track / output settings for rendering
```

## Composition order

`main.yaml`'s `defaults` list composes `algorithm` then `env`, so an env file can override algorithm
values by setting them under its own `algorithm:` block (the env tunes the shared defaults to itself).
The env files keep these overrides grouped under the same `# --- section ---` headers as
`algorithm/infoprop.yaml` so the two read the same way. Per-parameter detail lives as comments in the
YAML itself.

Selecting the env (`env=`, default `wheelbot`):
```bash
python -m infoprop_jax.main env=humanoid        # also: humanoid_race, ant
```
The name is looked up in `ENV_REGISTRY` (`../envs/__init__.py`) via the `env_name` key of the matching
`env/<name>.yaml`. Override anything on the CLI:
```bash
python -m infoprop_jax.main experiment=my_run \
    algorithm.num_model_envs=1000 algorithm.max_rollout_length=1000
```

## Training schedule

The schedule in `algorithm/infoprop.yaml` is hierarchical — trials → epochs → steps. Each of the
`num_trials` outer iterations collects `real_steps_per_trial` real transitions, refits the ensemble,
and runs `epochs_per_trial` agent-training epochs; every epoch re-initialises the model envs from real
states and takes `max_rollout_length` model env steps, with `utd_ratio` SAC gradient updates per step.
`random_init` controls how the very first dataset (before the first model fit) is collected: uniform
random actions (`True`, default) or the untrained policy (`False`). `model_subsampling` sets the
fraction of each step's transitions kept in the SAC replay buffer, which holds one epoch's worth of
data. `policy_network_layer_norm` and `q_network_layer_norm` toggle layer norm (critic layer norm
prevents unbounded Q-value growth on high-dimensional action spaces).

## Evaluation

`eval/video_eval.yaml` is read only when `video_eval=true`; see the top-level README's *Running
training* section for the invocation.
