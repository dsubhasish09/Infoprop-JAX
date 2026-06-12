# Environments

This package separates the **environment-agnostic Infoprop core** from **environment-specific MJX
simulators**. The pieces:

| File | Class | Role |
|---|---|---|
| `infoprop_wrappable_env.py` | `InfopropWrappable` | The methods and attributes a real env must define to work with Infoprop. **Start here to add a new env.** |
| `infoprop_env.py` | `InfopropEnv(brax Wrapper)` | The generic Infoprop core (fusion, Kalman, entropy, cutoffs, sampling). Written once; never re-implemented per env. |
| `default_wrappable.py` | `DefaultInfopropWrappable(Wrapper, InfopropWrappable)` | Ready-made class that makes a stock Brax env Infoprop-compatible: model state = observation, no context. |
| `contract_validation.py` | `validate_infoprop_contract(env)` | Startup check (under `jax.eval_shape`) that the env defines every required method with the right shapes. |
| `wheelbot/wheelbot_brax_mjx.py` | `WheelbotEnv(PipelineEnv, InfopropWrappable)` | Example env: real MuJoCo/MJX physics **and** the Infoprop methods, with a variant/invariant state split. |
| `humanoid/humanoid_mjx.py` | `HumanoidEnv(PipelineEnv, InfopropWrappable)` | MuJoCo MJX tutorial Humanoid; local floating-base model state + integrated odometry context. |
| `humanoid/humanoid_race_mjx.py` | `HumanoidRaceEnv(HumanoidEnv)` | Humanoid racing on scaled Wheelbot tracks; reuses the parent's Infoprop methods. |
| `quadruped/ant.py` | `AntEnv(DefaultInfopropWrappable)` | Stock Brax ant made compatible with no hand-written methods - the default-wrapping example. |

A real env is used directly for data collection / evaluation; the model environment is built by
wrapping a real env: `InfopropEnv(MyEnv(cfg), min_log_var=…, max_log_var=…)`.
Because `InfopropEnv` is a Brax `Wrapper`, it passes through the underlying env's attributes
(`dt`, `action_size`, sizes, the buffer-layout declarations) automatically and stacks with the
training wrappers in `../algorithms/util/custom_wrapper.py`.

## What a new environment must define (`InfopropWrappable`)

`InfopropWrappable` is a small base class with no Brax functionality of its own: it declares the
methods Infoprop will call on your env, plus the generic history helpers `shift_phys` /
`shift_action`. Your env inherits from it **together with** a Brax base class that provides the
usual env machinery - `PipelineEnv` if you write your own physics, or `brax.envs.base.Wrapper` if
you build on an existing env.

A new environment sets these **attributes** in `__init__`: `model_state_size`, `context_size`,
`full_state_size`, `obs_history`, `act_history` (`dt`/`action_size`/`observation_size` come from
the Brax base class).

**Required methods**

| Method | Signature |
|---|---|
| `preprocess` | `(state, action) -> (nn_input, curr_model_state, curr_context, applied_action, processed_action)` |
| `postprocess` | `(state, applied_action, next_model_state, next_context, processed_action) -> State` |
| `reset_from_buffer` | `(rng, init_transition) -> State` |
| `reset`, `step` | standard Brax env (real physics) |

**Optional**: `augment_prediction(member_mean, member_var, curr_model_state, curr_context)
-> (mean, var)` - identity by default; **must be overridden whenever `context_size > 0`** to
append integrated dims and propagate their variance *before* fusion (maps `[E, model_state_size]`
to `[E, full_state_size]`; the startup check enforces the shapes).

**Buffer layout declarations** (these let the training code store and restore your env's state
without knowing your key names):
`dummy_physics_transition` (sizes the physics buffer + declares context fields; its
`state_extras` must contain `truncation`),
`extract_physics_transition(prev, next, policy_extras)` (builds the model-training transition),
optional `context_from_transition(transition)` (reads context from buffer extras for cutoff
evaluation; default returns an empty vector), and
`reset_carry_keys` (the dynamic info keys of your env that the auto-reset wrapper restores on
`done`).

<!-- Fast model rollouts (skipping the MJX `pipeline_state` during model rollouts) are an optimisation
entirely inside your env: return `pipeline_state=None` to skip the rebuild. Whatever you choose,
the State **structure must be identical** across `postprocess` and `reset_from_buffer` - JAX
requires this because both run inside the same `lax.scan`. The common pattern is a
`self.fast_model_rollout` flag read from cfg. -->

## State vectors

- **`model_state`** (`model_state_size`) - the dims the ensemble predicts deltas of.
- **`context`** (`context_size`) - extra dims reconstructed by integration in `augment_prediction`;
  `0` if the NN predicts the whole next state.
- **`full_state`** = `model_state + context` (`full_state_size`) - the entropy / cutoff space.

## Who manages which `info` keys

`InfopropEnv` writes/reads only its own keys, with fixed names you never touch: `model`,
`model_obs_mean/std`, `next_state_delta_mean/std`, `per_step_cutoff`, `accumulated_cutoff`,
`binning_entropy`, `accumulated/current_conditional_entropy`, `rng`, `info_cutoff`. Every other
key (the model-state, history, context, task id) belongs to your environment - name them freely;
only your env's methods read or write them.

The model environment carries these uncertainty quantities in `state.info` during a rollout:

| Key | Description |
|---|---|
| `accumulated_conditional_entropy` | Sum of per-step H(s̃) over the rollout so far |
| `current_conditional_entropy` | Per-step H(s̃) for the most recent step |
| `per_step_cutoff` / `accumulated_cutoff` | λ₁ / λ₂ thresholds (computed from the real-data buffer) |
| `info_cutoff` | 1 when the rollout was terminated by an entropy violation this step |

## Naming and startup checks

`ENV_REGISTRY` in `__init__.py` is a dictionary mapping the config-facing name to the environment
class. Importing `infoprop_jax.envs` - which the training script does - passes every entry to
Brax's `envs.register_environment`, so `brax.envs.get_environment(<name>)` can build them by name.
Hydra selects the env via `config/env/<name>.yaml`, whose `env_name` key must match the dictionary
key.

`validate_infoprop_contract(env)` (in `contract_validation.py`) runs once at the top of
`infoprop.train` and checks that your environment defines everything with the right shapes. It
calls every required method under `jax.eval_shape` - no physics compute - and verifies the
declared sizes, the `dummy_physics_transition` shapes, the `preprocess` return shapes, the
`augment_prediction` `[E, model_state] → [E, full_state]` mapping, and that `postprocess` and
`reset_from_buffer` produce structurally identical States (the `lax.scan` requirement above). A
new env that passes these checks will run; one that doesn't fails at startup with a pointed error
instead of an opaque scan/vmap trace.

---

## Walkthrough A - wrapping a stock Brax env (`DefaultInfopropWrappable`)

The fast path for any preexisting Brax env with a flat observation vector.
`DefaultInfopropWrappable` already defines all the required methods for the simple case:

* model state = the env's observation (`model_state_size = observation_size`);
* no context (`context_size = 0`, identity `augment_prediction`);
* NN input = concatenated observation/action histories;
* real-env reward and termination come from the wrapped env unchanged.

Constraints: dict observations are unsupported; model rollouts always return
`pipeline_state=None`, so model-env video rendering is not available (the real-env path is
unaffected). Do **not** wrap an env that already defines the Infoprop methods itself - the
wrapper's own methods would silently take precedence; subclass that env instead.

Example: the ant (`quadruped/ant.py`, config `../config/env/ant.yaml`).

**1. Write an observation-based reward function.** In imagined rollouts there is no physics
`pipeline_state`, so the stock reward logic cannot run. Supply a pure-jax, per-sample
`reward_fn(obs, action, next_obs) -> (reward, done)` (it runs inside the vmapped model step):

```python
def ant_reward(obs, action, next_obs):
  """Obs-based proxy of the Brax 'ant' reward (default flags: positions excluded
  from obs, no contact cost).

  Obs layout: [z, quat(4), joint qpos(8), qvel(14)] = 27; x-velocity ~= qvel[0]
  = next_obs[13]. Healthy iff z in [0.2, 1.0]; reward = forward velocity +
  healthy bonus (1.0) - ctrl cost (0.5 * ||a||^2). The usual deviation (same
  proxy MBPO uses): velocity is read from the generalized velocity instead of
  Brax's torso-COM finite difference, so model-rollout rewards differ slightly
  in scale from the real-env eval reward.
  """
  x_velocity = next_obs[13]
  z = next_obs[0]
  is_healthy = jp.where(z < 0.2, 0.0, 1.0)
  is_healthy = jp.where(z > 1.0, 0.0, is_healthy)
  ctrl_cost = 0.5 * jp.sum(jp.square(action))
  reward = x_velocity + 1.0 - ctrl_cost
  done = 1.0 - is_healthy
  return reward, done
```

The proxy does not need to match the stock reward exactly - it only drives the SAC updates on
imagined data; evaluation always runs in the real env. (Subclasses may override `_get_rew`
instead of passing a `reward_fn`.)

**2. Subclass `DefaultInfopropWrappable`:**

```python
class AntEnv(DefaultInfopropWrappable):
  """Stock Brax ant, Infoprop-wrappable out of the box."""

  def __init__(self, cfg: DictConfig = DictConfig({}), eval_mode: bool = False,
               **kwargs):
    # eval_mode is accepted for registry-call compatibility; the stock env has
    # no eval variant. The stock class is constructed directly (not via
    # brax.envs.get_environment) because this env is registered as 'ant' and
    # would shadow the stock entry in Brax's global registry.
    inner = Ant(backend=cfg.get('backend', 'mjx'))
    super().__init__(inner, ant_reward, cfg)
```

**3. Add it to the env dictionary.** Export the class from the subpackage's `__init__.py` and add
one entry to `ENV_REGISTRY` in `envs/__init__.py`:

```python
ENV_REGISTRY = {
    ...
    "ant": AntEnv,
}
```

**4. Add the config** at `../config/env/ant.yaml`. `env_name` must match the `ENV_REGISTRY` key;
`obs_history`/`act_history` are read by `DefaultInfopropWrappable`; per-env algorithm overrides
go under `algorithm:`:

```yaml
# @package _global_
env:
  name: Ant
  env_name: ant
  backend: mjx

  # History parameters for model learning.
  obs_history: 1
  act_history: 0

algorithm:
  target_entropy: -8   # = -action_dim for ant
  discounting: 0.99
```

**5. Run:**

```bash
python -m infoprop_jax.main env=ant
```

The startup checks (`validate_infoprop_contract`) run automatically.

## Walkthrough B - a full custom env (`InfopropWrappable`)

For envs that need their own physics, a control prior, or a model_state/context factorisation.
References: `wheelbot/wheelbot_brax_mjx.py` (variant/invariant split, control prior, history
inputs) and `humanoid/humanoid_mjx.py` (floating-base state, odometry context).

**1. Choose the state factorisation.** Decide which dims the ensemble predicts (`model_state`)
and which are reconstructed by integration (`context`). Context exists for dims the dynamics are
invariant to - typically world-frame pose: the model then generalises across the workspace and
model rollouts can start anywhere. Two reference designs:

* Humanoid: `model_state` = local floating-base representation (orientation, joint qpos, qvel in
  body frame), `context` = integrated odometry `[yaw, x, y]` (`context_size = 3`).
* Wheelbot: variant state `s^V` (11 dims) vs. invariant state `s^I` (5 dims: yaw, wheel angles,
  x, y) - see `wheelbot/README.md`.

Set `context_size = 0` if the NN predicts the whole next state (no `augment_prediction` needed).

**2. `__init__`.** Subclass `(PipelineEnv, InfopropWrappable)`, build the MJX `sys`, and set the
five required attributes:

```python
class MyEnv(PipelineEnv, InfopropWrappable):
  def __init__(self, cfg: DictConfig = DictConfig({}), eval_mode: bool = False, **kwargs):
    ...  # load mjcf, super().__init__(sys=..., backend='mjx', n_frames=...)
    self.model_state_size = ...
    self.context_size = ...
    self.full_state_size = self.model_state_size + self.context_size
    self.obs_history = cfg.get('obs_history', 1)
    self.act_history = cfg.get('act_history', 0)
    self.fast_model_rollout = cfg.get('fast_model_rollout', False)  # optional
```

**3. Real-env path.** `reset`/`step` implement the real physics as usual, and additionally keep
your env's `state.info` keys up to date each step: the current model state, the context, and the
input histories (use the base helpers `shift_phys`/`shift_action`). The invariant to keep: the
last history slot equals the current model state. See `WheelbotEnv.reset`/`step`.

**4. Implement the three required methods** (full semantics in the `infoprop_wrappable_env.py`
docstrings). One model step runs:

```
preprocess -> NN forward -> decode -> augment_prediction -> infoprop_core -> postprocess
   (env)      (InfopropEnv)  (InfopropEnv)    (env, opt.)     (InfopropEnv)     (env)
```

* `preprocess(state, action)` returns
  `(nn_input, curr_model_state, curr_context, applied_action, processed_action)`. Inject any
  control prior into `applied_action` (the action sent to the dynamics); `processed_action` is
  the action used for observation/reward (typically the clipped RL action).
* `postprocess(state, applied_action, next_model_state, next_context, processed_action)` rebuilds
  a valid Brax `State`: new observation, your env's `info` keys, reward, done. The rng and
  entropy-accumulation keys managed by the training code are already set when it is called.
* `reset_from_buffer(rng, init_transition)` starts a model rollout from a sampled real
  transition: `init_transition.observation` holds the model-state/action history, and
  `init_transition.extras['state_extras']` the context fields declared by
  `dummy_physics_transition`.

`postprocess` and `reset_from_buffer` must produce structurally identical States (consistent
`pipeline_state` present/`None` - the fast-rollout choice described above).

**5. `augment_prediction`** (must be overridden when `context_size > 0`): map per-member
`(mean, var)` from model-state space (`[E, model_state_size]`) to the full output space
(`[E, full_state_size]`) by appending the integrated context dims **and propagating their
variance** - this runs *before* ensemble fusion, so the appended dims take part in the Kalman
update and the entropy cutoffs. See `HumanoidEnv.augment_prediction` for the odometry integration
(yaw/x/y from body-frame velocities, with first-order variance propagation).

**6. Declare the buffer layout.**

* `dummy_physics_transition` - a zero-filled `Transition` that sizes the physics replay buffer:
  `observation` of shape `(model_state_size * obs_history + action_size * act_history,)`,
  `next_observation` of shape `(model_state_size,)`, and `state_extras` declaring `truncation`
  plus any per-transition context fields (e.g. the wheelbot's `invariant_physics_state`). This is
  the one place where the buffer layout is defined.
* `extract_physics_transition(prev_state, next_state, policy_extras)` - build the
  (history → next_model_state) transition from your env's `info` keys.
* `context_from_transition(transition)` - read the context vector back from buffer extras
  (override only when `context_size > 0`).
* `reset_carry_keys` - list the dynamic `info` keys of your env that the auto-reset wrappers must
  restore on `done` (model state, context, histories, applied action, …).

**7. Add to the env dictionary, configure, run** - same as Walkthrough A steps 3–5: export the
class, add the `ENV_REGISTRY` entry, create `../config/env/<name>.yaml` with a matching
`env_name`, and launch `python -m infoprop_jax.main env=<name>`. `validate_infoprop_contract`
will catch size mismatches, missing `truncation`, wrong return shapes, and
`postprocess`/`reset_from_buffer` structure divergence before any training starts.

## File map

| Path | Role |
|---|---|
| `infoprop_wrappable_env.py` | The required methods/attributes for new envs + generic history utilities. |
| `infoprop_env.py` | The generic Infoprop core wrapper. |
| `default_wrappable.py` | Ready-made compatibility class for stock Brax envs. |
| `contract_validation.py` | Startup checks (shape-level, via `jax.eval_shape`). |
| `wheelbot/` | Example env: simulator, trajectory logic, docs, and assets. |
| `humanoid/` | Example env adapted from the MuJoCo MJX tutorial, plus the racing variant (`humanoid_race_mjx.py`, `race_track.py`). |
| `quadruped/` | Stock Brax ant via `DefaultInfopropWrappable` (default-wrapping example). |
