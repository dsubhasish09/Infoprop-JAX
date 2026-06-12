"""InfopropWrappable: what a real MJX env must define to be Infoprop-wrappable.

`InfopropWrappable` is a small base class with no Brax functionality of its own: it only
declares the methods Infoprop calls on the env, plus shared history helpers. The usual
Brax env machinery (`reset`, `step`, `dt`, `action_size`, `observation_size`) comes from
a second base class, in one of two ways:

  * a hand-written env inherits from both: ``class MyEnv(PipelineEnv, InfopropWrappable)``,
    implementing its *real* physics on the `PipelineEnv` side and the methods below;
  * a class built around an existing stock Brax env inherits from
    ``(brax.envs.base.Wrapper, InfopropWrappable)`` — see
    `DefaultInfopropWrappable` in default_wrappable.py for the ready-made simple case
    (model state == observation, no context).

Either way the result can be turned into a learned-dynamics model environment:

    model_env = InfopropEnv(MyEnv(cfg), min_log_var=..., max_log_var=...)

`InfopropEnv` (see infoprop_env.py) owns the fixed Infoprop core math and calls these
methods on the wrapped env. One model step runs:

    preprocess -> NN forward -> decode -> augment_prediction -> infoprop_core -> postprocess
       (here)     (InfopropEnv)  (InfopropEnv)    (here, opt.)    (InfopropEnv)   (here)

`preprocess` also maps the RL action to the action applied to the dynamics (a control prior,
if any); `postprocess` rebuilds the State and reward. There are no separate "step half" methods.

State vectors:
  * model_state  (model_state_size)  - the dims the ensemble predicts deltas of.
  * context      (context_size)      - extra dims reconstructed by integration
                                       (set context_size == 0 if the NN predicts the
                                       whole next state directly).
  * full_state   (full_state_size = model_state_size + context_size) - the entropy /
                                       cutoff space.

Required attributes the subclass sets in __init__: `model_state_size`, `context_size`,
`full_state_size`, `obs_history`, `act_history`. (`dt` / `action_size` /
`observation_size` come from the Brax base class: `PipelineEnv` provides them directly,
`brax.envs.base.Wrapper` passes them through from the inner env.)

Fast model rollouts (skipping the MJX `pipeline_state` during model rollouts) are an
optional optimisation entirely inside your env: the training code and wrappers only react
to the *structure* of the State you return (`pipeline_state is None` => skipped). If you
want it, keep a flag (e.g. `self.fast_model_rollout`, read from cfg) and apply it
consistently in `postprocess` and `reset_from_buffer`.

`info` keys: the keys holding the model state, histories, context and task id belong to
*your environment* — name them whatever you like; only your methods touch them. Declare
the dynamic ones in `reset_carry_keys` and the per-transition context fields via
`dummy_physics_transition`.
"""

from typing import Dict, List, Tuple

import jax
from jax import numpy as jp
from brax.envs.base import State
from brax.training.types import Transition


class InfopropWrappable:
  """Declares the env-specific methods the Infoprop core needs.

  The standard Brax env machinery (`reset`, `step`, `dt`, `action_size`,
  `observation_size`) comes from a second base class — `PipelineEnv` (real physics)
  or `brax.envs.base.Wrapper` (building on an existing env). The default
  `augment_prediction` (identity) plus a `preprocess` that passes the action through
  make a plain env with `context_size == 0` work with no extra structure.
  """

  # --------------------------------------------------------------- required methods
  def preprocess(
      self, state: State, action: jp.ndarray
  ) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
    """Map a State + RL action to the model inputs and the actions to apply.

    Returns ``(nn_input, curr_model_state, curr_context, applied_action, processed_action)``:
      * ``nn_input`` is fed to the ensemble;
      * ``curr_model_state`` (shape ``(model_state_size,)``) is what the decoded delta
        is integrated onto;
      * ``curr_context`` (shape ``(context_size,)``, possibly empty) is the extra state
        ``augment_prediction`` needs;
      * ``applied_action`` is the action sent to the dynamics — inject any control
        prior here (default: just the RL action);
      * ``processed_action`` is the action used for observation/reward (typically the
        clipped RL action).
    """
    raise NotImplementedError

  def postprocess(self, state: State, applied_action: jp.ndarray,
                  next_model_state: jp.ndarray, next_context: jp.ndarray,
                  processed_action: jp.ndarray) -> State:
    """Rebuild a valid Brax State from a sampled next model state + context.

    Must set the new observation, this env's own `info` keys, reward and done. The
    rng + entropy-accumulation keys managed by the training code are already set on
    ``state`` before this is called.

    Building the MJX ``pipeline_state`` is the env's own choice (the "fast rollout"
    optimisation): set ``pipeline_state=None`` to skip it. Whatever you choose, the
    State structure must be consistent across ``postprocess`` and
    ``reset_from_buffer`` (they are carried together under ``scan``). A common
    pattern is to gate it on a ``self.fast_model_rollout`` flag of your env.
    """
    raise NotImplementedError

  def reset_from_buffer(self, rng: jax.Array,
                        init_transition: Transition) -> State:
    """Reset a model rollout from a sampled real-data physics transition.

    ``init_transition.observation`` is the model-state/action history;
    ``init_transition.extras['state_extras']`` holds the context fields declared by
    ``dummy_physics_transition``. As in ``postprocess``, whether to build the MJX
    ``pipeline_state`` is the env's own choice — but it must match the structure
    ``postprocess`` produces (consistent ``pipeline_state`` present/absent under scan).
    """
    raise NotImplementedError

  # --------------------------------------------------------------- optional methods
  def augment_prediction(
      self, member_mean: jp.ndarray, member_var: jp.ndarray,
      curr_model_state: jp.ndarray, curr_context: jp.ndarray
  ) -> Tuple[jp.ndarray, jp.ndarray]:
    """Map per-member ``(mean, var)`` in model-state space to full output space.

    Default identity: the NN predicts the entire next state (``context_size == 0``).
    Override to append integrated/derived dims and propagate their variance (must
    happen *before* fusion). Operates on a single rollout sample:
    ``member_mean``/``member_var`` are ``[E, model_state_size]``.
    """
    return member_mean, member_var

  # ------------------------------------------------------ buffer layout declarations
  @property
  def dummy_physics_transition(self) -> Transition:
    """Zero-filled `Transition` that sizes the physics replay buffer and, via its
    ``extras['state_extras']``, declares the per-transition context fields."""
    raise NotImplementedError

  def extract_physics_transition(self, prev_state: State, next_state: State,
                                 policy_extras: Dict[str, jp.ndarray]) -> Transition:
    """Build the (history -> next_model_state) transition with its context extras,
    from this env's own `info` keys, for the physics replay buffer."""
    raise NotImplementedError

  def context_from_transition(self, transition: Transition) -> jp.ndarray:
    """Return the context vector for a physics-buffer transition.

    The default supports the common ``context_size == 0`` case. Environments that
    use context should override this to read their own fields from
    ``transition.extras['state_extras']``.
    """
    return jp.zeros((self.context_size,))

  @property
  def reset_carry_keys(self) -> List[str]:
    """This env's dynamic info keys that the auto-reset wrappers restore on `done`."""
    raise NotImplementedError

  # ------------------------------------------------------------------- history utils
  def shift_action(self, curr_act_history: jp.ndarray, to_add: jp.ndarray) -> jp.ndarray:
    if self.act_history == 0:
      return curr_act_history
    return jp.concatenate([curr_act_history[self.action_size:], to_add])

  def shift_phys(self, curr_phys_history: jp.ndarray, to_add: jp.ndarray) -> jp.ndarray:
    if self.obs_history == 0:
      return curr_phys_history
    return jp.concatenate([curr_phys_history[self.model_state_size:], to_add])
