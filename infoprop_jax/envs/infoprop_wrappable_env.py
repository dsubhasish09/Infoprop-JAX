"""InfopropWrappable: the contract a real MJX env must satisfy to be Infoprop-wrappable.

A concrete environment subclasses this, implements the usual Brax `PipelineEnv`
methods (`reset`, `step`, observation, reward) with its *real* physics, and additionally
implements the Infoprop hooks below. It can then be turned into a learned-dynamics model
environment by wrapping it:

    model_env = InfopropEnv(MyEnv(cfg), min_log_var=..., max_log_var=..., fast_model_rollout=...)

`InfopropEnv` (see infoprop_env.py) owns the fixed Infoprop core math and calls these hooks
on the wrapped env. One model step runs:

    preprocess -> NN forward -> decode -> augment_prediction -> infoprop_core -> postprocess
       (here)     (InfopropEnv)  (InfopropEnv)    (here, opt.)    (InfopropEnv)   (here)

`preprocess` also maps the RL action to the action applied to the dynamics (a control prior,
if any); `postprocess` rebuilds the State and reward. There are no separate "step half" hooks.

State vectors:
  * model_state  (model_state_size)  - the dims the ensemble predicts deltas of.
  * context      (context_size)      - extra dims reconstructed by integration
                                       (set context_size == 0 if the NN predicts the
                                       whole next state directly).
  * full_state   (full_state_size = model_state_size + context_size) - the entropy /
                                       cutoff space.

Required attributes the subclass sets in __init__: `model_state_size`, `context_size`,
`full_state_size`, `obs_history`, `act_history`. (`dt` / `action_size` come from PipelineEnv.)

`info`-key ownership: keys holding the model state, histories, context and task id are
*env-owned* — name them whatever you like; only your hooks touch them. Declare the dynamic
ones in `reset_carry_keys` and the per-transition context fields via `dummy_physics_transition`.
"""

import jax
from jax import numpy as jp
from brax.envs.base import PipelineEnv, State


class InfopropWrappable(PipelineEnv):
  """Base class declaring the env-specific hooks the Infoprop core needs.

  Subclasses additionally implement the standard `PipelineEnv` interface (`reset`,
  `step`) with their real physics. The default `augment_prediction` (identity) plus a
  `preprocess` that passes the action through make a plain env with `context_size == 0`
  work with no extra structure.
  """

  # ----------------------------------------------------------------- required hooks
  def preprocess(self, state: State, action: jp.ndarray):
    """Map a State + RL action to the model inputs and the actions to apply.

    Returns ``(nn_input, curr_model_state, curr_context, applied_action, processed_action)``:
      * ``nn_input`` is fed to the ensemble;
      * ``curr_model_state`` (shape ``(model_state_size,)``) is what the decoded delta
        is integrated onto;
      * ``curr_context`` (shape ``(context_size,)``, possibly empty) is the extra state
        the ``augment_prediction`` hook needs;
      * ``applied_action`` is the action sent to the dynamics — inject any control
        prior here (default: just the RL action);
      * ``processed_action`` is the action used for observation/reward (typically the
        clipped RL action).
    """
    raise NotImplementedError

  def postprocess(self, state, applied_action, next_model_state, next_context,
                  processed_action, build_pipeline_state):
    """Rebuild a valid Brax State from a sampled next model state + context.

    Must set the new observation, env-owned `info`, reward and done. Build the MJX
    ``pipeline_state`` only if ``build_pipeline_state`` is True (it is False during
    fast model rollouts). The framework-owned rng + entropy accumulation are already
    set on ``state`` before this is called.
    """
    raise NotImplementedError

  def reset_from_buffer(self, rng, init_transition, build_pipeline_state):
    """Reset a model rollout from a sampled real-data physics transition.

    ``init_transition.observation`` is the model-state/action history;
    ``init_transition.extras['state_extras']`` holds the context fields declared by
    ``dummy_physics_transition``. Build the MJX ``pipeline_state`` only if
    ``build_pipeline_state`` is True (matches the step's structure under scan).
    """
    raise NotImplementedError

  def _get_obs(self, *args, **kwargs):
    raise NotImplementedError

  def _get_rew(self, state, action):
    raise NotImplementedError

  # ------------------------------------------------------------- optional hooks
  def augment_prediction(self, member_mean, member_var, curr_model_state, curr_context):
    """Map per-member ``(mean, var)`` in model-state space to full output space.

    Default identity: the NN predicts the entire next state (``context_size == 0``).
    Override to append integrated/derived dims and propagate their variance (must
    happen *before* fusion). Operates on a single rollout sample (the framework vmaps
    it over the batch): ``member_mean``/``member_var`` are ``[E, model_state_size]``,
    ``curr_model_state`` ``[model_state_size]``, ``curr_context`` ``[context_size]``.
    """
    return member_mean, member_var

  # ------------------------------------------------------------- data contract
  @property
  def dummy_physics_transition(self):
    """Zero-filled `Transition` that sizes the physics replay buffer and, via its
    ``extras['state_extras']``, declares the per-transition context fields."""
    raise NotImplementedError

  def extract_physics_transition(self, prev_state, next_state, policy_extras):
    """Build the (history -> next_model_state) transition with its context extras,
    from this env's own `info` keys, for the physics replay buffer."""
    raise NotImplementedError

  @property
  def reset_carry_keys(self):
    """Env-owned dynamic info keys the auto-reset wrappers revert on `done`."""
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
