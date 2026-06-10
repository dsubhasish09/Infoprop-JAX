"""InfopropWrappable: the contract a real MJX env must satisfy to be Infoprop-wrappable.

A concrete environment subclasses this, implements the usual Brax `PipelineEnv`
methods (`reset`, `step`, observation, reward) with its *real* physics, and additionally
implements the Infoprop hooks below. It can then be turned into a learned-dynamics model
environment by wrapping it:

    model_env = InfopropEnv(MyEnv(cfg), min_log_var=..., max_log_var=...)

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

Fast model rollouts (skipping the MJX `pipeline_state` during model rollouts) are an
optional, purely env-side optimisation: the framework and wrappers are agnostic to it and
only react to the *structure* of the State you emit (`pipeline_state is None` => skipped).
If you want it, own a flag (e.g. `self.fast_model_rollout`, read from cfg) and gate
`pipeline_state` construction on it consistently in `postprocess` and `reset_from_buffer`.

`info`-key ownership: keys holding the model state, histories, context and task id are
*env-owned* — name them whatever you like; only your hooks touch them. Declare the dynamic
ones in `reset_carry_keys` and the per-transition context fields via `dummy_physics_transition`.
"""

from typing import Dict, List, Tuple

import jax
from jax import numpy as jp
from brax.envs.base import PipelineEnv, State
from brax.training.types import Transition


class InfopropWrappable(PipelineEnv):
  """Base class declaring the env-specific hooks the Infoprop core needs.

  Subclasses additionally implement the standard `PipelineEnv` interface (`reset`,
  `step`) with their real physics. The default `augment_prediction` (identity) plus a
  `preprocess` that passes the action through make a plain env with `context_size == 0`
  work with no extra structure.
  """

  # ----------------------------------------------------------------- required hooks
  def preprocess(
      self, state: State, action: jp.ndarray
  ) -> Tuple[jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray, jp.ndarray]:
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

  def postprocess(self, state: State, applied_action: jp.ndarray,
                  next_model_state: jp.ndarray, next_context: jp.ndarray,
                  processed_action: jp.ndarray) -> State:
    """Rebuild a valid Brax State from a sampled next model state + context.

    Must set the new observation, env-owned `info`, reward and done. The
    framework-owned rng + entropy accumulation are already set on ``state`` before
    this is called.

    Building the MJX ``pipeline_state`` is the env's own choice (the "fast rollout"
    optimisation): set ``pipeline_state=None`` to skip it. Whatever you choose, the
    State structure must be consistent across ``postprocess`` and
    ``reset_from_buffer`` (they are carried together under ``scan``). A common
    pattern is to gate it on an env-owned ``self.fast_model_rollout`` flag.
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

  def _get_obs(self, *args, **kwargs) -> jp.ndarray:
    raise NotImplementedError

  def _get_rew(
      self, state: State, action: jp.ndarray
  ) -> Tuple[jp.ndarray, jp.ndarray, Dict[str, jp.ndarray]]:
    """Compute the step reward. Returns ``(reward, done, reward_metrics)``."""
    raise NotImplementedError

  # ------------------------------------------------------------- optional hooks
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

  # ------------------------------------------------------------- data contract
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
    use context should override this to read their env-owned fields from
    ``transition.extras['state_extras']``.
    """
    return jp.zeros((self.context_size,))

  @property
  def reset_carry_keys(self) -> List[str]:
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
