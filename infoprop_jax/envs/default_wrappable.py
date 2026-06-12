# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""DefaultInfopropWrappable: make a stock Brax env Infoprop-compatible.

Wraps any existing Brax `PipelineEnv` (ant, humanoid, ...) and defines all the
`InfopropWrappable` methods for the simple case:

  * model state == the env's observation (``model_state_size = observation_size``);
  * no context (``context_size == 0``, identity ``augment_prediction``);
  * NN input = concatenated observation/action histories.

Real-env reward and termination come from the wrapped env unchanged. In *imagined*
rollouts there is no physics ``pipeline_state``, so the stock reward logic cannot run;
instead a constructor-supplied ``reward_fn(obs, action, next_obs) -> (reward, done)``
is used (pure jax, per-sample — it runs inside the vmapped model step). Subclasses may
override `_get_rew` instead of passing a ``reward_fn``.

Model rollouts always emit ``pipeline_state=None`` (there is no general inverse map
from observation to qpos/qvel), so model-env video rendering is not supported; the
real-env path keeps the inner pipeline state and is unaffected.

Note: the Infoprop methods are defined directly on this class, so `Wrapper.__getattr__`
never falls through to the inner env for them. Wrapping an env that itself defines the
Infoprop methods will silently ignore that env's versions — subclass that env
instead of wrapping it here.
"""

from typing import Callable, Tuple

import jax
from jax import numpy as jp
from brax.envs.base import Env, State, Wrapper
from brax.training.types import Transition
from omegaconf import DictConfig

from infoprop_jax.envs.infoprop_wrappable_env import InfopropWrappable

RewardFn = Callable[[jp.ndarray, jp.ndarray, jp.ndarray],
                    Tuple[jp.ndarray, jp.ndarray]]


class DefaultInfopropWrappable(Wrapper, InfopropWrappable):
  """Make a stock Brax env Infoprop-wrappable with model state == observation."""

  def __init__(self, env: Env, reward_fn: RewardFn,
               cfg: DictConfig = DictConfig({})):
    # Must run first: Wrapper.__getattr__ recurses on any attribute access
    # before self.env is set.
    super().__init__(env)
    if not isinstance(env.observation_size, int):
      raise ValueError(
          'DefaultInfopropWrappable requires a flat observation; got '
          f'observation_size={env.observation_size!r} (dict observations are '
          'unsupported).')
    self._reward_fn = reward_fn
    self.obs_history = cfg.get('obs_history', 1)
    self.act_history = cfg.get('act_history', 0)
    self.model_state_size = env.observation_size
    self.context_size = 0
    self.full_state_size = self.model_state_size

  # ------------------------------------------------------------- real env path
  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    info = dict(state.info)
    info['physics_state'] = state.obs
    info['prev_physics_state'] = state.obs
    info['applied_action'] = jp.zeros(self.action_size)
    # Tiling the initial obs (rather than zero-fill + warmup steps) satisfies the
    # invariant "last history slot == current model state" immediately; exact for
    # obs_history == 1.
    info['phys_state_history'] = jp.tile(state.obs, self.obs_history)
    info['act_history'] = jp.zeros(self.action_size * self.act_history)
    return state.replace(info=info)

  def step(self, state: State, action: jp.ndarray) -> State:
    prev_obs = state.obs
    nstate = self.env.step(state, action)
    info = dict(nstate.info)
    info['prev_physics_state'] = prev_obs
    info['physics_state'] = nstate.obs
    info['applied_action'] = action
    info['phys_state_history'] = self.shift_phys(
        info['phys_state_history'], nstate.obs)
    info['act_history'] = self.shift_action(info['act_history'], action)
    return nstate.replace(info=info)

  # ----------------------------------------------------------- Infoprop methods
  def preprocess(self, state: State, action: jp.ndarray):
    nn_input = jp.concatenate(
        [state.info['phys_state_history'], state.info['act_history']], axis=-1)
    return (nn_input, state.info['physics_state'], jp.zeros((0,)), action,
            action)

  def postprocess(self, state: State, applied_action: jp.ndarray,
                  next_model_state: jp.ndarray, next_context: jp.ndarray,
                  processed_action: jp.ndarray) -> State:
    info = dict(state.info)
    info['prev_physics_state'] = info['physics_state']
    info['physics_state'] = next_model_state
    info['applied_action'] = applied_action
    info['phys_state_history'] = self.shift_phys(
        info['phys_state_history'], next_model_state)
    info['act_history'] = self.shift_action(info['act_history'],
                                            applied_action)
    state = state.replace(
        pipeline_state=None, obs=next_model_state, info=info)
    reward, done, _ = self._get_rew(state, processed_action)
    return state.replace(reward=reward, done=done)

  def reset_from_buffer(self, rng: jax.Array,
                        init_transition: Transition) -> State:
    ms = self.model_state_size
    init_history = init_transition.observation
    phys_history = init_history[:ms * self.obs_history]
    act_history = init_history[ms * self.obs_history:]
    physics_state = phys_history[-ms:]
    info = {
        'physics_state': physics_state,
        'prev_physics_state': physics_state,
        'applied_action': jp.zeros(self.action_size),
        'phys_state_history': phys_history,
        'act_history': act_history,
        'accumulated_conditional_entropy': jp.zeros((self.full_state_size,)),
        'current_conditional_entropy': jp.zeros((self.full_state_size,)),
    }
    return State(None, physics_state, jp.zeros(()), jp.zeros(()), {}, info)

  def _get_rew(self, state: State, action: jp.ndarray):
    reward, done = self._reward_fn(state.info['prev_physics_state'], action,
                                   state.info['physics_state'])
    return reward, done, {}

  # -------------------------------------------------- buffer layout declarations
  @property
  def dummy_physics_transition(self) -> Transition:
    ms, oh, ah = self.model_state_size, self.obs_history, self.act_history
    return Transition(
        observation=jp.zeros(ms * oh + self.action_size * ah),
        action=jp.zeros(self.action_size),
        reward=0.0,
        discount=0.0,
        next_observation=jp.zeros(ms),
        extras={
            'state_extras': {'truncation': 0.0},
            'policy_extras': {},
        },
    )

  def extract_physics_transition(self, prev_state: State, next_state: State,
                                 policy_extras) -> Transition:
    return Transition(
        observation=jp.concatenate(
            [prev_state.info['phys_state_history'],
             prev_state.info['act_history']], axis=-1),
        action=next_state.info['applied_action'],
        reward=jp.zeros(next_state.reward.shape, dtype=jp.float32),
        discount=1 - next_state.done,
        next_observation=next_state.info['physics_state'],
        extras={
            'policy_extras': policy_extras,
            'state_extras': {
                'truncation': next_state.info.get('truncation', 0.0),
            },
        },
    )

  @property
  def reset_carry_keys(self):
    return [
        'physics_state',
        'prev_physics_state',
        'applied_action',
        'phys_state_history',
        'act_history',
    ]
