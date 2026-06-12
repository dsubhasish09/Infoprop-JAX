# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""Ant: the stock Brax quadruped adapted via `DefaultInfopropWrappable`.

The example of wrapping a preexisting Brax env without writing any Infoprop methods:
model state == observation, no context, and an observation-based reward for imagined
rollouts (the stock reward reads the physics ``pipeline_state``, which model rollouts
don't have). Real-env reward and termination come from the stock env unchanged.
"""

from brax.envs.ant import Ant
from jax import numpy as jp
from omegaconf import DictConfig

from infoprop_jax.envs.default_wrappable import DefaultInfopropWrappable


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
