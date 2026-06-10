# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""
Infoprop Dyna training loop for the Mini Wheelbot racing task.

Implements model-based RL combining:
  - A probabilistic ensemble dynamics model trained via negative log-likelihood.
  - InfoProp uncertainty quantification: Kalman-filtered rollout steps with
    per-step and accumulated entropy-based termination thresholds.
  - Soft Actor-Critic (SAC) policy optimization on synthetic model rollouts.
  - Episodic resampling of rollout initial states from the real-data replay buffer.
"""

import functools
import os
import time
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
from absl import logging
from brax import base
from brax import envs
from brax import envs as ENV
from brax.io import model as brax_model
from brax.training import gradients
from brax.training import replay_buffers
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from infoprop_jax.algorithms.util.agent_learning import sac_losses
from infoprop_jax.algorithms.util.agent_learning import sac_networks
from brax.training.types import Params
from brax.training.types import Policy
from brax.training.types import PRNGKey
from flax.training.train_state import TrainState

from infoprop_jax.algorithms.util.custom_evaluator import CustomEvaluator
from infoprop_jax.algorithms.util.custom_wrapper import wrap_custom, wrap
from infoprop_jax.algorithms.util.model_learning.model_dataset import ReplayBufferPhysicsState
from infoprop_jax.algorithms.util.model_learning.model_trainer import compute_loss
from infoprop_jax.envs.infoprop_env import InfopropEnv as Model_Wheelbot
from infoprop_jax.envs.wheelbot.wheelbot_brax_mjx import Wheelbot as Real_Wheelbot

State = envs.State
Env = envs.Env
Metrics = types.Metrics
Transition = types.Transition

ReplayBufferState = Any

# Physics-state column indices shared by get_cutoffs and the model environments
_EULER_RATE_IDX = jnp.array([2, 5, 6])   # roll_dot, pitch_dot, yaw_dot
_BODY_VEL_IDX   = jnp.array([7, 8, 9])  # body vx, vy, vz

@flax.struct.dataclass
class TrainingState:
  """Flax struct holding all mutable training state.

  Bundles SAC network parameters (policy, Q, alpha), the learned dynamics model,
  running-statistics normalizer, and normalisation constants for model inputs/outputs.
  Kept on a single device.
  """

  policy_optimizer_state: optax.OptState
  policy_params: Params
  q_optimizer_state: optax.OptState
  q_params: Params
  target_q_params: Params
  gradient_steps: jnp.ndarray
  env_steps: jnp.ndarray
  alpha_optimizer_state: optax.OptState
  alpha_params: Params
  normalizer_params: running_statistics.RunningStatisticsState
  model_state: TrainState
  model_obs_mean: jax.Array
  model_obs_std: jax.Array
  next_state_delta_mean: jnp.ndarray 
  next_state_delta_std: jnp.ndarray

def Rx(theta):
    """
    Rotation matrix around x-axis.
    """
    return jnp.array([[1, 0, 0],
                      [0, jnp.cos(theta), -jnp.sin(theta)],
                      [0, jnp.sin(theta), jnp.cos(theta)]])

def Ry(theta):
    """
    Rotation matrix around y-axis.
    """
    return jnp.array([[jnp.cos(theta), 0, jnp.sin(theta)],
                      [0, 1, 0],
                      [-jnp.sin(theta), 0, jnp.cos(theta)]])

def Rz(theta):
    """
    Rotation matrix around z-axis.
    """
    return jnp.array([[jnp.cos(theta), -jnp.sin(theta), 0],
                      [jnp.sin(theta), jnp.cos(theta), 0],
                      [0, 0, 1]])

def YRP(yaw, roll, pitch):
   """ 
   Combined rotation matrix for yaw, roll, and pitch.
   """
   return Rz(yaw) @ Rx(roll) @ Ry(pitch)

def _unpmap(v):
    """No-op: retained for structural symmetry; there is no device axis to strip."""
    return v

def tree_where(condition, v1, v2):
    """
    Returns a pytree with the same structure as v1 and v2, where each leaf
    is the result of jnp.where(condition, v1, v2).
    """
    return  jax.tree_util.tree_map(
        lambda x, y: jnp.where(condition, x, y), v1, v2
    )

def tree_repeat(v, n):
   """
   Repeats the elements of a pytree n times.
   """
   return jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), n, axis=0), v)

def _init_training_state(
    key: PRNGKey,
    obs_size: int,
    local_devices_to_use: int,
    sac_network: sac_networks.SACNetworks,
    alpha_optimizer: optax.GradientTransformation,
    policy_optimizer: optax.GradientTransformation,
    q_optimizer: optax.GradientTransformation,
    model_state,
    model_obs_mean: jax.Array,
    model_obs_std: jax.Array,
    next_state_delta_mean: jnp.ndarray,
    next_state_delta_std: jnp.ndarray,
    policy_params: Optional[Params] = None,
    normalizer_params: Optional[running_statistics.RunningStatisticsState] = None,
    initial_log_alpha: float = 0.0,
) -> TrainingState:
  """Initialise the full TrainingState for a single-device JIT training loop."""
  key_policy, key_q = jax.random.split(key)
  log_alpha = jnp.asarray(initial_log_alpha, dtype=jnp.float32)
  alpha_optimizer_state = alpha_optimizer.init(log_alpha)

  policy_params = sac_network.policy_network.init(key_policy) if policy_params is None else policy_params
  policy_optimizer_state = policy_optimizer.init(policy_params)
  q_params = sac_network.q_network.init(key_q)
  q_optimizer_state = q_optimizer.init(q_params)

  normalizer_params = running_statistics.init_state(
      specs.Array((obs_size,), jnp.dtype('float32'))
  ) if normalizer_params is None else normalizer_params

  training_state = TrainingState(
      policy_optimizer_state=policy_optimizer_state,
      policy_params=policy_params,
      q_optimizer_state=q_optimizer_state,
      q_params=q_params,
      target_q_params=q_params,
      gradient_steps=jnp.zeros(()),
      env_steps=jnp.zeros(()),
      alpha_optimizer_state=alpha_optimizer_state,
      alpha_params=log_alpha,
      normalizer_params=normalizer_params,
        model_state=model_state,
        model_obs_mean=model_obs_mean,
        model_obs_std=model_obs_std,
        next_state_delta_mean=next_state_delta_mean,
        next_state_delta_std=next_state_delta_std,
  )
  return training_state

def train(
    environment,
    model_environment,
    episode_length: int,
    wrap_env: bool = True,
    wrap_env_fn: Optional[Callable[[Any], Any]] = None,
    action_repeat: int = 1,
    num_envs: int = 1,
    num_real_envs: int = 1,
    num_real_eval_envs: int = 1000,
    model_learning_rate: float = 1e-3,
    model_weight_decay: float = 1e-4,
    agent_learning_rate: float = 1e-4,
    discounting: float = 0.9,
    seed: int = 0,
    model_batch_size: int = 256,
    agent_batch_size: int = 256,
    num_trials: int = 1,
    normalize_observations: bool = False,
    max_devices_per_host: Optional[int] = None,
    reward_scaling: float = 1.0,
    tau: float = 0.005,
    min_physics_replay_size: int = 0,
    max_physics_replay_size: Optional[int] = None,
    min_model_replay_size: int = 0,
    max_model_replay_size: Optional[int] = None,
    grad_updates_per_model_step: int = 1,
    num_resampling_epochs: int = 10,
    num_training_steps_per_model_train: int = 100,
    network_factory: types.NetworkFactory[
        sac_networks.SACNetworks
    ] = sac_networks.make_sac_networks,
    checkpoint_logdir: Optional[str] = None,
    agent_dir: Optional[str] = None,
    model_dir: Optional[str] = None,
    agent_hidden_layer_sizes: Tuple[int] = (256, 256),
    model_hidden_layer_sizes: Tuple[int] = (256, 256),
    target_entropy: Optional[float] = None,
    min_log_var: float = -4,
    max_log_var: float = -2,
    max_rollout_length: int = 1000,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.95,
    patience: int = 10,
    model_layer_norm: bool = True,
    agent_layer_norm: bool = True,
    eval_environment = None,
    progress_fn = None,
    randomization_fn: Optional[
        Callable[[base.System, jnp.ndarray], Tuple[base.System, base.System]]
    ] = None,
    obs_history: int = 1,
    act_history: int = 0,
    env_cfg = None,
    tune_entropy: bool = True,
    alpha: float = 0.1,
    reset_agent_per_trial: bool = False,
    reset_model_replay_buffer: bool = False,
    reset_model_per_trial: bool = False,
    fast_model_rollout: bool = True,
):
  """Main Infoprop Dyna training loop.

  Alternates between:
    1. Model training: fit the probabilistic ensemble on real-data transitions.
    2. Cutoff computation: derive per-step (lambda_1) and accumulated (lambda_2)
       entropy thresholds from the training buffer.
    3. Agent training: run SAC updates on model-generated rollouts, with episodic
       resampling of initial states from the real-data buffer.
    4. Real-world data collection: step the MJX environment with the current policy.

  Args:
      environment: Registered Brax environment name for real (MJX) rollouts.
      model_environment: Registered Brax environment name for model rollouts.
      ... (remaining args are passed through from Hydra config)
  """
#   jax.config.update("jax_log_compiles", True)
  process_id = jax.process_index()
  local_devices_to_use = 1
  device_count = 1
  logging.info(
      'single-device mode; local_device_count: %s; total_device_count: %s',
      local_devices_to_use,
      device_count,
  )

  # The number of environment steps executed for every `actor_step()` call.
  env_steps_per_actor_step = action_repeat * num_envs
  # equals to ceil(min_replay_size / env_steps_per_actor_step)
  num_prefill_actor_steps = -(-min_model_replay_size // num_envs)
  num_prefill_real_actor_steps = -(-min_physics_replay_size // num_real_envs)
 
  rng = jax.random.PRNGKey(seed)
  rng, key = jax.random.split(rng)
  key, evaluator_key = jax.random.split(key)

  # real environment
  env = environment
  if wrap_env:
    if wrap_env_fn is not None:
      wrap_for_training = wrap_env_fn
    elif isinstance(env, envs.Env):
      wrap_for_training = wrap
    else:
      wrap_for_training = wrap

    v_randomization_fn = None
    if randomization_fn is not None:
      v_randomization_fn = functools.partial(
          randomization_fn,
          rng=jax.random.split(
              key, num_envs
          ),
      )

    env = wrap_for_training(
        env,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )  # pytype: disable=wrong-keyword-args
    eval_env = envs.training.wrap(
        eval_environment,
        episode_length=episode_length,
        action_repeat=action_repeat,
        randomization_fn=v_randomization_fn,
    )

  # Initializing agent
  obs_size = env.observation_size
  if isinstance(obs_size, Dict):
    raise NotImplementedError('Dictionary observations not implemented in SAC')
  action_size = env.action_size

  normalize_fn = lambda x, y: x
  if normalize_observations:
    normalize_fn = running_statistics.normalize
  sac_network = network_factory(
      observation_size=obs_size,
      action_size=action_size,
      preprocess_observations_fn=normalize_fn,
      hidden_layer_sizes = agent_hidden_layer_sizes,
      # TODO Need to put these in config
      policy_network_layer_norm=agent_layer_norm,
        q_network_layer_norm=agent_layer_norm,
  )
  make_policy = sac_networks.make_inference_fn(sac_network)

  evaluator = CustomEvaluator(
      eval_env,
      functools.partial(make_policy, deterministic=True),
      num_eval_envs=num_real_eval_envs,
      episode_length=episode_length,
      action_repeat=action_repeat,
      key=evaluator_key,
  )

  alpha_optimizer = optax.adam(learning_rate=3e-4)

  policy_optimizer = optax.adam(learning_rate=agent_learning_rate)
  q_optimizer = optax.adam(learning_rate=agent_learning_rate)

  alpha_loss, critic_loss, actor_loss = sac_losses.make_losses(
      sac_network=sac_network,
      reward_scaling=reward_scaling,
      discounting=discounting,
      action_size=action_size,
      target_entropy=target_entropy,
  )
  alpha_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      alpha_loss, alpha_optimizer, pmap_axis_name=None, has_aux=True
  )
  critic_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      critic_loss, q_optimizer, pmap_axis_name=None
  )
  actor_update = gradients.gradient_update_fn(  # pytype: disable=wrong-arg-types  # jax-ndarray
      actor_loss, policy_optimizer, pmap_axis_name=None
  )
  
  # Initializing replay buffers
  dummy_obs = jnp.zeros((obs_size,))
  dummy_action = jnp.zeros((action_size,))
  dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=dummy_obs,
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.0, 'track_seed': 0,'invariant_physics_state':jnp.zeros((5,))}, 'policy_extras': {}},
  )
  # model replay buffer
  model_replay_buffer =  replay_buffers.UniformSamplingQueue(
      max_replay_size=max_model_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=agent_batch_size * grad_updates_per_model_step * num_training_steps_per_model_train // device_count,
  )
  # real replay buffer
  replay_buffer = replay_buffers.UniformSamplingQueue(
      max_replay_size=max_physics_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=agent_batch_size * grad_updates_per_model_step // device_count,
  )


  # need to make this adaptive
  #physics replay buffer
  dummy_obs_history = jnp.zeros((11 * obs_history,))
  dummy_obs = jnp.zeros((11,))
  dummy_action_history = jnp.zeros((2 * act_history,))
  dummy_action = jnp.zeros((2,))
  dummy_transition = Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
      observation=jnp.concatenate([dummy_obs_history, dummy_action_history], axis=-1),
      action=dummy_action,
      reward=0.0,
      discount=0.0,
      next_observation=dummy_obs,
      extras={'state_extras': {'truncation': 0.0, 'track_seed': 0, 'invariant_physics_state':jnp.zeros((5,))}, 'policy_extras': {}},
  )
  replay_buffer_physics_state = ReplayBufferPhysicsState(
      max_replay_size=max_physics_replay_size // device_count,
      dummy_data_sample=dummy_transition,
      sample_batch_size=model_batch_size * grad_updates_per_model_step // device_count,
  )
  
  # initialize model trainer
  model_trainer = model_environment.init_NN_trainer(
        seed=seed,
        learning_rate=model_learning_rate,
        weight_decay=model_weight_decay,
        hidden_layer_sizes=model_hidden_layer_sizes,
        model_layer_norm=model_layer_norm,
    )
  

  global_key, local_key = jax.random.split(rng)
  local_key = jax.random.fold_in(local_key, process_id)

  model_state, _, model_obs_mean, model_obs_std, next_state_delta_mean, next_state_delta_std, local_key = model_trainer.init(local_key)
  _initial_model_state = model_state
  # Store the apply_fn on the env instance; it's a static function reference
  # that never changes, so it must not be carried in the scan state.
  model_environment._model_apply_fn = model_state.apply_fn


  def model_sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    """Performs a single step of SGD on the model"""
    training_state, key = carry
    model_ = training_state.model_state
    model_obs_mean = training_state.model_obs_mean
    model_obs_std = training_state.model_obs_std
    next_state_delta_mean = training_state.next_state_delta_mean
    next_state_delta_std = training_state.next_state_delta_std

    curr_rng, key = jax.random.split(key, 2)
    model_, model_obs_mean, model_obs_std, next_state_delta_mean, next_state_delta_std, _ = model_trainer.update_step(
                transitions,
                model_,
                model_obs_mean,
                model_obs_std,
                next_state_delta_mean,
                next_state_delta_std,
            obs_history,
            act_history,
            11,
            2,
                curr_rng,
            )

    metrics = {
    }

    new_training_state = training_state.replace(
      model_state=model_,
      model_obs_mean=model_obs_mean,
      model_obs_std=model_obs_std,
        next_state_delta_mean=next_state_delta_mean,
        next_state_delta_std=next_state_delta_std,
    )
    return (new_training_state, key), metrics

  def update_stats(training_state, full_training_transitions):
    """Update running normalisation statistics for observations and next-state deltas."""
    model_dataset_train = full_training_transitions
    obs = model_dataset_train.observation
    next_obs = model_dataset_train.next_observation
    action = model_dataset_train.action
    model_inp = jnp.concatenate((obs, action), axis=-1)
    model_obs_mean = jnp.mean(model_inp, axis=0)
    model_obs_std = jnp.std(model_inp, axis=0)
    curr_obs = obs[:, (obs_history -1) * 11 : obs_history *11]
    target = (next_obs - curr_obs) / 0.006
    next_state_delta_mean = jnp.mean(target, axis=0)
    next_state_delta_std = jnp.std(target, axis=0)

    training_state = training_state.replace(
        model_obs_mean=model_obs_mean,
        model_obs_std=model_obs_std,
        next_state_delta_mean=next_state_delta_mean,
        next_state_delta_std=next_state_delta_std,
    )
    return training_state
  update_stats = jax.jit(update_stats)
      

  def model_training_step(
      training_state: TrainingState,
      training_transitions: Transition,
      validation_transitions: Transition,
      key: PRNGKey,
      best_training_state: TrainingState ,
      loss: float,
      steps_since_last_improvement: int,
  ) -> Tuple[
      TrainingState,
      Union[envs.State, envs.State],
      ReplayBufferState,
      Metrics,
  ]:
    """Run one full epoch of ensemble model SGD with early stopping (patience-based)."""
    model_dataset_train = training_transitions
    model_dataset_val = validation_transitions
    
    model_dataset_train_ = model_dataset_train

    def f(carry, unused):
        """Performs a single step of training."""
        training_state, best_training_state, key, loss, steps_since_last_improvement = carry
        # an epoch of model training
        key, sgd_key = jax.random.split(key)
        (training_state, _), _ = jax.lax.scan(
            model_sgd_step, (training_state, sgd_key), model_dataset_train_
        )

        model_ = training_state.model_state
        model_obs_mean = training_state.model_obs_mean
        model_obs_std = training_state.model_obs_std
        next_state_delta_mean = training_state.next_state_delta_mean
        next_state_delta_std = training_state.next_state_delta_std
        key, loss_key = jax.random.split(key)
        loss_ = compute_loss(
            model_,
            model_dataset_val,
            model_obs_mean,
            model_obs_std,
            next_state_delta_mean,
            next_state_delta_std,
            obs_history,
            act_history,
            11,
            2,
            loss_key,
        )
        improved = loss_ < loss
        # print(best_training_state.model_state.step.shape)
        loss = jnp.where(improved, loss_, loss)
        steps_since_last_improvement = jnp.where(improved, 0, steps_since_last_improvement + 1)
        best_model_params = tree_where(
            improved, training_state.model_state.params, best_training_state.model_state.params)
        best_model_obs_mean = jnp.where(    
            improved, training_state.model_obs_mean, best_training_state.model_obs_mean)    
        best_model_obs_std = jnp.where(
            improved, training_state.model_obs_std, best_training_state.model_obs_std)
        best_model_next_state_delta_mean = jnp.where(
            improved, training_state.next_state_delta_mean, best_training_state.next_state_delta_mean)
        best_model_next_state_delta_std = jnp.where(
            improved, training_state.next_state_delta_std, best_training_state.next_state_delta_std)
        
        best_model_state = training_state.model_state.replace(params = best_model_params)

        best_training_state = best_training_state.replace(
            model_state=best_model_state,
            model_obs_mean=best_model_obs_mean,
            model_obs_std=best_model_obs_std,
            next_state_delta_mean=best_model_next_state_delta_mean,
            next_state_delta_std=best_model_next_state_delta_std,
        )
        
        return (training_state, best_training_state, key, loss, steps_since_last_improvement), loss_

    (training_state, best_training_state, key, loss, steps_since_last_improvement), losses = jax.lax.scan(f, (training_state, best_training_state, key, loss, steps_since_last_improvement), (), length=patience)


    return training_state, best_training_state, loss, steps_since_last_improvement, losses
   
  model_training_step = jax.jit(model_training_step)


  def run_model_eval(
        training_state: TrainingState,
        val_transitions: Transition,
        key: PRNGKey,
  ):
    """Evaluate the current ensemble on a held-out validation split."""
    model_dataset_val = val_transitions
    model_ = training_state.model_state
    model_obs_mean = training_state.model_obs_mean
    model_obs_std = training_state.model_obs_std
    next_state_delta_mean = training_state.next_state_delta_mean
    next_state_delta_std = training_state.next_state_delta_std
    loss = compute_loss(
        model_,
        model_dataset_val,
        model_obs_mean,
        model_obs_std,
        next_state_delta_mean,
        next_state_delta_std,
        obs_history,
        act_history,
        11,
        2,
        key,
    )
    return training_state, loss
  run_model_eval = jax.jit(run_model_eval)

  def model_training_loop(
      training_state: TrainingState,
      training_transitions: Transition,
        validation_transitions: Transition,
      key: PRNGKey,
      max_iterations: int = 100,
  ) -> Tuple[TrainingState, ReplayBufferState]:
    """Orchestrate model training epochs, tracking the best validation loss."""

    key, eval_key = jax.random.split(key)
    training_state, loss = run_model_eval(training_state, validation_transitions, eval_key)
    logging.info('Initial validation loss: %s', loss)
    best_training_state, loss, steps_since_last_improvement = training_state, loss, jnp.array(0)

    for i in range(max_iterations):
        key, step_key = jax.random.split(key)
        training_state, best_training_state, loss, steps_since_last_improvement, losses = model_training_step(
            training_state, training_transitions, validation_transitions, step_key, best_training_state, loss, steps_since_last_improvement
        )
        # logging.info('Losses: %s', losses)
        if steps_since_last_improvement is not None and steps_since_last_improvement >= patience:
            logging.info('Early stopping triggered')
            break

    logging.info('Training loop completed after %s steps', i)
    logging.info('Final validation loss: %s', loss)
    

    return best_training_state, loss

  def get_cutoffs(training_state, full_transitions):
    """Compute InfoProp rollout termination thresholds from the real-data buffer.

    Runs a forward pass of the ensemble on all stored transitions, then computes:
      - Epistemic variance (ensemble disagreement).
      - Kalman-fused posterior and conditional entropy per state dimension.
      - per_step_cutoff  (lambda_1): upper-quantile of per-step conditional entropy.
      - accumulated_cutoff (lambda_2): lower-quantile * max_rollout_length.

    Returns:
        per_step_cutoff, accumulated_cutoff, binning_entropy
    """
    model_ = training_state.model_state
    model_obs_mean = training_state.model_obs_mean
    model_obs_std = training_state.model_obs_std
    next_state_delta_mean = training_state.next_state_delta_mean
    next_state_delta_std = training_state.next_state_delta_std

    model_dataset_val = full_transitions

    curr_physics_state = model_dataset_val.observation[:, (obs_history -1) * 11 : obs_history *11]
    curr_odom_state = jnp.zeros((curr_physics_state.shape[0], 5))
    applied_torque = model_dataset_val.action
    # Forward pass through the ensemble: each of the E members predicts (mean, logvar)
    means_, logvars_ = model_.apply_fn(
            {"params": model_.params}, model_dataset_val.observation, applied_torque, model_obs_mean, model_obs_std
        )
    # Rescale model output variances to physical units (undo normalisation, apply dt)
    dt = model_environment.dt
    VARS = jnp.exp(logvars_)
    vars_ = VARS * (next_state_delta_std + 1e-6) ** 2 * dt ** 2
    R = jax.vmap(YRP)(curr_odom_state[:,0], curr_physics_state[:,0], curr_physics_state[:,1])[:,None, :,:]
    vars__1 = (dt/2) ** 2 * vars_[:,:,_EULER_RATE_IDX]
    # x,y variance from body_vel propagation; z variance comes from model directly at index 10
    vars__2 = (dt/2) ** 2 * (jnp.square(R) @ vars_[:,:,_BODY_VEL_IDX, None]).squeeze(-1)[:,:,:2]
    vars__ = jnp.concatenate((vars__1, vars__2), axis=-1)
    vars = jnp.concatenate((vars_, vars__), axis=-1)
    means_ = (means_ * (next_state_delta_std + 1e-6) + next_state_delta_mean) * dt + curr_physics_state[:, None, :]
    means__1 = curr_odom_state[:,None, 0:3] + dt * (curr_physics_state[:,_EULER_RATE_IDX][:,None,:] + means_[:,:,_EULER_RATE_IDX]) / 2
    # x,y integration only; z is now directly predicted by the model at physics_state index 10
    means__2 = curr_odom_state[:,None, 3:5] + dt * (R @ curr_physics_state[:,_BODY_VEL_IDX][:,None,:, None] + R @ means_[:,:,_BODY_VEL_IDX][:,:,:,None]).squeeze(-1)[:,:,:2] / 2
    means__ = jnp.concatenate((means__1, means__2), axis=-1)
    means = jnp.concatenate((means_, means__), axis=-1)

    # Precision-weighted fusion of ensemble predictions (inverse-variance pooling)
    inv_vars = 1 / (vars + 1e-12)
    fused_var = 1 / jnp.mean(inv_vars, axis=1)
    fused_mean = fused_var * jnp.mean(means * inv_vars, axis=1)
    # Epistemic variance: disagreement between ensemble means
    epist_var = jnp.mean((means - fused_mean[:, None, :]) ** 2, axis=1)
    # Kalman gain: K = Sigma_GT / (Sigma_GT + Sigma_epist)
    kalman_gain = jnp.clip((fused_var) / (fused_var + epist_var), 0, 1)
    conditional_var = ((1 - kalman_gain) * fused_var)

    # Per-step information loss H(s_tilde): differential entropy of the filtered state
    diff_entropy = 0.5 * jnp.log2(2 * jnp.pi * jnp.e * conditional_var)
    binning_entropy = jnp.quantile(diff_entropy, lower_quantile, axis=0) - 1
    conditional_entropy = jnp.clip(diff_entropy - binning_entropy, 0, None)

    # Lambda_1: per-step cutoff — upper quantile of information loss over training buffer
    per_step_cutoff = jnp.quantile(conditional_entropy, upper_quantile, axis=0)
    # Lambda_2: accumulated cutoff — lower quantile * horizon
    accumulated_cutoff = jnp.quantile(conditional_entropy, lower_quantile, axis=0) * max_rollout_length

    return per_step_cutoff, accumulated_cutoff, binning_entropy
  
  get_cutoffs = jax.jit(get_cutoffs)

  def model_actor_step(
    model_env: Env,
    model_env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
        ) -> Tuple[State, Transition]:
        """Carries out one model-based rollout step"""
        actions, policy_extras = policy(model_env_state.obs, key)
        nstate = model_env.step(model_env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}

        return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=model_env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        )


  def actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
        ) -> Tuple[State, Transition]:
        """Carries out one step using the policy in the real environment"""
        actions, policy_extras = policy(env_state.obs, key)
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        next_state = nstate.info['physics_state']
        return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        ), Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=jnp.concatenate([env_state.info['phys_state_history'], env_state.info['act_history']], axis=-1),
            action=nstate.info['applied_torque'],
            reward=jnp.zeros(nstate.reward.shape, dtype=jnp.float32),
            discount=1 - nstate.done,
            next_observation=next_state,
            extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        )
  
  def random_actor_step(
    env: Env,
    env_state: State,
    policy: Policy,
    key: PRNGKey,
    extra_fields: Sequence[str] = (),
        ) -> Tuple[State, Transition]:
        """Carries out one step using a random policy in the real environment."""
        actions = jax.random.uniform(key, (env_state.obs.shape[0], 2), minval=-1.0, maxval=1.0)
        policy_extras = {}
        nstate = env.step(env_state, actions)
        state_extras = {x: nstate.info[x] for x in extra_fields}
        next_state = nstate.info['physics_state']
        return nstate, Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=env_state.obs,
            action=actions,
            reward=nstate.reward,
            discount=1 - nstate.done,
            next_observation=nstate.obs,
            extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        ), Transition(  # pytype: disable=wrong-arg-types  # jax-ndarray
            observation=jnp.concatenate([env_state.info['phys_state_history'], env_state.info['act_history']], axis=-1),
            action=nstate.info['applied_torque'],
            reward=jnp.zeros(nstate.reward.shape, dtype=jnp.float32),
            discount=1 - nstate.done,
            next_observation=next_state,
            extras={'policy_extras': policy_extras, 'state_extras': state_extras},
        )
  
  def get_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      env_state: Union[envs.State, envs.State],
      buffer_state: ReplayBufferState,
      physics_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      Union[envs.State, envs.State],
      ReplayBufferState,
  ]:
    """Get experience from the real environment using current policy and insert it into the replay buffer."""
    policy = make_policy((normalizer_params, policy_params))
    # curr_state = jnp.concatenate([*env_state.pipeline_state.qpos, *env_state.pipeline_state.qvel], axis=-1)
    env_state, transitions, transitions_state = actor_step(
        env, env_state, policy, key, extra_fields=('truncation','track_seed','invariant_physics_state',)
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
    )
    
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    physics_buffer_state = replay_buffer_physics_state.insert(physics_buffer_state, transitions_state)
    return normalizer_params, env_state, buffer_state, physics_buffer_state
  
  def get_random_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      env_state: Union[envs.State, envs.State],
      buffer_state: ReplayBufferState,
      physics_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      Union[envs.State, envs.State],
      ReplayBufferState,
  ]:
    """Get experience from the real environment using a random policy and insert it into the replay buffer."""
    policy = make_policy((normalizer_params, policy_params))
    # curr_state = jnp.concatenate([*env_state.pipeline_state.qpos, *env_state.pipeline_state.qvel], axis=-1)
    env_state, transitions, transitions_state = random_actor_step(
        env, env_state, policy, key, extra_fields=('truncation','track_seed','invariant_physics_state',)
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        transitions.observation,
    )
    
    buffer_state = replay_buffer.insert(buffer_state, transitions)
    physics_buffer_state = replay_buffer_physics_state.insert(physics_buffer_state, transitions_state)
    return normalizer_params, env_state, buffer_state, physics_buffer_state

  def prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      physics_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
    """Warm-start the replay buffer with initial policy (or random) transitions."""

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, physics_buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state, physics_buffer_state = get_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          env_state,
          buffer_state,
          physics_buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
        #   env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, physics_buffer_state, new_key), ()

    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, physics_buffer_state, key),
        (),
        length=num_prefill_real_actor_steps,
    )[0]

  prefill_replay_buffer = jax.jit(prefill_replay_buffer)

  def random_prefill_replay_buffer(
      training_state: TrainingState,
      env_state: envs.State,
      buffer_state: ReplayBufferState,
      physics_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
    """Warm-start the replay buffer with initial policy (or random) transitions."""

    def f(carry, unused):
      del unused
      training_state, env_state, buffer_state, physics_buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, env_state, buffer_state, physics_buffer_state = get_random_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          env_state,
          buffer_state,
          physics_buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
        #   env_steps=training_state.env_steps + env_steps_per_actor_step,
      )
      return (new_training_state, env_state, buffer_state, physics_buffer_state, new_key), ()

    return jax.lax.scan(
        f,
        (training_state, env_state, buffer_state, physics_buffer_state, key),
        (),
        length=num_prefill_real_actor_steps,
    )[0]

  random_prefill_replay_buffer = jax.jit(random_prefill_replay_buffer)
  
  def get_model_based_experience(
      normalizer_params: running_statistics.RunningStatisticsState,
      policy_params: Params,
      model_env_state: Union[envs.State, envs.State],
      model_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      running_statistics.RunningStatisticsState,
      Union[envs.State, envs.State],
      ReplayBufferState,
  ]:
    """Get synthetic experience from the model environment and insert it into the replay buffer."""
    # nonlocal model_env 
    policy = make_policy((normalizer_params, policy_params))
    model_env_state, model_transitions = model_actor_step(
        model_env, model_env_state, policy, key, extra_fields=('truncation','track_seed')
    )

    normalizer_params = running_statistics.update(
        normalizer_params,
        model_transitions.observation,
    )

    model_buffer_state = model_replay_buffer.insert(model_buffer_state, model_transitions)
    return normalizer_params, model_env_state, model_buffer_state
  
  def fill_model_replay_buffer(
      training_state: TrainingState,
      model_env_state: envs.State,
      model_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, PRNGKey]:
    """Pre-fill the model replay buffer with synthetic rollouts before agent training."""

    def f(carry, unused):
      del unused
      training_state, model_env_state, model_buffer_state, key = carry
      key, new_key = jax.random.split(key)
      new_normalizer_params, model_env_state, model_buffer_state, = get_model_based_experience(
          training_state.normalizer_params,
          training_state.policy_params,
          model_env_state,
          model_buffer_state,
          key,
      )
      new_training_state = training_state.replace(
          normalizer_params=new_normalizer_params,
      )
      return (new_training_state, model_env_state, model_buffer_state, new_key), ()

    return jax.lax.scan(
        f,
        (training_state, model_env_state, model_buffer_state, key),
        (),
        length=num_prefill_actor_steps,
    )[0]

  fill_model_replay_buffer = jax.jit(fill_model_replay_buffer)

  def agent_sgd_step(
      carry: Tuple[TrainingState, PRNGKey], transitions: Transition
  ) -> Tuple[Tuple[TrainingState, PRNGKey], Metrics]:
    """Single SAC gradient update: update actor, critic, and entropy coefficient."""
    training_state, key = carry

    key, key_alpha, key_critic, key_actor = jax.random.split(key, 4)

    if tune_entropy:
      (alpha_loss_val, actor_entropy), alpha_params, alpha_optimizer_state = alpha_update(
          training_state.alpha_params,
          training_state.policy_params,
          training_state.normalizer_params,
          transitions,
          key_alpha,
          optimizer_state=training_state.alpha_optimizer_state,
      )
    else:
      _, actor_entropy = alpha_loss(
          training_state.alpha_params,
          training_state.policy_params,
          training_state.normalizer_params,
          transitions,
          key_alpha,
      )
      alpha_loss_val = jnp.zeros(())
      alpha_params = training_state.alpha_params
      alpha_optimizer_state = training_state.alpha_optimizer_state
    alpha = jnp.exp(training_state.alpha_params)
    critic_loss, q_params, q_optimizer_state = critic_update(
        training_state.q_params,
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.target_q_params,
        alpha,
        transitions,
        key_critic,
        optimizer_state=training_state.q_optimizer_state,
    )
    actor_loss, policy_params, policy_optimizer_state = actor_update(
        training_state.policy_params,
        training_state.normalizer_params,
        training_state.q_params,
        alpha,
        transitions,
        key_actor,
        optimizer_state=training_state.policy_optimizer_state,
    )

    new_target_q_params = jax.tree_util.tree_map(
        lambda x, y: x * (1 - tau) + y * tau,
        training_state.target_q_params,
        q_params,
    )

    metrics = {
        'critic_loss': critic_loss,
        'actor_loss': actor_loss,
        'alpha_loss': alpha_loss_val,
        'alpha': jnp.exp(alpha_params),
        'actor_entropy': actor_entropy,
        'target_entropy': jnp.asarray(target_entropy, dtype=jnp.float32),
    }

    new_training_state = TrainingState(
        policy_optimizer_state=policy_optimizer_state,
        policy_params=policy_params,
        q_optimizer_state=q_optimizer_state,
        q_params=q_params,
        target_q_params=new_target_q_params,
        gradient_steps=training_state.gradient_steps + 1,
        env_steps=training_state.env_steps,
        alpha_optimizer_state=alpha_optimizer_state,
        alpha_params=alpha_params,
        normalizer_params=training_state.normalizer_params,
        model_state=training_state.model_state,
        model_obs_mean=training_state.model_obs_mean,
        model_obs_std=training_state.model_obs_std,
        next_state_delta_mean=training_state.next_state_delta_mean,
        next_state_delta_std=training_state.next_state_delta_std,
    )
    return (new_training_state, key), metrics
  
  def agent_training_step(
      training_state: TrainingState,
      model_env_state: envs.State,
      model_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[
      TrainingState,
      Union[envs.State, envs.State],
      ReplayBufferState,
      Metrics,
  ]:
    """Collect one batch of model rollouts and perform `grad_updates_per_model_step` SAC updates."""
    experience_key, training_key = jax.random.split(key)

    # Extract model params from the env state before the scan so they are
    # captured as a closure constant rather than carried as mutable loop state.
    # This prevents XLA from including ~34 MB of params in the scan carry for
    # every one of the num_training_steps_per_model_train iterations.
    _model_params = model_env_state.info['model']
    _lean_info = {k: v for k, v in model_env_state.info.items() if k != 'model'}
    _lean_env_state = model_env_state.replace(info=_lean_info)

    def f(carry, unused):
        normalizer_params, policy_params, lean_env_state, model_buffer_state, experience_key = carry
        experience_key, new_key = jax.random.split(experience_key)
        # Temporarily inject model params (from closure) for the step function.
        full_info = dict(lean_env_state.info)
        full_info['model'] = _model_params
        full_env_state = lean_env_state.replace(info=full_info)
        new_normalizer_params, full_env_state, model_buffer_state = get_model_based_experience(
            normalizer_params,
            policy_params,
            full_env_state,
            model_buffer_state,
            experience_key,
        )
        # Strip model params from the output carry.
        lean_info_out = {k: v for k, v in full_env_state.info.items() if k != 'model'}
        lean_env_state_out = full_env_state.replace(info=lean_info_out)
        return (new_normalizer_params, policy_params, lean_env_state_out, model_buffer_state, new_key), ()

    (normalizer_params, _, _lean_env_state, model_buffer_state, _), _ = jax.lax.scan(
        f,
        (training_state.normalizer_params, training_state.policy_params, _lean_env_state, model_buffer_state, experience_key),
        (),
        length=num_training_steps_per_model_train,
    )
    # Reattach model params so the returned state matches the caller's expectation.
    final_info = dict(_lean_env_state.info)
    final_info['model'] = _model_params
    model_env_state = _lean_env_state.replace(info=final_info)
    
    training_state = training_state.replace(
        normalizer_params=normalizer_params,
        env_steps=training_state.env_steps + grad_updates_per_model_step,
    )

    model_buffer_state, transitions = model_replay_buffer.sample(model_buffer_state)
    # Change the front dimension of transitions so 'update_step' is called
    # grad_updates_per_step times by the scan.
    transitions = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, (grad_updates_per_model_step*num_training_steps_per_model_train, -1) + x.shape[1:]),
        transitions,
    )
    (training_state, _), metrics = jax.lax.scan(
        agent_sgd_step, (training_state, training_key), transitions
    )

    metrics['buffer_current_size'] = model_replay_buffer.size(model_buffer_state)
    return training_state, model_env_state, model_buffer_state, metrics
  
  def agent_training_epoch(
      training_state: TrainingState,
      model_env_state: envs.State,
      model_buffer_state: ReplayBufferState,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    """Performs a single epoch of training for the agent."""

    # def f(carry, unused_t):
    #   ts, es, bs, k = carry
    #   k, new_key = jax.random.split(k)
    #   ts, es, bs, metrics = agent_training_step(ts, es, bs, k)
    #   return (ts, es, bs, new_key), metrics

    # (training_state, model_env_state, model_buffer_state, key), metrics = jax.lax.scan(
    #     f,
    #     (training_state, model_env_state, model_buffer_state, key),
    #     (),
    #     length=num_training_steps_per_model_train,
    # )
    training_state, model_env_state, model_buffer_state, metrics = agent_training_step(training_state, model_env_state, model_buffer_state, key)
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, model_env_state, model_buffer_state, metrics

  agent_training_epoch = jax.jit(agent_training_epoch)

  def _reset_agent_params(training_state: TrainingState) -> TrainingState:
    """Restore SAC agent fields to their original initial values."""
    return training_state.replace(
        policy_params=_initial_policy_params,
        policy_optimizer_state=_initial_policy_optimizer_state,
        q_params=_initial_q_params,
        q_optimizer_state=_initial_q_optimizer_state,
        target_q_params=_initial_q_params,
        alpha_params=_initial_alpha_params,
        alpha_optimizer_state=_initial_alpha_optimizer_state,
    )

  _reset_agent_params = jax.jit(_reset_agent_params)

  def _reset_model_params(training_state: TrainingState) -> TrainingState:
    """Re-initialise only the ensemble model fields of training_state on-device."""
    return training_state.replace(
        model_state=_initial_model_state,
    )

  _reset_model_params = jax.jit(_reset_model_params)
  _reset_model_buffer = jax.jit(model_replay_buffer.init)

  def agent_training_step_with_resampling(
      training_state: TrainingState,
      init_model_env_states: envs.State,
      physics_buffer_state,
      model_buffer_state: ReplayBufferState,
      per_step_cutoff,
      accumulated_cutoff,
      binning_entropy,
      key: PRNGKey,
  ):
    """Run multiple agent training epochs, re-initialising model env states from the physics buffer each epoch.

    Initial states are drawn from real-data transitions and augmented with invariance
    transforms (random track context) to diversify the rollout starting distribution.
    """
    def f(carry, unused): 
        training_state, model_buffer_state, key = carry
        model_env_state = unused

        epoch_key, key = jax.random.split(key)
        training_state, model_env_state, model_buffer_state, metrics = agent_training_epoch(
            training_state, model_env_state, model_buffer_state, epoch_key
        )
        metrics['average_rollout_length'] =(model_env_state.info["total_done_steps"] / model_env_state.info["num_inits"] + jnp.sum((1-model_env_state.done)*model_env_state.info["steps"]) / jnp.sum(1-model_env_state.done))
        return (training_state, model_buffer_state, key), metrics
    
    (training_state, model_buffer_state, key), metrics = jax.lax.scan(
                                                                                        f,
                                                                                        (training_state, model_buffer_state, key),
                                                                                        (init_model_env_states),
                                                                                    )
    
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    return training_state, model_buffer_state, key, metrics

  agent_training_step_with_resampling = jax.jit(agent_training_step_with_resampling)

  
  def training_epoch_with_timing(
      training_state: TrainingState,
      init_model_env_states: envs.State,
      physics_buffer_state,
      model_buffer_state: ReplayBufferState,
      per_step_cutoff,
      accumulated_cutoff,
      binning_entropy,
      key: PRNGKey,
  ) -> Tuple[TrainingState, envs.State, ReplayBufferState, Metrics]:
    """Performs a single cycle of agent training with timing."""
    nonlocal training_walltime
    t = time.time()
    (training_state, model_buffer_state, key, metrics) = agent_training_step_with_resampling(
        training_state, init_model_env_states, physics_buffer_state, model_buffer_state, per_step_cutoff, accumulated_cutoff, binning_entropy, key
    )
    metrics = jax.tree_util.tree_map(jnp.mean, metrics)
    jax.tree_util.tree_map(lambda x: x.block_until_ready(), metrics)

    epoch_training_time = time.time() - t
    training_walltime += epoch_training_time
    sps = (
        env_steps_per_actor_step * num_training_steps_per_model_train
    ) / epoch_training_time
    metrics = {
        'training/sps': sps,
        'training/walltime': training_walltime,
        **{f'training/{name}': value for name, value in metrics.items()},
    }
    return training_state, model_buffer_state, key, metrics  # pytype: disable=bad-return-type  # py311-upgrade
  
  def run_eval_and_collect_data(training_state, env_state, buffer_state, physics_buffer_state, prefill_key):
    """
    Collects real experience using the current policy.
    """
    training_state, env_state, buffer_state, physics_buffer_state, _ = prefill_replay_buffer(
        training_state, env_state, buffer_state, physics_buffer_state, prefill_key
    ) 
    return training_state, env_state, buffer_state, physics_buffer_state
  
  def run_eval_and_collect_random_data(training_state, env_state, buffer_state, physics_buffer_state, prefill_key):
    """
    Collects random experience using a random policy.
    """
    training_state, env_state, buffer_state, physics_buffer_state, _ = random_prefill_replay_buffer(
        training_state, env_state, buffer_state, physics_buffer_state, prefill_key
    ) 
    return training_state, env_state, buffer_state, physics_buffer_state
  
  training_walltime = time.time()

  
  normalizer_params, policy_params = None, None
  # Training state init
  training_state = _init_training_state(
      key=global_key,
      obs_size=obs_size,
      local_devices_to_use=local_devices_to_use,
      sac_network=sac_network,
      alpha_optimizer=alpha_optimizer,
      policy_optimizer=policy_optimizer,
      q_optimizer=q_optimizer,
        policy_params=policy_params,
        normalizer_params=normalizer_params,
        model_state=model_state,
        model_obs_mean=model_obs_mean,
        model_obs_std=model_obs_std,
        next_state_delta_mean=next_state_delta_mean,
        next_state_delta_std=next_state_delta_std,
        initial_log_alpha=float(jnp.log(alpha)) if not tune_entropy else 0.0,
  )

  _initial_policy_params = training_state.policy_params
  _initial_policy_optimizer_state = training_state.policy_optimizer_state
  _initial_q_params = training_state.q_params
  _initial_q_optimizer_state = training_state.q_optimizer_state
  _initial_alpha_params = training_state.alpha_params
  _initial_alpha_optimizer_state = training_state.alpha_optimizer_state

  local_key, rb_key1, rb_key2, rb_key3 = jax.random.split(local_key, 4)

  # buffer states init
  buffer_state = replay_buffer.init(rb_key1)
  physics_buffer_state = replay_buffer_physics_state.init(rb_key2)
  model_buffer_init_key = rb_key3
  model_buffer_state = model_replay_buffer.init(model_buffer_init_key)

  jit_env_reset = jax.jit(env.reset)

  # collect an initial dataset by running a random policy on the environment
  logging.info('Collecting initial random physics dataset')
  curr_env_key, local_key = jax.random.split(local_key)
  env_keys = jax.random.split(curr_env_key, num_real_envs)
  env_state = jit_env_reset(env_keys)
  prefill_key, local_key = jax.random.split(local_key)
  training_state, env_state, buffer_state, physics_buffer_state = run_eval_and_collect_data(training_state, env_state, buffer_state, physics_buffer_state, prefill_key)
  replay_size = replay_buffer_physics_state.size(physics_buffer_state)
  logging.info('physics replay size: %s', replay_size)


  # initialize model_env
  model_env = wrap_custom(
            model_environment,
            replay_buffer_physics_state,
            episode_length=episode_length,
            action_repeat=action_repeat,
            reset_pipeline_state=not fast_model_rollout,
  ) 
  model_env_reset = jax.jit(jax.vmap(model_env.reset, in_axes=(0, None)))

  # Hoisted outside the trial loop so JAX jit cache hits every iteration.
  _unflatten = jax.jit(replay_buffer_physics_state._unflatten_fn)
  _unflatten_vmap = jax.jit(jax.vmap(replay_buffer_physics_state._unflatten_fn))

# train, and collect data num_trial times
  num_steps = 0
  num_real_transitions = min_physics_replay_size
  for iteration in range(num_trials):
        # model training
        t0 = time.time()
        if reset_model_per_trial:
            logging.info('Resetting model parameters and optimizer state...')
            training_state = _reset_model_params(training_state)
        logging.info('Starting Model Training...')
        # All index arrays have fixed Python-int sizes so every downstream jit
        # compiles once. We always permute max_physics_replay_size indices and map
        # into the valid range via modulo — replay_size_int is a runtime value (not a
        # shape), so wrapping it in jnp.array prevents JAX from folding it as a literal.
        replay_size_int = int(replay_size)
        data = physics_buffer_state.data  # [max_physics_replay_size, raw_dim] — fixed

        fixed_train_size = int(0.8 * max_physics_replay_size / model_batch_size) * model_batch_size
        fixed_val_size   = max_physics_replay_size - fixed_train_size
        num_updates_in_epoch = fixed_train_size // model_batch_size

        local_key, perm_key = jax.random.split(local_key)
        full_perm  = jax.random.permutation(perm_key, max_physics_replay_size)
        valid_perm = full_perm % jnp.array(replay_size_int, dtype=jnp.int32)
        train_perm = valid_perm[:fixed_train_size]
        val_perm   = valid_perm[fixed_train_size:]

        # full_transitions — fixed shape, always valid data, compiles once
        full_transitions = _unflatten(data[valid_perm, :])

        # train_data — fixed shape [fixed_train_size, raw_dim], compiles once
        train_data = data[train_perm, :]
        local_key, *shuffle_keys = jax.random.split(local_key, 9)
        shuffle_keys = jnp.stack(shuffle_keys, axis=0)
        shuffle_idx = jax.vmap(lambda k: jax.random.permutation(k, fixed_train_size))(shuffle_keys)
        train_data = train_data[shuffle_idx, :]

        # val_data — fixed shape [fixed_val_size, raw_dim], compiles once
        val_data = data[val_perm, :]

        def _reshape_train(x):
            # x: (8, fixed_train_size, leaf_dim) from _unflatten_vmap
            swapped = x.swapaxes(0, 1)  # (fixed_train_size, 8, leaf_dim)
            return jnp.reshape(swapped, (num_updates_in_epoch, -1) + swapped.shape[1:])
            # result: (num_updates_in_epoch, model_batch_size, 8, leaf_dim)
        train_transitions = jax.tree_util.tree_map(_reshape_train, _unflatten_vmap(train_data))

        val_transitions = _unflatten(val_data)

        training_state = update_stats(training_state, full_transitions)
        training_key, local_key = jax.random.split(local_key)
        training_state, val_loss = model_training_loop(
                                                                    training_state, train_transitions, val_transitions, training_key
                                                            )
        
        per_step_cutoff, accumulated_cutoff, binning_entropy = get_cutoffs(
            training_state, full_transitions
        )
        logging.info('per_step_cutoff: %s', per_step_cutoff)
        logging.info('accumulated_cutoff: %s', accumulated_cutoff)
        logging.info('binning_entropy: %s', binning_entropy)

        # sample initial states
        logging.info('Starting Agent training...')
        curr_env_key, local_key = jax.random.split(local_key)
        env_keys = jax.random.split(curr_env_key, num_envs * num_resampling_epochs)
        env_keys = jnp.reshape(
            env_keys, (num_resampling_epochs, num_envs) + env_keys.shape[1:]
        )
        init_model_env_states = model_env_reset(env_keys, physics_buffer_state)
        info = init_model_env_states.info
        info['model'] = tree_repeat(_unpmap(training_state.model_state.params), num_resampling_epochs)
        scalar_fields = {
            'model_obs_mean':       training_state.model_obs_mean,
            'model_obs_std':        training_state.model_obs_std,
            'next_state_delta_mean': training_state.next_state_delta_mean,
            'next_state_delta_std': training_state.next_state_delta_std,
            'per_step_cutoff':      per_step_cutoff,
            'accumulated_cutoff':   accumulated_cutoff,
            'binning_entropy':      binning_entropy,
        }
        repeated = jax.tree_util.tree_map(
            lambda x: jnp.repeat(x[None], num_resampling_epochs, axis=0), scalar_fields
        )
        info.update(repeated)
        info['rng'] = env_keys
        init_model_env_states = init_model_env_states.replace(info=info)
        # local_key, buf_reset_key = jax.random.split(local_key)
        if reset_model_replay_buffer:
            logging.info('Resetting model replay buffer...')
            evolved_key = model_buffer_state.key
            model_buffer_state = _reset_model_buffer(model_buffer_init_key)
            model_buffer_state = model_buffer_state.replace(key=evolved_key)

        if reset_agent_per_trial:
            logging.info('Resetting agent parameters (policy, critic, alpha, target Q)...')
            training_state = _reset_agent_params(training_state)

            

        training_state, model_buffer_state, local_key, metrics = training_epoch_with_timing(
            training_state, init_model_env_states, physics_buffer_state, model_buffer_state, per_step_cutoff, accumulated_cutoff, binning_entropy, local_key
        )
        metrics['model/val_loss'] = jnp.mean(val_loss)
        logging.info('Agent training completed...')
        logging.info('Critic Loss: %s', metrics['training/critic_loss'])
        logging.info('Actor Loss: %s', metrics['training/actor_loss'])
        logging.info('Alpha Loss: %s', metrics['training/alpha_loss'])
        logging.info('Alpha: %s', metrics['training/alpha'])
        logging.info('Actor Entropy: %s', metrics['training/actor_entropy'])
        logging.info('Target Entropy: %s', metrics['training/target_entropy'])
        logging.info('Average rollout length: %s', metrics['training/average_rollout_length'])
        replay_size = model_replay_buffer.size(model_buffer_state)
        logging.info('Model replay size: %s', replay_size)
        

        logging.info('Evaluating Agent and Model...')
        params = _unpmap((training_state.normalizer_params, training_state.policy_params))

        # logging.info('Running Evaluator...')
        metrics = evaluator.run_evaluation(
                                                params,
                                                metrics,
                                            )
        logging.info('Eval Episode Reward: %s', metrics['eval/episode_reward'])
        logging.info('Eval Episode Reward Std: %s', metrics['eval/episode_reward_std'])
        logging.info('Eval Avg Episode Length: %s', metrics['eval/avg_episode_length'])
        num_steps += num_resampling_epochs * grad_updates_per_model_step * num_training_steps_per_model_train
        metrics['num_real_transitions'] = num_real_transitions
        progress_fn(num_steps, metrics)

        if agent_dir:
            os.makedirs(agent_dir, exist_ok=True)
            ckpt_path = os.path.join(agent_dir, f'brax_policy_{iteration}')
            brax_model.save_params(ckpt_path, params)
            logging.info('Policy params saved to %s', ckpt_path)

        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            model_ckpt = {
                'params': jax.tree_util.tree_map(np.array, _unpmap(training_state.model_state).params),
                'model_obs_mean': np.array(_unpmap(training_state.model_obs_mean)),
                'model_obs_std': np.array(_unpmap(training_state.model_obs_std)),
                'next_state_delta_mean': np.array(_unpmap(training_state.next_state_delta_mean)),
                'next_state_delta_std': np.array(_unpmap(training_state.next_state_delta_std)),
                'per_step_cutoff': np.array(per_step_cutoff),
                'accumulated_cutoff': np.array(accumulated_cutoff),
                'binning_entropy': np.array(binning_entropy),
            }
            model_ckpt_path = os.path.join(model_dir, f'model_state_{iteration}')
            brax_model.save_params(model_ckpt_path, model_ckpt)
            logging.info('Model state saved to %s', model_ckpt_path)

        # collecting fresh experience using current policy
        logging.info('Collecting real experience...')
        curr_env_key, local_key = jax.random.split(local_key)
        env_keys = jax.random.split(curr_env_key, num_real_envs)
        env_state = jit_env_reset(env_keys)
        prefill_key, local_key = jax.random.split(local_key)
        training_state, env_state, buffer_state, physics_buffer_state = run_eval_and_collect_data(training_state, env_state, buffer_state, physics_buffer_state, prefill_key)
        replay_size = replay_buffer_physics_state.size(physics_buffer_state)
        logging.info('physics replay size: %s', replay_size)
        t1 = time.time()
        logging.info('Full iteration (model + agent + real data collection) took: %s seconds', t1 - t0)
        num_real_transitions += min_physics_replay_size
  return (make_policy, params, metrics)
