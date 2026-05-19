"""
Brax environment wrappers for episode bookkeeping and observation normalisation.
"""
from typing import Callable, Dict, Optional, Tuple

from brax.base import System
from brax.envs.base import Env, State, Wrapper
from flax import struct
import jax
from jax import numpy as jp
from functools import partial

from brax.envs.wrappers.training import (
    EpisodeWrapper,
    AutoResetWrapper,
    VmapWrapper,
    DomainRandomizationVmapWrapper,
)


def wrap_custom(
    env: Env,
    replay_buffer,
    episode_length: int = 1000,
    action_repeat: int = 1,
    reset_pipeline_state: bool = True,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System]]
    ] = None,
) -> Wrapper:
    """Apply episode-tracking wrapper to the environment."""
    env = VmapInfopropWrapper(env, replay_buffer)
    env = CustomEpisodeWrapper(env, episode_length, action_repeat)
    env = CustomAutoResetWrapper(env, reset_pipeline_state=reset_pipeline_state)
    return env


def wrap(
    env: Env,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System]]
    ] = None,
) -> Wrapper:
    """Apply the standard Brax VmapWrapper and EpisodeWrapper.

    Args:
        env: environment to be wrapped
        episode_length: length of episode
        action_repeat: how many repeated actions to take per step
        randomization_fn: randomization function that produces a vectorized system
          and in_axes to vmap over

    Returns:
        An environment that is wrapped with Episode and AutoReset wrappers. If the
        environment did not already have batch dimensions, it is additionally Vmap
        wrapped.
    """
    if randomization_fn is None:
        env = VmapWrapper(env)
    else:
        env = DomainRandomizationVmapWrapper(env, randomization_fn)
    env = EpisodeWrapper(env, episode_length, action_repeat)
    env = CustomAutoResetWrapper2(env)
    return env


class VmapInfopropWrapper(Wrapper):
    """Vectorizes Brax env."""

    def __init__(self, env: Env, replay_buffer, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer
        self._vmap_first_half = jax.vmap(env.first_half_of_step)
        self._vmap_second_half = jax.vmap(env.second_half_of_step)

    def reset(self, rng: jax.Array, physics_buffer_state) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        n_envs = rng.shape[0]
        physics_buffer_state, init_robot_transition = self.replay_buffer.get_model_dataset(
            rng[0], physics_buffer_state, max_samples=n_envs
        )
        init_robot_state = init_robot_transition.observation
        track_seed = init_robot_transition.extras['state_extras']['track_seed']
        invariant_physics_state = init_robot_transition.extras['state_extras']['invariant_physics_state']
        init_xy = invariant_physics_state[:, 3:5]
        init_angle = invariant_physics_state[:, 0]
        init_state = jax.vmap(self.env.reset_with_init_robot_state)(
            rng, init_robot_state, track_seed, init_xy, init_angle
        )
        info = init_state.info
        info['info_cutoff'] = jp.zeros(n_envs)
        init_state = init_state.replace(info=info)
        return init_state

    def step(self, state: State, action: jax.Array) -> State:
        info = state.info
        model = info.pop('model')
        obs_mean = info.pop('model_obs_mean')
        obs_std = info.pop('model_obs_std')
        next_state_delta_mean = info.pop('next_state_delta_mean')
        next_state_delta_std = info.pop('next_state_delta_std')
        per_step_cut_off = info.pop('per_step_cutoff')
        accumulated_cutoff = info.pop('accumulated_cutoff')
        binning_entropy = info.pop('binning_entropy')
        num_inits = info.pop('num_inits')
        total_done_steps = info.pop('total_done_steps')

        state = state.replace(info=info)

        applied_torque, action_clipped = self._vmap_first_half(state, action)

        next_physics_state, rng, conditional_entropy, next_odom_state = (
            self.env.batched_model_step(
                state, applied_torque, model, obs_mean, obs_std,
                next_state_delta_mean, next_state_delta_std, binning_entropy,
            )
        )

        next_state = self._vmap_second_half(
            state, applied_torque, next_physics_state, action_clipped,
            conditional_entropy, rng, next_odom_state,
        )

        next_state = self.env.batch_entropy_cutoff(
            next_state, conditional_entropy,
            next_state.info['accumulated_conditional_entropy'],
            per_step_cut_off, accumulated_cutoff,
        )

        info = next_state.info
        info['model'] = model
        info['model_obs_mean'] = obs_mean
        info['model_obs_std'] = obs_std
        info['next_state_delta_mean'] = next_state_delta_mean
        info['next_state_delta_std'] = next_state_delta_std
        info['per_step_cutoff'] = per_step_cut_off
        info['accumulated_cutoff'] = accumulated_cutoff
        info['binning_entropy'] = binning_entropy
        info['num_inits'] = num_inits
        info['total_done_steps'] = total_done_steps
        next_state = next_state.replace(info=info)
        return next_state


class CustomEpisodeWrapper(Wrapper):
    """Maintains episode step count and sets done at episode end."""

    def __init__(self, env: Env, episode_length: int, action_repeat: int):
        super().__init__(env)
        self.episode_length = episode_length
        self.action_repeat = action_repeat

    def reset(self, rng: jax.Array, physics_buffer_state) -> State:
        state = self.env.reset(rng, physics_buffer_state)
        state.info['steps'] = jp.zeros(rng.shape[:-1])
        state.info['truncation'] = jp.zeros(rng.shape[:-1])
        # Keep separate record of episode done as state.info['done'] can be erased
        # by AutoResetWrapper
        state.info['episode_done'] = jp.zeros(rng.shape[:-1])
        episode_metrics = dict()
        episode_metrics['sum_reward'] = jp.zeros(rng.shape[:-1])
        episode_metrics['length'] = jp.zeros(rng.shape[:-1])
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jp.zeros(rng.shape[:-1])
        state.info['episode_metrics'] = episode_metrics
        return state

    def step(self, state: State, action: jax.Array) -> State:
        def f(state, _):
            nstate = self.env.step(state, action)
            return nstate, nstate.reward

        state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
        state = state.replace(reward=jp.sum(rewards, axis=0))
        steps = state.info['steps'] + self.action_repeat
        one = jp.ones_like(state.done)
        zero = jp.zeros_like(state.done)
        episode_length = jp.array(self.episode_length, dtype=jp.int32)
        done = jp.where(steps >= episode_length, one, state.done)
        state.info['truncation'] = jp.where(
            steps >= episode_length, 1 - state.done, zero
        )
        state.info['truncation'] = jp.where(
            jp.logical_or(state.info['truncation'], state.info['info_cutoff']), one, zero
        )
        state.info['steps'] = steps

        # Aggregate state metrics into episode metrics
        prev_done = state.info['episode_done']
        state.info['episode_metrics']['sum_reward'] += jp.sum(rewards, axis=0)
        state.info['episode_metrics']['sum_reward'] *= (1 - prev_done)
        state.info['episode_metrics']['length'] += self.action_repeat
        state.info['episode_metrics']['length'] *= (1 - prev_done)
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                state.info['episode_metrics'][metric_name] += state.metrics[metric_name]
                state.info['episode_metrics'][metric_name] *= (1 - prev_done)
        state.info['episode_done'] = done
        return state.replace(done=done)


class CustomAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def __init__(self, env: Env, reset_pipeline_state: bool = True):
        super().__init__(env)
        self.reset_pipeline_state = reset_pipeline_state

    def reset(self, rng: jax.Array, physics_buffer_state) -> State:
        state = self.env.reset(rng, physics_buffer_state)
        if self.reset_pipeline_state:
            state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        state.info['first_physics_state'] = state.info['physics_state']
        state.info['first_accumulated_conditional_entropy'] = (
            state.info['accumulated_conditional_entropy']
        )
        state.info['first_current_conditional_entropy'] = (
            state.info['current_conditional_entropy']
        )
        state.info['first_applied_torque'] = state.info['applied_torque']
        state.info['first_invariant_physics_state'] = state.info['invariant_physics_state']
        state.info['first_phys_state_history'] = state.info['phys_state_history']
        state.info['first_act_history'] = state.info['act_history']
        state.info['num_inits'] = 0
        state.info['total_done_steps'] = 0
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        done = state.done

        def where_done(x, y):
            if x.ndim > 1:
                d = jp.reshape(done, [done.shape[0]] + [1] * (x.ndim - 1))  # type: ignore
            else:
                d = done
            return jp.where(d, x, y)

        first = {
            'obs': state.info['first_obs'],
            'physics_state': state.info['first_physics_state'],
            'accumulated_conditional_entropy': state.info['first_accumulated_conditional_entropy'],
            'current_conditional_entropy': state.info['first_current_conditional_entropy'],
            'applied_torque': state.info['first_applied_torque'],
            'invariant_physics_state': state.info['first_invariant_physics_state'],
            'phys_state_history': state.info['first_phys_state_history'],
            'act_history': state.info['first_act_history'],
            'steps': jp.zeros_like(state.info['steps']),
        }
        current = {
            'obs': state.obs,
            'physics_state': state.info['physics_state'],
            'accumulated_conditional_entropy': state.info['accumulated_conditional_entropy'],
            'current_conditional_entropy': state.info['current_conditional_entropy'],
            'applied_torque': state.info['applied_torque'],
            'invariant_physics_state': state.info['invariant_physics_state'],
            'phys_state_history': state.info['phys_state_history'],
            'act_history': state.info['act_history'],
            'steps': state.info['steps'],
        }
        if self.reset_pipeline_state:
            first['pipeline_state'] = state.info['first_pipeline_state']
            current['pipeline_state'] = state.pipeline_state
        reset = jax.tree.map(where_done, first, current)

        info = state.info
        info['physics_state'] = reset['physics_state']
        info['accumulated_conditional_entropy'] = reset['accumulated_conditional_entropy']
        info['current_conditional_entropy'] = reset['current_conditional_entropy']
        info['applied_torque'] = reset['applied_torque']
        info['invariant_physics_state'] = reset['invariant_physics_state']
        info['phys_state_history'] = reset['phys_state_history']
        info['act_history'] = reset['act_history']
        info['num_inits'] = info['num_inits'] + jp.sum(state.done)
        info['total_done_steps'] = info['total_done_steps'] + jp.sum(
            state.done * state.info['steps']
        )
        info['steps'] = reset['steps']
        pipeline_state = reset['pipeline_state'] if self.reset_pipeline_state else state.pipeline_state
        return state.replace(pipeline_state=pipeline_state, obs=reset['obs'], info=info)


class CustomAutoResetWrapper2(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        state.info['first_physics_state'] = state.info['physics_state']
        state.info['first_applied_torque'] = state.info['applied_torque']
        state.info['first_invariant_physics_state'] = state.info['invariant_physics_state']
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        pipeline_state = jax.tree.map(
            where_done, state.info['first_pipeline_state'], state.pipeline_state
        )
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)
        physics_state = jax.tree.map(
            where_done, state.info['first_physics_state'], state.info['physics_state']
        )
        applied_torque = jax.tree.map(
            where_done, state.info['first_applied_torque'], state.info['applied_torque']
        )
        invariant_physics_state = jax.tree.map(
            where_done,
            state.info['first_invariant_physics_state'],
            state.info['invariant_physics_state'],
        )
        steps = jax.tree.map(
            where_done, jp.zeros_like(state.info['steps']), state.info['steps']
        )
        info = state.info
        info['physics_state'] = physics_state
        info['applied_torque'] = applied_torque
        info['invariant_physics_state'] = invariant_physics_state
        info['steps'] = steps
        return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)
