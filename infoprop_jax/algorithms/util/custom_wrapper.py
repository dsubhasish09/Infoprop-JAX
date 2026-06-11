"""
Infoprop training wrappers (Brax `Wrapper` subclasses).

The model-rollout stack wraps an ``InfopropEnv`` (built via ``wrap_custom``):

    CustomAutoResetWrapper -> CustomEpisodeWrapper -> VmapInfopropWrapper -> InfopropEnv -> <env>

  * ``VmapInfopropWrapper``  - resets a batch of rollouts from the physics replay buffer and runs
    the batched Infoprop step (``InfopropEnv.batched_step``), popping the shared (non-per-env) info
    so it is not vmapped.
  * ``CustomEpisodeWrapper`` - episode step counting, truncation (incl. entropy-cutoff truncation),
    and per-episode metric aggregation.
  * ``CustomAutoResetWrapper`` - reverts env-owned + entropy state to its episode-start values on
    ``done`` (driven by the env's ``reset_carry_keys``), and tracks rollout-length counters.

The real env is wrapped with ``wrap`` (standard Vmap/Episode + ``CustomAutoResetWrapper2``).

Both auto-reset wrappers stash the *actually reached* post-step values before reverting:
``pre_reset_obs`` (and, for the real env, ``pre_reset_<carry_key>``). Transition builders
must read these for next-state fields — ``obs``/carry keys are already reverted to the
episode start on ``done``, which would otherwise corrupt bootstrapped SAC targets on
truncation and inject reset "teleports" into the model-training data.
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
    randomization_fn: Optional[
        Callable[[System], Tuple[System, System]]
    ] = None,
) -> Wrapper:
    """Apply episode-tracking wrapper to the environment."""
    env = VmapInfopropWrapper(env, replay_buffer)
    env = CustomEpisodeWrapper(env, episode_length, action_repeat)
    env = CustomAutoResetWrapper(env)
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
    """Batches Infoprop model rollouts.

    ``reset`` samples a batch of real-data physics transitions from the replay buffer and turns each
    into an imagined-rollout initial state via the env's ``reset_from_buffer``. ``step`` runs the
    batched Infoprop step, popping the shared (non-per-env) info entries — ensemble params,
    normalisation stats, cutoffs, and the auto-reset counters — so they are broadcast, not vmapped.
    """

    def __init__(self, env: Env, replay_buffer, batch_size: Optional[int] = None):
        super().__init__(env)
        self.batch_size = batch_size
        self.replay_buffer = replay_buffer

    def reset(self, rng: jax.Array, physics_buffer_state) -> State:
        if self.batch_size is not None:
            rng = jax.random.split(rng, self.batch_size)
        n_envs = rng.shape[0]
        physics_buffer_state, init_transition = self.replay_buffer.get_model_dataset(
            rng[0], physics_buffer_state, max_samples=n_envs
        )
        # The env owns how a sampled physics transition becomes an initial state,
        # including whether it builds a pipeline_state. We stay agnostic to that.
        init_state = jax.vmap(self.env.reset_from_buffer, in_axes=(0, 0))(
            rng, init_transition)
        info = init_state.info
        info['info_cutoff'] = jp.zeros(n_envs)
        init_state = init_state.replace(info=info)
        return init_state

    def step(self, state: State, action: jax.Array) -> State:
        """Vectorized step: pop the shared (non-per-env) info entries — model params,
        normalisation stats, cutoffs and the autoreset counters — so they are not vmapped,
        run the batched Infoprop step, then reinsert them."""
        info = state.info
        model_params = info.pop('model')
        obs_mean = info.pop('model_obs_mean')
        obs_std = info.pop('model_obs_std')
        next_state_delta_mean = info.pop('next_state_delta_mean')
        next_state_delta_std = info.pop('next_state_delta_std')
        per_step_cutoff = info.pop('per_step_cutoff')
        accumulated_cutoff = info.pop('accumulated_cutoff')
        binning_entropy = info.pop('binning_entropy')
        num_inits = info.pop('num_inits')
        total_done_steps = info.pop('total_done_steps')

        state = state.replace(info=info)

        next_state = self.env.batched_step(
            state, action, model_params, obs_mean, obs_std,
            next_state_delta_mean, next_state_delta_std,
            per_step_cutoff, accumulated_cutoff, binning_entropy,
        )

        info = next_state.info
        info['model'] = model_params
        info['model_obs_mean'] = obs_mean
        info['model_obs_std'] = obs_std
        info['next_state_delta_mean'] = next_state_delta_mean
        info['next_state_delta_std'] = next_state_delta_std
        info['per_step_cutoff'] = per_step_cutoff
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

        # Aggregate state metrics into episode metrics; zero them on episode reset.
        prev_done = state.info['episode_done']
        ep = state.info['episode_metrics']
        ep['sum_reward'] = jp.where(prev_done, 0.0, ep['sum_reward'] + jp.sum(rewards, axis=0))
        ep['length'] = jp.where(prev_done, 0.0, ep['length'] + self.action_repeat)
        for metric_name in state.metrics.keys():
            if metric_name != 'reward':
                ep[metric_name] = jp.where(
                    prev_done, 0.0, ep[metric_name] + state.metrics[metric_name]
                )
        state.info['episode_done'] = done
        return state.replace(done=done)


class CustomAutoResetWrapper(Wrapper):
    """Automatically resets Brax envs that are done."""

    def __init__(self, env: Env):
        super().__init__(env)
        # Env-owned dynamic info keys (env-declared) plus the framework-owned entropy
        # accumulators are reverted to their episode-start values on `done`.
        self._carry_keys = list(env.reset_carry_keys) + [
            'accumulated_conditional_entropy', 'current_conditional_entropy']

    def reset(self, rng: jax.Array, physics_buffer_state) -> State:
        state = self.env.reset(rng, physics_buffer_state)
        # Agnostic to fast rollouts: snapshot pipeline_state only if the env built
        # one (it is None in fast-rollout mode). ``None`` is a static pytree node,
        # so this check is valid under jit/scan.
        if state.pipeline_state is not None:
            state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        for k in self._carry_keys:
            state.info[f'first_{k}'] = state.info[k]
        # The observation actually reached this step, before any auto-reset revert.
        # Consumers that bootstrap on truncation (SAC critic) must use this as
        # next_observation, since `obs` is reverted to the episode start on done.
        state.info['pre_reset_obs'] = state.obs
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

        first = {k: state.info[f'first_{k}'] for k in self._carry_keys}
        first['obs'] = state.info['first_obs']
        first['steps'] = jp.zeros_like(state.info['steps'])
        current = {k: state.info[k] for k in self._carry_keys}
        current['obs'] = state.obs
        current['steps'] = state.info['steps']
        has_pipeline_state = state.pipeline_state is not None
        if has_pipeline_state:
            first['pipeline_state'] = state.info['first_pipeline_state']
            current['pipeline_state'] = state.pipeline_state
        reset = jax.tree.map(where_done, first, current)

        info = state.info
        # Preserve the true post-step observation before reverting on done.
        info['pre_reset_obs'] = state.obs
        for k in self._carry_keys:
            info[k] = reset[k]
        info['num_inits'] = info['num_inits'] + jp.sum(state.done)
        info['total_done_steps'] = info['total_done_steps'] + jp.sum(
            state.done * state.info['steps']
        )
        info['steps'] = reset['steps']
        pipeline_state = reset['pipeline_state'] if has_pipeline_state else state.pipeline_state
        return state.replace(pipeline_state=pipeline_state, obs=reset['obs'], info=info)


class CustomAutoResetWrapper2(Wrapper):
    """Automatically resets Brax envs that are done."""

    def reset(self, rng: jax.Array) -> State:
        state = self.env.reset(rng)
        # Snapshot pipeline_state only if the env built one (agnostic to fast rollouts).
        if state.pipeline_state is not None:
            state.info['first_pipeline_state'] = state.pipeline_state
        state.info['first_obs'] = state.obs
        for k in self.env.reset_carry_keys:
            state.info[f'first_{k}'] = state.info[k]
        # The values actually reached this step, before any auto-reset revert.
        # Consumers of the *next* state (physics transitions for model training,
        # bootstrapped SAC targets) must read these, since obs and the env-owned
        # carry keys are reverted to the episode start on done.
        state.info['pre_reset_obs'] = state.obs
        for k in self.env.reset_carry_keys:
            state.info[f'pre_reset_{k}'] = state.info[k]
        return state

    def step(self, state: State, action: jax.Array) -> State:
        state = state.replace(done=jp.zeros_like(state.done))
        state = self.env.step(state, action)

        def where_done(x, y):
            done = state.done
            if done.shape:
                done = jp.reshape(done, [x.shape[0]] + [1] * (len(x.shape) - 1))  # type: ignore
            return jp.where(done, x, y)

        # Reset pipeline_state and obs (separate pytree structures). pipeline_state
        # is reverted only if the env built one (it is None in fast-rollout mode).
        if state.pipeline_state is not None:
            pipeline_state = jax.tree.map(
                where_done, state.info['first_pipeline_state'], state.pipeline_state
            )
        else:
            pipeline_state = state.pipeline_state
        obs = jax.tree.map(where_done, state.info['first_obs'], state.obs)

        # Reset env-owned info-dict fields (env-declared) + steps in one combined pass.
        carry_keys = self.env.reset_carry_keys
        first_info = {k: state.info[f'first_{k}'] for k in carry_keys}
        first_info['steps'] = jp.zeros_like(state.info['steps'])
        curr_info = {k: state.info[k] for k in carry_keys}
        curr_info['steps'] = state.info['steps']
        info = state.info
        # Preserve the true post-step values before reverting on done.
        info['pre_reset_obs'] = state.obs
        for k in carry_keys:
            info[f'pre_reset_{k}'] = state.info[k]
        info.update(jax.tree.map(where_done, first_info, curr_info))
        return state.replace(pipeline_state=pipeline_state, obs=obs, info=info)
