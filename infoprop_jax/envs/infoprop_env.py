"""InfopropEnv: the generic, env-agnostic Infoprop Dyna model environment.

`InfopropEnv` is a Brax `Wrapper` that turns any `InfopropWrappable` real MJX env into a
learned-dynamics model environment, replacing physics with a probabilistic ensemble:

    model_env = InfopropEnv(MyEnv(cfg), min_log_var=..., max_log_var=..., fast_model_rollout=...)

It owns the *fixed* Infoprop core math (decode, ensemble fusion + Kalman update, conditional
entropy, sampling, rollout-termination cutoffs, ensemble-trainer wiring) and delegates every
env-specific concern to the wrapped env's hooks (`preprocess`, `augment_prediction`,
`postprocess`). The whole per-sample step lives in ``_single_step``; ``batched_step`` vmaps it
over the rollout batch. Because it is a `Wrapper`, the underlying env's interface (`dt`,
`action_size`, `observation_size`, sizes, the data contract) is forwarded automatically, and
`InfopropEnv` composes uniformly with the Infoprop training wrappers in
`algorithms/util/custom_wrapper.py`.

The learned-model state — the apply function, the ensemble params carried in `info`, and the
decode/fuse math — lives entirely here; the wrapped real env knows nothing about the NN.
"""

import jax
from jax import numpy as jp
from brax.envs.base import Wrapper, State

from infoprop_jax.algorithms.util.model_learning.model_trainer import ModelTrainer


class InfopropEnv(Wrapper):
  """Generic Infoprop model environment wrapping an `InfopropWrappable` env.

  Attributes (model-only; env attributes forward to the wrapped env):
      min_log_var / max_log_var: ensemble log-variance clipping bounds.
      fast_model_rollout: if True, skip building the MJX pipeline_state during rollouts.
      _model_apply_fn: set externally after `init_NN_trainer(...).init(...)`.
  """

  def __init__(self, env, *, min_log_var: float = -4, max_log_var: float = -2,
               fast_model_rollout: bool = True):
    super().__init__(env)  # sets self.env = wrappable
    self.min_log_var = min_log_var
    self.max_log_var = max_log_var
    self.fast_model_rollout = fast_model_rollout

  # ------------------------------------------------------------- fixed core math
  def decode_delta(self, raw_mean, raw_logvar, curr_model_state,
                   next_state_delta_mean, next_state_delta_std):
    """Un-normalise the predicted delta and integrate onto the current state.

    ``raw_mean`` / ``raw_logvar`` are ``[E, model_state_size]``; ``curr_model_state``
    is ``[model_state_size]``. Returns per-member ``(member_mean, member_var)``.
    """
    dt = self.dt
    std = next_state_delta_std + 1e-6
    member_mean = (raw_mean * std + next_state_delta_mean) * dt + curr_model_state
    member_var = jp.exp(raw_logvar) * std ** 2 * dt ** 2
    return member_mean, member_var

  def _fuse(self, member_mean, member_var):
    """Precision-weighted ensemble fusion + Kalman update (single sample, axis 0 = E)."""
    inv_vars = 1 / (member_var + 1e-12)
    fused_var = 1 / jp.mean(inv_vars, axis=0)
    fused_mean = fused_var * jp.mean(member_mean * inv_vars, axis=0)
    epist_var = jp.mean((member_mean - fused_mean[None, :]) ** 2, axis=0)
    kalman_gain = jp.clip(fused_var / (fused_var + epist_var), 0, 1)
    conditional_var = (1 - kalman_gain) * fused_var
    return fused_mean, fused_var, kalman_gain, conditional_var, epist_var

  def infoprop_core(self, member_mean, member_var, binning_entropy, rng):
    """The fixed Infoprop step: fuse, compute conditional entropy, sample next state.

    ``member_mean`` / ``member_var`` are ``[E, full_state_size]`` (single sample).
    Returns ``(next_full_state, conditional_entropy, diagnostics)``.
    """
    fused_mean, fused_var, kalman_gain, conditional_var, epist_var = self._fuse(
        member_mean, member_var)
    conditional_entropy = jp.clip(
        0.5 * jp.log2(2 * jp.pi * jp.e * conditional_var) - binning_entropy, 0, None)
    next_full = fused_mean + jax.random.normal(
        rng, shape=(fused_mean.shape[0],)) * jp.sqrt(fused_var)
    diagnostics = {
        'kalman_gain': kalman_gain,
        'conditional_var': conditional_var,
        'fused_var': fused_var,
        'fused_mean': fused_mean,
        'epist_var': epist_var,
    }
    return next_full, conditional_entropy, diagnostics

  def _predict_member(self, raw_mean, raw_logvar, curr_model_state, curr_context,
                      next_state_delta_mean, next_state_delta_std):
    """decode + env augment for a single sample -> per-member full ``(mean, var)``."""
    member_mean, member_var = self.decode_delta(
        raw_mean, raw_logvar, curr_model_state, next_state_delta_mean, next_state_delta_std)
    return self.env.augment_prediction(member_mean, member_var, curr_model_state, curr_context)

  # ----------------------------------------------------------- model-step entries
  def _single_step(self, state, action, model_params, obs_mean, obs_std,
                   next_state_delta_mean, next_state_delta_std,
                   per_step_cutoff, accumulated_cutoff, binning_entropy):
    """One full learned-dynamics step for a single rollout sample.

    preprocess (env) -> NN -> decode -> augment (env) -> fuse / entropy / sample ->
    framework finalize (rng + entropy accumulation) -> postprocess (env) ->
    entropy-based termination. ``batched_step`` vmaps this over the rollout batch.
    """
    nn_input, curr_model_state, curr_context, applied_action, processed_action = (
        self.env.preprocess(state, action))
    curr_rng, rng = jax.random.split(state.info['rng'])
    raw_mean, raw_logvar = self._model_apply_fn(
        {"params": model_params}, nn_input, applied_action, obs_mean, obs_std)
    member_mean, member_var = self._predict_member(
        raw_mean, raw_logvar, curr_model_state, curr_context,
        next_state_delta_mean, next_state_delta_std)
    next_full, conditional_entropy, _ = self.infoprop_core(
        member_mean, member_var, binning_entropy, curr_rng)
    next_model_state = next_full[:self.model_state_size]
    next_context = next_full[self.model_state_size:]

    # framework-owned: advance rng + accumulate entropy
    info = dict(state.info)
    info['rng'] = rng
    info['accumulated_conditional_entropy'] = (
        state.info['accumulated_conditional_entropy'] + conditional_entropy)
    info['current_conditional_entropy'] = conditional_entropy
    state = state.replace(info=info)

    # env-owned: rebuild the State + reward (action-prep already done in preprocess)
    state = self.env.postprocess(
        state, applied_action, next_model_state, next_context, processed_action,
        not self.fast_model_rollout)

    # entropy-based termination, OR'd with the env's reward-based done
    violation = jp.logical_or(
        (conditional_entropy > per_step_cutoff).any(),
        (state.info['accumulated_conditional_entropy'] > accumulated_cutoff).any())
    info = state.info
    info['info_cutoff'] = jp.where(violation, 1.0 - state.done, 0.0)
    done = jp.where(violation, 1.0, state.done)
    return state.replace(done=done, info=info)

  def batched_step(self, state, action, model_params, obs_mean, obs_std,
                   next_state_delta_mean, next_state_delta_std,
                   per_step_cutoff, accumulated_cutoff, binning_entropy):
    """Vectorise ``_single_step`` over a batch of rollouts. The shared model params,
    normalisation stats and cutoffs are broadcast (in_axes=None), not mapped."""
    return jax.vmap(
        self._single_step,
        in_axes=(0, 0, None, None, None, None, None, None, None, None))(
        state, action, model_params, obs_mean, obs_std,
        next_state_delta_mean, next_state_delta_std,
        per_step_cutoff, accumulated_cutoff, binning_entropy)

  def batched_diff_entropy(self, raw_mean, raw_logvar, curr_model_state, curr_context,
                           next_state_delta_mean, next_state_delta_std):
    """Per-sample differential entropy ``0.5*log2(2*pi*e*conditional_var)`` over a buffer
    of raw NN outputs (no sampling, no binning). Used by the cutoff computation.
    ``raw_mean`` etc. are ``[N, E, model_state_size]``."""

    def one(raw_mean_i, raw_logvar_i, curr_ms_i, curr_ctx_i):
      member_mean, member_var = self._predict_member(
          raw_mean_i, raw_logvar_i, curr_ms_i, curr_ctx_i,
          next_state_delta_mean, next_state_delta_std)
      _, _, _, conditional_var, _ = self._fuse(member_mean, member_var)
      return 0.5 * jp.log2(2 * jp.pi * jp.e * conditional_var)

    return jax.vmap(one)(raw_mean, raw_logvar, curr_model_state, curr_context)

  # ----------------------------------------------------------------- step (eval/single)
  def step(self, state: State, action: jp.ndarray) -> State:
    """Advance one step using the learned model (single, un-batched; eval path).

    The ensemble params, normalisation stats and cutoffs are read from `info`
    (injected via ``put_in_NN_params_and_rng``). The batched training path goes
    through ``batched_step`` instead.
    """
    return self._single_step(
        state, action, state.info['model'], state.info['model_obs_mean'],
        state.info['model_obs_std'], state.info['next_state_delta_mean'],
        state.info['next_state_delta_std'], state.info['per_step_cutoff'],
        state.info['accumulated_cutoff'], state.info['binning_entropy'])

  # ------------------------------------------------------------------- trainer wiring
  def put_in_NN_params_and_rng(self, model, model_obs_mean, model_obs_std,
                               next_state_delta_mean, next_state_delta_std,
                               per_step_cutoff, accumulated_cutoff, binning_entropy,
                               rng, state):
    """Inject the ensemble params, normalisation stats, and entropy cutoffs into info."""
    info = state.info
    info['model'] = model
    info['model_obs_mean'] = model_obs_mean
    info['model_obs_std'] = model_obs_std
    info['next_state_delta_mean'] = next_state_delta_mean
    info['next_state_delta_std'] = next_state_delta_std
    info['per_step_cutoff'] = per_step_cutoff
    info['accumulated_cutoff'] = accumulated_cutoff
    info['binning_entropy'] = binning_entropy
    info['rng'] = rng
    return state.replace(info=info)

  def init_NN_trainer(self, seed, learning_rate, weight_decay, hidden_layer_sizes,
                      model_layer_norm):
    """Instantiate the ensemble ModelTrainer sized from the wrapped env's attributes.

    The caller must set ``self._model_apply_fn = model_state.apply_fn`` after
    ``trainer.init(...)``.
    """
    return ModelTrainer(
        seed=seed,
        observation_size=self.model_state_size,
        action_size=self.action_size,
        model_lr=learning_rate,
        model_wd=weight_decay,
        model_hidden_dims=hidden_layer_sizes[0],
        model_num_layers=len(hidden_layer_sizes) + 1,
        model_min_log_var=self.min_log_var,
        model_max_log_var=self.max_log_var,
        model_layer_norm=model_layer_norm,
        obs_history=self.obs_history,
        act_history=self.act_history,
    )
