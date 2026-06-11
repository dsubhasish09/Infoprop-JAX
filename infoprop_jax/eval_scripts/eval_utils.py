# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""Shared helpers for the checkpoint evaluation scripts (video_eval*)."""

import os

import jax
import jax.numpy as jnp
from brax.training.acme import running_statistics
from brax.training.agents.sac import networks as sac_networks


def resolve_iteration(log_dir, requested):
    """Pick the checkpoint index: validate `requested`, or take the latest found."""
    policy_dir = os.path.join(log_dir, 'policy')
    if not os.path.isdir(policy_dir):
        raise FileNotFoundError(f'Policy directory not found: {policy_dir}')
    indices = []
    for name in os.listdir(policy_dir):
        if name.startswith('brax_policy_'):
            try:
                indices.append(int(name.split('_')[-1]))
            except ValueError:
                pass
    if not indices:
        raise FileNotFoundError(f'No brax_policy_* files found in {policy_dir}')
    if requested is not None:
        if requested not in indices:
            raise ValueError(
                f'Requested iteration {requested} not found in {policy_dir} '
                f'(available: {sorted(indices)})')
        return requested
    return max(indices)


def infer_sizes(policy_params):
    """Infer (obs_size, action_size) from the saved SAC policy MLP shapes."""
    layers = policy_params['params']
    sorted_keys = sorted(
        (k for k in layers if k.startswith('hidden_')),
        key=lambda k: int(k.split('_')[1]))
    obs_size = layers[sorted_keys[0]]['kernel'].shape[0]
    # SAC last layer outputs [mean, log_std] concatenated → 2 * action_size
    action_size = layers[sorted_keys[-1]]['kernel'].shape[1] // 2
    return obs_size, action_size


def build_policy_inference(algo_cfg, obs_size, action_size,
                           normalizer_params, policy_params):
    """Rebuild the SAC inference function for a saved policy.

    Returns ``(params, jit_inf)`` with ``jit_inf(params, obs, rng) -> (action, extras)``
    (deterministic policy).
    """
    normalize_fn = (running_statistics.normalize
                    if algo_cfg.normalize_observations else (lambda x, y: x))
    sac_net = sac_networks.make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        hidden_layer_sizes=tuple(algo_cfg.agent_hidden_layer_sizes),
        # agent_layer_norm fallback covers configs saved before the per-network
        # layer-norm keys (and its later removal).
        policy_network_layer_norm=algo_cfg.get(
            'policy_network_layer_norm', algo_cfg.get('agent_layer_norm', False)),
        q_network_layer_norm=algo_cfg.get(
            'q_network_layer_norm', algo_cfg.get('agent_layer_norm', False)),
    )
    make_policy = sac_networks.make_inference_fn(sac_net)
    params = (normalizer_params, policy_params)

    @jax.jit
    def jit_inf(p, obs, rng):
        return make_policy(p, deterministic=True)(obs, rng)

    return params, jit_inf


def wire_model_env(env_model, algo_cfg, model_ckpt):
    """Instantiate the ensemble for an `InfopropEnv` and load checkpointed params.

    Sets ``env_model._model_apply_fn`` and returns the Flax TrainState holding the
    loaded ensemble params.
    """
    model_trainer = env_model.init_NN_trainer(
        seed=0, learning_rate=1e-3, weight_decay=1e-4,
        hidden_layer_sizes=tuple(algo_cfg.model_hidden_layer_sizes),
        model_layer_norm=algo_cfg.model_layer_norm,
    )
    model_state_template, _, _, _, _, _, _ = model_trainer.init(jax.random.PRNGKey(0))
    model_state = model_state_template.replace(
        params=jax.tree_util.tree_map(jnp.array, model_ckpt['params']))
    env_model._model_apply_fn = model_state.apply_fn
    return model_state


def inject_model_params(env_model, model_state, model_ckpt, rng, state):
    """Put the loaded ensemble params, normalisation stats and cutoffs into `info`."""
    g = lambda k: jnp.array(model_ckpt[k])
    return env_model.put_in_NN_params_and_rng(
        model_state.params, g('model_obs_mean'), g('model_obs_std'),
        g('next_state_delta_mean'), g('next_state_delta_std'),
        g('per_step_cutoff'), g('accumulated_cutoff'), g('binning_entropy'),
        rng, state,
    )
