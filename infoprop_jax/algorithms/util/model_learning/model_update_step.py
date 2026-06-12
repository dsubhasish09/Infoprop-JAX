"""
NLL-based gradient update for one ensemble member.

Provides the per-member training loss (negative log-likelihood under a diagonal
Gaussian) and the gradient step that updates a single Flax TrainState.
"""
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

import flax
from typing import Any, Dict, Union
import numpy as np
from brax.training.types import Transition

DataType = Union[np.ndarray, Dict[str, "DataType"]]
Params = flax.core.FrozenDict[str, Any]

_LOG_2PI = float(jnp.log(2 * jnp.pi))


def update_nll(
    model: TrainState,
    batch: Transition,
    obs_mean: jnp.ndarray,
    obs_std: jnp.ndarray,
    next_state_delta_mean: jnp.ndarray,
    next_state_delta_std: jnp.ndarray,
    obs_history: int,
    act_history: int,
    obs_size: int,
    act_size: int,
    dt: float,
) -> Tuple[TrainState, Dict[str, float]]:
    """Compute the NLL loss and apply one gradient step to the dynamics model.

    Loss per state dimension:
        L = 0.5 * [(target - mean)^2 * exp(-logvar) + logvar + log(2*pi)]
    where target = (next_state - state) / dt, normalised by the running delta stats.

    Args:
        model: Flax TrainState for one ensemble member.
        batch: Transition batch (observation, action, next_observation).
        obs_mean / obs_std: Input normalisation statistics.
        next_state_delta_mean / std: Output normalisation statistics.
        dt: Control timestep used to convert (s_{t+1} - s_t) into a rate.

    Returns:
        Updated TrainState after one AdamW step.
    """
    state = batch.observation
    action = batch.action
    next_state = batch.next_observation
    # Target: normalised next-state delta = ((s_{t+1} - s_t) / dt - delta_mean) / delta_std
    target_state = ((next_state - state[:, :, (obs_history - 1) * obs_size: obs_history * obs_size]) / dt - next_state_delta_mean) / (next_state_delta_std + 1e-6)  # target is delta_state

    def model_loss_fn(model_params: Params) -> Tuple[jnp.ndarray, Dict[str, float]]:
        mean, logvar = model.apply_fn(
            {"params": model_params}, state, action, obs_mean, obs_std
        )
        # Negative log-likelihood for a diagonal Gaussian (heteroscedastic)
        state_loss = 0.5 * (jnp.square(target_state - mean) * jnp.exp(-logvar) + logvar + _LOG_2PI).mean(axis=(0, 1)).sum()
        return state_loss

    grads = jax.grad(model_loss_fn)(model.params)
    new_model = model.apply_gradients(grads=grads)
    return new_model


def per_ensemble_nll(
    model: TrainState,
    batch: Transition,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    next_state_delta_mean: jnp.ndarray,
    next_state_delta_std: jnp.ndarray,
    obs_history: int,
    act_history: int,
    obs_size: int,
    act_size: int,
    dt: float,
) -> jax.Array:
    """Compute per-ensemble NLL for validation (no gradient, all members in parallel)."""
    state = batch.observation
    action = batch.action
    next_state = batch.next_observation
    target_state = ((next_state - state[:, (obs_history - 1) * obs_size: obs_history * obs_size]) / dt - next_state_delta_mean) / (next_state_delta_std + 1e-6)  # target is delta_state
    mean, logvar = model.apply_fn(
        {"params": model.params}, state, action, obs_mean, obs_std
    )
    state_loss = 0.5 * (jnp.square(target_state[:, jnp.newaxis, :] - mean) * jnp.exp(-logvar) + logvar + _LOG_2PI).mean(axis=(0, 1)).sum()
    return state_loss
