"""
Orchestrator for training the probabilistic ensemble dynamics model.

Wraps GaussianEnsembleModel initialisation and the per-step NLL update into a
NamedTuple-based stateless interface compatible with JAX's functional style.
"""
import functools
from typing import Dict, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from jax import numpy as jp
import optax
from flax.training.train_state import TrainState
from typing import Any, Dict, Union
import numpy as np
import flax
from .model_dataset import ModelDataset as Dataset


from .model_update_step import (
    update_nll,
    per_ensemble_nll,
)
from infoprop_jax.algorithms.util.nn.gaussian_env_model import GaussianEnsembleModel

DataType = Union[np.ndarray, Dict[str, "DataType"]]
Params = flax.core.FrozenDict[str, Any]


def _update_jit(
    model: TrainState,
    batch: Dataset,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    next_state_delta_mean: jnp.ndarray,
    next_state_delta_std: jnp.ndarray,
    obs_history: int,
    act_history: int,
    obs_size: int,
    act_size: int,
    dt: float,
    rng: jax.Array,
) -> Tuple[TrainState, Dict[str, float]]:
    """JIT-compiled wrapper: call update_nll and return updated model + normalisation stats."""
    update_rng, rng = jax.random.split(rng)
    return (
        update_nll(model, batch, obs_mean, obs_std, next_state_delta_mean, next_state_delta_std, obs_history,
        act_history,
        obs_size,
        act_size,
        dt,),
        obs_mean,
        obs_std,
        next_state_delta_mean,
        next_state_delta_std,
        update_rng,
    )


def compute_loss(
    model: TrainState,
    batch: Dataset,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    next_state_delta_mean: jnp.ndarray,
    next_state_delta_std: jnp.ndarray,
    obs_history: int,
    act_history: int,
    obs_size: int,
    act_size: int,
    dt: float,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute mean NLL across all ensemble members (for early-stopping / validation)."""
    return per_ensemble_nll(model, batch, obs_mean, obs_std, next_state_delta_mean, next_state_delta_std, obs_history, act_history, obs_size, act_size, dt).mean()


@functools.partial(jax.jit)
def compute_per_ensemble_loss(
    model: TrainState,
    batch: Dataset,
    obs_mean: jax.Array,
    obs_std: jax.Array,
    rng: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    return per_ensemble_nll(model, batch, obs_mean, obs_std)


class ModelTrainer(NamedTuple):
    """Configuration and factory for the ensemble model trainer.

    Fields:
        num_ensemble: Number of ensemble members.
        num_elites: Number of best-performing members to select after training.
        hidden_layer_sizes: MLP hidden dimensions.
        obs_size / act_size / state_size: Dimension of inputs/outputs.
        learning_rate / weight_decay: AdamW optimiser hyperparameters.
    """

    seed: int
    observation_size: Tuple[int]
    action_size: Tuple[int]
    model_lr: float = 3e-4
    model_wd: float = 1e-4
    model_hidden_dims: int = 200
    model_num_layers: int = 4
    n_ensemble: int = 8
    n_elites: int = 5
    patience: int = 10
    model_min_log_var: float = -4
    model_max_log_var: float = -2
    model_layer_norm: bool = True
    obs_history: int = 1
    act_history: int = 0

    def init(self, rng: jax.Array):
        """Initialise the GaussianEnsembleModel and return a Flax TrainState with AdamW optimiser.

        Returns:
            model: Flax TrainState (params + optimiser state).
            elite_indices: Initial elite set (all members).
            obs_mean / obs_std: Zero-initialised normalisation arrays.
            next_state_delta_mean / std: Zero-initialised delta normalisation arrays.
        """
        observations = jp.zeros(self.obs_history * self.observation_size + self.act_history * self.action_size)
        actions = jp.zeros(self.action_size)

        rng, model_key = jax.random.split(rng)

        model_def = GaussianEnsembleModel(
            self.model_hidden_dims,
            self.model_num_layers,
            self.n_ensemble,
            self.observation_size,
            log_min=self.model_min_log_var,
            log_max=self.model_max_log_var,
            model_layer_norm=self.model_layer_norm
        )
        # normalization stats
        dummy_inp = jnp.concatenate([observations, actions], axis=-1)
        _obs_mean = jnp.zeros_like(dummy_inp)
        _obs_std = jnp.ones_like(dummy_inp)
        model_params = model_def.init(
            model_key, observations, actions, _obs_mean, _obs_std
        )["params"]
        model = TrainState.create(
            apply_fn=model_def.apply,
            params=model_params,
            tx=optax.adamw(learning_rate=self.model_lr, weight_decay=self.model_wd),
        )

        _rng = rng
        _elites = jnp.arange(self.n_elites)

        _next_state_delta_mean = jnp.zeros(self.observation_size)
        _next_state_delta_std = jnp.ones(self.observation_size)
        return model, _elites, _obs_mean, _obs_std, _next_state_delta_mean, _next_state_delta_std, _rng

    def update_step(self, batch: Dataset, model, obs_mean, obs_std, next_state_delta_mean, next_state_delta_std, obs_history, act_history, obs_size, act_size, dt, rng) -> Dict[str, float]:
        """Apply one NLL gradient step to the model using the provided batch."""
        train_rng, rng = jax.random.split(rng)
        (new_model, obs_mean, obs_std, next_state_delta_mean, next_state_delta_std, train_rng) = _update_jit(
            model,
            batch,
            obs_mean,
            obs_std,
            next_state_delta_mean,
            next_state_delta_std,
            obs_history,
            act_history,
            obs_size,
            act_size,
            dt,
            train_rng,
        )

        return new_model, obs_mean, obs_std, next_state_delta_mean, next_state_delta_std, train_rng
