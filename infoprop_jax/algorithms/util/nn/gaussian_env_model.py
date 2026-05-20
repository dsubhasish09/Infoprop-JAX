"""
Probabilistic ensemble dynamics model for InfoProp Dyna.

Implements a probabilistic ensemble of E neural networks {p_e}, each predicting
a Gaussian distribution over next-state deltas:
    p_e(s_{t+1} | s_t, a_t) = N(mu_e(s_t, a_t), Sigma_e(s_t, a_t))

All E members share the same MLP architecture but are trained independently
(different parameter initialisations; the training loop uses separate dataset batches).
"""
from typing import Optional, Sequence

# import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from .mlp import MLP


class GaussianEnsembleModel(nn.Module):
    """Flax module implementing a probabilistic ensemble of MLP dynamics models.

    Each of the `num_ensemble` members maps (observation, action) -> (mean, logvar)
    over the normalised next-state delta. The forward pass is vmapped over ensemble
    members so all E predictions are computed in one JIT-compiled call.

    Attributes:
        hidden_dims: Width of each hidden layer.
        num_layers: Number of hidden layers per member.
        num_ensemble: Number of ensemble members.
        output_dim: Dimensionality of the predicted state delta.
        log_min / log_max: Clipping bounds for predicted log-variance.
        model_layer_norm: Whether to apply layer normalisation in the MLP.
    """

    hidden_dims: int
    num_layers: int
    num_ensemble: int
    output_dim: int
    dropout_rate: Optional[float] = None
    log_min: Optional[float] = -10
    log_max: Optional[float] = -4
    low: Optional[jnp.ndarray] = None
    high: Optional[jnp.ndarray] = None
    add_weight_norm: bool = False
    model_layer_norm: bool = True

    @nn.compact
    def __call__(
        self,
        observations: jnp.ndarray,
        action: jnp.ndarray,
        mean: jnp.ndarray,
        std: jnp.ndarray,
        training: bool = False,
    ):
        """Forward pass: predict (means, logvars) for all ensemble members.

        Args:
            x: Observation history (batch_size, obs_dim).
            action: Applied torques (batch_size, act_dim).
            obs_mean / obs_std: Running statistics for input normalisation.

        Returns:
            means:   (num_ensemble, batch_size, output_dim) — predicted next-state deltas.
            logvars: (num_ensemble, batch_size, output_dim) — log-variance of predictions.
        """
        layers = [self.hidden_dims] * (self.num_layers - 1)
        state = jnp.concatenate([observations, action], axis=-1)
        if len(state.shape) < 2 or state.shape[-2] != self.num_ensemble:
            state = jnp.expand_dims(state, axis=-2).repeat(self.num_ensemble, axis=-2)
        # Normalise input: (obs - mean) / (std + eps)
        state_inp = (state - mean) / (std + 1e-6)
        # Vmap applies one MLP forward pass per ensemble member in parallel
        outputs = nn.vmap(
            MLP,
            variable_axes={"params": 0},
            split_rngs={"params": True},
            in_axes=(-2, None),
            out_axes=-2,
            axis_size=self.num_ensemble,
        )(
            layers,
            activations=nn.silu,
            activate_final=True,
            dropout_rate=self.dropout_rate,
            add_weight_norm=self.add_weight_norm,
            layer_norm=self.model_layer_norm,
        )(
            state_inp, training
        )

        means = nn.Dense(self.output_dim)(outputs)

        # offset_state = jnp.concatenate(
        #     [observations, jnp.zeros_like(observations[..., :1])], axis=-1
        # )
        # if len(offset_state.shape) < 2 or offset_state.shape[-2] != self.num_ensemble:
        #     offset_state = jnp.expand_dims(offset_state, axis=-2).repeat(
        #         self.num_ensemble, axis=-2
        #     )

        # means_and_rewards = means_and_rewards + offset_state

        # Separate head for log-variance; clipped to [log_min, log_max]
        logvar = nn.Dense(self.output_dim)(outputs)

        logvar = self.log_max - nn.softplus(self.log_max - logvar)
        logvar = self.log_min + nn.softplus(logvar - self.log_min)
        return means, logvar
