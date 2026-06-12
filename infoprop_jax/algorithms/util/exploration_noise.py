"""Temporally correlated latent exploration noise for SAC action sampling.

Generates noise `eta` that is marginally N(0, 1) per step but correlated over
time, so it can replace the i.i.d. standard-normal sample in the tanh-normal
policy without changing the per-step action marginals.

Two flavors:
  - 'ar1': a single normalized AR(1) (Ornstein-Uhlenbeck-like) process.
  - 'pink': a sum of AR(1) processes with octave-spaced time constants
    (tau = 1, 2, 4, ...), giving roughly equal energy per octave, i.e. an
    approximate 1/f spectrum.

The noise state has shape [batch, action_size, num_filters] and is meant to be
carried through the existing rollout scans.
"""

from typing import Callable, Tuple

import jax
import jax.numpy as jnp
from brax.training.types import PRNGKey

NoiseState = jnp.ndarray
InitFn = Callable[[PRNGKey, int, int], NoiseState]
SampleFn = Callable[[NoiseState, PRNGKey, jnp.ndarray], Tuple[NoiseState, jnp.ndarray]]


def make_noise_fns(noise_type: str, beta: float, num_filters: int) -> Tuple[InitFn, SampleFn]:
  """Returns (init_fn, sample_fn) for the requested correlated noise process.

  init_fn(key, batch_size, action_size) -> noise_state sampled from the
  stationary distribution N(0, 1), shape [batch, action_size, k].

  sample_fn(noise_state, key, done) -> (new_noise_state, eta) where eta has
  shape [batch, action_size] and unit marginal variance. `done` (shape
  [batch]) marks envs whose current obs is a fresh episode start; their noise
  state is re-drawn so correlation never crosses an (auto)reset boundary.
  """
  if noise_type == 'ar1':
    alphas = jnp.array([1.0 - beta])
  elif noise_type == 'pink':
    taus = 2.0 ** jnp.arange(num_filters)
    alphas = jnp.exp(-1.0 / taus)
  else:
    raise ValueError(f'Unknown exploration noise type: {noise_type!r}')

  k = alphas.shape[0]
  # AR(1) with this innovation scale has unit stationary variance per filter.
  innovation_scales = jnp.sqrt(1.0 - alphas**2)
  output_scale = 1.0 / jnp.sqrt(k)

  def init_fn(key: PRNGKey, batch_size: int, action_size: int) -> NoiseState:
    return jax.random.normal(key, (batch_size, action_size, k))

  def sample_fn(
      noise_state: NoiseState, key: PRNGKey, done: jnp.ndarray
  ) -> Tuple[NoiseState, jnp.ndarray]:
    reset_key, step_key = jax.random.split(key)
    fresh = jax.random.normal(reset_key, noise_state.shape)
    noise_state = jnp.where(done[:, None, None].astype(bool), fresh, noise_state)
    eps = jax.random.normal(step_key, noise_state.shape)
    noise_state = alphas * noise_state + innovation_scales * eps
    eta = jnp.sum(noise_state, axis=-1) * output_scale
    return noise_state, eta

  return init_fn, sample_fn
