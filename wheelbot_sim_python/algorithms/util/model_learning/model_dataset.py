"""
Replay buffer and dataset utilities for the dynamics model training data.

Provides:
  - ReplayBufferPhysicsState: stores (obs_history, action, next_state) transitions
    collected from the real MJX environment.
  - ModelDataset: a thin wrapper around Transition that adds epoch-iteration support
    for supervised model training.
"""
from brax.training import replay_buffers
import jax.numpy as jp
import jax
from brax.training.types import Transition


class ReplayBufferPhysicsState(replay_buffers.UniformSamplingQueue):
    """Replay buffer for real-world physics transitions.

    Extends Brax's UniformSamplingQueue with helpers to convert the raw buffer
    contents into ModelDataset batches for supervised model training.
    """

    def get_model_dataset(self, sample_key, buffer_state: replay_buffers.ReplayBufferState, max_samples: int = 2560):
        """Sample `num_samples` transitions uniformly and return as a ModelDataset."""
        idx = jax.random.randint(
            sample_key,
            (max_samples,),
            minval=buffer_state.sample_position,
            maxval=buffer_state.insert_position,
        )
        batch = jp.take(buffer_state.data, idx, axis=0, mode='wrap')
        transitions = self._unflatten_fn(batch)
        model_dataset = ModelDataset(
            observation=transitions.observation,
            action=transitions.action,
            reward=transitions.reward,
            discount=transitions.discount,
            next_observation=transitions.next_observation,
            extras=transitions.extras,
        )
        return buffer_state, model_dataset

    def get_all(self, buffer_state, batch_size: int = 0):
        """Return all transitions in the buffer as a ModelDataset, optionally batched."""
        size = buffer_state.insert_position - buffer_state.sample_position
        batch_size = jp.where(batch_size == 0, size, batch_size)
        num_batches = size // batch_size
        idx = jp.arange(buffer_state.sample_position, buffer_state.sample_position + num_batches * batch_size)
        batch = jp.take(buffer_state.data, idx, axis=0, mode='wrap')
        transitions = self._unflatten_fn(batch)
        model_dataset = ModelDataset(
            observation=transitions.observation,
            action=transitions.action,
            reward=transitions.reward,
            discount=transitions.discount,
            next_observation=transitions.next_observation,
            extras=transitions.extras,
        )
        return buffer_state, model_dataset


class ModelDataset(Transition):
    """Transition subclass with an epoch-iteration generator for SGD training."""

    def get_epoch_iter(self, batch_size: int):
        """Yield shuffled mini-batches of size `batch_size` for one training epoch."""
        num_batches = self.observation.shape[0] // batch_size
        random_idxs = jp.arange(self.observation.shape[0])
        for i in range(num_batches):
            idxs = random_idxs[i * batch_size : (i + 1) * batch_size]
            sample = ModelDataset(
                observation=self.observation[idxs],
                action=self.action[idxs],
                reward=self.reward[idxs],
                discount=self.discount[idxs],
                next_observation=self.next_observation[idxs],
                extras={}
            )
            yield sample
