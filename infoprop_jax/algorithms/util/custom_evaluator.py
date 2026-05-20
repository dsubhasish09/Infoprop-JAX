"""
Custom Brax evaluator that returns mean and standard deviation of episode returns.
"""
# Copyright 2024 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Brax training acting functions."""
from brax.training.acting import Evaluator
import time
from typing import Callable, Union

from brax import envs
from brax.training.types import Metrics
from brax.training.types import Policy
from brax.training.types import PolicyParams
from brax.training.types import PRNGKey
import jax
import numpy as np

State = envs.State
Env = Union[envs.Env, envs.Wrapper]


class CustomEvaluator(Evaluator):
    """Evaluator running parallel episodes and reporting return statistics.

    Wraps a Brax environment with episode tracking and computes mean/std of
    total episode rewards across all parallel environments.
    """

    def __init__(
        self,
        eval_env: envs.Env,
        eval_policy_fn: Callable[[PolicyParams], Policy],
        num_eval_envs: int,
        episode_length: int,
        action_repeat: int,
        key: PRNGKey,
    ):
        """Init.

        Args:
          eval_env: Batched environment to run evals on.
          eval_policy_fn: Function returning the policy from the policy parameters.
          num_eval_envs: Each env will run 1 episode in parallel for each eval.
          episode_length: Maximum length of an episode.
          action_repeat: Number of physics steps per env step.
          key: RNG key.
        """
        super().__init__(eval_env, eval_policy_fn, num_eval_envs, episode_length,
                         action_repeat, key)
        self._initial_key = key

    def run_evaluation(
        self,
        policy_params: PolicyParams,
        training_metrics: Metrics,
        aggregate_episodes: bool = True,
    ) -> Metrics:
        """Run one epoch of evaluation."""
        self._key = self._initial_key
        self._key, unroll_key = jax.random.split(self._key)

        t = time.time()
        eval_state = self._generate_eval_unroll(
            self._eval_state_to_donate, policy_params, unroll_key
        )
        self._eval_state_to_donate = eval_state
        eval_metrics = eval_state.info['eval_metrics']
        eval_metrics.active_episodes.block_until_ready()
        epoch_eval_time = time.time() - t
        metrics = {}
        for fn in [np.mean, np.std, np.histogram]:
            suffix = '_std' if fn == np.std else '_histogram' if fn == np.histogram else ''
            metrics.update({
                f'eval/episode_{name}{suffix}': (
                    fn(value, bins=512) if aggregate_episodes and fn == np.histogram
                    else fn(value) if aggregate_episodes and fn != np.histogram
                    else value
                )
                for name, value in eval_metrics.episode_metrics.items()
            })
        metrics['eval/avg_episode_length'] = np.mean(eval_metrics.episode_steps)
        metrics['eval/epoch_eval_time'] = epoch_eval_time
        metrics['eval/sps'] = self._steps_per_unroll / epoch_eval_time
        self._eval_walltime = self._eval_walltime + epoch_eval_time
        metrics = {
            'eval/walltime': self._eval_walltime,
            **training_metrics,
            **metrics,
        }

        return metrics  # pytype: disable=bad-return-type  # jax-ndarray
