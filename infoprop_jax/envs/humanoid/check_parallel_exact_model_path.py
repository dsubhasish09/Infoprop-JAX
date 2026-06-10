"""Check the parallel model-rollout wrapper path with exact Humanoid physics.

This exercises the same wrapper stack used for model rollouts in training:

    CustomAutoResetWrapper -> CustomEpisodeWrapper -> VmapInfopropWrapper

but replaces the learned ensemble step with an exact MJX step reconstructed from
the model rollout state. It compares the result to the vectorized real-env path
for the same initial states and batched action sequence.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=. python -m infoprop_jax.envs.humanoid.check_parallel_exact_model_path
"""

from __future__ import annotations

import argparse

import jax
from jax import numpy as jp
import numpy as np
from brax.training.types import Transition
from omegaconf import OmegaConf

from infoprop_jax.algorithms.util.custom_wrapper import wrap, wrap_custom
from infoprop_jax.envs.humanoid.humanoid_mjx import HumanoidEnv
from infoprop_jax.envs.infoprop_env import InfopropEnv


class _StaticReplayBuffer:
    """Replay-buffer shim returning a fixed batch of initial transitions."""

    def __init__(self, transitions: Transition):
        self.transitions = transitions

    def get_model_dataset(self, sample_key, buffer_state, max_samples: int):
        return buffer_state, self.transitions


class ExactPhysicsInfopropEnv(InfopropEnv):
    """InfopropEnv with exact MJX physics standing in for the learned model."""

    def _single_step(
        self,
        state,
        action,
        model_params,
        obs_mean,
        obs_std,
        next_state_delta_mean,
        next_state_delta_std,
        per_step_cutoff,
        accumulated_cutoff,
        binning_entropy,
    ):
        _, _, _, applied_action, processed_action = self.env.preprocess(state, action)

        pipeline_state = state.info["exact_pipeline_state"]

        exact_data = self.env.pipeline_step(pipeline_state, applied_action)
        next_model_state = self.env._physics_state(exact_data)
        next_context = self.env._invariant_physics_state(exact_data)

        info = dict(state.info)
        info["current_conditional_entropy"] = jp.zeros((self.full_state_size,))
        info["accumulated_conditional_entropy"] = jp.zeros((self.full_state_size,))
        info["info_cutoff"] = jp.asarray(0.0)
        state = state.replace(info=info)

        next_state = self.env.postprocess(
            state,
            applied_action,
            next_model_state,
            next_context,
            processed_action,
        )
        info = dict(next_state.info)
        info["exact_pipeline_state"] = exact_data
        info["current_conditional_entropy"] = jp.zeros((self.full_state_size,))
        info["accumulated_conditional_entropy"] = jp.zeros((self.full_state_size,))
        info["info_cutoff"] = jp.asarray(0.0)
        return next_state.replace(info=info)


def _make_init_transitions(env: HumanoidEnv, real_state) -> Transition:
    return Transition(
        observation=jp.concatenate(
            [real_state.info["phys_state_history"], real_state.info["act_history"]],
            axis=-1,
        ),
        action=jp.zeros((real_state.obs.shape[0], env.action_size)),
        reward=jp.zeros(real_state.reward.shape),
        discount=jp.ones(real_state.reward.shape),
        next_observation=real_state.info["physics_state"],
        extras={
            "policy_extras": {},
            "state_extras": {
                "truncation": jp.zeros(real_state.reward.shape),
                "invariant_physics_state": real_state.info["invariant_physics_state"],
            },
        },
    )


def _action_sequence(key, num_steps: int, batch_size: int, action_size: int, mode: str, scale: float):
    if mode == "zeros":
        return jp.zeros((num_steps, batch_size, action_size))
    if mode == "random":
        return scale * jax.random.uniform(
            key, (num_steps, batch_size, action_size), minval=-1.0, maxval=1.0
        )
    raise ValueError(mode)


def _max_abs(a, b) -> float:
    return float(jp.max(jp.abs(a - b)))


def _put_dummy_shared_info(state, env: HumanoidEnv, exact_pipeline_state):
    info = dict(state.info)
    info["exact_pipeline_state"] = exact_pipeline_state
    info["model"] = jp.asarray(0.0)
    info["model_obs_mean"] = jp.zeros(env.model_state_size * env.obs_history + env.action_size * env.act_history + env.action_size)
    info["model_obs_std"] = jp.ones_like(info["model_obs_mean"])
    info["next_state_delta_mean"] = jp.zeros(env.model_state_size)
    info["next_state_delta_std"] = jp.ones(env.model_state_size)
    info["per_step_cutoff"] = jp.ones(env.full_state_size) * 1e9
    info["accumulated_cutoff"] = jp.ones(env.full_state_size) * 1e9
    info["binning_entropy"] = jp.zeros(env.full_state_size)
    return state.replace(info=info)


def run(seed: int, batch_size: int, num_steps: int, episode_length: int, action_mode: str, action_scale: float, tol: float):
    cfg = OmegaConf.load("infoprop_jax/config/env/humanoid.yaml")
    real_base = HumanoidEnv(cfg)
    exact_base = HumanoidEnv(cfg)

    real_env = wrap(real_base, episode_length=episode_length, action_repeat=1)

    rng = jax.random.PRNGKey(seed)
    rng, reset_key, action_key, model_reset_key = jax.random.split(rng, 4)
    reset_keys = jax.random.split(reset_key, batch_size)
    real_state = jax.jit(real_env.reset)(reset_keys)

    init_transitions = _make_init_transitions(real_base, real_state)
    replay = _StaticReplayBuffer(init_transitions)
    exact_model_env = ExactPhysicsInfopropEnv(exact_base)
    model_env = wrap_custom(
        exact_model_env,
        replay,
        episode_length=episode_length,
        action_repeat=1,
    )
    model_reset_keys = jax.random.split(model_reset_key, batch_size)
    model_state = model_env.reset(model_reset_keys, None)
    model_state = _put_dummy_shared_info(model_state, exact_base, real_state.pipeline_state)

    real_step = jax.jit(real_env.step)
    model_step = jax.jit(model_env.step)
    actions = _action_sequence(
        action_key, num_steps, batch_size, real_base.action_size, action_mode, action_scale
    )

    max_diffs = {
        "obs": 0.0,
        "reward": 0.0,
        "done": 0.0,
        "physics_state": 0.0,
        "context": 0.0,
        "phys_state_history": 0.0,
        "act_history": 0.0,
    }
    first_bad = None

    print(
        "Parallel exact-model wrapper check "
        f"(batch={batch_size}, steps={num_steps}, episode_length={episode_length}, "
        f"action_mode={action_mode}, action_scale={action_scale}, tol={tol})"
    )

    for i, action in enumerate(actions):
        real_state = real_step(real_state, action)
        model_state = model_step(model_state, action)

        diffs = {
            "obs": _max_abs(real_state.obs, model_state.obs),
            "reward": _max_abs(real_state.reward, model_state.reward),
            "done": _max_abs(real_state.done, model_state.done),
            "physics_state": _max_abs(
                real_state.info["physics_state"], model_state.info["physics_state"]
            ),
            "context": _max_abs(
                real_state.info["invariant_physics_state"],
                model_state.info["invariant_physics_state"],
            ),
            "phys_state_history": _max_abs(
                real_state.info["phys_state_history"],
                model_state.info["phys_state_history"],
            ),
            "act_history": _max_abs(
                real_state.info["act_history"], model_state.info["act_history"]
            ),
        }
        for key, value in diffs.items():
            max_diffs[key] = max(max_diffs[key], value)
        if first_bad is None and any(value > tol for value in diffs.values()):
            first_bad = i

    failures = []
    for key in sorted(max_diffs):
        value = max_diffs[key]
        print(f"{key:24s} max_abs_over_rollout={value:.9g}")
        if not np.isfinite(value) or value > tol:
            failures.append(f"{key}: {value} > {tol}")
    print(f"first step over tolerance: {first_bad}")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)
    print("\nPASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=20)
    parser.add_argument("--episode-length", type=int, default=1000)
    parser.add_argument("--action-mode", choices=("random", "zeros"), default="random")
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    args = parser.parse_args()
    run(
        args.seed,
        args.batch_size,
        args.num_steps,
        args.episode_length,
        args.action_mode,
        args.action_scale,
        args.tol,
    )


if __name__ == "__main__":
    main()
