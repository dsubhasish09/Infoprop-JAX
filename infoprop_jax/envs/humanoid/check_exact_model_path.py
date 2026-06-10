"""Sanity-check Humanoid real rollout vs. exact-physics Infoprop hook path.

This replaces the learned model prediction with the original MJX physics step:

    preprocess(state, action)
      -> pipeline_step(current_pipeline_state, applied_action)
      -> extract next model_state/context from exact MJX data
      -> postprocess(...)

If the Humanoid Infoprop hooks are consistent with the real env path, this should
produce the same observations, physics states, context, rewards, dones, applied
actions, and histories as repeated ``env.step(state, action)`` calls for the same
initial state and action sequence.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=. python -m infoprop_jax.envs.humanoid.check_exact_model_path
"""

from __future__ import annotations

import argparse

import jax
from jax import numpy as jp
import numpy as np
from omegaconf import OmegaConf

from infoprop_jax.envs.humanoid.humanoid_mjx import HumanoidEnv


def _max_abs(a, b) -> float:
    return float(jp.max(jp.abs(a - b)))


def _exact_model_step(env: HumanoidEnv, state, action, *, build_pipeline_state: bool):
    """Run the Infoprop hook path with exact MJX physics standing in for the NN."""
    # The env owns whether postprocess builds a pipeline_state; drive it here.
    env.fast_model_rollout = not build_pipeline_state
    _, _, _, applied_action, processed_action = env.preprocess(state, action)
    exact_data = env.pipeline_step(state.pipeline_state, applied_action)
    next_model_state = env._physics_state(exact_data)
    next_context = env._invariant_physics_state(exact_data)
    next_state = env.postprocess(
        state,
        applied_action,
        next_model_state,
        next_context,
        processed_action,
    )
    # In normal fast model rollout, pipeline_state remains None. This check still
    # needs exact MJX data to produce the next exact-physics step, so keep it as
    # private carry state while all fast-path-visible quantities are produced by
    # postprocess under the env's fast_model_rollout=True setting.
    if not build_pipeline_state:
        next_state = next_state.replace(pipeline_state=exact_data)
    return next_state


def _compare(label: str, real_value, exact_value, failures: list[str], tol: float):
    diff = _max_abs(real_value, exact_value)
    print(f"{label:34s} max_abs={diff:.9g}")
    if not np.isfinite(diff) or diff > tol:
        failures.append(f"{label}: {diff} > {tol}")


def _action_sequence(key, num_steps: int, action_size: int, mode: str):
    if mode == "zeros":
        return jp.zeros((num_steps, action_size))
    if mode == "random":
        return jax.random.uniform(
            key, (num_steps, action_size), minval=-1.5, maxval=1.5
        )
    raise ValueError(f"unknown action mode: {mode}")


def run(
    seed: int,
    num_cases: int,
    num_steps: int,
    tol: float,
    build_pipeline_state: bool,
    action_mode: str,
):
    cfg = OmegaConf.load("infoprop_jax/config/env/humanoid.yaml")
    env = HumanoidEnv(cfg)

    failures: list[str] = []
    rng = jax.random.PRNGKey(seed)

    print(
        "Humanoid exact-model-path check "
        f"(cases={num_cases}, steps={num_steps}, tol={tol}, "
        f"build_pipeline_state={build_pipeline_state}, action_mode={action_mode})"
    )
    print(
        f"sizes: model_state={env.model_state_size}, context={env.context_size}, "
        f"obs={env.observation_size}, action={env.action_size}"
    )

    reset_fn = jax.jit(env.reset)
    real_step_fn = jax.jit(env.step)
    exact_step_fn = jax.jit(
        lambda state, action: _exact_model_step(
            env, state, action, build_pipeline_state=build_pipeline_state
        )
    )

    for case in range(num_cases):
        rng, reset_key, action_key = jax.random.split(rng, 3)
        actions = _action_sequence(action_key, num_steps, env.action_size, action_mode)

        # Reset twice with the same key so mutable info dict updates in one path cannot
        # affect the other path.
        real_state = reset_fn(reset_key)
        exact_state = reset_fn(reset_key)

        print(f"\ncase {case}")
        max_diffs: dict[str, float] = {}

        def record(label: str, real_value, exact_value):
            diff = _max_abs(real_value, exact_value)
            max_diffs[label] = max(max_diffs.get(label, 0.0), diff)

        first_bad_step = None
        for step, action in enumerate(actions):
            real_state = real_step_fn(real_state, action)
            exact_state = exact_step_fn(exact_state, action)

            record("obs", real_state.obs, exact_state.obs)
            record(
                "physics_state",
                real_state.info["physics_state"],
                exact_state.info["physics_state"],
            )
            record(
                "invariant_physics_state",
                real_state.info["invariant_physics_state"],
                exact_state.info["invariant_physics_state"],
            )
            record(
                "phys_state_history",
                real_state.info["phys_state_history"],
                exact_state.info["phys_state_history"],
            )
            record(
                "act_history",
                real_state.info["act_history"],
                exact_state.info["act_history"],
            )
            record(
                "applied_action",
                real_state.info["applied_action"],
                exact_state.info["applied_action"],
            )
            record("reward", real_state.reward, exact_state.reward)
            record("done", real_state.done, exact_state.done)
            for key in sorted(real_state.info["reward_metrics"]):
                record(
                    f"reward_metrics.{key}",
                    real_state.info["reward_metrics"][key],
                    exact_state.info["reward_metrics"][key],
                )

            if build_pipeline_state:
                record(
                    "pipeline_state.qpos",
                    real_state.pipeline_state.qpos,
                    exact_state.pipeline_state.qpos,
                )
                record(
                    "pipeline_state.qvel",
                    real_state.pipeline_state.qvel,
                    exact_state.pipeline_state.qvel,
                )

            if first_bad_step is None and any(v > tol for v in max_diffs.values()):
                first_bad_step = step

        for label in sorted(max_diffs):
            diff = max_diffs[label]
            print(f"{label:34s} max_abs_over_rollout={diff:.9g}")
            if not np.isfinite(diff) or diff > tol:
                failures.append(f"case {case} {label}: {diff} > {tol}")
        if first_bad_step is not None:
            print(f"first step over tolerance: {first_bad_step}")

    if failures:
        print("\nFAILED")
        for failure in failures:
            print(f"  {failure}")
        raise SystemExit(1)

    print("\nPASSED")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-cases", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--action-mode", choices=("random", "zeros"), default="random")
    parser.add_argument(
        "--build-pipeline-state",
        action="store_true",
        help="Also rebuild pipeline_state in the exact model path and compare qpos/qvel.",
    )
    args = parser.parse_args()
    run(
        args.seed,
        args.num_cases,
        args.num_steps,
        args.tol,
        args.build_pipeline_state,
        args.action_mode,
    )


if __name__ == "__main__":
    main()
