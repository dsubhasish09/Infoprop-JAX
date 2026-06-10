"""Smoke-test saved Humanoid model through the parallel training wrapper path.

This uses the actual model rollout stack used by training:

    CustomAutoResetWrapper -> CustomEpisodeWrapper -> VmapInfopropWrapper -> InfopropEnv

and compares it against the vectorized real env wrapper for the same initial batch
and action sequence. A comparison plot is saved for one selected batch index.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=. MPLCONFIGDIR=/tmp/mpl \
    python -m infoprop_jax.envs.humanoid.smoke_parallel_saved_model_rollout \
      exp/humanoid_infoprop/0/2026.06.08/231834 --iteration 0 --num-steps 50
"""

from __future__ import annotations

import argparse
import os

import jax
from jax import numpy as jp
import numpy as np
from brax.io import model as brax_model
from brax.training.types import Transition
from omegaconf import OmegaConf

from infoprop_jax.algorithms.util.custom_wrapper import wrap, wrap_custom
from infoprop_jax.envs.humanoid.humanoid_mjx import HumanoidEnv
from infoprop_jax.envs.infoprop_env import InfopropEnv
from infoprop_jax.envs.humanoid.smoke_saved_model_rollout import _plot_rollout, _stats


class _StaticReplayBuffer:
    def __init__(self, transitions: Transition):
        self.transitions = transitions

    def get_model_dataset(self, sample_key, buffer_state, max_samples: int):
        return buffer_state, self.transitions


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


def _tree_repeat(x, n):
    return jax.tree_util.tree_map(lambda y: jp.repeat(y[None], n, axis=0), x)


def _max_abs(a, b) -> float:
    return float(jp.max(jp.abs(a - b)))


def run(
    log_dir: str,
    iteration: int,
    num_steps: int,
    seed: int,
    batch_size: int,
    action_scale: float,
    plot_index: int,
    plot_path: str | None,
):
    hydra_cfg = OmegaConf.load(os.path.join(log_dir, ".hydra", "config.yaml"))
    env_cfg = hydra_cfg.env
    algo_cfg = hydra_cfg.algorithm

    model_path = os.path.join(log_dir, "model", f"model_state_{iteration}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    ckpt = brax_model.load_params(model_path)

    real_base = HumanoidEnv(env_cfg)
    model_base = HumanoidEnv(env_cfg)
    real_env = wrap(real_base, episode_length=algo_cfg.episode_length, action_repeat=1)

    rng = jax.random.PRNGKey(seed)
    rng, reset_key, model_reset_key, action_key, rollout_rng = jax.random.split(rng, 5)
    reset_keys = jax.random.split(reset_key, batch_size)
    real_state = jax.jit(real_env.reset)(reset_keys)

    replay = _StaticReplayBuffer(_make_init_transitions(real_base, real_state))
    infoprop_env = InfopropEnv(
        model_base,
        min_log_var=algo_cfg.min_log_var,
        max_log_var=algo_cfg.max_log_var,
    )
    trainer = infoprop_env.init_NN_trainer(
        seed=0,
        learning_rate=algo_cfg.model_learning_rate,
        weight_decay=algo_cfg.model_weight_decay,
        hidden_layer_sizes=tuple(algo_cfg.model_hidden_layer_sizes),
        model_layer_norm=algo_cfg.model_layer_norm,
    )
    template_state, _, _, _, _, _, _ = trainer.init(jax.random.PRNGKey(0))
    model_state = template_state.replace(
        params=jax.tree_util.tree_map(jp.asarray, ckpt["params"])
    )
    infoprop_env._model_apply_fn = model_state.apply_fn

    model_env = wrap_custom(
        infoprop_env,
        replay,
        episode_length=algo_cfg.episode_length,
        action_repeat=1,
    )
    model_reset_keys = jax.random.split(model_reset_key, batch_size)
    model_state_env = model_env.reset(model_reset_keys, None)

    info = dict(model_state_env.info)
    info["model"] = model_state.params
    shared_fields = {
        "model_obs_mean": jp.asarray(ckpt["model_obs_mean"]),
        "model_obs_std": jp.asarray(ckpt["model_obs_std"]),
        "next_state_delta_mean": jp.asarray(ckpt["next_state_delta_mean"]),
        "next_state_delta_std": jp.asarray(ckpt["next_state_delta_std"]),
        "per_step_cutoff": jp.asarray(ckpt["per_step_cutoff"]),
        "accumulated_cutoff": jp.asarray(ckpt["accumulated_cutoff"]),
        "binning_entropy": jp.asarray(ckpt["binning_entropy"]),
    }
    info.update(shared_fields)
    info["rng"] = jax.random.split(rollout_rng, batch_size)
    model_state_env = model_state_env.replace(info=info)

    actions = action_scale * jax.random.uniform(
        action_key, (num_steps, batch_size, real_base.action_size), minval=-1.0, maxval=1.0
    )

    real_step = jax.jit(real_env.step)
    model_step = jax.jit(model_env.step)

    real_phys = []
    model_phys = []
    real_context = []
    model_context = []
    real_rewards = []
    model_rewards = []
    info_cutoffs = []
    dones = []
    max_diffs = {"obs": 0.0, "physics_state": 0.0, "context": 0.0, "reward": 0.0}

    print(
        "Parallel saved-model smoke: "
        f"log_dir={log_dir}, iteration={iteration}, batch={batch_size}, "
        f"steps={num_steps}, seed={seed}, action_scale={action_scale}"
    )
    print(
        f"sizes: model_state={real_base.model_state_size}, context={real_base.context_size}, "
        f"obs={real_base.observation_size}, action={real_base.action_size}"
    )
    _stats("actions", actions)

    for action in actions:
        real_state = real_step(real_state, action)
        model_state_env = model_step(model_state_env, action)

        max_diffs["obs"] = max(max_diffs["obs"], _max_abs(real_state.obs, model_state_env.obs))
        max_diffs["physics_state"] = max(
            max_diffs["physics_state"],
            _max_abs(real_state.info["physics_state"], model_state_env.info["physics_state"]),
        )
        max_diffs["context"] = max(
            max_diffs["context"],
            _max_abs(
                real_state.info["invariant_physics_state"],
                model_state_env.info["invariant_physics_state"],
            ),
        )
        max_diffs["reward"] = max(max_diffs["reward"], _max_abs(real_state.reward, model_state_env.reward))

        real_phys.append(real_state.info["physics_state"])
        model_phys.append(model_state_env.info["physics_state"])
        real_context.append(real_state.info["invariant_physics_state"])
        model_context.append(model_state_env.info["invariant_physics_state"])
        real_rewards.append(real_state.reward)
        model_rewards.append(model_state_env.reward)
        info_cutoffs.append(model_state_env.info["info_cutoff"])
        dones.append(model_state_env.done)

    real_phys = jp.asarray(real_phys)
    model_phys = jp.asarray(model_phys)
    real_context = jp.asarray(real_context)
    model_context = jp.asarray(model_context)
    real_rewards = jp.asarray(real_rewards)
    model_rewards = jp.asarray(model_rewards)
    info_cutoffs = jp.asarray(info_cutoffs)
    dones = jp.asarray(dones)

    print("\nmax absolute drift over batched rollout")
    for key, value in max_diffs.items():
        print(f"{key:28s} {value:.9g}")
    print("")
    _stats("real rewards", real_rewards)
    _stats("model rewards", model_rewards)
    _stats("reward diff", model_rewards - real_rewards)
    print(f"model done fraction          {float(jp.mean(dones)):.6g}")
    print(f"model info_cutoff fraction   {float(jp.mean(info_cutoffs)):.6g}")

    plot_index = int(plot_index)
    if not 0 <= plot_index < batch_size:
        raise ValueError(f"plot_index must be in [0, {batch_size}), got {plot_index}")
    if plot_path is None:
        plot_path = os.path.join(
            log_dir,
            "humanoid_parallel_saved_model_smoke",
            f"iter_{iteration}_seed_{seed}_idx_{plot_index}.png",
        )
    _plot_rollout(
        real_phys[:, plot_index],
        model_phys[:, plot_index],
        real_context[:, plot_index],
        model_context[:, plot_index],
        real_rewards[:, plot_index],
        model_rewards[:, plot_index],
        plot_path,
    )
    print(f"\nsaved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--plot-index", type=int, default=0)
    parser.add_argument("--plot-path", default=None)
    args = parser.parse_args()
    run(
        args.log_dir,
        args.iteration,
        args.num_steps,
        args.seed,
        args.batch_size,
        args.action_scale,
        args.plot_index,
        args.plot_path,
    )


if __name__ == "__main__":
    main()
