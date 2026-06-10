"""Smoke-test a saved Humanoid dynamics model against real MJX physics.

The script starts the real env and the Infoprop model env from the same reset
state, drives both with the same random action sequence, and reports divergence.

Run:
    JAX_PLATFORMS=cpu PYTHONPATH=. python -m infoprop_jax.envs.humanoid.smoke_saved_model_rollout \
        exp/humanoid_infoprop/0/2026.06.08/231834 --iteration 0 --num-steps 50
"""

from __future__ import annotations

import argparse
import os

import jax
from jax import numpy as jp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from brax.io import model as brax_model
from brax.training.types import Transition
from omegaconf import OmegaConf

from infoprop_jax.envs.humanoid.humanoid_mjx import HumanoidEnv
from infoprop_jax.envs.infoprop_env import InfopropEnv


HUMANOID_STATE_LABELS = [
    "z",
    "roll",
    "pitch",
    "yaw_rate",
    "roll_rate",
    "pitch_rate",
    "body_vx",
    "body_vy",
    "body_vz",
] + [f"joint_qpos_{i}" for i in range(21)] + [f"joint_qvel_{i}" for i in range(21)]

HUMANOID_CONTEXT_LABELS = ["yaw", "x", "y"]


def _stats(name: str, x):
    x = jp.asarray(x)
    print(
        f"{name:28s} min={float(jp.min(x)):.6g} max={float(jp.max(x)):.6g} "
        f"mean={float(jp.mean(x)):.6g} std={float(jp.std(x)):.6g}"
    )


def _max_abs(a, b) -> float:
    return float(jp.max(jp.abs(a - b)))


def _plot_rollout(
    real_phys,
    model_phys,
    real_context,
    model_context,
    real_rewards,
    model_rewards,
    out_path: str,
):
    real_phys = np.asarray(real_phys)
    model_phys = np.asarray(model_phys)
    real_context = np.asarray(real_context)
    model_context = np.asarray(model_context)
    real_rewards = np.asarray(real_rewards)
    model_rewards = np.asarray(model_rewards)

    cmp_len = min(len(real_phys), len(model_phys))
    real_phys = real_phys[:cmp_len]
    model_phys = model_phys[:cmp_len]
    real_context = real_context[:cmp_len]
    model_context = model_context[:cmp_len]
    real_rewards = real_rewards[:cmp_len]
    model_rewards = model_rewards[:cmp_len]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    # Full state overview: 51 model-state dims + 3 context dims + reward.
    labels = HUMANOID_STATE_LABELS + HUMANOID_CONTEXT_LABELS + ["reward"]
    real_full = np.concatenate([real_phys, real_context, real_rewards[:, None]], axis=-1)
    model_full = np.concatenate([model_phys, model_context, model_rewards[:, None]], axis=-1)

    n = real_full.shape[-1]
    ncols = 5
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 2.4 * nrows), squeeze=False)
    fig.suptitle("Humanoid Saved Model Rollout: Model vs Real", fontsize=14)
    steps = np.arange(cmp_len)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        if i >= n:
            ax.axis("off")
            continue
        ax.plot(steps, real_full[:, i], label="real", linewidth=1.6)
        ax.plot(steps, model_full[:, i], label="model", linewidth=1.3, alpha=0.85)
        ax.set_title(labels[i], fontsize=9)
        ax.grid(True, alpha=0.25)
        if i == 0:
            ax.legend(fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _transition_from_state(env: HumanoidEnv, state) -> Transition:
    return Transition(
        observation=jp.concatenate(
            [state.info["phys_state_history"], state.info["act_history"]], axis=-1
        ),
        action=jp.zeros(env.action_size),
        reward=jp.asarray(0.0),
        discount=jp.asarray(1.0),
        next_observation=state.info["physics_state"],
        extras={
            "policy_extras": {},
            "state_extras": {
                "truncation": jp.asarray(0.0),
                "invariant_physics_state": state.info["invariant_physics_state"],
            },
        },
    )


def run(
    log_dir: str,
    iteration: int,
    num_steps: int,
    seed: int,
    action_scale: float,
    plot_path: str | None,
):
    hydra_cfg = OmegaConf.load(os.path.join(log_dir, ".hydra", "config.yaml"))
    env_cfg = hydra_cfg.env
    algo_cfg = hydra_cfg.algorithm

    model_path = os.path.join(log_dir, "model", f"model_state_{iteration}")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(model_path)
    ckpt = brax_model.load_params(model_path)

    real_env = HumanoidEnv(env_cfg)
    model_env = InfopropEnv(
        HumanoidEnv(env_cfg),
        min_log_var=algo_cfg.min_log_var,
        max_log_var=algo_cfg.max_log_var,
    )
    trainer = model_env.init_NN_trainer(
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
    model_env._model_apply_fn = model_state.apply_fn

    rng = jax.random.PRNGKey(seed)
    rng, reset_key, action_key, model_rng = jax.random.split(rng, 4)
    actions = action_scale * jax.random.uniform(
        action_key, (num_steps, real_env.action_size), minval=-1.0, maxval=1.0
    )

    real_state = real_env.reset(reset_key)
    init_transition = _transition_from_state(real_env, real_state)
    model_state_env = model_env.reset_from_buffer(model_rng, init_transition, False)
    model_state_env = model_env.put_in_NN_params_and_rng(
        model_state.params,
        jp.asarray(ckpt["model_obs_mean"]),
        jp.asarray(ckpt["model_obs_std"]),
        jp.asarray(ckpt["next_state_delta_mean"]),
        jp.asarray(ckpt["next_state_delta_std"]),
        jp.asarray(ckpt["per_step_cutoff"]),
        jp.asarray(ckpt["accumulated_cutoff"]),
        jp.asarray(ckpt["binning_entropy"]),
        model_rng,
        model_state_env,
    )

    real_step = jax.jit(real_env.step)
    model_step = jax.jit(model_env.step)

    max_diffs = {
        "obs": 0.0,
        "physics_state": 0.0,
        "context": 0.0,
        "reward": 0.0,
    }
    real_rewards = []
    model_rewards = []
    real_phys = []
    model_phys = []
    real_context = []
    model_context = []
    info_cutoffs = []
    dones = []
    first_done = None

    print(
        f"Saved-model smoke: log_dir={log_dir}, iteration={iteration}, "
        f"steps={num_steps}, seed={seed}, action_scale={action_scale}"
    )
    print(
        f"sizes: model_state={real_env.model_state_size}, context={real_env.context_size}, "
        f"obs={real_env.observation_size}, action={real_env.action_size}"
    )
    _stats("actions", actions)

    for i, action in enumerate(actions):
        real_state = real_step(real_state, action)
        model_state_env = model_step(model_state_env, action)

        max_diffs["obs"] = max(max_diffs["obs"], _max_abs(real_state.obs, model_state_env.obs))
        max_diffs["physics_state"] = max(
            max_diffs["physics_state"],
            _max_abs(
                real_state.info["physics_state"],
                model_state_env.info["physics_state"],
            ),
        )
        max_diffs["context"] = max(
            max_diffs["context"],
            _max_abs(
                real_state.info["invariant_physics_state"],
                model_state_env.info["invariant_physics_state"],
            ),
        )
        max_diffs["reward"] = max(
            max_diffs["reward"], _max_abs(real_state.reward, model_state_env.reward)
        )
        real_rewards.append(real_state.reward)
        model_rewards.append(model_state_env.reward)
        real_phys.append(real_state.info["physics_state"])
        model_phys.append(model_state_env.info["physics_state"])
        real_context.append(real_state.info["invariant_physics_state"])
        model_context.append(model_state_env.info["invariant_physics_state"])
        info_cutoffs.append(model_state_env.info.get("info_cutoff", jp.asarray(0.0)))
        dones.append(model_state_env.done)
        if first_done is None and float(model_state_env.done) > 0.5:
            first_done = i

    real_rewards = jp.asarray(real_rewards)
    model_rewards = jp.asarray(model_rewards)
    info_cutoffs = jp.asarray(info_cutoffs)
    dones = jp.asarray(dones)

    print("\nmax absolute drift over rollout")
    for key, value in max_diffs.items():
        print(f"{key:28s} {value:.9g}")
    print("")
    _stats("real rewards", real_rewards)
    _stats("model rewards", model_rewards)
    _stats("reward diff", model_rewards - real_rewards)
    print(f"model done fraction          {float(jp.mean(dones)):.6g}")
    print(f"model info_cutoff fraction   {float(jp.mean(info_cutoffs)):.6g}")
    print(f"first model done step        {first_done}")
    print("\nfinal state samples")
    print("real physics head ", np.asarray(real_state.info["physics_state"][:9]))
    print("model physics head", np.asarray(model_state_env.info["physics_state"][:9]))
    print("real context      ", np.asarray(real_state.info["invariant_physics_state"]))
    print("model context     ", np.asarray(model_state_env.info["invariant_physics_state"]))

    if plot_path is None:
        plot_path = os.path.join(
            log_dir, "humanoid_saved_model_smoke", f"iter_{iteration}_seed_{seed}.png"
        )
    _plot_rollout(
        jp.asarray(real_phys),
        jp.asarray(model_phys),
        jp.asarray(real_context),
        jp.asarray(model_context),
        real_rewards,
        model_rewards,
        plot_path,
    )
    print(f"\nsaved plot: {plot_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir")
    parser.add_argument("--iteration", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--plot-path", default=None)
    args = parser.parse_args()
    run(
        args.log_dir,
        args.iteration,
        args.num_steps,
        args.seed,
        args.action_scale,
        args.plot_path,
    )


if __name__ == "__main__":
    main()
