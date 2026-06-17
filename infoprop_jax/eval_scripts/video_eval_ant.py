# Copyright (c) 2026 Devdutt Subhasish
# SPDX-License-Identifier: MIT
"""Video evaluation for InfoProp Dyna checkpoints (Ant).

Renders a real-physics rollout of a saved policy, then replays the same action
sequence through the learned model environment (branched from the real initial
state via ``reset_from_buffer``) and compares the two.

Unlike the wheelbot/humanoid evals there is no model video: Ant is wrapped via
``DefaultInfopropWrappable`` (model state == observation), and model rollouts
emit ``pipeline_state=None`` because there is no inverse map from observation to
qpos/qvel. So this produces real.mp4 and a physics comparison plot only — the
model rollout still runs, it just isn't rendered.

Can be invoked in two ways:

1. Via the Hydra main entry point (recommended) — ant checkpoints are
   auto-detected from the saved training config:
       python -m infoprop_jax.main video_eval=true eval.log_dir=exp/.../135429

2. As a standalone script:
       python -m infoprop_jax.eval_scripts.video_eval_ant <log_dir> [options]
"""

import argparse
import os
import sys
import time

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mediapy as media
import numpy as np
from brax.io import model as brax_model
from brax.training.types import Transition
from omegaconf import OmegaConf

from infoprop_jax.envs.infoprop_env import InfopropEnv
from infoprop_jax.envs.quadruped.ant import AntEnv
from infoprop_jax.eval_scripts.eval_utils import (
    build_policy_inference,
    infer_sizes,
    inject_model_params,
    resolve_iteration,
    wire_model_env,
)

_CAMERA = 'track'
# (label, index into the 26-D ant observation). Positions (x, y) are excluded
# from the obs by default, so only z is available among the root translations.
_ROOT = [
    ('z', 0), ('yaw', 1), ('roll', 2), ('pitch', 3),
    ('yaw_rate', 15), ('roll_rate', 16), ('pitch_rate', 17),
    ('body_vx', 12), ('body_vy', 13), ('body_vz', 14),
]
_N_JOINTS = 8  # qpos at obs[4:12], qvel at obs[18:26]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _do_real_rollout(jit_step, jit_inf, params, state, rng, max_steps):
    ps_list, phys_list, ctrl_list, rew_list = [], [], [], []
    for step in range(max_steps):
        rng, curr_rng = jax.random.split(rng)
        ctrl, _ = jit_inf(params, state.obs, curr_rng)
        ns = jit_step(state, ctrl)
        ps_list.append(ns.pipeline_state)
        phys_list.append(np.array(ns.info['physics_state']))
        ctrl_list.append(np.array(ctrl))
        rew_list.append(float(ns.reward))
        state = ns
        if (step + 1) % 500 == 0:
            print(f'  real step {step + 1}', flush=True)
        if float(ns.done) > 0.5:
            print(f'  real rollout done (unhealthy) at step {step + 1}, '
                  f'z={float(ns.info["physics_state"][0]):.3f}', flush=True)
            break
    return ps_list, np.stack(phys_list), np.stack(ctrl_list), np.array(rew_list)


def _do_model_rollout(jit_step_model, state, ctrls):
    phys_list, rew_list = [], []
    for step, ctrl in enumerate(ctrls):
        ns = jit_step_model(state, jnp.array(ctrl))
        phys_list.append(np.array(ns.info['physics_state']))
        rew_list.append(float(ns.reward))
        state = ns
        if (step + 1) % 500 == 0:
            print(f'  model step {step + 1}', flush=True)
        if float(ns.done) > 0.5:
            print(f'  model rollout done at step {step + 1} '
                  f'(info_cutoff={float(ns.info["info_cutoff"]):.0f}, '
                  f'z={float(ns.info["physics_state"][0]):.3f})', flush=True)
            break
    return np.stack(phys_list), np.array(rew_list)


def _physics_plot(real_phys, model_phys, real_rew, model_rew, out_path, iteration):
    cmp_len = min(len(real_phys), len(model_phys))
    real_phys, model_phys = real_phys[:cmp_len], model_phys[:cmp_len]
    qpos_mae = np.abs(real_phys[:, 4:4 + _N_JOINTS]
                      - model_phys[:, 4:4 + _N_JOINTS]).mean(axis=1)
    qvel_mae = np.abs(real_phys[:, 18:18 + _N_JOINTS]
                      - model_phys[:, 18:18 + _N_JOINTS]).mean(axis=1)

    fig, axes = plt.subplots(4, 4, figsize=(32, 16))
    fig.suptitle(f'Physics State Comparison: Model vs Real (Iter {iteration})', fontsize=16)
    for i, (label, idx) in enumerate(_ROOT):
        ax = axes[i // 4, i % 4]
        ax.plot(real_phys[:, idx],  label=f'{label}_real',  alpha=0.8, color='blue', linewidth=2)
        ax.plot(model_phys[:, idx], label=f'{label}_model', alpha=0.8, color='red',  linewidth=2)
        ax.set_title(label, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
        ax.set_xlabel('Time Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)

    for ax, vals, title in [(axes[2, 2], qpos_mae, f'joint qpos MAE ({_N_JOINTS} joints)'),
                            (axes[2, 3], qvel_mae, f'joint qvel MAE ({_N_JOINTS} joints)')]:
        ax.plot(vals, color='purple', linewidth=2)
        ax.set_title(title, fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Time Step', fontsize=8)
        ax.set_ylabel('MAE', fontsize=8)

    ax = axes[3, 0]
    ax.plot(real_rew,  label='reward_real',  alpha=0.8, color='blue', linewidth=2)
    ax.plot(model_rew, label='reward_model', alpha=0.8, color='red',  linewidth=2)
    ax.set_title('reward', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('Time Step', fontsize=8)
    ax.set_ylabel('Value', fontsize=8)
    for ax in (axes[3, 1], axes[3, 2], axes[3, 3]):
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def _init_transition_from_state(env, state):
    """Branch-point transition for ``reset_from_buffer``: only ``observation`` (the
    concatenated history) is read by ``DefaultInfopropWrappable.reset_from_buffer``;
    the remaining fields are filler to satisfy the Transition layout."""
    return Transition(
        observation=jnp.concatenate(
            [state.info['phys_state_history'], state.info['act_history']], axis=-1),
        action=jnp.zeros(env.action_size),
        reward=jnp.asarray(0.0),
        discount=jnp.asarray(1.0),
        next_observation=state.info['physics_state'],
        extras={
            'policy_extras': {},
            'state_extras': {'truncation': jnp.asarray(0.0)},
        },
    )


# ── Main entry point ──────────────────────────────────────────────────────────

def run_loaded(eval_cfg, log_dir, env_cfg, algo_cfg):
    """Run ant video evaluation with a pre-loaded training config."""
    # ── Resolve iteration / output dir ────────────────────────────────────────
    iteration = resolve_iteration(log_dir, eval_cfg.iteration)
    print(f'Evaluating ant checkpoint, iteration {iteration}', flush=True)

    output_dir = eval_cfg.output_dir or os.path.join(log_dir, 'video_eval', f'iter_{iteration}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Outputs → {output_dir}', flush=True)

    # ── Load policy ───────────────────────────────────────────────────────────
    policy_path = os.path.join(log_dir, 'policy', f'brax_policy_{iteration}')
    print(f'Loading policy from {policy_path}', flush=True)
    normalizer_params, policy_params = brax_model.load_params(policy_path)
    obs_size, action_size = infer_sizes(policy_params)
    print(f'  obs_size={obs_size}  action_size={action_size}', flush=True)

    # ── Build real env ────────────────────────────────────────────────────────
    print('Building real environment...', flush=True)
    t0 = time.time()
    env_real = AntEnv(cfg=env_cfg)
    jit_reset_real = jax.jit(env_real.reset)
    jit_step_real  = jax.jit(env_real.step)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # ── Build SAC inference fn ────────────────────────────────────────────────
    params, jit_inf = build_policy_inference(
        algo_cfg, obs_size, action_size, normalizer_params, policy_params)

    # ── Real rollout ──────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(eval_cfg.rng_seed)
    rng, reset_rng, eval_rng, model_reset_rng, model_env_rng = jax.random.split(rng, 5)

    print('Resetting real env and running rollout...', flush=True)
    t0 = time.time()
    init_state  = jit_reset_real(reset_rng)
    init_r_ps   = init_state.pipeline_state
    init_r_phys = np.array(init_state.info['physics_state'])

    ps_r, phys_r, ctrls, rew_r = _do_real_rollout(
        jit_step_real, jit_inf, params, init_state, eval_rng, eval_cfg.max_steps)
    rlen_r = len(ps_r)
    real_phys_arr = np.concatenate([init_r_phys[None], phys_r], axis=0)
    print(f'Real rollout: {rlen_r} steps in {time.time()-t0:.1f}s '
          f'(return {rew_r.sum():.2f})', flush=True)

    # ── Render real video ─────────────────────────────────────────────────────
    print(f'Rendering real video ({rlen_r+1} frames)...', flush=True)
    t0 = time.time()
    frames = env_real.render([init_r_ps] + ps_r, camera=_CAMERA, width=640, height=480)
    del ps_r
    path_r = os.path.join(output_dir, 'real.mp4')
    media.write_video(path_r, frames, fps=1.0 / env_real.dt)
    del frames
    print(f'  saved in {time.time()-t0:.1f}s → {path_r}', flush=True)

    # ── Model rollout (optional; no video — model pipeline_state is None) ──────
    model_state_path = os.path.join(log_dir, 'model', f'model_state_{iteration}')
    do_model = not eval_cfg.no_model and os.path.isfile(model_state_path)
    if not eval_cfg.no_model and not os.path.isfile(model_state_path):
        print(f'Warning: model checkpoint not found at {model_state_path}, skipping model rollout.',
              flush=True)

    if do_model:
        print('Loading model checkpoint and building model env...', flush=True)
        t0 = time.time()
        model_ckpt = brax_model.load_params(model_state_path)

        env_model = InfopropEnv(
            AntEnv(cfg=env_cfg),
            min_log_var=algo_cfg.min_log_var, max_log_var=algo_cfg.max_log_var)
        jit_step_model  = jax.jit(env_model.step)
        jit_reset_model = jax.jit(env_model.reset_from_buffer)

        model_state = wire_model_env(env_model, algo_cfg, model_ckpt)
        print(f'  model env ready in {time.time()-t0:.1f}s', flush=True)

        print('Resetting model env (branching from the real initial state)...', flush=True)
        t0 = time.time()
        init_transition = _init_transition_from_state(env_real, init_state)
        state_model = jit_reset_model(model_reset_rng, init_transition)
        state_model = inject_model_params(
            env_model, model_state, model_ckpt, model_env_rng, state_model)
        init_m_phys = np.array(state_model.info['physics_state'])

        phys_m, rew_m = _do_model_rollout(jit_step_model, state_model, ctrls)
        rlen_m = len(phys_m)
        model_phys_arr = np.concatenate([init_m_phys[None], phys_m], axis=0)
        print(f'Model rollout: {rlen_m} steps in {time.time()-t0:.1f}s '
              f'(return {rew_m.sum():.2f})', flush=True)

        print('Creating physics comparison plot...', flush=True)
        t0 = time.time()
        plot_path = os.path.join(output_dir, 'physics_comparison.png')
        _physics_plot(real_phys_arr, model_phys_arr, rew_r, rew_m, plot_path, iteration)
        print(f'  saved in {time.time()-t0:.1f}s → {plot_path}', flush=True)

    print('Done.', flush=True)


def run(eval_cfg):
    """Standalone entry: load the training config from the checkpoint dir, then run."""
    log_dir = os.path.abspath(eval_cfg.log_dir)
    hydra_cfg_path = os.path.join(log_dir, '.hydra', 'config.yaml')
    if not os.path.isfile(hydra_cfg_path):
        raise FileNotFoundError(f'Hydra config not found: {hydra_cfg_path}')
    train_cfg = OmegaConf.load(hydra_cfg_path)
    return run_loaded(eval_cfg, log_dir, train_cfg.env, train_cfg.algorithm)


# ── Standalone argparse entry point ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Render ant policy evaluation video from a training checkpoint.')
    parser.add_argument('log_dir',
                        help='Hydra training output directory, e.g. exp/test/27752/2026.06.01/135429')
    parser.add_argument('--iteration', type=int, default=None, dest='iteration',
                        help='Policy checkpoint index (default: latest)')
    parser.add_argument('--output-dir', default=None, dest='output_dir',
                        help='Output directory (default: <log_dir>/video_eval/iter_<N>/)')
    parser.add_argument('--no-model', action='store_true', dest='no_model',
                        help='Skip model rollout and physics comparison plot')
    parser.add_argument('--max-steps', type=int, default=1000, dest='max_steps',
                        help='Max steps for real rollout (default: 1000)')
    parser.add_argument('--rng-seed', type=int, default=0, dest='rng_seed',
                        help='JAX RNG seed (default: 0)')
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))


if __name__ == '__main__':
    main()
