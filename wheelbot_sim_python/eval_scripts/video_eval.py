"""Video evaluation for InfoProp Dyna checkpoints.

Can be invoked in two ways:

1. Via the Hydra main entry point (recommended):
       python -m wheelbot_sim_python.main video_eval=true eval.log_dir=exp/test/27752/2026.05.05/135429
   Override any eval param on the CLI, e.g. eval.track_seed=42 eval.iteration=3

2. As a standalone script:
       python -m wheelbot_sim_python.eval_scripts.video_eval <log_dir> [options]
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
from brax import envs
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.sac import networks as sac_networks
from omegaconf import OmegaConf

from wheelbot_sim_python.envs.wheelbot_brax_mjx import Wheelbot as RealWheelbot
from wheelbot_sim_python.envs.wheelbot_brax_infoprop import Wheelbot as ModelWheelbot


# ── Helpers ───────────────────────────────────────────────────────────────────

def _combine_physics_states(physics_11d, invariant_5d):
    """Merge variant (11D) and invariant (5D) state vectors into a 16D state."""
    return jnp.array([
        invariant_5d[0],   # yaw
        physics_11d[0],    # roll
        physics_11d[1],    # pitch
        physics_11d[2],    # yaw_rate
        physics_11d[3],    # roll_rate
        physics_11d[4],    # pitch_rate
        invariant_5d[1],   # driving_wheel_angle
        physics_11d[5],    # driving_wheel_angular_velocity
        invariant_5d[2],   # reaction_wheel_angle
        physics_11d[6],    # reaction_wheel_angular_velocity
        invariant_5d[3],   # x
        invariant_5d[4],   # y
        physics_11d[10],   # z (moved from invariant to variant physics state)
        physics_11d[7],    # x_rate
        physics_11d[8],    # y_rate
        physics_11d[9],    # z_rate
    ])


def _resolve_iteration(log_dir, requested):
    policy_dir = os.path.join(log_dir, 'policy')
    if not os.path.isdir(policy_dir):
        raise FileNotFoundError(f'Policy directory not found: {policy_dir}')
    indices = []
    for name in os.listdir(policy_dir):
        if name.startswith('brax_policy_'):
            try:
                indices.append(int(name.split('_')[-1]))
            except ValueError:
                pass
    if not indices:
        raise FileNotFoundError(f'No brax_policy_* files found in {policy_dir}')
    if requested is not None:
        if requested not in indices:
            raise ValueError(
                f'Requested iteration {requested} not found in {policy_dir} '
                f'(available: {sorted(indices)})')
        return requested
    return max(indices)


def _infer_sizes(policy_params):
    layers = policy_params['params']
    sorted_keys = sorted(
        (k for k in layers if k.startswith('hidden_')),
        key=lambda k: int(k.split('_')[1]))
    obs_size = layers[sorted_keys[0]]['kernel'].shape[0]
    # SAC last layer outputs [mean, log_std] concatenated → 2 * action_size
    action_size = layers[sorted_keys[-1]]['kernel'].shape[1] // 2
    return obs_size, action_size


def _do_real_rollout(jit_step, jit_inf, params, state, rng, max_steps):
    ps_list, phys_list, inv_list, ctrl_list = [], [], [], []
    for step in range(max_steps):
        rng, curr_rng = jax.random.split(rng)
        ctrl, _ = jit_inf(params, state.obs, curr_rng)
        ns = jit_step(state, ctrl)
        ps_list.append(ns.pipeline_state)
        phys_list.append(np.array(ns.info['physics_state']))
        inv_list.append(np.array(ns.info['invariant_physics_state']))
        ctrl_list.append(np.array(ctrl))
        state = ns
        if (step + 1) % 500 == 0:
            print(f'  real step {step + 1}', flush=True)
        if float(ns.done) > 0.5:
            print(state.obs[-11:])
            break
    return ps_list, np.stack(phys_list), np.stack(inv_list), np.stack(ctrl_list)


def _do_model_rollout(jit_step_model, state, ctrls):
    ps_list, phys_list, inv_list = [], [], []
    for step, ctrl in enumerate(ctrls):
        ns = jit_step_model(state, jnp.array(ctrl))
        ps_list.append(ns.pipeline_state)
        phys_list.append(np.array(ns.info['physics_state']))
        inv_list.append(np.array(ns.info['invariant_physics_state']))
        state = ns
        if (step + 1) % 500 == 0:
            print(f'  model step {step + 1}', flush=True)
        if float(ns.done) > 0.5:
            break
    return ps_list, np.stack(phys_list), np.stack(inv_list)


def _physics_plot(real_phys, real_inv, model_phys, model_inv, out_path, iteration):
    cmp_len = min(len(real_phys), len(model_phys))
    real_16d  = np.array(jax.vmap(_combine_physics_states)(
        jnp.array(real_phys[:cmp_len]), jnp.array(real_inv[:cmp_len])))
    model_16d = np.array(jax.vmap(_combine_physics_states)(
        jnp.array(model_phys[:cmp_len]), jnp.array(model_inv[:cmp_len])))
    labels = [
        'yaw', 'roll', 'pitch',
        'yaw_rate', 'roll_rate', 'pitch_rate',
        'driving_wheel_angle', 'driving_wheel_angular_velocity',
        'reaction_wheel_angle', 'reaction_wheel_angular_velocity',
        'x', 'y', 'z', 'x_rate', 'y_rate', 'z_rate',
    ]
    fig, axes = plt.subplots(4, 4, figsize=(32, 16))
    fig.suptitle(f'Physics State Comparison: Model vs Real (Iter {iteration})', fontsize=16)
    for i in range(16):
        ax = axes[i // 4, i % 4]
        ax.plot(real_16d[:, i],  label=f'{labels[i]}_real',  alpha=0.8, color='blue',  linewidth=2)
        ax.plot(model_16d[:, i], label=f'{labels[i]}_model', alpha=0.8, color='red',   linewidth=2)
        ax.set_title(labels[i], fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 2))
        ax.set_xlabel('Time Step', fontsize=8)
        ax.set_ylabel('Value', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


# ── Main entry point (called by Hydra dispatcher or standalone argparse) ──────

def run(eval_cfg):
    """Run video evaluation.

    Args:
        eval_cfg: Object with attributes matching config/eval/video_eval.yaml:
                  log_dir, iteration, track_seed, output_dir, no_model,
                  max_steps, rng_seed.  Accepts OmegaConf DictConfig or
                  argparse Namespace — both support attribute access.
    """
    log_dir = os.path.abspath(eval_cfg.log_dir)

    # ── Load training config from the checkpoint dir ──────────────────────────
    hydra_cfg_path = os.path.join(log_dir, '.hydra', 'config.yaml')
    if not os.path.isfile(hydra_cfg_path):
        raise FileNotFoundError(f'Hydra config not found: {hydra_cfg_path}')
    train_cfg = OmegaConf.load(hydra_cfg_path)
    env_cfg   = train_cfg.env
    algo_cfg  = train_cfg.algorithm

    # ── Resolve iteration ─────────────────────────────────────────────────────
    iteration = _resolve_iteration(log_dir, eval_cfg.iteration)
    print(f'Evaluating iteration {iteration} on track seed {eval_cfg.track_seed}', flush=True)

    # ── Output dir ────────────────────────────────────────────────────────────
    output_dir = eval_cfg.output_dir or os.path.join(log_dir, 'video_eval', f'iter_{iteration}')
    os.makedirs(output_dir, exist_ok=True)
    print(f'Outputs → {output_dir}', flush=True)

    # ── Load policy ───────────────────────────────────────────────────────────
    policy_path = os.path.join(log_dir, 'policy', f'brax_policy_{iteration}')
    print(f'Loading policy from {policy_path}', flush=True)
    normalizer_params, policy_params = brax_model.load_params(policy_path)
    obs_size, action_size = _infer_sizes(policy_params)
    print(f'  obs_size={obs_size}  action_size={action_size}', flush=True)

    # ── Build real env ────────────────────────────────────────────────────────
    print('Building real environment...', flush=True)
    t0 = time.time()
    envs.register_environment('_eval_real', RealWheelbot)
    env_real = envs.get_environment('_eval_real', cfg=env_cfg,
                                    visualize=True, track_seed=eval_cfg.track_seed)
    jit_reset_real = jax.jit(env_real.reset_to_start)
    jit_step_real  = jax.jit(env_real.step)
    print(f'  done in {time.time()-t0:.1f}s', flush=True)

    # ── Build SAC inference fn ────────────────────────────────────────────────
    normalize_fn = running_statistics.normalize if algo_cfg.normalize_observations else (lambda x, y: x)
    sac_net = sac_networks.make_sac_networks(
        observation_size=obs_size,
        action_size=action_size,
        preprocess_observations_fn=normalize_fn,
        hidden_layer_sizes=tuple(algo_cfg.agent_hidden_layer_sizes),
        policy_network_layer_norm=algo_cfg.agent_layer_norm,
        q_network_layer_norm=algo_cfg.agent_layer_norm,
    )
    make_policy = sac_networks.make_inference_fn(sac_net)
    params = (normalizer_params, policy_params)

    @jax.jit
    def jit_inf(p, obs, rng):
        return make_policy(p, deterministic=True)(obs, rng)

    # ── Real rollout ──────────────────────────────────────────────────────────
    rng = jax.random.PRNGKey(eval_cfg.rng_seed)
    rng, reset_rng, eval_rng, model_reset_rng, model_env_rng = jax.random.split(rng, 5)

    print('Resetting real env and running rollout...', flush=True)
    t0 = time.time()
    init_state = jit_reset_real(reset_rng)
    init_history            = jnp.concatenate(
        [init_state.info['phys_state_history'], init_state.info['act_history']], axis=-1)
    invariant_physics_state = init_state.info['invariant_physics_state']
    track_seed_val          = init_state.info['track_seed']
    init_r_ps               = init_state.pipeline_state
    init_r_phys             = np.array(init_state.info['physics_state'])
    init_r_inv              = np.array(init_state.info['invariant_physics_state'])

    ps_r, phys_r, inv_r, ctrls = _do_real_rollout(
        jit_step_real, jit_inf, params, init_state, eval_rng, eval_cfg.max_steps)
    rlen_r = len(ps_r)
    real_phys_10d = np.concatenate([init_r_phys[None], phys_r], axis=0)
    real_inv_6d   = np.concatenate([init_r_inv[None],  inv_r],  axis=0)
    print(f'Real rollout: {rlen_r} steps in {time.time()-t0:.1f}s', flush=True)

    # ── Render real video ─────────────────────────────────────────────────────
    print(f'Rendering real video ({rlen_r+1} frames)...', flush=True)
    t0 = time.time()
    frames = env_real.render([init_r_ps] + ps_r, camera='floating')
    del ps_r
    path_r = os.path.join(output_dir, 'real.mp4')
    media.write_video(path_r, frames, fps=1.0 / env_real.dt)
    del frames
    print(f'  saved in {time.time()-t0:.1f}s → {path_r}', flush=True)

    # ── Model rollout (optional) ──────────────────────────────────────────────
    model_state_path = os.path.join(log_dir, 'model', f'model_state_{iteration}')
    do_model = not eval_cfg.no_model and os.path.isfile(model_state_path)
    if not eval_cfg.no_model and not os.path.isfile(model_state_path):
        print(f'Warning: model checkpoint not found at {model_state_path}, skipping model rollout.',
              flush=True)

    if do_model:
        print('Loading model checkpoint and building model env...', flush=True)
        t0 = time.time()
        model_ckpt = brax_model.load_params(model_state_path)

        envs.register_environment('_eval_model', ModelWheelbot)
        env_model = envs.get_environment(
            '_eval_model', cfg=env_cfg, visualize=True, track_seed=eval_cfg.track_seed,
            min_log_var=algo_cfg.min_log_var, max_log_var=algo_cfg.max_log_var,
            fast_model_rollout=False)
        jit_step_model  = jax.jit(env_model.step)
        jit_reset_model = jax.jit(env_model.reset_with_init_robot_state_eval)

        model_trainer = env_model.init_NN_trainer(
            seed=0, learning_rate=1e-3, weight_decay=1e-4,
            hidden_layer_sizes=tuple(algo_cfg.model_hidden_layer_sizes),
            model_layer_norm=algo_cfg.model_layer_norm,
        )
        model_state_template, _, _, _, _, _, _ = model_trainer.init(jax.random.PRNGKey(0))
        model_state = model_state_template.replace(
            params=jax.tree_util.tree_map(jnp.array, model_ckpt['params']))

        model_obs_mean        = jnp.array(model_ckpt['model_obs_mean'])
        model_obs_std         = jnp.array(model_ckpt['model_obs_std'])
        next_state_delta_mean = jnp.array(model_ckpt['next_state_delta_mean'])
        next_state_delta_std  = jnp.array(model_ckpt['next_state_delta_std'])
        per_step_cutoff       = jnp.array(model_ckpt['per_step_cutoff'])
        accumulated_cutoff    = jnp.array(model_ckpt['accumulated_cutoff'])
        binning_entropy       = jnp.array(model_ckpt['binning_entropy'])
        print(f'  model env ready in {time.time()-t0:.1f}s', flush=True)

        print('Resetting model env and running rollout...', flush=True)
        t0 = time.time()
        state_model = jit_reset_model(
            model_reset_rng, init_history, track_seed_val,
            invariant_physics_state[3:5], invariant_physics_state[0])
        state_model = env_model.put_in_NN_params_and_rng(
            model_state, model_obs_mean, model_obs_std,
            next_state_delta_mean, next_state_delta_std,
            per_step_cutoff, accumulated_cutoff, binning_entropy,
            model_env_rng, state_model,
        )
        _mi = dict(state_model.info)
        for _k in ('kalman_gain', 'conditional_var', 'fused_var', 'fused_mean', 'epist_var'):
            if _k not in _mi:
                _mi[_k] = jnp.zeros(16)
        state_model = state_model.replace(info=_mi)
        init_m_ps   = state_model.pipeline_state
        init_m_phys = np.array(state_model.info['physics_state'])
        init_m_inv  = np.array(state_model.info['invariant_physics_state'])

        ps_m, phys_m, inv_m = _do_model_rollout(jit_step_model, state_model, ctrls)
        rlen_m = len(ps_m)
        model_phys_10d = np.concatenate([init_m_phys[None], phys_m], axis=0)
        model_inv_6d   = np.concatenate([init_m_inv[None],  inv_m],  axis=0)
        print(f'Model rollout: {rlen_m} steps in {time.time()-t0:.1f}s', flush=True)

        print(f'Rendering model video ({rlen_m+1} frames)...', flush=True)
        t0 = time.time()
        frames = env_model.render([init_m_ps] + ps_m, camera='floating')
        del ps_m
        path_m = os.path.join(output_dir, 'model.mp4')
        media.write_video(path_m, frames, fps=1.0 / env_model.dt)
        del frames
        print(f'  saved in {time.time()-t0:.1f}s → {path_m}', flush=True)

        print('Creating physics comparison plot...', flush=True)
        t0 = time.time()
        plot_path = os.path.join(output_dir, 'physics_comparison.png')
        _physics_plot(real_phys_10d, real_inv_6d, model_phys_10d, model_inv_6d,
                      plot_path, iteration)
        print(f'  saved in {time.time()-t0:.1f}s → {plot_path}', flush=True)

    print('Done.', flush=True)


# ── Standalone argparse entry point ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Render policy evaluation videos from a training checkpoint.')
    parser.add_argument('log_dir',
                        help='Hydra training output directory, e.g. exp/test/27752/2026.05.05/135429')
    parser.add_argument('--iteration', type=int, default=None, dest='iteration',
                        help='Policy checkpoint index (default: latest)')
    parser.add_argument('--track-seed', type=int, default=21, dest='track_seed',
                        help='Track seed to evaluate on (default: 21)')
    parser.add_argument('--output-dir', default=None, dest='output_dir',
                        help='Output directory (default: <log_dir>/video_eval/iter_<N>/)')
    parser.add_argument('--no-model', action='store_true', dest='no_model',
                        help='Skip model rollout and physics comparison plot')
    parser.add_argument('--max-steps', type=int, default=10000, dest='max_steps',
                        help='Max steps for real rollout (default: 2000)')
    parser.add_argument('--rng-seed', type=int, default=0, dest='rng_seed',
                        help='JAX RNG seed (default: 0)')
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as e:
        sys.exit(str(e))


if __name__ == '__main__':
    main()
