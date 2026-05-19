"""Compare fast and pipeline-backed InfoProp model rollouts.

This script loads a trained run directory, initializes two learned-model
environments from the same state, and rolls them forward side by side:

  - pipeline mode: fast_model_rollout=False, rebuilds pipeline_state each step
  - fast mode:     fast_model_rollout=True, builds observations directly

It reports numerical deviations in the training-relevant state.  The
pipeline_state itself is intentionally stale in fast mode and is only checked
optionally via qpos/qvel diagnostics.

Example:
    python -m wheelbot_sim_python.eval_scripts.compare_fast_model_rollout \
        exp/test/6/2026.05.19/123456 --iteration 3 --track-seed 21 --max-steps 500
"""

import argparse
import os
import sys
import time
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from brax import envs
from brax.io import model as brax_model
from brax.training.acme import running_statistics
from brax.training.agents.sac import networks as sac_networks
from omegaconf import OmegaConf

from wheelbot_sim_python.envs.wheelbot_brax_infoprop import Wheelbot as ModelWheelbot
from wheelbot_sim_python.envs.wheelbot_brax_mjx import Wheelbot as RealWheelbot


def _resolve_iteration(log_dir: str, requested):
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
                f'(available: {sorted(indices)})'
            )
        return requested
    return max(indices)


def _infer_sizes(policy_params) -> Tuple[int, int]:
    layers = policy_params['params']
    sorted_keys = sorted(
        (k for k in layers if k.startswith('hidden_')),
        key=lambda k: int(k.split('_')[1]),
    )
    obs_size = layers[sorted_keys[0]]['kernel'].shape[0]
    action_size = layers[sorted_keys[-1]]['kernel'].shape[1] // 2
    return obs_size, action_size


def _load_model_state(env_model, algo_cfg, model_ckpt):
    model_trainer = env_model.init_NN_trainer(
        seed=0,
        learning_rate=1e-3,
        weight_decay=1e-4,
        hidden_layer_sizes=tuple(algo_cfg.model_hidden_layer_sizes),
        model_layer_norm=algo_cfg.model_layer_norm,
    )
    model_state_template, _, _, _, _, _, _ = model_trainer.init(jax.random.PRNGKey(0))
    return model_state_template.replace(
        params=jax.tree_util.tree_map(jnp.array, model_ckpt['params'])
    )


def _inject_model(env_model, state, model_state, model_ckpt, rng):
    state = env_model.put_in_NN_params_and_rng(
        model_state,
        jnp.array(model_ckpt['model_obs_mean']),
        jnp.array(model_ckpt['model_obs_std']),
        jnp.array(model_ckpt['next_state_delta_mean']),
        jnp.array(model_ckpt['next_state_delta_std']),
        jnp.array(model_ckpt['per_step_cutoff']),
        jnp.array(model_ckpt['accumulated_cutoff']),
        jnp.array(model_ckpt['binning_entropy']),
        rng,
        state,
    )
    info = dict(state.info)
    for key in ('kalman_gain', 'conditional_var', 'fused_var', 'fused_mean', 'epist_var'):
        if key not in info:
            info[key] = jnp.zeros(16)
    return state.replace(info=info)


def _max_abs(a, b) -> float:
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b))))


def _state_diffs(state_pipeline, state_fast, model_input_torque) -> Dict[str, float]:
    keys = {
        'obs': (state_pipeline.obs, state_fast.obs),
        'physics': (
            state_pipeline.info['physics_state'],
            state_fast.info['physics_state'],
        ),
        'invariant': (
            state_pipeline.info['invariant_physics_state'],
            state_fast.info['invariant_physics_state'],
        ),
        'reward': (state_pipeline.reward, state_fast.reward),
        'done': (state_pipeline.done, state_fast.done),
        'model_input_torque': (
            state_pipeline.info['applied_torque'],
            model_input_torque,
        ),
        'applied_torque': (
            state_pipeline.info['applied_torque'],
            state_fast.info['applied_torque'],
        ),
        'current_entropy': (
            state_pipeline.info['current_conditional_entropy'],
            state_fast.info['current_conditional_entropy'],
        ),
        'accum_entropy': (
            state_pipeline.info['accumulated_conditional_entropy'],
            state_fast.info['accumulated_conditional_entropy'],
        ),
    }
    return {name: _max_abs(a, b) for name, (a, b) in keys.items()}


def _pipeline_q_diff(state_pipeline, state_fast) -> Tuple[float, float]:
    return (
        _max_abs(state_pipeline.pipeline_state.qpos, state_fast.pipeline_state.qpos),
        _max_abs(state_pipeline.pipeline_state.qvel, state_fast.pipeline_state.qvel),
    )


def run(args):
    log_dir = os.path.abspath(args.log_dir)
    hydra_cfg_path = os.path.join(log_dir, '.hydra', 'config.yaml')
    if not os.path.isfile(hydra_cfg_path):
        raise FileNotFoundError(f'Hydra config not found: {hydra_cfg_path}')

    train_cfg = OmegaConf.load(hydra_cfg_path)
    env_cfg = train_cfg.env
    algo_cfg = train_cfg.algorithm
    iteration = _resolve_iteration(log_dir, args.iteration)

    policy_path = os.path.join(log_dir, 'policy', f'brax_policy_{iteration}')
    model_path = os.path.join(log_dir, 'model', f'model_state_{iteration}')
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model checkpoint not found: {model_path}')

    print(f'Run: {log_dir}', flush=True)
    print(f'Iteration: {iteration}', flush=True)
    print(f'Track seed: {args.track_seed}', flush=True)
    print(f'Max steps: {args.max_steps}', flush=True)
    print('Actions: pipeline normal step; fast direct_step uses pipeline applied_torque', flush=True)

    normalizer_params, policy_params = brax_model.load_params(policy_path)
    model_ckpt = brax_model.load_params(model_path)
    obs_size, action_size = _infer_sizes(policy_params)

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
    policy = make_policy((normalizer_params, policy_params), deterministic=True)

    @jax.jit
    def jit_policy(obs, rng):
        action, _ = policy(obs, rng)
        return action

    envs.register_environment('_cmp_real', RealWheelbot)
    env_real = envs.get_environment(
        '_cmp_real', cfg=env_cfg, visualize=False, track_seed=args.track_seed
    )
    init_real = jax.jit(env_real.reset_to_start)(jax.random.PRNGKey(args.rng_seed))
    init_history = jnp.concatenate(
        [init_real.info['phys_state_history'], init_real.info['act_history']], axis=-1
    )
    invariant_init = init_real.info['invariant_physics_state']
    track_seed_val = init_real.info['track_seed']

    envs.register_environment('_cmp_model_pipeline', ModelWheelbot)
    env_pipeline = envs.get_environment(
        '_cmp_model_pipeline',
        cfg=env_cfg,
        visualize=False,
        track_seed=args.track_seed,
        min_log_var=algo_cfg.min_log_var,
        max_log_var=algo_cfg.max_log_var,
        fast_model_rollout=False,
    )
    envs.register_environment('_cmp_model_fast', ModelWheelbot)
    env_fast = envs.get_environment(
        '_cmp_model_fast',
        cfg=env_cfg,
        visualize=False,
        track_seed=args.track_seed,
        min_log_var=algo_cfg.min_log_var,
        max_log_var=algo_cfg.max_log_var,
        fast_model_rollout=True,
    )

    model_state = _load_model_state(env_pipeline, algo_cfg, model_ckpt)
    reset_rng = jax.random.PRNGKey(args.rng_seed + 1)
    model_rng = jax.random.PRNGKey(args.rng_seed + 2)

    reset_pipeline = jax.jit(env_pipeline.reset_with_init_robot_state_eval)
    reset_fast = jax.jit(env_fast.reset_with_init_robot_state_eval)
    state_pipeline = reset_pipeline(
        reset_rng, init_history, track_seed_val, invariant_init[3:5], invariant_init[0]
    )
    state_fast = reset_fast(
        reset_rng, init_history, track_seed_val, invariant_init[3:5], invariant_init[0]
    )
    state_pipeline = _inject_model(env_pipeline, state_pipeline, model_state, model_ckpt, model_rng)
    state_fast = _inject_model(env_fast, state_fast, model_state, model_ckpt, model_rng)

    step_pipeline = jax.jit(env_pipeline.step)
    direct_step_fast = jax.jit(env_fast.direct_step)

    initial_obs_diff = _max_abs(state_pipeline.obs, state_fast.obs)
    print(f'Initial obs max abs diff: {initial_obs_diff:.6e}', flush=True)

    worst = {}
    first_exceed = None
    t0 = time.time()
    rng = jax.random.PRNGKey(args.rng_seed + 3)
    for step in range(args.max_steps):
        rng, key_pipeline = jax.random.split(rng)
        policy_action = jit_policy(state_pipeline.obs, key_pipeline)

        next_pipeline = step_pipeline(state_pipeline, policy_action)
        model_input_torque = next_pipeline.info['applied_torque']
        next_fast = direct_step_fast(state_fast, model_input_torque)
        diffs = _state_diffs(next_pipeline, next_fast, model_input_torque)
        for name, value in diffs.items():
            worst[name] = max(worst.get(name, 0.0), value)

        max_training_diff = max(diffs.values())
        if first_exceed is None and max_training_diff > args.tolerance:
            first_exceed = (step + 1, diffs)
            print(f'First diff > tolerance at step {step + 1}: {diffs}', flush=True)
            if args.stop_on_first_diff:
                state_pipeline, state_fast = next_pipeline, next_fast
                break

        state_pipeline, state_fast = next_pipeline, next_fast
        if args.print_every and (step + 1) % args.print_every == 0:
            qpos_diff, qvel_diff = _pipeline_q_diff(state_pipeline, state_fast)
            print(
                f'step {step + 1}: max_training_diff={max_training_diff:.6e} '
                f'qpos_diff={qpos_diff:.6e} qvel_diff={qvel_diff:.6e}',
                flush=True,
            )
        if bool(state_pipeline.done) or bool(state_fast.done):
            print(
                f'Stopped at step {step + 1}: '
                f'pipeline_done={float(state_pipeline.done):.1f} '
                f'fast_done={float(state_fast.done):.1f}',
                flush=True,
            )
            break

    qpos_diff, qvel_diff = _pipeline_q_diff(state_pipeline, state_fast)
    print(f'Completed in {time.time() - t0:.1f}s', flush=True)
    print('Worst training-relevant max abs diffs:', flush=True)
    for name in sorted(worst):
        print(f'  {name}: {worst[name]:.6e}', flush=True)
    print('Final pipeline_state diagnostic diffs, expected nonzero in fast mode:', flush=True)
    print(f'  qpos: {qpos_diff:.6e}', flush=True)
    print(f'  qvel: {qvel_diff:.6e}', flush=True)
    if first_exceed is None:
        print(f'No training-relevant diff exceeded tolerance {args.tolerance:.3e}.', flush=True)
    else:
        step, diffs = first_exceed
        print(f'First tolerance exceedance was step {step}: {diffs}', flush=True)


def main():
    parser = argparse.ArgumentParser(
        description='Compare fast and pipeline-backed InfoProp model rollouts.'
    )
    parser.add_argument('log_dir', help='Hydra training output directory.')
    parser.add_argument('--iteration', type=int, default=None, help='Checkpoint iteration; default latest.')
    parser.add_argument('--track-seed', type=int, default=21, dest='track_seed')
    parser.add_argument('--max-steps', type=int, default=500, dest='max_steps')
    parser.add_argument('--rng-seed', type=int, default=0, dest='rng_seed')
    parser.add_argument('--tolerance', type=float, default=1e-5)
    parser.add_argument(
        '--stop-on-first-diff',
        action='store_true',
        help='Stop as soon as any training-relevant field exceeds tolerance.',
    )
    parser.add_argument(
        '--print-every',
        type=int,
        default=50,
        help='Progress print interval; use 0 to disable.',
    )
    args = parser.parse_args()
    try:
        run(args)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(str(exc))


if __name__ == '__main__':
    main()
