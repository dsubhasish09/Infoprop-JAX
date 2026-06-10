"""
Hydra entry point for Infoprop Dyna training.

Reads configuration from infoprop_jax/config/, sets up Weights & Biases
logging, defines domain randomisation, registers Brax environments, and calls
the single-GPU `train` function from infoprop.py.

Run with:
    python -m infoprop_jax.main
"""
from datetime import datetime
import functools
import os

from brax import envs
from infoprop_jax.algorithms import infoprop
# Importing the envs package registers all bundled environments with Brax's
# global registry, so they resolve via ``envs.get_environment(<name>)`` below.
import infoprop_jax.envs  # noqa: F401  (import for registration side effect)
from infoprop_jax.envs.infoprop_env import InfopropEnv
from brax.io import model
import jax
import jax.numpy as jp

import wandb
import hydra
import omegaconf
from omegaconf import OmegaConf


@hydra.main(config_path="config", config_name="main", version_base=None)
def main(cfg: omegaconf.DictConfig):
    """Configure and launch one InfoProp Dyna training run.

    Responsibilities:
      - Parse Hydra config (algorithm + env sections).
      - Initialise / resume a W&B run.
      - Define domain_randomize: samples robot physical parameters from Gaussian
        distributions for domain randomisation during real-env rollouts.
      - Register the MJX and InfoProp Brax environments.
      - Call infoprop.train with all hyperparameters from config.

    Args:
        cfg: Hydra DictConfig composed from config/main.yaml.
    """
    # Get Hydra output directory
    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    # Setup wandb with resume support
    wandb_resume = None
    wandb_id = cfg.get('wandb_run_id', None)

    # register the environment
    wnb_cfg = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=False,
    )

    wandb.init(
        # set the wandb project where this run will be logged
        project=cfg.wandb_project,

        # track hyperparameters and run metadata
        config=wnb_cfg,
        job_type="train",
        group=cfg.algorithm.name,
        name=cfg.experiment,
        resume=wandb_resume,
        id=wandb_id,
    )
    wandb.define_metric("agent_train/step")
    wandb.define_metric("model_train/step")
    wandb.define_metric("agent_train/actor_loss", step_metric="agent_train/step")
    wandb.define_metric("agent_train/critic_loss", step_metric="agent_train/step")
    wandb.define_metric("agent_train/alpha_loss", step_metric="agent_train/step")
    wandb.define_metric("agent_train/alpha", step_metric="agent_train/step")
    wandb.define_metric("agent_train/actor_entropy", step_metric="agent_train/step")
    wandb.define_metric("agent_train/target_entropy", step_metric="agent_train/step")
    wandb.define_metric("agent_train/average_rollout_length", step_metric="agent_train/step")
    wandb.define_metric("model_train/val_loss", step_metric="model_train/step")
    wandb.define_metric("agent_eval/return_mean", step_metric="model_train/step")
    wandb.define_metric("agent_eval/return_histogram", step_metric="model_train/step")

    # post evaluation logging callback
    def progress(num_steps, metrics):
        t = datetime.now()
        times.append(t)

        if "training/actor_loss" in metrics:
            data = {
                "agent_train/step": num_steps,
                "model_train/step": metrics['num_real_transitions'],
                "agent_train/actor_loss": metrics['training/actor_loss'],
                "agent_train/critic_loss": metrics['training/critic_loss'],
                "agent_train/alpha_loss": metrics['training/alpha_loss'],
                "agent_train/alpha": metrics['training/alpha'],
                "agent_train/actor_entropy": metrics['training/actor_entropy'],
                "agent_train/target_entropy": metrics['training/target_entropy'],
                "agent_train/average_rollout_length": metrics['training/average_rollout_length'],
                "model_train/val_loss": metrics['model/val_loss'],
                "agent_eval/return_mean": metrics['eval/episode_reward'],
                "agent_eval/return_histogram": wandb.Histogram(
                    np_histogram=metrics['eval/episode_reward_histogram']
                ),
            }
            wandb.log(data)

    # instantiate the environment
    env_name = cfg.env.get('env_name', str(cfg.env.name).lower())
    env = envs.get_environment(env_name, cfg=cfg.env)
    eval_env = envs.get_environment(env_name, cfg=cfg.env, eval_mode=True)

    # The model env is the generic InfopropEnv wrapping a fresh wrappable env.
    # Whether that env uses fast rollouts is its own concern (read from cfg.env).
    infoprop_env = InfopropEnv(
        envs.get_environment(env_name, cfg=cfg.env),
        min_log_var=cfg.algorithm.min_log_var,
        max_log_var=cfg.algorithm.max_log_var,
    )

    # training function
    train_cfg = cfg.algorithm
    train_fn = infoprop.train
    train_fn = functools.partial(
        train_fn,
        episode_length=train_cfg.episode_length,
        max_physics_replay_size=train_cfg.max_physics_replay_size,
        min_physics_replay_size=train_cfg.min_physics_replay_size,
        max_model_replay_size=train_cfg.max_model_replay_size,
        min_model_replay_size=train_cfg.min_model_replay_size,
        agent_learning_rate=train_cfg.agent_learning_rate,
        agent_batch_size=train_cfg.agent_batch_size,
        agent_hidden_layer_sizes=train_cfg.agent_hidden_layer_sizes,
        agent_layer_norm=train_cfg.agent_layer_norm,
        policy_network_layer_norm=train_cfg.get(
            'policy_network_layer_norm', train_cfg.agent_layer_norm),
        q_network_layer_norm=train_cfg.get(
            'q_network_layer_norm', train_cfg.agent_layer_norm),
        grad_updates_per_model_step=train_cfg.grad_updates_per_model_step,
        num_resampling_epochs=train_cfg.num_resampling_epochs,
        num_training_steps_per_model_train=train_cfg.num_training_steps_per_model_train,
        model_learning_rate=train_cfg.model_learning_rate,
        model_weight_decay=train_cfg.model_weight_decay,
        model_batch_size=train_cfg.model_batch_size,
        model_hidden_layer_sizes=train_cfg.model_hidden_layer_sizes,
        model_layer_norm=train_cfg.model_layer_norm,
        patience=train_cfg.patience,
        target_entropy=train_cfg.target_entropy,
        num_envs=train_cfg.num_model_envs,
        max_rollout_length=train_cfg.max_rollout_length,
        lower_quantile=train_cfg.lower_quantile,
        upper_quantile=train_cfg.upper_quantile,
        action_repeat=train_cfg.action_repeat,
        obs_history=cfg.env.get('obs_history', 1),
        act_history=cfg.env.get('act_history', 0),
        num_real_envs=train_cfg.num_real_train_envs,
        num_real_eval_envs=train_cfg.num_real_eval_envs,
        discounting=train_cfg.discounting,
        num_trials=train_cfg.num_trials,
        normalize_observations=train_cfg.normalize_observations,
        tau=train_cfg.tau,
        tune_entropy=train_cfg.tune_entropy,
        alpha=train_cfg.alpha,
        reset_agent_per_trial=train_cfg.reset_agent_per_trial,
        reset_model_replay_buffer=train_cfg.reset_model_replay_buffer,
        reset_model_per_trial=train_cfg.reset_model_per_trial,
        agent_dir=os.path.join(output_dir, 'policy'),
        model_dir=os.path.join(output_dir, 'model'),
        seed=cfg.seed,
    )

    times = [datetime.now()]

    _, params, _ = train_fn(
        environment=env,
        model_environment=infoprop_env,
        eval_environment=eval_env,
        progress_fn=progress,
        randomization_fn=None,
    )

    print(f'time to jit: {times[1] - times[0]}')
    print(f'time to train: {times[-1] - times[1]}')

    # save the model
    model_path = output_dir + '/final_mjx_brax_policy'
    model.save_params(model_path, params)
    wandb.finish()


if __name__ == "__main__":
    main()
