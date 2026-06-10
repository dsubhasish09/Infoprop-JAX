#!/bin/bash
#SBATCH -p c23g

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=humanoid_infoprop
#SBATCH --output=humanoid_%A_%a.out
#SBATCH --account=rwth2087
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
#SBATCH --array=1

export CUDA_VISIBLE_DEVICES=0

source .venv/bin/activate
module load FFmpeg

export MUJOCO_GL=egl

# Prevent JAX from pre-allocating 75% of GPU memory at import time.
# Without this, JAX grabs a fixed chunk before any array is created,
# causing OOM on large replay buffers.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

python -m infoprop_jax.main \
    algorithm=infoprop \
    env=humanoid \
    experiment=humanoid_infoprop \
    seed=0 \
    algorithm.num_model_envs=1000 \
    algorithm.max_model_replay_size=1000000 \
    algorithm.num_training_steps_per_model_train=1000 \
    algorithm.grad_updates_per_model_step=10 \
    algorithm.num_resampling_epochs=10 \
    algorithm.target_entropy=-21 \
    algorithm.tune_entropy=true \
    algorithm.max_rollout_length=1000 \
    algorithm.upper_quantile=1.0
