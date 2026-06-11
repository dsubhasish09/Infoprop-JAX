#!/bin/bash
#SBATCH -p c23g

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wheelbot_experiment
#SBATCH --output=output_.out

#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --array=1

export CUDA_VISIBLE_DEVICES=0

# Deterministic JAX/XLA flags - disable autotuning for consistent performance
export XLA_FLAGS='--xla_gpu_autotune_level=0'
export TF_DETERMINISTIC_OPS=1

source .venv/bin/activate
module load FFmpeg

export MUJOCO_GL=egl

python -m infoprop_jax.main video_eval=true eval.log_dir=exp/humanoid_race_infoprop/0/2026.06.11/021836 eval.iteration=19 eval.track_seed=100