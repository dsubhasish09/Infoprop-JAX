#!/bin/bash
#SBATCH -p c23g

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wheelbot_experiment
#SBATCH --output=output_.out
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
git checkout generalize
python -m infoprop_jax.main \
    algorithm=infoprop \
    experiment=main_test_ \
    seed=0