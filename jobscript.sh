#!/bin/bash
#SBATCH -p c23g

#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=wheelbot_experiment
#SBATCH --output=output_.out
#SBATCH --account=rwth2087
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --array=1-5

export CUDA_VISIBLE_DEVICES=0

source .venv/bin/activate
module load FFmpeg

export MUJOCO_GL=egl

python -m wheelbot_sim_python.main \
    algorithm=infoprop \
    experiment=default_run_refactored \
    seed=$RANDOM \
    algorithm.num_model_envs=1000 \
    algorithm.grad_updates_per_model_step=10 \
    algorithm.reset_agent_per_trial=False \
    algorithm.reset_model_replay_buffer=True \
    algorithm.reset_model_per_trial=False \
    env.rew_scale=1.0 \
    algorithm.tau=0.005 \
    algorithm.alpha=0.1
