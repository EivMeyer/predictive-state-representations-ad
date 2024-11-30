#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --partition=gpu80G,gpu40G,prismgpup
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --ntasks-per-node=1
#SBATCH --mem=373G
#SBATCH --gres=gpu:1
#SBATCH --job-name=prad-pretrain
#SBATCH --output=logs/%x-%j.txt

set -e

export CUDA_LAUNCH_BLOCKING=1

# Remove --nv flag if you don't need GPU support
srun singularity exec --nv \
    --bind $(pwd):/app/psr-ad \
    --bind $(pwd)/output:/app/psr-ad/output \
    --bind /home/users/tdupuis/codes/crgeo_scenarios:/app/psr-ad/scenarios \
    /home/users/tdupuis/codes/psr-ad_latest.sif \
    python3 train_model.py training.model_type="AutoEncoderModelV0" wandb.offline=true