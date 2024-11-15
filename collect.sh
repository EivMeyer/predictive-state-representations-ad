#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH --partition=bigmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=collect_crgeo

set -e

# Configure environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export WANDB_MODE=offline
export WANDB__SERVICE_WAIT=30

source prad/bin/activate

srun --output=logs/%x-%j.txt ./parallel_dataset_collection.sh -e 64 -w 8 -r 128