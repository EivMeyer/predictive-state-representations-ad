#!/usr/bin/env bash
#SBATCH --time=24:00:00
#SBATCH -p bigmem
#SBATCH -N 1
#SBATCH --cpus-per-task=72
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --job-name=collect_crgeo
#SBATCH --output=logs/%x-%j.txt

set -e

# Since you're on CPU partition, you probably don't need this
# export CUDA_VISIBLE_DEVICES=0

# # Remove --nv flag if you don't need GPU support
srun singularity exec \
    --bind $(pwd):/app/psr-ad \
    --bind $(pwd)/output:/app/psr-ad/output \
    --bind /home/users/tdupuis/codes/crgeo_scenarios:/app/psr-ad/scenarios \
    /home/users/tdupuis/codes/psr-ad_latest.sif \
    python3 change_dataset_batch_size.py ./output/dataset/ ./output/dataset2 100000
    
    # ./parallel_dataset_collection.sh dataset.collect_from_trajectories=true commonroad.async_reset=false -e 100000 -w 72 -r 100000

