#!/bin/bash
#SBATCH --time=0-1:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --exclude=watgpu108,watgpu408,watgpu1008
#SBATCH --error=train.log
#SBATCH --mail-user=k24liang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Interactive allocation example:
# salloc --gres=gpu:1 --cpus-per-task=12 --mem=128G --time=6:00:00 --exclude=watgpu108,watgpu408

# Environment setup
eval "$(conda shell.bash hook)"
conda activate sweetdreams

torchrun --nproc_per_node=2 train/train_vqvae.py exp.run_name='rgb perceptive_weight 0.5'
