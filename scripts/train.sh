#!/bin/bash
#SBATCH --time=0-3:00
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

# Hydra multirun (-m) reuses a single Python process across sweeps, which collides
# with torchrun's NCCL rendezvous on the second sweep (port stuck in TIME_WAIT,
# rank 1 hits "Connection refused" on the next dist.broadcast). Loop in shell so
# each value gets its own torchrun invocation with a fresh rendezvous port.

# for disc_weight in 0.0 0.005 0.01 0.1; do
torchrun \
    --nproc_per_node=2 \
    train/train_vqvae.py \
    exp.run_name="vqvae + ball loss" \
    discriminator.enabled=false