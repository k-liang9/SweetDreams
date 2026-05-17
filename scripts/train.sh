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

RUN_SHA=44d60c41f874abffc1800a7a9715e8d3315d5eda
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE=$REPO_ROOT/../SweetDreams-runs/$SLURM_JOB_ID
if [ ! -d "$WORKTREE" ]; then
    git -C "$REPO_ROOT" worktree add -d "$WORKTREE" "$RUN_SHA"
fi
cd "$WORKTREE"

torchrun \
    --nproc_per_node=2 \
    train/train_vqvae.py \
    exp.run_name="vqvae + ball loss" \
    data.h5_path="$REPO_ROOT/data/breakout.h5"