#!/bin/bash
#SBATCH --time=0-2:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --exclude=watgpu108,watgpu408,watgpu1008
#SBATCH --error=train-%j.log
#SBATCH --mail-user=k24liang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Interactive allocation example:
# salloc --gres=gpu:1 --cpus-per-task=12 --mem=128G --time=6:00:00 --exclude=watgpu108,watgpu408

# Environment setup
eval "$(conda shell.bash hook)"
conda activate sweetdreams

RUN_SHA=1a1015576be82de7afb69c90c167826476d7db5a
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE=$REPO_ROOT/../SweetDreams-runs/$SLURM_JOB_ID
if [ ! -d "$WORKTREE" ]; then
    git -C "$REPO_ROOT" worktree add -d "$WORKTREE" "$RUN_SHA"
fi
cd "$WORKTREE"

MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

torchrun \
    --master_port="$MASTER_PORT" \
    --nproc_per_node=2 \
    train/train_vqvae.py \
    exp.run_name="FINAL: VQGAN 8X8" \
    data.h5_path="$REPO_ROOT/data/breakout.h5"
