#!/bin/bash
#SBATCH --time=0-5:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --exclude=watgpu108,watgpu408,watgpu1008
#SBATCH --error=train-%j.log
#SBATCH --mail-user=k24liang@uwaterloo.ca
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Environment setup
eval "$(conda shell.bash hook)"
conda activate sweetdreams

RUN_SHA=b1d93ed25e159f38aae88c6c2ab19318ed65bb68
REPO_ROOT=$(git rev-parse --show-toplevel)
WORKTREE=$REPO_ROOT/../SweetDreams-runs/$SLURM_JOB_ID
if [ ! -d "$WORKTREE" ]; then
    git -C "$REPO_ROOT" worktree add -d "$WORKTREE" "$RUN_SHA"
fi
cd "$WORKTREE"

MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))

export NCCL_P2P_DISABLE=1

torchrun \
    --master_port="$MASTER_PORT" \
    --nproc_per_node=2 \
    train/train_world_model.py \
    exp.run_name="world model" \
    data.h5_path="$REPO_ROOT/data/breakout.h5" \
    checkpoints.tokenizer="$REPO_ROOT/weights/tokenizer.pt"
