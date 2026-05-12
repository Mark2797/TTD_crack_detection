#!/bin/bash -l
#SBATCH --job-name=ttd-train
#SBATCH --account=csci5527
#SBATCH --partition=msigpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=/users/7/yu001011/csci5527/logs/%x-%j.out

set -euo pipefail

# Use absolute paths for Slurm compatibility
WORKSPACE_ROOT="/users/7/yu001011/csci5527"
PROJECT_ROOT="$WORKSPACE_ROOT/CSCI5527-final"
SCRIPT_DIR="$PROJECT_ROOT/baseline_models_scripts"

cd "$PROJECT_ROOT"
source "$WORKSPACE_ROOT/.venv/bin/activate"

python "$SCRIPT_DIR/train_ttd.py" \
  --project-root "$PROJECT_ROOT" \
  --experiment all \
  --arch Unet \
  --encoder resnet34 \
  --encoder-weights imagenet \
  --epochs 100 \
  --batch-size 16 \
  --img-size 512 \
  --lr 1e-4 \
  --quiet
