#!/bin/bash
#SBATCH --job-name=chinchilla_isoflops
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

module load cuda12.6/toolkit/12.6

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"


echo "change uv cache dir to /var/scratch/dpp2567/.uv_cache"
source /var/scratch/dpp2567/miniconda3/etc/profile.d/conda.sh
export UV_CACHE_DIR=/var/scratch/dpp2567/.uv_cache
export XDG_CACHE_HOME=/var/scratch/dpp2567/.cache
export UV_PROJECT_ENVIRONMENT=/var/scratch/dpp2567/.venvs/cs336-assignment2-systems

conda activate base

echo "------Running chinchilla_isoflops.py with uv and Python------"

uv run python chinchilla_isoflops.py \
  --input data/isoflops_curves.json \
  --output-dir results/chinchilla_isoflops \
  --predict-budgets 1e23 1e24 \
  --extrapolate-to 1e24

