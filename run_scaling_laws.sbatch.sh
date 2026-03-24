#!/bin/bash
#SBATCH --job-name=scaling_laws
#SBATCH --time=04:00:00
#SBATCH --output=log/%x_%j.log
#SBATCH --error=log/%x_%j.log

set -euo pipefail
export PYTHONUNBUFFERED=1

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "$PROJECT_ROOT"

mkdir -p log results/scaling_laws

if [[ -z "${SCALING_API_KEY:-}" ]]; then
  echo "ERROR: SCALING_API_KEY is not set."
  echo "Set it before sbatch, for example:"
  echo "  export SCALING_API_KEY='your_api_key_here'"
  exit 1
fi

echo "Running scaling_laws.py with uv"
uv run python scaling_laws.py \
  --api-key-env SCALING_API_KEY \
  --api-url "http://hyperturing.stanford.edu:8000" \
  --max-budget 2e18 \
  --target-flops 1e19 \
  --output-dir results/scaling_laws

echo "Done. Key outputs:"
echo "  results/scaling_laws/summary.txt"
echo "  results/scaling_laws/predictions.json"

