#!/bin/bash -l
#SBATCH --job-name=cnr-baselines
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/results/baselines}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

if [[ ! -f "$CACHE_PATH" ]]; then
  if [[ ! -f "$PKL_PATH" ]]; then
    echo "Cache not found at $CACHE_PATH and raw pickle not found at $PKL_PATH" >&2
    exit 1
  fi
  python -m project.prepare_dataset \
    --pkl-path "$PKL_PATH" \
    --cache-path "$CACHE_PATH"
fi

python -m project.train \
  --dataset-npz "$CACHE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda \
  "$@"
