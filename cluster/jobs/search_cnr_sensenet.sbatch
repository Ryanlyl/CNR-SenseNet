#!/bin/bash -l
#SBATCH --job-name=cnr-search
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/results/cnr_sensenet_search_gpu}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-gpu_search}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m project.search_cnr_sensenet \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --device cuda \
  --aux-branch-type autocorr \
  --autocorr-max-lag 8 \
  --lr-values 1e-3 5e-4 3e-4 \
  --weight-decay-values 0.0 1e-5 1e-4 \
  --dropout-values 0.1 0.2 0.3 \
  --batch-size-values 512 1024 2048 \
  --epochs-values 10 20 30 \
  --patience 3 \
  --resume \
  --save-best-checkpoint \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  "$@"
