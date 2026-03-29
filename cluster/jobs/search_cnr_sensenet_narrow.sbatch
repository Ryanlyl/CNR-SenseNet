#!/bin/bash -l
#SBATCH --job-name=cnr-search-r2
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/results/cnr_sensenet_search_round2}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-gpu_search_round2}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
source activate "$ENV_NAME"

python -m project.search_cnr_sensenet \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --device cuda \
  --lr-values 8e-4 1e-3 1.2e-3 \
  --weight-decay-values 0.0 1e-6 1e-5 \
  --dropout-values 0.15 0.2 0.25 \
  --batch-size-values 256 512 768 \
  --epochs-values 30 40 \
  --patience 3 \
  --top-k 20 \
  --resume \
  --save-best-checkpoint \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  "$@"
