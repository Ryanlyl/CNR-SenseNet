#!/bin/bash -l
#SBATCH --job-name=cnr-ablation
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/results/cnr_sensenet_ablation}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cnr_sensenet_ablation}"
SEEDS="${SEEDS:-42 43 44}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
source activate "$ENV_NAME"

python -m project.ablation \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --device cuda \
  --seeds $SEEDS \
  --lr 1e-3 \
  --weight-decay 0.0 \
  --dropout 0.2 \
  --batch-size 512 \
  --epochs 30 \
  --patience 3 \
  --threshold-mode balanced_acc \
  --calibration-split val \
  --snr-loss-weighting two_band \
  --low-snr-cutoff -10 \
  --low-snr-positive-weight 3.0 \
  --mid-snr-cutoff -6 \
  --mid-snr-positive-weight 2.0 \
  "$@"
