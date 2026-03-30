#!/bin/bash -l
#SBATCH --job-name=cnr-explain
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/plots/cnr_sensenet_explainability}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cnr_sensenet_explainability}"
SEED="${SEED:-42}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m project.explainability \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --seed "$SEED" \
  --device cuda \
  --save-checkpoint \
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
