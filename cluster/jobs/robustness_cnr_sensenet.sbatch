#!/bin/bash -l
#SBATCH --job-name=cnr-robust
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/project/results/cnr_sensenet_robustness}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-cnr_sensenet_robustness}"
SEED="${SEED:-42}"
PERTURBATIONS="${PERTURBATIONS:-extra_awgn phase_rotation gain_imbalance impulsive_noise}"
AWGN_STD_VALUES="${AWGN_STD_VALUES:-0.05 0.1 0.2 0.35}"
PHASE_MAX_DEG_VALUES="${PHASE_MAX_DEG_VALUES:-5 15 30 45}"
GAIN_IMBALANCE_VALUES="${GAIN_IMBALANCE_VALUES:-0.05 0.1 0.2 0.3}"
IMPULSE_PROB_VALUES="${IMPULSE_PROB_VALUES:-0.01 0.03 0.05 0.1}"
IMPULSE_SCALE="${IMPULSE_SCALE:-3.0}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$OUTPUT_DIR"

module load Miniforge3
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

python -m project.robustness \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --output-dir "$OUTPUT_DIR" \
  --output-prefix "$OUTPUT_PREFIX" \
  --seed "$SEED" \
  --device cuda \
  --save-checkpoint \
  --aux-branch-type autocorr \
  --autocorr-max-lag 8 \
  --perturbations $PERTURBATIONS \
  --awgn-std-values $AWGN_STD_VALUES \
  --phase-max-deg-values $PHASE_MAX_DEG_VALUES \
  --gain-imbalance-values $GAIN_IMBALANCE_VALUES \
  --impulse-prob-values $IMPULSE_PROB_VALUES \
  --impulse-scale "$IMPULSE_SCALE" \
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
