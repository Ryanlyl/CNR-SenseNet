#!/bin/bash -l
#SBATCH --job-name=cnr-prepare
#SBATCH --output=slurm-%x-%j.out

set -eo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
SUMMARY_JSON="${SUMMARY_JSON:-$REPO_ROOT/project/results/dataset/prepare_dataset_summary.json}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$(dirname "$SUMMARY_JSON")"

module load Miniforge3
source activate "$ENV_NAME"

python -m project.prepare_dataset \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --summary-json "$SUMMARY_JSON" \
  "$@"
