#!/bin/bash -l
#SBATCH --job-name=cnr-eval
#SBATCH --output=slurm-%x-%j.out

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$HOME/CNR-SenseNet}"
ENV_NAME="${ENV_NAME:-cnr-sensenet}"
PKL_PATH="${PKL_PATH:-$REPO_ROOT/project/data/RML2016.10a_dict.pkl}"
CACHE_PATH="${CACHE_PATH:-$REPO_ROOT/project/data/processed/cluster_signal_noise.npz}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-}"

if [[ -z "$CHECKPOINT_PATH" ]]; then
  if [[ $# -lt 1 ]]; then
    echo "Usage: sbatch cluster/jobs/evaluate_checkpoint.sbatch /path/to/checkpoint.pt [args...]"
    exit 1
  fi
  CHECKPOINT_PATH="$1"
  shift
fi

OUTPUT_JSON="${OUTPUT_JSON:-${CHECKPOINT_PATH%.*}_eval.json}"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$CACHE_PATH")" "$(dirname "$OUTPUT_JSON")"

module load Miniforge3
source activate "$ENV_NAME"

python -m project.evaluate \
  --checkpoint "$CHECKPOINT_PATH" \
  --dataset-npz "$CACHE_PATH" \
  --pkl-path "$PKL_PATH" \
  --cache-path "$CACHE_PATH" \
  --output-json "$OUTPUT_JSON" \
  "$@"
